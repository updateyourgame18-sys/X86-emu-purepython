"""
disk.py — Disk image backend for x86emu Phase 4
Supports:
  - Local file (for PC testing)
  - HTTP lazy fetch (for Pythonista — fetches sectors on demand via Range requests)
  - LRU sector cache (avoids re-fetching hot sectors)

Sector size: 512 bytes (standard ATA)
Addressing: LBA28 (up to 128 GB disks)

Usage:
    # Local file
    disk = Disk.from_file("dsl.img")

    # HTTP lazy (Pythonista)
    disk = Disk.from_http("http://192.168.1.x:8080/dsl.img")

    # Read sector
    data = disk.read_sector(0)   # MBR

    # Write sector (cached only, not persisted to HTTP source)
    disk.write_sector(0, data)
"""

import struct
import os
import sys

SECTOR_SIZE  = 512
CACHE_MAX    = 512    # max sectors in LRU cache (~256 KB)


# LRU sector cache

class SectorCache:
    def __init__(self, maxsize=CACHE_MAX):
        self.maxsize = maxsize
        self._cache  = {}        # lba -> bytearray
        self._order  = []        # LRU order (most recent last)

    def get(self, lba):
        if lba in self._cache:
            # Move to end (most recently used)
            self._order.remove(lba)
            self._order.append(lba)
            return self._cache[lba]
        return None

    def put(self, lba, data):
        if lba in self._cache:
            self._order.remove(lba)
        elif len(self._cache) >= self.maxsize:
            # Evict least recently used
            old = self._order.pop(0)
            del self._cache[old]
        self._cache[lba] = bytearray(data[:SECTOR_SIZE])
        self._order.append(lba)

    def dirty_write(self, lba, data):
        """Write to cache only (not back to source)."""
        self.put(lba, data)

    def stats(self):
        return {'size': len(self._cache), 'maxsize': self.maxsize}



# Base Disk class

class Disk:
    def __init__(self):
        self.sector_count = 0
        self.cache        = SectorCache()
        self._hits        = 0
        self._misses      = 0
        self._writes      = 0

    # --- Override in subclasses ---
    def _fetch_sector(self, lba) -> bytes:
        raise NotImplementedError

    # --- Public API ---
    def read_sector(self, lba) -> bytearray:
        if lba >= self.sector_count:
            return bytearray(SECTOR_SIZE)
        cached = self.cache.get(lba)
        if cached is not None:
            self._hits += 1
            return cached
        self._misses += 1
        data = self._fetch_sector(lba)
        self.cache.put(lba, data)
        return self.cache.get(lba)

    def write_sector(self, lba, data):
        self._writes += 1
        self.cache.dirty_write(lba, data)

    def read_sectors(self, lba, count) -> bytearray:
        result = bytearray()
        for i in range(count):
            result += self.read_sector(lba + i)
        return result

    def stats(self):
        return {
            'sectors':  self.sector_count,
            'size_mb':  self.sector_count * SECTOR_SIZE // (1024 * 1024),
            'hits':     self._hits,
            'misses':   self._misses,
            'writes':   self._writes,
            'cache':    self.cache.stats(),
        }

    # --- Factory methods ---
    @classmethod
    def from_file(cls, path):
        d = FileDisk(path)
        return d

    @classmethod
    def from_http(cls, url, sector_count=None):
        d = HTTPDisk(url, sector_count)
        return d

    @classmethod
    def from_bytes(cls, data):
        d = BytesDisk(data)
        return d



# FileDisk — backed by a local file

class FileDisk(Disk):
    def __init__(self, path):
        super().__init__()
        self.path = path
        size = os.path.getsize(path)
        self.sector_count = size // SECTOR_SIZE
        self._f = open(path, 'rb')
        print(f"[disk] Opened {path}: {self.sector_count} sectors ({size // (1024*1024)} MB)")

    def _fetch_sector(self, lba):
        self._f.seek(lba * SECTOR_SIZE)
        data = self._f.read(SECTOR_SIZE)
        if len(data) < SECTOR_SIZE:
            data += b'\x00' * (SECTOR_SIZE - len(data))
        return data

    def __del__(self):
        try:
            self._f.close()
        except Exception:
            pass



# HTTPDisk — lazy HTTP fetch via Range requests (Pythonista-friendly)

class HTTPDisk(Disk):
    def __init__(self, url, sector_count=None):
        super().__init__()
        self.url = url
        self._probe(sector_count)

    def _probe(self, sector_count):
        """HEAD request to get Content-Length and verify Range support."""
        import urllib.request
        try:
            req = urllib.request.Request(self.url, method='HEAD')
            with urllib.request.urlopen(req, timeout=10) as resp:
                headers = resp.headers
                cl = headers.get('Content-Length')
                if cl:
                    total_bytes       = int(cl)
                    self.sector_count = total_bytes // SECTOR_SIZE
                    accept_ranges     = headers.get('Accept-Ranges', '')
                    if 'bytes' not in accept_ranges:
                        print("[disk] WARNING: server may not support Range requests")
                    print(f"[disk] HTTP disk: {self.url}")
                    print(f"[disk] Size: {total_bytes // (1024*1024)} MB, "
                          f"{self.sector_count} sectors")
                elif sector_count:
                    self.sector_count = sector_count
                    print(f"[disk] HTTP disk (no Content-Length): assuming {sector_count} sectors")
                else:
                    self.sector_count = 0
                    print("[disk] WARNING: cannot determine disk size")
        except Exception as e:
            print(f"[disk] HTTP probe failed: {e}")
            if sector_count:
                self.sector_count = sector_count

    def _fetch_sector(self, lba):
        import urllib.request
        start = lba * SECTOR_SIZE
        end   = start + SECTOR_SIZE - 1
        req   = urllib.request.Request(
            self.url,
            headers={'Range': f'bytes={start}-{end}'}
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read(SECTOR_SIZE)
                if len(data) < SECTOR_SIZE:
                    data += b'\x00' * (SECTOR_SIZE - len(data))
                return data
        except Exception as e:
            print(f"[disk] HTTP fetch sector {lba} failed: {e}")
            return b'\x00' * SECTOR_SIZE

    def prefetch(self, lba_list):
        """
        Prefetch multiple sectors in parallel using threads.
        Call this when you know which sectors will be needed soon
        (e.g. after reading MBR, prefetch partition table sectors).
        """
        import threading
        def fetch_one(lba):
            if self.cache.get(lba) is None:
                data = self._fetch_sector(lba)
                self.cache.put(lba, data)

        threads = [threading.Thread(target=fetch_one, args=(lba,))
                   for lba in lba_list if self.cache.get(lba) is None]
        for t in threads: t.start()
        for t in threads: t.join()



# BytesDisk — backed by in-memory bytes (for testing)

class BytesDisk(Disk):
    def __init__(self, data):
        super().__init__()
        self._data        = bytearray(data)
        self.sector_count = len(data) // SECTOR_SIZE
        # Pad to sector boundary
        remainder = len(data) % SECTOR_SIZE
        if remainder:
            self._data += bytearray(SECTOR_SIZE - remainder)
            self.sector_count += 1
        print(f"[disk] BytesDisk: {self.sector_count} sectors "
              f"({len(self._data) // 1024} KB)")

    def _fetch_sector(self, lba):
        offset = lba * SECTOR_SIZE
        return bytes(self._data[offset:offset + SECTOR_SIZE])

    def write_sector(self, lba, data):
        """BytesDisk supports actual writes."""
        super().write_sector(lba, data)
        offset = lba * SECTOR_SIZE
        self._data[offset:offset + SECTOR_SIZE] = bytearray(data[:SECTOR_SIZE])



# Disk image builder — creates minimal test images

class DiskBuilder:
    """Build minimal disk images for testing without needing real disk files."""

    SECTOR = 512

    @staticmethod
    def build_test_disk(kernel_code: bytes, kernel_load_addr: int = 0x10000) -> bytes:
        """
        Build a minimal bootable disk image:
          Sector 0: MBR with bootloader that loads sectors 1-N into memory
                    then jumps to kernel_load_addr
          Sectors 1-N: kernel_code (padded to sector boundary)

        The MBR bootloader:
          - Uses INT 13h AH=0x02 (BIOS disk read) in real mode
          - Loads kernel_code into memory at kernel_load_addr
          - Far jumps to CS=0, IP=kernel_load_addr
        """
        S = DiskBuilder.SECTOR

        # How many sectors does the kernel occupy?
        kernel_sectors = (len(kernel_code) + S - 1) // S
        kernel_padded  = kernel_code + b'\x00' * (kernel_sectors * S - len(kernel_code))

        # ---- MBR bootloader (hand-assembled) ----
        mbr = bytearray(S)

        # We encode a simple INT 13h loader.
        # Memory layout:
        #   0x0000:7C00 — MBR loaded here by BIOS
        #   kernel_load_addr — where we load the kernel

        load_seg    = (kernel_load_addr >> 4) & 0xFFFF
        load_off    = kernel_load_addr & 0xF

        code = bytearray()
        # XOR AX, AX; MOV DS,AX; MOV ES,AX; MOV SS,AX
        code += b'\x31\xC0\x8E\xD8\x8E\xC0\x8E\xD0'
        # MOV SP, 0x7C00
        code += b'\xBC\x00\x7C'

        # INT 13h read:
        #   AH=0x02 (read sectors)
        #   AL=sector_count
        #   CH=0 (cylinder 0)
        #   CL=2 (start at sector 2, 1-based)
        #   DH=0 (head 0)
        #   DL=0x80 (first hard disk)
        #   ES:BX = destination buffer
        code += bytes([0xB8, kernel_sectors, 0x02])  # MOV AX, 0x02 | sectors<<8
        code += bytes([0xBB, load_off & 0xFF, (load_off >> 8) & 0xFF])  # MOV BX, load_off
        code += bytes([0xB9, 0x00, 0x00])   # MOV CX, 0x0002 — cylinder 0, sector 2
        code[-2] = 0x00
        code[-1] = 0x02
        code += bytes([0xBA, 0x00, 0x80])   # MOV DX, 0x0080 — DH=0 DL=0x80
        # MOV ES, load_seg
        code += bytes([0xB8, load_seg & 0xFF, (load_seg >> 8) & 0xFF])  # MOV AX, load_seg
        code += b'\x8E\xC0'                 # MOV ES, AX
        code += b'\xCD\x13'                 # INT 13h
        # JC error (carry = read error)
        code += b'\x72\x05'                 # JC +5 (to error HLT)
        # Far jump to loaded kernel: JMP FAR 0000:kernel_load_addr
        code += b'\xEA'
        code += struct.pack('<H', kernel_load_addr & 0xFFFF)
        code += struct.pack('<H', (kernel_load_addr >> 16) & 0xFFFF)
        # Error: HLT
        code += b'\xF4\xEB\xFE'

        assert len(code) <= 446, f"MBR code too large: {len(code)} bytes"
        mbr[:len(code)] = code

        # Partition table (1 partition, type 0x83 Linux, start LBA=1)
        pt_offset = 446
        # Status=0x80 (bootable), CHS start, type, CHS end, LBA start, LBA size
        def chs(c, h, s): return bytes([h, (s & 0x3F) | ((c>>8)<<6), c & 0xFF])
        mbr[pt_offset:pt_offset+16] = (
            b'\x80' +           # bootable
            chs(0, 0, 2) +      # CHS start (cylinder 0, head 0, sector 2)
            b'\x83' +           # type: Linux
            chs(0, 0, kernel_sectors + 1) +  # CHS end
            struct.pack('<I', 1) +            # LBA start
            struct.pack('<I', kernel_sectors) # LBA size
        )

        # Boot signature
        mbr[510] = 0x55
        mbr[511] = 0xAA

        # Assemble full image
        image = bytes(mbr) + kernel_padded
        return image

    @staticmethod
    def build_minimal_linux_disk(mem_size_mb: int = 32) -> bytes:
        """
        Build a disk image that a real GRUB legacy can boot from.
        Placeholder — in Phase 5 we'll populate this with a real
        kernel image fetched over HTTP.
        For now returns a test image with a stub 'kernel'.
        """
        # Stub kernel: sets up GDT, enters pmode, writes to VGA, HLTs
        # This is exactly what Phase 2 tested — proves disk load works
        stub = (
            b'\xFA'             # CLI
            b'\x31\xC0'         # XOR AX,AX
            b'\x8E\xD8'         # MOV DS,AX
            # Write 'D','S','K' to VGA (we're in real mode still)
            b'\xBB\x00\x80'     # MOV BX, 0xB800 (VGA segment)
            b'\x8E\xC3'         # MOV ES, BX (BX already has 0xB800... wait)
        )
        # Simpler stub: just print via INT 10h and HLT
        msg   = b"DISK OK - kernel stub loaded!\r\n"
        stub2 = bytearray()
        stub2 += b'\x31\xC0\x8E\xD8\x8E\xC0'      # set DS=ES=0
        stub2 += b'\xBE' + struct.pack('<H', 0x10000 + len(stub2) + 9)
        # (address of msg — will patch below)
        loop  = len(stub2)
        stub2 += b'\xAC'                             # LODSB
        stub2 += b'\x08\xC0'                         # OR AL,AL
        stub2 += b'\x74\x07'                         # JZ done
        stub2 += b'\xB4\x0E\xBB\x07\x00\xCD\x10'    # INT 10h TTY
        stub2 += bytes([0xEB, (loop - len(stub2) - 2) & 0xFF])  # JMP loop
        stub2 += b'\xF4\xEB\xFE'                     # HLT; JMP $
        stub2 += msg
        # Patch SI to point at message
        msg_addr = 0x10000 + 9
        struct.pack_into('<H', stub2, 3, msg_addr & 0xFFFF)

        return DiskBuilder.build_test_disk(bytes(stub2), kernel_load_addr=0x10000)
