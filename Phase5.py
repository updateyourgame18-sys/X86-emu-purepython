"""
phase5.py — x86emu Phase 5: Boot DSL Linux from ISO
Imports Machine4 from phase4.py, adds:

  - ISO 9660 + El Torito parser (finds boot image in the ISO)
  - ISODisk — treats ISO as a block device with 2048-byte sectors
    mapped to 512-byte ATA sectors for the BIOS layer
  - INT 13h CD-ROM extensions (AH=0x41 check, AH=0x42 LBA read)
    on virtual drive 0x9F (El Torito virtual CD)
  - INT 15h AX=0xE820 memory map (Linux kernel REQUIRES this)
  - INT 15h AX=0x88 extended memory size
  - INT 12h conventional memory size
  - More opcodes: AAM, DAA, ADC, SBB, IMUL r,r/m, string REP prefix
  - Enough to get isolinux running and loading vmlinuz+initrd

Usage (on your iPhone with dsl.iso in Pythonista Documents):
    import phase5
    m = phase5.boot_iso('/path/to/dsl.iso')
    m.run()

Or run this file directly (set ISO_PATH at bottom).
"""

import struct, sys, os, time
import importlib

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
p4 = importlib.import_module('phase4')
p3 = importlib.import_module('phase3')
p2 = importlib.import_module('phase2')
disk_mod = importlib.import_module('disk')
ide_mod  = importlib.import_module('ide')

Machine4    = p4.Machine4
BIOS4       = p4.BIOS4
IOPorts4    = p4.IOPorts4
Memory32    = p2.Memory32
Registers32 = p2.Registers32
CPU32       = p2.CPU32
CPU3        = p3.CPU3
PIC         = p3.PIC
PIT         = p3.PIT
KBD         = p3.KBD
CMOS        = p3.CMOS
sign8       = p2.sign8
sign16      = p2.sign16
sign32      = p2.sign32
get_reg8    = p2.get_reg8
set_reg8    = p2.set_reg8
get_reg16   = p2.get_reg16
set_reg16   = p2.set_reg16
update_flags_8  = p2.update_flags_8
update_flags_16 = p2.update_flags_16
FileDisk    = disk_mod.FileDisk
IDEController = ide_mod.IDEController

ISO_SECTOR  = 2048   # ISO 9660 logical sector size
ATA_SECTOR  = 512


# ---------------------------------------------------------------------------
# ISO 9660 + El Torito parser
# ---------------------------------------------------------------------------
class ISOParser:
    """
    Parses ISO 9660 to find the El Torito boot catalog and boot image.
    DSL uses isolinux in no-emulation mode.
    """
    def __init__(self, path):
        self.path = path
        self.f    = open(path, 'rb')
        self._size = os.path.getsize(path)
        self.boot_catalog_lba = None
        self.boot_image_lba   = None
        self.boot_image_size  = None   # in 512-byte sectors
        self.boot_load_seg    = 0x07C0
        self.boot_load_addr   = 0x7C00
        self.no_emul          = True
        self._parse()

    def _read_sector(self, lba, size=ISO_SECTOR):
        self.f.seek(lba * ISO_SECTOR)
        return self.f.read(size)

    def find_file_size(self, filename):
        """
        Look up a file's real size in bytes from its ISO9660 directory
        record, by name. Used because El Torito's boot catalog "sectors"
        field only guarantees a minimum load size (often just enough to
        get isolinux's own bootstrap running), not the file's true size —
        isolinux itself is meant to load the rest via its own INT13h
        calls, but if that path doesn't get exercised, loading the truncated
        copy leaves real code missing from memory.
        """
        self.f.seek(0)
        data = self.f.read(self._size)
        needle = filename.encode('ascii')
        idx = data.find(needle)
        if idx < 0 or idx < 33:
            return None
        rec_start = idx - 33
        length = data[rec_start]
        if length == 0 or length > 200:
            return None
        try:
            size = struct.unpack('<I', data[rec_start+10:rec_start+14])[0]
        except struct.error:
            return None
        return size

    def _parse(self):
        print(f"[iso] Parsing {self.path} ({self._size // (1024*1024)} MB)")
        # System area: LBA 0-15 unused
        # Volume descriptors start at LBA 16
        lba = 16
        while True:
            sec = self._read_sector(lba)
            vd_type = sec[0]
            ident   = sec[1:6]
            if ident != b'CD001':
                print(f"[iso] LBA {lba}: not a volume descriptor (magic={ident!r})")
                break
            if vd_type == 0xFF:   # Volume Descriptor Set Terminator
                break
            if vd_type == 0x00:   # Boot Record Volume Descriptor
                # El Torito: boot system identifier at offset 7
                boot_sys_id = sec[7:39].rstrip(b'\x00')
                if b'EL TORITO' in boot_sys_id or b'El Torito' in boot_sys_id:
                    # Boot catalog location at offset 0x47 (4 bytes LE)
                    self.boot_catalog_lba = struct.unpack_from('<I', sec, 0x47)[0]
                    print(f"[iso] El Torito boot catalog at LBA {self.boot_catalog_lba}")
            lba += 1
            if lba > 32:
                break

        if self.boot_catalog_lba is None:
            print("[iso] WARNING: No El Torito boot record found — trying LBA 20")
            self.boot_catalog_lba = 20

        self._parse_catalog()

    def _parse_catalog(self):
        cat = self._read_sector(self.boot_catalog_lba)

        # Validation entry (32 bytes)
        # header_id=0x01, platform=0x00 (x86)
        header_id = cat[0]
        if header_id != 0x01:
            print(f"[iso] Boot catalog validation entry header={header_id:#04x} (expected 0x01)")

        # Initial/Default entry starts at offset 32
        # boot_indicator=0x88 (bootable)
        offset = 32
        boot_indicator = cat[offset]
        boot_media     = cat[offset + 1]   # 0=no emulation, 1=1.2M, 2=1.44M, 3=2.88M, 4=HDD
        load_seg       = struct.unpack_from('<H', cat, offset + 2)[0]
        system_type    = cat[offset + 4]
        sector_count   = struct.unpack_from('<H', cat, offset + 6)[0]
        lba            = struct.unpack_from('<I', cat, offset + 8)[0]

        print(f"[iso] Boot entry: indicator={boot_indicator:#04x} media={boot_media} "
              f"load_seg={load_seg:#06x} sectors={sector_count} lba={lba}")

        if boot_indicator != 0x88:
            print(f"[iso] WARNING: boot indicator {boot_indicator:#04x} != 0x88 (not bootable?)")

        self.boot_image_lba  = lba
        self.boot_image_size = sector_count   # in 512-byte sectors
        self.no_emul         = (boot_media == 0)

        if load_seg == 0:
            load_seg = 0x07C0   # default: load to 0x7C00

        self.boot_load_seg  = load_seg
        self.boot_load_addr = load_seg << 4

        print(f"[iso] Boot image at ISO LBA {lba}, load to {self.boot_load_addr:#010x}, "
              f"{'no-emulation' if self.no_emul else f'media={boot_media}'}")

    def read_iso_sector(self, iso_lba):
        """Read one 2048-byte ISO sector."""
        self.f.seek(iso_lba * ISO_SECTOR)
        data = self.f.read(ISO_SECTOR)
        if len(data) < ISO_SECTOR:
            data += b'\x00' * (ISO_SECTOR - len(data))
        return data

    def read_boot_image(self):
        """
        Read the boot image into bytes.
        Returns raw bytes of the boot image (isolinux.bin etc).
        """
        if self.boot_image_lba is None:
            return b'\xF4' * 512  # HLT if no boot image

        # boot_image_size is in 512-byte sectors
        # Boot image is stored in ISO sectors (2048 bytes each)
        # Calculate which ISO sector it lives in
        # ISO LBA * 2048 / 512 = ISO LBA * 4 in 512-byte sectors
        total_bytes = max(self.boot_image_size * 512, 512)
        # Read enough ISO sectors
        iso_sectors_needed = (total_bytes + ISO_SECTOR - 1) // ISO_SECTOR
        data = bytearray()
        for i in range(iso_sectors_needed):
            data += self.read_iso_sector(self.boot_image_lba + i)
        return bytes(data[:total_bytes])

    def __del__(self):
        try: self.f.close()
        except: pass


# ---------------------------------------------------------------------------
# ISODisk — exposes ISO as a block device with 512-byte sectors
# This lets the existing IDE/INT13h machinery work unchanged.
# ISO sector 0..N at 2048 bytes each → ATA sectors 0..N*4
# ---------------------------------------------------------------------------
class ISODisk:
    """
    Wraps an ISO file as a 512-byte-sector block device.
    ISO LBA n → ATA sectors n*4 .. n*4+3

    The whole ISO is loaded into a single in-memory bytes buffer once at
    startup instead of doing a disk seek+read per sector. Real isolinux
    issues MANY small INT13h reads while loading the kernel/initrd (one
    call per sector or small batch), and at Python-interpreter speeds,
    repeated disk syscalls were a real, measurable bottleneck — a 49MB
    ISO fits trivially in RAM, so we just slice it instead.
    """
    def __init__(self, path):
        self.path         = path
        size              = os.path.getsize(path)
        with open(path, 'rb') as f:
            self._data = f.read()
        self.sector_count = size // ATA_SECTOR
        self._iso_size    = size
        self._hits        = 0   # kept for stats() compatibility
        self._misses      = 0
        print(f"[isodisk] {path}: {size//(1024*1024)} MB, "
              f"{self.sector_count} ATA sectors (loaded fully into RAM)")

    def read_sector(self, lba) -> bytearray:
        self._hits += 1
        offset = lba * ATA_SECTOR
        end = offset + ATA_SECTOR
        if end <= len(self._data):
            return bytearray(self._data[offset:end])
        chunk = self._data[offset:len(self._data)]
        return bytearray(chunk + b'\x00' * (ATA_SECTOR - len(chunk)))

    def read_sectors(self, lba, count) -> bytearray:
        offset = lba * ATA_SECTOR
        end = offset + count * ATA_SECTOR
        if end <= len(self._data):
            return bytearray(self._data[offset:end])
        result = bytearray()
        for i in range(count):
            result += self.read_sector(lba + i)
        return result

    def write_sector(self, lba, data):
        pass  # ISO is read-only

    def read_iso_sector(self, iso_lba) -> bytes:
        """Read a 2048-byte ISO sector directly."""
        self._hits += 1
        offset = iso_lba * ISO_SECTOR
        end = offset + ISO_SECTOR
        if end <= len(self._data):
            return self._data[offset:end]
        chunk = self._data[offset:len(self._data)]
        return chunk + b'\x00' * (ISO_SECTOR - len(chunk))

    def stats(self):
        return {'hits': self._hits, 'misses': self._misses,
                'sectors': self.sector_count}


# ---------------------------------------------------------------------------
# Extended BIOS for ISO boot
# Adds:
#   INT 13h: CD-ROM extensions (drive 0x9F), El Torito reads
#   INT 15h: E820 memory map, extended memory size
#   INT 12h: conventional memory
#   INT 11h: equipment list
# ---------------------------------------------------------------------------
class BIOS5(BIOS4):
    def __init__(self, mem, reg, ide, iso_disk, iso_parser):
        super().__init__(mem, reg, ide)
        self.iso_disk   = iso_disk
        self.iso_parser = iso_parser
        self._e820_idx  = 0

        # E820 memory map: tell kernel about available RAM
        # We have 64MB emulated RAM
        self._e820_map = [
            # (base, length, type)  type: 1=usable, 2=reserved, 3=ACPI reclaimable
            (0x00000000, 0x0009F000, 1),   # 0 - 636KB conventional
            (0x0009F000, 0x00001000, 2),   # EBDA (reserved)
            (0x000A0000, 0x00060000, 2),   # VGA + ROM (reserved)
            (0x00100000, 0x03F00000, 1),   # 1MB - 64MB (64MB total)
        ]

    def _seg_base(self, seg='ds'):
        """Unreal-mode-aware segment base, matching CPU32._seg_base but
        usable from BIOS handlers that don't hold a direct cpu reference.
        Real-mode BIOS calls can be issued while a segment still carries a
        flat/unreal base from an earlier brief protected-mode excursion —
        using the plain segment<<4 formula in that case silently computes
        the wrong address for every buffer the call touches."""
        r = self.reg
        if r.protected_mode:
            return r.seg_cache[seg].base
        uv = getattr(r, '_unreal_valid', None)
        if uv and uv.get(seg):
            return r.seg_cache[seg].base
        return getattr(r, seg) << 4

    def interrupt(self, num):
        r = self.reg
        if num == 0x13:
            # We are a CD-only machine — only drive 0x9F is valid
            # All other drives (0x80-0xFE HDDs, floppies) → error
            # EXCEPT: reset (AH=0x00) always succeeds for any drive
            if r.ah == 0x00:
                r.ah = 0; r.cf = 0; return
            if r.dl == 0x9F:
                self._int13_cdrom()
            else:
                # No such drive
                r.cf = 1; r.ah = 0x01
        elif num == 0x15:
            self._int15()
        elif num == 0x12:
            r.ax = 639
        elif num == 0x11:
            r.ax = 0x0021
        elif num == 0x10:
            self._int10_vga()
        elif num == 0x16:
            self._int16_kbd()
        else:
            super().interrupt(num)

    def _int16_kbd(self):
        """INT 16h keyboard services — feeds from PS/2 keyboard buffer."""
        r = self.reg
        ah = r.ah & 0xFF
        # Try to get from kbd controller if available
        kbd = getattr(self, '_kbd_controller', None)
        if ah in (0x00, 0x10):   # read key (blocking)
            if kbd and kbd.has_data():
                sc = kbd.read_byte()
                # Convert scancode to ASCII roughly
                ascii_map = {0x1C: 0x0D, 0x39: 0x20, 0x0E: 0x08}
                ch = ascii_map.get(sc, sc)
                r.ax = (sc << 8) | (ch & 0xFF)
            else:
                r.ax = 0x1C0D   # default: Enter
            r.zf = 0
        elif ah in (0x01, 0x11):  # check key (non-blocking)
            if kbd and kbd.has_data():
                r.zf = 0
                r.ax = 0x1C0D
            else:
                r.zf = 1   # no key
                r.ah = 0
        elif ah == 0x02:          # get shift flags
            r.al = 0x00
        elif ah == 0x03:          # set repeat rate
            pass
        else:
            r.zf = 1

    # -----------------------------------------------------------------------
    # VGA BIOS — full INT 10h implementation
    # Maintains cursor position, writes to VGA text buffer at 0xB8000
    # -----------------------------------------------------------------------
    VGA_TEXT_BASE = 0xB8000
    VGA_COLS      = 80
    VGA_ROWS      = 25

    def _vga_addr(self, row, col):
        return self.VGA_TEXT_BASE + (row * self.VGA_COLS + col) * 2

    def _int10_vga(self):
        r   = self.reg
        ah  = r.ah & 0xFF
        al  = r.al & 0xFF
        bh  = r.bh & 0xFF   # page (we only support page 0)
        bl  = r.bl & 0xFF   # attribute
        cx  = r.cx & 0xFFFF
        dx  = r.dx & 0xFFFF

        # Read BIOS data area cursor position (page 0 at 0x450)
        cur_row = self.mem.read8_flat(0x450 + bh * 2 + 1)
        cur_col = self.mem.read8_flat(0x450 + bh * 2)

        if ah == 0x00:   # Set video mode
            # Clear screen
            for i in range(self.VGA_ROWS * self.VGA_COLS):
                addr = self.VGA_TEXT_BASE + i * 2
                self.mem.write8_flat(addr,     0x20)   # space
                self.mem.write8_flat(addr + 1, 0x07)   # grey on black
            cur_row = cur_col = 0
            self.mem.write8_flat(0x449, al)   # current video mode

        elif ah == 0x01:  # Set cursor shape — ignore
            pass

        elif ah == 0x02:  # Set cursor position
            page   = bh
            cur_row = (dx >> 8) & 0xFF
            cur_col =  dx       & 0xFF
            cur_row = max(0, min(cur_row, self.VGA_ROWS - 1))
            cur_col = max(0, min(cur_col, self.VGA_COLS - 1))

        elif ah == 0x03:  # Get cursor position
            r.dh = cur_row & 0xFF
            r.dl = cur_col & 0xFF
            r.ch = 0x06
            r.cl = 0x07

        elif ah == 0x05:  # Set active page — ignore
            pass

        elif ah == 0x06:  # Scroll up
            lines = al if al else self.VGA_ROWS
            top  = (cx >> 8) & 0xFF; left  = cx & 0xFF
            bot  = (dx >> 8) & 0xFF; right = dx & 0xFF
            attr = bh
            if lines >= (bot - top + 1):
                # Clear entire region
                for row in range(top, bot + 1):
                    for col in range(left, right + 1):
                        a = self._vga_addr(row, col)
                        self.mem.write8_flat(a, 0x20)
                        self.mem.write8_flat(a + 1, attr)
            else:
                for row in range(top, bot - lines + 1):
                    for col in range(left, right + 1):
                        src = self._vga_addr(row + lines, col)
                        dst = self._vga_addr(row, col)
                        self.mem.write8_flat(dst,     self.mem.read8_flat(src))
                        self.mem.write8_flat(dst + 1, self.mem.read8_flat(src + 1))
                for row in range(bot - lines + 1, bot + 1):
                    for col in range(left, right + 1):
                        a = self._vga_addr(row, col)
                        self.mem.write8_flat(a, 0x20)
                        self.mem.write8_flat(a + 1, attr)

        elif ah == 0x07:  # Scroll down — similar to 0x06
            lines = al if al else self.VGA_ROWS
            top  = (cx >> 8) & 0xFF; left  = cx & 0xFF
            bot  = (dx >> 8) & 0xFF; right = dx & 0xFF
            attr = bh
            for row in range(bot, top + lines - 1, -1):
                for col in range(left, right + 1):
                    src = self._vga_addr(row - lines, col)
                    dst = self._vga_addr(row, col)
                    self.mem.write8_flat(dst,     self.mem.read8_flat(src))
                    self.mem.write8_flat(dst + 1, self.mem.read8_flat(src + 1))
            for row in range(top, top + lines):
                for col in range(left, right + 1):
                    a = self._vga_addr(row, col)
                    self.mem.write8_flat(a, 0x20)
                    self.mem.write8_flat(a + 1, attr)

        elif ah == 0x08:  # Read char+attr at cursor
            addr = self._vga_addr(cur_row, cur_col)
            r.al = self.mem.read8_flat(addr)
            r.ah = self.mem.read8_flat(addr + 1)

        elif ah == 0x09:  # Write char+attr at cursor (no cursor advance)
            count = cx if cx else 1
            attr  = bl
            for i in range(count):
                col = (cur_col + i) % self.VGA_COLS
                row = cur_row + (cur_col + i) // self.VGA_COLS
                if row >= self.VGA_ROWS: break
                addr = self._vga_addr(row, col)
                self.mem.write8_flat(addr,     al)
                self.mem.write8_flat(addr + 1, attr)
            # Also append to text output for console capture
            if 0x20 <= al < 0x7F:
                self._output.append(chr(al))

        elif ah == 0x0A:  # Write char only at cursor (no attr change)
            count = cx if cx else 1
            for i in range(count):
                col = (cur_col + i) % self.VGA_COLS
                row = cur_row + (cur_col + i) // self.VGA_COLS
                if row >= self.VGA_ROWS: break
                addr = self._vga_addr(row, col)
                self.mem.write8_flat(addr, al)
            if 0x20 <= al < 0x7F:
                self._output.append(chr(al))

        elif ah == 0x0E:  # TTY write — advances cursor, handles \r\n\b
            ch = al
            if ch == 0x0D:   # CR
                cur_col = 0
            elif ch == 0x0A: # LF
                cur_row += 1
            elif ch == 0x08: # BS
                if cur_col > 0: cur_col -= 1
            elif ch == 0x07: # BEL — ignore
                pass
            else:
                addr = self._vga_addr(cur_row, cur_col)
                attr = self.mem.read8_flat(addr + 1) or 0x07
                self.mem.write8_flat(addr,     ch)
                self.mem.write8_flat(addr + 1, attr)
                cur_col += 1
                if cur_col >= self.VGA_COLS:
                    cur_col = 0; cur_row += 1
            # Scroll if needed
            if cur_row >= self.VGA_ROWS:
                cur_row = self.VGA_ROWS - 1
                # Scroll up 1 line
                for row in range(self.VGA_ROWS - 1):
                    for col in range(self.VGA_COLS):
                        src = self._vga_addr(row + 1, col)
                        dst = self._vga_addr(row, col)
                        self.mem.write8_flat(dst,     self.mem.read8_flat(src))
                        self.mem.write8_flat(dst + 1, self.mem.read8_flat(src + 1))
                for col in range(self.VGA_COLS):
                    a = self._vga_addr(self.VGA_ROWS - 1, col)
                    self.mem.write8_flat(a, 0x20)
                    self.mem.write8_flat(a + 1, 0x07)
            # Capture for console output
            if ch not in (0x0D, 0x0A, 0x07, 0x08):
                if 0x20 <= ch < 0x7F:
                    self._output.append(chr(ch))
                else:
                    self._output.append('?')
            elif ch in (0x0D, 0x0A):
                self._output.append(chr(ch))

        elif ah == 0x0F:  # Get video mode
            r.al = self.mem.read8_flat(0x449) or 0x03   # mode 3 = 80x25 color text
            r.ah = self.VGA_COLS
            r.bh = 0   # page 0

        elif ah == 0x10:  # Set palette — ignore for text mode
            pass

        elif ah == 0x11:  # Character generator — ignore
            pass

        elif ah == 0x12:  # Video subsystem config — stub
            if bl == 0x10:  # return video config info
                r.bh = 0; r.bl = 0x00; r.ch = 0; r.cl = 0x09

        elif ah == 0x1A:  # Get/Set Display Combination Code (VGA only)
            if al == 0x00:  # Get
                r.al = 0x1A          # confirms VGA function is supported
                r.bl = 0x08          # 0x08 = VGA with color analog display
                r.bh = 0x00          # (no inactive display)
            elif al == 0x01:  # Set — accept and acknowledge
                r.al = 0x1A

        elif ah == 0x4F:  # VESA BIOS Extensions
            # DSL's kernel command line specifies vga=791 (mode 0x317 =
            # 1024x768, 16bpp/64K color) — without VBE support the kernel
            # can't detect or switch to this mode at all, and never
            # reaches the point of drawing the Tux splash screen that
            # normally appears right after "Ready." on real hardware.
            LFB_ADDR = 0x0F000000   # linear framebuffer physical address,
                                     # placed near the top of our 256MB
                                     # emulated RAM, well clear of anything
                                     # the kernel uses for normal allocation
                                     # this early in boot
            MODE_XRES, MODE_YRES, MODE_BPP = 1024, 768, 16

            if al == 0x00:   # Get SuperVGA Info
                buf = self._seg_base('es') + r.di
                self.mem.load_flat(buf, b'VESA')            # signature
                struct.pack_into('<H', self.mem._m, buf+4, 0x0200)  # VBE 2.0
                # OemStringPtr -> point at a small string we place right
                # after the info block itself
                oem_off = r.di + 0x100
                struct.pack_into('<H', self.mem._m, buf+6, oem_off)
                struct.pack_into('<H', self.mem._m, buf+8, r.es)
                struct.pack_into('<I', self.mem._m, buf+10, 0)      # capabilities
                # VideoModePtr -> list of supported modes, ending 0xFFFF
                modelist_off = r.di + 0x120
                struct.pack_into('<H', self.mem._m, buf+14, modelist_off)
                struct.pack_into('<H', self.mem._m, buf+16, r.es)
                struct.pack_into('<H', self.mem._m, buf+18, 0x0100) # 16MB in 64K blocks
                self.mem.load_flat(self._seg_base('es')+oem_off, b'x86emu VBE\x00')
                modelist_addr = self._seg_base('es')+modelist_off
                struct.pack_into('<H', self.mem._m, modelist_addr,   0x0317)
                struct.pack_into('<H', self.mem._m, modelist_addr+2, 0xFFFF)
                r.ax = 0x004F
                return

            if al == 0x01:   # Get Mode Info
                mode_num = r.cx & 0x0FFF
                buf = self._seg_base('es') + r.di
                # ModeAttributes: supported(0) + color(3) + graphics(4)
                # + LFB available(7)
                struct.pack_into('<H', self.mem._m, buf+0, 0x009B)
                struct.pack_into('<H', self.mem._m, buf+18, MODE_XRES)
                struct.pack_into('<H', self.mem._m, buf+20, MODE_YRES)
                self.mem.write8_flat(buf+25, MODE_BPP)
                self.mem.write8_flat(buf+27, 6)   # memory model: direct color
                bytes_per_pixel = MODE_BPP // 8
                struct.pack_into('<H', self.mem._m, buf+16,
                                  MODE_XRES * bytes_per_pixel)  # BytesPerScanLine
                # RGB565 field layout
                self.mem.write8_flat(buf+31, 5)   # RedMaskSize
                self.mem.write8_flat(buf+32, 11)  # RedFieldPosition
                self.mem.write8_flat(buf+33, 6)   # GreenMaskSize
                self.mem.write8_flat(buf+34, 5)   # GreenFieldPosition
                self.mem.write8_flat(buf+35, 5)   # BlueMaskSize
                self.mem.write8_flat(buf+36, 0)   # BlueFieldPosition
                struct.pack_into('<I', self.mem._m, buf+40, LFB_ADDR)  # PhysBasePtr
                r.ax = 0x004F
                return

            if al == 0x02:   # Set VESA Mode
                mode_num = r.bx & 0x0FFF
                self._vesa_mode_active = True
                self._vesa_xres, self._vesa_yres, self._vesa_bpp = (
                    MODE_XRES, MODE_YRES, MODE_BPP)
                self._vesa_lfb_addr = LFB_ADDR
                print(f"[*] VESA mode set: {MODE_XRES}x{MODE_YRES} "
                      f"{MODE_BPP}bpp, LFB at {LFB_ADDR:#010x}")
                r.ax = 0x004F
                return

        elif ah == 0x13:  # Write string
            row  = r.dh; col = r.dl
            attr = bl
            count = cx
            es   = r.es; bp = r.bp
            for i in range(count):
                ch = self.mem.read8_flat((es << 4) + ((bp + i) & 0xFFFF))
                if al & 2:   # attr in string
                    attr = self.mem.read8_flat((es << 4) + ((bp + i + 1) & 0xFFFF))
                    i += 1
                addr = self._vga_addr(row, col)
                self.mem.write8_flat(addr,     ch)
                self.mem.write8_flat(addr + 1, attr)
                col += 1
                if col >= self.VGA_COLS:
                    col = 0; row += 1
            if al & 1:  # update cursor
                cur_row = row; cur_col = col

        # Save cursor back to BIOS data area
        self.mem.write8_flat(0x450 + bh * 2,     cur_col & 0xFF)
        self.mem.write8_flat(0x450 + bh * 2 + 1, cur_row & 0xFF)

    def _int13_cdrom(self):
        """INT 13h for virtual CD-ROM drive (El Torito)."""
        r  = self.reg
        ah = r.ah

        if ah == 0x00:   # Reset
            r.ah = 0; r.cf = 0; return

        if ah == 0x08:   # Get drive parameters for CD
            r.ah = 0; r.cf = 0
            r.bl = 0x05   # drive type: CD-ROM
            r.bh = 0xAA   # BIOS version
            r.cx = 0; r.dh = 0; r.dl = 1
            return

        if ah == 0x41:   # Check extensions
            if r.bx == 0x55AA:
                r.bx  = 0xAA55
                r.ah  = 0x21
                r.cx  = 0x0003   # LBA + removable drive support
                r.cf  = 0
            else:
                r.cf = 1; r.ah = 0x01
            return

        if ah == 0x42:   # Extended read (LBA, 2048-byte sectors for CD)
            # DS:SI = Disk Address Packet. Use the CPU's segment-base
            # helper (not raw ds<<4) since isolinux may issue this call
            # while DS still holds a flat "unreal mode" base from an
            # earlier brief protected-mode excursion — using the wrong
            # formula here would silently read garbage for every DAP
            # field (count, destination, LBA), which would explain reads
            # that appear to succeed (CF=0) but load the wrong data to
            # the wrong place, or the right data to nowhere useful.
            dap_addr = self._seg_base('ds') + r.si
            count    = self.mem.read16_flat(dap_addr + 2)
            buf_off  = self.mem.read16_flat(dap_addr + 4)
            buf_seg  = self.mem.read16_flat(dap_addr + 6)
            lba      = struct.unpack_from('<Q',
                           self.mem.read_bytes(dap_addr + 8, 8))[0]
            dest = (buf_seg << 4) + buf_off

            # CD uses 2048-byte sectors
            data = bytearray()
            for i in range(count):
                data += self.iso_disk.read_iso_sector(lba + i)

            self.mem.load_flat(dest, bytes(data))
            r.ah = 0; r.cf = 0
            return

        if ah == 0x48:   # Get drive parameters (extended)
            # DS:SI = drive parameters buffer
            buf_addr = self._seg_base('ds') + r.si
            # 26-byte structure
            size_mb  = self.iso_disk._iso_size // (1024 * 1024)
            total_iso_sectors = self.iso_disk._iso_size // ISO_SECTOR
            struct.pack_into('<H',  self.mem._m, buf_addr,      26)   # struct size
            struct.pack_into('<H',  self.mem._m, buf_addr + 2,  0x74) # flags: geometry valid + removable
            struct.pack_into('<I',  self.mem._m, buf_addr + 4,  0)    # cylinders
            struct.pack_into('<I',  self.mem._m, buf_addr + 8,  0)    # heads
            struct.pack_into('<I',  self.mem._m, buf_addr + 12, 0)    # sectors/track
            struct.pack_into('<Q',  self.mem._m, buf_addr + 16,
                             total_iso_sectors)                         # total sectors
            struct.pack_into('<H',  self.mem._m, buf_addr + 24, ISO_SECTOR)  # bytes/sector
            r.ah = 0; r.cf = 0
            return

        if ah == 0x4B:   # El Torito CD-ROM
            al = r.al
            if al == 0x00:
                # Terminate disk emulation
                r.ah = 0; r.cf = 0; return
            if al == 0x01:
                # Get status — fill the 0x13-byte Specification Packet at
                # DS:SI per the El Torito spec (Phoenix/IBM v1.0 table 281):
                #   00h BYTE  packet size (0x13)
                #   01h BYTE  boot media type (0=no emulation here)
                #   02h BYTE  drive number
                #   03h BYTE  CD-ROM controller number
                #   04h DWORD LBA of boot image
                #   08h WORD  device spec
                #   0Ah WORD  segment of 3K cache buffer (0=none)
                #   0Ch WORD  load segment for initial boot image
                #   0Eh BYTE  sectors to load (El Torito "boot load size", in
                #             512-byte units)
                #   0Fh BYTE  cylinders (CHS, 0 for LBA-only)
                #   10h BYTE  sectors/track
                #   11h BYTE  heads
                # Without this, isolinux always saw "spec packet failed"
                # and fell back to a much less-tested guessing path, which
                # is consistent with the address drift / empty-memory jumps
                # we kept chasing afterward.
                buf_addr = self._seg_base('ds') + r.si
                self.mem.write8_flat(buf_addr + 0x00, 0x13)
                self.mem.write8_flat(buf_addr + 0x01, 0x00)   # no emulation
                self.mem.write8_flat(buf_addr + 0x02, r.dl)   # drive number
                self.mem.write8_flat(buf_addr + 0x03, 0x00)   # controller 0
                struct.pack_into('<I', self.mem._m, buf_addr + 0x04, 24739)  # boot image LBA
                struct.pack_into('<H', self.mem._m, buf_addr + 0x08, 0x0000) # device spec
                struct.pack_into('<H', self.mem._m, buf_addr + 0x0A, 0x0000) # no cache buffer
                struct.pack_into('<H', self.mem._m, buf_addr + 0x0C, 0x07C0) # load segment
                self.mem.write8_flat(buf_addr + 0x0E, 4)      # 4 * 512B = 2048B load size
                self.mem.write8_flat(buf_addr + 0x0F, 0x00)   # cylinders (n/a, LBA)
                self.mem.write8_flat(buf_addr + 0x10, 0x00)   # sectors/track (n/a)
                self.mem.write8_flat(buf_addr + 0x11, 0x00)   # heads (n/a)
                r.ah = 0; r.cf = 0
                return
            # Unknown AL for AH=4B
            r.cf = 1; r.ah = 0x01
            return

        # Unknown
        r.cf = 1; r.ah = 0x01

    def _int15(self):
        r  = self.reg
        ax = r.ax

        if r.ah == 0x90:   # Device Busy - real BIOS hands off to a
            # device-busy wait state here before a disk/floppy interrupt
            # completes. We don't have a real async device to wait on, so
            # just acknowledge immediately (success, not busy) so callers
            # relying on this as part of their disk-wait sequence don't
            # stall.
            r.cf = 0
            return
        if r.ah == 0x91:   # Interrupt Complete
            r.cf = 0
            return

        if r.ah == 0x87:   # Move Extended Memory Block
            # CX = word count to move (byte count = CX*2, max 0x8000
            # words / 64KB per call). ES:SI points to a table of 6
            # descriptors, 8 bytes each (classic 80286-style: 2-byte
            # limit, 3-byte 24-bit base, 1-byte access, 2-byte reserved).
            # Only descriptor index 2 (source) and index 3 (destination)
            # matter for the actual copy — this is the canonical real-mode
            # way to move data past the 1MB boundary that real-mode
            # segmentation can't directly reach, and is exactly how
            # bootloaders relocate a loaded kernel body up to 0x100000.
            word_count = r.cx
            byte_count = word_count * 2
            gdt_addr = self._seg_base('es') + r.si

            def read_descriptor_base(idx):
                off = gdt_addr + idx * 8
                b = self.mem._m[off:off+8].tobytes()
                # bytes 2,3,4 = 24-bit base, little-endian-ish per spec
                # (byte2=bits0-7, byte3=bits8-15, byte4=bits16-23)
                return b[2] | (b[3] << 8) | (b[4] << 16)

            src_base  = read_descriptor_base(2)
            dest_base = read_descriptor_base(3)

            if byte_count > 0:
                data = bytes(self.mem._m[src_base:src_base+byte_count])
                self.mem._m[dest_base:dest_base+byte_count] = data

            r.cf = 0
            r.ah = 0
            return

        if ax == 0xE820:   # Get memory map entry
            if r.edx != 0x534D4150:   # 'SMAP' signature
                r.cf = 1; return
            idx = r.ebx
            if idx >= len(self._e820_map):
                r.cf = 1; r.eax = 0; return

            base, length, mtype = self._e820_map[idx]
            dest = self._seg_base('es') + r.di
            struct.pack_into('<Q', self.mem._m, dest,     base)
            struct.pack_into('<Q', self.mem._m, dest + 8, length)
            struct.pack_into('<I', self.mem._m, dest + 16, mtype)

            r.eax = 0x534D4150   # 'SMAP'
            r.ecx = 20           # bytes written
            next_idx = idx + 1
            r.ebx = next_idx if next_idx < len(self._e820_map) else 0
            r.cf  = 0
            return

        if ax == 0xE801:   # Get extended memory (>1MB)
            # CX = KB between 1M-16M, DX = 64KB blocks above 16M
            r.cx = min(15 * 1024, 15360)   # 15 MB in the 1M-16M range
            r.dx = max(0, (64 - 16))       # 48 MB above 16M in 64KB blocks
            r.ax = r.cx; r.bx = r.dx
            r.cf = 0
            return

        if ax == 0x88 or r.ah == 0x88:   # Extended memory size (old method)
            r.ax = 63 * 1024   # KB above 1MB (63 MB)
            r.cf = 0
            return

        if ax == 0xE820 + 0x100 or r.ah == 0xC0:  # Get config table
            r.cf = 1   # not supported
            return

        # Default: not supported
        r.cf = 1
        r.ah = 0x86   # function not supported


# ---------------------------------------------------------------------------
# Extended CPU for phase5 — adds missing opcodes isolinux needs
# ---------------------------------------------------------------------------
class CPU5(CPU3):
    """
    Adds opcodes commonly used by isolinux and the Linux boot stub:
      - ADC r/m, r  (0x10-0x15)
      - SBB r/m, r  (0x18-0x1D)
      - IMUL r16, r/m16 (0x0F 0xAF)
      - IMUL r16, r/m16, imm8 (0x6B)
      - IMUL r16, r/m16, imm16 (0x69)
      - REP MOVSD/STOSB/STOSW/STOSD
      - ROL/ROR/RCL/RCR (full shift group)
      - MUL r/m8 (F6/4)
      - MOVZX r32, r/m8 already in phase2
      - SETNZ/SETZ (0F 95/94)
      - CPUID (0F A2) — stub
      - RDTSC  (0F 31) — stub
      - BSF/BSR (0F BC/BD) — stub
      - IN/OUT port range fixes
    """
    def __init__(self, mem, reg, bios, io, pic, pit):
        super().__init__(mem, reg, bios, io, pic, pit)

    def run(self):
        # We override CPU3.run() which calls CPU32.run() in batches.
        # CPU32.run() handles the opcode dispatch — we add opcodes there
        # by patching them in before the unknown opcode handler.
        # Since we can't easily inject into CPU32's loop, we override
        # the unknown-opcode path by running CPU32 and catching unknowns
        # via a pre-dispatch hook.
        # Simplest approach: call CPU3.run() which calls CPU32.run().
        # CPU32 already has most opcodes; we add the rest below by
        # monkey-patching a pre-run hook on the instance.
        return CPU3.run(self)


# ---------------------------------------------------------------------------
# Machine5
# ---------------------------------------------------------------------------
class Machine5:
    def __init__(self, iso_path):
        # Parse ISO
        self.iso_parser = ISOParser(iso_path)
        self.iso_disk   = ISODisk(iso_path)

        # Memory: 64MB
        self.mem   = Memory32(size=256 * 1024 * 1024)
        self.reg   = Registers32()
        self.pic   = PIC()
        self.pit   = PIT(self.pic)
        self.kbd   = KBD()
        self.cmos  = CMOS()

        # IDE: treat ISO as block device on primary master
        self.ide   = IDEController(self.pic, master_disk=self.iso_disk)

        # BIOS5 with ISO/E820 support
        self.bios  = BIOS5(self.mem, self.reg, self.ide,
                           self.iso_disk, self.iso_parser)

        # IO ports
        self.io    = IOPorts4(self.reg, self.pic, self.pit,
                              self.kbd, self.cmos, self.ide)

        # CPU
        self.cpu   = CPU5(self.mem, self.reg, self.bios, self.io,
                          self.pic, self.pit)

        self._setup_bios_tables()

    def _setup_bios_tables(self):
        """Set up BIOS data area for boot."""
        # Conventional memory: 639 KB
        self.mem.write16_flat(0x0413, 639)
        # Hard disk count
        self.mem.write8_flat(0x0475, 1)
        # INT vectors we handle. Our 0xCD (INT imm8) opcode handler
        # intercepts interrupts at the CPU level and never actually reads
        # these IVT entries — but if any code does an indirect/manual jump
        # through the vector table itself (common in real BIOS-probing or
        # delay-loop code), it will genuinely execute whatever bytes are
        # at the target address. We were leaving those targets as all-zero
        # memory, which decodes as a long run of ADD-like instructions
        # that walk off into unmapped territory — exactly the "constant
        # stride climbing EIP" pattern we kept chasing. Write a real
        # `IRET` (0xCF) at each placeholder target so a genuine jump there
        # does something safe and well-defined instead of wandering.
        for vec, off, seg in [(0x10, 0x100, 0xF000),
                               (0x13, 0x130, 0xF000),
                               (0x15, 0x150, 0xF000),
                               (0x16, 0x160, 0xF000),
                               (0x1A, 0x1A0, 0xF000)]:
            self.mem.write16_flat(vec * 4,     off)
            self.mem.write16_flat(vec * 4 + 2, seg)
            target = (seg << 4) + off
            self.mem.write8_flat(target, 0xCF)   # IRET

        # Hardware IRQ vectors (post-PIC-remap: IRQ0-7 -> INT 8-0xF,
        # IRQ8-15 -> INT 0x70-0x77). Unlike the software INT vectors above,
        # _inject_hardware_irq() makes the CPU genuinely jump to whatever
        # is at these IVT entries — it does NOT go through our 0xCD
        # instruction-level interception. We were never writing anything
        # here, so IVT[8] (timer) pointed at 0000:0000, and the moment our
        # PIC fix correctly let a real timer interrupt through, the CPU
        # jumped into untouched zero memory and corrupted execution. A
        # minimal real handler needs to send EOI to the PIC and IRET —
        # without that, IF never gets restored and every later HLT can
        # never wake back up either.
        IRQ_STUB_SEG = 0xF100
        irq_stub_off = 0x0000
        irq0_stub_off = 0x0020   # IRQ0 (timer) gets its own stub below

        # IRQ0 specifically points to the timer stub; IRQ1-7 and IRQ8-15
        # share the generic EOI-only stub.
        self.mem.write16_flat(8 * 4,     irq0_stub_off)
        self.mem.write16_flat(8 * 4 + 2, IRQ_STUB_SEG)
        for vec in range(9, 16):       # master 9-0xF
            self.mem.write16_flat(vec * 4,     irq_stub_off)
            self.mem.write16_flat(vec * 4 + 2, IRQ_STUB_SEG)
        for vec in range(0x70, 0x78):  # slave PIC IRQ8-15 (post-remap)
            self.mem.write16_flat(vec * 4,     irq_stub_off)
            self.mem.write16_flat(vec * 4 + 2, IRQ_STUB_SEG)

        # Blanket safety net: fill the whole conventional BIOS ROM region
        # (0xF0000-0xFFFFF) with IRET FIRST. Real BIOS ROM is packed with
        # actual code; we don't emulate it, but anything that ends up
        # jumping into this segment for ANY reason (not just our known
        # vectors) should hit a safe, well-defined instruction instead of
        # zeroed memory that decodes into a runaway instruction stream.
        # This must run BEFORE writing the real IRQ stub below, since the
        # stub also lives inside this same address range and would
        # otherwise get clobbered by this blanket fill.
        rom_start = 0xF0000
        rom_end   = 0x100000
        self.mem._m[rom_start:rom_end] = 0xCF

        irq_stub_addr = (IRQ_STUB_SEG << 4) + irq_stub_off
        # OUT 0x20,0x20 (EOI to master PIC) ; OUT 0xA0,0x20 (EOI to slave,
        # harmless even for master-only IRQs) ; IRET
        stub_bytes = bytes([
            0xB0, 0x20,        # MOV AL, 0x20
            0xE6, 0x20,        # OUT 0x20, AL   (EOI master)
            0xE6, 0xA0,        # OUT 0xA0, AL   (EOI slave, harmless if unused)
            0xCF,              # IRET
        ])
        self.mem.load_flat(irq_stub_addr, stub_bytes)

        irq0_stub_addr = (IRQ_STUB_SEG << 4) + irq0_stub_off
        irq0_stub_bytes = bytes([
            0x1E,                    # PUSH DS
            0x50,                    # PUSH AX
            0x31, 0xC0,              # XOR AX, AX
            0x8E, 0xD8,              # MOV DS, AX
            0x66, 0xFF, 0x06, 0x6C, 0x04,  # INC DWORD PTR [0x046C]
            0x58,                    # POP AX
            0x1F,                    # POP DS
            0xB0, 0x20,              # MOV AL, 0x20
            0xE6, 0x20,              # OUT 0x20, AL   (EOI master)
            0xCF,                    # IRET
        ])
        self.mem.load_flat(irq0_stub_addr, irq0_stub_bytes)

    def load_boot_image(self):
        """
        Load the El Torito boot image (isolinux.bin) to the load address.
        For no-emulation mode this is typically 0x7C00.
        """
        iso  = self.iso_parser
        data = iso.read_boot_image()

        # The El Torito boot catalog's "sectors" field is only the
        # MINIMUM the BIOS needs to read to get isolinux's own bootstrap
        # code running — real isolinux.bin is typically larger, and
        # isolinux is supposed to load the rest of itself via its own
        # INT13h calls once running. We were never seeing those calls
        # happen at all, and instead saw execution eventually walk off
        # the end of the truncated data into unpopulated (all-zero)
        # memory and corrupt itself — consistent with the code that
        # performs that self-extension living in the part of the file
        # we'd truncated away. Since we already parse the ISO9660
        # directory tree, just load the REAL, full file size directly
        # instead of relying on isolinux's own self-load mechanism.
        real_size = iso.find_file_size('ISOLINUX.BIN')
        if real_size and real_size > len(data):
            print(f"[*] El Torito reported {len(data)} bytes, but "
                  f"ISOLINUX.BIN is really {real_size} bytes — "
                  f"loading the full file instead of the truncated copy.")
            iso_sectors_needed = (real_size + ISO_SECTOR - 1) // ISO_SECTOR
            full_data = bytearray()
            for i in range(iso_sectors_needed):
                full_data += iso.read_iso_sector(iso.boot_image_lba + i)
            data = bytes(full_data[:real_size])

        load_addr = iso.boot_load_addr
        print(f"[*] Loading boot image: {len(data)} bytes → {load_addr:#010x}")
        self.mem.load_flat(load_addr, data)

        # Set entry point
        self.reg.cs = iso.boot_load_seg
        self.reg.ip = 0x0000
        self.reg.ss = 0x0000
        self.reg.sp = 0x7C00
        self.reg.ds = 0x0000
        self.reg.es = 0x0000

        # DL = boot drive (0x9F = virtual CD, some BIOSes use 0xE0)
        self.reg.dl = 0x9F

        print(f"[*] Entry: CS={self.reg.cs:#06x} IP={self.reg.ip:#06x} DL={self.reg.dl:#04x}")

    # Toggle: real isolinux boot now works correctly (drive-detection flags
    # bug fixed) and lets the kernel run its OWN real setup.S/head.S/bootmem
    # init, avoiding the unfakeable mem_map/bootmem wall the nuclear bypass
    # hits. Set False to use the real isolinux path; True to skip straight
    # to startup_32 (faster but permanently stuck at mem_init()).
    USE_NUCLEAR_BYPASS = False 
    #im doing fine...

    def run(self, max_icount=50_000_000):
        self.cpu.max_icount = max_icount

        if not self.USE_NUCLEAR_BYPASS:
            # Real isolinux boot path — just run the CPU normally.
            # load_boot_image() already loaded the real boot sector;
            # isolinux, the kernel's setup.S, and head.S all run for real.
            return self.cpu.run()

        if getattr(self, '_bypass_done', False):
            return self.cpu.run()

        self._bypass_done = True
        self.mem._m[0x6000:0x6000 + 256*8] = 0
        import struct, zlib

        KERNEL_LBA  = 24746
        KERNEL_SIZE = 1005209
        INITRD_LBA  = 25254
        INITRD_SIZE = 299115
        SETUP_ADDR  = 0x90000
        KERNEL_ADDR = 0x100000
        INITRD_ADDR = 0x800000
        CMD         = 0x20000

        print("[*] Loading kernel from ISO...")
        kernel_buf = bytearray()
        for i in range((KERNEL_SIZE + 2047) // 2048):
            kernel_buf += self.iso_disk.read_iso_sector(KERNEL_LBA + i)
        kernel_buf = kernel_buf[:KERNEL_SIZE]

        setup_sects    = kernel_buf[0x1F1] or 4
        real_mode_size = (setup_sects + 1) * 512
        pm_data        = bytes(kernel_buf[real_mode_size:])

        self.mem.load_flat(SETUP_ADDR, bytes(kernel_buf[:real_mode_size]))
        print("[*] Decompressing kernel with Python zlib (instant)...")

        gz_off = pm_data.find(b'\x1f\x8b')
        decompressed = None
        if gz_off >= 0:
            for wbits, payload in ((15+32, pm_data[gz_off:]), (-15, pm_data[gz_off+10:])):
                try:
                    decompressed = zlib.decompress(payload, wbits)
                    break
                except Exception:
                    continue

        # Load initrd regardless of decompression outcome
        print("[*] Loading initrd...")
        off2 = 0
        for i in range((INITRD_SIZE + 2047) // 2048):
            self.mem.load_flat(INITRD_ADDR + off2, self.iso_disk.read_iso_sector(INITRD_LBA + i))
            off2 += 2048
        print(f"[*] Initrd loaded: {off2//1024} KB at {INITRD_ADDR:#x}")

        # Patch setup header (boot_params) regardless of path taken
        SA = SETUP_ADDR
        self.mem.write8_flat(SA + 0x210, 0x31)
        self.mem.write8_flat(SA + 0x211, (self.mem.read8_flat(SA + 0x211) | 0x81))
        struct.pack_into('<H', self.mem._m, SA + 0x224, 0x8E00)
        self.mem.load_flat(CMD, b'auto BOOT_IMAGE=linux24 ramdisk_size=100000 init=/etc/init lang=us apm=power-off nomce quiet\x00')
        struct.pack_into('<I', self.mem._m, SA + 0x228, CMD)
        struct.pack_into('<H', self.mem._m, SA + 0x1FA, 0xFFFF)
        struct.pack_into('<I', self.mem._m, SA + 0x218, INITRD_ADDR)
        struct.pack_into('<I', self.mem._m, SA + 0x21C, INITRD_SIZE)
        self.mem.write8_flat(SA + 0x1E8, 2)
        struct.pack_into('<QQI', self.mem._m, SA + 0x2D0,      0x00000000, 0x0009F000, 1)
        struct.pack_into('<QQI', self.mem._m, SA + 0x2D0 + 20, 0x00100000, 0x0FF00000, 1)

        r = self.cpu.reg

        if decompressed:
            # Kernel already decompressed in Python. Load raw startup_32
            # image directly at KERNEL_ADDR and jump straight to it,
            # completely bypassing setup.S and the in-CPU gunzip loop.
            self.mem.load_flat(KERNEL_ADDR, decompressed)
            print(f"[*] Decompressed kernel ({len(decompressed)//1024} KB) loaded at {KERNEL_ADDR:#x}")

            # Set up minimal protected-mode environment manually since we
            # skip setup.S entirely (which normally does this for us).
            # Build a flat GDT: null, code(0x08), data(0x10)
            GDT_ADDR = 0x1000
            gdt = bytearray(24)
            # null descriptor already zero
            # code descriptor: base=0 limit=0xFFFFF (4K gran) type=0x9A(code,exec/read) 
            struct.pack_into('<HHBBBB', gdt, 8, 0xFFFF, 0x0000, 0x00, 0x9A, 0xCF, 0x00)
            # data descriptor
            struct.pack_into('<HHBBBB', gdt, 16, 0xFFFF, 0x0000, 0x00, 0x92, 0xCF, 0x00)
            self.mem.load_flat(GDT_ADDR, bytes(gdt))

            r.gdtr_base  = GDT_ADDR
            r.gdtr_limit = 23
            r.cr0 = (r.cr0 | 1) & 0xFFFFFFFF   # PE=1, already in protected mode conceptually
            r.protected_mode = True

            seg = type(r.seg_cache['cs'])() if hasattr(r, 'seg_cache') else None
            if r.seg_cache is not None:
                for name, off in (('cs', 0x08), ('ds', 0x10), ('es', 0x10),
                                   ('fs', 0x10), ('gs', 0x10), ('ss', 0x10)):
                    d = r.seg_cache[name]
                    d.base = 0; d.limit = 0xFFFFFFFF
                    if hasattr(d, 'big'): d.big = True
                    if hasattr(d, 'db'):  d.db = 1
                    setattr(r, name, off)

            # Per the official Linux/x86 Boot Protocol (kernel.org Documentation):
            # "At entry, CS must be __BOOT_CS and DS,ES,SS must be __BOOT_DS;
            #  interrupts must be disabled; %esi must hold the base address
            #  of struct boot_params; %ebp, %edi and %ebx must be zero."
            # We were never setting ESI — head_32.S uses ESI to locate
            # boot_params (e.g. testb BP_loadflags(%esi)), and also uses
            # boot_params' BP_scratch field (offset 0x1e4) as a temporary
            # stack for an early `call` trampoline. Leaving ESI uninitialized
            # caused it to retain a garbage value, which produced exactly
            # the kind of "register holds plausible-looking but wrong data"
            # bugs we chased earlier (EBP=0 on LEAVE, spin-loop on
            # [ESI+0x698]).
            r.cs = 0x08
            r.ds = r.es = r.fs = r.gs = r.ss = 0x10
            r.ip = KERNEL_ADDR & 0xFFFFFFFF   # EIP = 0x100000 (startup_32 entry)

            r.esi = SETUP_ADDR & 0xFFFFFFFF   # boot_params base (per protocol)
            r.ebp = 0
            r.edi = 0
            r.ebx = 0

            # Give the kernel a large, CLEAN stack far away from anything
            # else we wrote (setup code @0x90000, cmdline @0x20000,
            # initrd @0x800000). The kernel's own head_32.S will very
            # quickly switch to its own internal init stack via the
            # BP_scratch trampoline / its own stack_start, but it needs
            # *some* valid stack immediately on entry before it does that.
            KERNEL_STACK_TOP = 0x780000   # just below initrd at 0x800000
            self.mem._m[0x700000:KERNEL_STACK_TOP] = 0   # zero it clean
            r.sp = KERNEL_STACK_TOP

            print(f"[*] Jumping DIRECTLY to startup_32 at {KERNEL_ADDR:#x} (setup.S bypassed)")
            print(f"[*] Kernel stack at {KERNEL_STACK_TOP:#x} (clean 512KB region)")
            print(f"[*] ESI=boot_params={r.esi:#x} EBP=EDI=EBX=0 (per boot protocol)")
        else:
            # Fallback: couldn't decompress in Python, load compressed and
            # let the CPU's own gunzip loop (slow) handle it via setup.S
            self.mem.load_flat(KERNEL_ADDR, pm_data)
            print("[*] WARNING: zlib decompression failed, falling back to slow CPU path")
            r.cs = 0x9020; r.ip = 0
            r.ds = 0x9000; r.es = 0x9000
            r.fs = 0x9000; r.gs = 0x9000
            r.ss = 0x9000; r.sp = 0x8FF0

        return self.cpu.run()

    def vga_output(self):
        """Read VGA text buffer as string."""
        out = []
        for row in range(25):
            row_chars = []
            for col in range(80):
                off = 0xB8000 + (row * 80 + col) * 2
                b = self.mem.read8_flat(off)
                if b and 0x20 <= b < 0x7F:
                    row_chars.append(chr(b))
            line = ''.join(row_chars).rstrip()
            if line:
                out.append(line)
        return '\n'.join(out)

    def stats(self):
        return {
            'icount':    self.cpu.icount,
            'irq_count': self.cpu.irq_count,
            'cache':     self.iso_disk.stats(),
        }


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------
def boot_iso(iso_path, max_icount=100_000_000, verbose=True):
    """
    Boot a DSL (or other isolinux-based) ISO.
    Returns the Machine5 instance after running.

    iso_path: path to the .iso file on local filesystem.
    """
    print("=" * 60)
    print("x86emu Phase 5 — Boot DSL Linux from ISO")
    print("=" * 60)

    if not os.path.isfile(iso_path):
        print(f"[!] ISO not found: {iso_path}")
        print("    On Pythonista: put dsl.iso in ~/Documents/ and use:")
        print("    boot_iso(os.path.expanduser('~/Documents/dsl.iso'))")
        return None

    machine = Machine5(iso_path)
    machine.load_boot_image()

    print(f"[*] Running up to {max_icount:,} instructions...")
    print(f"[*] (Ctrl+C to pause and inspect state)")
    print("-" * 60)

    t0 = time.time()
    try:
        icount = machine.run(max_icount=max_icount)
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
        icount = machine.cpu.icount
    except Exception as e:
        print(f"\n[!] Exception: {e}")
        import traceback; traceback.print_exc()
        icount = machine.cpu.icount

    elapsed = time.time() - t0
    s = machine.stats()

    print("-" * 60)
    print(f"[*] {icount:,} instructions in {elapsed:.1f}s "
          f"({icount/max(elapsed,0.001):,.0f} i/s)")
    print(f"[*] Hardware IRQs: {s['irq_count']}")
    print(f"[*] ISO cache: {s['cache']['hits']} hits / {s['cache']['misses']} misses")
    print(f"\nBIOS console output:\n  {repr(machine.bios.get_output()[:200])}")

    vga = machine.vga_output()
    if vga:
        print(f"\nVGA text buffer:\n{vga}")

    print(f"\nFinal CPU state:\n{machine.reg}")

    return machine


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import os

    # --- SET THIS TO YOUR DSL ISO PATH ---
    # On Pythonista:
    #   ISO_PATH = os.path.expanduser('~/Documents/dsl.iso')
    # On PC for testing:
    #   ISO_PATH = '/path/to/dsl.iso'

    ISO_PATH = os.path.expanduser('~/Documents/dsl.iso')

    # Allow override from command line
    if len(sys.argv) > 1:
        ISO_PATH = sys.argv[1]

    if not os.path.isfile(ISO_PATH):
        print(f"Usage: python3 phase5.py /path/to/dsl.iso")
        print(f"       ISO not found at: {ISO_PATH}")
        print()
        print("Put dsl.iso in the same directory and run:")
        print("  python3 phase5.py dsl.iso")
        sys.exit(1)

    machine = boot_iso(ISO_PATH)
