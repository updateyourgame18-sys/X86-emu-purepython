"""
phase4.py — x86emu Phase 4: IDE Block Device + Disk Boot
Imports Machine3 from phase3.py, adds IDE controller.

New in this phase:
  - IDE ATA PIO controller (primary channel)
  - Disk image backend (local file, HTTP lazy, or in-memory bytes)
  - INT 13h BIOS disk services (AH=00/02/08/41/42) — real mode disk reads
  - IRQ14 wired through PIC for DMA-less PIO completion
  - BIOS disk parameter table at 0x104 (for boot sector geometry queries)
  - Full boot path: MBR → bootloader → 'kernel' loaded from disk
  - Test: boots a hand-built disk image, kernel stub prints "DISK OK"

For Pythonista / DSL:
  - Swap BytesDisk for HTTPDisk pointing at your PC-served DSL image
  - Phase 5 will handle the actual Linux kernel boot sequence
"""

import struct, sys, time, os
import importlib

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
p3 = importlib.import_module('phase3')
p2 = importlib.import_module('phase2')
disk_mod = importlib.import_module('disk')
ide_mod  = importlib.import_module('ide')

Machine3    = p3.Machine3
Memory32    = p2.Memory32
Registers32 = p2.Registers32
BIOS        = p2.BIOS
CPU32       = p2.CPU32
CPU3        = p3.CPU3
IOPorts3    = p3.IOPorts3
PIC         = p3.PIC
PIT         = p3.PIT
KBD         = p3.KBD
CMOS        = p3.CMOS
sign8       = p2.sign8

Disk        = disk_mod.Disk
DiskBuilder = disk_mod.DiskBuilder
BytesDisk   = disk_mod.BytesDisk
FileDisk    = disk_mod.FileDisk
HTTPDisk    = disk_mod.HTTPDisk
IDEController = ide_mod.IDEController


# ---------------------------------------------------------------------------
# Extended BIOS with INT 13h disk services
# ---------------------------------------------------------------------------
class BIOS4(BIOS):
    """
    Extends Phase 1 BIOS with INT 13h disk services.
    Real-mode bootloaders use INT 13h to read sectors before switching
    to protected mode (GRUB, isolinux, etc all do this).
    """
    def __init__(self, mem, reg, ide):
        super().__init__(mem, reg)
        self.ide = ide

    def interrupt(self, num):
        if num == 0x13:
            self._int13()
        else:
            super().interrupt(num)

    def _int13(self):
        r   = self.reg
        ah  = r.ah
        dl  = r.dl   # drive: 0x80=first HDD, 0x9F=virtual CD

        # Only respond to drives we actually have
        # 0x00-0x7F = floppy (none), 0x80 = first HDD, 0x9F = virtual CD
        # All other drives → error
        have_drive = (dl == 0x80 and self.ide.drives[0] is not None)

        if ah == 0x00:   # Reset disk — always succeed
            r.ah = 0; r.cf = 0; return

        if ah == 0x02:   # Read sectors (CHS)
            if not have_drive:
                r.ah = 0x01; r.cf = 1; return
            al  = r.al
            ch  = r.ch; cl = r.cl; dh = r.dh
            es  = r.es; bx = r.bx
            cylinder = ch | ((cl >> 6) << 8)
            sector   = (cl & 0x3F) - 1
            head     = dh
            heads = 16; spt = 63
            lba = (cylinder * heads + head) * spt + sector
            drv  = self.ide.drives[0]
            data = drv.disk.read_sectors(lba, al)
            dest = (es << 4) + bx
            self.mem.load_flat(dest, data[:al * 512])
            r.ah = 0; r.al = al; r.cf = 0
            return

        if ah == 0x08:   # Get drive parameters
            if not have_drive:
                r.cf = 1; r.ah = 0x01; return
            drv   = self.ide.drives[0]
            sects = drv.disk.sector_count
            heads = 16; spt = 63
            cyls  = min(sects // (heads * spt), 1023)
            r.ah  = 0; r.al = 0
            r.ch  = cyls & 0xFF
            r.cl  = spt | ((cyls >> 8) << 6)
            r.dh  = heads - 1
            r.dl  = 1
            r.cf  = 0
            return

        if ah == 0x41:   # Check extensions present
            # Only respond to drives we actually have
            if have_drive and r.bx == 0x55AA:
                r.bx = 0xAA55; r.ah = 0x21; r.cx = 0x0003; r.cf = 0
            else:
                r.cf = 1; r.ah = 0x01
            return

        if ah == 0x42:   # Extended read (LBA)
            if not have_drive:
                r.cf = 1; r.ah = 0x01; return
            dap_addr = (r.ds << 4) + r.si
            count    = self.mem.read16_flat(dap_addr + 2)
            buf_off  = self.mem.read16_flat(dap_addr + 4)
            buf_seg  = self.mem.read16_flat(dap_addr + 6)
            lba      = struct.unpack_from('<Q', self.mem.read_bytes(dap_addr + 8, 8))[0]
            drv  = self.ide.drives[0]
            data = drv.disk.read_sectors(lba, count)
            dest = (buf_seg << 4) + buf_off
            self.mem.load_flat(dest, data)
            r.ah = 0; r.cf = 0
            return

        # Unknown subfunction
        r.cf = 1; r.ah = 0x01



# ---------------------------------------------------------------------------
# Extended IO ports — adds IDE to Phase 3 IO
# ---------------------------------------------------------------------------
class IOPorts4(IOPorts3):
    def __init__(self, reg, pic, pit, kbd, cmos, ide):
        super().__init__(reg, pic, pit, kbd, cmos)
        self.ide = ide

    def read(self, port):
        # Primary IDE channel
        if 0x1F0 <= port <= 0x1F7 or port == 0x3F6:
            return self.ide.read(port)
        # Secondary IDE channel
        if 0x170 <= port <= 0x177 or port == 0x376:
            return self.ide.read_secondary(port)
        return super().read(port)

    def write(self, port, val):
        if 0x1F0 <= port <= 0x1F7 or port == 0x3F6:
            self.ide.write(port, val)
            return
        if 0x170 <= port <= 0x177 or port == 0x376:
            self.ide.write_secondary(port, val)
            return
        super().write(port, val)


# ---------------------------------------------------------------------------
# Machine4
# ---------------------------------------------------------------------------
class Machine4:
    def __init__(self, disk=None):
        self.mem   = Memory32(size=64 * 1024 * 1024)   # 64 MB for kernel
        self.reg   = Registers32()
        self.pic   = PIC()
        self.pit   = PIT(self.pic)
        self.kbd   = KBD()
        self.cmos  = CMOS()
        self.ide   = IDEController(self.pic, master_disk=disk)
        self.bios  = BIOS4(self.mem, self.reg, self.ide)
        self.io    = IOPorts4(self.reg, self.pic, self.pit,
                              self.kbd, self.cmos, self.ide)
        self.cpu   = CPU3(self.mem, self.reg, self.bios, self.io,
                          self.pic, self.pit)

        # Write BIOS disk parameter table to 0x104 (IVT slot for INT 41h)
        self._setup_bios_tables()

    def _setup_bios_tables(self):
        """
        Set up BIOS data area and IVT entries expected by bootloaders.
        0x0000:0x0413 — memory size in KB (below 640K line)
        0x0040:0x0075 — number of hard disks
        INT 13h vector at 0x0000:0x004C (4 bytes: offset, segment)
        """
        # Memory size: 639 KB (conventional)
        self.mem.write16_flat(0x0413, 639)
        # Number of hard disks at BIOS data area 0x475
        self.mem.write8_flat(0x0475, 1 if self.ide.drives[0] else 0)
        # IVT: INT 13h — point to a real-mode BIOS stub
        # We don't need a real handler because CPU handles INT via BIOS object
        # Just make sure it's non-zero so bootloaders don't freak out
        self.mem.write16_flat(0x004C, 0x0000)  # offset
        self.mem.write16_flat(0x004E, 0xF000)  # segment (ROM area)

    def load_at(self, addr, data):
        self.mem.load_flat(addr, data)

    def set_entry(self, cs, ip):
        self.reg.cs = cs
        self.reg.ip = ip
        self.reg.ss = 0x0000
        self.reg.sp = 0x7C00
        self.reg.ds = 0x0000
        self.reg.es = 0x0000

    def run(self, max_icount=20_000_000):
        self.cpu.max_icount = max_icount
        return self.cpu.run()

    def stats(self):
        return {
            'icount':    self.cpu.icount,
            'irq_count': self.cpu.irq_count,
            'ide':       self.ide.stats(),
        }


# ---------------------------------------------------------------------------
# Build a real-mode MBR bootloader that uses INT 13h
# to load our test kernel and then executes it
# ---------------------------------------------------------------------------
def build_test_mbr(kernel_load_addr=0x10000, kernel_sectors=1):
    """
    Build a 512-byte MBR using a clean two-pass approach so
    message address arithmetic is always exact.
    """
    load_seg = (kernel_load_addr >> 4) & 0xFFFF
    boot_msg = b"Booting x86emu disk...\r\n\x00"

    def make_code(msg_addr):
        c = bytearray()
        c += b'\x31\xC0\x8E\xD8\x8E\xC0\x8E\xD0'      # XOR AX,AX; MOV DS/ES/SS,AX
        c += b'\xBC\x00\x7C'                             # MOV SP, 0x7C00
        c += b'\xFA'                                     # CLI
        # Print loop
        c += b'\xBE' + struct.pack('<H', msg_addr)       # MOV SI, msg_addr
        loop = len(c)
        c += b'\xAC'                                     # LODSB
        c += b'\x20\xC0'                                 # AND AL, AL
        jz_pos = len(c)
        c += b'\x74\x00'                                 # JZ done_print (patch below)
        c += b'\xB4\x0E\xBB\x07\x00\xCD\x10'            # INT 10h TTY
        c += bytes([0xEB, (loop - len(c) - 2) & 0xFF])  # JMP loop
        # done_print — patch JZ offset
        done_print = len(c)
        c[jz_pos + 1] = (done_print - jz_pos - 2) & 0xFF
        # INT 13h read
        c += bytes([0xB8, kernel_sectors & 0xFF, 0x02])  # MOV AX, 0x0200|count
        c += b'\x31\xDB'                                 # XOR BX, BX
        c += b'\xB9\x02\x00'                             # MOV CX, 0x0002 (CH=0 cyl, CL=2 sector)
        c += b'\xBA\x80\x00'                             # MOV DX, 0x0080 (DH=0 head, DL=0x80 hdd)
        # Load ES = load_seg without touching AX
        c += bytes([0xBB, load_seg & 0xFF, (load_seg >> 8) & 0xFF])  # MOV BX, load_seg
        c += b'\x8E\xC3'                                 # MOV ES, BX
        c += b'\x31\xDB'                                 # XOR BX, BX (buffer offset = 0)
        c += b'\xCD\x13'                                 # INT 13h
        c += b'\x72\x07'                                 # JC error
        # Far jump to kernel: CS=load_seg, IP=0
        c += b'\xEA' + struct.pack('<HH', 0x0000, load_seg)
        # error: print '!' halt
        c += b'\xB4\x0E\xB0\x21\xBB\x07\x00\xCD\x10\xF4\xEB\xFE'
        return bytes(c)

    # Two-pass: first pass to measure, second to fix address
    code_v1  = make_code(0x0000)
    msg_addr = 0x7C00 + len(code_v1)
    code     = make_code(msg_addr)

    assert len(code) + len(boot_msg) <= 510, \
        f"MBR too large: {len(code)+len(boot_msg)}"
    mbr = bytearray(512)
    mbr[:len(code)] = code
    mbr[len(code):len(code)+len(boot_msg)] = boot_msg
    mbr[510] = 0x55
    mbr[511] = 0xAA
    return bytes(mbr)


def build_kernel_stub():
    """
    Minimal real-mode 'kernel' stub loaded at 0x10000 (CS=0x1000, IP=0).
    Prints "DISK OK - Phase 4 Pass!" via INT 10h then halts.
    """
    msg = b"DISK OK - Phase 4 Pass!\r\n\x00"

    def make_stub(msg_off):
        s = bytearray()
        s += b'\xFA'                        # CLI
        s += b'\xB8\x00\x10'               # MOV AX, 0x1000
        s += b'\x8E\xD8'                   # MOV DS, AX
        s += b'\x8E\xC0'                   # MOV ES, AX
        s += b'\x31\xC0'                   # XOR AX, AX
        s += b'\x8E\xD0'                   # MOV SS, AX
        s += b'\xBC\x00\x7C'               # MOV SP, 0x7C00
        s += b'\xFB'                       # STI
        # SI = msg_off (relative to DS=0x1000, so linear = 0x10000 + msg_off)
        s += b'\xBE' + struct.pack('<H', msg_off)
        loop = len(s)
        s += b'\xAC'                       # LODSB
        s += b'\x20\xC0'                   # AND AL, AL
        jz_pos = len(s)
        s += b'\x74\x00'                   # JZ done (patch)
        s += b'\xB4\x0E\xBB\x07\x00'      # MOV AH,0xE; MOV BX,7
        s += b'\xCD\x10'                   # INT 10h
        s += bytes([0xEB, (loop - len(s) - 2) & 0xFF])  # JMP loop
        done = len(s)
        s[jz_pos+1] = (done - jz_pos - 2) & 0xFF
        s += b'\xF4\xEB\xFE'               # HLT; JMP $
        return bytes(s)

    # Two-pass: find where msg lands
    stub_v1  = make_stub(0)
    msg_off  = len(stub_v1)
    stub     = make_stub(msg_off) + msg

    # Pad to sector boundary
    while len(stub) % 512:
        stub += b'\x00'
    return stub


# ---------------------------------------------------------------------------
# Build complete disk image: MBR + kernel stub
# ---------------------------------------------------------------------------
def build_test_image():
    kernel  = build_kernel_stub()
    k_sects = len(kernel) // 512
    mbr     = build_test_mbr(kernel_load_addr=0x10000, kernel_sectors=k_sects)
    image   = mbr + kernel
    return image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 55)
    print("x86emu Phase 4 — IDE Block Device + Disk Boot")
    print("=" * 55)

    # Build test disk image in memory
    image = build_test_image()
    print(f"[*] Test disk image: {len(image)} bytes "
          f"({len(image)//512} sectors)")

    disk = BytesDisk(image)

    machine = Machine4(disk=disk)

    # Load MBR into emulated RAM at 0x7C00 (BIOS puts it there)
    machine.load_at(0x7C00, image[:512])
    machine.set_entry(0x0000, 0x7C00)

    # Zero IDT
    machine.mem._m[0x6000:0x6000 + 256*8] = 0

    print(f"[*] Entry: CS=0000 IP=7C00 (MBR)")
    print(f"[*] Running...\n")
    print("-" * 55)

    t0 = time.time()
    try:
        icount = machine.run(max_icount=5_000_000)
    except KeyboardInterrupt:
        print("\n[*] Interrupted")
        icount = machine.cpu.icount
    except Exception as e:
        print(f"\n[!] Exception: {e}")
        import traceback; traceback.print_exc()
        icount = machine.cpu.icount

    elapsed = time.time() - t0
    print("-" * 55)

    s = machine.stats()
    print(f"\n[*] {icount:,} instructions in {elapsed:.3f}s "
          f"({icount/max(elapsed,0.001):,.0f} i/s)")
    print(f"[*] Hardware IRQs: {s['irq_count']}")
    print(f"[*] IDE IRQs fired: {s['ide'].get('irq_fired', 0)}")
    disk_s = s['ide'].get('disk', {})
    print(f"[*] Disk cache hits/misses: "
          f"{disk_s.get('hits',0)}/{disk_s.get('misses',0)}")

    bios_out = machine.bios.get_output()
    print(f"\nBIOS output:\n  {repr(bios_out)}")

    if "DISK OK" in bios_out and "Phase 4" in bios_out:
        print("\n✓ Phase 4 PASS — MBR loaded, INT 13h disk read worked, "
              "kernel stub executed")
    elif "Booting" in bios_out:
        print("\n~ Phase 4 PARTIAL — MBR ran but kernel didn't print output")
        print(f"  Final: CS={machine.reg.cs:04X} IP={machine.reg.ip:08X} "
              f"halted={machine.cpu.halted}")
    else:
        print("\n✗ Phase 4 FAIL — MBR didn't even print boot message")
        print(f"  CF={machine.reg.cf} AH={machine.reg.ah:#04x}")
        print(machine.reg)

    print()
    print("=" * 55)
    print("To use with a real disk image (local file):")
    print("  disk = FileDisk('dsl.img')")
    print("  machine = Machine4(disk=disk)")
    print()
    print("To use with HTTP lazy fetch (Pythonista):")
    print("  disk = HTTPDisk('http://192.168.1.X:8080/dsl.img')")
    print("  machine = Machine4(disk=disk)")
    print("=" * 55)
