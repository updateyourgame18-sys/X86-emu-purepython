"""
ide.py — ATA PIO IDE controller emulator for x86emu Phase 4

Emulates the primary IDE channel (ports 0x1F0-0x1F7, 0x3F6).
Supports:
  - ATA IDENTIFY (0xEC)
  - READ SECTORS PIO (0x20)
  - WRITE SECTORS PIO (0x30)
  - LBA28 addressing
  - BSY/DRQ/RDY status bits
  - IRQ14 on completion (wired to PIC)

Secondary channel (0x170-0x177) stubbed as empty.

Port map (primary):
  0x1F0 — Data register (16-bit reads/writes)
  0x1F1 — Error (read) / Features (write)
  0x1F2 — Sector count
  0x1F3 — LBA bits 0-7
  0x1F4 — LBA bits 8-15
  0x1F5 — LBA bits 16-23
  0x1F6 — Drive/Head (LBA bits 24-27, drive select, LBA mode bit)
  0x1F7 — Status (read) / Command (write)
  0x3F6 — Alternate status (read) / Device control (write)
"""

import struct

# ATA status bits
ATA_SR_BSY  = 0x80   # Busy
ATA_SR_DRDY = 0x40   # Drive ready
ATA_SR_DF   = 0x20   # Drive write fault
ATA_SR_DSC  = 0x10   # Drive seek complete
ATA_SR_DRQ  = 0x08   # Data request
ATA_SR_CORR = 0x04   # Corrected data
ATA_SR_IDX  = 0x02   # Index
ATA_SR_ERR  = 0x01   # Error

# ATA error bits
ATA_ER_BBK  = 0x80   # Bad block
ATA_ER_UNC  = 0x40   # Uncorrectable data
ATA_ER_MC   = 0x20   # Media changed
ATA_ER_IDNF = 0x10   # ID not found
ATA_ER_MCR  = 0x08   # Media change requested
ATA_ER_ABRT = 0x04   # Aborted command
ATA_ER_TK0NF= 0x02   # Track 0 not found
ATA_ER_AMNF = 0x01   # Address mark not found

# ATA commands
ATA_CMD_READ_PIO      = 0x20
ATA_CMD_WRITE_PIO     = 0x30
ATA_CMD_IDENTIFY      = 0xEC
ATA_CMD_SET_FEATURES  = 0xEF
ATA_CMD_CACHE_FLUSH   = 0xE7

SECTOR_SIZE = 512


class ATADrive:
    """
    Single ATA drive backed by a disk.Disk instance.
    Handles command processing and data buffering.
    """
    def __init__(self, disk, drive_num=0):
        self.disk      = disk
        self.drive_num = drive_num   # 0=master, 1=slave

        # Registers
        self.error      = 0
        self.sec_count  = 1
        self.lba0       = 0   # LBA[7:0]
        self.lba1       = 0   # LBA[15:8]
        self.lba2       = 0   # LBA[23:16]
        self.lba3       = 0   # LBA[27:24] + flags in drive/head reg

        # State machine
        self.status     = ATA_SR_DRDY   # ready, not busy
        self._cmd       = 0
        self._drq_buf   = bytearray()   # data buffer for DRQ transfers
        self._drq_pos   = 0             # read position in buffer
        self._drq_write = bytearray()   # write buffer accumulator

        # Build IDENTIFY response once
        self._identify_data = self._build_identify()

    def _build_identify(self) -> bytes:
        """
        Build a 512-byte IDENTIFY DEVICE response.
        Enough for Linux to detect geometry and capabilities.
        """
        d = bytearray(512)

        sectors = self.disk.sector_count

        # Word 0: general config (0x0040 = fixed disk)
        struct.pack_into('<H', d, 0,   0x0040)
        # Word 1: cylinders
        cyls = min(sectors // (16 * 63), 65535)
        struct.pack_into('<H', d, 2,   cyls)
        # Word 3: heads
        struct.pack_into('<H', d, 6,   16)
        # Word 6: sectors per track
        struct.pack_into('<H', d, 12,  63)

        # Words 10-19: serial number (20 chars, space padded, byte-swapped)
        serial = b'X86EMU0000          '
        for i, b in enumerate(serial[:20]):
            d[20 + i] = b

        # Words 23-26: firmware revision (8 chars)
        fw = b'0.4     '
        for i, b in enumerate(fw[:8]):
            d[46 + i] = b

        # Words 27-46: model number (40 chars)
        model = b'x86emu Virtual Disk                     '
        for i, b in enumerate(model[:40]):
            d[54 + i] = b

        # Word 47: max sectors per interrupt (0x8010 = 16)
        struct.pack_into('<H', d, 94,  0x8010)

        # Word 49: capabilities (LBA supported = bit 9, DMA = bit 8)
        struct.pack_into('<H', d, 98,  0x0300)

        # Word 51: PIO timing mode
        struct.pack_into('<H', d, 102, 0x0200)

        # Word 53: fields valid (words 54-58 valid)
        struct.pack_into('<H', d, 106, 0x0001)

        # Words 54-58: current geometry
        struct.pack_into('<H', d, 108, cyls)
        struct.pack_into('<H', d, 110, 16)
        struct.pack_into('<H', d, 112, 63)
        cur_sectors = cyls * 16 * 63
        struct.pack_into('<I', d, 114, cur_sectors)

        # Word 60-61: total LBA sectors (28-bit)
        struct.pack_into('<I', d, 120, min(sectors, 0x0FFFFFFF))

        # Word 80: ATA version (ATA-1 through ATA-4)
        struct.pack_into('<H', d, 160, 0x001E)

        # Word 83: command set supported (LBA48 not supported)
        struct.pack_into('<H', d, 166, 0x0000)

        return bytes(d)

    def _lba28(self) -> int:
        return self.lba0 | (self.lba1 << 8) | (self.lba2 << 16) | ((self.lba3 & 0xF) << 24)

    def execute_command(self, cmd):
        """Process an ATA command written to the command register."""
        self._cmd = cmd
        self.error = 0

        if cmd == ATA_CMD_IDENTIFY:
            self.status    = ATA_SR_DRDY | ATA_SR_DRQ
            self._drq_buf  = bytearray(self._identify_data)
            self._drq_pos  = 0
            return True   # signal IRQ

        elif cmd == ATA_CMD_READ_PIO:
            lba   = self._lba28()
            count = self.sec_count if self.sec_count != 0 else 256
            data  = self.disk.read_sectors(lba, count)
            self.status   = ATA_SR_DRDY | ATA_SR_DRQ
            self._drq_buf = bytearray(data)
            self._drq_pos = 0
            return True

        elif cmd == ATA_CMD_WRITE_PIO:
            lba   = self._lba28()
            count = self.sec_count if self.sec_count != 0 else 256
            self._write_lba   = lba
            self._write_count = count
            self._drq_write   = bytearray()
            self.status = ATA_SR_DRDY | ATA_SR_DRQ
            return False   # no IRQ yet, wait for data

        elif cmd == ATA_CMD_CACHE_FLUSH:
            self.status = ATA_SR_DRDY
            return True

        elif cmd == ATA_CMD_SET_FEATURES:
            # Accept silently
            self.status = ATA_SR_DRDY
            return True

        else:
            # Unknown command — abort
            self.error  = ATA_ER_ABRT
            self.status = ATA_SR_DRDY | ATA_SR_ERR
            return False

    def read_data16(self) -> int:
        """Read 2 bytes from DRQ buffer (data port 0x1F0)."""
        if self._drq_pos + 2 <= len(self._drq_buf):
            val = struct.unpack_from('<H', self._drq_buf, self._drq_pos)[0]
            self._drq_pos += 2
            if self._drq_pos >= len(self._drq_buf):
                self.status = ATA_SR_DRDY   # DRQ cleared, transfer done
            return val
        return 0xFFFF

    def write_data16(self, val) -> bool:
        """
        Write 2 bytes to write buffer.
        Returns True when a full sector is ready to commit.
        """
        self._drq_write += struct.pack('<H', val)
        if len(self._drq_write) >= SECTOR_SIZE:
            # Commit sector
            self.disk.write_sector(self._write_lba, self._drq_write[:SECTOR_SIZE])
            self._write_lba   += 1
            self._write_count -= 1
            self._drq_write    = bytearray()
            if self._write_count <= 0:
                self.status = ATA_SR_DRDY
                return True   # all sectors written, signal IRQ
        return False

    def drq_remaining(self) -> int:
        return max(0, len(self._drq_buf) - self._drq_pos)


class IDEController:
    """
    Primary IDE channel emulator.
    Ports 0x1F0-0x1F7, 0x3F6.
    Fires IRQ14 via PIC on command completion.
    """
    def __init__(self, pic, master_disk=None, slave_disk=None):
        self.pic = pic

        # Create drives
        self.drives = [None, None]
        if master_disk:
            self.drives[0] = ATADrive(master_disk, drive_num=0)
            print(f"[ide] Master: {master_disk.sector_count} sectors "
                  f"({master_disk.sector_count * SECTOR_SIZE // (1024*1024)} MB)")
        if slave_disk:
            self.drives[1] = ATADrive(slave_disk, drive_num=1)
            print(f"[ide] Slave:  {slave_disk.sector_count} sectors")

        # Selected drive (0=master, 1=slave)
        self._selected = 0

        # Device control register (0x3F6)
        self._dev_ctrl  = 0x00
        self._irq_fired = 0   # debug counter

    def _drive(self) -> ATADrive:
        """Return currently selected drive, or None."""
        return self.drives[self._selected]

    def _fire_irq14(self):
        """Signal IRQ14 (primary IDE) through PIC."""
        self.pic.raise_irq(14)
        self._irq_fired += 1

    
    # IO port read
    
    def read(self, port) -> int:
        drv = self._drive()

        if port == 0x1F0:   # Data (16-bit)
            if drv and (drv.status & ATA_SR_DRQ):
                return drv.read_data16()
            return 0xFFFF

        elif port == 0x1F1:  # Error
            return drv.error if drv else 0x01

        elif port == 0x1F2:  # Sector count
            return drv.sec_count if drv else 0

        elif port == 0x1F3:  # LBA[7:0]
            return drv.lba0 if drv else 0

        elif port == 0x1F4:  # LBA[15:8]
            return drv.lba1 if drv else 0

        elif port == 0x1F5:  # LBA[23:16]
            return drv.lba2 if drv else 0

        elif port == 0x1F6:  # Drive/Head
            if drv:
                return 0xE0 | (self._selected << 4) | (drv.lba3 & 0xF)
            return 0xE0 | (self._selected << 4)

        elif port == 0x1F7:  # Status
            if drv:
                return drv.status
            return 0x00   # no drive

        elif port == 0x3F6:  # Alternate status
            if drv:
                return drv.status
            return 0x00

        # Secondary channel — no drive attached
        elif 0x170 <= port <= 0x177 or port == 0x376:
            return 0xFF

        return 0xFF

    
    # IO port write
    
    def write(self, port, val):
        drv = self._drive()

        if port == 0x1F0:   # Data (16-bit write)
            if drv and (drv.status & ATA_SR_DRQ):
                fire = drv.write_data16(val)
                if fire:
                    self._fire_irq14()

        elif port == 0x1F1:  # Features — ignore
            pass

        elif port == 0x1F2:  # Sector count
            if drv: drv.sec_count = val if val else 256

        elif port == 0x1F3:  # LBA[7:0]
            if drv: drv.lba0 = val

        elif port == 0x1F4:  # LBA[15:8]
            if drv: drv.lba1 = val

        elif port == 0x1F5:  # LBA[23:16]
            if drv: drv.lba2 = val

        elif port == 0x1F6:  # Drive/Head
            self._selected = (val >> 4) & 1
            drv = self._drive()
            if drv:
                drv.lba3 = val & 0xF
                # LBA mode bit (bit 6) — we always use LBA, ignore

        elif port == 0x1F7:  # Command
            if drv:
                fire = drv.execute_command(val)
                if fire:
                    self._fire_irq14()

        elif port == 0x3F6:  # Device control
            self._dev_ctrl = val
            # Bit 2 = SRST (software reset)
            if val & 0x04:
                for d in self.drives:
                    if d:
                        d.status = ATA_SR_BSY
                        d.error  = 0x01  # diagnostic OK after reset
            elif not (val & 0x04) and (self._dev_ctrl & 0x04):
                # Reset released
                for d in self.drives:
                    if d:
                        d.status = ATA_SR_DRDY

    
    # Secondary channel stubs (Linux probes these)
    
    def read_secondary(self, port):
        return 0xFF

    def write_secondary(self, port, val):
        pass

    def stats(self):
        d = self.drives[0]
        if d:
            return {
                'irq_fired': self._irq_fired,
                'disk':      d.disk.stats(),
            }
        return {}
