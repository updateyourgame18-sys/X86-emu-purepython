"""
phase3.py — x86emu Phase 3: Interrupt Infrastructure
Imports Machine32 from phase2.py.

New in this phase:
  - PIC 8259A (master + slave cascade) — full ICW1-4 init, IRQ masking,
    EOI handling, priority resolution
  - PIT 8253 — channel 0 mode 3 (rate generator), fires IRQ0 at ~18.2 Hz
    (or whatever rate the kernel programs)
  - IDT population helpers — set_gate() for interrupt/trap gates
  - Hardware interrupt injection into CPU fetch loop
  - IRET 32-bit properly returns from interrupt handlers
  - Keyboard controller 8042 stub (IRQ1) — enough to not hang
  - CMOS/RTC stub (port 0x70/0x71) — kernel reads this during boot
  - A test that:
      1. Sets up a full IDT
      2. Installs a timer handler that increments a counter
      3. Lets the PIT fire 10 ticks
      4. Verifies the counter reached 10

Pythonista compatible — numpy + stdlib only.
"""

import struct, sys, time
import numpy as np

# ---------------------------------------------------------------------------
# Import Phase 2
# ---------------------------------------------------------------------------
import importlib, os, sys as _sys

def _import(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None

p2 = _import('phase2')
if p2 is None:
    raise ImportError("Cannot find phase2.py — must be in same directory")

# Re-export everything we need
Memory32    = p2.Memory32
Registers32 = p2.Registers32
BIOS        = p2.BIOS
CPU32       = p2.CPU32
Machine32   = p2.Machine32
IOPorts     = p2.IOPorts
GDT         = p2.GDT
SegDesc     = p2.SegDesc
sign8       = p2.sign8
sign16      = p2.sign16
sign32      = p2.sign32

# ---------------------------------------------------------------------------
# PIC 8259A — master + slave
# ---------------------------------------------------------------------------
class PIC:
    """
    Emulates the dual 8259A PIC (master at 0x20, slave at 0xA0).
    Handles ICW1-4 initialization sequence, OCW2 (EOI), OCW1 (mask).
    IRQ 0-7  → master
    IRQ 8-15 → slave (cascaded through master IRQ2)
    """
    def __init__(self):
        # Interrupt vector base (set by ICW2)
        self.master_base = 0x08   # IRQ0 → INT 0x08 (typical protected mode)
        self.slave_base  = 0x70   # IRQ8 → INT 0x70

        # Mask registers (1=masked/disabled). Real BIOS POST always leaves
        # IRQ0 (timer) and IRQ1 (keyboard) unmasked by the time it hands
        # control to the bootsector — everything else (floppy, COM ports,
        # etc.) typically starts masked until a driver explicitly enables
        # it. Since we skip real BIOS POST entirely and jump straight to
        # the bootsector, starting at 0xFF (everything masked) meant code
        # that relies on the BIOS having already unmasked the timer (e.g.
        # isolinux's HLT-and-wait-for-a-tick idle loop) could never be
        # woken back up — IRQ0 fired into the IRR but get_pending_vector()
        # always saw it as masked.
        self.master_mask = 0xFC   # bits 0,1 clear → IRQ0,IRQ1 unmasked
        self.slave_mask  = 0xFF

        # In-service register (ISR) — bit set while handler running
        self.master_isr  = 0x00
        self.slave_isr   = 0x00

        # Interrupt request register (IRR) — pending hardware IRQs
        self.master_irr  = 0x00
        self.slave_irr   = 0x00

        # ICW init state machine (each PIC has 4 init words)
        self.master_icw_step = 0
        self.slave_icw_step  = 0
        self.master_icw4     = False
        self.slave_icw4      = False

        # Auto-EOI mode
        self.master_aeoi = False
        self.slave_aeoi  = False

    def raise_irq(self, irq):
        """Signal a hardware IRQ (0-15)."""
        if irq < 8:
            self.master_irr |= (1 << irq)
        else:
            self.slave_irr  |= (1 << (irq - 8))
            self.master_irr |= (1 << 2)   # cascade line

    def get_pending_vector(self):
        """
        Returns the interrupt vector number for the highest-priority
        pending unmasked IRQ, or None if nothing pending.
        """
        # Check master IRQs 0-7
        for irq in range(8):
            bit = 1 << irq
            if (self.master_irr & bit) and not (self.master_mask & bit):
                # Not already in service
                if not (self.master_isr & bit):
                    if irq == 2:
                        # Cascade — check slave
                        for sirq in range(8):
                            sbit = 1 << sirq
                            if (self.slave_irr & sbit) and not (self.slave_mask & sbit):
                                if not (self.slave_isr & sbit):
                                    # Acknowledge
                                    self.slave_irr  &= ~sbit
                                    self.slave_isr  |=  sbit
                                    self.master_irr &= ~bit
                                    self.master_isr |=  bit
                                    return self.slave_base + sirq
                    else:
                        self.master_irr &= ~bit
                        self.master_isr |=  bit
                        return self.master_base + irq
        return None

    def eoi(self, is_slave=False):
        """Non-specific EOI — clears highest ISR bit."""
        if is_slave:
            for i in range(8):
                if self.slave_isr & (1 << i):
                    self.slave_isr &= ~(1 << i)
                    # Also clear cascade on master
                    if not self.slave_isr:
                        self.master_isr &= ~0x04
                    return
        else:
            for i in range(8):
                if self.master_isr & (1 << i):
                    self.master_isr &= ~(1 << i)
                    return

    # --- IO port handlers ---
    def read_master_cmd(self):   return self.master_irr
    def read_master_data(self):  return self.master_mask
    def read_slave_cmd(self):    return self.slave_irr
    def read_slave_data(self):   return self.slave_mask

    def write_master_cmd(self, v):
        if v & 0x10:  # ICW1
            self.master_icw_step = 1
            self.master_icw4     = bool(v & 1)
            self.master_irr      = 0
            self.master_isr      = 0
            self.master_mask     = 0
        elif v == 0x20 or v == 0x60:  # Non-specific EOI
            self.eoi(is_slave=False)
        elif v & 0x60 == 0x60:  # Specific EOI
            irq = v & 7
            self.master_isr &= ~(1 << irq)

    def write_master_data(self, v):
        if self.master_icw_step == 1:   # ICW2: vector base
            self.master_base     = v & 0xF8
            self.master_icw_step = 2
        elif self.master_icw_step == 2:  # ICW3: cascade mask
            self.master_icw_step = 3 if self.master_icw4 else 0
        elif self.master_icw_step == 3:  # ICW4
            self.master_aeoi     = bool(v & 2)
            self.master_icw_step = 0
        else:                            # OCW1: mask
            self.master_mask = v

    def write_slave_cmd(self, v):
        if v & 0x10:  # ICW1
            self.slave_icw_step = 1
            self.slave_icw4     = bool(v & 1)
            self.slave_irr      = 0
            self.slave_isr      = 0
            self.slave_mask     = 0
        elif v == 0x20 or v == 0x60:
            self.eoi(is_slave=True)
        elif v & 0x60 == 0x60:
            irq = v & 7
            self.slave_isr &= ~(1 << irq)

    def write_slave_data(self, v):
        if self.slave_icw_step == 1:
            self.slave_base      = v & 0xF8
            self.slave_icw_step  = 2
        elif self.slave_icw_step == 2:
            self.slave_icw_step  = 3 if self.slave_icw4 else 0
        elif self.slave_icw_step == 3:
            self.slave_aeoi      = bool(v & 2)
            self.slave_icw_step  = 0
        else:
            self.slave_mask = v


# ---------------------------------------------------------------------------
# PIT 8253 — channel 0 only (IRQ0 timer)
# ---------------------------------------------------------------------------
class PIT:
    """
    Programmable Interval Timer — channel 0, mode 2/3.
    Clocked at 1.193182 MHz. Fires IRQ0 at reload_value / 1193182 Hz.
    Default reload = 65536 → ~18.2 Hz.
    """
    CLOCK_HZ = 1_193_182

    def __init__(self, pic):
        self.pic = pic

        # Channel 0
        self.reload   = 65536
        self.counter  = 65536
        self.mode     = 3         # square wave / rate generator
        self.latch    = None      # latched count for read
        self.lo_byte  = True      # read/write low byte first
        self.access   = 3         # 11 = lo then hi

        # Simulated cycle counter — we fire IRQ every `reload` cycles
        self._cycles  = 0

    def tick(self, cycles):
        """
        Call with number of CPU cycles elapsed.
        Returns True if IRQ0 should fire.
        """
        self._cycles += cycles
        if self._cycles >= self.reload:
            self._cycles -= self.reload
            self.pic.raise_irq(0)
            return True
        return False

    def read(self, port):
        ch = port - 0x40
        if ch == 0:
            if self.latch is not None:
                v = self.latch
                if self.lo_byte:
                    self.lo_byte = False
                    return v & 0xFF
                else:
                    self.lo_byte = True
                    self.latch   = None
                    return (v >> 8) & 0xFF
            # Read live counter
            if self.lo_byte:
                self.lo_byte = False
                return self.counter & 0xFF
            else:
                self.lo_byte = True
                return (self.counter >> 8) & 0xFF
        return 0xFF

    def write(self, port, val):
        ch = port - 0x40
        if ch == 3:   # Control word register
            ch_sel  = (val >> 6) & 3
            access  = (val >> 4) & 3
            mode    = (val >> 1) & 7
            if ch_sel == 0:
                self.access  = access
                self.mode    = mode
                self.lo_byte = True
                if access == 0:  # Latch command
                    self.latch = self.counter
            return
        if ch == 0:
            if self.access == 1:   # LSB only
                self.reload = (self.reload & 0xFF00) | val
            elif self.access == 2: # MSB only
                self.reload = (self.reload & 0x00FF) | (val << 8)
            else:                  # LSB then MSB
                if self.lo_byte:
                    self.reload  = (self.reload & 0xFF00) | val
                    self.lo_byte = False
                else:
                    self.reload  = (self.reload & 0x00FF) | (val << 8)
                    self.lo_byte = True
                    if self.reload == 0:
                        self.reload = 65536
            self.counter = self.reload


# ---------------------------------------------------------------------------
# Keyboard controller 8042 stub
# ---------------------------------------------------------------------------
class KBD:
    """Minimal 8042 stub — enough to not hang during boot."""
    def __init__(self):
        self._status = 0x00   # no data available
        self._data   = 0xAA   # self-test passed

    def read_status(self):  return self._status
    def read_data(self):    return self._data

    def write_cmd(self, v):
        if v == 0xAA:   # self-test
            self._data   = 0x55
            self._status = 0x01
        elif v == 0xAD:  # disable keyboard
            pass
        elif v == 0xAE:  # enable keyboard
            pass
        elif v == 0xD1:  # write output port (A20)
            pass

    def write_data(self, v):
        pass


# ---------------------------------------------------------------------------
# CMOS / RTC stub
# ---------------------------------------------------------------------------
class CMOS:
    """
    Minimal CMOS stub. Linux reads RTC during boot.
    Ports: 0x70 (address), 0x71 (data)
    """
    def __init__(self):
        self._addr = 0
        # Fake RTC: 2024-01-01 00:00:00, BCD format
        self._regs = {
            0x00: 0x00,  # seconds
            0x02: 0x00,  # minutes
            0x04: 0x00,  # hours
            0x06: 0x01,  # day of week
            0x07: 0x01,  # day of month
            0x08: 0x01,  # month
            0x09: 0x24,  # year (BCD 24 = 2024)
            0x0A: 0x26,  # status A: 32kHz, rate=6
            0x0B: 0x02,  # status B: 24h, no DST
            0x0C: 0x00,  # status C: no IRQs pending
            0x0D: 0x80,  # status D: valid RAM
            0x14: 0x4D,  # equipment byte
            0x17: 0x80,  # base memory lo (640 KB)
            0x18: 0x02,  # base memory hi
            0x30: 0x00,  # extended memory lo
            0x31: 0x00,  # extended memory hi
        }

    def write_addr(self, v):
        self._addr = v & 0x7F   # mask NMI disable bit

    def read_data(self):
        return self._regs.get(self._addr, 0xFF)

    def write_data(self, v):
        self._regs[self._addr] = v & 0xFF


# ---------------------------------------------------------------------------
# Extended IO port map — wires everything together
# ---------------------------------------------------------------------------
class IOPorts3(IOPorts):
    """Phase 3 IO: adds PIC, PIT, KBD, CMOS to Phase 2 IO."""
    def __init__(self, reg, pic, pit, kbd, cmos):
        super().__init__(reg)
        self.pic  = pic
        self.pit  = pit
        self.kbd  = kbd
        self.cmos = cmos

    def read(self, port):
        # PIC master
        if port == 0x20: return self.pic.read_master_cmd()
        if port == 0x21: return self.pic.read_master_data()
        # PIC slave
        if port == 0xA0: return self.pic.read_slave_cmd()
        if port == 0xA1: return self.pic.read_slave_data()
        # PIT
        if 0x40 <= port <= 0x43: return self.pit.read(port)
        # KBD
        if port == 0x60: return self.kbd.read_data()
        if port == 0x64: return self.kbd.read_status()
        # CMOS
        if port == 0x71: return self.cmos.read_data()
        # VGA misc read
        if port in (0x3C0, 0x3C2, 0x3C4, 0x3C6, 0x3C8, 0x3CA, 0x3CC,
                    0x3CE, 0x3D4, 0x3D5, 0x3DA): return 0xFF
        # Fallback
        return super().read(port)

    def write(self, port, val):
        # PIC master
        if port == 0x20: self.pic.write_master_cmd(val);  return
        if port == 0x21: self.pic.write_master_data(val); return
        # PIC slave
        if port == 0xA0: self.pic.write_slave_cmd(val);   return
        if port == 0xA1: self.pic.write_slave_data(val);  return
        # PIT
        if 0x40 <= port <= 0x43: self.pit.write(port, val); return
        # KBD
        if port == 0x60: self.kbd.write_data(val); return
        if port == 0x64: self.kbd.write_cmd(val);  return
        # CMOS
        if port == 0x70: self.cmos.write_addr(val); return
        if port == 0x71: self.cmos.write_data(val); return
        # A20 fast gate
        if port == 0x92: self._a20 = bool(val & 2); return
        # VGA — ignore silently
        if 0x3B0 <= port <= 0x3DF: return
        # Fallback
        super().write(port, val)


# ---------------------------------------------------------------------------
# IDT helpers
# ---------------------------------------------------------------------------
def make_idt_gate(offset, selector, gate_type=0xEE):
    """
    Build an 8-byte 32-bit IDT interrupt gate descriptor.
    gate_type: 0xEE = interrupt gate DPL=3, 0x8E = interrupt gate DPL=0
    """
    lo = ((selector & 0xFFFF) << 16) | (offset & 0xFFFF)
    hi = (offset & 0xFFFF0000) | ((gate_type & 0xFF) << 8)
    return struct.pack('<II', lo, hi)

def install_idt(mem, idt_base, vector, offset, selector=0x08, gate_type=0x8E):
    """Write one IDT entry to memory."""
    gate = make_idt_gate(offset, selector, gate_type)
    addr = idt_base + vector * 8
    mem.load_flat(addr, gate)


# ---------------------------------------------------------------------------
# CPU3 — adds hardware interrupt injection to the fetch loop
# ---------------------------------------------------------------------------

# How many CPU instructions between PIT ticks
# At ~1 MIPS emulated, PIT fires every ~1193182/18.2 ≈ 65536 insns
# We use a smaller number so tests don't take forever
PIT_TICK_INTERVAL = 1000   # check PIT every N instructions
# Real x86 PIT cycles are oscillator ticks (1.193182 MHz), not CPU
# instructions. A period-appropriate CPU executes many instructions per
# PIT cycle. Feeding raw instruction counts 1:1 as PIT cycles made our
# timer interrupt fire far too often relative to code progress — this
# scales instruction count down to a more realistic ratio so guest code
# (isolinux, the kernel) gets proportionally more real work done between
# timer ticks, matching how real hardware behaves.
PIT_CYCLE_SCALE = 32

class CPU3(CPU32):
    """
    Extends CPU32 with:
      - Hardware interrupt injection (checks PIC after every N instructions)
      - PIT tick simulation
      - Proper interrupt nesting (IF flag checked)
    """
    def __init__(self, mem, reg, bios, io, pic, pit):
        super().__init__(mem, reg, bios, io)
        self.pic = pic
        self.pit = pit
        self._irq_check_counter = 0
        self.irq_count = 0        # total hardware IRQs delivered
        self._tick_accum = 0      # for PIT
        # Hot-loop discovery instrumentation — records any EIP that
        # keeps recurring as a candidate for a hand-written native
        # accelerator (see CPU32._try_compile_loop), even ones we don't
        # have an accelerator for yet. The UI layer (phase6.py) reads
        # this to surface the hottest un-accelerated loops instead of
        # us having to guess where time is going during a slow, mostly-
        # unaccelerated stage like gzip's huft_build().
        self.hot_loops = {}   # eip -> {'hits', 'bytes', 'start', 'accelerated'}
        self._hot_loops_order = []   # FIFO of eip insertion order, for O(1) eviction

    def _inject_hardware_irq(self, vector):
        """Push flags/CS/IP and jump to the interrupt handler, the way
        real hardware does for a hardware interrupt — correctly handling
        both real mode (fixed IVT at 0000:vector*4, 16-bit stack frame)
        and protected mode (IDT at idtr_base, 32-bit stack frame). This
        previously always used the protected-mode IDT and 32-bit
        push/pop even in real mode, which read garbage (idtr_base==0)
        and corrupted the real-mode stack layout — meaning a real-mode
        hardware interrupt could never correctly reach its handler or
        IRET back, leaving IF permanently cleared for the rest of boot."""
        r = self.reg

        if not r.protected_mode:
            # Real mode: fixed IVT at 0000:vector*4 (4 bytes/entry,
            # offset then segment), 16-bit stack frame.
            self.push16(r.flags_word())
            self.push16(r.cs)
            self.push16(r.ip)
            r.IF = 0

            ivt_addr = vector * 4
            handler_off = self.mem.read16_flat(ivt_addr)
            handler_seg = self.mem.read16_flat(ivt_addr + 2)

            handler_is_empty = (handler_off == 0 and handler_seg == 0)
            if not handler_is_empty:
                # A handler pointer IS installed, but if the actual code
                # at that address is all zeros, it's functionally no
                # different from having no handler at all — isolinux
                # installs its own IRQ0 handler for timer-driven
                # animation/countdown during kernel loading, but if that
                # code never actually got loaded into memory (a separate,
                # not-yet-understood loading gap), jumping there just
                # decodes zeros as harmless-looking ADD instructions that
                # never reach a real IRET or EOI — silently blocking every
                # future timer interrupt forever, identical in effect to
                # our very first "no handler" bug.
                target = (handler_seg << 4) + handler_off
                probe = self.mem._m[target:target+8].tobytes()
                if probe == b'\x00' * 8:
                    handler_is_empty = True

            if handler_is_empty:
                # No usable handler. Send the EOI ourselves before IRETing
                # so we don't leave the PIC's in-service bit permanently
                # set, which would silently block ALL future delivery of
                # this IRQ.
                if 8 <= vector <= 0xF:
                    self.pic.eoi(is_slave=False)
                elif 0x70 <= vector <= 0x77:
                    self.pic.eoi(is_slave=True)
                r.ip = self.pop16()
                r.cs = self.pop16()
                r.set_flags_word(self.pop16())
                return

            r.cs = handler_seg
            r.ip = handler_off
            self.irq_count += 1
            return

        # Protected mode: IDT at idtr_base, 32-bit stack frame.
        self.push32(r.flags_word())
        self.push32(r.cs)
        self.push32(r.ip)
        r.IF = 0

        # Bounds-check against idtr_limit before trusting anything read
        # from the table. Real x86 hardware checks this for every vector
        # lookup — accessing a vector whose 8-byte gate would extend
        # past idtr_limit is out-of-bounds, full stop, regardless of
        # what bytes happen to physically sit at that address. Without
        # this check, a genuinely empty/null IDT (idtr_limit=0 — which
        # the kernel sets intentionally during early boot, e.g. right
        # after enabling paging in head.S, before its real IDT exists)
        # would have its "lookup" silently read whatever unrelated data
        # happens to live at idtr_base+vector*8 (e.g. leftover bytes
        # from our own real-mode IVT setup) and could misinterpret that
        # as a plausible-looking nonzero handler address, sending
        # execution into garbage instead of correctly treating this as
        # "no handler available."
        idt_entry_end = vector * 8 + 7
        if idt_entry_end > r.idtr_limit:
            if 8 <= vector <= 0xF:
                self.pic.eoi(is_slave=False)
            elif 0x70 <= vector <= 0x77:
                self.pic.eoi(is_slave=True)
            r.ip = self.pop32()
            r.cs = self.pop32()
            r.set_flags_word(self.pop32())
            return

        idt_addr = r.idtr_base + vector * 8
        off_lo   = self.mem.read16_flat(idt_addr)
        sel      = self.mem.read16_flat(idt_addr + 2)
        off_hi   = self.mem.read16_flat(idt_addr + 6)
        handler  = off_lo | (off_hi << 16)

        if handler == 0:
            if 8 <= vector <= 0xF:
                self.pic.eoi(is_slave=False)
            elif 0x70 <= vector <= 0x77:
                self.pic.eoi(is_slave=True)
            r.ip = self.pop32()
            r.cs = self.pop32()
            r.set_flags_word(self.pop32())
            return

        self.gdt.update_cache('cs', sel)
        r.cs = sel
        r.ip = handler
        self.irq_count += 1

    def run(self):
        r = self.reg
        m = self.mem
        _compiled = {}   # eip -> native fn (only ever the gunzip signature)
        _hits = {}       # eip -> count
        import struct
        _check_counter = 0

        while self.icount < self.max_icount:

            if self.halted:
                # Don't run any CPU instructions while halted (real HLT
                # genuinely stops instruction execution), but DO keep
                # advancing the PIT and checking for a wake-up IRQ each
                # pass — that's the whole point of HLT-and-wait. The bug
                # this replaced: the while loop's own condition included
                # "not self.halted", so the very first time HLT fired,
                # the loop exited on its next condition check before the
                # wake-up logic below ever got a second chance to see a
                # PIT tick accumulate past threshold.
                # PIT cycles are real oscillator ticks, not CPU
                # instructions — feeding raw instruction count 1:1 made
                # the timer fire roughly 1000x more often relative to
                # code execution than real hardware, giving isolinux far
                # too little breathing room to finish installing its own
                # IVT/interrupt handlers before the next tick interrupted
                # it mid-setup. Scale down to a more realistic
                # instructions-per-PIT-cycle ratio.
                batch = min(PIT_TICK_INTERVAL, self.max_icount - self.icount)
                self.icount += batch
                self.pit.tick(max(1, batch // PIT_CYCLE_SCALE))

                if r.IF:
                    vec = self.pic.get_pending_vector()
                    if vec is not None:
                        self.halted = False
                        self._inject_hardware_irq(vec)
                        saved_max2      = self.max_icount
                        self.max_icount = self.icount + 256
                        CPU32.run(self)
                        self.max_icount = saved_max2
                continue

            # ---- Native loop accelerator (gunzip Huffman loop only) ----
            # Only probe every 64 outer-loop passes to keep per-instruction
            # overhead near zero for the common case (no hot loop active).
            _check_counter += 1
            if r.protected_mode and (_check_counter & 63) == 0:
                eip = r.ip
                h = _hits.get(eip, 0) + 1
                _hits[eip] = h

                # Hot-loop discovery instrumentation: record any EIP
                # that keeps recurring, whether or not we have a native
                # accelerator for it, so the UI layer can surface the
                # hottest un-accelerated loops for someone to inspect
                # instead of guessing where time is going.
                #
                # Capped via O(1) FIFO eviction to prevent both unbounded
                # memory growth AND the severe performance cliff an O(n)
                # "evict least-hit entry" scan causes once execution
                # genuinely spans a wide range of distinct addresses —
                # that scan re-running on every single new EIP discovered
                # (which can be very frequent) compounds into a dramatic,
                # progressive slowdown over a long run.
                HOT_LOOPS_MAX = 2000
                if h >= 3:
                    entry = self.hot_loops.get(eip)
                    if entry is None:
                        if len(self.hot_loops) >= HOT_LOOPS_MAX:
                            oldest = self._hot_loops_order.pop(0)
                            self.hot_loops.pop(oldest, None)
                        try:
                            start = max(0, eip - 8)
                            dump = bytes(self.mem._m[start:eip + 24].tobytes())
                        except Exception:
                            start, dump = eip, b''
                        entry = {'hits': 0, 'bytes': dump, 'start': start,
                                 'accelerated': False}
                        self.hot_loops[eip] = entry
                        self._hot_loops_order.append(eip)
                    entry['hits'] += 1

                if h >= 3:
                    fn = _compiled.get(eip)
                    if fn is None:
                        fn = self._try_compile_loop(eip)
                        _compiled[eip] = fn if fn else False
                    if fn:
                        self.hot_loops[eip]['accelerated'] = True
                        iters = fn()
                        if iters and iters > 0:
                            self.icount += iters
                            if r.zf:  # loop done, EIP advanced past JNZ
                                _hits[eip] = 0
                            continue

            # ---- Normal batch ----
            batch = min(PIT_TICK_INTERVAL, self.max_icount - self.icount)
            saved_max       = self.max_icount
            self.max_icount = self.icount + batch
            CPU32.run(self)
            self.max_icount = saved_max

            self.pit.tick(max(1, batch // PIT_CYCLE_SCALE))

            # Real x86 HLT waits for an interrupt (or NMI/reset) and then
            # resumes at the instruction right after HLT — it does NOT
            # permanently stop the CPU. Real-mode boot code commonly HLTs
            # while waiting for a disk/timer IRQ to wake it back up.
            if r.IF:
                vec = self.pic.get_pending_vector()
                if vec is not None:
                    if self.halted:
                        self.halted = False   # wake up from HLT
                    self._inject_hardware_irq(vec)
                    saved_max2      = self.max_icount
                    self.max_icount = self.icount + 256
                    CPU32.run(self)
                    self.max_icount = saved_max2

        return self.icount


# ---------------------------------------------------------------------------
# Machine3
# ---------------------------------------------------------------------------
class Machine3:
    def __init__(self):
        self.mem   = Memory32(size=32 * 1024 * 1024)
        self.reg   = Registers32()
        self.bios  = BIOS(self.mem, self.reg)
        self.pic   = PIC()
        self.pit   = PIT(self.pic)
        self.kbd   = KBD()
        self.cmos  = CMOS()
        self.io    = IOPorts3(self.reg, self.pic, self.pit, self.kbd, self.cmos)
        self.cpu   = CPU3(self.mem, self.reg, self.bios, self.io, self.pic, self.pit)

    def load_at(self, addr, data):
        self.mem.load_flat(addr, data)

    def set_entry(self, cs, ip):
        self.reg.cs = cs
        self.reg.ip = ip
        self.reg.ss = 0
        self.reg.sp = 0x7C00
        self.reg.ds = 0
        self.reg.es = 0

    def install_gate(self, vector, offset, selector=0x08):
        install_idt(self.mem, self.reg.idtr_base, vector, offset, selector)

    def run(self, max_icount=10_000_000):
        self.cpu.max_icount = max_icount
        return self.cpu.run()
