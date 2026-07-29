"""
x86emu_phase1.py
Real-mode x86 emulator, Phase 1.
Runs a 512-byte bootsector image in 16-bit real mode.

Target: boot a bootsector that prints "Hello from x86emu!" via BIOS INT 10h.

Pythonista compatible — only numpy + stdlib.
"""

import numpy as np
import struct
import sys


# Memory

MEM_SIZE = 1 * 1024 * 1024   # 1 MB — real mode address space

class Memory:
    def __init__(self):
        self._m = np.zeros(MEM_SIZE, dtype=np.uint8)

    def linear(self, seg, off):
        """Real-mode segmented address → linear."""
        return ((seg & 0xFFFF) << 4) + (off & 0xFFFF)

    # --- raw ---
    def rb(self, addr):
        return int(self._m[addr & 0xFFFFF])

    def wb(self, addr, v):
        self._m[addr & 0xFFFFF] = v & 0xFF

    # --- segmented ---
    def read8(self, seg, off):
        return self.rb(self.linear(seg, off))

    def read16(self, seg, off):
        la = self.linear(seg, off)
        return int(self._m[la]) | (int(self._m[(la+1) & 0xFFFFF]) << 8)

    def write8(self, seg, off, v):
        self.wb(self.linear(seg, off), v)

    def write16(self, seg, off, v):
        la = self.linear(seg, off)
        self._m[la]             = v & 0xFF
        self._m[(la+1)&0xFFFFF] = (v >> 8) & 0xFF

    def load(self, addr, data):
        n = len(data)
        self._m[addr:addr+n] = np.frombuffer(data, dtype=np.uint8)

    def read_bytes(self, addr, n):
        return self._m[addr:addr+n].tobytes()



# Registers
class Registers:
    def __init__(self):
        # General purpose (16-bit view; we store 32-bit for easy masking)
        self.ax = self.bx = self.cx = self.dx = 0
        self.si = self.di = self.bp = self.sp = 0
        # Segment registers
        self.cs = self.ds = self.es = self.ss = self.fs = self.gs = 0
        # Instruction pointer
        self.ip = 0
        # FLAGS
        self.cf = self.zf = self.sf = self.of_ = self.pf = self.af = 0
        self.df = 0   # direction flag
        self.IF = 1   # interrupt enable

    # --- 8-bit halves ---
    @property
    def al(self): return self.ax & 0xFF
    @al.setter
    def al(self, v): self.ax = (self.ax & 0xFF00) | (v & 0xFF)

    @property
    def ah(self): return (self.ax >> 8) & 0xFF
    @ah.setter
    def ah(self, v): self.ax = (self.ax & 0x00FF) | ((v & 0xFF) << 8)

    @property
    def bl(self): return self.bx & 0xFF
    @bl.setter
    def bl(self, v): self.bx = (self.bx & 0xFF00) | (v & 0xFF)

    @property
    def bh(self): return (self.bx >> 8) & 0xFF
    @bh.setter
    def bh(self, v): self.bx = (self.bx & 0x00FF) | ((v & 0xFF) << 8)

    @property
    def cl(self): return self.cx & 0xFF
    @cl.setter
    def cl(self, v): self.cx = (self.cx & 0xFF00) | (v & 0xFF)

    @property
    def ch(self): return (self.cx >> 8) & 0xFF
    @ch.setter
    def ch(self, v): self.cx = (self.cx & 0x00FF) | ((v & 0xFF) << 8)

    @property
    def dl(self): return self.dx & 0xFF
    @dl.setter
    def dl(self, v): self.dx = (self.dx & 0xFF00) | (v & 0xFF)

    @property
    def dh(self): return (self.dx >> 8) & 0xFF
    @dh.setter
    def dh(self, v): self.dx = (self.dx & 0x00FF) | ((v & 0xFF) << 8)

    def flags_word(self):
        f = 0x0002  # bit 1 always set
        if self.cf:  f |= 0x0001
        if self.pf:  f |= 0x0004
        if self.af:  f |= 0x0010
        if self.zf:  f |= 0x0040
        if self.sf:  f |= 0x0080
        if self.IF:  f |= 0x0200
        if self.df:  f |= 0x0400
        if self.of_: f |= 0x0800
        # Preserve extended 32-bit EFLAGS bits (IOPL, NT, RF, VM, AC, VIF,
        # VIP, ID) round-trip through PUSHFD/POPFD even though i don't
        # actively emulate their hardware effects. Linux head.S CPU
        # detection toggles AC (bit18) and ID (bit21) via PUSHFD/POPFD to
        # probe for 486/CPUID support — if these bits don't round-trip,
        # the kernel misdetects the CPU and takes the wrong init path.
        f |= getattr(self, '_eflags_ext', 0) & 0x3F3000  # bits 12-13,14,16-21
        return f

    def set_flags_word(self, f):
        self.cf  = (f >> 0)  & 1
        self.pf  = (f >> 2)  & 1
        self.af  = (f >> 4)  & 1
        self.zf  = (f >> 6)  & 1
        self.sf  = (f >> 7)  & 1
        self.IF  = (f >> 9)  & 1
        self.df  = (f >> 10) & 1
        self.of_ = (f >> 11) & 1
        # Store extended bits (IOPL=12-13, NT=14, RF=16, VM=17, AC=18,
        # VIF=19, VIP=20, ID=21) so they round-trip correctly even though
        # i don't emulate their hardware behavior. either.
        self._eflags_ext = f & 0x3F3000

    def __repr__(self):
        return (f"AX={self.ax:04X} BX={self.bx:04X} CX={self.cx:04X} DX={self.dx:04X} "
                f"SI={self.si:04X} DI={self.di:04X} BP={self.bp:04X} SP={self.sp:04X}\n"
                f"CS={self.cs:04X} DS={self.ds:04X} ES={self.es:04X} SS={self.ss:04X} "
                f"IP={self.ip:04X} FL={self.flags_word():04X} "
                f"CF={self.cf} ZF={self.zf} SF={self.sf} OF={self.of_}")



# BIOS

class BIOS:
    def __init__(self, mem, reg):
        self.mem = mem
        self.reg = reg
        self._output = []   # collect INT 10h output

    def interrupt(self, num):
        r = self.reg
        if num == 0x10:
            self._int10()
        elif num == 0x16:
            self._int16()
        elif num == 0x19:
            raise SystemExit("INT 19h: reboot")
        else:
            # silently ignore unknown BIOS ints for now
            pass

    def _int10(self):
        r = self.reg
        ah = r.ah
        if ah == 0x0E:
            # TTY write character
            ch = chr(r.al & 0x7F)
            self._output.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()
        elif ah == 0x00:
            pass  # set video mode — ignore
        elif ah == 0x01:
            pass  # set cursor shape — ignore
        elif ah == 0x02:
            pass  # set cursor pos — ignore
        elif ah == 0x03:
            # get cursor pos — return 0
            r.dx = 0
            r.cx = 0
        else:
            pass  # unimplemented INT 10h sub

    def _int16(self):
        # keyboard services — stub
        r = self.reg
        ah = r.ah
        if ah == 0x00 or ah == 0x10:
            # read key — return Enter (0x1C0D)
            r.ax = 0x1C0D
        elif ah == 0x01 or ah == 0x11:
            # check key — no key available
            r.ah = 0
            self.reg.zf = 1

    def get_output(self):
        return ''.join(self._output)



# CPU helpers

def sign8(v):  return v if v < 0x80 else v - 0x100
def sign16(v): return v if v < 0x8000 else v - 0x10000

def parity(v):
    v &= 0xFF
    v ^= v >> 4; v ^= v >> 2; v ^= v >> 1
    return (~v) & 1

def update_flags_8(r, result, cf=None, of_=None):
    result8 = result & 0xFF
    r.zf = 1 if result8 == 0 else 0
    r.sf = (result8 >> 7) & 1
    r.pf = parity(result8)
    if cf  is not None: r.cf  = cf
    if of_ is not None: r.of_ = of_

def update_flags_16(r, result, cf=None, of_=None):
    result16 = result & 0xFFFF
    r.zf = 1 if result16 == 0 else 0
    r.sf = (result16 >> 15) & 1
    r.pf = parity(result16)
    if cf  is not None: r.cf  = cf
    if of_ is not None: r.of_ = of_



# ModRM decoder

def decode_modrm(cpu, byte):
    """
    Decode a ModRM byte.
    Returns (mod, reg, rm).
    Also advances cpu.reg.ip past any displacement bytes and returns
    the effective address (or None if register).
    """
    mod = (byte >> 6) & 3
    reg = (byte >> 3) & 7
    rm  = byte & 7
    r   = cpu.reg
    m   = cpu.mem

    if mod == 3:
        return mod, reg, rm, None  # register operand

    # Compute EA
    bases = {
        0: lambda: r.bx + r.si,
        1: lambda: r.bx + r.di,
        2: lambda: r.bp + r.si,
        3: lambda: r.bp + r.di,
        4: lambda: r.si,
        5: lambda: r.di,
        6: lambda: r.bp,
        7: lambda: r.bx,
    }

    if mod == 0 and rm == 6:
        # direct address
        disp = cpu.fetch16()
        ea = disp
    else:
        ea = bases[rm]() & 0xFFFF
        if mod == 1:
            ea = (ea + sign8(cpu.fetch8())) & 0xFFFF
        elif mod == 2:
            ea = (ea + cpu.fetch16()) & 0xFFFF

    return mod, reg, rm, ea



# Register index → name helpers

REG16 = ['ax','cx','dx','bx','sp','bp','si','di']
REG8  = ['al','cl','dl','bl','ah','ch','dh','bh']
SEG   = ['es','cs','ss','ds','fs','gs']

def get_reg16(r, idx): return getattr(r, REG16[idx]) & 0xFFFF
def set_reg16(r, idx, v): setattr(r, REG16[idx], v & 0xFFFF)
def get_reg8(r, idx):  return getattr(r, REG8[idx]) & 0xFF
def set_reg8(r, idx, v): setattr(r, REG8[idx], v & 0xFF)
def get_seg(r, idx):   return getattr(r, SEG[idx]) & 0xFFFF
def set_seg(r, idx, v): setattr(r, SEG[idx], v & 0xFFFF)



# CPU

class CPU:
    def __init__(self, mem, reg, bios):
        self.mem  = mem
        self.reg  = reg
        self.bios = bios
        self.halted   = False
        self.icount   = 0
        self.max_icount = 10_000_000
        # segment override prefix
        self._seg_override = None

    #fetch helpers
    def fetch8(self):
        v = self.mem.read8(self.reg.cs, self.reg.ip)
        self.reg.ip = (self.reg.ip + 1) & 0xFFFF
        return v

    def fetch16(self):
        lo = self.fetch8()
        hi = self.fetch8()
        return lo | (hi << 8)

    def _seg(self, default='ds'):
        if self._seg_override is not None:
            s = self._seg_override
            self._seg_override = None
            return getattr(self.reg, s) & 0xFFFF
        return getattr(self.reg, default) & 0xFFFF

    #stack
    def push16(self, v):
        self.reg.sp = (self.reg.sp - 2) & 0xFFFF
        self.mem.write16(self.reg.ss, self.reg.sp, v & 0xFFFF)

    def pop16(self):
        v = self.mem.read16(self.reg.ss, self.reg.sp)
        self.reg.sp = (self.reg.sp + 2) & 0xFFFF
        return v

    #ALU ops
    def _add16(self, a, b, carry=0):
        r = self.reg
        res = a + b + carry
        cf  = 1 if res > 0xFFFF else 0
        res16 = res & 0xFFFF
        of_ = 1 if (not ((a ^ b) & 0x8000)) and ((a ^ res16) & 0x8000) else 0
        update_flags_16(r, res16, cf=cf, of_=of_)
        r.af = 1 if ((a & 0xF) + (b & 0xF) + carry) > 0xF else 0
        return res16

    def _sub16(self, a, b, borrow=0):
        r = self.reg
        res = a - b - borrow
        cf  = 1 if res < 0 else 0
        res16 = res & 0xFFFF
        of_ = 1 if ((a ^ b) & 0x8000) and ((a ^ res16) & 0x8000) else 0
        update_flags_16(r, res16, cf=cf, of_=of_)
        r.af = 1 if ((a & 0xF) < (b & 0xF) + borrow) else 0
        return res16

    def _add8(self, a, b, carry=0):
        r = self.reg
        res = a + b + carry
        cf  = 1 if res > 0xFF else 0
        res8 = res & 0xFF
        of_ = 1 if (not ((a ^ b) & 0x80)) and ((a ^ res8) & 0x80) else 0
        update_flags_8(r, res8, cf=cf, of_=of_)
        r.af = 1 if ((a & 0xF) + (b & 0xF) + carry) > 0xF else 0
        return res8

    def _sub8(self, a, b, borrow=0):
        r = self.reg
        res = a - b - borrow
        cf  = 1 if res < 0 else 0
        res8 = res & 0xFF
        of_ = 1 if ((a ^ b) & 0x80) and ((a ^ res8) & 0x80) else 0
        update_flags_8(r, res8, cf=cf, of_=of_)
        r.af = 1 if ((a & 0xF) < (b & 0xF) + borrow) else 0
        return res8

    #get/set mem or reg operand from ModRM 
    def _modrm_read16(self, mod, rm, ea):
        if mod == 3:
            return get_reg16(self.reg, rm)
        return self.mem.read16(self._seg(), ea)

    def _modrm_write16(self, mod, rm, ea, v):
        if mod == 3:
            set_reg16(self.reg, rm, v)
        else:
            self.mem.write16(self._seg(), ea, v)

    def _modrm_read8(self, mod, rm, ea):
        if mod == 3:
            return get_reg8(self.reg, rm)
        return self.mem.read8(self._seg(), ea)

    def _modrm_write8(self, mod, rm, ea, v):
        if mod == 3:
            set_reg8(self.reg, rm, v)
        else:
            self.mem.write8(self._seg(), ea, v)

    #main execute loop
    def run(self):
        r = self.reg
        m = self.mem

        while not self.halted and self.icount < self.max_icount:
            self.icount += 1
            op = self.fetch8()

            #PREFIX
            if op == 0x26: self._seg_override = 'es'; continue
            if op == 0x2E: self._seg_override = 'cs'; continue
            if op == 0x36: self._seg_override = 'ss'; continue
            if op == 0x3E: self._seg_override = 'ds'; continue
            if op == 0x64: self._seg_override = 'fs'; continue
            if op == 0x65: self._seg_override = 'gs'; continue
            if op == 0xF2: continue  # REPNE prefix (stub)
            if op == 0xF3: continue  # REP prefix (stub)

            #NOP
            if op == 0x90: continue

            #HLT
            if op == 0xF4:
                self.halted = True
                break

            # CLI / STI
            if op == 0xFA: r.IF = 0; continue
            if op == 0xFB: r.IF = 1; continue

            # CLD / STD
            if op == 0xFC: r.df = 0; continue
            if op == 0xFD: r.df = 1; continue

            # STC / CLC / CMC
            if op == 0xF9: r.cf = 1; continue
            if op == 0xF8: r.cf = 0; continue
            if op == 0xF5: r.cf ^= 1; continue

            # ---- MOV r16, imm16 (B8+r) ----
            if 0xB8 <= op <= 0xBF:
                set_reg16(r, op - 0xB8, self.fetch16())
                continue

            # ---- MOV r8, imm8 (B0+r) i feel like adding bloat to my code because im already getting bored
            if 0xB0 <= op <= 0xB7:
                set_reg8(r, op - 0xB0, self.fetch8())
                continue

            # ---- MOV r/m16, r16 (89) ----
            if op == 0x89:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                v = get_reg16(r, reg)
                self._modrm_write16(mod, rm, ea, v)
                continue

            # ---- MOV r16, r/m16 (8B) ----
            if op == 0x8B:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                v = self._modrm_read16(mod, rm, ea)
                set_reg16(r, reg, v)
                continue

            # ---- MOV r/m8, r8 (88) ----
            if op == 0x88:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                v = get_reg8(r, reg)
                self._modrm_write8(mod, rm, ea, v)
                continue

            # ---- MOV r8, r/m8 (8A) ----
            if op == 0x8A:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                v = self._modrm_read8(mod, rm, ea)
                set_reg8(r, reg, v)
                continue

            # ---- MOV r/m16, imm16 (C7 /0) ----
            if op == 0xC7:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                imm = self.fetch16()
                self._modrm_write16(mod, rm, ea, imm)
                continue

            # ---- MOV r/m8, imm8 (C6 /0) ----
            if op == 0xC6:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                imm = self.fetch8()
                self._modrm_write8(mod, rm, ea, imm)
                continue

            # ---- MOV sreg, r/m16 (8E) ----
            if op == 0x8E:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                v = self._modrm_read16(mod, rm, ea)
                set_seg(r, reg, v)
                continue

            # ---- MOV r/m16, sreg (8C) ----
            if op == 0x8C:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                v = get_seg(r, reg)
                self._modrm_write16(mod, rm, ea, v)
                continue

            # ---- MOV AX, [imm16] (A1) ----
            if op == 0xA1:
                addr = self.fetch16()
                r.ax = m.read16(self._seg(), addr)
                continue

            # ---- MOV [imm16], AX (A3) ----
            if op == 0xA3:
                addr = self.fetch16()
                m.write16(self._seg(), addr, r.ax)
                continue

            # ---- PUSH r16 (50+r) ----
            if 0x50 <= op <= 0x57:
                self.push16(get_reg16(r, op - 0x50))
                continue

            # ---- POP r16 (58+r) ----
            if 0x58 <= op <= 0x5F:
                set_reg16(r, op - 0x58, self.pop16())
                continue

            # PUSH imm16 (68)
            if op == 0x68:
                self.push16(self.fetch16())
                continue
              #PUSH imm8 sign-extended (6A) 
            if op == 0x6A:
                self.push16(sign8(self.fetch8()) & 0xFFFF)
                continue

            # PUSH sreg
            if op == 0x06: self.push16(r.es); continue
            if op == 0x0E: self.push16(r.cs); continue
            if op == 0x16: self.push16(r.ss); continue
            if op == 0x1E: self.push16(r.ds); continue

            # ---- POP sreg ----
            if op == 0x07: r.es = self.pop16(); continue
            if op == 0x17: r.ss = self.pop16(); continue
            if op == 0x1F: r.ds = self.pop16(); continue

            # ---- PUSHF / POPF ----
            if op == 0x9C: self.push16(r.flags_word()); continue
            if op == 0x9D: r.set_flags_word(self.pop16()); continue

            # ---- XCHG AX, r16 (90+r) already handled by NOP for 90 ----
            if 0x91 <= op <= 0x97:
                idx = op - 0x90
                tmp = get_reg16(r, idx)
                set_reg16(r, idx, r.ax)
                r.ax = tmp
                continue

            # ---- ADD r/m8, r8 (00) ----
            if op == 0x00:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                a = self._modrm_read8(mod, rm, ea)
                b = get_reg8(r, reg)
                self._modrm_write8(mod, rm, ea, self._add8(a, b))
                continue

            # ---- ADD r/m16, r16 (01) ----
            if op == 0x01:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                a = self._modrm_read16(mod, rm, ea)
                b = get_reg16(r, reg)
                self._modrm_write16(mod, rm, ea, self._add16(a, b))
                continue

            # ---- ADD r8, r/m8 (02) ----
            if op == 0x02:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                a = get_reg8(r, reg)
                b = self._modrm_read8(mod, rm, ea)
                set_reg8(r, reg, self._add8(a, b))
                continue

            # ---- ADD r16, r/m16 (03) ----
            if op == 0x03:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                a = get_reg16(r, reg)
                b = self._modrm_read16(mod, rm, ea)
                set_reg16(r, reg, self._add16(a, b))
                continue

            # ---- ADD AL, imm8 (04) ----
            if op == 0x04:
                r.al = self._add8(r.al, self.fetch8())
                continue

            # ---- ADD AX, imm16 (05) ----
            if op == 0x05:
                r.ax = self._add16(r.ax, self.fetch16())
                continue

            # ---- SUB r/m8, r8 (28) ----
            if op == 0x28:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                a = self._modrm_read8(mod, rm, ea)
                b = get_reg8(r, reg)
                self._modrm_write8(mod, rm, ea, self._sub8(a, b))
                continue

            # ---- SUB r/m16, r16 (29) ----
            if op == 0x29:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                a = self._modrm_read16(mod, rm, ea)
                b = get_reg16(r, reg)
                self._modrm_write16(mod, rm, ea, self._sub16(a, b))
                continue

            # ---- SUB r8, r/m8 (2A) ----
            if op == 0x2A:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                a = get_reg8(r, reg)
                b = self._modrm_read8(mod, rm, ea)
                set_reg8(r, reg, self._sub8(a, b))
                continue

            # ---- SUB r16, r/m16 (2B) ----
            if op == 0x2B:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                a = get_reg16(r, reg)
                b = self._modrm_read16(mod, rm, ea)
                set_reg16(r, reg, self._sub16(a, b))
                continue

            # ---- SUB AL, imm8 (2C) ----
            if op == 0x2C:
                r.al = self._sub8(r.al, self.fetch8())
                continue

            # ---- SUB AX, imm16 (2D) ----
            if op == 0x2D:
                r.ax = self._sub16(r.ax, self.fetch16())
                continue

            # ---- INC r16 (40+r) ----
            if 0x40 <= op <= 0x47:
                idx = op - 0x40
                v = get_reg16(r, idx)
                res = self._add16(v, 1)
                # INC doesn't affect CF
                cf_save = r.cf
                set_reg16(r, idx, res)
                r.cf = cf_save
                continue

            # ---- DEC r16 (48+r) ----
            if 0x48 <= op <= 0x4F:
                idx = op - 0x48
                v = get_reg16(r, idx)
                cf_save = r.cf
                res = self._sub16(v, 1)
                set_reg16(r, idx, res)
                r.cf = cf_save
                continue

            # ---- 0x80 / 0x81 / 0x83 group ----
            if op in (0x80, 0x81, 0x83):
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                if op == 0x80:
                    imm = self.fetch8()
                    a   = self._modrm_read8(mod, rm, ea)
                    if   reg == 0: res = self._add8(a, imm);  self._modrm_write8(mod,rm,ea,res)
                    elif reg == 1: res = self._add8(a, imm, carry=r.cf); self._modrm_write8(mod,rm,ea,res)
                    elif reg == 2: res = self._sub8(a, imm);  self._modrm_write8(mod,rm,ea,res)  # ADC stub
                    elif reg == 3: res = self._sub8(a, imm, borrow=r.cf); self._modrm_write8(mod,rm,ea,res)
                    elif reg == 4: res = a & imm; update_flags_8(r, res, cf=0, of_=0); self._modrm_write8(mod,rm,ea,res)
                    elif reg == 5: res = a - imm; self._sub8(a, imm)  # CMP
                    elif reg == 6: res = a ^ imm; update_flags_8(r, res, cf=0, of_=0); self._modrm_write8(mod,rm,ea,res)
                    elif reg == 7: self._sub8(a, imm)  # CMP
                elif op == 0x81:
                    imm = self.fetch16()
                    a   = self._modrm_read16(mod, rm, ea)
                    if   reg == 0: res = self._add16(a, imm);  self._modrm_write16(mod,rm,ea,res)
                    elif reg == 1: res = self._add16(a, imm, carry=r.cf); self._modrm_write16(mod,rm,ea,res)
                    elif reg == 2: res = self._sub16(a, imm);  self._modrm_write16(mod,rm,ea,res)
                    elif reg == 3: res = self._sub16(a, imm, borrow=r.cf); self._modrm_write16(mod,rm,ea,res)
                    elif reg == 4: res = a & imm; update_flags_16(r, res, cf=0, of_=0); self._modrm_write16(mod,rm,ea,res)
                    elif reg == 5: self._sub16(a, imm)  # CMP
                    elif reg == 6: res = a ^ imm; update_flags_16(r, res, cf=0, of_=0); self._modrm_write16(mod,rm,ea,res)
                    elif reg == 7: self._sub16(a, imm)  # CMP
                else:  # 0x83 sign-extended imm8
                    imm = sign8(self.fetch8()) & 0xFFFF
                    a   = self._modrm_read16(mod, rm, ea)
                    if   reg == 0: res = self._add16(a, imm);  self._modrm_write16(mod,rm,ea,res)
                    elif reg == 1: res = self._add16(a, imm, carry=r.cf); self._modrm_write16(mod,rm,ea,res)
                    elif reg == 2: res = self._sub16(a, imm);  self._modrm_write16(mod,rm,ea,res)
                    elif reg == 3: res = self._sub16(a, imm, borrow=r.cf); self._modrm_write16(mod,rm,ea,res)
                    elif reg == 4: res = a & imm; update_flags_16(r, res, cf=0, of_=0); self._modrm_write16(mod,rm,ea,res)
                    elif reg == 5: self._sub16(a, imm)  # CMP
                    elif reg == 6: res = a ^ imm; update_flags_16(r, res, cf=0, of_=0); self._modrm_write16(mod,rm,ea,res)
                    elif reg == 7: self._sub16(a, imm)  # CMP
                continue

            # ---- CMP r/m16, r16 (39) ----
            if op == 0x39:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                self._sub16(self._modrm_read16(mod, rm, ea), get_reg16(r, reg))
                continue

            # ---- CMP r16, r/m16 (3B) ----
            if op == 0x3B:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                self._sub16(get_reg16(r, reg), self._modrm_read16(mod, rm, ea))
                continue

            # ---- CMP AL, imm8 (3C) ----
            if op == 0x3C:
                self._sub8(r.al, self.fetch8())
                continue

            # ---- CMP AX, imm16 (3D) ----
            if op == 0x3D:
                self._sub16(r.ax, self.fetch16())
                continue

            # ---- AND r/m8, r8 (20) ----
            if op == 0x20:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                res = self._modrm_read8(mod, rm, ea) & get_reg8(r, reg)
                update_flags_8(r, res, cf=0, of_=0)
                self._modrm_write8(mod, rm, ea, res)
                continue

            # ---- AND r/m16, r16 (21) ----
            if op == 0x21:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                res = self._modrm_read16(mod, rm, ea) & get_reg16(r, reg)
                update_flags_16(r, res, cf=0, of_=0)
                self._modrm_write16(mod, rm, ea, res)
                continue

            # ---- AND r8, r/m8 (22) ----
            if op == 0x22:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                res = get_reg8(r, reg) & self._modrm_read8(mod, rm, ea)
                update_flags_8(r, res, cf=0, of_=0)
                set_reg8(r, reg, res)
                continue

            # ---- AND AX, imm16 (25) ----
            if op == 0x25:
                r.ax &= self.fetch16()
                update_flags_16(r, r.ax, cf=0, of_=0)
                continue

            # ---- OR r/m8, r8 (08) ----
            if op == 0x08:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                res = self._modrm_read8(mod, rm, ea) | get_reg8(r, reg)
                update_flags_8(r, res, cf=0, of_=0)
                self._modrm_write8(mod, rm, ea, res)
                continue

            # ---- OR r/m16, r16 (09) ----
            if op == 0x09:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                res = self._modrm_read16(mod, rm, ea) | get_reg16(r, reg)
                update_flags_16(r, res, cf=0, of_=0)
                self._modrm_write16(mod, rm, ea, res)
                continue

            # OR r8, r/m8 (0A) 
            if op == 0x0A:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                res = get_reg8(r, reg) | self._modrm_read8(mod, rm, ea)
                update_flags_8(r, res, cf=0, of_=0)
                set_reg8(r, reg, res)
                continue

            # OR AX, imm16 (0D)
            if op == 0x0D:
                r.ax |= self.fetch16()
                update_flags_16(r, r.ax, cf=0, of_=0)
                continue

            # XOR r/m8, r8 (30)
            if op == 0x30:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                res = self._modrm_read8(mod, rm, ea) ^ get_reg8(r, reg)
                update_flags_8(r, res, cf=0, of_=0)
                self._modrm_write8(mod, rm, ea, res)
                continue

            # XOR r/m16, r16 (31)
            if op == 0x31:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                res = self._modrm_read16(mod, rm, ea) ^ get_reg16(r, reg)
                update_flags_16(r, res, cf=0, of_=0)
                self._modrm_write16(mod, rm, ea, res)
                continue

            #XOR r8, r/m8 (32)
            if op == 0x32:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                res = get_reg8(r, reg) ^ self._modrm_read8(mod, rm, ea)
                update_flags_8(r, res, cf=0, of_=0)
                set_reg8(r, reg, res)
                continue

            #XOR r16, r/m16 (33)
            if op == 0x33:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                res = get_reg16(r, reg) ^ self._modrm_read16(mod, rm, ea)
                update_flags_16(r, res, cf=0, of_=0)
                set_reg16(r, reg, res)
                continue

            #XOR AL, imm8 (34)
            if op == 0x34:
                r.al ^= self.fetch8()
                update_flags_8(r, r.al, cf=0, of_=0)
                continue

            # XOR AX, imm16 (35)
            if op == 0x35:
                r.ax ^= self.fetch16()
                update_flags_16(r, r.ax, cf=0, of_=0)
                continue

            #TEST r/m16, r16 (85)
            if op == 0x85:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                res = self._modrm_read16(mod, rm, ea) & get_reg16(r, reg)
                update_flags_16(r, res, cf=0, of_=0)
                continue

            #NOT / NEG / MUL / IMUL / DIV / IDIV / INC /DEC  r/m (F6/F7)
            if op == 0xF7:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                v = self._modrm_read16(mod, rm, ea)
                if reg == 0 or reg == 1:  # TEST
                    imm = self.fetch16()
                    update_flags_16(r, v & imm, cf=0, of_=0)
                elif reg == 2:  # NOT
                    self._modrm_write16(mod, rm, ea, (~v) & 0xFFFF)
                elif reg == 3:  # NEG
                    res = self._sub16(0, v)
                    self._modrm_write16(mod, rm, ea, res)
                elif reg == 4:  # MUL AX = AX * r/m16
                    res = r.ax * v
                    r.ax = res & 0xFFFF
                    r.dx = (res >> 16) & 0xFFFF
                    r.cf = r.of_ = 1 if r.dx else 0
                elif reg == 5:  # IMUL
                    res = sign16(r.ax) * sign16(v)
                    r.ax = res & 0xFFFF
                    r.dx = (res >> 16) & 0xFFFF
                elif reg == 6:  # DIV
                    dividend = (r.dx << 16) | r.ax
                    if v == 0: raise ZeroDivisionError("DIV by zero")
                    r.ax = (dividend // v) & 0xFFFF
                    r.dx = (dividend % v) & 0xFFFF
                elif reg == 7:  # IDIV
                    dividend = sign16(r.dx) * 0x10000 + r.ax
                    if v == 0: raise ZeroDivisionError("IDIV by zero")
                    r.ax = (dividend // sign16(v)) & 0xFFFF
                    r.dx = (dividend % sign16(v)) & 0xFFFF
                continue

            # FE group: INC/DEC r/m8
            if op == 0xFE:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                v = self._modrm_read8(mod, rm, ea)
                cf_save = r.cf
                if reg == 0:
                    res = self._add8(v, 1)
                else:
                    res = self._sub8(v, 1)
                self._modrm_write8(mod, rm, ea, res)
                r.cf = cf_save
                continue

            # FF group: INC/DEC/CALL/JMP/PUSH r/m16 
            if op == 0xFF:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                v = self._modrm_read16(mod, rm, ea)
                cf_save = r.cf
                if reg == 0:
                    res = self._add16(v, 1); self._modrm_write16(mod,rm,ea,res); r.cf=cf_save
                elif reg == 1:
                    res = self._sub16(v, 1); self._modrm_write16(mod,rm,ea,res); r.cf=cf_save
                elif reg == 2:  # CALL r/m16
                    self.push16(r.ip)
                    r.ip = v
                elif reg == 4:  # JMP r/m16
                    r.ip = v
                elif reg == 6:  # PUSH r/m16
                    self.push16(v)
                continue

            # SHL/SHR/SAR/ROL/ROR group (D0-D3)
            if op in (0xD0, 0xD1, 0xD2, 0xD3):
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                is16 = op in (0xD1, 0xD3)
                count = 1 if op in (0xD0, 0xD1) else (r.cl & 0x1F)
                if is16:
                    v = self._modrm_read16(mod, rm, ea)
                    if reg == 4:    # SHL
                        for _ in range(count): r.cf = (v>>15)&1; v = (v<<1)&0xFFFF
                        update_flags_16(r, v, cf=r.cf)
                    elif reg == 5:  # SHR
                        for _ in range(count): r.cf = v&1; v >>= 1
                        update_flags_16(r, v, cf=r.cf)
                    elif reg == 7:  # SAR
                        for _ in range(count): r.cf = v&1; v = (sign16(v) >> 1) & 0xFFFF
                        update_flags_16(r, v, cf=r.cf)
                    elif reg == 0:  # ROL
                        for _ in range(count): r.cf=(v>>15)&1; v=((v<<1)|(v>>15))&0xFFFF
                    elif reg == 1:  # ROR
                        for _ in range(count): r.cf=v&1; v=((v>>1)|(v<<15))&0xFFFF
                    self._modrm_write16(mod, rm, ea, v)
                else:
                    v = self._modrm_read8(mod, rm, ea)
                    if reg == 4:
                        for _ in range(count): r.cf=(v>>7)&1; v=(v<<1)&0xFF
                        update_flags_8(r, v, cf=r.cf)
                    elif reg == 5:
                        for _ in range(count): r.cf=v&1; v>>=1
                        update_flags_8(r, v, cf=r.cf)
                    elif reg == 7:
                        for _ in range(count): r.cf=v&1; v=(v|0x100 if v&0x80 else v)>>1
                        update_flags_8(r, v, cf=r.cf)
                    self._modrm_write8(mod, rm, ea, v)
                continue

            # JMP short (EB)
            if op == 0xEB:
                off = sign8(self.fetch8())
                r.ip = (r.ip + off) & 0xFFFF
                continue

            # JMP near (E9)
            if op == 0xE9:
                off = sign16(self.fetch16())
                r.ip = (r.ip + off) & 0xFFFF
                continue

            #JMP far (EA)
            if op == 0xEA:
                new_ip  = self.fetch16()
                new_cs  = self.fetch16()
                r.cs    = new_cs
                r.ip    = new_ip
                continue

            # CALL near (E8)
            if op == 0xE8:
                off = sign16(self.fetch16())
                self.push16(r.ip)
                r.ip = (r.ip + off) & 0xFFFF
                continue

            # CALL far (9A)
            if op == 0x9A:
                new_ip = self.fetch16()
                new_cs = self.fetch16()
                self.push16(r.cs)
                self.push16(r.ip)
                r.cs = new_cs
                r.ip = new_ip
                continue

            #RET near (C3)
            if op == 0xC3:
                r.ip = self.pop16()
                continue

            #RET far (CB)
            if op == 0xCB:
                r.ip = self.pop16()
                r.cs = self.pop16()
                continue

            #RET near imm16 (C2)
            if op == 0xC2:
                imm = self.fetch16()
                r.ip = self.pop16()
                r.sp = (r.sp + imm) & 0xFFFF
                continue

            # Conditional jumps (short)
            JCOND = {
                0x70: lambda r: r.of_,            # JO
                0x71: lambda r: not r.of_,         # JNO
                0x72: lambda r: r.cf,              # JB/JC
                0x73: lambda r: not r.cf,          # JNB/JNC/JAE
                0x74: lambda r: r.zf,              # JE/JZ
                0x75: lambda r: not r.zf,          # JNE/JNZ
                0x76: lambda r: r.cf or r.zf,      # JBE/JNA
                0x77: lambda r: not(r.cf or r.zf), # JA/JNBE
                0x78: lambda r: r.sf,              # JS
                0x79: lambda r: not r.sf,          # JNS
                0x7A: lambda r: r.pf,              # JP/JPE
                0x7B: lambda r: not r.pf,          # JNP/JPO
                0x7C: lambda r: r.sf != r.of_,     # JL/JNGE
                0x7D: lambda r: r.sf == r.of_,     # JGE/JNL
                0x7E: lambda r: r.zf or r.sf!=r.of_, # JLE/JNG
                0x7F: lambda r: not r.zf and r.sf==r.of_, # JG/JNLE
            }
            if op in JCOND:
                off = sign8(self.fetch8())
                if JCOND[op](r):
                    r.ip = (r.ip + off) & 0xFFFF
                continue

            #LOOP / LOOPE / LOOPNE
            if op == 0xE2:
                off = sign8(self.fetch8())
                r.cx = (r.cx - 1) & 0xFFFF
                if r.cx != 0:
                    r.ip = (r.ip + off) & 0xFFFF
                continue

            if op == 0xE1:
                off = sign8(self.fetch8())
                r.cx = (r.cx - 1) & 0xFFFF
                if r.cx != 0 and r.zf:
                    r.ip = (r.ip + off) & 0xFFFF
                continue

            if op == 0xE0:
                off = sign8(self.fetch8())
                r.cx = (r.cx - 1) & 0xFFFF
                if r.cx != 0 and not r.zf:
                    r.ip = (r.ip + off) & 0xFFFF
                continue

            # ---- JCXZ (E3) ----
            if op == 0xE3:
                off = sign8(self.fetch8())
                if r.cx == 0:
                    r.ip = (r.ip + off) & 0xFFFF
                continue

            # INT imm8 (CD)
            if op == 0xCD:
                num = self.fetch8()
                self.push16(r.flags_word())
                self.push16(r.cs)
                self.push16(r.ip)
                r.IF = 0
                self.bios.interrupt(num)
                # The BIOS handler sets real result flags (e.g. CF on INT 13h
                # success/failure) directly on `r`. Everyone must NOT clobber them
                # by popping the pre-call flags we pushed above — that would
                # silently discard the handler's result every single time,
                # which is exactly what was happening here: isolinux's drive
                # detection loop (PUSHA; INT 13h,AH=4B; POPA; JC skip) could
                # never see a real CF=1 from our handler because we always
                # restored the original (pre-interrupt) CF right after.
                # IRET still needs to restore IP/CS to resume after the INT,
                # but flags must come from the BIOS handler's result, not
                # from the saved pre-call copy.
                r.ip = self.pop16()
                r.cs = self.pop16()
                self.pop16()  # discard saved pre-call flags — handler's
                              # flags (already set on r by bios.interrupt)
                              # are what the caller should observe
                continue

            # IRET (CF)
            if op == 0xCF:
                r.ip = self.pop16()
                r.cs = self.pop16()
                r.set_flags_word(self.pop16())
                continue

            #IN AL, imm8 / IN AX, imm8
            if op == 0xE4:
                self.fetch8(); r.al = 0; continue
            if op == 0xE5:
                self.fetch8(); r.ax = 0; continue
            if op == 0xEC:
                r.al = 0; continue
            if op == 0xED:
                r.ax = 0; continue

            #OUT imm8/DX, AL/AX
            if op in (0xE6, 0xE7, 0xEE, 0xEF):
                if op in (0xE6, 0xE7): self.fetch8()
                continue  # ignore port writes for now

            #LEA r16, m (8D)
            if op == 0x8D:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                set_reg16(r, reg, ea if ea is not None else 0)
                continue

            #LDS / LES
            if op == 0xC5:  # LDS
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                v   = m.read16(self._seg(), ea)
                seg = m.read16(self._seg(), (ea+2)&0xFFFF)
                set_reg16(r, reg, v); r.ds = seg
                continue
            if op == 0xC4:  # LES
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                v   = m.read16(self._seg(), ea)
                seg = m.read16(self._seg(), (ea+2)&0xFFFF)
                set_reg16(r, reg, v); r.es = seg
                continue

            #XCHG r/m16, r16 (87)
            if op == 0x87:
                mrm = self.fetch8()
                mod, reg, rm, ea = decode_modrm(self, mrm)
                a = self._modrm_read16(mod, rm, ea)
                b = get_reg16(r, reg)
                self._modrm_write16(mod, rm, ea, b)
                set_reg16(r, reg, a)
                continue

            #STOSB / STOSW
            if op == 0xAA:
                m.write8(r.es, r.di, r.al)
                r.di = (r.di + (1 if not r.df else -1)) & 0xFFFF
                continue
            if op == 0xAB:
                m.write16(r.es, r.di, r.ax)
                r.di = (r.di + (2 if not r.df else -2)) & 0xFFFF
                continue

            # LODSB / LODSW 
            if op == 0xAC:
                r.al = m.read8(self._seg('ds'), r.si)
                r.si = (r.si + (1 if not r.df else -1)) & 0xFFFF
                continue
            if op == 0xAD:
                r.ax = m.read16(self._seg('ds'), r.si)
                r.si = (r.si + (2 if not r.df else -2)) & 0xFFFF
                continue

            #  MOVSB / MOVSW
            if op == 0xA4:
                m.write8(r.es, r.di, m.read8(self._seg('ds'), r.si))
                r.si = (r.si + (1 if not r.df else -1)) & 0xFFFF
                r.di = (r.di + (1 if not r.df else -1)) & 0xFFFF
                continue
            if op == 0xA5:
                m.write16(r.es, r.di, m.read16(self._seg('ds'), r.si))
                r.si = (r.si + (2 if not r.df else -2)) & 0xFFFF
                r.di = (r.di + (2 if not r.df else -2)) & 0xFFFF
                continue

            # SCASB 
            if op == 0xAE:
                self._sub8(r.al, m.read8(r.es, r.di))
                r.di = (r.di + (1 if not r.df else -1)) & 0xFFFF
                continue

            # CBW 
            if op == 0x98:
                r.ax = sign8(r.al) & 0xFFFF
                continue

            #  CWD 
            if op == 0x99:
                r.dx = 0xFFFF if r.ax & 0x8000 else 0
                continue

            # LAHF / SAHF
            if op == 0x9F:
                r.ah = r.flags_word() & 0xFF
                continue
            if op == 0x9E:
                f = r.flags_word()
                r.set_flags_word((f & 0xFF00) | r.ah)
                continue

            # 0x0F two-byte opcodes
            if op == 0x0F:
                op2 = self.fetch8()
                # 0F 84..8F: near conditional jumps
                if 0x84 <= op2 <= 0x8F:
                    off = sign16(self.fetch16())
                    JCOND2 = {
                        0x84: lambda r: r.zf,
                        0x85: lambda r: not r.zf,
                        0x82: lambda r: r.cf,
                        0x83: lambda r: not r.cf,
                        0x86: lambda r: r.cf or r.zf,
                        0x87: lambda r: not(r.cf or r.zf),
                        0x8C: lambda r: r.sf != r.of_,
                        0x8D: lambda r: r.sf == r.of_,
                        0x8E: lambda r: r.zf or r.sf!=r.of_,
                        0x8F: lambda r: not r.zf and r.sf==r.of_,
                        0x88: lambda r: r.sf,
                        0x89: lambda r: not r.sf,
                    }
                    if op2 in JCOND2 and JCOND2[op2](r):
                        r.ip = (r.ip + off) & 0xFFFF
                else:
                    # unknown 0F opcode — skip
                    pass
                continue

            #Unknown opcode
            r.ip = (r.ip - 1) & 0xFFFF  # rewind
            print(f"\n[!] Unknown opcode 0x{op:02X} at CS:IP={r.cs:04X}:{r.ip:04X}")
            print(r)
            break

        return self.icount



# Bootsector builder (pure Python — no NASM needed)
# Prints "Hello from x86emu!" via INT 10h then halts.
def build_bootsector():
    """
    Assemble a minimal bootsector that prints a string via INT 10h.
    Returns exactly 512 bytes with 0x55AA signature.

    Memory layout when BIOS loads us:
      CS=0x0000, IP=0x7C00, SS=0x0000, SP=0x7C00
    """
    msg = b"Hello from x86emu!\r\n"

    code = bytearray()

    # Set up segments
    code += b'\x31\xC0'             # XOR AX, AX
    code += b'\x8E\xD8'             # MOV DS, AX
    code += b'\x8E\xC0'             # MOV ES, AX

    # Point SI at message (relative to load address 0x7C00)
    # We'll place the message right after the code
    # msg_offset = 0x7C00 + len(code) + 7  (7 = bytes of following instructions before msg)
    # We'll use a label trick: encode SI load after we know code size

    # MOV SI, msg_addr (placeholder — patch below)
    code += b'\xBE\x00\x00'         # MOV SI, imm16  (bytes 7,8 = address)
    si_patch_offset = len(code) - 2

    # Print loop: LODSB; test AL,AL; jz done; INT 10h; jmp loop
    loop_start = len(code)
    code += b'\xAC'                 # LODSB
    code += b'\x08\xC0'             # OR AL, AL
    code += b'\x74\x07'             # JZ done (+7)
    code += b'\xB4\x0E'             # MOV AH, 0x0E
    code += b'\xBB\x07\x00'         # MOV BX, 7  (page 0, fg color 7)
    code += b'\xCD\x10'             # INT 10h
    # JMP back to loop_start relative
    jmp_back = loop_start - (len(code) + 2)
    code += bytes([0xEB, jmp_back & 0xFF])   # JMP SHORT loop_start

    # done: HLT
    code += b'\xF4'                 # HLT
    code += b'\xEB\xFE'             # JMP $ (infinite loop safety net)

    # Message data
    msg_offset_in_sector = 0x7C00 + len(code)
    # Patch SI load
    code[si_patch_offset]     = msg_offset_in_sector & 0xFF
    code[si_patch_offset + 1] = (msg_offset_in_sector >> 8) & 0xFF

    code += msg

    # Pad to 510 bytes and add boot signature
    assert len(code) <= 510, f"Bootsector code too large: {len(code)} bytes"
    code += b'\x00' * (510 - len(code))
    code += b'\x55\xAA'

    return bytes(code)



# Machine: adds my chaos together :D

class Machine:
    def __init__(self):
        self.mem  = Memory()
        self.reg  = Registers()
        self.bios = BIOS(self.mem, self.reg)
        self.cpu  = CPU(self.mem, self.reg, self.bios)

    def load_bootsector(self, data: bytes):
        assert len(data) == 512
        assert data[-2:] == b'\x55\xAA', "Missing boot signature 0x55AA"
        self.mem.load(0x7C00, data)
        # BIOS hands control to 0000:7C00
        self.reg.cs = 0x0000
        self.reg.ip = 0x7C00
        self.reg.ss = 0x0000
        self.reg.sp = 0x7C00   # stack grows down from 0x7C00
        self.reg.ds = 0x0000
        self.reg.es = 0x0000

    def run(self, max_icount=500_000):
        self.cpu.max_icount = max_icount
        return self.cpu.run()



# Main

if __name__ == '__main__':
    import time

    print("=" * 50)
    print("x86emu Phase 1 — Real Mode Bootsector")
    print("=" * 50)

    machine = Machine()

    # Build our test bootsector
    sector = build_bootsector()
    machine.load_bootsector(sector)

    print("Output from bootsector:")
    print("-" * 50)

    t0 = time.time()
    icount = machine.run()
    elapsed = time.time() - t0

    print()
    print("-" * 50)
    print(f"Halted after {icount} instructions in {elapsed:.3f}s")
    print(f"({icount/elapsed:.0f} instructions/sec)")
    print()
    print("Final register state:")
    print(machine.reg)

    output = machine.bios.get_output()
    print()
    if "Hello from x86emu!" in output:
        print("✓ Phase 1 PASS — bootsector ran successfully")
    else:
        print("✗ Phase 1 FAIL — expected 'Hello from x86emu!' in output")
        print(f"  Got: {repr(output)}")
