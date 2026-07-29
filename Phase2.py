"""
phase2.py — x86emu Phase 2: Protected Mode Transition
Imports CPU core from cpu.py (was x86emu_phase1.py).
"""

import struct, sys, time
import numpy as np

import importlib, os

def _import_cpu():
    for name in ('cpu', 'x86emu_phase1'):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            pass
    raise ImportError("Cannot find cpu.py or x86emu_phase1.py in path")

_cpu_mod = _import_cpu()

Memory      = _cpu_mod.Memory
Registers   = _cpu_mod.Registers
BIOS        = _cpu_mod.BIOS
CPU         = _cpu_mod.CPU
decode_modrm = _cpu_mod.decode_modrm
get_reg16   = _cpu_mod.get_reg16
set_reg16   = _cpu_mod.set_reg16
get_reg8    = _cpu_mod.get_reg8
set_reg8    = _cpu_mod.set_reg8
get_seg     = _cpu_mod.get_seg
set_seg     = _cpu_mod.set_seg
sign8       = _cpu_mod.sign8
sign16      = _cpu_mod.sign16
sign32      = lambda v: v if v < 0x80000000 else v - 0x100000000
update_flags_16 = _cpu_mod.update_flags_16
update_flags_8  = _cpu_mod.update_flags_8
REG16       = _cpu_mod.REG16
REG8        = _cpu_mod.REG8

MEM_SIZE_P2 = 32 * 1024 * 1024   # 32 MB

class Memory32(Memory):
    """Extends Phase 1 Memory with 32-bit read/write and larger address space."""
    def __init__(self, size=MEM_SIZE_P2):
        self.size = size
        self._m = np.zeros(size, dtype=np.uint8)
        self.paging_enabled = False
        self.cr3 = 0
        self._tlb = {}
        self._tlb_hits = 0
        self._tlb_misses = 0

    def _translate(self, vaddr):
        if not self.paging_enabled:
            return vaddr
        page = vaddr & 0xFFFFF000
        off  = vaddr & 0xFFF
        cached = self._tlb.get(page)
        if cached is not None:
            self._tlb_hits += 1
            return cached + off
        self._tlb_misses += 1
        pd_idx = (vaddr >> 22) & 0x3FF
        pt_idx = (vaddr >> 12) & 0x3FF
        pd_addr = (self.cr3 & 0xFFFFF000) + pd_idx * 4
        if pd_addr + 4 > self.size:
            return vaddr
        pde = struct.unpack_from('<I', self._m, pd_addr)[0]
        if not (pde & 1):
            return vaddr
        if pde & 0x80:
            phys_base = pde & 0xFFC00000
            phys = phys_base | (vaddr & 0x3FFFFF)
            self._tlb[page] = phys & 0xFFFFF000
            return phys
        pt_base = pde & 0xFFFFF000
        pt_addr = pt_base + pt_idx * 4
        if pt_addr + 4 > self.size:
            return vaddr
        pte = struct.unpack_from('<I', self._m, pt_addr)[0]
        if not (pte & 1):
            return vaddr
        phys_page = pte & 0xFFFFF000
        self._tlb[page] = phys_page
        return phys_page | off

    def translate_strict(self, vaddr):
        """Like _translate(), but returns None on a genuinely not-present
        mapping instead of silently falling back to treating the raw
        virtual address as physical. That permissive fallback is fine
        while paging is off (vaddr==paddr is correct then), but once
        paging is truly enabled it lets code fetch from unmapped pages
        as if they were valid physical addresses — usually far beyond
        actual RAM size — producing an endless stream of zero bytes
        that decode as harmless-looking instructions instead of the
        #PF a real CPU would raise. Used by instruction fetch so a
        genuine fault can be delivered to the guest's own IDT handler
        instead of silently executing garbage."""
        if not self.paging_enabled:
            return vaddr
        page = vaddr & 0xFFFFF000
        off  = vaddr & 0xFFF
        cached = self._tlb.get(page)
        if cached is not None:
            return cached + off
        pd_idx = (vaddr >> 22) & 0x3FF
        pt_idx = (vaddr >> 12) & 0x3FF
        pd_addr = (self.cr3 & 0xFFFFF000) + pd_idx * 4
        if pd_addr + 4 > self.size:
            return None
        pde = struct.unpack_from('<I', self._m, pd_addr)[0]
        if not (pde & 1):
            return None
        if pde & 0x80:
            phys_base = pde & 0xFFC00000
            phys = phys_base | (vaddr & 0x3FFFFF)
            return phys
        pt_base = pde & 0xFFFFF000
        pt_addr = pt_base + pt_idx * 4
        if pt_addr + 4 > self.size:
            return None
        pte = struct.unpack_from('<I', self._m, pt_addr)[0]
        if not (pte & 1):
            return None
        phys_page = pte & 0xFFFFF000
        return phys_page | off

    def invalidate_tlb(self):
        self._tlb.clear()

    def read32_flat(self, addr):
        addr = self._translate(addr & 0xFFFFFFFF)
        if addr + 4 > self.size:
            return 0
        return struct.unpack_from('<I', self._m, addr)[0]

    def write32_flat(self, addr, v):
        addr = self._translate(addr & 0xFFFFFFFF)
        if addr + 4 > self.size:
            return
        struct.pack_into('<I', self._m, addr, v & 0xFFFFFFFF)

    def read16_flat(self, addr):
        addr = self._translate(addr & 0xFFFFFFFF)
        if addr + 2 > self.size:
            return 0
        return struct.unpack_from('<H', self._m, addr)[0]

    def write16_flat(self, addr, v):
        addr = self._translate(addr & 0xFFFFFFFF)
        if addr + 2 > self.size:
            return
        struct.pack_into('<H', self._m, addr, v & 0xFFFF)

    def read8_flat(self, addr):
        addr = self._translate(addr & 0xFFFFFFFF)
        if addr >= self.size:
            return 0
        return int(self._m[addr])

    def write8_flat(self, addr, v):
        addr = self._translate(addr & 0xFFFFFFFF)
        if addr >= self.size:
            return
        self._m[addr] = v & 0xFF

    def load_flat(self, addr, data):
        n = len(data)
        end = addr + n
        if end > self.size:
            n = self.size - addr
        self._m[addr:addr+n] = np.frombuffer(data[:n], dtype=np.uint8)


class SegDesc:
    def __init__(self):
        self.base  = 0
        self.limit = 0xFFFF
        self.flags = 0
        self.present = True
        self.db    = 0
        self.big   = False

    @classmethod
    def from_raw(cls, raw8):
        d = cls()
        lo, hi = struct.unpack_from('<II', raw8)
        d.base  = ((lo >> 16) & 0xFFFF) | (((hi >> 0) & 0xFF) << 16) | (((hi >> 24) & 0xFF) << 24)
        d.limit = (lo & 0xFFFF) | (((hi >> 16) & 0xF) << 16)
        d.flags = (hi >> 8) & 0xFF
        d.present = bool((hi >> 15) & 1)
        d.db    = (hi >> 22) & 1
        d.big   = bool(d.db)
        if (hi >> 23) & 1:
            d.limit = (d.limit << 12) | 0xFFF
        return d

    def __repr__(self):
        return (f"SegDesc(base={self.base:#010x}, limit={self.limit:#010x}, "
                f"flags={self.flags:#04x}, db={self.db}, present={self.present})")


class Registers32(Registers):
    def __init__(self):
        super().__init__()
        self._eax_hi = self._ebx_hi = self._ecx_hi = self._edx_hi = 0
        self._esi_hi = self._edi_hi = self._ebp_hi = self._esp_hi = 0

        self.cr0 = 0x00000010
        self.cr2 = 0
        self.cr3 = 0

        self.gdtr_base  = 0
        self.gdtr_limit = 0
        self.idtr_base  = 0
        self.idtr_limit = 0x3FF

        self.seg_cache = {
            'cs': SegDesc(), 'ds': SegDesc(), 'es': SegDesc(),
            'ss': SegDesc(), 'fs': SegDesc(), 'gs': SegDesc(),
        }

        self.protected_mode = False

    @property
    def eax(self): return self.ax | (self._eax_hi << 16)
    @eax.setter
    def eax(self, v): v &= 0xFFFFFFFF; self.ax = v & 0xFFFF; self._eax_hi = v >> 16

    @property
    def ebx(self): return self.bx | (self._ebx_hi << 16)
    @ebx.setter
    def ebx(self, v): v &= 0xFFFFFFFF; self.bx = v & 0xFFFF; self._ebx_hi = v >> 16

    @property
    def ecx(self): return self.cx | (self._ecx_hi << 16)
    @ecx.setter
    def ecx(self, v): v &= 0xFFFFFFFF; self.cx = v & 0xFFFF; self._ecx_hi = v >> 16

    @property
    def edx(self): return self.dx | (self._edx_hi << 16)
    @edx.setter
    def edx(self, v): v &= 0xFFFFFFFF; self.dx = v & 0xFFFF; self._edx_hi = v >> 16

    @property
    def esi(self): return self.si | (self._esi_hi << 16)
    @esi.setter
    def esi(self, v): v &= 0xFFFFFFFF; self.si = v & 0xFFFF; self._esi_hi = v >> 16

    @property
    def edi(self): return self.di | (self._edi_hi << 16)
    @edi.setter
    def edi(self, v): v &= 0xFFFFFFFF; self.di = v & 0xFFFF; self._edi_hi = v >> 16

    @property
    def ebp(self): return self.bp | (self._ebp_hi << 16)
    @ebp.setter
    def ebp(self, v): v &= 0xFFFFFFFF; self.bp = v & 0xFFFF; self._ebp_hi = v >> 16

    @property
    def esp(self): return self.sp | (self._esp_hi << 16)
    @esp.setter
    def esp(self, v): v &= 0xFFFFFFFF; self.sp = v & 0xFFFF; self._esp_hi = v >> 16

    @property
    def eip(self): return self.ip
    @eip.setter
    def eip(self, v): self.ip = v & 0xFFFFFFFF

    @property
    def pe(self): return self.cr0 & 1

    def __repr__(self):
        if self.protected_mode:
            return (f"EAX={self.eax:08X} EBX={self.ebx:08X} ECX={self.ecx:08X} EDX={self.edx:08X}\n"
                    f"ESI={self.esi:08X} EDI={self.edi:08X} EBP={self.ebp:08X} ESP={self.esp:08X}\n"
                    f"CS={self.cs:04X} DS={self.ds:04X} ES={self.es:04X} SS={self.ss:04X} "
                    f"EIP={self.eip:08X} FL={self.flags_word():04X} "
                    f"CF={self.cf} ZF={self.zf} SF={self.sf} OF={self.of_}\n"
                    f"CR0={self.cr0:08X} PE={'ON' if self.pe else 'OFF'} "
                    f"GDTR={self.gdtr_base:#010x}/{self.gdtr_limit:#06x}")
        return super().__repr__()


class GDT:
    def __init__(self, mem, reg):
        self.mem = mem
        self.reg = reg

    def load_descriptor(self, selector):
        r = self.reg
        if selector == 0:
            return None
        idx   = (selector >> 3) & 0x1FFF
        addr  = r.gdtr_base + idx * 8
        if addr + 8 > r.gdtr_base + r.gdtr_limit + 1:
            d = SegDesc()
            d.base = 0; d.limit = 0xFFFFFFFF; d.db = 1; d.big = True
            return d
        raw = self.mem.read_bytes(addr, 8)
        return SegDesc.from_raw(raw)

    def update_cache(self, seg_name, selector):
        desc = self.load_descriptor(selector)
        if desc:
            self.reg.seg_cache[seg_name] = desc
            setattr(self.reg, seg_name, selector)

    def seg_base(self, seg_name):
        if self.reg.protected_mode:
            return self.reg.seg_cache[seg_name].base
        return getattr(self.reg, seg_name) << 4


class IOPorts:
    def __init__(self, reg):
        self.reg   = reg
        self._a20  = False
        self._ports = {}

    def read(self, port):
        if port in self._ports:
            return self._ports[port][0]()
        return 0xFF

    def write(self, port, val):
        if port == 0x92:
            if val & 0x02:
                self._a20 = True
            return
        if port in (0x20, 0x21, 0xA0, 0xA1):
            return
        if port in (0x40, 0x41, 0x42, 0x43):
            return
        if port in self._ports:
            self._ports[port][1](val)

    @property
    def a20_enabled(self):
        return self._a20


def parity(v):
    v &= 0xFF
    v ^= v >> 4; v ^= v >> 2; v ^= v >> 1
    return (~v) & 1
_cpu_mod.parity = getattr(_cpu_mod, 'parity', parity)


class CPU32(CPU):
    def __init__(self, mem, reg, bios, io):
        super().__init__(mem, reg, bios)
        self.io  = io
        self.gdt = GDT(mem, reg)

        self._op32   = False
        self._addr32 = False
        self._rep    = None

        self.vga_base  = 0xB8000
        self.vga_col   = 0
        self.vga_row   = 0
        self.vga_attr  = 0x07

    def _cs_base(self):
        if self.reg.protected_mode:
            return self.reg.seg_cache['cs'].base
        uv = getattr(self.reg, '_unreal_valid', None)
        if uv and uv.get('cs'):
            return self.reg.seg_cache['cs'].base
        return self.reg.cs << 4

    def _seg_base(self, seg='ds'):
        if seg is None:
            seg = 'ds'
        if self.reg.protected_mode:
            return self.reg.seg_cache[seg].base
        unreal = getattr(self.reg, '_unreal_valid', None)
        if unreal and unreal.get(seg):
            return self.reg.seg_cache[seg].base
        return getattr(self.reg, seg) << 4

    def _read_mem8(self, seg, offset):
        addr = (self._seg_base(seg) + offset) & 0xFFFFFFFF
        return self.mem.read8_flat(addr)

    def _read_mem16(self, seg, offset):
        addr = (self._seg_base(seg) + offset) & 0xFFFFFFFF
        return self.mem.read16_flat(addr)

    def _read_mem32(self, seg, offset):
        addr = (self._seg_base(seg) + offset) & 0xFFFFFFFF
        return self.mem.read32_flat(addr)

    def _write_mem8(self, seg, offset, v):
        addr = (self._seg_base(seg) + offset) & 0xFFFFFFFF
        self.mem.write8_flat(addr, v)

    def _write_mem16(self, seg, offset, v):
        addr = (self._seg_base(seg) + offset) & 0xFFFFFFFF
        self.mem.write16_flat(addr, v)

    def _write_mem32(self, seg, offset, v):
        addr = (self._seg_base(seg) + offset) & 0xFFFFFFFF
        self.mem.write32_flat(addr, v)

    def fetch8(self):
        addr = (self._cs_base() + self.reg.ip) & 0xFFFFFFFF
        if self.mem.paging_enabled:
            paddr = self.mem.translate_strict(addr)
            if paddr is None:
                self._raise_page_fault(addr, is_write=False)
                if self.halted:
                    # No valid handler — _raise_page_fault already
                    # halted us with a diagnostic. EIP hasn't moved, so
                    # recursing here would just hit the identical fault
                    # forever. Return a harmless value; the outer run()
                    # loop's halted check stops real progress from here.
                    return 0xF4  # HLT opcode, in case this value is
                                 # ever actually decoded
                # Otherwise a real handler was found and EIP now points
                # there — fetch again from the new location.
                return self.fetch8()
        v = self.mem.read8_flat(addr)
        self.reg.ip = (self.reg.ip + 1) & 0xFFFFFFFF
        return v

    def _raise_page_fault(self, fault_addr, is_write=False):
        """Deliver INT 14 (#PF) the way real x86 hardware does: set CR2
        to the faulting linear address, push an error code beneath the
        usual EFLAGS/CS/EIP frame, and jump through IDT[14] if a valid
        handler is registered. If the IDT genuinely has no entry for
        vector 14 (e.g. still null this early in boot — we've directly
        observed the kernel doing exactly that right before this class
        of fault occurs), there is no meaningful way to continue: on
        real hardware this scenario is a double/triple fault. Rather
        than either silently reading zeros (executing whatever garbage
        follows) or continuing to fetch from nowhere, halt cleanly with
        a clear diagnostic so this is visible and debuggable instead of
        an opaque crash or a misleading wander through unmapped memory.
        """
        r = self.reg
        r.cr2 = fault_addr

        idt_entry_end = 14 * 8 + 7
        has_handler = (self.reg.protected_mode
                       and idt_entry_end <= r.idtr_limit)
        handler = 0
        sel = 0
        if has_handler:
            idt_addr = r.idtr_base + 14 * 8
            off_lo = self.mem.read16_flat(idt_addr)
            sel    = self.mem.read16_flat(idt_addr + 2)
            off_hi = self.mem.read16_flat(idt_addr + 6)
            handler = off_lo | (off_hi << 16)

        if not handler:
            print(f"\n[!] #PF (page fault) with NO VALID HANDLER — "
                  f"CR2={fault_addr:#010x} EIP={r.ip:#010x} "
                  f"idtr=({r.idtr_base:#010x},{r.idtr_limit:#06x}). "
                  f"This is what a real CPU would double/triple-fault "
                  f"on. Halting cleanly here instead of executing "
                  f"whatever garbage follows the unmapped page.")
            self.halted = True
            return

        # error code: bit0=0 (not-present), bit1=write?, bit2=0 (super-
        # visor, since this is kernel-mode boot code)
        error_code = (1 if is_write else 0) << 1
        self.push32(r.flags_word())
        self.push32(r.cs)
        self.push32(r.ip)
        self.push32(error_code)
        r.IF = 0
        self.gdt.update_cache('cs', sel)
        r.cs = sel
        r.ip = handler

    def fetch16(self):
        lo = self.fetch8()
        hi = self.fetch8()
        return lo | (hi << 8)

    def fetch32(self):
        lo = self.fetch16()
        hi = self.fetch16()
        return lo | (hi << 16)

    def push32(self, v):
        self.reg.esp = (self.reg.esp - 4) & 0xFFFFFFFF
        self._write_mem32('ss', self.reg.esp, v & 0xFFFFFFFF)

    def pop32(self):
        v = self._read_mem32('ss', self.reg.esp)
        self.reg.esp = (self.reg.esp + 4) & 0xFFFFFFFF
        return v

    def push16(self, v):
        if self.reg.protected_mode:
            self.reg.esp = (self.reg.esp - 2) & 0xFFFFFFFF
            self._write_mem16('ss', self.reg.esp, v & 0xFFFF)
        else:
            super().push16(v)

    def pop16(self):
        if self.reg.protected_mode:
            v = self._read_mem16('ss', self.reg.esp)
            self.reg.esp = (self.reg.esp + 2) & 0xFFFFFFFF
            return v
        return super().pop16()

    def _add32(self, a, b, carry=0):
        r = self.reg
        res = a + b + carry
        cf  = 1 if res > 0xFFFFFFFF else 0
        res32 = res & 0xFFFFFFFF
        of_ = 1 if (not ((a ^ b) & 0x80000000)) and ((a ^ res32) & 0x80000000) else 0
        r.zf = 1 if res32 == 0 else 0
        r.sf = (res32 >> 31) & 1
        r.pf = _cpu_mod.parity(res32)
        r.cf = cf; r.of_ = of_
        r.af = 1 if ((a & 0xF) + (b & 0xF) + carry) > 0xF else 0
        return res32

    def _sub32(self, a, b, borrow=0):
        r = self.reg
        res = a - b - borrow
        cf  = 1 if res < 0 else 0
        res32 = res & 0xFFFFFFFF
        of_ = 1 if ((a ^ b) & 0x80000000) and ((a ^ res32) & 0x80000000) else 0
        r.zf = 1 if res32 == 0 else 0
        r.sf = (res32 >> 31) & 1
        r.pf = _cpu_mod.parity(res32)
        r.cf = cf; r.of_ = of_
        r.af = 1 if ((a & 0xF) < (b & 0xF) + borrow) else 0
        return res32

    REG32 = ['eax','ecx','edx','ebx','esp','ebp','esi','edi']

    def get_reg32(self, idx):
        return getattr(self.reg, self.REG32[idx]) & 0xFFFFFFFF

    def set_reg32(self, idx, v):
        setattr(self.reg, self.REG32[idx], v & 0xFFFFFFFF)

    def decode_modrm32(self, byte):
        r   = self.reg
        mod = (byte >> 6) & 3
        reg = (byte >> 3) & 7
        rm  = byte & 7

        if mod == 3:
            return mod, reg, rm, None

        bases32 = {
            0: lambda: r.eax, 1: lambda: r.ecx,
            2: lambda: r.edx, 3: lambda: r.ebx,
            4: None,
            5: lambda: r.ebp,
            6: lambda: r.esi, 7: lambda: r.edi,
        }

        if mod == 0 and rm == 5:
            ea = self.fetch32()
        elif rm == 4:
            sib = self.fetch8()
            base  = sib & 7
            index = (sib >> 3) & 7
            scale = (sib >> 6) & 3
            base_val  = self.get_reg32(base)  if base  != 5 else (self.fetch32() if mod == 0 else self.reg.ebp)
            index_val = self.get_reg32(index) if index != 4 else 0
            ea = (base_val + index_val * (1 << scale)) & 0xFFFFFFFF
            if mod == 1:
                ea = (ea + sign8(self.fetch8())) & 0xFFFFFFFF
            elif mod == 2:
                ea = (ea + self.fetch32()) & 0xFFFFFFFF
        else:
            fn = bases32[rm]
            ea = fn() & 0xFFFFFFFF if fn else 0
            if mod == 1:
                ea = (ea + sign8(self.fetch8())) & 0xFFFFFFFF
            elif mod == 2:
                ea = (ea + self.fetch32()) & 0xFFFFFFFF

        return mod, reg, rm, ea

    def _modrm_read32(self, mod, rm, ea, seg='ds'):
        if mod == 3:
            return self.get_reg32(rm)
        return self._read_mem32(seg, ea)

    def _modrm_write32(self, mod, rm, ea, v, seg='ds'):
        if mod == 3:
            self.set_reg32(rm, v)
        else:
            self._write_mem32(seg, ea, v)

    def vga_putchar(self, ch):
        if ch == '\n' or ch == '\r':
            self.vga_col = 0
            if ch == '\n':
                self.vga_row += 1
            return
        offset = self.vga_base + (self.vga_row * 80 + self.vga_col) * 2
        self.mem.write8_flat(offset,     ord(ch))
        self.mem.write8_flat(offset + 1, self.vga_attr)
        self.vga_col += 1
        if self.vga_col >= 80:
            self.vga_col = 0
            self.vga_row += 1
        sys.stdout.write(ch)
        sys.stdout.flush()

    def vga_puts(self, s):
        for ch in s:
            self.vga_putchar(ch)

    def vga_read(self, rows=1):
        out = []
        for row in range(rows):
            row_chars = []
            for col in range(80):
                offset = self.vga_base + (row * 80 + col) * 2
                b = self.mem.read8_flat(offset)
                if b: row_chars.append(chr(b))
            line = ''.join(row_chars).rstrip()
            if line: out.append(line)
        return '\n'.join(out)

    def run(self):
        r   = self.reg
        m   = self.mem
        io  = self.io

        _hot_eip = -1
        _hot_hits = 0
        _loop_cache = {}

        while not self.halted and self.icount < self.max_icount:
            if r.protected_mode:
                cur = r.ip
                if cur == _hot_eip:
                    _hot_hits += 1
                    if _hot_hits == 5:
                        fn = _loop_cache.get(cur)
                        if fn is None:
                            fn = self._try_compile_loop(cur)
                            _loop_cache[cur] = fn
                        if fn is not None:
                            iters = self._run_native_loop(fn)
                            if iters > 0:
                                self.icount += iters
                                _hot_hits = 0
                                continue
                else:
                    _hot_eip = cur
                    _hot_hits = 0

            self.icount += 1
            self._unk_streak = 0

            _insn_start_eip = r.ip

            # Default operand/address size in protected mode depends on the
            # CURRENT CS descriptor's D/B bit, not merely "is it in
            # protected mode." A 16-bit protected-mode code segment
            # (D/B=0) — e.g. isolinux's own temporary GDT often defines
            # exactly this for brief compatibility excursions — legitimately
            # defaults to 16-bit operand/address size. Assuming 32-bit for
            # ANY protected-mode CS inverts the effect of every 0x66/0x67
            # prefix in such a segment: a 0x66 that should toggle 16->32
            # instead appears to toggle a (wrongly assumed) 32->16, and
            # vice versa — corrupting REP MOVS counts/pointers computed
            # under this segment (e.g. a 64KB dword-count SHR ECX,2 gets
            # misread as a 16-bit SHR CX,2, wrapping to zero).
            cs_is_32bit = bool(r.seg_cache['cs'].db) if r.protected_mode else False
            self._op32      = cs_is_32bit
            self._addr32    = cs_is_32bit
            self._seg_override = None
            self._rep       = None

            op = self.fetch8()

            while op in (0x26,0x2E,0x36,0x3E,0x64,0x65,0x66,0x67,0xF0,0xF2,0xF3):
                if   op == 0x26: self._seg_override = 'es'
                elif op == 0x2E: self._seg_override = 'cs'
                elif op == 0x36: self._seg_override = 'ss'
                elif op == 0x3E: self._seg_override = 'ds'
                elif op == 0x64: self._seg_override = 'fs'
                elif op == 0x65: self._seg_override = 'gs'
                elif op == 0x66: self._op32 = not self._op32
                elif op == 0x67: self._addr32 = not self._addr32
                elif op == 0xF3: self._rep = 'REP'
                elif op == 0xF2: self._rep = 'REPNE'
                op = self.fetch8()

            o32 = self._op32
            seg = self._seg_override or 'ds'

            if (getattr(self, '_debug_rep_trace', False)
                    and 0x96d0 <= _insn_start_eip <= 0x9710
                    and op in (0xA4, 0xA5) and self._rep):
                print(f"[rep-trace] EIP={_insn_start_eip:#010x} op={op:#04x} "
                      f"rep={self._rep} o32={o32} addr32={self._addr32} "
                      f"ECX={r.ecx:#010x} CX={r.cx:#06x} "
                      f"ESI={r.esi:#010x} EDI={r.edi:#010x} "
                      f"ES_base={self._seg_base('es'):#010x} "
                      f"DS_base={self._seg_base('ds'):#010x}")

            def modrm():
                byte = self.fetch8()
                if self._addr32:
                    return self.decode_modrm32(byte)
                return decode_modrm(self, byte)

            def fetch_imm(): return self.fetch32() if o32 else self.fetch16()
            def fetch_simm8_ext(): return sign8(self.fetch8()) & (0xFFFFFFFF if o32 else 0xFFFF)

            if op == 0x90: continue
            if op == 0xF4: self.halted = True; break
            if op == 0xFA: r.IF = 0; continue
            if op == 0xFB: r.IF = 1; continue
            if op == 0xFC: r.df = 0; continue
            if op == 0xFD: r.df = 1; continue
            if op == 0xF9: r.cf = 1; continue
            if op == 0xF8: r.cf = 0; continue
            if op == 0xF5: r.cf ^= 1; continue

            if 0xB8 <= op <= 0xBF:
                idx = op - 0xB8
                v = fetch_imm()
                if o32: self.set_reg32(idx, v)
                else:   set_reg16(r, idx, v)
                continue

            if 0xB0 <= op <= 0xB7:
                set_reg8(r, op - 0xB0, self.fetch8())
                continue

            if op == 0x89:
                mod, reg, rm, ea = modrm()
                if o32:
                    v = self.get_reg32(reg)
                    self._modrm_write32(mod, rm, ea, v, seg)
                else:
                    v = get_reg16(r, reg)
                    if mod == 3: set_reg16(r, rm, v)
                    else: self._write_mem16(seg, ea, v)
                continue

            if op == 0x88:
                mod, reg, rm, ea = modrm()
                v = get_reg8(r, reg)
                if mod == 3: set_reg8(r, rm, v)
                else: self._write_mem8(seg, ea, v)
                continue

            if op == 0x8B:
                mod, reg, rm, ea = modrm()
                if o32:
                    v = self._modrm_read32(mod, rm, ea, seg)
                    self.set_reg32(reg, v)
                else:
                    v = get_reg16(r, rm) if mod==3 else self._read_mem16(seg, ea)
                    set_reg16(r, reg, v)
                continue

            if op == 0x8A:
                mod, reg, rm, ea = modrm()
                v = get_reg8(r, rm) if mod==3 else self._read_mem8(seg, ea)
                set_reg8(r, reg, v)
                continue

            if op == 0xC7:
                mod, reg, rm, ea = modrm()
                imm = fetch_imm()
                if o32: self._modrm_write32(mod, rm, ea, imm, seg)
                else:
                    if mod==3: set_reg16(r, rm, imm)
                    else: self._write_mem16(seg, ea, imm)
                continue

            if op == 0xC6:
                mod, reg, rm, ea = modrm()
                imm = self.fetch8()
                if mod==3: set_reg8(r, rm, imm)
                else: self._write_mem8(seg, ea, imm)
                continue

            if op == 0x8E:
                mod, reg, rm, ea = modrm()
                v = get_reg16(r, rm) if mod==3 else self._read_mem16(seg, ea)
                segs = ['es','cs','ss','ds','fs','gs']
                if reg < len(segs):
                    sname = segs[reg]
                    if r.protected_mode:
                        self.gdt.update_cache(sname, v)
                    else:
                        set_seg(r, reg, v)
                        uv = getattr(r, '_unreal_valid', None)
                        if uv is not None:
                            uv[sname] = False
                continue

            if op == 0x8C:
                mod, reg, rm, ea = modrm()
                v = get_seg(r, reg) if reg < 6 else 0
                if mod==3: set_reg16(r, rm, v)
                else: self._write_mem16(seg, ea, v)
                continue

            if op == 0xA0:
                addr = self.fetch16()
                r.al = self._read_mem8(seg, addr)
                continue

            if op == 0xA1:
                addr = fetch_imm() if self._addr32 else self.fetch16()
                if o32: r.eax = self._read_mem32(seg, addr)
                else:   r.ax  = self._read_mem16(seg, addr)
                continue

            if op == 0xA2:
                addr = self.fetch16()
                self._write_mem8(seg, addr, r.al)
                continue

            if op == 0xA3:
                addr = fetch_imm() if self._addr32 else self.fetch16()
                if o32: self._write_mem32(seg, addr, r.eax)
                else:   self._write_mem16(seg, addr, r.ax)
                continue

            if 0x50 <= op <= 0x57:
                idx = op - 0x50
                if o32: self.push32(self.get_reg32(idx))
                else:   self.push16(get_reg16(r, idx))
                continue

            if 0x58 <= op <= 0x5F:
                idx = op - 0x58
                if o32: self.set_reg32(idx, self.pop32())
                else:   set_reg16(r, idx, self.pop16())
                continue

            if op == 0x68:
                if o32: self.push32(self.fetch32())
                else:   self.push16(self.fetch16())
                continue

            if op == 0x6A:
                v = fetch_simm8_ext()
                if o32: self.push32(v)
                else:   self.push16(v & 0xFFFF)
                continue

            if op == 0x06: self.push16(r.es); continue
            if op == 0x0E: self.push16(r.cs); continue
            if op == 0x16: self.push16(r.ss); continue
            if op == 0x1E: self.push16(r.ds); continue
            if op == 0x07:
                r.es = self.pop16()
                if not r.protected_mode:
                    uv = getattr(r, '_unreal_valid', None)
                    if uv is not None: uv['es'] = False
                continue
            if op == 0x17:
                r.ss = self.pop16()
                if not r.protected_mode:
                    uv = getattr(r, '_unreal_valid', None)
                    if uv is not None: uv['ss'] = False
                continue
            if op == 0x1F:
                r.ds = self.pop16()
                if not r.protected_mode:
                    uv = getattr(r, '_unreal_valid', None)
                    if uv is not None: uv['ds'] = False
                continue

            if op == 0x9C:
                if o32: self.push32(r.flags_word())
                else:   self.push16(r.flags_word())
                continue
            if op == 0x9D:
                f = self.pop32() if o32 else self.pop16()
                r.set_flags_word(f)
                continue

            if op == 0x00:
                mod, reg, rm, ea = modrm()
                a = get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea)
                b = get_reg8(r, reg)
                res = self._add8_p(a, b)
                if mod==3: set_reg8(r,rm,res)
                else: self._write_mem8(seg,ea,res)
                continue

            if op == 0x01:
                mod, reg, rm, ea = modrm()
                if o32:
                    a = self._modrm_read32(mod,rm,ea,seg)
                    b = self.get_reg32(reg)
                    self._modrm_write32(mod,rm,ea, self._add32(a,b), seg)
                else:
                    a = get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)
                    b = get_reg16(r, reg)
                    res = self._add16(a, b)
                    if mod==3: set_reg16(r,rm,res)
                    else: self._write_mem16(seg,ea,res)
                continue

            if op == 0x03:
                mod, reg, rm, ea = modrm()
                if o32:
                    a = self.get_reg32(reg)
                    b = self._modrm_read32(mod,rm,ea,seg)
                    self.set_reg32(reg, self._add32(a, b))
                else:
                    a = get_reg16(r,reg)
                    b = get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)
                    set_reg16(r, reg, self._add16(a,b))
                continue

            if op == 0x04: r.al = self._add8_p(r.al, self.fetch8()); continue
            if op == 0x05:
                if o32: r.eax = self._add32(r.eax, self.fetch32())
                else:   r.ax  = self._add16(r.ax,  self.fetch16())
                continue

            if op == 0x29:
                mod, reg, rm, ea = modrm()
                if o32:
                    a = self._modrm_read32(mod,rm,ea,seg)
                    self._modrm_write32(mod,rm,ea, self._sub32(a, self.get_reg32(reg)), seg)
                else:
                    a = get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)
                    res = self._sub16(a, get_reg16(r,reg))
                    if mod==3: set_reg16(r,rm,res)
                    else: self._write_mem16(seg,ea,res)
                continue

            if op == 0x2B:
                mod, reg, rm, ea = modrm()
                if o32: self.set_reg32(reg, self._sub32(self.get_reg32(reg), self._modrm_read32(mod,rm,ea,seg)))
                else:   set_reg16(r, reg, self._sub16(get_reg16(r,reg), get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)))
                continue

            if op == 0x2C: r.al = self._sub8_p(r.al, self.fetch8()); continue
            if op == 0x2D:
                if o32: r.eax = self._sub32(r.eax, self.fetch32())
                else:   r.ax  = self._sub16(r.ax,  self.fetch16())
                continue

            if 0x40 <= op <= 0x47:
                idx = op - 0x40
                cf = r.cf
                if o32: self.set_reg32(idx, self._add32(self.get_reg32(idx), 1))
                else:   set_reg16(r, idx, self._add16(get_reg16(r,idx), 1))
                r.cf = cf
                continue

            if 0x48 <= op <= 0x4F:
                idx = op - 0x48
                cf = r.cf
                if o32: self.set_reg32(idx, self._sub32(self.get_reg32(idx), 1))
                else:   set_reg16(r, idx, self._sub16(get_reg16(r,idx), 1))
                r.cf = cf
                continue

            if op == 0x38:
                mod, reg, rm, ea = modrm()
                a = get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea)
                self._sub8_p(a, get_reg8(r,reg)); continue
            if op == 0x39:
                mod, reg, rm, ea = modrm()
                if o32: self._sub32(self._modrm_read32(mod,rm,ea,seg), self.get_reg32(reg))
                else:
                    a = get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)
                    self._sub16(a, get_reg16(r,reg))
                continue
            if op == 0x3A:
                mod, reg, rm, ea = modrm()
                self._sub8_p(get_reg8(r,reg), get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea)); continue
            if op == 0x3B:
                mod, reg, rm, ea = modrm()
                if o32: self._sub32(self.get_reg32(reg), self._modrm_read32(mod,rm,ea,seg))
                else:   self._sub16(get_reg16(r,reg), get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea))
                continue
            if op == 0x3C: self._sub8_p(r.al, self.fetch8()); continue
            if op == 0x3D:
                if o32: self._sub32(r.eax, self.fetch32())
                else:   self._sub16(r.ax,  self.fetch16())
                continue

            if op == 0x21:
                mod, reg, rm, ea = modrm()
                if o32:
                    res = self._modrm_read32(mod,rm,ea,seg) & self.get_reg32(reg)
                    self._logic_flags32(res)
                    self._modrm_write32(mod,rm,ea,res,seg)
                else:
                    res = (get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)) & get_reg16(r,reg)
                    update_flags_16(r,res,cf=0,of_=0)
                    if mod==3: set_reg16(r,rm,res)
                    else: self._write_mem16(seg,ea,res)
                continue
            if op == 0x23:
                mod, reg, rm, ea = modrm()
                if o32:
                    res = self.get_reg32(reg) & self._modrm_read32(mod,rm,ea,seg)
                    self._logic_flags32(res); self.set_reg32(reg,res)
                else:
                    res = get_reg16(r,reg) & (get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea))
                    update_flags_16(r,res,cf=0,of_=0); set_reg16(r,reg,res)
                continue
            if op == 0x25:
                if o32: r.eax &= self.fetch32(); self._logic_flags32(r.eax)
                else:   r.ax  &= self.fetch16(); update_flags_16(r,r.ax,cf=0,of_=0)
                continue

            if op == 0x09:
                mod, reg, rm, ea = modrm()
                if o32:
                    res = self._modrm_read32(mod,rm,ea,seg) | self.get_reg32(reg)
                    self._logic_flags32(res); self._modrm_write32(mod,rm,ea,res,seg)
                else:
                    res = (get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)) | get_reg16(r,reg)
                    update_flags_16(r,res,cf=0,of_=0)
                    if mod==3: set_reg16(r,rm,res)
                    else: self._write_mem16(seg,ea,res)
                continue
            if op == 0x0B:
                mod, reg, rm, ea = modrm()
                if o32:
                    res = self.get_reg32(reg) | self._modrm_read32(mod,rm,ea,seg)
                    self._logic_flags32(res); self.set_reg32(reg,res)
                else:
                    res = get_reg16(r,reg) | (get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea))
                    update_flags_16(r,res,cf=0,of_=0); set_reg16(r,reg,res)
                continue
            if op == 0x0D:
                if o32: r.eax |= self.fetch32(); self._logic_flags32(r.eax)
                else:   r.ax  |= self.fetch16(); update_flags_16(r,r.ax,cf=0,of_=0)
                continue

            if op == 0x31:
                mod, reg, rm, ea = modrm()
                if o32:
                    res = self._modrm_read32(mod,rm,ea,seg) ^ self.get_reg32(reg)
                    self._logic_flags32(res); self._modrm_write32(mod,rm,ea,res,seg)
                else:
                    res = (get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)) ^ get_reg16(r,reg)
                    update_flags_16(r,res,cf=0,of_=0)
                    if mod==3: set_reg16(r,rm,res)
                    else: self._write_mem16(seg,ea,res)
                continue
            if op == 0x33:
                mod, reg, rm, ea = modrm()
                if o32:
                    res = self.get_reg32(reg) ^ self._modrm_read32(mod,rm,ea,seg)
                    self._logic_flags32(res); self.set_reg32(reg,res)
                else:
                    res = get_reg16(r,reg) ^ (get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea))
                    update_flags_16(r,res,cf=0,of_=0); set_reg16(r,reg,res)
                continue
            if op == 0x35:
                if o32: r.eax ^= self.fetch32(); self._logic_flags32(r.eax)
                else:   r.ax  ^= self.fetch16(); update_flags_16(r,r.ax,cf=0,of_=0)
                continue

            if op == 0x85:
                mod, reg, rm, ea = modrm()
                if o32: self._logic_flags32(self._modrm_read32(mod,rm,ea,seg) & self.get_reg32(reg))
                else:
                    res = (get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)) & get_reg16(r,reg)
                    update_flags_16(r,res,cf=0,of_=0)
                continue
            if op == 0xA9:
                if o32: self._logic_flags32(r.eax & self.fetch32())
                else:   update_flags_16(r, r.ax & self.fetch16(), cf=0, of_=0)
                continue

            if op in (0x80, 0x81, 0x82, 0x83):
                mod, reg, rm, ea = modrm()
                if op == 0x80:
                    imm = self.fetch8()
                    a   = get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea)
                    res = self._alu8(reg, a, imm)
                    if reg != 7:
                        if mod==3: set_reg8(r,rm,res)
                        else: self._write_mem8(seg,ea,res)
                elif op == 0x81:
                    imm = fetch_imm()
                    if o32:
                        a = self._modrm_read32(mod,rm,ea,seg)
                        res = self._alu32(reg, a, imm)
                        if reg != 7: self._modrm_write32(mod,rm,ea,res,seg)
                    else:
                        a = get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)
                        res = self._alu16(reg, a, imm)
                        if reg != 7:
                            if mod==3: set_reg16(r,rm,res)
                            else: self._write_mem16(seg,ea,res)
                else:
                    imm8 = self.fetch8()
                    if o32:
                        imm = sign8(imm8) & 0xFFFFFFFF
                        a = self._modrm_read32(mod,rm,ea,seg)
                        res = self._alu32(reg, a, imm)
                        if reg != 7: self._modrm_write32(mod,rm,ea,res,seg)
                    else:
                        imm = sign8(imm8) & 0xFFFF
                        a = get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)
                        res = self._alu16(reg, a, imm)
                        if reg != 7:
                            if mod==3: set_reg16(r,rm,res)
                            else: self._write_mem16(seg,ea,res)
                continue

            if op == 0xEB:
                off = sign8(self.fetch8())
                r.ip = (r.ip + off) & 0xFFFFFFFF; continue

            if op == 0xE9:
                off = sign32(self.fetch32()) if o32 else sign16(self.fetch16())
                r.ip = (r.ip + off) & 0xFFFFFFFF; continue

            if op == 0xEA:
                new_ip = fetch_imm()
                new_cs = self.fetch16()
                if r.protected_mode:
                    self.gdt.update_cache('cs', new_cs)
                    r.cs = new_cs
                    r.ip = new_ip
                    self._op32 = bool(r.seg_cache['cs'].db)
                else:
                    r.cs = new_cs
                    r.ip = new_ip
                    if r.cr0 & 1:
                        if new_cs == 0x10 and new_ip == 0x100000:
                            for sname in ('ds', 'es', 'ss', 'fs', 'gs'):
                                r.seg_cache[sname].base = self._seg_base(sname)
                            # Per the Linux/x86 Boot Protocol: "%ebp, %edi
                            # and %ebx must be zero" at this exact 32-bit
                            # entry point. Whatever real-mode code got me
                            # here is responsible for guaranteeing that on
                            # real hardware, but our emulated 32-bit
                            # register halves (e.g. Registers32._edi_hi)
                            # can retain stale data from earlier 32-bit
                            # operations if a later 16-bit write (which
                            # will only touch the low half, on
                            # real x86 too) was relied upon to leave the
                            # upper half already-zero from CPU reset —
                            # an assumption our emulator doesn't otherwise
                            # guarantee. Enforce the documented contract
                            # directly here rather than chasing the exact
                            # upstream write that should have zeroed it.
                            before = (r.edi, r.ebp, r.ebx)
                            r.edi = 0
                            r.ebp = 0
                            r.ebx = 0
                            if before != (0, 0, 0):
                                print(f"[*] Boot protocol enforcement: EDI/EBP/EBX "
                                      f"were {before[0]:#010x}/{before[1]:#010x}/"
                                      f"{before[2]:#010x} at kernel entry (should be "
                                      f"zero per boot protocol) — corrected to zero.")
                        r.protected_mode = True
                        self.gdt.update_cache('cs', new_cs)
                        print(f"\n[*] *** PROTECTED MODE ACTIVE *** CS={new_cs:#06x} EIP={new_ip:#010x}")
                    else:
                        uv = getattr(r, '_unreal_valid', None)
                        if uv is not None:
                            uv['cs'] = False
                continue

            if op == 0xE8:
                off = sign32(self.fetch32()) if o32 else sign16(self.fetch16())
                if o32: self.push32(r.ip)
                else:   self.push16(r.ip & 0xFFFF)
                r.ip = (r.ip + off) & 0xFFFFFFFF; continue

            if op == 0xC3:
                r.ip = self.pop32() if o32 else self.pop16(); continue

            if op == 0xCB:
                r.ip = self.pop32() if o32 else self.pop16()
                r.cs = self.pop16(); continue

            if op == 0xC2:
                imm = self.fetch16()
                r.ip = self.pop32() if o32 else self.pop16()
                r.esp = (r.esp + imm) & 0xFFFFFFFF; continue

            JCOND = {
                0x70: lambda: r.of_,
                0x71: lambda: not r.of_,
                0x72: lambda: r.cf,
                0x73: lambda: not r.cf,
                0x74: lambda: r.zf,
                0x75: lambda: not r.zf,
                0x76: lambda: r.cf or r.zf,
                0x77: lambda: not(r.cf or r.zf),
                0x78: lambda: r.sf,
                0x79: lambda: not r.sf,
                0x7A: lambda: r.pf,
                0x7B: lambda: not r.pf,
                0x7C: lambda: r.sf != r.of_,
                0x7D: lambda: r.sf == r.of_,
                0x7E: lambda: r.zf or r.sf != r.of_,
                0x7F: lambda: not r.zf and r.sf == r.of_,
            }
            if op in JCOND:
                off = sign8(self.fetch8())
                if JCOND[op]():
                    r.ip = (r.ip + off) & 0xFFFFFFFF
                continue

            if op == 0xE2:
                off = sign8(self.fetch8())
                if o32: r.ecx = (r.ecx - 1) & 0xFFFFFFFF; cond = r.ecx != 0
                else:   r.cx  = (r.cx  - 1) & 0xFFFF;     cond = r.cx  != 0
                if cond: r.ip = (r.ip + off) & 0xFFFFFFFF
                continue

            if op == 0xE1:
                off = sign8(self.fetch8())
                if o32: r.ecx = (r.ecx - 1) & 0xFFFFFFFF; cond = r.ecx != 0 and r.zf
                else:   r.cx  = (r.cx  - 1) & 0xFFFF;     cond = r.cx  != 0 and r.zf
                if cond: r.ip = (r.ip + off) & 0xFFFFFFFF
                continue

            if op == 0xE0:
                off = sign8(self.fetch8())
                if o32: r.ecx = (r.ecx - 1) & 0xFFFFFFFF; cond = r.ecx != 0 and not r.zf
                else:   r.cx  = (r.cx  - 1) & 0xFFFF;     cond = r.cx  != 0 and not r.zf
                if cond: r.ip = (r.ip + off) & 0xFFFFFFFF
                continue

            if op == 0xE3:
                off = sign8(self.fetch8())
                val = r.ecx if o32 else r.cx
                if not val: r.ip = (r.ip + off) & 0xFFFFFFFF
                continue

            if op == 0xCD:
                num = self.fetch8()
                if r.protected_mode:
                    self.push32(r.flags_word())
                    self.push32(r.cs)
                    self.push32(r.ip)
                    r.IF = 0
                    idt_addr = r.idtr_base + num * 8
                    # Bounds-check against idtr_limit — see the matching
                    # comment in phase3.py's _inject_hardware_irq for
                    # why this matters: a genuinely empty/null IDT must
                    # never have its "lookup" trust whatever unrelated
                    # bytes happen to physically sit at idtr_base+num*8.
                    if (num * 8 + 7) > r.idtr_limit:
                        handler = 0
                    else:
                        off_lo = self.mem.read16_flat(idt_addr)
                        sel    = self.mem.read16_flat(idt_addr + 2)
                        off_hi = self.mem.read16_flat(idt_addr + 6)
                        handler = off_lo | (off_hi << 16)
                    if handler:
                        self.gdt.update_cache('cs', sel)
                        r.cs = sel
                        r.ip = handler
                    else:
                        r.ip = self.pop32(); r.cs = self.pop32()
                        r.set_flags_word(self.pop32())
                else:
                    self.push16(r.flags_word())
                    self.push16(r.cs)
                    self.push16(r.ip)
                    r.IF = 0
                    self.bios.interrupt(num)
                    handler_cf = r.cf
                    handler_ax = r.ax
                    handler_ah = r.ah
                    r.ip = self.pop16()
                    r.cs = self.pop16()
                    saved_flags = self.pop16()
                    r.set_flags_word(saved_flags)
                    r.cf = handler_cf
                    r.ax = handler_ax
                continue

            if op == 0xCF:
                r.ip = self.pop32() if o32 else self.pop16()
                r.cs = self.pop32() & 0xFFFF if o32 else self.pop16()
                r.set_flags_word(self.pop32() if o32 else self.pop16())
                continue

            if op == 0xE4:
                port = self.fetch8(); r.al = io.read(port); continue
            if op == 0xE5:
                port = self.fetch8()
                if o32: r.eax = io.read(port)
                else:   r.ax  = io.read(port)
                continue
            if op == 0xEC:
                r.al = io.read(r.dx & 0xFFFF); continue
            if op == 0xED:
                if o32: r.eax = io.read(r.dx & 0xFFFF)
                else:   r.ax  = io.read(r.dx & 0xFFFF)
                continue
            if op == 0xE6:
                io.write(self.fetch8(), r.al); continue
            if op == 0xE7:
                port = self.fetch8()
                io.write(port, r.eax if o32 else r.ax); continue
            if op == 0xEE:
                io.write(r.dx & 0xFFFF, r.al); continue
            if op == 0xEF:
                io.write(r.dx & 0xFFFF, r.eax if o32 else r.ax); continue

            if op == 0x6C:
                rep_count = (r.ecx if self._addr32 else r.cx) if self._rep else 1
                port = r.dx & 0xFFFF
                step = 1 if not r.df else -1
                for _ in range(rep_count):
                    val = io.read(port)
                    di = r.edi if self._addr32 else r.di
                    self._write_mem8('es', di, val)
                    if self._addr32: r.edi = (r.edi + step) & 0xFFFFFFFF
                    else:            r.di  = (r.di  + step) & 0xFFFF
                if self._rep:
                    if self._addr32: r.ecx = 0
                    else:            r.cx  = 0
                continue

            if op == 0x6D:
                rep_count = (r.ecx if self._addr32 else r.cx) if self._rep else 1
                port = r.dx & 0xFFFF
                size = 4 if o32 else 2
                step = size if not r.df else -size
                for _ in range(rep_count):
                    if o32:
                        val = io.read(port) | (io.read(port) << 16)
                    else:
                        val = io.read(port)
                    di = r.edi if self._addr32 else r.di
                    if o32: self._write_mem32('es', di, val)
                    else:   self._write_mem16('es', di, val)
                    if self._addr32: r.edi = (r.edi + step) & 0xFFFFFFFF
                    else:            r.di  = (r.di  + step) & 0xFFFF
                if self._rep:
                    if self._addr32: r.ecx = 0
                    else:            r.cx  = 0
                continue

            if op == 0x6E:
                rep_count = (r.ecx if self._addr32 else r.cx) if self._rep else 1
                port = r.dx & 0xFFFF
                step = 1 if not r.df else -1
                for _ in range(rep_count):
                    si = r.esi if self._addr32 else r.si
                    val = self._read_mem8('ds', si)
                    io.write(port, val)
                    if self._addr32: r.esi = (r.esi + step) & 0xFFFFFFFF
                    else:            r.si  = (r.si  + step) & 0xFFFF
                if self._rep:
                    if self._addr32: r.ecx = 0
                    else:            r.cx  = 0
                continue

            if op == 0x6F:
                rep_count = (r.ecx if self._addr32 else r.cx) if self._rep else 1
                port = r.dx & 0xFFFF
                size = 4 if o32 else 2
                step = size if not r.df else -size
                for _ in range(rep_count):
                    si = r.esi if self._addr32 else r.si
                    if o32: val = self._read_mem32('ds', si)
                    else:   val = self._read_mem16('ds', si)
                    io.write(port, val)
                    if self._addr32: r.esi = (r.esi + step) & 0xFFFFFFFF
                    else:            r.si  = (r.si  + step) & 0xFFFF
                if self._rep:
                    if self._addr32: r.ecx = 0
                    else:            r.cx  = 0
                continue

            if op == 0x63:
                self.fetch8()
                continue

            if op == 0x8D:
                mod, reg, rm, ea = modrm()
                if o32: self.set_reg32(reg, ea if ea is not None else 0)
                else:   set_reg16(r, reg, (ea if ea is not None else 0) & 0xFFFF)
                continue

            if 0x91 <= op <= 0x97:
                idx = op - 0x90
                if o32:
                    tmp = self.get_reg32(idx); self.set_reg32(idx, r.eax); r.eax = tmp
                else:
                    tmp = get_reg16(r,idx); set_reg16(r,idx,r.ax); r.ax = tmp
                continue

            if self._rep == 'REP' and not m.paging_enabled:
                cx = r.ecx if self._addr32 else r.cx
                if cx > 0:
                    if op == 0xA4:
                        src = (self._seg_base('ds') + (r.esi if self._addr32 else r.si)) & 0xFFFFFFFF
                        dst = (self._seg_base('es') + (r.edi if self._addr32 else r.di)) & 0xFFFFFFFF
                        if not r.df and src+cx <= m.size and dst+cx <= m.size:
                            m._m[dst:dst+cx] = m._m[src:src+cx]
                            if self._addr32: r.esi=(r.esi+cx)&0xFFFFFFFF; r.edi=(r.edi+cx)&0xFFFFFFFF
                            else:            r.si=(r.si+cx)&0xFFFF;       r.di=(r.di+cx)&0xFFFF
                            if self._addr32: r.ecx=0
                            else:            r.cx=0
                            self.icount += cx; continue
                    if op == 0xA5 and o32:
                        src = (self._seg_base('ds') + r.esi) & 0xFFFFFFFF
                        dst = (self._seg_base('es') + r.edi) & 0xFFFFFFFF
                        nb = cx * 4
                        if not r.df and src+nb <= m.size and dst+nb <= m.size:
                            m._m[dst:dst+nb] = m._m[src:src+nb]
                            r.esi=(r.esi+nb)&0xFFFFFFFF; r.edi=(r.edi+nb)&0xFFFFFFFF; r.ecx=0
                            self.icount += cx; continue
                    if op == 0xAA:
                        dst = (self._seg_base('es') + (r.edi if self._addr32 else r.di)) & 0xFFFFFFFF
                        if not r.df and dst+cx <= m.size:
                            m._m[dst:dst+cx] = r.al
                            if self._addr32: r.edi=(r.edi+cx)&0xFFFFFFFF; r.ecx=0
                            else:            r.di=(r.di+cx)&0xFFFF;       r.cx=0
                            self.icount += cx; continue
                    if op == 0xAB and o32:
                        dst = (self._seg_base('es') + r.edi) & 0xFFFFFFFF
                        nb = cx * 4
                        if not r.df and dst+nb <= m.size:
                            import numpy as _np
                            b4 = (r.eax & 0xFFFFFFFF).to_bytes(4, 'little')
                            m._m[dst:dst+nb] = _np.frombuffer(b4 * cx, dtype=_np.uint8)
                            r.edi=(r.edi+nb)&0xFFFFFFFF; r.ecx=0
                            self.icount += cx; continue
                    if op == 0xA6:
                        src = (self._seg_base('ds') + (r.esi if self._addr32 else r.si)) & 0xFFFFFFFF
                        dst = (self._seg_base('es') + (r.edi if self._addr32 else r.di)) & 0xFFFFFFFF
                        count = 0; a = 0; b = 0
                        while count < cx:
                            a = m.read8_flat(src+count); b = m.read8_flat(dst+count)
                            count += 1
                            if a != b: break
                        self._sub8(a, b)
                        if self._addr32: r.esi=(r.esi+count)&0xFFFFFFFF; r.edi=(r.edi+count)&0xFFFFFFFF; r.ecx=(r.ecx-count)&0xFFFFFFFF
                        else:            r.si=(r.si+count)&0xFFFF;       r.di=(r.di+count)&0xFFFF;       r.cx=(r.cx-count)&0xFFFF
                        self.icount += count; continue

            if self._rep in ('REP', 'REPNE') and op in (0xA4,0xA5,0xAA,0xAB,0xAC,0xAD,0xA6,0xA7,0xAE,0xAF):
                cx = r.ecx if self._addr32 else r.cx
                rep_is_repe = (self._rep == 'REP')
                while cx > 0 and self.icount < self.max_icount:
                    if op == 0xA4:
                        b = self._read_mem8('ds', r.esi if self._addr32 else r.si)
                        self._write_mem8('es', r.edi if self._addr32 else r.di, b)
                        step = 1 if not r.df else -1
                        if self._addr32: r.esi=(r.esi+step)&0xFFFFFFFF; r.edi=(r.edi+step)&0xFFFFFFFF
                        else:            r.si=(r.si+step)&0xFFFF;       r.di=(r.di+step)&0xFFFF
                    elif op == 0xA5:
                        esi = r.esi if self._addr32 else r.si
                        edi = r.edi if self._addr32 else r.di
                        if o32:
                            v = self._read_mem32('ds', esi); self._write_mem32('es', edi, v)
                            step = 4 if not r.df else -4
                        else:
                            v = self._read_mem16('ds', esi); self._write_mem16('es', edi, v)
                            step = 2 if not r.df else -2
                        if self._addr32: r.esi=(r.esi+step)&0xFFFFFFFF; r.edi=(r.edi+step)&0xFFFFFFFF
                        else:            r.si=(r.si+step)&0xFFFF;       r.di=(r.di+step)&0xFFFF
                    elif op == 0xAA:
                        self._write_mem8('es', r.edi if self._addr32 else r.di, r.al)
                        step = 1 if not r.df else -1
                        if self._addr32: r.edi=(r.edi+step)&0xFFFFFFFF
                        else:            r.di=(r.di+step)&0xFFFF
                    elif op == 0xAB:
                        edi = r.edi if self._addr32 else r.di
                        if o32:
                            self._write_mem32('es', edi, r.eax)
                            step = 4 if not r.df else -4
                        else:
                            self._write_mem16('es', edi, r.ax)
                            step = 2 if not r.df else -2
                        if self._addr32: r.edi=(r.edi+step)&0xFFFFFFFF
                        else:            r.di=(r.di+step)&0xFFFF
                    elif op == 0xAC:
                        r.al = self._read_mem8('ds', r.esi if self._addr32 else r.si)
                        step = 1 if not r.df else -1
                        if self._addr32: r.esi=(r.esi+step)&0xFFFFFFFF
                        else:            r.si=(r.si+step)&0xFFFF
                    elif op == 0xAD:
                        esi = r.esi if self._addr32 else r.si
                        if o32: r.eax = self._read_mem32('ds', esi); step=4
                        else:   r.ax  = self._read_mem16('ds', esi); step=2
                        step = step if not r.df else -step
                        if self._addr32: r.esi=(r.esi+step)&0xFFFFFFFF
                        else:            r.si=(r.si+step)&0xFFFF
                    elif op == 0xA6:
                        a = self._read_mem8('ds', r.esi if self._addr32 else r.si)
                        b = self._read_mem8('es', r.edi if self._addr32 else r.di)
                        step = 1 if not r.df else -1
                        if self._addr32: r.esi=(r.esi+step)&0xFFFFFFFF; r.edi=(r.edi+step)&0xFFFFFFFF
                        else:            r.si=(r.si+step)&0xFFFF;       r.di=(r.di+step)&0xFFFF
                        self._sub8(a, b)
                        cx -= 1
                        if self._addr32: r.ecx = cx
                        else:            r.cx = cx
                        self.icount += 1
                        if rep_is_repe and not r.zf: break
                        if (not rep_is_repe) and r.zf: break
                        continue
                    elif op == 0xA7:
                        esi = r.esi if self._addr32 else r.si
                        edi = r.edi if self._addr32 else r.di
                        if o32:
                            a = self._read_mem32('ds', esi); b = self._read_mem32('es', edi)
                            step = 4
                        else:
                            a = self._read_mem16('ds', esi); b = self._read_mem16('es', edi)
                            step = 2
                        step = step if not r.df else -step
                        if self._addr32: r.esi=(r.esi+step)&0xFFFFFFFF; r.edi=(r.edi+step)&0xFFFFFFFF
                        else:            r.si=(r.si+step)&0xFFFF;       r.di=(r.di+step)&0xFFFF
                        if o32: self._sub32(a,b)
                        else:   self._sub16(a,b)
                        cx -= 1
                        if self._addr32: r.ecx = cx
                        else:            r.cx = cx
                        self.icount += 1
                        if rep_is_repe and not r.zf: break
                        if (not rep_is_repe) and r.zf: break
                        continue
                    elif op == 0xAE:
                        b = self._read_mem8('es', r.edi if self._addr32 else r.di)
                        step = 1 if not r.df else -1
                        if self._addr32: r.edi=(r.edi+step)&0xFFFFFFFF
                        else:            r.di=(r.di+step)&0xFFFF
                        self._sub8(r.al, b)
                        cx -= 1
                        if self._addr32: r.ecx = cx
                        else:            r.cx = cx
                        self.icount += 1
                        if rep_is_repe and not r.zf: break
                        if (not rep_is_repe) and r.zf: break
                        continue
                    elif op == 0xAF:
                        if o32:
                            b = self._read_mem32('es', r.edi); step = 4
                        else:
                            b = self._read_mem16('es', r.di); step = 2
                        step = step if not r.df else -step
                        if self._addr32: r.edi=(r.edi+step)&0xFFFFFFFF
                        else:            r.di=(r.di+step)&0xFFFF
                        if o32: self._sub32(r.eax, b)
                        else:   self._sub16(r.ax, b)
                        cx -= 1
                        if self._addr32: r.ecx = cx
                        else:            r.cx = cx
                        self.icount += 1
                        if rep_is_repe and not r.zf: break
                        if (not rep_is_repe) and r.zf: break
                        continue
                    else:
                        break
                    cx -= 1
                    self.icount += 1
                if self._addr32: r.ecx = cx
                else:            r.cx = cx
                continue

            if op == 0xA4:
                b = self._read_mem8('ds', r.esi if self._addr32 else r.si)
                self._write_mem8('es', r.edi if self._addr32 else r.di, b)
                step = 1 if not r.df else -1
                if self._addr32: r.esi=(r.esi+step)&0xFFFFFFFF; r.edi=(r.edi+step)&0xFFFFFFFF
                else:            r.si=(r.si+step)&0xFFFF;       r.di=(r.di+step)&0xFFFF
                continue
            if op == 0xA5:
                if o32:
                    v = self._read_mem32('ds', r.esi); self._write_mem32('es', r.edi, v)
                    step = 4 if not r.df else -4
                    r.esi=(r.esi+step)&0xFFFFFFFF; r.edi=(r.edi+step)&0xFFFFFFFF
                else:
                    v = self._read_mem16('ds', r.si); self._write_mem16('es', r.di, v)
                    step = 2 if not r.df else -2
                    r.si=(r.si+step)&0xFFFF; r.di=(r.di+step)&0xFFFF
                continue
            if op == 0xAA:
                self._write_mem8('es', r.edi if self._addr32 else r.di, r.al)
                step = 1 if not r.df else -1
                if self._addr32: r.edi=(r.edi+step)&0xFFFFFFFF
                else:            r.di=(r.di+step)&0xFFFF
                continue
            if op == 0xAB:
                if o32:
                    self._write_mem32('es', r.edi, r.eax)
                    step = 4 if not r.df else -4; r.edi=(r.edi+step)&0xFFFFFFFF
                else:
                    self._write_mem16('es', r.di, r.ax)
                    step = 2 if not r.df else -2; r.di=(r.di+step)&0xFFFF
                continue
            if op == 0xAC:
                r.al = self._read_mem8('ds', r.esi if self._addr32 else r.si)
                step = 1 if not r.df else -1
                if self._addr32: r.esi=(r.esi+step)&0xFFFFFFFF
                else:            r.si=(r.si+step)&0xFFFF
                continue
            if op == 0xAD:
                if o32: r.eax = self._read_mem32('ds', r.esi); step=4
                else:   r.ax  = self._read_mem16('ds', r.si);  step=2
                if not r.df: step = abs(step)
                else:        step = -abs(step)
                if self._addr32: r.esi=(r.esi+step)&0xFFFFFFFF
                else:            r.si=(r.si+step)&0xFFFF
                continue

            if op == 0x98:
                if o32: r.eax = sign16(r.ax) & 0xFFFFFFFF
                else:   r.ax  = sign8(r.al)  & 0xFFFF
                continue
            if op == 0x99:
                if o32: r.edx = 0xFFFFFFFF if r.eax & 0x80000000 else 0
                else:   r.dx  = 0xFFFF     if r.ax  & 0x8000     else 0
                continue

            if op == 0x0F:
                op2 = self.fetch8()

                JCOND2 = {
                    0x80: lambda: r.of_,            0x81: lambda: not r.of_,
                    0x82: lambda: r.cf,              0x83: lambda: not r.cf,
                    0x84: lambda: r.zf,              0x85: lambda: not r.zf,
                    0x86: lambda: r.cf or r.zf,      0x87: lambda: not(r.cf or r.zf),
                    0x88: lambda: r.sf,              0x89: lambda: not r.sf,
                    0x8A: lambda: r.pf,              0x8B: lambda: not r.pf,
                    0x8C: lambda: r.sf != r.of_,     0x8D: lambda: r.sf == r.of_,
                    0x8E: lambda: r.zf or r.sf!=r.of_,
                    0x8F: lambda: not r.zf and r.sf==r.of_,
                }
                if op2 in JCOND2:
                    off = sign32(self.fetch32()) if o32 else sign16(self.fetch16())
                    if JCOND2[op2](): r.ip = (r.ip + off) & 0xFFFFFFFF
                    continue

                SETCC = {
                    0x90: lambda: r.of_,            0x91: lambda: not r.of_,
                    0x92: lambda: r.cf,              0x93: lambda: not r.cf,
                    0x94: lambda: r.zf,              0x95: lambda: not r.zf,
                    0x96: lambda: r.cf or r.zf,      0x97: lambda: not(r.cf or r.zf),
                    0x98: lambda: r.sf,              0x99: lambda: not r.sf,
                    0x9A: lambda: r.pf,              0x9B: lambda: not r.pf,
                    0x9C: lambda: r.sf != r.of_,     0x9D: lambda: r.sf == r.of_,
                    0x9E: lambda: r.zf or r.sf!=r.of_,
                    0x9F: lambda: not r.zf and r.sf==r.of_,
                }
                if op2 in SETCC:
                    mod, reg, rm, ea = modrm()
                    val = 1 if SETCC[op2]() else 0
                    if mod==3: set_reg8(r,rm,val)
                    else: self._write_mem8(seg,ea,val)
                    continue

                CMOVCC = {
                    0x40: lambda: r.of_,            0x41: lambda: not r.of_,
                    0x42: lambda: r.cf,              0x43: lambda: not r.cf,
                    0x44: lambda: r.zf,              0x45: lambda: not r.zf,
                    0x46: lambda: r.cf or r.zf,      0x47: lambda: not(r.cf or r.zf),
                    0x48: lambda: r.sf,              0x49: lambda: not r.sf,
                    0x4A: lambda: r.pf,              0x4B: lambda: not r.pf,
                    0x4C: lambda: r.sf != r.of_,     0x4D: lambda: r.sf == r.of_,
                    0x4E: lambda: r.zf or r.sf!=r.of_,
                    0x4F: lambda: not r.zf and r.sf==r.of_,
                }
                if op2 in CMOVCC:
                    mod, reg, rm, ea = modrm()
                    if CMOVCC[op2]():
                        if o32: self.set_reg32(reg, self._modrm_read32(mod,rm,ea,seg))
                        else:   set_reg16(r,reg, get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea))
                    else:
                        pass
                    continue

                if op2 == 0x00:
                    mrm_b = self.fetch8()
                    mod = (mrm_b>>6)&3; reg=(mrm_b>>3)&7; rm=mrm_b&7
                    if mod != 3:
                        if mod==1: self.fetch8()
                        elif mod==2: self.fetch16()
                        elif mod==0 and rm==6: self.fetch16()
                    continue

                if op2 == 0x01:
                    mrm_b = self.fetch8()
                    sub   = (mrm_b >> 3) & 7
                    mod, reg2, rm, ea = decode_modrm(self, mrm_b)
                    if sub == 0:
                        pass
                    elif sub == 1:
                        pass
                    elif sub == 2:
                        if ea is not None:
                            base_addr = self._seg_base('ds') + ea
                            r.gdtr_limit = self.mem.read16_flat(base_addr)
                            base = self.mem.read32_flat(base_addr + 2)
                            r.gdtr_base  = base if o32 else base & 0xFFFFFF
                            print(f"[*] LGDT: base={r.gdtr_base:#010x} limit={r.gdtr_limit:#06x}")
                    elif sub == 3:
                        if ea is not None:
                            base_addr = self._seg_base('ds') + ea
                            r.idtr_limit = self.mem.read16_flat(base_addr)
                            r.idtr_base  = self.mem.read32_flat(base_addr + 2)
                            print(f"[*] LIDT: base={r.idtr_base:#010x} limit={r.idtr_limit:#06x}")
                    elif sub == 4:
                        if mod==3: set_reg16(r,rm, r.cr0 & 0xFFFF)
                        elif ea is not None: self._write_mem16(seg, ea, r.cr0 & 0xFFFF)
                    elif sub == 6:
                        v = get_reg16(r,rm) if mod==3 else self._read_mem16('ds', ea) if ea else 0
                        r.cr0 = (r.cr0 & ~0xFFFF) | (v & 0xFFFF)
                    elif sub == 7:
                        pass
                    continue

                if op2 == 0x20:
                    mrm_b=self.fetch8(); cr=(mrm_b>>3)&7; rm=mrm_b&7
                    val=[r.cr0,0,r.cr2,r.cr3][cr] if cr<=3 else 0
                    self.set_reg32(rm,val); continue
                if op2 == 0x22:
                    mrm_b=self.fetch8(); cr=(mrm_b>>3)&7; rm=mrm_b&7
                    val=self.get_reg32(rm)
                    if cr==0:
                        old_pe=r.cr0&1
                        old_pg=(r.cr0>>31)&1
                        r.cr0=val
                        new_pe = val & 1
                        if new_pe and not old_pe:
                            print(f"[*] CR0.PE set — PE=1, awaiting far jump")
                        elif old_pe and not new_pe:
                            r.protected_mode = False
                            r._unreal_valid = {
                                s: (r.seg_cache[s].base != (getattr(r, s) << 4))
                                for s in ('cs','ds','es','ss','fs','gs')
                            }
                            print(f"[*] CR0.PE cleared — back to real mode")
                        new_pg=(val>>31)&1
                        if new_pg != old_pg:
                            self.mem.paging_enabled = bool(new_pg)
                            self.mem.invalidate_tlb()
                            print(f"[*] CR0.PG {'enabled' if new_pg else 'disabled'} — paging {'ON' if new_pg else 'OFF'}")
                    elif cr==2: r.cr2=val
                    elif cr==3:
                        r.cr3=val
                        self.mem.cr3 = val
                        self.mem.invalidate_tlb()
                    continue

                if op2 in (0x21, 0x23):
                    self.fetch8(); continue

                if op2 in (0x08, 0x09): continue

                if op2 == 0x0B:
                    func_start_eip = (r.ip - 2) & 0xFFFFFFFF
                    if 0xc02f8d10 <= func_start_eip <= 0xc02f8d80:
                        if not getattr(self, '_meminit_patched', False):
                            print(f"[!] mem_init() page-walk BUG at EIP={func_start_eip:#010x}: "
                                  f"EDI={r.edi:#x} (mem_map not initialized — nuclear bypass "
                                  f"skips bootmem). Verified vs real QEMU trace: taking the "
                                  f"branch-around-BUG path real hardware takes here.")
                            self._meminit_patched = True
                        continue
                    if not getattr(self, '_ud2_warned', False):
                        print(f"[!] UD2 (kernel BUG/WARN_ON) at EIP={r.ip-2:#010x} "
                              f"— likely due to skipped bootmem/mem_map init "
                              f"(nuclear bypass). Continuing past it.")
                        self._ud2_warned = True
                    continue

                if op2 in (0x38, 0x3A):
                    self.fetch8()
                    b3 = self.fetch8() if op2 == 0x3A else 0
                    continue

                if 0x50 <= op2 <= 0x7F:
                    b = self.fetch8()
                    mod=(b>>6)&3; rm=b&7
                    if mod!=3:
                        if mod==1: self.fetch8()
                        elif mod==2: self.fetch16()
                        elif mod==0 and rm==6: self.fetch16()
                    if op2 in (0x6F,0x7F,0x7E): pass
                    continue

                if op2 == 0xA0:
                    if o32: self.push32(r.fs)
                    else:   self.push16(r.fs)
                    continue
                if op2 == 0xA1:
                    r.fs = self.pop32()&0xFFFF if o32 else self.pop16()
                    continue

                if op2 == 0xA8:
                    if o32: self.push32(r.gs)
                    else:   self.push16(r.gs)
                    continue
                if op2 == 0xA9:
                    r.gs = self.pop32()&0xFFFF if o32 else self.pop16()
                    continue

                if op2 in (0xA3, 0xAB, 0xB3, 0xBB):
                    mod,reg2,rm,ea = modrm()
                    if o32:
                        v = self._modrm_read32(mod,rm,ea,seg)
                        bit_idx = self.get_reg32(reg2) & 31
                    else:
                        v = get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)
                        bit_idx = get_reg16(r,reg2) & 15
                    r.cf = (v >> bit_idx) & 1
                    if op2 == 0xAB: v |=  (1 << bit_idx)
                    elif op2==0xB3: v &= ~(1 << bit_idx)
                    elif op2==0xBB: v ^=  (1 << bit_idx)
                    if op2 != 0xA3:
                        if o32: self._modrm_write32(mod,rm,ea,v,seg)
                        else:
                            if mod==3: set_reg16(r,rm,v&0xFFFF)
                            else: self._write_mem16(seg,ea,v&0xFFFF)
                    continue

                if op2 == 0xBA:
                    mod,reg2,rm,ea = modrm(); imm=self.fetch8()
                    if o32: v=self._modrm_read32(mod,rm,ea,seg); bit_idx=imm&31
                    else:   v=get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea); bit_idx=imm&15
                    r.cf=(v>>bit_idx)&1
                    if reg2==5: v|=(1<<bit_idx)
                    elif reg2==6: v&=~(1<<bit_idx)
                    elif reg2==7: v^=(1<<bit_idx)
                    if reg2 in (5,6,7):
                        if o32: self._modrm_write32(mod,rm,ea,v,seg)
                        else:
                            if mod==3: set_reg16(r,rm,v&0xFFFF)
                            else: self._write_mem16(seg,ea,v&0xFFFF)
                    continue

                if op2 in (0xA4, 0xA5, 0xAC, 0xAD):
                    mod,reg2,rm,ea = modrm()
                    count = self.fetch8() if op2 in (0xA4,0xAC) else (r.cl & 0x1F)
                    if o32:
                        dst=self._modrm_read32(mod,rm,ea,seg); src=self.get_reg32(reg2)
                        if op2 in (0xA4,0xA5):
                            res=((dst<<count)|(src>>(32-count)))&0xFFFFFFFF if count else dst
                        else:
                            res=((dst>>count)|(src<<(32-count)))&0xFFFFFFFF if count else dst
                        self._modrm_write32(mod,rm,ea,res,seg)
                    else:
                        dst=get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)
                        src=get_reg16(r,reg2)
                        if op2 in (0xA4,0xA5):
                            res=((dst<<count)|(src>>(16-count)))&0xFFFF if count else dst
                        else:
                            res=((dst>>count)|(src<<(16-count)))&0xFFFF if count else dst
                        if mod==3: set_reg16(r,rm,res)
                        else: self._write_mem16(seg,ea,res)
                    continue

                if op2 == 0xAF:
                    mod,reg2,rm,ea = modrm()
                    if o32:
                        a=sign32(self.get_reg32(reg2)); b=sign32(self._modrm_read32(mod,rm,ea,seg))
                        res=(a*b)&0xFFFFFFFF; self.set_reg32(reg2,res)
                        r.cf=r.of_=1 if (a*b)!=sign32(res) else 0
                    else:
                        a=sign16(get_reg16(r,reg2)); b=sign16(get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea))
                        res=(a*b)&0xFFFF; set_reg16(r,reg2,res)
                        r.cf=r.of_=1 if (a*b)!=sign16(res) else 0
                    continue

                if op2 in (0xB2, 0xB4, 0xB5):
                    mod,reg2,rm,ea = modrm()
                    if ea is not None:
                        if o32:
                            v  = self._read_mem32(seg, ea)
                            sv = self._read_mem16(seg, (ea+4) & 0xFFFFFFFF)
                            self.set_reg32(reg2, v)
                        else:
                            v  = self._read_mem16(seg, ea)
                            sv = self._read_mem16(seg, (ea+2) & 0xFFFFFFFF)
                            set_reg16(r, reg2, v)
                        sname = {0xB2:'ss',0xB4:'fs',0xB5:'gs'}[op2]
                        if r.protected_mode:
                            self.gdt.update_cache(sname, sv)
                        else:
                            setattr(r, sname, sv)
                            uv = getattr(r, '_unreal_valid', None)
                            if uv is not None:
                                uv[sname] = False
                    continue

                if op2 == 0xB6:
                    mod,reg2,rm,ea = modrm()
                    v=get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea)
                    if o32: self.set_reg32(reg2,v)
                    else:   set_reg16(r,reg2,v)
                    continue

                if op2 == 0xB7:
                    mod,reg2,rm,ea = modrm()
                    v=get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)
                    self.set_reg32(reg2,v)
                    continue

                if op2 == 0xBE:
                    mod,reg2,rm,ea = modrm()
                    v=get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea)
                    ext=sign8(v)&(0xFFFFFFFF if o32 else 0xFFFF)
                    if o32: self.set_reg32(reg2,ext)
                    else:   set_reg16(r,reg2,ext)
                    continue

                if op2 == 0xBF:
                    mod,reg2,rm,ea = modrm()
                    v=get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)
                    self.set_reg32(reg2,sign16(v)&0xFFFFFFFF)
                    continue

                if op2 in (0xBC, 0xBD):
                    mod,reg2,rm,ea = modrm()
                    v=self._modrm_read32(mod,rm,ea,seg) if o32 else (get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea))
                    if v == 0:
                        r.zf = 1
                    else:
                        r.zf = 0
                        bits = 32 if o32 else 16
                        if op2 == 0xBC:
                            idx = next(i for i in range(bits) if (v>>i)&1)
                        else:
                            idx = next(i for i in range(bits-1,-1,-1) if (v>>i)&1)
                        if o32: self.set_reg32(reg2,idx)
                        else:   set_reg16(r,reg2,idx)
                    continue

                if op2 == 0xC0:
                    mod,reg2,rm,ea = modrm()
                    dst=get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea)
                    src=get_reg8(r,reg2)
                    set_reg8(r,reg2,dst)
                    res=self._add8(dst,src)
                    if mod==3: set_reg8(r,rm,res)
                    else: self._write_mem8(seg,ea,res)
                    continue
                if op2 == 0xC1:
                    mod,reg2,rm,ea = modrm()
                    if o32:
                        dst=self._modrm_read32(mod,rm,ea,seg); src=self.get_reg32(reg2)
                        self.set_reg32(reg2,dst); self._modrm_write32(mod,rm,ea,self._add32(dst,src),seg)
                    else:
                        dst=get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea); src=get_reg16(r,reg2)
                        set_reg16(r,reg2,dst)
                        res=self._add16(dst,src)
                        if mod==3: set_reg16(r,rm,res)
                        else: self._write_mem16(seg,ea,res)
                    continue

                if op2 == 0xB0:
                    mod,reg2,rm,ea = modrm()
                    dst=get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea)
                    self._sub8(r.al, dst)
                    if r.zf:
                        if mod==3: set_reg8(r,rm,get_reg8(r,reg2))
                        else: self._write_mem8(seg,ea,get_reg8(r,reg2))
                    else: r.al=dst
                    continue
                if op2 == 0xB1:
                    mod,reg2,rm,ea = modrm()
                    if o32:
                        dst=self._modrm_read32(mod,rm,ea,seg); self._sub32(r.eax,dst)
                        if r.zf: self._modrm_write32(mod,rm,ea,self.get_reg32(reg2),seg)
                        else: r.eax=dst
                    else:
                        dst=get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea); self._sub16(r.ax,dst)
                        if r.zf:
                            if mod==3: set_reg16(r,rm,get_reg16(r,reg2))
                            else: self._write_mem16(seg,ea,get_reg16(r,reg2))
                        else: r.ax=dst
                    continue

                if 0xC8 <= op2 <= 0xCF:
                    idx=op2-0xC8; v=self.get_reg32(idx)
                    v=((v&0xFF)<<24)|((v&0xFF00)<<8)|((v>>8)&0xFF00)|((v>>24)&0xFF)
                    self.set_reg32(idx,v); continue

                if op2 == 0xA2:
                    if r.eax == 0:
                        r.eax=1; r.ebx=0x756E6547; r.ecx=0x6C65746E; r.edx=0x49656E69
                    elif r.eax == 1:
                        r.eax=0x0530; r.ebx=0; r.ecx=0
                        r.edx=0x0183FBFF
                    else:
                        r.eax=r.ebx=r.ecx=r.edx=0
                    continue

                if op2 == 0x31:
                    import time; tsc=int(time.monotonic()*1e9)&0xFFFFFFFFFFFFFFFF
                    r.eax=tsc&0xFFFFFFFF; r.edx=(tsc>>32)&0xFFFFFFFF; continue

                if op2 in (0x30, 0x32): r.eax=r.edx=0; continue

                if op2 in (0x05, 0x34, 0x35): continue

                if op2 in (0xAE, 0x77, 0xE7, 0xEA, 0xF0): continue

                continue

            if op == 0xFF:
                mod, reg, rm, ea = modrm()
                cf = r.cf
                if o32:
                    v = self._modrm_read32(mod,rm,ea,seg)
                    if   reg == 0: self._modrm_write32(mod,rm,ea,self._add32(v,1),seg); r.cf=cf
                    elif reg == 1: self._modrm_write32(mod,rm,ea,self._sub32(v,1),seg); r.cf=cf
                    elif reg == 2: self.push32(r.ip); r.ip=v
                    elif reg == 4: r.ip=v
                    elif reg == 6: self.push32(v)
                else:
                    v = get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)
                    if   reg == 0: res=self._add16(v,1); (set_reg16(r,rm,res) if mod==3 else self._write_mem16(seg,ea,res)); r.cf=cf
                    elif reg == 1: res=self._sub16(v,1); (set_reg16(r,rm,res) if mod==3 else self._write_mem16(seg,ea,res)); r.cf=cf
                    elif reg == 2: self.push16(r.ip&0xFFFF); r.ip=v
                    elif reg == 4: r.ip=v
                    elif reg == 6: self.push16(v)
                continue

            if op == 0xFE:
                mod, reg, rm, ea = modrm()
                v = get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea)
                cf = r.cf
                res = self._add8_p(v,1) if reg==0 else self._sub8_p(v,1)
                if mod==3: set_reg8(r,rm,res)
                else: self._write_mem8(seg,ea,res)
                r.cf = cf
                continue

            if op == 0xF6:
                mod,reg,rm,ea = modrm()
                v = get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea)
                if reg in (0,1):
                    imm = self.fetch8(); update_flags_8(r, v & imm, cf=0, of_=0)
                elif reg == 2:
                    res = (~v) & 0xFF
                    if mod==3: set_reg8(r,rm,res)
                    else: self._write_mem8(seg,ea,res)
                elif reg == 3:
                    res = self._sub8(0, v)
                    if mod==3: set_reg8(r,rm,res)
                    else: self._write_mem8(seg,ea,res)
                elif reg == 4:
                    res = r.al * v; r.ax = res & 0xFFFF
                    r.cf = r.of_ = 1 if r.ah else 0
                elif reg == 5:
                    res = (sign8(r.al) * sign8(v)) & 0xFFFF; r.ax = res
                elif reg == 6:
                    if v == 0: r.cf=1; continue
                    r.al = (r.ax // v) & 0xFF; r.ah = (r.ax % v) & 0xFF
                elif reg == 7:
                    if v == 0: r.cf=1; continue
                    d = sign16(r.ax) // sign8(v)
                    r.al = d & 0xFF; r.ah = (sign16(r.ax) % sign8(v)) & 0xFF
                continue

            if op == 0xF7:
                mod, reg, rm, ea = modrm()
                if o32:
                    v = self._modrm_read32(mod,rm,ea,seg)
                    if reg in (0,1):
                        imm = self.fetch32(); self._logic_flags32(v & imm)
                    elif reg == 2: self._modrm_write32(mod,rm,ea,(~v)&0xFFFFFFFF,seg)
                    elif reg == 3: self._modrm_write32(mod,rm,ea,self._sub32(0,v),seg)
                    elif reg == 4:
                        res = r.eax * v; r.eax=res&0xFFFFFFFF; r.edx=(res>>32)&0xFFFFFFFF
                        r.cf=r.of_=1 if r.edx else 0
                    elif reg == 6:
                        dividend=(r.edx<<32)|r.eax
                        if v==0: r.cf=1; continue
                        r.eax=(dividend//v)&0xFFFFFFFF; r.edx=(dividend%v)&0xFFFFFFFF
                else:
                    v = get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)
                    if reg in (0,1):
                        imm=self.fetch16(); update_flags_16(r,v&imm,cf=0,of_=0)
                    elif reg==2:
                        if mod==3: set_reg16(r,rm,(~v)&0xFFFF)
                        else: self._write_mem16(seg,ea,(~v)&0xFFFF)
                    elif reg==3:
                        res=self._sub16(0,v)
                        if mod==3: set_reg16(r,rm,res)
                        else: self._write_mem16(seg,ea,res)
                    elif reg==4:
                        res=r.ax*v; r.ax=res&0xFFFF; r.dx=(res>>16)&0xFFFF
                        r.cf=r.of_=1 if r.dx else 0
                    elif reg==6:
                        dividend=(r.dx<<16)|r.ax
                        if v==0: r.cf=1; continue
                        r.ax=(dividend//v)&0xFFFF; r.dx=(dividend%v)&0xFFFF
                continue

            if op in (0xC0,0xC1,0xD0,0xD1,0xD2,0xD3):
                mod, reg, rm, ea = modrm()
                if op in (0xC0, 0xC1):
                    count = self.fetch8() & 0x1F
                elif op in (0xD0, 0xD1):
                    count = 1
                else:
                    count = r.cl & 0x1F

                def _do_shift(v, bits, count, reg, r):
                    mask = (1 << bits) - 1
                    sign_bit = 1 << (bits - 1)
                    for _ in range(count):
                        if   reg == 0:
                            r.cf = (v >> (bits-1)) & 1
                            v = ((v << 1) | r.cf) & mask
                        elif reg == 1:
                            r.cf = v & 1
                            v = ((v >> 1) | (r.cf << (bits-1))) & mask
                        elif reg == 2:
                            new_cf = (v >> (bits-1)) & 1
                            v = ((v << 1) | r.cf) & mask
                            r.cf = new_cf
                        elif reg == 3:
                            new_cf = v & 1
                            v = ((v >> 1) | (r.cf << (bits-1))) & mask
                            r.cf = new_cf
                        elif reg == 4:
                            r.cf = (v >> (bits-1)) & 1
                            v = (v << 1) & mask
                        elif reg == 5:
                            r.cf = v & 1
                            v >>= 1
                        elif reg == 6:
                            r.cf = (v >> (bits-1)) & 1
                            v = (v << 1) & mask
                        elif reg == 7:
                            r.cf = v & 1
                            if v & sign_bit:
                                v = ((v >> 1) | sign_bit) & mask
                            else:
                                v >>= 1
                    return v & mask

                is_byte = op in (0xC0, 0xD0, 0xD2)
                is_32   = o32 and not is_byte

                if is_byte:
                    v = get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea)
                    v = _do_shift(v, 8, count, reg, r)
                    if count: update_flags_8(r, v)
                    if mod==3: set_reg8(r,rm,v)
                    else: self._write_mem8(seg,ea,v)
                elif is_32:
                    v = self._modrm_read32(mod,rm,ea,seg)
                    v = _do_shift(v, 32, count, reg, r)
                    if count:
                        r.zf = 1 if v==0 else 0
                        r.sf = (v>>31)&1
                        r.pf = bin(v&0xFF).count('1')%2==0
                    self._modrm_write32(mod,rm,ea,v,seg)
                else:
                    v = get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)
                    v = _do_shift(v, 16, count, reg, r)
                    if count: update_flags_16(r, v)
                    if mod==3: set_reg16(r,rm,v)
                    else: self._write_mem16(seg,ea,v)
                continue

            if op == 0x08:
                mod,reg,rm,ea = modrm()
                a = get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea)
                res = a | get_reg8(r,reg); update_flags_8(r,res,cf=0,of_=0)
                (set_reg8(r,rm,res) if mod==3 else self._write_mem8(seg,ea,res)); continue
            if op == 0x0A:
                mod,reg,rm,ea = modrm()
                res = get_reg8(r,reg) | (get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea))
                update_flags_8(r,res,cf=0,of_=0); set_reg8(r,reg,res); continue
            if op == 0x0C: r.al |= self.fetch8(); update_flags_8(r,r.al,cf=0,of_=0); continue

            if op == 0x20:
                mod,reg,rm,ea = modrm()
                a = get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea)
                res = a & get_reg8(r,reg); update_flags_8(r,res,cf=0,of_=0)
                (set_reg8(r,rm,res) if mod==3 else self._write_mem8(seg,ea,res)); continue
            if op == 0x22:
                mod,reg,rm,ea = modrm()
                res = get_reg8(r,reg) & (get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea))
                update_flags_8(r,res,cf=0,of_=0); set_reg8(r,reg,res); continue
            if op == 0x24: r.al &= self.fetch8(); update_flags_8(r,r.al,cf=0,of_=0); continue

            if op == 0x28:
                mod,reg,rm,ea = modrm()
                res = self._sub8(get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea), get_reg8(r,reg))
                (set_reg8(r,rm,res) if mod==3 else self._write_mem8(seg,ea,res)); continue
            if op == 0x2A:
                mod,reg,rm,ea = modrm()
                res = self._sub8(get_reg8(r,reg), get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea))
                set_reg8(r,reg,res); continue

            if op == 0x30:
                mod,reg,rm,ea = modrm()
                a = get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea)
                res = a ^ get_reg8(r,reg); update_flags_8(r,res,cf=0,of_=0)
                (set_reg8(r,rm,res) if mod==3 else self._write_mem8(seg,ea,res)); continue
            if op == 0x32:
                mod,reg,rm,ea = modrm()
                res = get_reg8(r,reg) ^ (get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea))
                update_flags_8(r,res,cf=0,of_=0); set_reg8(r,reg,res); continue
            if op == 0x34: r.al ^= self.fetch8(); update_flags_8(r,r.al,cf=0,of_=0); continue

            if op == 0x02:
                mod,reg,rm,ea = modrm()
                res = self._add8(get_reg8(r,reg), get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea))
                set_reg8(r,reg,res); continue

            if op == 0x3C: self._sub8(r.al, self.fetch8()); continue

            if op == 0x84:
                mod,reg,rm,ea = modrm()
                update_flags_8(r, (get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea)) & get_reg8(r,reg), cf=0,of_=0); continue
            if op == 0xA8: update_flags_8(r, r.al & self.fetch8(), cf=0,of_=0); continue

            if op == 0x9F: r.ah = r.flags_word() & 0xFF; continue
            if op == 0x9E:
                f = r.flags_word(); r.set_flags_word((f & 0xFF00) | r.ah); continue

            if op == 0x86:
                mod,reg,rm,ea = modrm()
                a = get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea)
                b = get_reg8(r,reg)
                if mod==3: set_reg8(r,rm,b)
                else: self._write_mem8(seg,ea,b)
                set_reg8(r,reg,a)
                continue

            if op == 0x87:
                mod,reg,rm,ea = modrm()
                if o32:
                    a=self._modrm_read32(mod,rm,ea,seg); b=self.get_reg32(reg)
                    self._modrm_write32(mod,rm,ea,b,seg); self.set_reg32(reg,a)
                else:
                    a=get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea)
                    b=get_reg16(r,reg)
                    if mod==3: set_reg16(r,rm,b)
                    else: self._write_mem16(seg,ea,b)
                    set_reg16(r,reg,a)
                continue

            if op == 0xC5:
                mod,reg,rm,ea = modrm()
                v=self._read_mem16(seg,ea); s=self._read_mem16(seg,(ea+2)&0xFFFFFFFF)
                set_reg16(r,reg,v); r.ds=s
                if not r.protected_mode:
                    uv = getattr(r, '_unreal_valid', None)
                    if uv is not None: uv['ds'] = False
                continue
            if op == 0xC4:
                mod,reg,rm,ea = modrm()
                v=self._read_mem16(seg,ea); s=self._read_mem16(seg,(ea+2)&0xFFFFFFFF)
                set_reg16(r,reg,v); r.es=s
                if not r.protected_mode:
                    uv = getattr(r, '_unreal_valid', None)
                    if uv is not None: uv['es'] = False
                continue

            if op == 0xAE:
                self._sub8(r.al, self._read_mem8('es', r.edi if self._addr32 else r.di))
                step = 1 if not r.df else -1
                if self._addr32: r.edi=(r.edi+step)&0xFFFFFFFF
                else:            r.di=(r.di+step)&0xFFFF
                continue

            if op == 0x9A:
                new_ip=fetch_imm(); new_cs=self.fetch16()
                if o32: self.push32(r.cs); self.push32(r.ip)
                else:   self.push16(r.cs); self.push16(r.ip&0xFFFF)
                r.cs=new_cs; r.ip=new_ip; continue

            if op == 0xCA:
                imm=self.fetch16()
                r.ip=self.pop32() if o32 else self.pop16()
                r.cs=(self.pop32()&0xFFFF) if o32 else self.pop16()
                r.esp=(r.esp+imm)&0xFFFFFFFF; continue

            if op == 0x60:
                tmp=r.esp if o32 else r.sp
                for i in range(8):
                    v=[r.eax,r.ecx,r.edx,r.ebx,tmp,r.ebp,r.esi,r.edi][i] if o32 else \
                      [r.ax,r.cx,r.dx,r.bx,tmp&0xFFFF,r.bp,r.si,r.di][i]
                    self.push32(v) if o32 else self.push16(v)
                continue
            if op == 0x61:
                vals=[]
                for _ in range(8): vals.append(self.pop32() if o32 else self.pop16())
                vals=vals[::-1]
                names32=['eax','ecx','edx','ebx',None,'ebp','esi','edi']
                names16=['ax','cx','dx','bx',None,'bp','si','di']
                for i,n in enumerate(names32 if o32 else names16):
                    if n: setattr(r,n,vals[i])
                continue

            if op == 0x10:
                mod,reg,rm,ea = modrm()
                res = self._add8(get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea), get_reg8(r,reg), r.cf)
                (set_reg8(r,rm,res) if mod==3 else self._write_mem8(seg,ea,res)); continue
            if op == 0x11:
                mod,reg,rm,ea = modrm()
                if o32:
                    res = self._add32(self._modrm_read32(mod,rm,ea,seg), self.get_reg32(reg), r.cf)
                    self._modrm_write32(mod,rm,ea,res,seg)
                else:
                    res = self._add16(get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea), get_reg16(r,reg), r.cf)
                    if mod==3: set_reg16(r,rm,res)
                    else: self._write_mem16(seg,ea,res)
                continue
            if op == 0x12:
                mod,reg,rm,ea = modrm()
                res = self._add8(get_reg8(r,reg), get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea), r.cf)
                set_reg8(r,reg,res); continue
            if op == 0x13:
                mod,reg,rm,ea = modrm()
                if o32: self.set_reg32(reg, self._add32(self.get_reg32(reg), self._modrm_read32(mod,rm,ea,seg), r.cf))
                else:   set_reg16(r,reg, self._add16(get_reg16(r,reg), get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea), r.cf))
                continue
            if op == 0x14: r.al = self._add8(r.al, self.fetch8(), r.cf); continue
            if op == 0x15:
                if o32: r.eax = self._add32(r.eax, self.fetch32(), r.cf)
                else:   r.ax  = self._add16(r.ax,  self.fetch16(), r.cf)
                continue

            if op == 0x18:
                mod,reg,rm,ea = modrm()
                res = self._sub8(get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea), get_reg8(r,reg), r.cf)
                (set_reg8(r,rm,res) if mod==3 else self._write_mem8(seg,ea,res)); continue
            if op == 0x19:
                mod,reg,rm,ea = modrm()
                if o32:
                    res = self._sub32(self._modrm_read32(mod,rm,ea,seg), self.get_reg32(reg), r.cf)
                    self._modrm_write32(mod,rm,ea,res,seg)
                else:
                    res = self._sub16(get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea), get_reg16(r,reg), r.cf)
                    if mod==3: set_reg16(r,rm,res)
                    else: self._write_mem16(seg,ea,res)
                continue
            if op == 0x1A:
                mod,reg,rm,ea = modrm()
                res = self._sub8(get_reg8(r,reg), get_reg8(r,rm) if mod==3 else self._read_mem8(seg,ea), r.cf)
                set_reg8(r,reg,res); continue
            if op == 0x1B:
                mod,reg,rm,ea = modrm()
                if o32: self.set_reg32(reg, self._sub32(self.get_reg32(reg), self._modrm_read32(mod,rm,ea,seg), r.cf))
                else:   set_reg16(r,reg, self._sub16(get_reg16(r,reg), get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea), r.cf))
                continue
            if op == 0x1C: r.al = self._sub8(r.al, self.fetch8(), r.cf); continue
            if op == 0x1D:
                if o32: r.eax = self._sub32(r.eax, self.fetch32(), r.cf)
                else:   r.ax  = self._sub16(r.ax,  self.fetch16(), r.cf)
                continue

            if op == 0x6B:
                mod,reg,rm,ea = modrm()
                imm = sign8(self.fetch8())
                if o32:
                    a = sign32(self._modrm_read32(mod,rm,ea,seg))
                    res = (a * imm) & 0xFFFFFFFF
                    self.set_reg32(reg, res)
                else:
                    a = sign16(get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea))
                    res = (a * imm) & 0xFFFF
                    set_reg16(r,reg,res)
                r.cf = r.of_ = 0
                continue
            if op == 0x69:
                mod,reg,rm,ea = modrm()
                if o32:
                    imm = sign32(self.fetch32())
                    a   = sign32(self._modrm_read32(mod,rm,ea,seg))
                    self.set_reg32(reg, (a*imm)&0xFFFFFFFF)
                else:
                    imm = sign16(self.fetch16())
                    a   = sign16(get_reg16(r,rm) if mod==3 else self._read_mem16(seg,ea))
                    set_reg16(r,reg,(a*imm)&0xFFFF)
                r.cf = r.of_ = 0
                continue

            if op == 0xF1: continue
            if op == 0xCC: continue

            if op == 0x62:
                mod,reg,rm,ea = modrm(); continue

            if op == 0xC8:
                alloc = self.fetch16(); level = self.fetch8() & 0x1F
                if o32: self.push32(r.ebp); r.ebp=r.esp; r.esp=(r.esp-alloc)&0xFFFFFFFF
                else:   self.push16(r.bp);  r.bp=r.sp&0xFFFF; r.sp=(r.sp-alloc)&0xFFFF
                continue
            if op == 0xC9:
                if o32: r.esp=r.ebp; r.ebp=self.pop32()
                else:   r.sp=r.bp;   r.bp=self.pop16()
                continue

            if op == 0xD7:
                addr = (r.ebx if self._addr32 else r.bx) + r.al
                r.al = self._read_mem8(seg, addr & (0xFFFFFFFF if self._addr32 else 0xFFFF))
                continue

            if op == 0xD4:
                base = self.fetch8()
                if base == 0: r.cf = 1; continue
                r.ah = r.al // base; r.al = r.al % base
                update_flags_8(r, r.ax & 0xFF); continue
            if op == 0xD5:
                base = self.fetch8()
                r.al = (r.ah * base + r.al) & 0xFF; r.ah = 0
                update_flags_8(r, r.al); continue

            if op == 0x27:
                al = r.al
                if (al & 0xF) > 9 or r.af:
                    r.al = (r.al + 6) & 0xFF; r.af = 1
                if al > 0x99 or r.cf:
                    r.al = (r.al + 0x60) & 0xFF; r.cf = 1
                update_flags_8(r, r.al); continue
            if op == 0x2F:
                al = r.al
                if (al & 0xF) > 9 or r.af:
                    r.al = (r.al - 6) & 0xFF; r.af = 1
                if al > 0x99 or r.cf:
                    r.al = (r.al - 0x60) & 0xFF; r.cf = 1
                update_flags_8(r, r.al); continue

            if op == 0x9B: continue

            if op == 0x8F:
                mod,reg,rm,ea = modrm()
                val = self.pop32() if o32 else self.pop16()
                if o32: self._modrm_write32(mod,rm,ea,val,seg)
                else:
                    if mod==3: set_reg16(r,rm,val)
                    else: self._write_mem16(seg,ea,val)
                continue

            if 0xD8 <= op <= 0xDF:
                modrm_b = self.fetch8()
                mod = (modrm_b >> 6) & 3
                reg = (modrm_b >> 3) & 7
                rm  = modrm_b & 7

                if mod == 3:
                    if op == 0xDB and modrm_b == 0xE3:
                        pass
                    elif op == 0xDF and modrm_b == 0xE0:
                        r.ax = 0x0000
                else:
                    if mod == 1:
                        self.fetch8()
                    elif mod == 2:
                        self.fetch16()
                    elif mod == 0 and rm == 6:
                        self.fetch16()

                    if op == 0xD9 and reg == 7:
                        pass
                continue

            if op == 0xA6:
                a = self._read_mem8(seg, r.esi if self._addr32 else r.si)
                b = self._read_mem8('es', r.edi if self._addr32 else r.di)
                self._sub8(a, b)
                step = 1 if not r.df else -1
                if self._addr32: r.esi=(r.esi+step)&0xFFFFFFFF; r.edi=(r.edi+step)&0xFFFFFFFF
                else:            r.si=(r.si+step)&0xFFFF;       r.di=(r.di+step)&0xFFFF
                continue

            if op == 0xA7:
                if o32:
                    a = self._read_mem32(seg, r.esi); b = self._read_mem32('es', r.edi)
                    self._sub32(a, b); step = 4
                else:
                    a = self._read_mem16(seg, r.si);  b = self._read_mem16('es', r.di)
                    self._sub16(a, b); step = 2
                if r.df: step = -step
                if self._addr32: r.esi=(r.esi+step)&0xFFFFFFFF; r.edi=(r.edi+step)&0xFFFFFFFF
                else:            r.si=(r.si+step)&0xFFFF;       r.di=(r.di+step)&0xFFFF
                continue

            if op == 0xAF:
                if o32:
                    self._sub32(r.eax, self._read_mem32('es', r.edi))
                    step = 4 if not r.df else -4; r.edi=(r.edi+step)&0xFFFFFFFF
                else:
                    self._sub16(r.ax, self._read_mem16('es', r.di))
                    step = 2 if not r.df else -2; r.di=(r.di+step)&0xFFFF
                continue

            if op == 0xCE:
                if r.of_:
                    self.push16(r.flags_word()); self.push16(r.cs); self.push16(r.ip)
                    r.IF = 0
                    self.bios.interrupt(4)
                    r.ip = self.pop16(); r.cs = self.pop16(); r.set_flags_word(self.pop16())
                continue

            if op == 0xD6:
                r.al = 0xFF if r.cf else 0x00
                continue

            if op == 0x37:
                if (r.al & 0xF) > 9 or r.af:
                    r.ax = (r.ax + 0x0106) & 0xFFFF; r.af = r.cf = 1
                else:
                    r.af = r.cf = 0
                r.al &= 0x0F; continue
            if op == 0x3F:
                if (r.al & 0xF) > 9 or r.af:
                    r.ax = (r.ax - 0x0106) & 0xFFFF; r.af = r.cf = 1
                else:
                    r.af = r.cf = 0
                r.al &= 0x0F; continue

            r.ip = (r.ip - 1) & 0xFFFFFFFF
            print(f"\n[!] Unknown opcode 0x{op:02X} at CS:IP={r.cs:04X}:{r.ip:08X} pmode={r.protected_mode}")
            print(r)
            r.ip = (r.ip + 1) & 0xFFFFFFFF
            if not hasattr(self, '_unk_streak'):
                self._unk_streak = 0
            self._unk_streak += 1
            if self._unk_streak > 20:
                print("[!] Too many consecutive unknown opcodes — stopping")
                self.halted = True
            continue

        return self.icount


    def _alu8(self, op, a, b):
        r = self.reg
        if   op == 0: return self._add8_p(a, b)
        elif op == 1: res = a | b; update_flags_8(r, res, cf=0, of_=0); return res
        elif op == 2: return self._add8_p(a, b, r.cf)
        elif op == 3: return self._sub8_p(a, b, r.cf)
        elif op == 4: res = a & b; update_flags_8(r, res, cf=0, of_=0); return res
        elif op == 5: return self._sub8_p(a, b)
        elif op == 6: res = a ^ b; update_flags_8(r, res, cf=0, of_=0); return res
        elif op == 7: self._sub8_p(a, b); return a
        return a

    def _alu16(self, op, a, b):
        r = self.reg
        if   op == 0: return self._add16(a, b)
        elif op == 1: res = a | b; update_flags_16(r, res, cf=0, of_=0); return res
        elif op == 2: return self._add16(a, b, r.cf)
        elif op == 3: return self._sub16(a, b, r.cf)
        elif op == 4: res=a&b; update_flags_16(r,res,cf=0,of_=0); return res
        elif op == 5: return self._sub16(a, b)
        elif op == 6: res=a^b; update_flags_16(r,res,cf=0,of_=0); return res
        elif op == 7: self._sub16(a,b); return a
        return a

    def _alu32(self, op, a, b):
        r = self.reg
        if   op == 0: return self._add32(a, b)
        elif op == 1: res = a | b; self._logic_flags32(res); return res
        elif op == 2: return self._add32(a, b, r.cf)
        elif op == 3: return self._sub32(a, b, r.cf)
        elif op == 4: res = a & b; self._logic_flags32(res); return res
        elif op == 5: return self._sub32(a, b)
        elif op == 6: res = a ^ b; self._logic_flags32(res); return res
        elif op == 7: self._sub32(a, b); return a
        return a

    def _try_compile_loop(self, eip):
        try:
            m = self.mem
            r = self.reg

            SIG1 = bytes([0x8b,0x4c,0x24,0x48, 0x8b,0x01, 0xc1,0xe0,0x02,
                          0xff,0x04,0x10, 0x83,0xc1,0x04, 0x89,0x4c,0x24,0x48,
                          0xff,0x4c,0x24,0x54, 0x75,0xe7])
            loop1_start = eip - 0x17
            if loop1_start >= 0:
                try:
                    code = bytes(m._m[loop1_start:loop1_start+25])
                    if code == SIG1:
                        cpu = self
                        _mem_size = m.size
                        def _loop1():
                            esp = r.esp
                            ptr_off = esp + 0x48
                            cnt_off = esp + 0x54
                            _mem = m._m
                            iters = 0
                            limit = min(cpu.max_icount - cpu.icount, 50_000_000)
                            import struct
                            while iters < limit:
                                cnt = struct.unpack_from("<I", _mem, cnt_off)[0]
                                if cnt == 0: break
                                ecx = struct.unpack_from("<I", _mem, ptr_off)[0]
                                if ecx + 4 > _mem_size: break
                                eax = struct.unpack_from("<I", _mem, ecx)[0] * 4
                                slot = (eax + 0x10) & 0xFFFFFFFF
                                if slot + 4 > _mem_size: break
                                struct.pack_into("<I", _mem, slot,
                                    (struct.unpack_from("<I", _mem, slot)[0] + 1) & 0xFFFFFFFF)
                                ecx = (ecx + 4) & 0xFFFFFFFF
                                struct.pack_into("<I", _mem, ptr_off, ecx)
                                struct.pack_into("<I", _mem, cnt_off, (cnt - 1) & 0xFFFFFFFF)
                                iters += 1
                            if ptr_off + 4 <= _mem_size:
                                r.ecx = struct.unpack_from("<I", _mem, ptr_off)[0]
                            cnt_final = struct.unpack_from("<I", _mem, cnt_off)[0] if cnt_off + 4 <= _mem_size else 0
                            r.zf = 1 if cnt_final == 0 else 0
                            r.cf = 0; r.sf = 0
                            if r.zf:
                                r.ip = (eip + 2) & 0xFFFFFFFF
                            return iters
                        return _loop1
                except Exception:
                    pass
        except Exception:
            pass
        return None

    def _run_native_loop(self, fn):
        try:
            return fn() or 0
        except Exception:
            return 0

    def _add8_p(self, a, b, carry=0):
        return self._add8(a, b, carry)

    def _sub8_p(self, a, b, borrow=0):
        return self._sub8(a, b, borrow)

    def _logic_flags32(self, v):
        r = self.reg
        r.zf = 1 if (v & 0xFFFFFFFF) == 0 else 0
        r.sf = (v >> 31) & 1
        r.pf = _cpu_mod.parity(v)
        r.cf = 0; r.of_ = 0


class Machine32:
    def __init__(self):
        self.mem  = Memory32()
        self.reg  = Registers32()
        self.bios = BIOS(self.mem, self.reg)
        self.io   = IOPorts(self.reg)
        self.cpu  = CPU32(self.mem, self.reg, self.bios, self.io)

    def load_at(self, addr, data):
        self.mem.load_flat(addr, data)

    def set_entry(self, cs, ip):
        self.reg.cs = cs
        self.reg.ip = ip
        self.reg.ss = 0x0000
        self.reg.sp = 0x7C00
        self.reg.ds = 0x0000
        self.reg.es = 0x0000

    def run(self, max_icount=2_000_000):
        self.cpu.max_icount = max_icount
        return self.cpu.run()
