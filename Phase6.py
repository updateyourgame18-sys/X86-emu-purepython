"""
phase6.py — x86emu Phase 6: Pythonista UI Layer
VGA framebuffer display + touch input → PS/2 mouse + keyboard

Features:
  - VGA text mode (80x25) rendered as a terminal on screen
  - VGA graphics mode (320x200 Mode 13h) rendered as pixels
  - VESA LFB mode (vga=791, 1024x768x16bpp) rendered as pixels — this is
    the mode DSL's vesafb driver switches into right after "Ready." in
    order to draw the Tux boot logo. Previously this file only ever read
    the legacy 0xB8000 text buffer, so once the kernel handed the console
    over to vesafb the screen went black forever (Tux was actually being
    drawn correctly into RAM at LFB_ADDR — nothing was ever reading it
    back out to render it).
  - Touch → PS/2 mouse:
      single tap   = right click
      double tap   = left click
      drag         = mouse move
  - Keyboard button in corner → text input sheet → injects keystrokes
  - Runs the emulator on a background thread so UI stays responsive
  - Live screen refresh every 100ms

Pythonista-specific: uses scene, ui, threading modules.
Falls back to a headless text mode if not running in Pythonista.
"""

import threading
import time
import struct
import sys
import os
import math


# Detect Pythonista

IN_PYTHONISTA = False
try:
    import scene
    import ui
    IN_PYTHONISTA = True
except ImportError:
    pass


# PS/2 mouse controller

class PS2Mouse:
    """
    Emulates an 8042-connected PS/2 mouse.
    Feeds mouse packets into the emulator's keyboard controller port.
    Mouse packets: 3 bytes
      Byte 0: flags  (bit0=left, bit1=right, bit2=middle, bit3=1, bit4=x-sign, bit5=y-sign)
      Byte 1: X movement (signed)
      Byte 2: Y movement (signed, positive = up in PS/2 convention)
    IRQ12 fires after each packet.
    """
    def __init__(self, pic):
        self.pic    = pic
        self._buf   = []      # pending bytes
        self._lock  = threading.Lock()
        self.enabled = True

        # Current button state
        self.left   = False
        self.right  = False
        self.middle = False

    def move(self, dx, dy, left=None, right=None, middle=None):
        """Queue a mouse movement/click packet."""
        if not self.enabled:
            return
        if left   is not None: self.left   = left
        if right  is not None: self.right  = right
        if middle is not None: self.middle = middle

        # Clamp to signed byte range
        dx = max(-127, min(127, int(dx)))
        dy = max(-127, min(127, int(dy)))

        flags = 0x08   # bit 3 always set
        if self.left:   flags |= 0x01
        if self.right:  flags |= 0x02
        if self.middle: flags |= 0x04
        if dx < 0: flags |= 0x10
        if dy < 0: flags |= 0x20

        packet = [flags, dx & 0xFF, (-dy) & 0xFF]  # PS/2 Y is inverted

        with self._lock:
            self._buf.extend(packet)
            self.pic.raise_irq(12)

    def click_left(self):
        self.move(0, 0, left=True)
        time.sleep(0.05)
        self.move(0, 0, left=False)

    def click_right(self):
        self.move(0, 0, right=True)
        time.sleep(0.05)
        self.move(0, 0, right=False)

    def read_byte(self):
        with self._lock:
            if self._buf:
                return self._buf.pop(0)
        return 0x00

    def has_data(self):
        with self._lock:
            return len(self._buf) > 0



# PS/2 Keyboard


# Scancode set 1 (XT scancodes) for common keys
SCANCODE_MAP = {
    '\x1b': 0x01,  # ESC
    '1': 0x02, '2': 0x03, '3': 0x04, '4': 0x05, '5': 0x06,
    '6': 0x07, '7': 0x08, '8': 0x09, '9': 0x0A, '0': 0x0B,
    '-': 0x0C, '=': 0x0D,
    '\x08': 0x0E,  # Backspace
    '\t':   0x0F,  # Tab
    'q': 0x10, 'w': 0x11, 'e': 0x12, 'r': 0x13, 't': 0x14,
    'y': 0x15, 'u': 0x16, 'i': 0x17, 'o': 0x18, 'p': 0x19,
    '[': 0x1A, ']': 0x1B,
    '\n': 0x1C, '\r': 0x1C,  # Enter
    'a': 0x1E, 's': 0x1F, 'd': 0x20, 'f': 0x21, 'g': 0x22,
    'h': 0x23, 'j': 0x24, 'k': 0x25, 'l': 0x26,
    ';': 0x27, "'": 0x28, '`': 0x29,
    '\\': 0x2B,
    'z': 0x2C, 'x': 0x2D, 'c': 0x2E, 'v': 0x2F, 'b': 0x30,
    'n': 0x31, 'm': 0x32, ',': 0x33, '.': 0x34, '/': 0x35,
    ' ': 0x39,  # Space
}

SHIFT_MAP = {
    '!': '1', '@': '2', '#': '3', '$': '4', '%': '5',
    '^': '6', '&': '7', '*': '8', '(': '9', ')': '0',
    '_': '-', '+': '=', '{': '[', '}': ']', '|': '\\',
    ':': ';', '"': "'", '<': ',', '>': '.', '?': '/',
    '~': '`',
    'Q': 'q', 'W': 'w', 'E': 'e', 'R': 'r', 'T': 't',
    'Y': 'y', 'U': 'u', 'I': 'i', 'O': 'o', 'P': 'p',
    'A': 'a', 'S': 's', 'D': 'd', 'F': 'f', 'G': 'g',
    'H': 'h', 'J': 'j', 'K': 'k', 'L': 'l',
    'Z': 'z', 'X': 'x', 'C': 'c', 'V': 'v', 'B': 'b',
    'N': 'n', 'M': 'm',
}

LSHIFT_SC = 0x2A
RSHIFT_SC = 0x36

class PS2Keyboard:
    def __init__(self, pic, mem=None):
        self.pic   = pic
        self.mem   = mem
        self._buf  = []
        self._lock = threading.Lock()

    def type_string(self, text):
        """Inject a string of characters as PS/2 scancodes."""
        for ch in text:
            self._type_char(ch)
        # Final Enter
        if text and text[-1] not in ('\n', '\r'):
            self._type_char('\r')

    def inject_bios_buffer(self, ch):
        """Write a character directly into the classic BIOS keyboard
        buffer (0x41A head, 0x41C tail, 0x41E-0x43D 16-entry circular
        buffer of ASCII+scancode word pairs), bypassing the normal
        IRQ1-driven path entirely. Needed when guest code polls this
        buffer directly with interrupts disabled — real Linux video.S
        runs exactly this way, so the IRQ1 handler that would normally
        populate the buffer can never actually fire to service a
        keypress, even though a real keypress at the PS/2 controller
        level would be sitting there waiting."""
        if self.mem is None:
            return
        BUF_START, BUF_END = 0x41E, 0x43E
        head = self.mem.read16_flat(0x41A)
        tail = self.mem.read16_flat(0x41C)
        next_tail = tail + 2
        if next_tail >= BUF_END:
            next_tail = BUF_START
        if next_tail == head:
            return  # buffer full
        sc = SCANCODE_MAP.get(ch, 0)
        self.mem.write8_flat(tail, ord(ch) & 0xFF)      # ASCII
        self.mem.write8_flat(tail + 1, sc & 0xFF)       # scan code
        self.mem.write16_flat(0x41C, next_tail)

    def clear_bios_buffer(self):
        """Reset the BIOS keyboard buffer to empty (head==tail). Used to
        flush stale, never-consumed entries before injecting a keystroke
        meant for a specific later prompt — otherwise old characters
        (e.g. from an earlier auto-typed command) sit ahead of the new
        one in the circular buffer and get read first."""
        if self.mem is None:
            return
        BUF_START = 0x41E
        self.mem.write16_flat(0x41A, BUF_START)
        self.mem.write16_flat(0x41C, BUF_START)

    def _type_char(self, ch):
        needs_shift = False
        key = ch
        if ch in SHIFT_MAP:
            needs_shift = True
            key = SHIFT_MAP[ch]
        sc = SCANCODE_MAP.get(key)
        if sc is None:
            return
        with self._lock:
            if needs_shift:
                self._buf.append(LSHIFT_SC)        # shift press
            self._buf.append(sc)                    # key press
            self._buf.append(sc | 0x80)             # key release
            if needs_shift:
                self._buf.append(LSHIFT_SC | 0x80) # shift release
        self.pic.raise_irq(1)
        self.inject_bios_buffer(ch)

    def read_byte(self):
        with self._lock:
            if self._buf:
                return self._buf.pop(0)
        return 0x00

    def has_data(self):
        with self._lock:
            return len(self._buf) > 0



# Extended KBD controller that routes PS/2 mouse + keyboard

class KBD6:
    """
    Replaces Phase 3 KBD stub with real PS/2 routing.
    Port 0x60: data (keyboard or mouse depending on last command)
    Port 0x64: status
    """
    def __init__(self, ps2_kbd, ps2_mouse):
        self.kbd   = ps2_kbd
        self.mouse = ps2_mouse
        self._cmd  = None
        self._mouse_mode = False

    def has_data(self):
        return self.kbd.has_data() or (self._mouse_mode and self.mouse.has_data())

    def read_byte(self):
        return self.read_data()

    def read_data(self):
        if self._mouse_mode and self.mouse.has_data():
            return self.mouse.read_byte()
        if self.kbd.has_data():
            return self.kbd.read_byte()
        return 0x00

    def read_status(self):
        has = self.has_data()
        status = 0x00
        if has:
            status |= 0x01
        if self._mouse_mode and self.mouse.has_data():
            status |= 0x20
        return status

    def write_cmd(self, v):
        self._cmd = v
        if v == 0xA8:   self._mouse_mode = True
        elif v == 0xA7: self._mouse_mode = False
        elif v == 0xD4: self._mouse_mode = True

    def write_data(self, v):
        if self._cmd == 0xD4:
            if v == 0xFF:
                self.mouse._buf = [0xAA, 0x00]
            elif v == 0xF4:
                self.mouse.enabled = True
                self.mouse._buf = [0xFA]
            elif v == 0xF5:
                self.mouse.enabled = False
                self.mouse._buf = [0xFA]
            self._cmd = None



# VGA state reader

class VGAReader:
    """Reads VGA state from emulator memory and produces display data."""

    TEXT_BASE   = 0xB8000
    COLS        = 80
    ROWS        = 25
    GFX_BASE    = 0xA0000   # Mode 13h linear framebuffer
    GFX_W       = 320
    GFX_H       = 200

    # Standard 16-color VGA palette (RGB tuples)
    PALETTE16 = [
        (0,0,0), (0,0,170), (0,170,0), (0,170,170),
        (170,0,0), (170,0,170), (170,85,0), (170,170,170),
        (85,85,85), (85,85,255), (85,255,85), (85,255,255),
        (255,85,85), (255,85,255), (255,255,85), (255,255,255),
    ]

    def __init__(self, mem, vga_mode_ptr=None):
        self.mem = mem
        self._mode_ptr = vga_mode_ptr  # address in mem that holds current VGA mode
        self._mode13_palette = self._build_default_mode13_palette()

    def _build_default_mode13_palette(self):
        """Build the default 256-color Mode 13h palette."""
        pal = []
        # Standard VGA 256-color palette (simplified)
        for i in range(256):
            if i < 16:
                r, g, b = self.PALETTE16[i]
            elif i < 232:
                # 6x6x6 color cube
                idx = i - 16
                b_i = idx % 6; g_i = (idx // 6) % 6; r_i = idx // 36
                r = r_i * 51; g = g_i * 51; b = b_i * 51
            else:
                # Grayscale
                v = (i - 232) * 10 + 8
                r = g = b = v
            pal.append((r, g, b))
        return pal

    def is_graphics_mode(self):
        """Check if VGA is in graphics mode (Mode 13h or similar)."""
        if self._mode_ptr:
            return self.mem.read8_flat(self._mode_ptr) != 0
        # Heuristic: check if 0xA0000 area has non-zero content
        sample = self.mem._m[0xA0000:0xA0100]
        return any(b != 0 for b in sample)

    def read_text_screen(self):
        """
        Returns list of (char, fg_color, bg_color) for each cell.
        """
        cells = []
        for row in range(self.ROWS):
            for col in range(self.COLS):
                off  = self.TEXT_BASE + (row * self.COLS + col) * 2
                ch   = self.mem.read8_flat(off)
                attr = self.mem.read8_flat(off + 1)
                fg   = attr & 0x0F
                bg   = (attr >> 4) & 0x07
                cells.append((ch, fg, bg))
        return cells

    def read_gfx_screen(self):
        """Returns raw 320x200 pixel data as bytearray of palette indices."""
        return self.mem._m[self.GFX_BASE:self.GFX_BASE + self.GFX_W * self.GFX_H]

    def gfx_to_rgb(self, pixels):
        """Convert Mode 13h palette indices to RGB888 bytes."""
        out = bytearray(len(pixels) * 3)
        for i, p in enumerate(pixels):
            r, g, b = self._mode13_palette[p]
            out[i*3] = r; out[i*3+1] = g; out[i*3+2] = b
        return out



# Headless runner (non-Pythonista)

class HeadlessRunner:
    """
    Runs the emulator without UI, printing VGA text output periodically.
    Used when not in Pythonista.
    """
    def __init__(self, machine):
        self.machine = machine
        self.vga     = VGAReader(machine.mem)
        self._stop   = threading.Event()

    def run(self, max_icount=200_000_000):
        print("[phase6] Running headless (no Pythonista UI)")
        print("[phase6] Press Ctrl+C to stop\n")

        m = self.machine

        def emu_thread():
            try:
                m.run(max_icount=max_icount)
            except KeyboardInterrupt:
                pass

        t = threading.Thread(target=emu_thread, daemon=True)
        t.start()

        last_output = ''
        try:
            while t.is_alive():
                time.sleep(0.5)
                out = m.bios.get_output()
                if out != last_output:
                    new = out[len(last_output):]
                    sys.stdout.write(new)
                    sys.stdout.flush()
                    last_output = out
        except KeyboardInterrupt:
            self._stop.set()
            print("\n[phase6] Stopped")

        t.join(timeout=2)
        return self.machine



# Pythonista Scene-based UI

if IN_PYTHONISTA:
    import scene
    import ui

    # Monospace font for VGA text rendering
    VGA_FONT    = 'Courier'
    VGA_FONT_W  = 8    # approx char width at size 14
    VGA_FONT_H  = 14

    # CGA/VGA color table → Pythonista scene colors
    CGA_COLORS = [
        '#000000', '#0000AA', '#00AA00', '#00AAAA',
        '#AA0000', '#AA00AA', '#AA5500', '#AAAAAA',
        '#555555', '#5555FF', '#55FF55', '#55FFFF',
        '#FF5555', '#FF55FF', '#FFFF55', '#FFFFFF',
    ]

    
    # Low-level LFB write watchpoint
    #
    # fb_nonzero staying at 0 for a long time is ambiguous: it could mean
    # "the kernel hasn't reached fbcon/vesafb init yet" (just needs more
    # wall-clock time at ~40K instructions/sec), OR it could mean writes
    # ARE happening but resolving to the wrong physical address due to a
    # paging bug — the LFB sits at 0x0F000000 (~240MB), well outside the
    # 64MB the kernel believes it has (see the E820 map in phase5.py), so
    # the kernel MUST ioremap() it and go through a fresh page-table
    # mapping to reach it. Any bug in Memory32._translate()'s page walk
    # for that kind of mapping would silently misroute the writes.
    #
    # This hooks _translate() itself — the single chokepoint every
    # paging-aware memory access goes through — so we get a definitive
    # answer instead of guessing from a black screen.
    
    def install_lfb_watch(machine, lfb_addr=0x0F000000, lfb_size=1024*768*2):
        mem = machine.mem
        machine._lfb_watch_hits       = 0
        machine._lfb_watch_first_seen = None   # (icount, eip, vaddr, paddr)

        orig_translate = mem._translate

        def traced_translate(vaddr):
            paddr = orig_translate(vaddr)
            if lfb_addr <= paddr < lfb_addr + lfb_size:
                machine._lfb_watch_hits += 1
                if machine._lfb_watch_first_seen is None:
                    machine._lfb_watch_first_seen = (
                        machine.cpu.icount, machine.reg.ip, vaddr, paddr)
                    print(f"[lfb-watch] FIRST access resolving into LFB "
                          f"physical range: vaddr={vaddr:#x} -> "
                          f"paddr={paddr:#x} icount={machine.cpu.icount:,} "
                          f"EIP={machine.reg.ip:#010x} "
                          f"paging={mem.paging_enabled}")
            return paddr

        mem._translate = traced_translate
        print(f"[lfb-watch] installed on physical range "
              f"{lfb_addr:#x}..{lfb_addr+lfb_size:#x}")

    class VGAScene(scene.Scene):
        # Pythonista scene internals may try to set this attribute
        fixed_time_step = False

        def __init__(self, machine, ps2_mouse, ps2_kbd):
            self.machine   = machine
            self.ps2_mouse = ps2_mouse
            self.ps2_kbd   = ps2_kbd
            self.vga       = VGAReader(machine.mem)
            self._emu_thread  = None
            self._running     = True
            self._last_touch  = None
            self._tap_time    = 0
            self._tap_count   = 0
            self._drag_start  = None
            self._last_pos    = None
            self._kbd_btn_rect = None
            self._show_kbd    = False

            # Short instance tag prefixed on every console line from this
            # scene. If two runs ever end up alive at once (e.g. an old
            # Pythonista session left running in the background), their
            # interleaved [emu]/[hot-loop]/[decomp-watch] output stays
            # unambiguous instead of looking like one confusing,
            # self-contradictory timeline.
            self._tag = f'{id(self) & 0xFFFF:04x}'
            print(f"[phase6] instance tag={self._tag} starting — if you "
                  f"ever see a DIFFERENT tag interleaved in the console, "
                  f"you have two runs alive at once; stop the old one.")

            # --- VESA linear-framebuffer rendering state ---
            # Once the kernel's vesafb driver takes over (right after
            # "Ready." when booting with vga=791), it stops writing to
            # the legacy 0xB8000 text buffer entirely and draws straight
            # into the LFB at BIOS5._vesa_lfb_addr instead. Without this,
            # draw() had no code path that ever looked at that memory, so
            # the screen just stayed black forever even though Tux was
            # being rendered correctly in emulated RAM the whole time.
            self._fb_texture      = None   # cached scene image-name (str)
            self._fb_last_update  = 0.0    # throttle re-encoding
            self._fb_interval     = 0.15   # ~6-7 fps texture rebuild
            self._fb_scale        = 2      # downsample factor for speed
            self._fb_warned       = False
            self._fb_tmp_path     = None   # scratch PNG file, reused each frame
            self._fb_has_content  = False  # sticky: True once real (nonzero)
                                            # pixel data has ever appeared in
                                            # the LFB. Until then we keep
                                            # rendering the text console
                                            # instead — _vesa_mode_active
                                            # flips true early (during
                                            # real-mode video.S, well before
                                            # fbcon actually initializes),
                                            # so gating purely on that flag
                                            # blacks out the screen with zero
                                            # visibility into whether vesafb
                                            # ever actually starts drawing,
                                            # stalls, or fails to load.

        def setup(self):
            self.background_color = '#000000'
            install_lfb_watch(self.machine)
            # Start emulator thread
            self._emu_thread = threading.Thread(
                target=self._run_emu, daemon=True)
            self._emu_thread.start()

        def _run_emu(self):
            try:
                CHUNK = 200_000
                last_report = time.time()
                auto_typed = False
                video_prompt_typed = False
                AUTOTYPE_AT_ICOUNT = 600_000   # give isolinux time to
                                                # print its banner and reach
                                                # the boot: prompt, then
                                                # type the kernel name for
                                                # the user automatically —
                                                # text-matching on 'boot:'
                                                # in the console buffer
                                                # turned out to be too
                                                # timing-fragile across
                                                # different run speeds.
                VIDEO_PROMPT_AUTOTYPE_AT = 1_200_000  # the kernel's own
                                                # video mode selection
                                                # prompt (arch/i386/boot/
                                                # video.S) runs with
                                                # interrupts disabled, so
                                                # its advertised 30-second
                                                # BIOS-tick timeout can
                                                # never actually elapse —
                                                # send Space directly to
                                                # continue instead.

                # Kernel decompression output tracker. The i386 boot
                # protocol decompresses the kernel straight into
                # physical RAM starting at 0x100000 (1MB) — the same
                # address this codebase's own nuclear-bypass path uses
                # as KERNEL_ADDR. Watching nonzero-byte growth in that
                # region over time gives a DIRECT, honest signal for
                # "is decompression actually producing output" instead
                # of inferring progress indirectly from icount or EIP,
                # which look identical whether its slowly-but-really
                # progressing or genuinely stuck in a loop.
                #
                # IMPORTANT CAVEAT: decomp_out_nz==0 for a long stretch
                # is AMBIGUOUS by itself — gzip's inflate builds its
                # Huffman code tables before it ever emits an output
                # byte, so a slow (partially-accelerated) table-build
                # phase can legitimately show zero output growth for
                # millions of instructions with nothing actually wrong.
                # The EIP-range tracker below is the real tiebreaker:
                # EIP genuinely wandering across a wide span of code
                # means we're moving through different phases (healthy,
                # just slow); EIP stuck oscillating in a tiny range for
                # a long time means the emu actually stuck in one loop. 
                DECOMP_OUT_ADDR   = 0x100000
                DECOMP_OUT_SPAN   = 4 * 1024 * 1024   # sample first 4MB
                self.machine._decomp_output_nz    = 0
                self.machine._decomp_output_delta = 0
                self.machine._decomp_stall_checks = 0   # consecutive
                                                         # 3s windows with
                                                         # zero growth
                import collections
                eip_history = collections.deque(maxlen=10)   # ~30s window
                self.machine._eip_range_recent   = 0
                self.machine._eip_tight_loop     = False

                # Fine-grained pmode-entry tracer. the normal report
                # interval (~10s / ~600K instructions at this speed) is
                # too coarse to see WHERE linear drift actually starts —
                # it can only confirm "yes, drifting" after the fact
                # once several coarse samples already show a constant
                # step. This logs EIP every 100K instructions for the
                # first 3M instructions after protected mode first
                # engages, giving fine resolution right where the
                # kernel's decompressor actually starts running.
                pmode_entry_icount = None
                pmode_trace_done   = False
                PMODE_TRACE_STEP   = CHUNK   # can't resolve finer than the
                                              # outer loop's own chunk size —
                                              # EIP only changes once per
                                              # chunk boundary anyway
                PMODE_TRACE_SPAN   = 3_000_000
                next_trace_at      = None

                stuck_halt_count = 0
                while self.machine.cpu.icount < 2_000_000_000:
                    if not pmode_trace_done and self.machine.reg.protected_mode:
                        if pmode_entry_icount is None:
                            pmode_entry_icount = self.machine.cpu.icount
                            next_trace_at = pmode_entry_icount
                            print(f"[pmode-trace:{self._tag}] protected mode "
                                  f"entered at icount={pmode_entry_icount:,} "
                                  f"EIP={self.machine.reg.ip:#010x} — tracing "
                                  f"every {PMODE_TRACE_STEP:,} instructions "
                                  f"for the next {PMODE_TRACE_SPAN:,}")
                        while (next_trace_at is not None
                               and self.machine.cpu.icount >= next_trace_at):
                            print(f"[pmode-trace:{self._tag}] "
                                  f"icount={self.machine.cpu.icount:,} "
                                  f"(+{self.machine.cpu.icount - pmode_entry_icount:,}) "
                                  f"EIP={self.machine.reg.ip:#010x}")
                            next_trace_at += PMODE_TRACE_STEP
                        if (self.machine.cpu.icount - pmode_entry_icount
                                >= PMODE_TRACE_SPAN):
                            pmode_trace_done = True
                            print(f"[pmode-trace:{self._tag}] trace window done")

                    if self.machine.cpu.halted:
                        # CPU3.run() already handles HLT-and-wait-for-IRQ
                        # internally (it keeps ticking the PIT and
                        # checking for a wake-up vector each pass rather
                        # than truly stopping), so seeing halted=True at
                        # one checkpoint doesn't mean it's stuck forever —
                        # it may just need a few more chunks for a timer
                        # tick to accumulate past threshold. Only treat it
                        # as a real, permanent halt if it's STILL halted
                        # after several consecutive chunks worth of extra
                        # time to wake up on its own.
                        stuck_halt_count += 1
                        if stuck_halt_count == 1:
                            print(f"[emu:{self._tag}] CPU halted (waiting for interrupt) "
                                  f"at icount={self.machine.cpu.icount} "
                                  f"EIP={self.machine.reg.ip:#x} — giving it "
                                  f"time to wake up...")
                        if stuck_halt_count > 50:
                            print(f"[emu:{self._tag}] Still halted after extended wait "
                                  f"at icount={self.machine.cpu.icount} "
                                  f"EIP={self.machine.reg.ip:#x} — stopping.")
                            break
                    else:
                        stuck_halt_count = 0

                    if not auto_typed and self.machine.cpu.icount >= AUTOTYPE_AT_ICOUNT:
                        auto_typed = True
                        print("[emu] Auto-typing 'linux24' + Enter at the boot prompt...")
                        for ch in 'linux24':
                            self.ps2_kbd._type_char(ch)
                        self.ps2_kbd._type_char('\r')

                    if not video_prompt_typed and self.machine.cpu.icount >= VIDEO_PROMPT_AUTOTYPE_AT:
                        video_prompt_typed = True
                        print("[emu] Auto-typing Space to continue past the "
                              "kernel's video mode prompt...")
                        self.ps2_kbd.clear_bios_buffer()
                        self.ps2_kbd._type_char(' ')

                    target = self.machine.cpu.icount + CHUNK
                    self.machine.run(max_icount=target)

                    # Briefly yield after every chunk. A single 200K-
                    # instruction Python-level batch can hold the GIL
                    # long enough, especially combined with UI-thread
                    # contention, that iOS's watchdog treats Pythonista
                    # as unresponsive and kills it — independent of how
                    # much actual memory is in use. This costs a tiny
                    # amount of throughput in exchange for guaranteeing
                    # the main thread gets a real scheduling window.
                    time.sleep(0.001)

                    now = time.time()
                    if now - last_report > 3:
                        r = self.machine.reg
                        ips = CHUNK / max(now - last_report, 0.001)
                        # Sample the first 64KB of the VESA LFB (if active)
                        # so i can see the exact moment fbcon/vesafb for more suffering
                        # starts writing real pixel data (e.g. drawing
                        # the Tux boot logo) instead of just staring at
                        # a black screen and guessing whether anything
                        # is happening yet.
                        fb_info = ''
                        if getattr(self.machine.bios, '_vesa_mode_active', False):
                            try:
                                lfb = self.machine.bios._vesa_lfb_addr
                                sample = self.machine.mem._m[lfb:lfb + 65536]
                                nz = int((sample != 0).sum())
                                fb_info = f" fb_nonzero={nz}/65536"
                            except Exception:
                                fb_info = " fb_nonzero=?"
                        lfb_hits = getattr(self.machine, '_lfb_watch_hits', 0)

                        # Decompression-output progress: count nonzero
                        # bytes in the first 4MB starting at 0x100000
                        # (the standard i386 decompression target) and
                        # compare against the last check. This is a
                        # direct measurement of real forward progress,
                        # unlike icount/EIP which advance identically
                        # whether its slowly succeeding or genuinely
                        # stuck spinning in a loop.
                        decomp_info = ''
                        try:
                            sample = self.machine.mem._m[
                                DECOMP_OUT_ADDR:DECOMP_OUT_ADDR + DECOMP_OUT_SPAN]
                            nz = int((sample != 0).sum())
                            delta = nz - self.machine._decomp_output_nz
                            self.machine._decomp_output_nz    = nz
                            self.machine._decomp_output_delta = delta
                            decomp_info = f" decomp_out_nz={nz:,} (Δ{delta:+,})"

                            if not self.machine.reg.protected_mode or \
                               not getattr(self.machine.mem, 'paging_enabled', False):
                                # Only meaningful once its actually in
                                # the decompression/init window — before
                                # that this region is legitimately still
                                # all zero and that's not a stall signal.
                                pass
                            if delta == 0 and self.machine.cpu.icount > AUTOTYPE_AT_ICOUNT:
                                self.machine._decomp_stall_checks += 1
                            else:
                                self.machine._decomp_stall_checks = 0

                            if self.machine._decomp_stall_checks == 5:
                                print(f"[decomp-watch:{self._tag}] *** WARNING: zero growth in "
                                      f"decompression output region for "
                                      f"{self.machine._decomp_stall_checks * 3}s "
                                      f"straight while icount keeps advancing — "
                                      f"this looks like a genuine stuck loop, not "
                                      f"just slow progress. Worth investigating "
                                      f"phase2.py's CPU32 opcode handling around "
                                      f"EIP={r.ip:#010x}. ***")
                        except Exception as e:
                            decomp_info = f" decomp_out_nz=? ({e})"

                        # EIP-range tiebreaker. Sample current EIP into a
                        # rolling window and check how it's moving.
                        # Computed from however many samples exist so
                        # far — NOT gated on a full window — since
                        # showing 0x0 for the first ~10 reports (however
                        # long that takes at this run's speed) looks
                        # identical to "stuck" and is actively misleading.
                        #
                        # Three distinct patterns matter here:
                        #  - narrow range, EIP bouncing around inside it
                        #    -> genuinely stuck in one small loop
                        #  - wide range, EIP visiting varied addresses
                        #    -> healthy, just slow
                        #  - wide range, but EIP moving almost linearly
                        #    (near-constant step every sample) -> NOT a
                        #    loop at all. This is the signature of
                        #    execution having jumped into non-code
                        #    memory (e.g. compressed initrd data) and
                        #    the interpreter dutifully "executing"
                        #    whatever garbage bytes are there — most
                        #    random bytes decode to *some* valid-looking
                        #    x86 opcode, so it never crashes or hits the
                        #    unknown-opcode trap, it just wanders.
                        eip_history.append(self.machine.reg.ip)
                        eip_range     = 0
                        tight_loop    = False
                        linear_drift  = False
                        if len(eip_history) >= 2:
                            eip_range = max(eip_history) - min(eip_history)
                            tight_loop = (eip_range < 0x1000
                                          and len(eip_history) == eip_history.maxlen)
                        if len(eip_history) == eip_history.maxlen:
                            steps = [b - a for a, b in
                                     zip(eip_history, list(eip_history)[1:])]
                            if steps and all(s > 0 for s in steps):
                                avg = sum(steps) / len(steps)
                                if avg > 0 and all(
                                        abs(s - avg) < 0.15 * avg for s in steps):
                                    linear_drift = True
                        self.machine._eip_range_recent  = eip_range
                        self.machine._eip_tight_loop    = tight_loop
                        self.machine._eip_linear_drift  = linear_drift
                        eip_info = f" eip_range={eip_range:#x} (n={len(eip_history)})"
                        if tight_loop:
                            eip_info += " *** TIGHT LOOP — likely genuinely stuck ***"
                        if linear_drift:
                            eip_info += (" *** LINEAR DRIFT — EIP crawling at "
                                          "near-constant step through non-code "
                                          "memory, not a loop at all ***")

                        print(f"[emu:{self._tag}] icount={self.machine.cpu.icount:,} "
                              f"EIP={r.ip:#010x} pmode={r.protected_mode} "
                              f"paging={getattr(self.machine.mem,'paging_enabled',False)} "
                              f"({ips:.0f} i/s){fb_info} lfb_hits={lfb_hits}"
                              f"{decomp_info}{eip_info}")

                        # Hot-loop discovery report: the top few
                        # un-accelerated loops by hit count, with a raw
                        # byte dump. This is the artifact needed to
                        # write a new native accelerator (following the
                        # same pattern as CPU32._try_compile_loop's
                        # SIG1) for whatever's actually consuming time
                        # right now — no more guessing.
                        hot_loops = getattr(self.machine.cpu, 'hot_loops', None)
                        if hot_loops:
                            cold = [(eip, e) for eip, e in hot_loops.items()
                                    if not e.get('accelerated')]
                            cold.sort(key=lambda kv: kv[1]['hits'], reverse=True)
                            printed = set()
                            for eip, entry in cold[:3]:
                                hexdump = entry['bytes'].hex()
                                print(f"[hot-loop:{self._tag}] UNACCELERATED eip={eip:#010x} "
                                      f"hits={entry['hits']:,} "
                                      f"bytes(from {entry['start']:#010x})="
                                      f"{hexdump}")
                                printed.add(eip)
                            # Always also surface whatever the CURRENT
                            # live EIP is, even if some earlier loop has
                            # a higher cumulative hit count — this is
                            # the one we actually need bytes for right
                            # now if we're confirmed stuck.
                            cur_entry = hot_loops.get(r.ip)
                            if cur_entry and r.ip not in printed:
                                print(f"[hot-loop:{self._tag}] CURRENT EIP={r.ip:#010x} "
                                      f"hits={cur_entry['hits']:,} "
                                      f"bytes(from {cur_entry['start']:#010x})="
                                      f"{cur_entry['bytes'].hex()}")
                        last_report = now
            except Exception as e:
                print(f"[emu:{self._tag}] crashed: {e}")
                import traceback; traceback.print_exc()

        def update(self):
            # Redraw every frame (scene calls this at ~60fps but we
            # throttle actual VGA reads to ~15fps via draw())
            pass

        
        # VESA LFB reading / rendering
        
        def _read_lfb_rgb(self, scale=2):
            """
            Read the VESA linear framebuffer (RGB565, set up by
            BIOS5._int10_vga's AH=0x4F handler) out of emulated RAM and
            return a downsampled HxWx3 uint8 numpy array, plus the
            (width, height) of that downsampled image.

            Returns (None, 0, 0) if VESA mode isn't active or the LFB
            address would run past the end of emulated memory.
            """
            bios = self.machine.bios
            if not getattr(bios, '_vesa_mode_active', False):
                return None, 0, 0

            xres = getattr(bios, '_vesa_xres', 1024)
            yres = getattr(bios, '_vesa_yres', 768)
            lfb  = getattr(bios, '_vesa_lfb_addr', 0x0F000000)
            mem  = self.machine.mem

            nbytes = xres * yres * 2   # RGB565 = 2 bytes/pixel
            if lfb < 0 or lfb + nbytes > mem.size:
                if not self._fb_warned:
                    print(f"[fb] LFB range {lfb:#x}..{lfb+nbytes:#x} "
                          f"exceeds emulated RAM size {mem.size:#x} — "
                          f"cannot render framebuffer")
                    self._fb_warned = True
                return None, 0, 0

            import numpy as np
            raw = np.frombuffer(mem._m[lfb:lfb + nbytes].tobytes(),
                                 dtype='<u2').reshape(yres, xres)
            if scale > 1:
                raw = raw[::scale, ::scale]

            # RGB565 -> RGB888
            r = ((raw >> 11) & 0x1F).astype(np.uint8)
            g = ((raw >> 5)  & 0x3F).astype(np.uint8)
            b = ( raw        & 0x1F).astype(np.uint8)
            r = (r << 3) | (r >> 2)
            g = (g << 2) | (g >> 4)
            b = (b << 3) | (b >> 2)
            rgb = np.dstack([r, g, b])
            h, w = rgb.shape[0], rgb.shape[1]
            return rgb, w, h

        def _update_fb_texture(self):
            """Throttled read of the VESA LFB + texture rebuild. Sets
            self._fb_has_content (sticky) the first time real, nonzero
            pixel data is found — this is what draw() uses to decide
            whether it's worth switching away from the text console."""
            now = time.time()
            if now - self._fb_last_update <= self._fb_interval:
                return
            self._fb_last_update = now

            try:
                rgb, fw, fh = self._read_lfb_rgb(scale=self._fb_scale)
            except Exception as e:
                rgb = None
                print(f"[fb] read error: {e}")

            if rgb is None or fw <= 0 or fh <= 0:
                return

            # numpy vectorized check — cheap even at full 1024x768x2
            # bytes, and we only run this at most once every
            # self._fb_interval seconds anyway.
            if not self._fb_has_content and bool(rgb.any()):
                self._fb_has_content = True
                print("[fb] real pixel data detected in LFB — "
                      "switching display to framebuffer view")

            if not self._fb_has_content:
                # Don't bother building/encoding a texture for an
                # all-black frame — nothing to show yet, and draw()
                # won't render it anyway while _fb_has_content is False.
                return

            try:
                from PIL import Image
                import io
                pil_img = Image.fromarray(rgb, 'RGB')

                # scene.image() in this Pythonista build ONLY accepts a
                # string image identifier — not a ui.Image, not a
                # scene.Texture. The documented way to feed it a
                # dynamically-generated image is to write it to a file
                # and register it via scene.load_image_file(path), which
                # hands back the opaque string key scene.image() wants.
                # reuse a single scratch file across frames so this
                # doesn't leak temp files while booting.
                if self._fb_tmp_path is None:
                    import tempfile
                    fd, self._fb_tmp_path = tempfile.mkstemp(
                        suffix='.png', prefix='x86emu_fb_')
                    os.close(fd)

                pil_img.save(self._fb_tmp_path, format='PNG')
                self._fb_texture = scene.load_image_file(self._fb_tmp_path)
            except Exception as e:
                print(f"[fb] render/encode error: {e}")

        def _render_fb(self, W, H):
            """Draw the current framebuffer texture, letterboxed to
            preserve the guest's aspect ratio."""
            scene.background(0, 0, 0)
            if self._fb_texture is not None:
                bios = self.machine.bios
                xres = getattr(bios, '_vesa_xres', 1024)
                yres = getattr(bios, '_vesa_yres', 768)
                src_aspect = xres / float(yres)
                dst_aspect = W / float(H)
                if dst_aspect > src_aspect:
                    draw_h = H
                    draw_w = H * src_aspect
                else:
                    draw_w = W
                    draw_h = W / src_aspect
                ox = (W - draw_w) / 2.0
                oy = (H - draw_h) / 2.0
                # self._fb_texture is a string key returned by
                # scene.load_image_file() (see _update_fb_texture above)
                # — that's the only form scene.image() accepts here.
                scene.image(self._fb_texture, ox, oy, draw_w, draw_h)
            else:
                scene.fill(0, 1, 0)
                scene.text('Framebuffer active, rendering...', VGA_FONT, 16, W/2, H/2)

        
        # Boot splash — shown for every stage before real Tux pixels
        # exist, so waiting on the (currently unaccelerated) kernel
        # decompression stage is watchable instead of blank/garbled.
        
        def _boot_stage_info(self):
            """Returns (stage_text, progress_fraction, is_stalled). Most
            stages are illustrative markers of which real, distinct boot
            phase its in (there is no ground truth % for most of them).
            The decompression stage is the exception — its progress is
            driven by measured nonzero-byte growth in the actual
            decompression output region (0x100000+), tracked by
            _run_emu's periodic sampling, so the bar visibly creeps
            forward with real evidence instead of sitting frozen for
            however long this stage takes."""
            reg  = self.machine.reg
            mem  = self.machine.mem
            bios = self.machine.bios

            pmode        = reg.protected_mode
            paging       = getattr(mem, 'paging_enabled', False)
            vesa_on      = getattr(bios, '_vesa_mode_active', False)
            lfb_hits     = getattr(self.machine, '_lfb_watch_hits', 0)
            no_growth    = getattr(self.machine, '_decomp_stall_checks', 0) >= 5
            tight_loop   = getattr(self.machine, '_eip_tight_loop', False)
            linear_drift = getattr(self.machine, '_eip_linear_drift', False)
            # Genuinely stuck requires no_growth PLUS one of two distinct
            # EIP patterns: bouncing in a tiny range (real tight loop),
            # or crawling at a near-constant step through a wide range
            # (execution likely wandered into non-code memory — data
            # being misinterpreted as instructions). no_growth alone is
            # ambiguous since gzip builds Huffman tables before emitting
            # any output byte, which can legitimately take a long time.
            stalled = no_growth and (tight_loop or linear_drift)

            if not pmode:
                return "BIOS / isolinux — loading kernel image", 0.08, False
            if not paging:
                nz = getattr(self.machine, '_decomp_output_nz', 0)
                # 4MB sampled span — treat ~2MB of nonzero output as
                # "essentially done decompressing" for bar purposes.
                frac = 0.10 + 0.45 * min(1.0, nz / 2_000_000.0)
                if no_growth and linear_drift:
                    label = "STUCK — EIP drifting into non-code memory (real bug)"
                elif no_growth and tight_loop:
                    label = "STUCK — EIP trapped in a tight loop (real bug)"
                elif no_growth:
                    label = "Building Huffman tables (no output yet — checking...)"
                else:
                    label = "Decompressing kernel (this is the slow part)"
                return label, frac, stalled
            if not vesa_on:
                return "Kernel initializing...", 0.55, False
            if lfb_hits == 0:
                return "Negotiating VESA video mode...", 0.70, False
            if not self._fb_has_content:
                return "Framebuffer active — waiting for first pixels...", 0.88, False
            return "Almost there...", 0.97, False

        def _draw_boot_splash(self, W, H):
            stage_text, progress, stalled = self._boot_stage_info()
            t = time.time()

            card_w = min(560, W * 0.62)
            card_h = 226
            cx = W / 2.0
            cy = H * 0.38
            card_x = cx - card_w / 2.0
            card_y = cy - card_h / 2.0

            scene.fill(0.06, 0.07, 0.13, 0.90)
            scene.rect(card_x, card_y, card_w, card_h)
            accent = (0.85, 0.30, 0.25, 0.8) if stalled else (0.25, 0.55, 0.85, 0.6)
            scene.fill(*accent)
            scene.rect(card_x, card_y, card_w, 3)   # top accent line

            # Bouncing penguin placeholder — a fun stand-in for the real
            # Tux logo while we wait for the actual framebuffer content.
            bounce = math.sin(t * 2.2) * 8
            scene.fill(1, 1, 1)
            scene.text('\U0001F427', VGA_FONT, 52, cx, card_y + 58 + bounce)

            scene.fill(0.85, 0.9, 1.0)
            scene.text('x86emu — Booting Damn Small Linux',
                       VGA_FONT, 15, cx, card_y + 108)

            dots = '.' * (int(t * 2) % 4)
            scene.fill(0.85, 0.4, 0.35) if stalled else scene.fill(0.55, 0.85, 0.6)
            scene.text(stage_text + dots, VGA_FONT, 12, cx, card_y + 130)

            # Progress bar — for the decompression stage this is driven
            # by real measured output growth (see _boot_stage_info); for
            # other stages it's a fixed marker for that stage.
            bar_w = card_w - 60
            bar_h = 8
            bar_x = cx - bar_w / 2.0
            bar_y = card_y + 154
            scene.fill(0.18, 0.18, 0.24)
            scene.rect(bar_x, bar_y, bar_w, bar_h)
            scene.fill(0.85, 0.3, 0.25) if stalled else scene.fill(0.30, 0.70, 0.95)
            scene.rect(bar_x, bar_y, bar_w * progress, bar_h)

            icount    = self.machine.cpu.icount
            lfb_hits  = getattr(self.machine, '_lfb_watch_hits', 0)
            decomp_nz = getattr(self.machine, '_decomp_output_nz', 0)
            scene.fill(0.5, 0.5, 0.55)
            scene.text(f'{icount:,} instr   decomp_out={decomp_nz:,}B   lfb_hits={lfb_hits}',
                       VGA_FONT, 10, cx, card_y + 180)

            hot_loops = getattr(self.machine.cpu, 'hot_loops', None)
            if hot_loops:
                cold = [(e, entry) for e, entry in hot_loops.items()
                        if not entry.get('accelerated')]
                if cold:
                    cold.sort(key=lambda kv: kv[1]['hits'], reverse=True)
                    top_eip, top_entry = cold[0]
                    scene.fill(0.6, 0.6, 0.4)
                    scene.text(f'hottest loop: {top_eip:#010x}  '
                               f'({top_entry["hits"]:,} hits, unaccelerated)',
                               VGA_FONT, 10, cx, card_y + 196)

        def draw(self):
            W, H = self.size

            
            # VESA linear-framebuffer path. DSL's vesafb/fbcon driver
            # draws Tux and the rest of the graphical console straight
            # into the LFB, but _vesa_mode_active flips true early
            # (during real-mode video.S, well before fbcon actually
            # initializes) — so we don't switch the display over until
            # we've actually SEEN real pixel data land there. Until then
            # we fall through to the normal text-console rendering below,
            # so real kernel boot messages (and any vesafb failure) stay
            # visible instead of just showing black with no information.
            
            vesa_active = getattr(self.machine.bios, '_vesa_mode_active', False)
            if vesa_active:
                self._update_fb_texture()

            if vesa_active and self._fb_has_content:
                self._render_fb(W, H)

                # Keep the keyboard button available on top of the
                # framebuffer view too.
                btn_size = 44
                btn_x = 8; btn_y = H - btn_size - 4
                self._kbd_btn_rect = (btn_x, btn_y, btn_size, btn_size)
                scene.fill(0.2, 0.2, 0.2, 0.9)
                scene.rect(btn_x, btn_y, btn_size, btn_size)
                scene.fill(1, 1, 1)
                scene.text('⌨', VGA_FONT, 24, btn_x + btn_size/2, btn_y + btn_size/2)

                icount = self.machine.cpu.icount
                scene.fill(0.4, 0.4, 0.4)
                scene.text(f'{icount//1000}K | vesa {getattr(self.machine.bios,"_vesa_xres",0)}x'
                           f'{getattr(self.machine.bios,"_vesa_yres",0)}',
                           VGA_FONT, 10, W/2, H - 10)
                return

            scene.background(0, 0, 0)

            mem = self.machine.mem
            VGA_BASE = 0xB8000
            COLS = 80; ROWS = 25

            # Check if VGA has any content
            has_content = False
            for i in range(0, ROWS * COLS * 2, 2):
                if mem.read8_flat(VGA_BASE + i) not in (0, 0x20):
                    has_content = True
                    break

            char_w = W / 80.0
            char_h = (H - 20) / 25.0
            font_size = max(8, min(char_w * 1.4, char_h * 0.85))

            if has_content:
                # Render VGA text buffer
                for row in range(ROWS):
                    for col in range(COLS):
                        off  = VGA_BASE + (row * COLS + col) * 2
                        ch   = mem.read8_flat(off)
                        attr = mem.read8_flat(off + 1)
                        fg   = attr & 0x0F
                        bg   = (attr >> 4) & 0x07
                        x = col * char_w
                        y = row * char_h
                        if bg != 0:
                            r_c, g_c, b_c = self.vga.PALETTE16[bg]
                            scene.fill(r_c/255, g_c/255, b_c/255)
                            scene.rect(x, y, char_w + 0.5, char_h + 0.5)
                        if ch and ch != 0x20:
                            try:
                                display_ch = chr(ch) if 0x20 <= ch < 0x7F else '?'
                                r_c, g_c, b_c = self.vga.PALETTE16[fg]
                                scene.fill(r_c/255, g_c/255, b_c/255)
                                scene.text(display_ch, VGA_FONT, font_size,
                                           x + char_w * 0.5, y + char_h * 0.5)
                            except Exception:
                                pass
            else:
                # Black screen — kernel hasn't written to VGA text buffer yet.
                # Decompression is instant now (Python zlib), so if its
                # here it means the kernel is executing real init code,
                # possibly in high-half virtual addresses once paging is on.
                icount = self.machine.cpu.icount
                pmode  = self.machine.reg.protected_mode
                paging = getattr(self.machine.mem, 'paging_enabled', False)
                eip    = self.machine.reg.ip

                scene.fill(0, 1, 0)   # green text on black
                if paging:
                    scene.text('Kernel running (paging active)...', VGA_FONT, 16, W/2, H/2 - 20)
                    scene.text(f'{icount//1_000_000}M instructions', VGA_FONT, 12, W/2, H/2)
                    scene.text(f'EIP={eip:#010x} (virtual)', VGA_FONT, 10, W/2, H/2 + 20)
                    scene.text('(kernel init in progress — please wait)', VGA_FONT, 10, W/2, H/2 + 40)
                elif pmode:
                    scene.text('Kernel starting (protected mode)...', VGA_FONT, 16, W/2, H/2 - 20)
                    scene.text(f'{icount//1_000_000}M instructions', VGA_FONT, 12, W/2, H/2)
                    scene.text(f'EIP={eip:#010x}', VGA_FONT, 10, W/2, H/2 + 20)
                    scene.text('(setting up paging / GDT)', VGA_FONT, 10, W/2, H/2 + 40)
                else:
                    scene.text('Booting DSL Linux...', VGA_FONT, 16, W/2, H/2)
                    scene.text(f'{icount//1000}K instructions', VGA_FONT, 12, W/2, H/2 + 20)

            # Cursor blink
            if has_content:
                try:
                    import time
                    cur_col = mem.read8_flat(0x450)
                    cur_row = mem.read8_flat(0x451)
                    if int(time.time() * 2) % 2 == 0:
                        scene.fill(1, 1, 1, 0.8)
                        scene.rect(cur_col * char_w, cur_row * char_h + char_h - 3, char_w, 2)
                except Exception:
                    pass

            # Boot splash — a friendly animated overlay covering the
            # middle of the screen (deliberately away from the bottom
            # strip where raw boot text tends to cluster) so there's
            # something engaging to watch during the slow, unaccelerated
            # kernel-decompression stage instead of garbled text or
            # blank black.
            self._draw_boot_splash(W, H)

            # Keyboard button bottom-left
            btn_size = 44
            btn_x = 8; btn_y = H - btn_size - 4
            self._kbd_btn_rect = (btn_x, btn_y, btn_size, btn_size)
            scene.fill(0.2, 0.2, 0.2, 0.9)
            scene.rect(btn_x, btn_y, btn_size, btn_size)
            scene.fill(1, 1, 1)
            scene.text('⌨', VGA_FONT, 24, btn_x + btn_size/2, btn_y + btn_size/2)

            # Status bar
            icount = self.machine.cpu.icount
            mode = 'pmode' if self.machine.reg.protected_mode else 'rmode'
            scene.fill(0.4, 0.4, 0.4)
            scene.text(f'{icount//1000}K | {mode}', VGA_FONT, 10, W/2, H - 10)

            # Live LFB diagnostic — shown on-screen (not just console) so
            # it's visible while recording. vesa_on: has the guest ever
            # set VESA mode (via real-mode video.S). lfb_hits: total
            # memory accesses that have ever resolved to the LFB physical
            # range, from install_lfb_watch()'s hook on _translate() —
            # this is the truth
            #signal for "is anything actually
            # touching that memory yet," independent of pixel content.
            vesa_on  = getattr(self.machine.bios, '_vesa_mode_active', False)
            lfb_hits = getattr(self.machine, '_lfb_watch_hits', 0)
            halted   = self.machine.cpu.halted
            scene.fill(1.0, 0.3, 0.3) if halted else scene.fill(0.5, 0.5, 0.5)
            scene.text(f'vesa_on={vesa_on}  lfb_hits={lfb_hits}  '
                       f'halted={halted}',
                       VGA_FONT, 10, W/2, H - 26)

        def touch_began(self, touch):
            now = time.time()
            pos = touch.location

            # Check keyboard button
            if self._kbd_btn_rect:
                bx, by, bw, bh = self._kbd_btn_rect
                if bx <= pos.x <= bx+bw and by <= pos.y <= by+bh:
                    self._show_keyboard_input()
                    return

            # Track for tap detection
            if now - self._tap_time < 0.35:
                self._tap_count += 1
            else:
                self._tap_count = 1
            self._tap_time   = now
            self._drag_start = pos
            self._last_pos   = pos

        def touch_moved(self, touch):
            if self._last_pos is None:
                return
            pos  = touch.location
            dx   = pos.x - self._last_pos.x
            dy   = pos.y - self._last_pos.y
            # Scale: screen pixels → mouse mickeys
            # Screen is ~375px wide for 320 virtual pixels
            scale = 2.0
            self.ps2_mouse.move(dx * scale, dy * scale)
            self._last_pos = pos

        def touch_ended(self, touch):
            pos  = touch.location
            now  = time.time()

            if self._drag_start:
                dx = abs(pos.x - self._drag_start.x)
                dy = abs(pos.y - self._drag_start.y)
                moved = (dx**2 + dy**2) ** 0.5

                if moved < 10:   # tap, not drag
                    if self._tap_count >= 2:
                        # Double tap = left click
                        self.ps2_mouse.click_left()
                    else:
                        # Single tap = right click
                        self.ps2_mouse.click_right()

            self._drag_start = None
            self._last_pos   = None

        def _show_keyboard_input(self):
            """Toggle keyboard input. Uses a background thread with console dialog."""
            if getattr(self, '_kbd_overlay_active', False):
                self._kbd_overlay_active = False
                return
            self._kbd_overlay_active = True

            ps2_kbd = self.ps2_kbd

            def _kb_thread():
                try:
                    import console
                    while self._kbd_overlay_active:
                        try:
                            text = console.input_alert(
                                'VM Keyboard',
                                'Type below — each submission sends text + Enter',
                                '', 'Send', hide_cancel_button=False)
                            if text is None:
                                break
                            for ch in text:
                                ps2_kbd._type_char(ch)
                            ps2_kbd._type_char('\r')
                            self._user_typed = True
                        except KeyboardInterrupt:
                            break
                        except Exception:
                            break
                except Exception:
                    pass
                self._kbd_overlay_active = False

            import threading
            threading.Thread(target=_kb_thread, daemon=True).start()

        def _dismiss_keyboard(self):
            self._kbd_overlay_active = False

        def did_stop(self):
            if self._fb_tmp_path:
                try:
                    os.remove(self._fb_tmp_path)
                except OSError:
                    pass



# Main entry point

def launch(iso_path=None, machine=None):
    """
    Launch Phase 6 UI.

    Either pass an already-created Machine5, or an iso_path to create one.

    On Pythonista: opens a Scene with VGA display + touch input.
    Elsewhere: runs headless with console output.

    Example (Pythonista):
        import phase6, os
        phase6.launch(os.path.expanduser('~/Documents/Linux emu/dsl.iso'))
    """
    import importlib

    if machine is None:
        if iso_path is None:
            iso_path = os.path.expanduser('~/Documents/Linux emu/dsl.iso')
        p5 = importlib.import_module('phase5')
        print(f"[phase6] Creating Machine5 for {iso_path}")
        machine = p5.Machine5(iso_path)
        machine.load_boot_image()

    # Wire up PS/2 devices
    ps2_kbd   = PS2Keyboard(machine.pic, machine.mem)
    ps2_mouse = PS2Mouse(machine.pic)
    kbd_ctrl  = KBD6(ps2_kbd, ps2_mouse)

    # Patch the machine's IO to use our KBD6 controller
    machine.io.kbd = kbd_ctrl

    # Wire keyboard into BIOS INT 16h handler
    machine.bios._kbd_controller = kbd_ctrl

    if IN_PYTHONISTA:
        print("[phase6] Launching Pythonista VGA Scene...")
        vga_scene = VGAScene(machine, ps2_mouse, ps2_kbd)
        try:
            scene.run(vga_scene, scene.LANDSCAPE)
        except TypeError:
            # Older Pythonista scene.run signature
            scene.run(vga_scene)
    else:
        runner = HeadlessRunner(machine)
        runner.run()

    return machine

# Pythonista quick-launch shortcut
if __name__ == '__main__':
    ISO_PATH = os.path.expanduser('~/Documents/Linux emu/dsl.iso')
    if len(sys.argv) > 1:
        ISO_PATH = sys.argv[1]
    launch(iso_path=ISO_PATH)
