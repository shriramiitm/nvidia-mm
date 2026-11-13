#!/usr/bin/env python3
"""Draw the explanatory diagrams used in docs/report.md as SVG files.

Usage:  python3 scripts/make_figures.py        -> docs/figures/*.svg
"""
import os
from html import escape

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, LINE, FILL, SURF = "#0b0b0b", "#52514e", "#8a8984", "#c9c8c2", "#f1f0ec", "#fcfcfb"
FONT = "font-family='Helvetica, Arial, sans-serif'"
OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.p = w, h, []

    def rect(self, x, y, w, h, fill=FILL, stroke=LINE, sw=1, rx=0, op=1):
        self.p.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}' "
                      f"fill-opacity='{op}' stroke='{stroke}' stroke-width='{sw}'/>")

    def line(self, x1, y1, x2, y2, stroke=MUTED, sw=1.5, dash=None, arrow=False):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        m = " marker-end='url(#arr)'" if arrow else ""
        self.p.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{stroke}' "
                      f"stroke-width='{sw}'{d}{m}/>")

    def text(self, x, y, s, size=12, fill=INK2, anchor="start", weight="400"):
        self.p.append(f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' fill='{fill}' "
                      f"text-anchor='{anchor}' font-weight='{weight}'>{escape(s)}</text>")

    def save(self, name):
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, name), "w") as f:
            f.write(f"<svg xmlns='http://www.w3.org/2000/svg' width='{self.w}' height='{self.h}' "
                    f"viewBox='0 0 {self.w} {self.h}'>\n<defs><marker id='arr' viewBox='0 0 10 10' refX='9' "
                    f"refY='5' markerWidth='7' markerHeight='7' orient='auto'><path d='M0,0 L10,5 L0,10 z' "
                    f"fill='{MUTED}'/></marker></defs>\n<rect width='100%' height='100%' fill='{SURF}'/>\n"
                    + "\n".join(self.p) + "\n</svg>\n")


def fig_tiling():
    """Block/thread mapping of the tiled kernel and the walk along K."""
    s = Svg(820, 400)
    s.text(20, 28, "Tiled kernel: one block = one T×T tile of C, one thread = one element", 15, INK, weight="600")
    s.text(20, 48, "The block walks along K in steps of T, loading one tile of A and one of B into shared memory per step.", 12)
    # A: M x K
    ax, ay, aw, ah, T = 40, 150, 240, 160, 40
    s.rect(ax, ay, aw, ah)
    s.text(ax + aw / 2, ay + ah + 20, "A (M × K)", 12, INK, "middle", "600")
    for t in range(aw // T):
        s.rect(ax + t * T, ay + 60, T, T, fill=BLUE if t == 2 else "#d6e6f8", stroke=BLUE, op=0.85)
    s.text(ax + 2 * T + T / 2, ay + 60 + T / 2 + 4, "t", 12, "#ffffff", "middle", "600")
    s.text(ax - 6, ay + 60 + T / 2 + 4, "rows of", 10, INK2, "end")
    s.text(ax - 6, ay + 60 + T / 2 + 16, "block", 10, INK2, "end")
    # B: K x N
    bx, by, bw, bh = 340, 70, 200, 240
    s.rect(bx, by, bw, bh)
    s.text(bx + bw / 2, by - 8, "B (K × N)", 12, INK, "middle", "600")
    for t in range(bh // T):
        s.rect(bx + 120, by + t * T, T, T, fill=ORANGE if t == 2 else "#fbdccf", stroke=ORANGE, op=0.85)
    s.text(bx + 120 + T / 2, by + 2 * T + T / 2 + 4, "t", 12, "#ffffff", "middle", "600")
    # C
    cx, cy, cw, ch = 560, 150, 160, 160
    s.rect(cx, cy, cw, ch)
    s.text(cx + cw / 2, cy + ch + 20, "C (M × N)", 12, INK, "middle", "600")
    s.rect(cx + 60, cy + 60, T, T, fill=AQUA, stroke=AQUA)
    n = 5
    for i in range(1, n):
        s.line(cx + 60 + i * T / n, cy + 60, cx + 60 + i * T / n, cy + 60 + T, "#ffffff", 0.7)
        s.line(cx + 60, cy + 60 + i * T / n, cx + 60 + T, cy + 60 + i * T / n, "#ffffff", 0.7)
    s.text(cx + 60 + T / 2, cy + 52, "block (bx, by)", 10, INK2, "middle")
    s.line(cx + 60 + T + 30, cy + 60 + T / 2 + 18, cx + 60 + T + 4, cy + 60 + T / 2 + 2, arrow=True)
    s.text(cx + 60 + T + 32, cy + 60 + T / 2 + 22, "thread (tx, ty)", 10, INK2)
    s.text(cx + 60 + T + 32, cy + 60 + T / 2 + 34, "→ C[row][col]", 10, INK2)
    # equations
    s.text(20, 352, "row = by·T + ty,   col = bx·T + tx      (tx varies fastest inside a warp → neighbouring columns)", 12, INK)
    s.text(20, 372, "step t:  As[ty][tx] = A[row][t·T + tx],   Bs[ty][tx] = B[t·T + ty][col]   (0 outside the matrix)", 12, INK)
    s.text(20, 392, "then __syncthreads();  acc += Σk As[ty][k]·Bs[k][tx];  __syncthreads();   C[row][col] = acc if inside", 12, INK)
    s.save("tiling.svg")


def fig_global_access():
    """Which global addresses one warp touches: naive vs tiled (T = 32)."""
    s = Svg(820, 330)
    s.text(20, 28, "Global-memory access of one warp (32 threads), one load instruction", 15, INK, weight="600")
    s.text(20, 48, "Squares = consecutive floats of one matrix row; brackets = 32-byte sectors fetched; colour = bytes actually used.", 12)

    def row(y, label, used, sectors, note, color):
        s.text(20, y + 14, label, 12, INK, weight="600")
        x0, cw = 200, 14
        for i in range(40):
            s.rect(x0 + i * cw, y, cw - 1, 18, fill=color if i in used else FILL, stroke=LINE, sw=0.5)
        for a, b in sectors:
            s.line(x0 + a * cw, y + 24, x0 + (b + 1) * cw - 1, y + 24, color, 2.5)
        s.text(x0, y + 42, note, 11, INK2)

    row(80, "tiled: A or B row", set(range(32)), [(0, 7), (8, 15), (16, 23), (24, 31)],
        "32 neighbouring floats (tx = 0..31) → 4 full sectors, 100% of fetched bytes used (measured 100%)", BLUE)
    row(150, "naive: B row", set(range(16)), [(0, 7), (8, 15)],
        "warp = 16 × 2 threads; both thread rows read the same 16 floats → 2 full sectors", ORANGE)
    row(220, "naive: A", {0, 33}, [(0, 7), (32, 39)],
        "2 thread rows read 2 different A rows (1 float each, broadcast) → 2 sectors for 8 useful bytes", ORANGE)
    s.text(20, 300, "naive efficiency = (64 + 8) bytes used / (64 + 64) fetched = 56.25%  — exactly the value Nsight Compute reports", 12, INK)
    s.text(20, 318, "If threadIdx.x mapped to rows instead, every thread would hit its own sector: 4 of 32 bytes used (12.5%).", 12, INK2)
    s.save("global_access.svg")


def fig_banks():
    """Bank mapping for row access, column access, and padded column access (8 banks for legibility)."""
    s = Svg(820, 360)
    s.text(20, 28, "Shared-memory banks: why padding helps a column access, and why tiled doesn't need it", 15, INK, weight="600")
    s.text(20, 48, "Drawn with 8 banks and an 8-wide tile for legibility; the T4 has 32 banks and the pattern is the same at 32 × 32.", 12)
    B = 8
    cell = 24

    def panel(x0, title, ld, highlight, sub):
        s.text(x0, 82, title, 12, INK, weight="600")
        for r in range(6):
            for c in range(ld):
                bank = (r * ld + c) % B
                pad = c >= B
                hl = (r, c) in highlight
                fill = "#ffffff" if pad else (BLUE if hl else FILL)
                s.rect(x0 + c * cell, 94 + r * cell, cell - 2, cell - 2, fill=fill,
                       stroke=MUTED if pad else LINE, sw=0.8)
                s.text(x0 + c * cell + cell / 2 - 1, 94 + r * cell + 17, "pad" if pad else str(bank), 9 if pad else 11,
                       "#ffffff" if hl else (MUTED if pad else INK2), "middle")
        for i, line in enumerate(sub):
            s.text(x0, 262 + 16 * i, line, 11, INK2)

    panel(20, "row access, row length 8", 8, {(2, c) for c in range(8)},
          ["tiled: As[ty][tx], Bs[k][tx]", "8 threads → banks 0..7", "1 pass, no conflict"])
    panel(275, "column access, row length 8", 8, {(r, 3) for r in range(6)},
          ["tiled_ta: As[tx][ty]", "every row hits bank 3", "6-way conflict (32-way at 32 × 32)"])
    panel(530, "column access, row length 9", 9, {(r, 3) for r in range(6)},
          ["padded_ta: As[T][T+1]", "each row shifts one bank", "1 pass, no conflict"])
    s.text(20, 344, "Cost of +1: rows are no longer 16-byte aligned, so 4 neighbouring floats can't be read with one 128-bit load (LDS.128). "
           "regtile / vec pad by 4 instead.", 11, INK)
    s.save("banks.svg")


def fig_regtile():
    """Register tiling: per-thread 4x4 outer product from two 128-bit shared loads."""
    s = Svg(820, 330)
    s.text(20, 28, "Register tiling (regtile): each thread computes a 4 × 4 block of C", 15, INK, weight="600")
    s.text(20, 48, "Block = 16 × 16 threads → 64 × 64 tile of C; K step = 8. Per k: 2 shared loads (128-bit), 16 FMAs in registers.", 12)
    # As transposed: 8 x 64 (show 8 rows x 16 groups)
    x0, y0 = 30, 90
    s.text(x0, y0 - 10, "As[k][m] (A tile, transposed, row length 64 + 4)", 11, INK, weight="600")
    for k in range(8):
        for g in range(16):
            hl = (k == 3 and g == 5)
            s.rect(x0 + g * 14, y0 + k * 14, 13, 13, fill=BLUE if hl else FILL, stroke=LINE, sw=0.5)
    s.text(x0, y0 + 8 * 14 + 16, "k = 3, rows ty·4 .. ty·4+3 → one float4  a[0..3]", 11, INK2)
    # Bs: 8 x 64
    x1 = 300
    s.text(x1, y0 - 10, "Bs[k][n] (B tile, row length 64)", 11, INK, weight="600")
    for k in range(8):
        for g in range(16):
            hl = (k == 3 and g == 9)
            s.rect(x1 + g * 14, y0 + k * 14, 13, 13, fill=ORANGE if hl else FILL, stroke=LINE, sw=0.5)
    s.text(x1, y0 + 8 * 14 + 16, "k = 3, cols tx·4 .. tx·4+3 → one float4  b[0..3]", 11, INK2)
    # outer product
    x2, y2, c = 600, 128, 26
    s.text(x2 - 20, y0 - 10, "acc[i][j] += a[i] · b[j]", 11, INK, weight="600")
    for i in range(4):
        s.rect(x2 - 30, y2 + i * c, c - 3, c - 3, fill=BLUE, stroke=BLUE)
        s.text(x2 - 30 + (c - 3) / 2, y2 + i * c + 16, f"a{i}", 10, "#ffffff", "middle")
        s.rect(x2 + i * c, y2 - 30, c - 3, c - 3, fill=ORANGE, stroke=ORANGE)
        s.text(x2 + i * c + (c - 3) / 2, y2 - 14, f"b{i}", 10, "#ffffff", "middle")
        for j in range(4):
            s.rect(x2 + j * c, y2 + i * c, c - 3, c - 3, fill=AQUA, stroke=AQUA, op=0.8)
    s.text(x2 - 30, y2 + 4 * c + 18, "16 accumulators in registers", 11, INK2)
    s.text(20, 280, "Shared loads per multiply-add:  tiled 1.25 (scalar)  →  regtile 2 / 16 = 0.125 (128-bit)  →  vec128 (8 × 8) 4 / 64 = 0.0625", 12, INK)
    s.text(20, 300, "Each A value is reused 4× (TN) and each B value 4× (TM) from registers instead of being re-read from shared memory.", 12, INK2)
    s.text(20, 318, "Padding As by 4 makes the transposed store conflict-free and keeps rows 16-byte aligned for the float4 reads.", 12, INK2)
    s.save("regtile.svg")


if __name__ == "__main__":
    fig_tiling()
    fig_global_access()
    fig_banks()
    fig_regtile()
    print("wrote docs/figures/: tiling.svg global_access.svg banks.svg regtile.svg")
