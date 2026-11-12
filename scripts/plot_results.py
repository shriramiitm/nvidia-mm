#!/usr/bin/env python3
"""Make the report plots as standalone SVG files (no matplotlib / numpy needed).

Usage:  python3 scripts/plot_results.py [results]      -> results/plots/*.svg

Inputs:  results/results.csv           (scripts/bench.sh)
         results/profile_summary.csv   (scripts/summarize_profile.py)

Every mark carries an SVG <title>, so hovering a bar or point in a browser shows
its exact value. The numbers are also in the tables in docs/week*.md.
"""
import csv
import os
import sys
from html import escape

# Validated categorical slots (light surface): blue, orange, aqua.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#9cc3ef", "#4f93e0", "#1d5aa6"]  # one hue, light -> dark (ordinal: tile 8 / 16 / 32)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
FONT = "font-family='Helvetica, Arial, sans-serif'"

SHAPES = [(256, 256, 256), (1024, 1024, 1024), (4096, 4096, 4096), (2048, 512, 4096), (1000, 513, 769)]
LABEL = {s: f"{s[0]}×{s[1]}×{s[2]}" for s in SHAPES}


def nice_max(v):
    for mag in (0.1, 1, 10, 100, 1000, 10000):
        for step in (1, 2, 2.5, 5):
            if step * mag * 5 >= v:
                return step * mag * 5, step * mag
    return v, v / 5


class Svg:
    def __init__(self, w, h, title, subtitle):
        self.w, self.h, self.parts = w, h, []
        self.text(24, 30, title, 16, INK, weight="600")
        self.text(24, 50, subtitle, 12, INK2)

    def text(self, x, y, s, size=11, fill=INK2, anchor="start", weight="400", rotate=None):
        tr = f" transform='rotate({rotate} {x} {y})'" if rotate else ""
        self.parts.append(f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' fill='{fill}' "
                          f"text-anchor='{anchor}' font-weight='{weight}'{tr}>{escape(s)}</text>")

    def raw(self, s):
        self.parts.append(s)

    def legend(self, items, x, y):
        for label, color in items:
            self.raw(f"<rect x='{x}' y='{y - 9}' width='10' height='10' rx='2' fill='{color}'/>")
            self.text(x + 15, y, label, 11, INK2)
            x += 22 + 6.5 * len(label)

    def save(self, path):
        body = "\n".join(self.parts)
        with open(path, "w") as f:
            f.write(f"<svg xmlns='http://www.w3.org/2000/svg' width='{self.w}' height='{self.h}' "
                    f"viewBox='0 0 {self.w} {self.h}'>\n<rect width='100%' height='100%' fill='{SURFACE}'/>\n"
                    f"{body}\n</svg>\n")


def bar(svg, x, y0, w, h, color, tip):
    # 4px rounded data-end, square at the baseline
    r = min(4, w / 2, h)
    path = (f"M{x:.1f},{y0:.1f} V{y0 - h + r:.1f} Q{x:.1f},{y0 - h:.1f} {x + r:.1f},{y0 - h:.1f} "
            f"H{x + w - r:.1f} Q{x + w:.1f},{y0 - h:.1f} {x + w:.1f},{y0 - h + r:.1f} V{y0:.1f} Z")
    svg.raw(f"<path d='{path}' fill='{color}'><title>{escape(tip)}</title></path>")


def grouped_bars(path, title, subtitle, groups, series, values, unit, fmt="{:.0f}"):
    """values[(group, series_name)] = number"""
    W, H, L, R, T, B = 760, 400, 64, 20, 92, 56
    svg = Svg(W, H, title, subtitle)
    if len(series) > 1:  # a single series is named by the title, no legend box
        svg.legend([(n, c) for n, c in series], L, 74)
    vmax, step = nice_max(max(values.values()) * 1.12)
    ph, pw = H - T - B, W - L - R
    y = lambda v: T + ph - v / vmax * ph
    v = 0
    while v <= vmax + 1e-9:
        svg.raw(f"<line x1='{L}' x2='{W - R}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke='{GRID}' stroke-width='1'/>")
        svg.text(L - 8, y(v) + 4, f"{v:g}", 10, MUTED, "end")
        v += step
    svg.text(16, T + ph / 2, unit, 11, INK2, "middle", rotate=-90)
    gw = pw / len(groups)
    bw = min(34, (gw - 24) / len(series) - 2)
    for gi, g in enumerate(groups):
        gx = L + gi * gw + (gw - len(series) * (bw + 2) + 2) / 2
        for si, (name, color) in enumerate(series):
            val = values.get((g, name))
            if val is None:
                continue
            x = gx + si * (bw + 2)  # 2px surface gap between adjacent bars
            bar(svg, x, y(0), bw, y(0) - y(val), color, f"{g} · {name}: {fmt.format(val)} {unit}")
            svg.text(x + bw / 2, y(val) - 4, fmt.format(val), 9, INK2, "middle")
        svg.text(L + gi * gw + gw / 2, H - B + 18, g, 11, INK2, "middle")
    svg.raw(f"<line x1='{L}' x2='{W - R}' y1='{y(0):.1f}' y2='{y(0):.1f}' stroke='{MUTED}' stroke-width='1'/>")
    svg.save(path)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    out = os.path.join(root, "plots")
    os.makedirs(out, exist_ok=True)

    t = {}
    for r in csv.DictReader(open(os.path.join(root, "results.csv"))):
        s = (int(r["M"]), int(r["K"]), int(r["N"]))
        t[(r["impl"], r["tile"], s)] = {k: float(r[k]) for k in ("kernel_ms", "e2e_ms", "kernel_gflops")}

    groups = [LABEL[s] for s in SHAPES]

    # 1. The required versions, kernel-only throughput
    series = [("naive", S1), ("tiled (T=32)", S2), ("padded (T=32)", S3)]
    key = {"naive": ("naive", "NA"), "tiled (T=32)": ("tiled", "32"), "padded (T=32)": ("padded", "32")}
    vals = {(LABEL[s], n): t[key[n] + (s,)]["kernel_gflops"] for s in SHAPES for n, _ in series}
    cpu = ", ".join(f"{t[('cpu', 'NA', s)]['kernel_gflops']:.1f}" for s in SHAPES)
    grouped_bars(os.path.join(out, "gflops_by_impl.svg"), "Kernel throughput by implementation (Tesla T4)",
                 f"GFLOP/s = 2·M·N·K / kernel time, median of 20. Single-thread CPU for scale: {cpu} GFLOP/s.",
                 groups, series, vals, "GFLOP/s")

    # 2. Tile-size comparison for the tiled kernel
    series = [("T = 8", SEQ[0]), ("T = 16", SEQ[1]), ("T = 32", SEQ[2])]
    vals = {(LABEL[s], f"T = {T}"): t[("tiled", T, s)]["kernel_gflops"] for s in SHAPES for T in ("8", "16", "32")}
    vals.update({(LABEL[s], "naive"): t[("naive", "NA", s)]["kernel_gflops"] for s in SHAPES})
    grouped_bars(os.path.join(out, "tile_sizes.svg"), "Tiled kernel: tile size comparison",
                 "Kernel-only GFLOP/s. T = 32 is fastest on every shape; T = 8 is slower than naive on 4096³ and 1000×513×769.",
                 groups, [("naive", MUTED)] + series, vals, "GFLOP/s")

    # 3. Padding effect: tiled vs padded, and the transposed-A experiment, T = 32
    series = [("tiled", S1), ("padded", S2), ("tiled_ta (32-way store conflicts)", S3)]
    vals = {}
    for s in SHAPES:
        vals[(LABEL[s], "tiled")] = t[("tiled", "32", s)]["kernel_gflops"]
        vals[(LABEL[s], "padded")] = t[("padded", "32", s)]["kernel_gflops"]
        vals[(LABEL[s], "tiled_ta (32-way store conflicts)")] = t[("tiled_ta", "32", s)]["kernel_gflops"]
    grouped_bars(os.path.join(out, "padding_effect.svg"), "Does padding help? (T = 32)",
                 "Padding costs the plain tiled kernel 17–24% of its GFLOP/s. padded_ta (not shown) performs the same as padded.",
                 groups, series, vals, "GFLOP/s")

    # 4. Kernel share of end-to-end time (tiled, T = 32)
    vals = {(LABEL[s], "kernel share"): 100 * t[("tiled", "32", s)]["kernel_ms"] / t[("tiled", "32", s)]["e2e_ms"]
            for s in SHAPES}
    grouped_bars(os.path.join(out, "kernel_share_e2e.svg"), "Share of end-to-end GPU time spent in the kernel",
                 "tiled, T = 32. The rest is cudaMalloc, host↔device copies and cudaFree.",
                 groups, [("kernel share", S1)], vals, "% of end-to-end time", "{:.0f}%")

    # 5. Kernel time vs shared-memory wavefronts (profiler, 1024³)
    prof = [r for r in csv.DictReader(open(os.path.join(root, "profile_summary.csv")))
            if r["M"] == "1024" and r["impl"] != "naive"]
    W, H, L, R, T, B = 760, 420, 64, 24, 92, 56
    svg = Svg(W, H, "Kernel time follows shared-memory wavefronts (1024³, Nsight Compute)",
              "Each point is one kernel; label = tile size. Dashed line: 100 ns per million wavefronts. ncu runs at base clocks.")
    groups3 = [("tiled", S1, {"tiled"}), ("padded / padded_ta (identical)", S2, {"padded", "padded_ta"}),
               ("tiled_ta", S3, {"tiled_ta"})]
    svg.legend([(n, c) for n, c, _ in groups3], L, 74)
    pts = [((int(r["shld_wavefronts"]) + int(r["shst_wavefronts"])) / 1e6, int(r["time_ns"]) / 1e6, r) for r in prof]
    xmax, xs = nice_max(max(p[0] for p in pts) * 1.08)
    ymax, ys = nice_max(max(p[1] for p in pts) * 1.1)
    pw, ph = W - L - R, H - T - B
    X = lambda v: L + v / xmax * pw
    Y = lambda v: T + ph - v / ymax * ph
    v = 0
    while v <= ymax + 1e-9:
        svg.raw(f"<line x1='{L}' x2='{W - R}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='{GRID}'/>")
        svg.text(L - 8, Y(v) + 4, f"{v:g}", 10, MUTED, "end")
        v += ys
    v = 0
    while v <= xmax + 1e-9:
        svg.text(X(v), H - B + 16, f"{v:g}", 10, MUTED, "middle")
        v += xs
    svg.text(L + pw / 2, H - 14, "shared-memory wavefronts, load + store (millions)", 11, INK2, "middle")
    svg.text(16, T + ph / 2, "kernel time (ms)", 11, INK2, "middle", rotate=-90)
    svg.raw(f"<line x1='{X(0):.1f}' y1='{Y(0):.1f}' x2='{X(xmax):.1f}' y2='{Y(min(ymax, xmax * 0.1)):.1f}' "
            f"stroke='{MUTED}' stroke-width='1.5' stroke-dasharray='5 4'/>")
    svg.raw(f"<line x1='{L}' x2='{W - R}' y1='{Y(0):.1f}' y2='{Y(0):.1f}' stroke='{MUTED}'/>")
    for name, color, impls in groups3:
        for x, yv, r in pts:
            if r["impl"] in impls and not (r["impl"] == "padded_ta"):  # padded_ta sits exactly on padded
                tip = f"{r['impl']} T={r['tile']}: {x:.1f} M wavefronts, {yv:.2f} ms"
                svg.raw(f"<circle cx='{X(x):.1f}' cy='{Y(yv):.1f}' r='5' fill='{color}' stroke='{SURFACE}' "
                        f"stroke-width='2'><title>{escape(tip)}</title></circle>")
                svg.text(X(x) + 8, Y(yv) + 4, f"T={r['tile']}", 10, INK2)
    svg.save(os.path.join(out, "time_vs_wavefronts.svg"))
    print(f"wrote {out}/: gflops_by_impl.svg tile_sizes.svg padding_effect.svg kernel_share_e2e.svg time_vs_wavefronts.svg")


if __name__ == "__main__":
    main()
