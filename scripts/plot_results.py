#!/usr/bin/env python3
"""Make the report plots as standalone SVG files (no matplotlib / numpy needed).

Usage:  python3 scripts/plot_results.py [results]      -> results/plots/*.svg

Inputs (missing files or implementations are skipped):
         results/results.csv           (scripts/bench.sh)
         results/edge_sweep.csv        (scripts/edge_sweep.sh)
         results/profile_summary.csv   (scripts/summarize_profile.py)

Every mark carries an SVG <title>, so hovering a bar or point in a browser shows
its exact value. The numbers are also in the tables in docs/week*.md.
"""
import csv
import os
import sys
from html import escape

# Validated categorical slots in fixed order (light surface): blue, orange, aqua, yellow, magenta.
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
S6, S7, S8 = "#008300", "#6250d6", "#e34948"  # palette slots 6-8 (fixed order, validated adjacent)
SEQ = ["#9cc3ef", "#4f93e0", "#1d5aa6"]  # one hue, light -> dark (ordinal: tile 8 / 16 / 32)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
FONT = "font-family='Helvetica, Arial, sans-serif'"


def label(s):
    return f"{s[0]}×{s[1]}×{s[2]}"


def load(path):
    """{(impl, tile, (M, K, N)): {...}} plus the shapes in file order."""
    t, shapes = {}, []
    if not os.path.exists(path):
        return t, shapes
    for r in csv.DictReader(open(path)):
        s = (int(r["M"]), int(r["K"]), int(r["N"]))
        if s not in shapes:
            shapes.append(s)
        t[(r["impl"], r["tile"], s)] = {k: float(r[k]) for k in ("kernel_ms", "e2e_ms", "kernel_gflops")}
    return t, shapes


# (legend label, impl, tile column value, colour) for the main comparison
MAIN = [("naive", "naive", "NA", S1), ("tiled (T=32)", "tiled", "32", S2), ("padded (T=32)", "padded", "32", S3),
        ("regtile", "regtile", "64x64", S4), ("cuBLAS (reference)", "cublas", "NA", S5)]
# the optional extensions: register tiling, then float4 access, then the bigger tile
EXT = [("tiled (T=32)", "tiled", "32", S1), ("regtile", "regtile", "64x64", S2), ("vec (64×64, 4×4)", "vec", "64x64", S3),
       ("vec128 (128×128, 8×8)", "vec128", "128x128", S4), ("cuBLAS (reference)", "cublas", "NA", S5)]


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


# The optimization ladder, in presentation order: (step label, impl, tile column, colour).
LADDER = [("CPU, i-j-k", "cpu_ijk", "NA", MUTED), ("CPU, i-k-j", "cpu", "NA", MUTED),
          ("Naive GPU (uncoalesced)", "naive_uncoal", "NA", S1), ("Global-memory coalesced", "naive", "NA", S2),
          ("Shared memory (T=32)", "tiled", "32", S3), ("Shared + padding (T=32)", "padded", "32", S4),
          ("1D block tiling (8×1)", "tile1d", "64x64", S5), ("2D block tiling (8×8)", "tile2d", "128x128", S6),
          ("Vectorized (float4, 8×8)", "vec128", "128x128", S7), ("cuBLAS (reference)", "cublas", "NA", S8)]


def ladder_table(t, shape, out_md):
    """Markdown table in the style of the reference post: cumulative, one row per step."""
    M, K, N = shape
    fl, by = 2.0 * M * N * K, 4.0 * (M * K + K * N + M * N)
    base = t.get(("naive_uncoal", "NA", shape)) or t.get(("naive", "NA", shape))
    cub = t.get(("cublas", "NA", shape))
    lines = [f"Kernel-only, median of 20, {M}×{K}×{N}. Bandwidth = (A + B + C bytes) / time (effective).", "",
             "| Step | Kernel | Time (ms) | TFLOPS | Bandwidth (GB/s) | vs cuBLAS | Speedup over naive | Speedup over previous step |",
             "|---|---|---|---|---|---|---|---|"]
    prev = None
    for i, (name, impl, tile, _) in enumerate(LADDER):
        r = t.get((impl, tile, shape))
        if not r:
            continue
        ms = r["kernel_ms"]
        step = {"cpu_ijk": "2", "cpu": "2", "naive_uncoal": "3", "naive": "4", "tiled": "5", "padded": "5b",
                "tile1d": "6", "tile2d": "7", "vec128": "8"}.get(impl, "–")
        vs = f"{100 * cub['kernel_ms'] / ms:.1f}%" if cub else "–"
        ratio = base["kernel_ms"] / ms if base else None
        sp = "–" if ratio is None else (f"{ratio:.3f}×" if ratio < 0.1 else f"{ratio:.1f}×")
        sprev = f"{prev / ms:.2f}×" if prev and not impl.startswith("cpu") and impl != "cublas" and impl != "padded" else "–"
        lines.append(f"| {step} | {name} | {ms:.4g} | {fl / ms / 1e9:.3g} | {by / ms / 1e6:.3g} | {vs} | {sp} | {sprev} |")
        if not impl.startswith("cpu") and impl not in ("cublas", "padded"):
            prev = ms
    open(out_md, "w").write("\n".join(lines) + "\n")


def line_chart(path, title, subtitle, sizes, series):
    """series: [(label, colour, {size: GFLOP/s})]. x = matrix size (log2), y = TFLOP/s."""
    W, H, L, R, T, B = 800, 420, 64, 190, 92, 56
    svg = Svg(W, H, title, subtitle)
    ymax_v = max(v for _, _, d in series for v in d.values()) / 1000
    ymax, ys = nice_max(ymax_v * 1.1)
    pw, ph = W - L - R, H - T - B
    xs = sorted(sizes)
    X = lambda n: L + (xs.index(n) / max(1, len(xs) - 1)) * pw
    Y = lambda v: T + ph - v / ymax * ph
    v = 0
    while v <= ymax + 1e-9:
        svg.raw(f"<line x1='{L}' x2='{L + pw}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='{GRID}'/>")
        svg.text(L - 8, Y(v) + 4, f"{v:g}", 10, MUTED, "end")
        v += ys
    for n in xs:
        svg.text(X(n), H - B + 16, str(n), 10, MUTED, "middle")
    svg.text(L + pw / 2, H - 14, "matrix size n (M = N = K = n)", 11, INK2, "middle")
    svg.text(16, T + ph / 2, "TFLOP/s (kernel-only)", 11, INK2, "middle", rotate=-90)
    svg.raw(f"<line x1='{L}' x2='{L + pw}' y1='{Y(0):.1f}' y2='{Y(0):.1f}' stroke='{MUTED}'/>")
    ends = []
    for name, color, d in series:
        pts = [(X(n), Y(d[n] / 1000), n, d[n]) for n in xs if n in d]
        if not pts:
            continue
        svg.raw(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='"
                + " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in pts) + "'/>")
        for x, y, n, g in pts:
            svg.raw(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{color}' stroke='{SURFACE}' stroke-width='2'>"
                    f"<title>{escape(name)}, n = {n}: {g / 1000:.3g} TFLOP/s</title></circle>")
        ends.append([pts[-1][1], name, color])
    ends.sort()  # direct labels at the right end, nudged apart
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 13)
    for y, name, color in ends:
        svg.raw(f"<rect x='{L + pw + 22}' y='{y - 5:.1f}' width='8' height='8' rx='2' fill='{color}'/>")
        svg.text(L + pw + 34, y + 3, name, 10, INK2)
    svg.save(path)


def roofline(path, hw, prof_rows, bench, shape_m="1024"):
    """Log-log roofline: x = FLOP per DRAM byte (Nsight), y = achieved GFLOP/s (benchmark)."""
    peak, bw = hw["fp32_peak_tflops"] * 1000, hw["dram_bw_gbs"]  # GFLOP/s, GB/s
    W, H, L, R, T, B = 760, 440, 70, 24, 92, 56
    svg = Svg(W, H, f"Roofline: {hw.get('name', 'GPU')} (FP32 {peak / 1000:.2f} TFLOP/s, DRAM {bw:.0f} GB/s)",
              "Arithmetic intensity from Nsight DRAM bytes at 1024³; achieved throughput from the benchmark. Peak at boost clock.")
    import math
    x0, x1, y0, y1 = -1, 3, 1, math.log10(peak * 2)  # log10 ranges: 0.1..1000 FLOP/B, 10..2*peak GFLOP/s
    pw, ph = W - L - R, H - T - B
    X = lambda a: L + (math.log10(a) - x0) / (x1 - x0) * pw
    Y = lambda g: T + ph - (math.log10(g) - y0) / (y1 - y0) * ph
    for e in range(x0, x1 + 1):
        svg.raw(f"<line x1='{X(10 ** e):.1f}' x2='{X(10 ** e):.1f}' y1='{T}' y2='{T + ph}' stroke='{GRID}'/>")
        svg.text(X(10 ** e), H - B + 16, f"{10 ** e:g}", 10, MUTED, "middle")
    for e in range(int(y0), int(y1) + 1):
        if 10 ** e <= peak * 2:
            svg.raw(f"<line x1='{L}' x2='{L + pw}' y1='{Y(10 ** e):.1f}' y2='{Y(10 ** e):.1f}' stroke='{GRID}'/>")
            svg.text(L - 8, Y(10 ** e) + 4, f"{10 ** e:g}", 10, MUTED, "end")
    svg.text(L + pw / 2, H - 14, "arithmetic intensity (FLOP per DRAM byte, log scale)", 11, INK2, "middle")
    svg.text(16, T + ph / 2, "GFLOP/s (log scale)", 11, INK2, "middle", rotate=-90)
    ridge = peak / bw
    pts = [(10 ** x0, bw * 10 ** x0), (ridge, peak), (10 ** x1, peak)]
    svg.raw(f"<polyline fill='none' stroke='{INK2}' stroke-width='2' points='"
            + " ".join(f"{X(a):.1f},{Y(g):.1f}" for a, g in pts) + "'/>")
    svg.text(X(ridge) + 6, Y(peak) - 6, f"ridge {ridge:.0f} FLOP/B", 10, INK2)
    svg.text(X(10 ** x0 * 1.3), Y(bw * 10 ** x0 * 1.3) - 8, "memory-bound", 10, INK2)
    svg.text(X(10 ** x1) - 4, Y(peak) - 6, "compute-bound", 10, INK2, "end")
    for name, impl, tile, color in LADDER:
        if impl.startswith("cpu"):
            continue
        r = next((p for p in prof_rows if p["impl"] == impl and p["M"] == shape_m
                  and (p["tile"] == tile or tile not in ("32",))), None)
        b = bench.get((impl, tile, (1024, 1024, 1024)))
        if not r or not b or not r.get("dram_read_bytes"):
            continue
        n = 1024
        ai = 2.0 * n ** 3 / (int(r["dram_read_bytes"]) + int(r["dram_write_bytes"] or 0))
        g = b["kernel_gflops"]
        svg.raw(f"<circle cx='{X(ai):.1f}' cy='{Y(g):.1f}' r='5' fill='{color}' stroke='{SURFACE}' stroke-width='2'>"
                f"<title>{escape(name)}: {ai:.0f} FLOP/B, {g:.0f} GFLOP/s</title></circle>")
        dy = {"vec128": -9}.get(impl, 4)  # nudge labels of neighbouring points apart
        left = impl == "tile2d"
        svg.text(X(ai) + (-8 if left else 8), Y(g) + dy, name.split(" (")[0], 10, INK2, "end" if left else "start")
    svg.save(path)


def series_values(t, shapes, spec, field="kernel_gflops"):
    """Keep only series that have data; values keyed by (shape label, legend label)."""
    vals, used = {}, []
    for name, impl, tile, color in spec:
        got = False
        for sh in shapes:
            r = t.get((impl, tile, sh))
            if r:
                vals[(label(sh), name)] = r[field]
                got = True
        if got:
            used.append((name, color))
    return vals, used


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    out = os.path.join(root, "plots")
    os.makedirs(out, exist_ok=True)
    written = []

    t, shapes = load(os.path.join(root, "results.csv"))
    groups = [label(sh) for sh in shapes]

    if t:
        # 1. All implementations, kernel-only throughput
        vals, used = series_values(t, shapes, MAIN)
        cpu = ", ".join(f"{t[('cpu', 'NA', sh)]['kernel_gflops']:.1f}" for sh in shapes if ("cpu", "NA", sh) in t)
        sub = "GFLOP/s = 2·M·N·K / kernel time, median of 20."
        if cpu:
            sub += f" Single-thread CPU for scale: {cpu} GFLOP/s."
        grouped_bars(os.path.join(out, "gflops_by_impl.svg"), "Kernel throughput by implementation",
                     sub, groups, used, vals, "GFLOP/s")
        written.append("gflops_by_impl.svg")

        # 2. Tile-size comparison for the tiled kernel
        spec = [("naive", "naive", "NA", MUTED)] + [(f"T = {T}", "tiled", T, c) for T, c in zip(("8", "16", "32"), SEQ)]
        vals, used = series_values(t, shapes, spec)
        if len(used) > 1:
            grouped_bars(os.path.join(out, "tile_sizes.svg"), "Tiled kernel: tile size comparison",
                         "Kernel-only GFLOP/s for tile sizes 8, 16 and 32, with the naive kernel for reference.",
                         groups, used, vals, "GFLOP/s")
            written.append("tile_sizes.svg")

        # 3. Padding effect at T = 32
        spec = [("tiled", "tiled", "32", S1), ("padded", "padded", "32", S2),
                ("tiled_ta (32-way store conflicts)", "tiled_ta", "32", S3)]
        vals, used = series_values(t, shapes, spec)
        if len(used) > 1:
            grouped_bars(os.path.join(out, "padding_effect.svg"), "Does padding help? (T = 32)",
                         "padded = tiled + [T][T+1] shared arrays. padded_ta (not shown) performs the same as padded.",
                         groups, used, vals, "GFLOP/s")
            written.append("padding_effect.svg")

        # 4. Kernel share of end-to-end time (best hand-written kernel available)
        best = "regtile" if any(k[0] == "regtile" for k in t) else "tiled"
        tile = "64x64" if best == "regtile" else "32"
        vals = {(label(sh), "kernel share"): 100 * t[(best, tile, sh)]["kernel_ms"] / t[(best, tile, sh)]["e2e_ms"]
                for sh in shapes if (best, tile, sh) in t}
        if vals:
            grouped_bars(os.path.join(out, "kernel_share_e2e.svg"), "Share of end-to-end GPU time spent in the kernel",
                         f"{best}{'' if best == 'regtile' else ', T = 32'}. The rest is cudaMalloc, host↔device copies and cudaFree.",
                         groups, [("kernel share", S1)], vals, "% of end-to-end time", "{:.0f}%")
            written.append("kernel_share_e2e.svg")

        # 4b. Extensions: register tiling and vectorized access
        vals, used = series_values(t, shapes, EXT)
        if any(n.startswith(("regtile", "vec")) for n, _ in used):
            grouped_bars(os.path.join(out, "extensions.svg"), "Register tiling and vectorized access",
                         "Kernel-only GFLOP/s. vec = regtile + float4 loads/stores (same tiles); vec128 also doubles the tiles.",
                         groups, used, vals, "GFLOP/s")
            written.append("extensions.svg")

        # 5. Gap to cuBLAS: each kernel's throughput as % of cuBLAS
        if any(k[0] == "cublas" for k in t):
            vals, used = {}, []
            pct = [(n, i, tl, c) for (n, i, tl, _), c in zip([MAIN[0]] + EXT[:4], (S1, S2, S3, S4, S5))]
            for name, impl, tl, color in pct:
                got = False
                for sh in shapes:
                    r, c = t.get((impl, tl, sh)), t.get(("cublas", "NA", sh))
                    if r and c:
                        vals[(label(sh), name)] = 100 * r["kernel_gflops"] / c["kernel_gflops"]
                        got = True
                if got:
                    used.append((name, color))
            grouped_bars(os.path.join(out, "percent_of_cublas.svg"), "Gap to cuBLAS",
                         "Kernel-only throughput as a percentage of cuBLAS SGEMM on the same GPU and shape.",
                         groups, used, vals, "% of cuBLAS", "{:.0f}%")
            written.append("percent_of_cublas.svg")

    # 6. Edge-effect sweep
    e, eshapes = load(os.path.join(root, "edge_sweep.csv"))
    if e:
        vals, used = series_values(e, eshapes, EXT)
        grouped_bars(os.path.join(out, "edge_sweep.svg"), "Edge effects: n − 1, n, n + 1",
                     "Kernel-only GFLOP/s. n ± 1 are odd sizes: no row is 16-byte aligned, so vec / vec128 fall back to scalar access.",
                     [label(sh) for sh in eshapes], used, vals, "GFLOP/s")
        written.append("edge_sweep.svg")

    # 7. Kernel time vs shared-memory wavefronts (profiler, 1024³)
    ps = os.path.join(root, "profile_summary.csv")
    prof = [r for r in csv.DictReader(open(ps)) if r["M"] == "1024" and r["impl"] in
            ("tiled", "padded", "padded_ta", "tiled_ta") and r.get("shld_wavefronts")] if os.path.exists(ps) else []
    if prof:
        plot_wavefronts(prof, os.path.join(out, "time_vs_wavefronts.svg"))
        written.append("time_vs_wavefronts.svg")

    # 8. The optimization ladder (reference-post style)
    if t and (4096, 4096, 4096) in shapes:
        ladder_table(t, (4096, 4096, 4096), os.path.join(root, "ladder_4096.md"))
        vals = {}
        used = []
        short = {"cpu_ijk": "CPU ijk", "cpu": "CPU", "naive_uncoal": "Naive", "naive": "Coalesced",
                 "tiled": "Shared", "padded": "Padded", "tile1d": "1D tile", "tile2d": "2D tile",
                 "vec128": "Vectorized", "cublas": "cuBLAS"}
        for name, impl, tile, color in LADDER:
            r = t.get((impl, tile, (4096, 4096, 4096)))
            if r:
                vals[(short[impl], "TFLOP/s")] = r["kernel_gflops"] / 1000
                used.append(short[impl])
        if used:
            W = 760
            grouped_bars(os.path.join(out, "ladder_4096.svg"), "Optimization ladder at 4096³",
                         "Kernel-only TFLOP/s for each step, left to right in the order presented.",
                         used, [("TFLOP/s", S1)], vals, "TFLOP/s", "{:.2f}")
            written.append("ladder_4096.svg")

    # 9. Performance across matrix sizes, one chart per step (previous step, this step, cuBLAS)
    sw, sizes_sq = load(os.path.join(root, "sweep.csv"))
    if sw and t:  # add the 4096³ benchmark points (bench.sh) so the curves reach the largest size
        for (impl, tile, sh), r in t.items():
            if sh == (4096, 4096, 4096) and not impl.startswith("cpu") and (impl, tile, sh) not in sw:
                sw[(impl, tile, sh)] = r
                if sh not in sizes_sq:
                    sizes_sq.append(sh)
    if sw:
        ns = sorted({s[0] for s in sizes_sq})
        def curve(impl, tile):
            return {s[0]: sw[(impl, tile, s)]["kernel_gflops"] for s in sizes_sq if (impl, tile, s) in sw}
        gpu = [l for l in LADDER if not l[1].startswith("cpu") and l[1] != "cublas"]
        cub = curve("cublas", "NA")
        for i, (name, impl, tile, color) in enumerate(gpu):
            series = []
            if i > 0 and impl != "padded":
                prev = gpu[i - 1] if gpu[i - 1][1] != "padded" else gpu[i - 2]
                series.append((prev[0], MUTED, curve(prev[1], prev[2])))
            elif impl == "padded":
                series.append((gpu[i - 1][0], MUTED, curve(gpu[i - 1][1], gpu[i - 1][2])))
            series.append((name, color, curve(impl, tile)))
            if cub:
                series.append(("cuBLAS", S8, cub))
            if series[-2 if cub else -1][2]:
                line_chart(os.path.join(out, f"sweep_{i + 1}_{impl}.svg"), f"Step: {name}",
                           "TFLOP/s across matrix sizes, compared with the previous step and cuBLAS.", ns, series)
                written.append(f"sweep_{i + 1}_{impl}.svg")
        allc = [(name, color, curve(impl, tile)) for name, impl, tile, color in LADDER if not impl.startswith("cpu")]
        line_chart(os.path.join(out, "sweep_all.svg"), "Performance across matrix sizes: every step",
                   "Kernel-only TFLOP/s, median of 20, square matrices.", ns, [c for c in allc if c[2]])
        written.append("sweep_all.svg")
        cpu = [(name, c, curve(impl, tile)) for name, impl, tile, c in LADDER[:2]]
        cpu = [(n, S1 if i == 0 else S2, d) for i, (n, c, d) in enumerate(cpu) if d]
        if cpu:
            line_chart(os.path.join(out, "sweep_cpu.svg"), "CPU loop order: i-j-k vs i-k-j",
                       "Single thread, GFLOP/s shown as TFLOP/s; same arithmetic, different memory order.",
                       sorted({n for _, _, d in cpu for n in d}), cpu)
            written.append("sweep_cpu.svg")

    # 10. Roofline
    hwj = os.path.join(root, "hw_specs.json")
    if os.path.exists(hwj) and os.path.exists(ps) and t:
        import json
        hw = json.load(open(hwj))
        if hw.get("fp32_peak_tflops"):
            roofline(os.path.join(out, "roofline.svg"), hw, list(csv.DictReader(open(ps))), t)
            written.append("roofline.svg")

    print(f"wrote {out}/: " + " ".join(written))


def plot_wavefronts(prof, path):
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
            if r["impl"] in impls and r["impl"] != "padded_ta":  # padded_ta sits exactly on padded
                tip = f"{r['impl']} T={r['tile']}: {x:.1f} M wavefronts, {yv:.2f} ms"
                svg.raw(f"<circle cx='{X(x):.1f}' cy='{Y(yv):.1f}' r='5' fill='{color}' stroke='{SURFACE}' "
                        f"stroke-width='2'><title>{escape(tip)}</title></circle>")
                svg.text(X(x) + 8, Y(yv) + 4, f"T={r['tile']}", 10, INK2)
    svg.save(path)


if __name__ == "__main__":
    main()
