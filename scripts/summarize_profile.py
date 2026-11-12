#!/usr/bin/env python3
"""Collect the Nsight Compute metric CSVs written by scripts/profile.sh into one table.

Usage:  python3 scripts/summarize_profile.py [results/profile] > results/profile_summary.csv

Each input file results/profile/<impl>_T<tile>_<M>x<K>x<N>.csv (or naive_<shape>.csv)
holds one kernel launch with one metric per row. The output has one row per
(impl, tile, shape) and one column per metric (short names below).
"""
import csv
import glob
import io
import os
import re
import sys

SHORT = {
    "gpu__time_duration.sum": "time_ns",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "mem_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": "gld_requests",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "gld_sectors",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct": "gld_efficiency_pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": "gst_sectors",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "shld_conflicts",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "shst_conflicts",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": "shld_wavefronts",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": "shst_wavefronts",
    "smsp__inst_executed_op_shared_ld.sum": "shld_inst",
    "smsp__inst_executed_op_shared_st.sum": "shst_inst",
    "dram__bytes_read.sum": "dram_read_bytes",
    "dram__bytes_write.sum": "dram_write_bytes",
}
NAME = re.compile(r"^(?P<impl>[a-z_]+?)(?:_T(?P<tile>\d+))?_(?P<M>\d+)x(?P<K>\d+)x(?P<N>\d+)\.csv$")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/profile"
    rows = []
    for path in sorted(glob.glob(os.path.join(d, "*.csv"))):
        m = NAME.match(os.path.basename(path))
        if not m:
            continue
        text = open(path).read()
        start = text.find('"ID"')  # skip ==PROF== lines
        if start < 0:
            print(f"warning: no metrics in {path}", file=sys.stderr)
            continue
        row = {"impl": m["impl"], "tile": m["tile"] or "NA", "M": m["M"], "K": m["K"], "N": m["N"]}
        for r in csv.DictReader(io.StringIO(text[start:])):
            key = SHORT.get(r["Metric Name"])
            if key:
                row[key] = r["Metric Value"].replace(",", "")
                row["kernel"] = r["Kernel Name"].split("(")[0].replace("void ", "")
        rows.append(row)
    cols = ["impl", "tile", "M", "K", "N", "kernel"] + list(SHORT.values())
    w = csv.DictWriter(sys.stdout, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    order = {"naive": 0, "tiled": 1, "padded": 2, "tiled_ta": 3, "padded_ta": 4}
    for r in sorted(rows, key=lambda r: (-int(r["M"]), order.get(r["impl"], 9), int(r["tile"]) if r["tile"] != "NA" else 0)):
        w.writerow(r)


if __name__ == "__main__":
    main()
