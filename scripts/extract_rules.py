#!/usr/bin/env python3
"""Collect Nsight Compute rule messages (OPT / WRN) and headline metrics from the
details pages exported by profile.sh (results/profile/*.details.txt).

Usage:  python3 scripts/extract_rules.py [results/profile] > results/profile/rules_summary.md
"""
import glob
import os
import re
import sys

KEYS = ["Duration", "SM Frequency", "Compute (SM) Throughput", "Memory Throughput", "DRAM Throughput",
        "Registers Per Thread", "Grid Size", "Block Size", "Waves Per SM", "Theoretical Occupancy",
        "Achieved Occupancy", "Block Limit Registers", "Block Limit Shared Mem", "Warp Cycles Per Issued Instruction",
        "Executed Ipc Active", "L2 Hit Rate"]


def parse(path):
    lines = open(path, errors="replace").read().splitlines()
    metrics, rules, section, cur = {}, [], "", None
    for raw in lines:
        s = raw.strip()
        if s.startswith("Section:"):
            section = s[len("Section:"):].strip()
        if cur is not None:
            if not s or s.startswith("---") or s.startswith("Section:") or re.match(r"^(OPT|WRN|INF)\b", s):
                rules.append(cur)
                cur = None
            else:
                cur["text"] += " " + s
                continue
        m = re.match(r"^(OPT|WRN)\s+(.*)$", s)
        if m:
            cur = {"kind": m[1], "section": section, "text": m[2].strip()}
            continue
        for k in KEYS:
            if s.startswith(k + " ") and k not in metrics:
                parts = re.split(r"\s{2,}", s)
                if len(parts) >= 2:
                    metrics[k] = " ".join(parts[1:])
    if cur:
        rules.append(cur)
    for r in rules:
        r["text"] = re.sub(r"\s+", " ", r["text"]).strip()
    return metrics, rules


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/profile"
    files = sorted(glob.glob(os.path.join(d, "*.details.txt")))
    if not files:
        sys.exit(f"no *.details.txt in {d}")
    print("# Nsight Compute rule messages and headline metrics\n")
    print(f"Generated from {len(files)} details pages by scripts/extract_rules.py.\n")
    for f in files:
        name = os.path.basename(f).replace(".details.txt", "")
        metrics, rules = parse(f)
        print(f"## {name}\n")
        if metrics:
            print("| " + " | ".join(metrics) + " |")
            print("|" + "---|" * len(metrics))
            print("| " + " | ".join(metrics.values()) + " |\n")
        for r in rules:
            print(f"- **{r['kind']}** ({r['section']}): {r['text']}")
        print()


if __name__ == "__main__":
    main()
