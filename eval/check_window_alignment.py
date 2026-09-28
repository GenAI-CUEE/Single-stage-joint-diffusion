"""
Checks whether a built dataset folder's test windows match the paper's, using
splits/sirta_2023/test_window_timestamps.csv (frame timestamps for the 1998
canonical test windows, built from the paper's own dataset) as the reference.
No windowing is reimplemented here: this builds the given folder's own test
windows with dataloader.sirta_dataset.SeqDataset_SIRTA and only compares
frame timestamps against the reference file.

Usage:
  python eval/check_window_alignment.py --data-folder samples/sirta_2023
  python eval/check_window_alignment.py --data-folder sirta_2023_image64-localtime_ICHP-daytime-dhi --out alignment.csv

For each of the 1998 canonical test_ids, classifies the given folder's matching window as:
  identical       input and target frame timestamps both match the reference exactly.
  input_shifted   input frame timestamps differ; target frame timestamps match.
  target_shifted  target frame timestamps differ (checked first; takes precedence over
                  input_shifted if both differ).
  missing         the given data folder builds no test-mode window with that test_id at all.

IMPORTANT about "missing": this script cannot tell apart two different reasons a test_id
is missing -- (a) the data folder was simply never built to cover that window (e.g. a
partial/sample build covering only a few days), and (b) a full-year rebuild's SIRTA
re-issue-driven nan-dropping removed that specific window (see PORTING_NOTES.md, "Data").
Both look identical from here: a "missing" count on its own does not tell you which. Know
what your data folder was built from before interpreting it.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataloader.sirta_dataset import SeqDataset_SIRTA

REFERENCE_CSV = os.path.join(REPO_ROOT, "splits/sirta_2023/test_window_timestamps.csv")
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs/xiong.yaml")


def fmt_ref(row):
    h, m, day, month, year = (int(x) for x in row)
    return "%04d-%02d-%02d %02d:%02d" % (year, month, day, h, m)


def build_dataset(config_path, data_folder, mode):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["Data"]["data_folder"] = data_folder
    return SeqDataset_SIRTA(config, torch.device("cpu"), mode=mode,
                             attribute_list=["label", "image", "feature"])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-folder", required=True,
                     help="dataset folder to check, e.g. samples/sirta_2023")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                     help="template config supplying Data/ImageSequence settings "
                          "(default: %(default)s); its own data_folder is ignored")
    ap.add_argument("--mode", default="test", choices=["test", "valid", "train"])
    ap.add_argument("--out", default=None,
                     help="write the per-window classification to this CSV (test_id,class)")
    args = ap.parse_args()

    with open(REFERENCE_CSV, newline="") as f:
        reference = list(csv.DictReader(f))
    print("reference: %d canonical test windows (%s)" % (len(reference), REFERENCE_CSV))

    print(">>>> building %s-mode windows from %s..." % (args.mode, args.data_folder))
    ds = build_dataset(args.config, args.data_folder, args.mode)

    n = len(ds.data["ref_y"])
    built = {}
    for i in range(n):
        tid = fmt_ref(np.asarray(ds.data["ref_y"][i])[0])
        if tid not in built:
            built[tid] = i

    counts = {"identical": 0, "input_shifted": 0, "target_shifted": 0, "missing": 0}
    out_rows = []
    for ref in reference:
        tid = ref["test_id"]
        pos = built.get(tid)
        if pos is None:
            cls = "missing"
        else:
            ref_x = [fmt_ref(row) for row in np.asarray(ds.data["ref_x"][pos])]
            ref_y = [fmt_ref(row) for row in np.asarray(ds.data["ref_y"][pos])]
            csv_in = [ref["in_%d" % i] for i in range(1, 9)]
            csv_tgt = [ref["tgt_%d" % i] for i in range(1, 9)]
            if ref_y != csv_tgt:
                cls = "target_shifted"
            elif ref_x != csv_in:
                cls = "input_shifted"
            else:
                cls = "identical"
        counts[cls] += 1
        out_rows.append({"test_id": tid, "class": cls})

    print()
    print("=== counts (%d canonical windows) ===" % len(reference))
    for k in ("identical", "input_shifted", "target_shifted", "missing"):
        print("  %-15s %d" % (k, counts[k]))
    print()
    print("Note on 'missing': this script cannot tell a data folder that was never built to")
    print("cover a window apart from one where a full rebuild's re-issue nan-dropping removed")
    print("it -- both look identical here. Know what your data folder was built from.")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["test_id", "class"])
            w.writeheader()
            w.writerows(out_rows)
        print("wrote %d rows -> %s" % (len(out_rows), args.out))


if __name__ == "__main__":
    main()
