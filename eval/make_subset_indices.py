"""
Selects test windows by datetime identity, not position.

Given a dataset folder built by preprocessing/ (e.g. samples/sirta_2023) and a CSV
of requested test_ids -- first-target-frame (y0) timestamps, dataset_index,timestamp_local
format as in splits/sirta_2023/test_subset_ids.csv -- this builds every test window the
dataloader would build from that folder (dataloader.sirta_dataset.SeqDataset_SIRTA,
never reimplemented here) and looks each requested test_id up by its own target
timestamp. Position is not reusable across dataset folders built from different raw
files: dataloader/sirta_dataset.py::making_data2sequence slides positionally over
whatever rows survive the join, so a single missing minute shifts every later
window's position (see eval/reproduce_comparison.py::build_ground_truth, which
matches the shipped 1998 canonical test_ids the same way for the same reason).

Writes <dataset_folder>/test_subset_ids.csv and test_subset_indices.npy -- the
latter is what a config's Text2ImageJoint.Network.subset_indices (or the
equivalent Eval.subset_indices key another arm reads) points at. Any requested
test_id the dataset folder can't build is reported, never silently dropped.

Usage:
  python eval/make_subset_indices.py <dataset_folder> <test_ids_csv> [--config CONFIG] [--mode test]
"""
import argparse
import csv
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import yaml


def fmt_ref(ref_row):
    h, m, day, month, year = (int(x) for x in ref_row)
    return "%04d-%02d-%02d %02d:%02d" % (year, month, day, h, m)


def build_dataset(config_path, dataset_folder, mode):
    import torch
    from dataloader.sirta_dataset import SeqDataset_SIRTA

    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["Data"]["data_folder"] = dataset_folder

    return SeqDataset_SIRTA(config, torch.device("cpu"), mode=mode,
                             attribute_list=["label", "image", "feature"])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_folder", help="folder built by preprocessing/, e.g. samples/sirta_2023")
    ap.add_argument("test_ids_csv", help="CSV of requested test_ids: dataset_index,timestamp_local")
    ap.add_argument("--config", default="configs/joint_x0hat_seed1.yaml",
                     help="template config supplying Data/ImageSequence settings "
                          "(default: %(default)s); its own data_folder is ignored")
    ap.add_argument("--mode", default="test", choices=["test", "valid", "train"])
    args = ap.parse_args()

    print(">>>> building %s-mode windows from %s (config: %s)..."
          % (args.mode, args.dataset_folder, args.config))
    ds = build_dataset(args.config, args.dataset_folder, args.mode)

    n_local = len(ds.data["ref_y"])
    local_test_id = [fmt_ref(np.asarray(ds.data["ref_y"][i])[0]) for i in range(n_local)]
    id_to_idx = {}
    for i, tid in enumerate(local_test_id):
        if tid in id_to_idx:
            print("WARNING: duplicate test_id %s at positions %d and %d; keeping the first"
                  % (tid, id_to_idx[tid], i))
        else:
            id_to_idx[tid] = i

    with open(args.test_ids_csv, newline="") as f:
        requested = [row["timestamp_local"] for row in csv.DictReader(f)]

    found = [tid for tid in requested if tid in id_to_idx]
    missing = [tid for tid in requested if tid not in id_to_idx]

    print("requested: %d   found: %d   missing: %d" % (len(requested), len(found), len(missing)))
    if missing:
        print("MISSING test_ids (%s cannot build these windows):" % args.dataset_folder)
        for tid in missing:
            print("  ", tid)

    positions = np.array([id_to_idx[tid] for tid in found], dtype=np.int64)

    out_csv = os.path.join(args.dataset_folder, "test_subset_ids.csv")
    out_npy = os.path.join(args.dataset_folder, "test_subset_indices.npy")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset_index", "timestamp_local"])
        for pos, tid in zip(positions, found):
            w.writerow([int(pos), tid])
    np.save(out_npy, positions)
    print("wrote %d positions -> %s, %s" % (len(positions), out_csv, out_npy))

    mismatches = 0
    for pos, tid in zip(positions, found):
        actual = fmt_ref(np.asarray(ds.data["ref_y"][pos])[0])
        if actual != tid:
            mismatches += 1
            print("MISMATCH: position %d has test_id %s, expected %s" % (pos, actual, tid))
    if mismatches:
        print("VERIFICATION FAILED: %d position(s) do not match their requested test_id" % mismatches)
        sys.exit(1)
    print("verified: every written position's first-target timestamp equals its requested test_id")


if __name__ == "__main__":
    main()
