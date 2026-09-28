# prediction_samples

The forecasts behind the paper's comparison table (Table 2), one file per method, so the table can be
reproduced exactly by scoring them against your own copy of the SIRTA data.

## Download

The 15 `.npz` files are hosted on Dropbox:
<https://www.dropbox.com/scl/fo/c9z5wrxeemrwc7dlyf9tq/AMwm-yGZcrw62MVTwTLL1oY?rlkey=kcrc8epk8nbumthxuyiv1emr2&dl=0>

Put them in `prediction_samples/predictions_db/`. Verify them against `CHECKSUMS.tsv` before use:

```
cd prediction_samples/predictions_db && sha256sum -c ../CHECKSUMS.tsv
```

## Reproduce Table 2

```
python eval/reproduce_comparison.py                        # stored mode (default)
```

This scores every file against the ground truth in your local SIRTA data, computes the smart-persistence
reference from that data, and compares every cell with `expected_table2.csv`. It should end with
`183/183 cells PASS`.

```
python eval/reproduce_comparison.py --mode regenerate [--rows SunSet,Xiong] [--force]
```

This re-runs inference from the released weights instead. Deterministic rows reproduce exactly, and so do
both proposed-model rows, given the same data, subset order and batch size. SkyGPT and Xiong do not: their
ensembles are not reproducible run to run (see `seeded` below), and a fresh run lands within the table's
confidence intervals rather than on its printed values. Existing outputs are reused unless `--force` is
given; forced runs write to a new directory and never overwrite.

### If you download SIRTA's data today

In February 2025 SIRTA re-issued the 2023 irradiance files, marking as unusable some minutes that were still
provisional when the paper's copy was downloaded. No measurement on a test day changed, but the preprocessing
drops any window with a missing minute, so a dataset rebuilt from today's files lacks 86 of the 1998 test
samples (`rebuild_excluded.csv`). The driver detects this and compares against `expected_rebuild.csv`,
computed on the remaining 1912 samples; every cell should still match exactly. On any other sample set it
reports the missing samples and does not claim a match.

Use the split lists in `splits/sirta_2023/` rather than regenerating the split; see `PORTING_NOTES.md`.

## What each file contains

| key | shape | meaning |
|---|---|---|
| `S` or `ens_ghi` | (1998, M, 8) | GHI forecasts, W/m², for 1998 test samples, M members, 8 horizons. The key and dtype are exactly as the paper's scorer read them. |
| `mu_d` | (1998, 8) | Xiong only: the point forecast used for RMSE and FS. |
| `issue_time` | (1998,) | time of the last input frame |
| `target_time` | (1998, 8) | times of the eight forecast frames |
| `test_id` | (1998,) | the sample's key in `Test_Tr0p80-Val0p10-Test0p10-Frame-2x8-to-2x8.txt` (its first target time) |
| `test_index` | (1998,) | the line of that key in the same file, 0-based |
| `meta` | JSON string | method, Table 2 label, units, member count, sampling steps, `seeded`, checkpoint and source-file SHA-256 |

`MANIFEST.tsv` lists the keys, shapes and dtypes of every file.

For the deterministic methods, `S` holds one point forecast repeated M = 100 times, as the paper's
scorer received it.

**Times are Europe/Paris civil time** (CET/CEST, with daylight saving), written without an offset, as in
the source data.

**`seeded`** in `meta`: `true` for the deterministic methods; `partial` for the proposed model (seeded
initial noise per member; no GPU determinism flags are set, though that has not been observed to change its
output given the same data, subset order and batch size -- see `PORTING_NOTES.md`) and Xiong (seeded final
draw, unseeded forward pass); `false` for SkyGPT (no seed anywhere on its sampling path).

## What is not included

No ground truth or measured irradiance, no smart-persistence forecasts (they are built from measured
irradiance), no clear-sky irradiance and no sky images. These come from SIRTA, whose data policy
reserves distribution to SIRTA; the scoring code reads them from your own download.

## Thirteen non-contiguous windows

13 of the 1998 samples are not sixteen consecutive two-minute frames, as the paper discloses
(Section 4.1, Table C.3). They are kept, since every method is scored on the same samples. Their
`test_id`s:

```
2023-01-03 14:36:00   2023-01-03 14:44:00   2023-01-03 15:14:00   2023-01-03 15:22:00
2023-02-01 09:22:00   2023-02-01 09:24:00   2023-02-01 09:28:00   2023-03-01 07:34:00
2023-04-01 07:28:00   2023-04-02 07:26:00   2023-08-01 06:22:00   2023-09-03 08:18:00
2023-11-03 07:38:00
```

Five span a night (the 2023-03-01, 04-01, 04-02, 08-01 and 11-03 entries); the other eight skip one or
two frames within a day. The cause is in the sequencing step; see `PORTING_NOTES.md`.

## Files tracked here

`CHECKSUMS.tsv`, `MANIFEST.tsv`, `expected_table2.csv` (every Table 2 cell at full precision, with the
bootstrap settings: B = 5000, seed 20260806, day blocks), `expected_rebuild.csv` (the same on the 1912
samples obtainable from today's SIRTA files), `rebuild_excluded.csv`, `expected_synthetic.csv` (a
consistency-check table for `samples/synthetic_2023`, scored by `eval/check_synthetic.py` -- not a Table 2
reference; see `samples/synthetic_2023/README.md`) and this README. The `.npz` files are not in git.
