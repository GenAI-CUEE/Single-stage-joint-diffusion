# Porting notes

How this repository was assembled from the code that produced the paper's numbers, and every known
discrepancy, limitation or oddity found while doing it. Nothing here changes a reported result; it is
recorded so nobody has to rediscover it.

## How the code was assembled

- **Copied, not rewritten.** Model, data and scoring code was copied verbatim from the two research
  repositories (the comparison benchmark and the proposed model's repository), then edited in place in
  small reviewed passes. Edits are limited to import paths, internal names and comments that cited
  internal documents. A rewritten port would mean testing the port rather than the experiment.
- **Scoring code** (fair CRPS, day-block bootstrap, PICP) was copied from the benchmark's report scripts
  unchanged, and is called by the drivers rather than reimplemented.
- **Authored for the release:** `eval/reproduce_comparison.py`, a driver that loads forecasts and passes
  them to the copied scorer without computing any metric itself, and `train/proposed_one_step.py`, with
  an assertion that checks the gradient topology of the joint model.
- **Verification.** `eval/reproduce_comparison.py` (stored mode) reproduces every cell of Table 2 at full
  float precision, 183 of 183. Regenerating from the weights reproduces the deterministic rows exactly;
  SkyGPT and Xiong land within their confidence intervals (see below).

## Stochastic baselines

- **SkyGPT's sampling is unseeded.** Its token draw is a bare `torch.multinomial` with no generator, and
  its settings have `random_seed: false`, so every run draws a different ensemble. This is a property of
  the original code, kept unchanged. A fresh run gave RMSE 217.15 against the paper's 216.75.
- **Xiong's final draw is seeded, its forward pass is not bit-deterministic.** The mixture sampling uses
  a fixed seed, but the network pass that produces the mixture parameters runs without determinism
  flags. A fresh run gave RMSE 116.35 against the paper's 116.89.
- **The proposed model reproduces bit for bit, given the same data, subset order and batching.** No code
  path sets GPU determinism flags (see below), but that has not been observed to matter here. Running the
  released code with the shipped configs and weights on the paper's own dataset reproduced both proposed
  arms' forecasts bit for bit on all 1998 test windows (`np.array_equal` against
  `prediction_samples/predictions_db/propFuture.npz`'s and `propPast.npz`'s `S`: true for both; scored with
  the repo's own scorer, every `expected_table2.csv` cell for both arms matches exactly, point estimate and
  confidence interval alike). A separate run on a different, smaller subset also matched a first run of that
  same subset bit for bit, on every one of its 173 windows. Forecasts computed from *rebuilt* data can still
  differ from the shipped
  ones -- not from GPU/cuDNN nondeterminism, but because the inputs genuinely differ: a rebuild's images
  reproduce to within one intensity level even from the same raw files on the same machine (see "Images
  reproduce to within one intensity level" below), and some windows are built from shifted rows despite
  matching test_id (see the gap-window note under "Data" below for counts and mechanism).
- **The proposed model's noise depends on batch position, not on the window.** In the configuration behind
  Table 2 (`seeding_mode: periodic`), each ensemble member's initial noise is seeded once per batch
  (`seed + member`), so a window's draw depends on its position within its batch (with `test_batch_size: 4`,
  its position modulo 4 in the evaluation order), not on which window it is. Evaluating the full test set in
  its original order reproduces the paper's forecasts; evaluating a different selection of windows, or a
  dataset rebuilt from SIRTA's current files (which lacks 86 test windows and so shifts later positions),
  gives different, equally valid ensemble draws for most windows. Stored mode is unaffected. The code also
  offers `seeding_mode: corrected`, which seeds per window; the shipped configurations and Table 2's stored
  forecasts use `periodic`.
- No code path sets `torch.backends.cudnn.deterministic` or `torch.use_deterministic_algorithms`.
- For these reasons the paper's forecasts are shipped as stored predictions (`prediction_samples/`), and
  stored mode is the reproduction path.

## Data

- **Non-contiguous windows.** 13 of the 1998 scored test windows are not sixteen consecutive two-minute
  frames (five span a night, eight skip one or two frames), at a similar rate in training (0.45 %) and
  validation (0.50 %) windows. The sequencing step (`making_data2sequence()` in the preprocessing code,
  `preprocessing_SIRTA_sequence.py`, lines 475–481) checks that each half of a window lies within one
  day, but never compares the two halves' days, and never checks frame spacing. Not fixed: fixing it
  would change the dataset the paper was built on. The paper discloses it (Section 4.1, Table C.3); the
  13 windows are listed in `prediction_samples/README.md`.
- **The split is chronological within each month, not random.** In the preprocessing's
  `sirta_utils/train_valid_test_splitting_sequence.py`, line 178 calls `np.random.permutation(total_index)`
  without assigning the result, so no shuffle happens: in each month the earliest ~10 % of rows form the test
  set, the next ~10 % the validation set (off by one row), and the rest training. The test days are therefore
  the 1st–4th of each month; 32 of the 43 share no window with training or validation. The paper describes
  the split as made (Section 4.1). The lists actually used are shipped in `splits/sirta_2023/`; use them rather than
  regenerating the split, which would move the boundaries if any rows differ.
- **Missing values drop whole rows and windows.** `Step0_2_Clean_irradiance.py` (line 60) and
  `sirta_utils/tools.py` `merge_with_irradiance_csv()` (line 595) drop any row with a `nan` in *any* extracted
  column (`I`, `Diff`, `C`, `P`, `H`, …), including columns no model reads; `divide2frames()` (lines 653–657)
  then drops any window missing one of its 16 frames. A single missing minute can therefore remove a window
  whose GHI is complete.
- **SIRTA re-issued the 2023 irradiance files in February 2025.** Our copy of
  `radflux_1b_Lz2M1minIsolys2PrayDp-QC_v01` was downloaded in 2023–24, while many minutes were still awaiting
  validation. The re-issue changed no measurement on any test day but marked 1,946 GHI and 535 DHI minutes as
  unusable (`nan`). Because of the row and window dropping above, a dataset rebuilt from today's files lacks
  86 of the 1998 matched test windows (listed in `prediction_samples/rebuild_excluded.csv`); the other 1912
  keep their test_id, but matching by test_id alone does not guarantee the same window (see the next bullet).
  `eval/reproduce_comparison.py` detects the 86-window shortfall and compares against
  `prediction_samples/expected_rebuild.csv`. That table was scored on a rebuild, so by the mechanism in the
  next bullet it will include some shifted windows too -- their number was not counted, and the rebuild that
  produced the table is no longer available to check directly (see the next bullet for how to check any
  rebuild you do have). The paper discloses the 62 of these 86 whose targets or persistence origin are
  affected (Section 4.1, Table C.3).
- **A window that keeps its test_id can still be built from shifted rows.** `making_data2sequence()` slides
  positionally over whatever rows survive the drops above, so a window whose 16-frame span (8 input + 8
  target) crosses a dropped minute is built one row earlier or later than in the paper's dataset, even though
  its identity (first-target timestamp) is unchanged. Checked directly on a 173-window sample rebuild
  (`samples/sirta_2023`): of 173 windows matched by test_id to the paper's dataset, 166 are built from
  identical rows, 2 have shifted *input* rows only, and 5 have shifted *target* rows -- rerun the same check
  against any other rebuild with `eval/check_window_alignment.py` (see `README.md`). A target-row shift
  changes not just which measurement a window is scored against, but also the
  proposed model's own clear-sky-irradiance scaling factor for the affected horizons (`feature_y`'s `Iclr`,
  read from the same shifted rows), so its forecast for that window can differ from the shipped one by far
  more than ordinary run-to-run image noise -- not because it drew a different sample, but because it scored
  a genuinely different window under the same test_id. This is a property of the windowing, not of any one
  dataset build: `expected_rebuild.csv`'s 1912 windows were scored on a rebuild too, so they will include
  shifted windows by this same mechanism -- their number was not counted, since that rebuild is no longer
  available to check.
- **Raw timestamps are UTC** (`…Z`); they are converted to Europe/Paris civil time in
  `Step1_irradiance_and_Clear-sky_irradiance.ipynb`. The dataset's timestamps are Europe/Paris civil time
  (CET/CEST), stored without an offset.
- **Clear-sky irradiance** is computed, not measured: the Ineichen–Perez model in `pvlib`, with climatological
  Linke turbidity, in the same notebook.
- **One truncated source file.** Our local copy of the 2023-03-14 irradiance file holds 431 of 1440 minutes
  (an interrupted download). It is not a test day, so only training data for that day is reduced.
- **Two cloud-label options removed.** The research configurations named two cloud files that no run in this
  release reads: `yuhao_corrected_cloudcover_and_mask.pickle` (never opened) and
  `sirta_corrected_yuhao_elifan_cloud_2023.csv` (opened only for the `diff_label` and `cloudmask` outputs,
  which no configuration requests; it combines SIRTA's ELIFAN cloud fraction and Nie et al.'s cloud detector
  and served work outside this paper). Both options, and the loader code that served them, were removed; the
  dataloader returns bit-identical inputs for every shipped configuration.
- **Required data**, under exactly the folder name `sirta_2023_image64-localtime_ICHP-daytime-dhi`:
  `sirta_data_2023.csv`, `sirta_image_2023.pickle`, and the three
  `{Train,Valid,Test}_Tr0p80-Val0p10-Test0p10-Frame-2x8-to-2x8.txt` lists (the lists are shipped in
  `splits/sirta_2023/`). `preprocessing/` builds the first two from SIRTA's files. SIRTA data is not redistributed
  here.

## Preprocessing

- **Copied from the working tree of the preprocessing repository**, which had uncommitted changes; 9 of the 16
  files were modified after the 2023 dataset was built. Rebuilding January 2023's irradiance with the copied
  code reproduces all 8,277 rows of the dataset (measurements bit-exact, clear-sky quantities to about
  1e-14), so the later edits did not change the result.
- **Parameters made explicit.** Every script had the year hard-coded (to 2021 in the committed versions) and
  the image size to 256; both are now parameters defaulting to 2023 and 64. The committed configuration said
  16 input and 15 target frames; it now says 8 and 8, matching the run log shipped as
  `preprocessing/reference/making-h5-setting_2023_64_Frame-2x8-to-2x8.csv`.
- **Site elevation is a constant, 159.0 m.** The original notebook fetched it from `api.opentopodata.org` at
  run time; that value reproduces the dataset's clear-sky irradiance to about 1e-17, while the 13.0 m left in
  a comment would not. It is now fixed in the notebook, so a rebuild no longer depends on an external service.
- **Images reproduce to within one intensity level, not bit for bit.** Rebuilding 2023-01-03 finds all 243
  timestamps, with a largest pixel difference of 1 (of 255) on both exposures. Two builds from the same raw
  files, even on the same machine with the same package versions, can differ by one intensity level; the
  source of this run-to-run variation in the image step has not been identified. This does not affect
  stored-mode scoring, which uses irradiance only; regenerated forecasts from rebuilt images may differ
  slightly from the shipped ones.
- **Pinned versions** in `preprocessing/requirements.txt` (pvlib 0.15.2, Pillow 11.1.0, pandas 2.2.3,
  numpy 2.1.3, …) are those that produced the results above, not necessarily those used for the original
  build, which were not recorded.

## Smaller discrepancies

- **SkyGPT's settings dump names a 98-sample debug subset.** Its stored forecasts hold the canonical 1998
  samples, and the scorer checks the sample indices, so the reported numbers use the full subset. The
  settings entry is a leftover.
- **Every method is scored at its best-validation checkpoint**, not its final epoch.

## Known and accepted

- **Xiong's `forecast()` needs a non-empty validation split.** Regardless of `mode`, `forecast()` calls
  `self._prepare_gp(dataset, config)` (`models/Xiong.py:525`) before drawing any forecast. Because the
  shipped `configs/xiong.yaml` sets `Xiong.GP.add_observation_noise: true`, `_prepare_gp()` fits an
  observation-noise term on up to 200 batches of the *validation* split (`models/Xiong.py:758-775`); the
  batches are read with `shuffle=False`, so the fit is itself deterministic, not a source of run-to-run
  variation. A validation split with zero windows -- as in `samples/synthetic_2023`, which leaves Valid
  empty by design -- means that batch loop collects nothing, and `torch.cat(res)` (`models/Xiong.py:773`)
  then raises `RuntimeError: torch.cat(): expected a non-empty list of Tensors`. This is why Xiong is the
  one Table 2 method that cannot run on the synthetic set (see `samples/synthetic_2023/README.md`).
- **Verified for inference only.** The proposed model's `train()` and `evaluation()` ship complete but
  have not been exercised in this repository.
- **Importing this codebase no longer requires a GPU; loading checkpoints no longer does either.**
  `losses/Sunset/losses.py` used to call `torch.cuda.current_device()` unconditionally at import time
  (before its own later `torch.cuda.is_available()` guard ever ran), which crashed the whole import chain --
  every method, not only SUNSET, since `main.py` imports every model unconditionally -- the instant no CUDA
  device was visible. Fixed: that call is now itself guarded the same way `losses/ConvLSTM/losses.py`
  already guarded its equivalent line. `import models.Sunset` and importing `eval/run_config.py`'s module
  chain both now succeed with `CUDA_VISIBLE_DEVICES=""`.

  Actually running a method used to fail regardless, for an unrelated reason: `main.py`'s
  `_get_previous_training()` loaded the released checkpoints with `torch.load(generator_name,
  weights_only=False)` and no `map_location`, so PyTorch tried to deserialize the GPU-trained checkpoint
  straight onto a CUDA device and raised `RuntimeError: Attempting to deserialize object on a CUDA device
  but torch.cuda.is_available() is False`. **Fixed**: every reachable `torch.load` on the checkpoint-loading
  path (`main.py:199,212,217,228,241,254,260`; the frozen-regressor loader shared by `models/ConvLSTM.py`,
  `models/PhyDNet.py`, `models/PhyDNetGAN.py` and `models/SkyGPT.py`) now passes `map_location="cpu"`. Every
  one of these results is either consumed by `load_state_dict` (which re-places tensors onto the receiving
  module's own parameters, on whatever device they already live on -- so this cannot change a GPU run's
  result) or, at `main.py:254`/`260`, is a plain Python `epoch` int with no device concept at all. Several
  other `torch.load` call sites already passed a correct `map_location` and needed no change
  (`models/Xiong.py:117,200,730`, `models/Text2ImageJoint.py:220`,
  `models/Text2ImageJoint_utils/checkpoint_io.py:48`); `models/SkyGPT_utils/vqvae.py:106`
  (`load_from_checkpoint`) still has none, but is unreachable -- never called anywhere in this repository.
  Verified bit-for-bit unchanged on the GPU path (183/183 Table 2 cells, the proposed model's two arms
  `np.array_equal` against their `results/` reference on the full 1998-window set, and the synthetic sample
  set matching `expected_synthetic.csv`'s central values exactly) before being trusted.

  **Remaining CPU gap (not `torch.load`): the proposed model's two arms run, but outside tolerance.** On
  the synthetic set, 12 deterministic methods and SkyGPT (whose ensemble is already unseeded by design) all
  score CLOSE on CPU. `propFuture`/`propPast` run to completion but land well outside their tolerance (worst
  cell off by about 89x its tolerance for `propFuture`, 13x for `propPast`) -- not a precision shortfall but
  a genuinely different random draw. `models/Text2ImageJoint.py:138-139` seeds once per ensemble member with
  `torch.manual_seed`/`torch.cuda.manual_seed_all`, then `torch.randn_like(...)` (`models/Text2ImageJoint.py:374,378`)
  draws from whichever default generator the tensor's own device uses; PyTorch's CPU and CUDA default RNGs
  are different algorithms, so the same integer seed produces different numbers on each, and 50 diffusion
  steps compound the difference. Not fixed here (would mean changing how the ensemble is seeded, not just
  adding a `map_location`); tolerances were not loosened to paper over it.
- **Unreachable files retained:** `models/SkyGPT_utils/data.py` and `vgg.py`, kept so the upstream
  module stays whole.
- **`--force` in regenerate mode** is tested on SUNSET only. It works by switching off `load_ckpt` so a
  fresh output folder is created; SkyGPT's nested `load_ckpt` setting is handled but untested.

## Scope

The release reproduces Table 2. Other tables in the paper come from runs outside this scope, including
two that can no longer be re-run: the Appendix B design-space survey was scored at 20 sampling steps by a
script that no longer exists, and four checkpoints of the scalar-only ablation arm, including the one
reported, are lost.

## Licences

- The proposed model's weights derive from Stable Diffusion v1-5 and are subject to the CreativeML Open
  RAIL-M licence (see `NOTICE`).
- SkyGPT is included under its upstream licence (see `NOTICE`).
