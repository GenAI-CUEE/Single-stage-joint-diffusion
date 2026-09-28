# preprocessing

Builds the dataset folder the models read, `sirta_2023_image64-localtime_ICHP-daytime-dhi`, from SIRTA's raw
files. The code is copied from the preprocessing repository that built the paper's dataset; the only changes
are that year, image size and output locations are parameters, the site elevation is a constant, and the FTP
logins are read from the environment (see `PORTING_NOTES.md`, Preprocessing).

Needed only to rebuild the data. Reproducing Table 2 from the stored forecasts needs the built folder, not
this code's intermediate files.

## 1. Get the raw data from SIRTA

SIRTA distributes its data itself; it is not redistributed here. FTP access to the irradiance and camera
archives is provided by SIRTA on request. Set the two logins it gives you:

```
export SIRTA_FTP_IRRAD_USER=<irradiance login>
export SIRTA_FTP_CAM_USER=<camera login>
```

Then, for 2023:

```
bash preprocessing/fetch/download_radflux.sh   2023 <raw_root>/radflux
bash preprocessing/fetch/download_skyimages.sh 2023 <raw_root>/srf02
```

About 125 MB of irradiance files and 43 GB of zipped images (full resolution only). The irradiance product
used is `radflux_1b_Lz2M1minIsolys2PrayDp-QC_v01` (one-minute, UTC); the scripts mirror every file in each day
folder, and the extraction step picks this one.

The scripts expect the downloads under the names `SIRTA_irradiance/` and `SIRTA_skyimages/` in the directory
they run from; symbolic links to `<raw_root>/radflux` and `<raw_root>/srf02` are enough.

Both scripts also take optional `[month] [day]` arguments to mirror just one month or one day instead of the
whole year -- for example, `bash preprocessing/fetch/download_radflux.sh 2023 <raw_root>/radflux 04 01` fetches
only 2023-04-01.

**Downloading today gives a slightly different record.** In February 2025 SIRTA re-issued the 2023 irradiance
files, marking as unusable some minutes that were still provisional when the paper's copy was made. The build
then lacks 86 of the 1998 test samples used for Table 2; the scoring script detects this and compares against
`prediction_samples/expected_rebuild.csv` instead. See `prediction_samples/README.md`. The other 1998-86=1912
keep their identity (first-target timestamp) but are not all guaranteed to be built from the same rows as the
paper's: a window whose 16-frame span crosses a dropped minute shifts by a row despite matching that
timestamp. See `PORTING_NOTES.md`, "Data", for what that changes and how often it happens, and run
`python eval/check_window_alignment.py --data-folder preprocessing/sirta_2023_image64-localtime_ICHP-daytime-dhi`
(the folder step 2 below writes, before you place or link it in the repository root per `README.md`'s Data
step) against your own build to see exactly which windows match.

## 2. Build, in order

| Step | File | Does |
|---|---|---|
| 1 | `Step0_0_Extract_downloaded_image.py` | unzips each day's image archive |
| 2 | `Step0_1_Extract_irradiance.py` | reads the one-minute irradiance files (GHI, DHI and three meteorological columns) |
| 3 | `Step0_2_Clean_irradiance.py` | sets negative GHI to 0; drops any row with a missing value |
| 4 | `Step1_irradiance_and_Clear-sky_irradiance.ipynb` | clear-sky irradiance (pvlib, Ineichen–Perez), UTC → Europe/Paris, daytime window, daily sky condition |
| 5 | `Step2-1_Extract_image_and_local_datetime-pickle.py` | masks and crops the fisheye image, resizes to 64×64, joins images and irradiance by timestamp; writes `sirta_data_2023.csv` and `sirta_image_2023.pickle` |
| 6 | `Step2-2_remove_douplicates.ipynb` | removes duplicate timestamps from both |
| — | `Step4_Spliting_…_for_testing_valid_training.py` | **do not run**: it regenerates the split lists. Copy the three lists from `splits/sirta_2023/` into the dataset folder instead. |

**Environment.** Install the pinned packages into their own virtual environment, then register that
environment as a Jupyter kernel (the notebook steps below need to run under it, not under whatever kernel
happens to be the system default):

```
python3 -m venv <preproc_venv> && <preproc_venv>/bin/pip install -r preprocessing/requirements.txt ipykernel nbconvert
<preproc_venv>/bin/python -m ipykernel install --user --name sirta-preprocessing --display-name "SIRTA preprocessing"
```

**Run every step from inside `preprocessing/`** (`cd preprocessing`): `configs/`, `sirta_utils/` and
`solar_utils/` are already there, so nothing besides the two data folders needs linking. Link (or copy) the raw
downloads under the names the scripts expect, from inside `preprocessing/`:

```
cd preprocessing
ln -s <raw_root>/radflux SIRTA_irradiance
ln -s <raw_root>/srf02   SIRTA_skyimages
```

Then, with year 2023 and (where relevant) image size 64, still from inside `preprocessing/`:

1. `<preproc_venv>/bin/python Step0_0_Extract_downloaded_image.py 2023`
   -- unzips into `SIRTA_skyimages_unzip/`.
2. `<preproc_venv>/bin/python Step0_1_Extract_irradiance.py 2023`
   -- writes `SIRTA_irradiance_solys2_ICPH-daytime/2023.csv`.
3. `<preproc_venv>/bin/python Step0_2_Clean_irradiance.py 2023`
   -- writes `SIRTA_irradiance_solys2_ICPH-daytime/2023-cleaned.csv`.
4. ```
   <preproc_venv>/bin/jupyter nbconvert --to notebook --execute Step1_irradiance_and_Clear-sky_irradiance.ipynb \
     --output-dir <a folder outside the repo> --ExecutePreprocessor.kernel_name=sirta-preprocessing
   ```
   (set `Year_` in cell 2 first if not 2023) -- never `--inplace`: this writes the executed copy, with
   outputs, to `<a folder outside the repo>` and leaves the notebook in `preprocessing/` as shipped. Writes
   `SIRTA_irradiance_solys2_ICPH-daytime/2023-cleaned_sky-cond.csv`.
5. Headless machines need a non-interactive matplotlib backend, or this step aborts trying to save a preview
   plot:
   ```
   export MPLBACKEND=Agg
   <preproc_venv>/bin/python Step2-1_Extract_image_and_local_datetime-pickle.py 2023 64
   ```
   Writes `sirta_2023_image64-localtime_ICHP-daytime-dhi/sirta_data_2023.csv` and
   `sirta_2023_image64-localtime_ICHP-daytime-dhi/sirta_image_2023.pickle` (plus periodic preview PNGs under
   `Step2_Extract_image_and_local_datetime-pickle-2023/`).
6. ```
   <preproc_venv>/bin/jupyter nbconvert --to notebook --execute Step2-2_remove_douplicates.ipynb \
     --output-dir <a folder outside the repo> --ExecutePreprocessor.kernel_name=sirta-preprocessing
   ```
   (set `YEAR`/`IMSIZE` in cell 0 first if not 2023/64) -- overwrites the same `sirta_data_2023.csv` /
   `sirta_image_2023.pickle` in place, with duplicate timestamps removed.

The final `sirta_data_2023.csv` and `sirta_image_2023.pickle` end up under
`preprocessing/sirta_2023_image64-localtime_ICHP-daytime-dhi/`.

`configs/SIRTA_with_irr_setting_2023_64.yaml` and `reference/making-h5-setting_2023_64_Frame-2x8-to-2x8.csv`
record the settings of the paper's build (8 input and 8 target frames at two-minute cadence, 05:00–22:00).

## What to expect

- **Irradiance** reproduces the paper's dataset exactly: tested on January 2023, all 8,277 rows, measurements
  bit-exact and clear-sky values to about 1e-14.
- **Images** reproduce to within one intensity level of 255, not bit for bit (tested on 2023-01-03). Two
  builds from the same raw files, even on the same machine with the same package versions, can differ by one
  intensity level; the source of this run-to-run variation in the image step has not been identified.
  Stored-mode scoring is unaffected; forecasts regenerated from rebuilt images may differ slightly.
- **Known properties kept on purpose**, because changing them would change the dataset the paper used: a row
  with any missing value is dropped, even in a column no model reads; a window is dropped if any of its 16
  frames is missing; a window can span a night or a skipped frame (13 of the 1998 test samples; paper
  Section 4.1); and the split generator is chronological within each month. See `PORTING_NOTES.md`.
