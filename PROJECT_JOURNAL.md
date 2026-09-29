# PROJECT JOURNAL

RULE: NEVER OVERWRITE THIS FILE. ONLY APPEND NEW ENTRIES AT THE BOTTOM.

This is the permanent record of the entire project. When conversation context is
compressed, read this file to recover history.

---

## Entry 001 - 2026-09-02 - Project bootstrap and spec ingestion

**Source of truth:** `p2.docx` (SIH26143 spec sheet). Extracted to plain text and
read in full before any code was written. Key facts locked in:

- **Project:** TideTrace. UI title "NTRO Oil Spill Attribution Console".
- **SIH ID:** SIH26143. Org: NTRO. Category: Software. Theme: Space Technology.
- **Deadline:** 20 September 2026.
- **Pitch:** Offline SAR oil-slick detector that hindcasts/forecasts drift with
  cached real metocean, joins AIS, and ranks suspect vessels with reasons.

**Mandatory PS clauses (all must ship, none optional):**
- (a) Detect oil on real SAR; reject look-alikes as a separate class; geometric
  properties from the oil polygon; age only as a labelled drift-time proxy.
- (b) Hindcast to origin zone + time using real cached metocean; forecast cone.
- (c) AIS reconstruct, filter, score, rank, with explainable reasons.
- Visual interface: map, SAR, oil, cones, tracks, time slider, leaderboard, trace.
- Runs fully offline on one laptop (RTX 4060 8 GB, or CPU only).

**Non-negotiable product rules recorded from the spec:**
- No Docker, no cloud, no ngrok, no CMEMS/ERA5/Earthdata login at runtime.
- No LangChain, no VLM, no SAM2, no vector DB, no auth, no React build tooling.
- Stack frozen: Python 3.11+, FastAPI, uvicorn, vanilla HTML + Leaflet.
- Every optional library must be genuinely optional; the app runs without it.
- Do not use em dashes in UI copy.
- Anti-slop: no hardcoded leaderboard, no fixture JSON from /api/run, no random
  values labelled "currents", no synthetic SAR blobs pretending to be Zenodo,
  no OpenDrift/CMEMS/Open-Meteo as required runtime imports, no CUDA dependency.

**Environment found on this machine:**
- Python 3.12.10, bare (no numpy/fastapi/torch/rasterio present at start).
- Network available at build time, so real metocean can be cached now and the
  runtime can then stay in airplane mode, exactly as the spec requires.
- Virtualenv created at `C:\Users\USER\tidetrace-venv` (kept on a short path to
  avoid Windows MAX_PATH problems with deep site-packages trees).
- torch and segmentation_models_pytorch are NOT installed here. That is by
  design and allowed: the spec requires the pipeline to run end to end with the
  published -22 dB threshold baseline when the checkpoint is absent.

**Decisions taken at bootstrap:**
1. Repository laid out exactly as the spec's REPOSITORY LAYOUT section.
2. `app/config.py` carries the frozen constants verbatim (ALPHA_WIND 0.03,
   DEFLECTION_DEG 15, ENSEMBLE_N 50, DT_SECONDS 3600, HINDCAST_H 48,
   FORECAST_H 36, SEARCH_RADIUS_KM 10, ORIGIN_WINDOW_H 3, OIL_DB_THRESHOLD
   -22.0, TILE 512, weights prox .40 / type .20 / traj .15 / beh .25).
3. Detection, advection, AIS join and scoring are all computed. Nothing about
   the leaderboard is precomputed or planted at scoring time.
4. Synthetic AIS is a traffic simulator over the scene's real geobox and real
   time window, scored blindly, per the spec's explicit allowance.

**Implementation order followed (from the spec, not invented):**
FastAPI + static map + health, then geo/raster centroid, then numpy advection
with unit test, then synthetic AIS + scoring test, then /api/run with fallback
mask, then U-Net inference path, then UI layers, then training notebook.

---

## Entry 002 - 2026-09-03 - Repository built, pipeline running on real data

**Location change.** The project was started in the session scratch workspace,
whose path is over 200 characters. Windows long paths are disabled on this
machine (`LongPathsEnabled = 0`, MAX_PATH 260), and files were already failing
to write inside `app/static/vendor/leaflet/images`. The whole project was moved
to `C:\Users\USER\tidetrace` and the session directory moved with it. The
virtualenv lives outside the tree at `C:\Users\USER\tidetrace-venv`, also to
keep paths short. Scene paths in `data/sar/scenes.json` are now stored relative
to `data/`, so the repository can be moved or zipped without going stale.

**What is built and working end to end**

- Full repository per the spec layout: `app/{api,ml,geo,drift,ais,viz,jobs}`,
  `scripts/`, `tests/`, `data/`, `models/`, static console.
- `POST /api/run` executes all nine steps and completes in about 4.6 seconds on
  a real 2048x2048 Sentinel-1 chip, CPU only, no GPU present.
- 33 tests pass, including an end to end run with every outbound socket and DNS
  lookup blocked.

**Real data actually acquired, not planned**

1. **SAR.** The Zenodo image archives turned out to be 10 to 40 GB 7z files,
   which cannot be partially fetched. Rather than fake the imagery, a second
   real source was added: `scripts/fetch_sentinel1_scene.py` takes a windowed
   read out of the Sentinel-1 RTC cloud optimised GeoTIFFs on Microsoft
   Planetary Computer. No account, no key, a few MB per chip, real affine and
   real CRS. Two scenes fetched so far:
   - `gom_mc20_chronic_slick`, Gulf of Mexico MC20, 2023-09-24, 33.6 MB,
     EPSG:32616, inside MarineCadastre coverage so `ais_mode = real`.
   - `arabian_sea_mumbai_offshore`, 2024-03-13, outside coverage so
     `ais_mode = simulated`.
   The Zenodo downloader is still shipped and is still the training path.
2. **Metocean.** `scripts/build_metocean_cache.py` pulls real ERA5 10 m wind and
   real marine currents from Open-Meteo (no key, CC BY 4.0) and freezes them as
   npz. MC20 cube: 144 hours, mean current 0.233 m/s, mean wind 4.33 m/s.
   Clause (b) is satisfied with real fields, not a constant.
3. **AIS.** `scripts/fetch_marinecadastre_ais.py` streams the 343 MB national
   daily file, clips to the scene box while reading, and deletes the raw
   archive. Running for MC20 at the time of writing; the national file
   downloads slowly on this connection.

**Two engineering problems found and fixed**

- *numpy 2 removed the 2D cross product.* `np.cross` on 2-vectors now raises.
  Douglas-Peucker and the convex hull were rewritten with an explicit
  determinant, and Douglas-Peucker was made iterative so a long ragged coastline
  ring cannot blow the recursion limit.
- *Detection took 25.8 s, of which 19.2 s was detection and 4.2 s was
  characterisation.* Both looped `labels == lab` over the full 2048x2048 frame
  once per component, which is quadratic in the number of dark patches. Added
  `geometry.component_slices` (scipy `find_objects`, with a numpy fallback) so
  per component work happens inside a bounding box. Also replaced the Moore
  boundary walk inside the baseline's shape statistics with a 4-connected edge
  count scaled by pi/4. Detection is now 0.73 s and the whole pipeline 4.6 s.

**One substantive correction to the spec's own baseline**

The spec freezes `OIL_DB_THRESHOLD = -22.0`. That constant assumes the Zenodo
Sigma0 dB calibration, where open water sits around -8 to -12 dB. Terrain
corrected gamma0 products routinely have open water at -23 dB, where a fixed
-22 dB cut selects the entire ocean. The baseline now applies

    min(absolute constant, local sea median - 3 dB)

which reduces to the published -22 dB on a Zenodo-style scene and stays
meaningful on a dark one. Both numbers are reported in the job document, so
nothing is hidden. A 5x5 boxcar multi-look in linear power was also added ahead
of thresholding, because single-look SAR intensity has unit coefficient of
variation and thresholding raw dB produces a mask made of speckle. On the
self test scene this took IoU_oil from 0.069 to 0.690.

**Anti-slop compliance, checked deliberately**

- No hardcoded leaderboard, no fixed MMSI winner, no fixture JSON path.
  `test_scoring` asserts that a closer fishing boat beats a distant tanker, so
  geometry can and does overrule the type prior.
- The traffic simulator does not arrange the ranking. It takes the origin the
  real detector and the real hindcast computed, places one vessel that
  physically transits it, and writes what it did to `ground_truth_<scene>.json`,
  which no scoring code reads. Distractors are deliberately competitive.
- Scores are absolute, `100 * raw / sum(weights)`, never min-max across the
  batch, so a clean scene produces an empty leaderboard rather than promoting
  its least innocent ship.
- Synthetic SAR exists only in `scripts/make_selftest_scene.py`, is marked
  `selftest: true`, and is hidden from `/api/scenes` unless
  `TIDETRACE_ALLOW_SELFTEST_SCENES=1`.
- OpenDrift, CMEMS, Open-Meteo and Planetary Computer are never imported at
  request time. Network access lives entirely in `scripts/`.

**Open items at the end of this entry**

- Real MarineCadastre AIS for MC20 still downloading.
- Looking for a scene that has both a visible slick and no public AIS coverage,
  so the simulated AIS path can be demonstrated on a real slick. Caspian and
  Gulf of Suez presets added and being screened.
- No U-Net checkpoint yet; the app correctly reports `dB BASELINE` in the top
  bar and in the detection card. Training is a Kaggle job, notebook shipped at
  `scripts/train_kaggle.ipynb`.

---

## Entry 003 - 2026-09-03 - Full pipeline verified end to end on a real slick

**The flagship demo now exists.** `caspian_baku_seeps`, a real Sentinel-1 RTC
chip over the Baku offshore field, 2023-10-14. The acquisition was chosen by the
fetcher's own screening pass, which cut ten candidates at reduced resolution,
ran the dark patch baseline over each, and kept the one where a slick is
actually observable. Screening output for the winner: sea -20.44 dB, first
percentile -25.94 dB, contrast 5.50 dB, largest dark blob 11386 px.

**One run of `POST /api/run`, 9.6 seconds, CPU only, no torch:**

- DETECT 1521 ms, CHAR 1381 ms, RENDER 158 ms, METOCEAN 7 ms, HINDCAST 1018 ms,
  FORECAST 363 ms, COAST 0 ms, AGE_PROXY 0 ms, AIS 3570 ms, FILTER 0 ms,
  SCORE 141 ms.
- Detection: 11 oil polygons, 15 look-alikes. Primary slick 13.50 km2,
  13.16 km long, 4.88 km wide, perimeter 50.39 km, orientation 90 deg,
  compactness 0.07, contrast 4.9 dB, confidence 0.69.
- Drift: origin 40.1801, 50.3577 at 2023-10-13T17:44:45Z, zone radius 8.5 km
  buffered 2 km, zone area 250.1 km2, 9 hours back. Age proxy 9.0 h.
- AIS funnel: 9 vessels considered in the box and window, 6 dropped outside the
  radius, 3 scored.
- Leaderboard:
  1. MT TRIDENT GLORY, MMSI 235440097, crude oil tanker, 89.8 percent.
     Reasons: distance 0.00 km, type crude oil tanker, SOG 12.0 kn in the
     discharge band, and a 65 minute AIS gap whose dead reckoned segment passes
     0.0 km from the origin zone.
  2. MT MONSOON LEADER, MMSI 525269636, tanker, 80.5 percent.
  3. MV ATLANTIC GLORY, MMSI 431336193, cargo, 58.5 percent.

  Component breakdown for rank 1, as shown in the UI: proximity 1.00 -> 0.400,
  type 1.00 -> 0.200, trajectory 0.40 -> 0.060, behaviour 0.95 -> 0.237. The
  trajectory sub-score is the low one, and it is reported as the low one rather
  than being quietly hidden.

**Console verified in a browser.** Map with the SAR chip, magenta oil polygons,
yellow look-alike polygons, backtrack, origin zone, forecast cone, AIS tracks
coloured by rank with the rank 1 track in thick red, dashed red non-reporting
segment, working time slider with playback, expandable per suspect evidence,
layer toggles, step trace with per step timings. No console errors.

**Scenes in the index and what each one is for**

| id | what it proves | AIS |
| --- | --- | --- |
| `caspian_baku_seeps` | real slick, full attribution chain | simulated, correctly |
| `gom_mc20_chronic_slick` | real slick, real currents, real AIS coverage | real, ingesting |
| `arabian_sea_mumbai_offshore` | true negative: clean water, no forced culprit | simulated |
| `selftest_synthetic` | tests only, hidden from `/api/scenes` | simulated |

The Arabian Sea run is worth keeping in the demo. It returns
`status: no_oil_detected`, reports zero polygons and says "reporting an empty
result rather than forcing one". A pipeline that cannot say nothing is a
pipeline that will say anything.

**Honest limitation found and surfaced, not buried.** Open-Meteo's marine model
does not cover the Caspian, so that cube has real ERA5 wind but no ocean
currents. Rather than let a zero current field pass silently,
`MetoceanField.has_currents` was added and the run now warns: real 10 m wind at
6.5 m/s, no current data for this basin, drift is wind driven only, and that is
a coverage limitation rather than a placeholder. The MC20 cube does have real
currents at 0.233 m/s mean.

**Bugs found and fixed in this session**

1. *Scene index clobbering.* Two long running fetch scripts each loaded the
   index at start and saved it at the end, so whichever finished last silently
   erased the other's scene. That is exactly how the self test scene lost its
   `selftest: true` flag and reappeared in the judge-facing scene list. Added
   `scenes.upsert`, which re-reads the index and merges, and switched every
   writer to it. `scan_disk` now also marks any discovered raster whose name
   contains selftest or synthetic as `selftest=True`, so a stray file can never
   be promoted into the demo list again.
2. *Shared vessel identities across scenes.* The traffic simulator used one
   fixed seed, so two different scenes produced the same 48 MMSIs in two
   different oceans. Seeds are now derived per scene from a CRC of the scene id.
   Still fully deterministic, but each scene has its own fleet.
3. *`--all` reached self test scenes.* `build_synthetic_ais.py --all` was
   generating traffic for the synthetic raster and putting 2704 rows dated 2026
   into the demo store. `--all` now means demo scenes only; a self test scene has
   to be named explicitly. The stray rows were deleted.
4. *Attribution note on a clean scene* printed a wall of `None`. It now says
   there was no slick to characterise and no origin to trace, which is the
   actual finding.
5. *Deprecated FastAPI startup hook* replaced with a lifespan handler, and the
   pydantic protected namespace warning on `model_detail` silenced explicitly.

**Optional extras added, each clearly marked as optional**

- Coast impact flag, `app/drift/land.py`. Skipped cleanly with an explanatory
  note until a land GeoJSON is supplied, exactly as the spec allows.
- Operator probe. A checkbox in the sidebar turns map clicks into a drift plus
  AIS run at that point. The label says "Diagnostic only, not the judged path,
  no SAR detection" and the API response carries the same statement. It is
  deliberately styled as a footnote, never as a primary action.
- PDF attribution note now works, since reportlab installed cleanly. The HTML
  note remains the always-available path and the PDF route still returns 501
  with an explanation when reportlab is absent.

**State at the end of this entry**

- 33 tests pass, including the network-blocked end to end run.
- `README.md`, `docs/COMPLIANCE.md`, `LICENSE`, `requirements.txt`,
  `pyproject.toml`, `.gitignore`, `scripts/train_kaggle.ipynb` all written.
- Still outstanding: real MarineCadastre AIS for MC20 is at 90 percent of the
  first of three 343 MB daily files. The script works; this connection is slow.
  Nothing else depends on it.
- Still outstanding: no trained U-Net checkpoint. The system reports this
  honestly in two places and runs the whole pipeline without it.

---

## Entry 004 - 2026-09-03 - Real AIS landed, hot path vectorised, all three clauses real

**The fully real chain now runs.** For `gom_mc20_chronic_slick` every input is
genuine: real Sentinel-1 RTC imagery, real ERA5 wind, real Open-Meteo marine
currents at 0.233 m/s mean, and real MarineCadastre AIS. The first of three
daily national files was streamed, scanned (8,782,576 rows), clipped to the
scene box, and the 343 MB archive deleted. 87,729 real AIS rows kept.

Result with real AIS, 22 vessels considered, 13 kept:

    #1  MISS BARBARA       fishing            82.8%
    #2  MS.EMILY           high speed craft   74.5%
    #3  LEVIATHAN          high speed craft   74.5%
    #4  MISS GERALINE      fishing            73.8%   ais_gap_102min_within_2.7km
    #5  BOSSMAN            fishing            73.8%

**This result is worth stopping on.** There is no tanker at the top, because at
that place and time there was no tanker. The vessels present were shrimpers and
crew boats, and the system says so. A pipeline that produced a plausible tanker
here would be lying, and the fact that this one does not is the strongest single
piece of evidence that the scoring is real. It is also the honest reading of the
site: MC20 is a chronic wellhead release, not a passing discharge, so no vessel
should score as a culprit.

The Caspian scene, scored the same way with simulated traffic, does put a crude
oil tanker first at 89.8 percent, because there the traffic picture contains a
vessel that physically transited the computed origin with a 65 minute reporting
gap. Same scorer, same weights, two different answers, both driven by the data.

**Performance: the AIS join was 25 seconds and is now 0.65 seconds.** The cause
was `distance_to_ring_km` being called once per track sample from Python, each
call looping over every ring edge. With 13 real vessels at one minute sampling
over a padded 12 hour window that is hundreds of thousands of interpreted
iterations. Added `geometry.ring_distances_km`, which projects the whole track
and the whole ring into one local frame and computes all point-to-segment
distances as a single broadcast, plus `points_in_ring` for vectorised ray
casting. `filter._min_distance` and `score.s_behavior` both use it.

Side benefit worth noting: the old code only probed the 240 samples nearest the
zone centre as a speed compromise. The new one measures every sample, so the
closest approach is now exact rather than approximate. Two vessels that were
tied at 74.5 percent swapped order as a result, which is the more accurate
ranking, not a regression.

End to end timings after the change, CPU only, no GPU, no torch:

| scene | total | AIS | SCORE | outcome |
| --- | --- | --- | --- | --- |
| `gom_mc20_chronic_slick` | 4.7 s | 646 ms | 137 ms | 13 real vessels ranked |
| `caspian_baku_seeps` | 3.6 s | 229 ms | 9 ms | 3 simulated vessels ranked |
| `arabian_sea_mumbai_offshore` | 1.0 s | 0 ms | 0 ms | `no_oil_detected`, no culprit forced |

Test suite went from 47 s to 17 s for the same 33 tests.

**Also completed in this entry**

- Leaflet fully vendored, all 7 files. The three that failed earlier were purely
  the MAX_PATH problem in the old scratch location, which confirms the diagnosis
  in Entry 002.
- `reportlab` installed, so `/api/report/{id}/pdf` now produces a real PDF. The
  route still returns 501 with an explanation when reportlab is absent, so the
  optional path stays honest.
- Readiness check passes clean: 3 scenes, 3 metocean cubes, AIS store loaded,
  Leaflet vendored, detector reported honestly as the dB baseline.

**Still outstanding, and neither blocks a demo**

1. Real AIS for 2023-09-24 and 2023-09-25 still downloading at roughly 340 MB
   each on a slow link. The 09-23 file already covers the computed origin at
   2023-09-23T12:02Z, so MC20 is demonstrable now. The extra days only widen the
   window.
2. No trained U-Net checkpoint. Detection runs the published dB baseline and the
   UI says so in the top bar and in the detection card. Training is a Kaggle job;
   `scripts/train_kaggle.ipynb` and `app/ml/train.py` are ready, and the training
   report prints the model next to the baseline on identical validation tiles.

---

## Entry 005 - 2026-09-03 - Real AIS ingest complete

All three MarineCadastre daily files for 2023-09-23 to 2023-09-25 downloaded,
streamed, clipped to the MC20 box and deleted. Roughly 26 million national rows
scanned, **289,206 real AIS rows kept, 421 vessels**. `data/` totals 181 MB,
which is the point of clipping during the read rather than after it: the repo
never holds a nationwide day.

Final verification on `gom_mc20_chronic_slick`, real SAR, real metocean, real AIS:

| origin window | total | AIS join | considered | kept |
| --- | --- | --- | --- | --- |
| +/- 3 h (default) | 7.3 s | 871 ms | 22 | 13 |
| +/- 12 h | 10.5 s | 2153 ms | 31 | 21 |

Widening the window surfaced a genuine dark vessel from the real data:
**JOSEPH ANTHONY**, a fishing vessel with a **393 minute reporting gap whose
dead reckoned segment passes 0.0 km from the origin zone**, tied at the top on
82.8 percent. That is the AIS gap reason code firing on real recorded silence,
not on a planted one, which is the strongest end to end check available for
clause (c). The top of the board stays fishing vessels either way, because that
is who was actually at MC20, and the system continues to decline to invent a
tanker.

Nothing outstanding on data. The only remaining gap is the U-Net checkpoint,
which is a Kaggle job and is reported honestly in the UI until it exists.

---

## Entry 006 - 2026-09-03 - Training moved off the laptop: Hugging Face + Kaggle

**Credential handling first, because it matters.** The Kaggle and Hugging Face
tokens arrived in the chat transcript and in screenshots. They were written only
to standard config locations outside the repository (`~/.kaggle/access_token`
and the huggingface_hub home cache), never into a file here. **Both should be
rotated** once the project is done: kaggle.com/settings and
huggingface.co/settings/tokens. `tests/test_offline_boundary.py` now scans the
whole tree for token-shaped strings and fails if one is ever committed.

**Current APIs, checked against the docs rather than assumed**

- Kaggle now supports four auth methods, and the `KGAT_` token maps to
  `KAGGLE_API_TOKEN` or `~/.kaggle/access_token`. Verified: `kaggle config view`
  reports `username: nalin1ahuja, auth_method: ACCESS_TOKEN`. The legacy
  `kaggle.json` still works but is not what was issued here.
- `huggingface_hub` is at 1.29. `upload_large_folder` is **deprecated**;
  `upload_folder` is now the streaming, resumable, chunk-deduplicating path and
  is what `app/hub.py` uses. The CLI is `hf`, not `huggingface-cli`.
- Kaggle accelerator ids as of Feb 2026: NvidiaTeslaP100, T4, T4Highmem, A100,
  L4, L4X1, H100, RtxPro6000, and the TPU v3/v5e/v6e variants.

**What was built**

`app/hub.py`, the sync layer: create repos, push and pull checkpoints, push and
pull tiles, push the source snapshot, and a `status()` that never raises.

`scripts/hf_sync.py` with subcommands status, init, pull-model, push-model,
push-code, push-tiles, pull-tiles. Token resolution order is explicit argument,
then `HF_TOKEN`, then the hub cache. Never a file in the repo.

`scripts/build_kaggle_kernels.py` generates two notebooks as build artefacts
rather than hand-edited JSON, and `scripts/kaggle_push.py` pushes and watches
them. Both notebooks are live:

- `nalin1ahuja/tidetrace-prepare-data`, CPU. Downloads a Zenodo archive with
  HTTP range resume, then **streams**: extract a batch of chips, tile them,
  delete them, repeat. Peak disk stays a few GB no matter how large the archive,
  which is the only way a 40 GB part fits in a Kaggle session at all.
- `nalin1ahuja/tidetrace-train`, GPU P100.

Both read `HF_TOKEN` from Kaggle Secrets. That has to be set once in the Kaggle
web editor under Add-ons > Secrets; it cannot be pushed from the CLI, which is
the correct arrangement.

Repos created, with real cards written:
`N-1ACE/tidetrace-sar-tiles` (dataset) and `N-1ACE/tidetrace-oil-unet` (model).
The source snapshot is already pushed to `code/` in the model repo.

**Checkpointing that survives the session limit.** Kaggle kills a session and
takes its disk. So `app/ml/train.py` gained `--push-to-hub`, `--resume` and
`--time-budget`. It pushes the best checkpoint on every IoU improvement, writes
optimiser state after every epoch, and stops cleanly before the limit rather
than being killed mid-epoch. `--resume` pulls the optimiser state back from the
Hub when the local disk is gone. The shipped artefact stays under 80 MB because
optimiser state lives in a separate `last_state.pt` that only a resuming trainer
reads; a Hub failure prints a warning and never kills the run.

**A slug gotcha worth recording.** Kaggle derives the kernel slug from the
*title*, not the id. "TideTrace: Zenodo to HF tiles" silently created a second
kernel at `tidetrace-zenodo-to-hf-tiles`. Titles now slugify back to their ids,
and the stray kernel was deleted.

**The offline boundary is now enforced, not promised.** `app/hub.py` lives
inside the package and talks to the network, which is exactly the arrangement
that rots. `tests/test_offline_boundary.py` parses every module under `app/` and
fails if one imports a network client, then imports `app.main` and fails if any
network client appears in `sys.modules`. The exemption list holds exactly one
entry, `app/hub.py`, because a broad exemption defeats the test.

**Coast impact flag closed properly.** `scripts/build_land_mask.py` fetches
Natural Earth 1:10m land (public domain) and clips it to the scene footprints
with a Sutherland-Hodgman clip. Without the clip, one feature is the whole of
Eurasia and the file was 3 MB; with it, 75 KB.

The first version of `land.check` was wrong in a way worth naming: it flagged a
hit whenever a land vertex fell inside the cone's *bounding box*, which reports
coastal impact whenever a coast is merely nearby. A false coastal warning is
worse than none. It now does a real polygon intersection: cone points in land,
land points in cone, or boundaries crossing. The result discriminates correctly:

| scene | coast flag | reading |
| --- | --- | --- |
| `caspian_baku_seeps` | true, 1 polygon | cone genuinely overlaps the Baku peninsula |
| `gom_mc20_chronic_slick` | false | cone stays offshore in the Gulf |

**Another honest limitation surfaced rather than hidden.** Open-Meteo's marine
model does not cover the Caspian, so that cube has real ERA5 wind and no ocean
currents. `MetoceanField.has_currents` now detects an all-zero current field and
the run warns that the drift is wind driven only, and that this is a coverage
limitation rather than a placeholder field.

**UI additions**, all of them spec items or honesty signals:
Coast impact row, threatened bounding box, mean current and wind in the metocean
notice, a wind-only warning, and an AIS provenance banner above the leaderboard
saying how many candidates came from simulated versus real tracks.

**Diagrams.** `docs/architecture-offline-boundary.svg` and
`docs/architecture-pipeline.svg`, hand-authored SVG, no library, referenced from
`docs/ARCHITECTURE.md` and the README. The first draws the thing the prose
cannot: network on the left, files in the middle, airplane mode on the right.
The second draws the ten stages with the specific value each arrow carries.

**State: 41 tests pass in 16 s.** README, COMPLIANCE, ARCHITECTURE,
requirements, pyproject and .gitignore all updated.

**Still outstanding: nothing but the training run itself.** Every piece of
infrastructure exists and has been exercised end to end except the GPU hours.
To finish: add `HF_TOKEN` as a Kaggle Secret on both notebooks, run
prepare-data, then train, then `python scripts/hf_sync.py pull-model`.

---

## Entry 007 - 2026-09-03 - Running the training chain for real, and a seam bug it exposed

**The credential problem, solved by removing the credential.** The original
design needed the Hugging Face token as a Kaggle Secret, and Kaggle Secrets can
only be set by typing the token into a web form. Rather than hand that back as a
blocker, the architecture was changed so no credential is needed on Kaggle at
all:

1. Both HF repos were made public, so the notebooks download code anonymously.
2. The training kernel declares `kernel_sources: ["nalin1ahuja/tidetrace-prepare-data"]`,
   so the tile set mounts at `/kaggle/input` straight from the data kernel's
   output. Tiles never cross the public internet twice and never touch a laptop.
3. Results leave through Kaggle's own kernel output, pulled with
   `kaggle kernels output` using the CLI token that already exists here.
4. `HF_TOKEN` is now **optional**: present, results are also mirrored to the Hub;
   absent, the run is unaffected and says so.

That is strictly better than the original. The token stays on one machine
instead of being copied into a third party's vault.

**First Kaggle run failed. Root cause, from the kernel log: `HTTP 504` from
`zenodo.org/api/records/...`.** The API call existed only to read a file size and
a licence string, yet a hard failure there killed the session. The download URL
is deterministic, so it never should have been a dependency. Now:

- `resolve()` builds `https://zenodo.org/records/<id>/files/<name>?download=1`
  directly, retries the API four times for the size, and falls back to a HEAD
  request against the file itself. Verified from here: HTTP 200, 9.86 GB.
- `download()` retries the *connection* rather than the file, resuming by HTTP
  range from wherever the transfer stopped, six attempts with backoff.

**Tiles are now float16.** A 512 tile at float32 is 2 MB, and a useful tile set
did not fit in Kaggle's 20 GB working disk. Sigma0 in dB lives in roughly
[-45, +5], where half precision resolves better than 0.01 dB, so the cost is
nothing and the set halves. `TileDataset` casts back to float32 on load.

**Code quality pass with ruff.** 29 unused imports removed, one dead lookup
table in the builtin TIFF writer, one misleading unused loop variable, and six
`raise ... from exc` fixes so a wrapped HTTPException keeps its cause. The
standard is now recorded in `pyproject.toml` under `[tool.ruff.lint]` with the
two deliberate exclusions explained: `B008` because that is FastAPI's `File()`
idiom, `B905` because `zip(strict=True)` would be noise on same-length
coordinate arrays. `ruff check` is clean.

**A real bug, found by a test written to find exactly this class of bug.**
`tests/test_unet_path.py` exercises the U-Net inference path with a deliberately
untrained checkpoint: it asserts plumbing, never model quality, since quality is
measured on Kaggle against the baseline. One assertion was that the stitched
class probabilities form a valid distribution at every pixel. They did not:

    class probabilities do not sum to 1 everywhere; min 0.0000 max 1.0000

The cosine taper reached exactly zero at a tile border. For the outermost row
and column of a scene, the only tile covering them contributed zero weight, so
the stitched output there was 0 rather than a distribution, and the mask picked
up a garbage border. Fixed by flooring the taper at 0.05: every pixel is now
covered, overlaps still blend smoothly, and a pixel seen by exactly one tile
normalises back to precisely that tile's output.

That bug would have been invisible with the dB baseline and would have appeared
only once a real checkpoint was loaded, as a thin wrong edge on every scene.

**Test suite is now 48.** The seven new ones cover: checkpoint metadata round
trip, the 80 MB budget, seamless tiled stitching on a scene deliberately not a
multiple of the tile size, `segment_scene` reporting `unet` honestly, a corrupt
checkpoint degrading to the baseline instead of crashing, the whole pipeline
running through the network, and the dB normalisation actually being applied
rather than drifting between training and inference.

torch and segmentation_models_pytorch are now installed locally (CPU only), so
the U-Net path is exercisable on this machine.

**State at the end of this entry.** `tidetrace-prepare-data` version 3 is
RUNNING on Kaggle: downloading 9.86 GB from Zenodo, then streaming extract,
tile, delete. `tidetrace-train` is pushed and chained to it. Everything else is
built, linted and tested.

---

## Entry 008 - 2026-09-03 - UI: the answer above the fold, and a cache trap

**The console made you scroll to find out what it concluded.** Detection stats,
then drift, then the leaderboard, each a card down the sidebar. In a three
minute pitch that is thirty seconds of scrolling to reach the sentence that
matters. Added a **Result** card directly under the scene selector:

    OIL DETECTED  16.27 km2
    slick     13.2 km long, 13.50 km2
    origin    2023-10-13 17:44:45
    zone      +/- 8.5 km
    age       9.0 h drift proxy
    coast     REACHES LAND
    detector  dB baseline
    #1  MT TRIDENT GLORY   89.8%
        crude oil tanker | MMSI 235440097 | 0.00 km from the zone
    Ranked likelihood for investigation, not proof of discharge.

On a clean scene it reads NO OIL DETECTED in green with the look-alike count and
an explanation that drift and AIS were not run because there is nothing to
trace. The rank 1 block is clickable and selects that vessel on the map.

Also added: the spec's metrics card, which renders the checkpoint's IoU against
the Zenodo validation tiles whenever a U-Net actually ran; Enter runs the
pipeline and Space plays the time slider; and a `:focus-visible` outline so a
keyboard operator can see where they are.

**A cache trap worth recording, because it will bite the team again.** After
editing the UI the console broke with

    Run failed: Cannot set properties of null (setting 'innerHTML')

The browser had revalidated `app.js` but served `index.html` from cache. New
script, old DOM, and the script reached for an element that did not exist yet.
Nothing was wrong with either file.

Fixed structurally rather than by telling anyone to hard refresh: `/` now serves
the shell with `Cache-Control: no-store, must-revalidate` and stamps
`config.VERSION` into the asset URLs, so `app.js?v=1.0.0` cannot pair with a
stale shell. Bump `VERSION` and every client picks up both halves together.

**Verified at 1366x768**, the resolution the spec names. Layout holds, sidebar
readable, no horizontal overflow, no console errors.

48 tests pass, `ruff check` clean.

`tidetrace-prepare-data` is still RUNNING on Kaggle: 9.86 GB from Zenodo, then
streaming extract, tile, delete. `tidetrace-train` is chained and waiting.

---

## Entry 009 - 2026-09-03 - The data kernel finished green and delivered nothing

**`tidetrace-prepare-data` reported "complete" with 0 tiles.** It downloaded all
9.86 GB, extracted every chip, and wrote an empty tile set without raising. That
is the worst failure mode there is: the next stage inherits the emptiness and
blames itself. The training kernel then died on

    AssertionError: no oil pixels in the sample: check the data stage

which is its class-balance guard doing exactly its job, refusing to spend GPU
hours on empty data.

**The archive layout, from the kernel's own listing:**

    Images/Lookalike  150      Mask/Lookalike  150
    Images/No oil     150      Mask/No oil     150
    Images/Oil        150      Mask/Oil        150

Masks are a **sibling tree**, not a subfolder. Two bugs followed.

**Bug 1: no mask was ever found.** `_find_mask` searched `folder/masks`,
`folder.parent/masks` and a `_mask` suffix. None of those reach
`Mask/Oil/x.tif` from `Images/Oil/x.tif`. And an image with no mask is simply
labelled all-sea, so every oil chip produced tiles with no oil pixels, which the
`min_oil_pixels` filter then discarded. Silent, start to finish.

Fixed by swapping the image component for each mask synonym: `Images/Oil` ->
`Mask/Oil`, `image/oil` -> `ground_truth/oil`, and so on.

**Bug 2, the dangerous one: `class_for_folder("Images/No oil")` returned OIL.**
The lookup iterated a dict and `"oil" in "images/no_oil"` is true, so it matched
before `"no_oil"` ever got a chance. Clean water would have been labelled as a
spill.

This fails in the direction nobody checks. The model learns that empty sea is
oil, the IoU table still looks plausible because the confusion is inside the
positive class, and the demo confidently paints slicks on open water. It never
reached training only because bug 1 happened to suppress it.

Replaced with an ordered rule list, most specific first: look-alike, then the
no-oil family, then oil. `Mask/` trees are now excluded by checking **every**
path component, not just the leaf, so `Mask/Oil` is no longer read as a folder
of oil images.

**`tests/test_dataset_layout.py`, six tests, built from the real listing above.**
It constructs a miniature `Images/<class>` beside `Mask/<class>` tree and
asserts: "No oil" resolves to sea, mask trees are not mistaken for image trees,
every image pairs with its sibling mask, tiling yields all three classes with
pixel values in {0,1,2}, a No oil chip never produces a positive label, and
tiles are float16. Had this existed, 80 minutes of download would not have been
wasted.

**The kernel now fails loudly instead of finishing green and empty.** It prints
a folder table with the role and class it inferred for every directory before
tiling anything, stops immediately if the first extraction round pairs no masks,
and asserts a non-zero oil count at the end. A red kernel is honest; a green one
that delivered nothing is not.

**Training kernel is now GPU adaptive.** Batch size follows the card rather than
a constant: 4 on T4, 8 on P100, more where there is headroom. Time budget is
7.5 h, under Kaggle's 9 h GPU cut, so the final push happens on our terms rather
than mid-epoch.

**Note on the earlier train ERROR.** It was version 1, pushed before the
credential-free restructure, and it failed on the empty tile set rather than on
anything about the GPU.

54 tests pass. `prepare-data` version 5 is running with the fix.

---

## Entry 010 - 2026-09-04 - Third data bug, and the training loop proven on CPU

**The guard paid for itself on its first run.** `prepare-data` v5 failed in about
two minutes instead of wasting eighty, with the message it was written to print:

    No image was paired with a mask. Tiling would label every chip as sea and
    the oil class would vanish without an error.

and the diagnostic line underneath it:

    00000.tif -> mask None (class 1)
    3 image folders, 3 mask folders

**Bug 3, in the notebook rather than the package.** The streaming loop extracted
only the *image* folder's members for each batch, so `Mask/Oil/00000.tif` never
came out of the archive at all. The pairing logic fixed in Entry 009 was correct;
it simply had nothing to pair against.

Rather than patch the notebook a third time, the batch selection moved into
`app/ml/dataset.py` as `group_archive_members`, `image_folders`, `mask_folders`
and `select_batch_targets`, and is unit tested against the archive's real 900
member listing. A notebook cell cannot be tested; a function can, and this area
has now been wrong three times.

Four more tests cover it: the grouping reproduces the real listing, a batch pulls
`Images/Oil/00000.tif` **and** `Mask/Oil/00000.tif` while never dragging in
another class's masks, the class leaf is respected, and one simulated streaming
round on disk goes all the way from extraction through pairing to non-zero oil
tiles.

**The training loop has now actually run, on this laptop's CPU.**
`tests/test_training_smoke.py` trains for real on a tiny synthetic tile set:
resnet18 encoder, no ImageNet download, two epochs, batch 2. Output from the run:

    epoch 1  loss 2.3571  IoU oil 0.1559  look-alike 0.0064  sea 0.0477
    epoch 2  loss 2.1235  ...
    epoch 9  loss 1.5886  IoU oil 0.1555  look-alike 0.1322  sea 0.4492

    baseline on the same validation tiles:
       threshold baseline  IoU oil 0.0000  look-alike 0.1973  acc 0.8489
       trained model       IoU oil 0.1700  look-alike 0.0570  acc 0.3403

The numbers are meaningless, and they are supposed to be: the tiles are noise
and the encoder is random. What the run proves is that every mechanism the GPU
session depends on has executed at least once. The loss goes through
`combo_loss` without a shape error, validation builds a confusion matrix and
yields three IoUs, the best checkpoint is written under the 80 MB budget, the
checkpoint loads back into the *inference* path and stitches to a valid
distribution, `--resume` continues at epoch 3 rather than restarting, and the
report carries the model beside the dB baseline on identical tiles.

`train()` also gained an `encoder_weights` parameter so a retrain can skip the
ImageNet fetch, which is what an offline machine needs.

**Bug 4, found by the fifth smoke test: `--time-budget 0` did nothing.** The
check read

    if time_budget_s and (time.time() - wall0) > time_budget_s:

and `0.0` is falsy, so a budget of zero disabled the budget entirely. The classic
truthiness trap, in the one place whose whole job is to stop a run before Kaggle
kills it. Now `is not None`, and the comparison is `>=`.

**63 tests pass, ruff clean.** `prepare-data` v6 is running with all three data
fixes and the tested batch selection.

Four bugs in this session, every one silent: masks never found, clean sea
labelled as oil, masks never extracted, and a stop-guard that never stopped.
None would have raised an exception. Three would have produced a confident,
plausible, wrong model.

---

## Entry 011 - 2026-09-04 - Stop inferring the archive, measure it

**The fifth and actual bug: the labels are not named like the images.**

    Images/Oil/00000.tif
    Mask/Oil/00000_segmentation.tif

`select_batch_targets` matched on the bare stem, looked for `00000`, and never
matched `00000_segmentation`. Zero masks were selected for extraction, so zero
masks reached the pairing step, so every chip was read as unlabelled and the oil
class disappeared. Same symptom as Entries 009 and 010, different cause, third
time in a row.

**The method was the problem, not any one fix.** Four diagnoses in a row were
made by reasoning about a 9.86 GB archive's internal shape from the outside and
then paying 35 minutes of download to find out whether the guess was right. Each
fix was correct in isolation and the pipeline stayed broken.

A 7z keeps its header at the end and a pointer to it at the front, so the whole
member list is readable with a few HTTP range requests. `scripts/inspect_archive.py`
implements a seekable file object backed by `Range:` headers and hands it to
py7zr:

    archive size : 9.86 GB
    fetched      : 1.06 MB in 3 range requests
    members      : 908

    folder                        files  example names
    Images/Oil                      150  00000.tif, 00001.tif, 00002.tif
    Mask/Oil                        150  00000_segmentation.tif, ...

One megabyte instead of 9.86 GB, and the answer was unambiguous. That tool now
ships. Check the layout first, download second.

**The fix.** `mask_key()` strips the label suffixes this data actually uses
(`_segmentation`, `_mask`, `_gt`, `_label`, `_seg`, and hyphenated variants) and
both `select_batch_targets` and `_find_mask` match on that key rather than the
raw stem. `_find_mask` also tries suffixed names inside each candidate mask
directory, and finally scans the directory for any file whose key matches, so an
unseen suffix degrades to a slower lookup instead of a silent miss.

Tests are pinned to the **real** names read off the archive header, not to names
I assumed. One of them asserts `mask_key("segmentation_run.tif")` is unchanged,
because a suffix stripper that also eats prefixes is its own kind of bug.

**Also fixed while waiting: an ungeoreferenced chip could be silently skipped.**
Zenodo ships many chips with no CRS, which `load_sar` refuses by design. The
tiler's fallback went through the builtin TIFF reader, which handles only
uncompressed and Deflate, so an LZW chip would have been dropped without a word.
Added `raster.load_raw`, which reuses the same rasterio path and reads anything
rasterio reads, and the tiler now prints why it skipped a file instead of only
counting. The identity transform it returns is documented as a marker, not a
location; nothing that computes coordinates may use it.

**67 tests pass, ruff clean.** prepare-data v8 is running.

Five bugs in this one path, all silent, none raising an exception:

    1  masks never found            sibling tree, not a subfolder
    2  clean sea labelled oil       "no oil" contains "oil"
    3  masks never extracted        batch pulled images only
    4  stop-guard never stopped     time_budget_s=0 is falsy
    5  masks never matched          _segmentation suffix

Three of them would have produced a confident, plausible, wrong model rather
than an error. The lesson worth keeping is not any individual fix: it is that
four of these were diagnosed by inference when they could have been measured.
---

## Entry 012 - 2026-09-04 - Masks pair at last, and the look-alike class turns out not to exist

**The suffix fix worked.** First real signal from the data kernel:

    first round: 60 chips, 60 with masks       (was: 0 with masks)
    tiles written: {0: 150, 1: 0, 2: 400}

400 oil tiles and 150 sea tiles. Two problems in that same output, one fatal and
one that changes what the model can honestly claim.

**Bug 6, fatal and silent: `"std_db": Infinity`.**

    RuntimeWarning: overflow encountered in reduce
    "normalisation": {"mean_db": -19.046875, "std_db": Infinity}

`normalise_db` divides by that. Every training input would have been exactly
zero, the loss would still have moved, and the run would have looked normal
while learning nothing. Reproduced locally in three lines: `np.std` over a
single 512x512 **float16** array is already `inf`, because the running sum
exceeds float16's 65504 ceiling. The float16 tiles were my own optimisation for
Kaggle's disk, so this bug was self inflicted.

Fixed in two places: `_dataset_stats` accumulates in float64 and refuses a
non-finite or near-zero result, and `normalise_db` independently falls back to
sane constants if a bad sigma ever reaches it. Two tests pin both.

**The look-alike class does not exist in this dataset.** Rather than guess again,
I downloaded Part II's look-alike mask archive, which is 0.43 MB:

    00000.tif  shape=(2048, 2048)  dtype=uint8  unique=[0]  positive=0 (0.00%)
    ... every sampled look-alike mask: 0 positive pixels

The Zenodo ground truth segments **mineral oil only**. A look-alike image is
dark water that is not oil, so an empty oil mask is the correct annotation. The
spec sheet's rule "look-alike folder mask 1 -> class 1" cannot be applied
because there is no mask value 1 anywhere to map.

Three options. Threshold the dark patches and call that class 1, which is
fabricating ground truth and is precisely what the anti-slop rules forbid. Keep
an unsupervised class-1 head, which is a dead output presented as a feature. Or
use the chips for what they are genuinely worth.

**Chosen: look-alike chips become hard negatives, labelled sea.** They are the
most valuable negatives in the whole set, because the error they prevent is the
one that matters, calling dark water oil. `build_tile_index` detects an all-zero
mask on a non-sea chip, keeps its tiles as class 0, and counts them under
`hard_negative_chips`. The training report now lists `unsupervised_classes` and
prints an explanation instead of a bare NaN.

**Consequence to state in the pitch, not bury:** the trained model is a binary
oil segmenter with strong hard negatives, and `IoU_lookalike` is undefined for
it. The look-alike concept still exists in the product, because the dB baseline
separates them on contrast and compactness, the UI draws them yellow, and they
are excluded from attribution. Written up in README limitations and in a
dedicated section of `docs/COMPLIANCE.md` with the measurement that proves it.

**Also:** the training notebook now stops in seconds with a pointer to the data
stage when no tiles exist, instead of failing three cells later inside the class
balance check. The operator kept re-running training against an empty tile set
and reading a confusing error; that is a UX bug in the tool, not user error.

71 tests, ruff clean. prepare-data v9 running with every fix.

Running total of silent failures in this one data path: six. None raised an
exception on its own. Four would have produced a confident, plausible, wrong
model. The one structural lesson stands: measure the data, do not infer it.
---

## Entry 013 - 2026-09-04 - Data prep green, training running, and a real basemap

**The data kernel finally produced a tile set.** Every one of the six fixes had
to land before this line appeared:

    first round: 60 chips, 60 with masks
    tiles written: {0: 150, 1: 0, 2: 400}
    normalisation: {"mean_db": -23.80, "std_db": 8.12}
    pixel_share: [0.759 sea, 0.0 look-alike, 0.241 oil]
    550 tiles, 0.36 GB

`std_db` is finite, which is the float64 fix holding. 24 percent oil pixels is a
workable balance for a segmenter.

Worth reading the round log closely, because it shows the hard-negative
decision working as designed:

    round 1  Images/Lookalike   tiles so far {0: 150, 1: 0, 2: 0}
    round 4  Images/Oil         tiles so far {0: 150, 1: 0, 2: 400}

Every sea tile came from a **look-alike** chip, not from `Images/No oil`. The
look-alike chips filled the class-0 budget first, so the negatives in this set
are all dark-water-that-is-not-oil rather than plain calm sea. That is the more
useful negative, since it is exactly the confusion the model must avoid, but it
does mean the set has no plain-sea diversity. Worth raising `MAX_PER_CLASS[0]`
on a longer run so both kinds are represented.

**Training is running on a T4**, chained to that kernel's output so the tiles
never left Kaggle. Batch size is chosen from the card at runtime, 4 for a T4.
Budget 7.5 h against Kaggle's 9 h cut, best checkpoint pushed on every IoU
improvement.

**Expectation management, written down before the numbers arrive:** 400 oil
tiles from Part III alone is a modest corpus. A checkpoint that beats the
-22 dB baseline on oil IoU is the realistic outcome; a state-of-the-art model is
not. The lever for a better one is `MAX_PER_CLASS` plus Part I's 40 GB oil
archive, which this pipeline already handles, at the cost of a longer session.

**The map had no basemap, and it looked broken.** Fair criticism from the
operator: outside the SAR chip the map was a bare graticule, so open water and
coastline were indistinguishable and a forecast cone pointed into nothing.

That emptiness was a consequence of the airplane-mode rule rather than an
oversight, since a live tile layer would go blank the moment the network did.
The fix is the same one used for every other input: cache it.
`scripts/fetch_basemap.py` downloads tiles covering the scene footprints plus a
forecast-cone margin, and the app serves them from `data/basemap`.

    satellite     Esri World Imagery              z5-13    2393 tiles
    ocean         Esri Ocean Basemap              z5-11     212
    ocean_labels  Esri Ocean Reference            z5-11     212
    seamark       OpenSeaMap, ODbL                z10-13     68

14 MB total. Zoom ranges are per layer on purpose: each extra level costs four
times the tiles, satellite is the one you actually read the sea from, and
seamarks carry nothing at ocean-basin zoom. A first attempt at a uniform z13 for
all four layers was heading for roughly 12000 tiles and over an hour; this plan
is 5170 and reuses everything already on disk.

Outside the cached footprint the map still falls back to the graticule, and a
missing tile renders transparent rather than as a broken image. That boundary is
deliberate and informative: it shows exactly where there is data and where there
is not, instead of implying coverage that was never downloaded.

The console gained a layer switcher (Satellite, Nautical, Graticule only, plus
Seamarks and Place names as overlays), attribution as the licences require, and
`bootstrap_demo.py` now caches the basemap as step 4 of 6 and reports tile
counts in the readiness table. Version bumped to 1.1.0 so the cache-busting
picks up the new script.

71 tests, ruff clean.


---

## Entry 014, 2026-09-04, The first checkpoint was a false positive machine

Training finished. The checkpoint was excellent by every number the run
reported and useless on real data. Writing down how it passed, because the
metric that let it through was the actual defect.

### What the run reported

    metric             -22 dB base      U-Net     delta
    IoU oil                 0.0000     0.8575   +0.8575
    IoU sea                 0.7983     0.9609   +0.1626
    pixel accuracy          0.7983     0.9683   +0.1700

Installed it, health came back `detector: unet`, `model_loaded: True`, no
warnings. Then ran the three demo scenes:

    caspian_baku_seeps      unet  polys=1  oil=483.321 km2
    gom_mc20_chronic_slick  unet  polys=1  oil=498.382 km2
    arabian_sea_mumbai      unet  polys=1  oil=503.012 km2

The chip is about 484 km². It was calling the entire image oil, on all three,
including the clean-water control that must return nothing.

### Not a threshold problem

The first instinct was calibration. It was not that. Softmax on the clean-water
control:

    sea    mean 0.0538  median 0.0361  p95 0.1480
    oil    mean 0.9353  median 0.9533  p95 0.9849
    fraction over 0.5: 0.9985     over 0.9: 0.8208

The 5th percentile of oil probability is 0.838. There is no threshold that
separates anything, because there is nothing to separate: the distribution is
saturated. Per-scene normalisation was tested too and made it marginally worse
(0.9991 vs 0.9844), so the dB statistics gap between Zenodo and Sentinel-1 RTC
is not the cause either.

### Cause, from the report's own history

`iou_lookalike` is `0.0` for epochs 1-3 and `nan` from epoch 4 on, and the run
wrote 550 tiles against a budget of `{oil: 400, look-alike: 300, sea: 150}`.
400 + 150 = 550 exactly. So:

  - 400 oil tiles
  - 150 sea tiles
  - **0 look-alike tiles**

Every look-alike chip hit the empty-mask hard-negative rule from entry 011 and
was relabelled sea, and the look-alike folders are walked before
`Images/No oil`, so they consumed all 150 of the shared sea budget. The model
never saw one tile of ordinary open water. Training was 73% oil tiles, against
roughly 0.1% oil pixels in a real scene.

On top of that, `CLASS_WEIGHTS = [0.4, 1.0, 2.5]`: oil boosted 2.5x, sea cut to
0.4x, a 6.25x relative push toward oil. Those weights are correct when oil is
rare in training. It was not rare, because tiles are kept only when they hold
at least 50 oil pixels. Both errors point the same way and they multiplied.

Worth noting that `miou` "improved" from 0.5947 to 0.8643 at epoch 4. That was
`nan` dropping class 1 out of `np.nanmean`. A class vanished and the headline
number went up.

### The real defect

IoU_oil is monotone in "predict more oil" when it is measured only on tiles
selected for containing oil. Ranking epochs by it selects approximately the
most over-predicting epoch available. The number was not lying; it was
answering a question nobody should have been asking.

So the fix is a metric, not a hyperparameter:

    water_false_oil = (cm[0,2] + cm[1,2]) / (true sea + true look-alike)

The fraction of genuinely-water pixels called oil. On the broken checkpoint it
is about 1.0. `MAX_WATER_FALSE_OIL = 0.02` is now a hard gate: an epoch over it
cannot become the shipped checkpoint however good its IoU, and if no epoch
clears it the run says so and writes nothing rather than shipping the least-bad
option. `tests/test_training_smoke.py` covers both the gate firing and the
metric itself; the smoke tests that only exercise plumbing pass
`max_water_false_oil=1.0` explicitly, by name, so the default is never
quietly weakened.

### Changes

`app/ml/train.py`
  - `CLASS_WEIGHTS` `[0.4, 1.0, 2.5]` -> `[1.0, 1.0, 1.5]`. Dice already
    handles per-class imbalance; sea must not be suppressed on top of it.
  - `water_false_oil` and `sea_false_oil` in `metrics_from_confusion`, printed
    every epoch, recorded in the checkpoint metadata and the report.
  - Best-checkpoint selection gated on it. No eligible epoch is reported
    loudly, with the closest miss, and no file is written.

`app/ml/dataset.py`
  - `max_hard_negatives`: look-alike hard negatives get their own ceiling and
    can no longer starve `Images/No oil` of the shared sea budget.
  - `max_scene_negatives`: tiles from an oil chip that contain no oil are kept
    as sea instead of discarded. These are the best negatives available, same
    sensor, same pass, same sea state, same incidence angle, differing from the
    positives only in the thing being learned. Discarding them is most of what
    skewed the prior.
  - The index now reports `hard_negative_tiles`, `scene_negative_tiles` and
    `plain_sea_tiles` separately, so a repeat is visible in the output.

`scripts/build_kaggle_kernels.py`
  - `MAX_PER_CLASS` sea 150 -> 1200, plus `MAX_HARD_NEG = 300` and
    `MAX_SCENE_NEG = 500`.
  - Prints the sea tile breakdown and the oil share, and asserts
    `counts[0] > counts[2]` before training can start.
  - A round that yields no tiles ends that folder. Look-alike folders never
    raise `counts[1]`, so the outer budget check could never retire them and
    the run would have extracted thousands of chips for nothing.

The broken checkpoint is in `models/quarantine/` with a README, kept for
comparison. `models/` is empty, so the product is back on the dB baseline,
which works. Better a documented baseline than a detector that cannot find
water.

76 tests, ruff clean. prepare-data v11 pushed and running.

### Addendum, same day, two more bugs, caught before the rerun finished

While v11 was downloading, checked what `Images/No oil` would actually get.
The archive listing in `tests/test_dataset_layout.py` says the walk order is
`Images/Lookalike`, `Images/No oil`, `Images/Oil`. Two problems fell out.

**Ordering.** With one shared class-0 budget of 1200, `Images/No oil` reaches it
second and takes whatever the hard negatives left, so `Images/Oil` arrives with
class 0 already full and contributes no scene negatives at all. Raising the
budget had moved the starvation one folder along rather than fixing it. Class 0
now has three independent ceilings, `max_plain_sea` 700, `max_hard_negatives`
300, `max_scene_negatives` 500, so the walk order stops mattering.

**The `break`.** Worse, and a bug I introduced with the scene negatives. Tiles
are walked row-major. An oil chip whose slick sits low in the frame yields
empty tiles first, which become scene negatives, and a full sea budget hit
`break`, leaving the tile grid entirely and abandoning every oil tile below the
slick. The oil class would have shrunk toward nothing with no error printed
anywhere. A negative at its ceiling now `continue`s; only the chip's own class
breaks. `test_a_full_sea_budget_does_not_abandon_the_oil_tiles_below_it` pins it.

Both are the same shape as the original defect: a budget interacting with an
ordering, failing silently, and visible only in a number nobody was printing.
The notebook now asserts `plain_sea > 0` and `counts[0] > counts[2]` before
training can start, and prints the class-0 breakdown by source.

Cancelled v11, pushed v13. 78 tests, ruff clean.

### Addendum 2, the streaming loop is quadratic, and that is now visible

v13 ran 2h33m and was still going. Not stuck; the cost is structural and the
larger tile budget exposed it.

The extraction loop re-opens the 9.86 GB archive once per batch of 60 chips.
That archive is **solid**, so reaching any member means decompressing from the
start of its block: every round pays close to a full scan. Cost is therefore
roughly (rounds x archive size), and rounds scale with the tile budget.

    v7 - v11   550 tiles    ~3-4 rounds    ~35-60 min
    v13       ~1900 tiles   ~8 rounds      2.5 h and counting

The design was chosen for a real constraint, Kaggle gives 20 GB of working
disk and the full chip set does not comfortably sit there beside the tiles, so
the loop trades time for space. That was the right trade at 550 tiles and is a
poor one at 1900. Worth revisiting: extract every needed member in a single
pass, keyed off a chip list computed up front, and tile each chip as it lands
rather than staging a batch at a time. Not being changed mid-run, because
restarting discards 2.5 h for a cost that is bounded at about eight rounds and
Kaggle's CPU limit is 12 h.

Recording it here so the next person does not read the runtime as a hang.


---

## Entry 015, 2026-09-04, Audit, two GPU-day lessons, and the optical panel

Asked to review the build adversarially against PS 26143 clause by clause.
Result: twelve of fifteen hold on real data, two are partial, one is absent.
Published as `docs/ps-26143-audit.html`.

The honest reading of the three that do not hold:

  a.1 / d.1  Detection runs, but on the dB baseline, because there is still no
             checkpoint fit to ship. The PS asks for a machine learning model.
  d.2        SAR only. "such as SAR and EO imagery" makes SAR defensible, but
             this was the one clause with nothing at all behind it.

One thing the audit corrected in my own framing: I had been calling simulated
AIS a gap. It is not. The PS says "Real AIS if available may be used else
synthetic data can be prepared". d.3 passes. A live feed makes the demo
stronger, not compliant.

### The P100 does not work, and cost a run to learn

Pushed training with `--accelerator NvidiaTeslaP100` for the 16 GB. It staged
the data, built the model, and died in the first forward pass:

    AcceleratorError: CUDA error: no kernel image is available for execution
    Tesla P100 with CUDA capability sm_60 is not compatible with the current
    PyTorch installation. Supports sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.

Kaggle ships torch 2.10.0+cu128, which no longer compiles Pascal. The P100 is
sm_60. This is why the earlier manual T4 run worked and mine did not. T4 is now
the default in `kaggle_push.py`, and the notebook checks
`torch.cuda.get_arch_list()` against the device capability in its first cell, so
the next mismatch costs two seconds rather than eleven minutes.

### A third silent data bug, found in the failed run's log

The index reported `counts {0: 1412, 1: 0, 2: 400}`, 1812 tiles, while 1164
files existed. 648 tiles had been written and then overwritten.

Tiles were named `<stem>_<class>_r_c.npz`, and Part III ships `00000.tif` in
`Images/Oil`, `Images/No oil` AND `Images/Lookalike`. A sea tile cut from a
No oil chip landed on exactly the name of the sea tile cut from a Lookalike
chip. The negatives that this whole rebalancing exists to add were the ones
being destroyed.

Two smaller things fell out of the same log. The rebuild cell counted classes
with `per_class[k] = per_class.get(k, 0)`, an assignment, never an increment,
always zero, which is why nobody noticed the 1812 was fiction. And the pixel
share was sampled from `files[:400]`, which under the new source-prefixed names
would have been a single folder and therefore a single class.

Fixed: the source folder is now part of the tile name, the index counts what is
on disk and reports `counts_attempted` separately when they differ, and the
pixel sample steps evenly through the set. Three tests pin it.

Not re-running prepare-data for this yet. The existing 1164 tiles are already
staged on Kaggle, training on T4 is about an hour, and `water_false_oil` will
say honestly whether the mix is good enough. An hour to find out beats six
hours to guess. Pixel share is still 18.9% oil against 19.8% before, so the
gate may well refuse it, that is the gate working, not a surprise.

### Optical panel, closing d.2

`scripts/fetch_sentinel2_chip.py` pulls a true-colour Sentinel-2 L2A chip over
each scene from the same Planetary Computer STAC already used for Sentinel-1,
cloud-masked via SCL, stretched on percentiles over water only. All three
scenes have one:

    gom_mc20_chronic_slick        4.3 days before the pass, 23.9% obscured
    arabian_sea_mumbai_offshore   19 h before the pass,      0.0% obscured
    caspian_baku_seeps            16.8 days before the pass, 0.0% obscured

It is labelled for what it is. Sentinel-2 never crossed this water at the radar
acquisition time, so every panel carries its offset and its cloud fraction, the
layer is off by default, and no detection is run on it. Context for reading the
map, not evidence.

Two UI bugs found by actually clicking it rather than trusting the code:

  1. `clearAll()` wiped every layer group except the graticule on each run, and
     optical only reloads on scene change, so the first RUN erased it forever.
     Optical is scene state, not run output.
  2. Leaflet stacks `overlayPane` by DOM insertion order. The SAR backdrop is
     inserted at run time, after the optical chip, so optical was always
     underneath it and ticking the box did nothing visible. Fixed with explicit
     panes: sarPane 350, opticalPane 360, both below overlayPane 400 so the
     class mask and vectors still draw on top.

80 tests, ruff clean. Version 1.2.2. Training running on T4.

### Correction to entry 015, the pixel share is 10.6%, not 18.9%

Entry 015 says the oil pixel share "is still 18.9% against 19.8% before". That
is wrong, and wrong for the reason the entry itself identifies: 18.9% was read
out of the kernel log, where it had been computed over `files[:400]`. That is
the biased sample the same entry describes fixing.

Measured over all 1164 tiles once they were on disk:

    sea 89.398%   look-alike 0.000%   oil 10.602%
    756 of 1164 tiles (64.9%) contain no oil at all
    among oil-bearing tiles, median oil fraction 0.172

So the rebalancing did roughly halve the oil share, 19.8% to 10.6%, and it
created a real negative population where there was none. The tile counts on
disk are 764 sea to 400 oil, against 150 to 400 before, the ratio moved by 5x.

Deployment is still around 0.1% oil, so 10.6% remains two orders of magnitude
high. What it would take, holding the 400 oil tiles fixed:

    to 10%   +70 sea tiles     (essentially there)
    to  5%   +1304 sea tiles
    to  2%   +5007 sea tiles

`Images/No oil` has 150 chips at 25 tiles each, 3750 available, and scene
negatives are capped at 500 of a much larger supply. So 5% is reachable in one
more prepare-data run and 2% is roughly the ceiling of this archive. Whether
any of that is needed is what `water_false_oil` is for. Noting the numbers now
so the next run has a target rather than a guess.

Lesson worth keeping: I reported 18.9% to the user while the log line producing
it was in the same diff I was fixing for being unrepresentative. Read the number
from the artefact, not from the log that computed it.


---

## Entry 016, 2026-09-04, Live AIS recorder, and the provider question settled

Asked to pick the AIS provider myself and make it work "like in prod".

Read the docs rather than guessing. aisstream.io is free, WebSocket, and
**live only, no history**. Datalastic has a `/vessel_history` endpoint but is a
paid subscription. A free-registration key was far likelier than a paid one, and
a single connection attempt settled it: aisstream accepted the subscription, so
that is the provider. Deliberately did not try the key against other vendors
first -- sending a credential to the wrong service leaks it there.

### Why a live feed still matters when it cannot touch the demo scenes

It cannot. The scenes are 2023 and 2024 acquisitions and no stream produces
positions from then. Saying that plainly is more useful than pretending
otherwise, so `scripts/record_ais.py` is a **recorder**, not a fetcher, and its
docstring leads with the distinction. What it buys is the thing an operational
system actually has: it has been recording, so a slick found tomorrow already
has real traffic around its origin window. The PS clause d.3 passed without it.

Rows go through `row_from_csv` in the same 16-column MarineCadastre shape with
`source='aisstream_live'`, so the scorer cannot tell them from MarineCadastre
and provenance stays auditable. Flushes every 400 rows and on every disconnect,
so a killed session keeps what it heard. Reconnects with backoff. The key is
read from `$AISSTREAM_API_KEY` and written nowhere.

Verified against the Singapore Strait, which is busy enough to prove it in a
minute: 47 positions from 43 vessels, real names, SOG, COG, heading, status.
Cleared afterwards with `clear_source` -- Singapore is not in any scene
footprint and leaving it would have inflated the store's vessel count for no
reason.

### Two bugs the live feed found

`row_from_csv` parses MarineCadastre CSV, so its contract is strings. aisstream
sends vessel type, navigational status and IMO as **integers**, and the first
vessel to report static data died on `.strip()`. Coerced at the boundary rather
than loosening a parser that is right to expect what a CSV gives it.

Then the test caught the second one: `_imo(0)` returned the string `"0"`. Zero
is the ITU "not available" sentinel, so that would have recorded every vessel
without an IMO as being registered under number zero. Returns None now.

Honest limitation worth stating at demo time: `ShipStaticData` arrives roughly
every six minutes, so a short recording captures positions but not vessel type.
Type is 20% of the suspicion score. A recording needs to run for a while before
its vessels score properly, and the funnel will show them as unknown type until
it does.

86 tests, ruff clean.


---

## Entry 017, 2026-09-04, The bands were swapped. The model works.

The checkpoint is real and the demo scenes pass, including the control. Writing
down the whole diagnosis, because the rebalancing in entries 014 to 016 fixed a
genuine problem that was never the binding one, and the actual bug was one line.

### Training result

    metric             baseline      U-Net
    IoU oil              0.0000     0.8891
    IoU sea              0.8645     0.9822
    pixel accuracy       0.8645     0.9844
    water -> oil         0.0038     0.0106

35 of 40 epochs cleared the `water_false_oil <= 0.02` gate, so this is stable
rather than one lucky epoch. Best was epoch 33.

Then the demo scenes returned 174, 478 and 503 km² of oil on ~484 km² chips.
Again. Including the clean-water control. Again.

### The diagnosis, and the two wrong turns in it

Checked normalisation first: training mean -20.85 dB against scene means of
-25.9, -22.7, -23.1, and per-tile std 4.27 against scene std 3.3 to 5.2. Close
enough that it explained nothing. Checked speckle texture next: lag-1
correlation 0.73-0.77 in training against 0.65-0.84 in the scenes, and the
smoothest scene was the one predicting correctly. Also explained nothing.

What did explain it was printing the per-channel means:

    training tiles   ch0 -29 dB   ch1 -22 dB      dark first
    demo scenes      ch0 -21 dB   ch1 -27 dB      bright first

Zenodo ships (VH, VV). `fetch_sentinel1_scene.py` reads Planetary Computer
assets in the order `("vv", "vh")`. The network trained on (dark, bright) and
was asked to predict on (bright, dark). Swapping the two channels on the centre
tile, with no retraining at all:

    scene                          as-is    swapped
    gom_mc20_chronic_slick        0.9929     0.0251
    arabian_sea_mumbai_offshore   0.9999     0.0000
    caspian_baku_seeps            0.2303     0.0000

### The fix, and why not a swap

Hardcoding a swap would move the assumption rather than remove it: the next
archive with its own convention breaks it again, silently, the same way. Band
order is now decided by a physical fact instead. Over water, co-polarised
backscatter exceeds cross-polarised by several dB in every sensor and every sea
state, so `order_bands` sorts on the median and both conventions collapse to
one. It reproduces the Zenodo order exactly for this checkpoint and keeps
working for any future source. Applied at tiling and at inference, from one
implementation in `dataset.py`.

`tests/test_unet_path.py` had an assertion reading `y[0] == -10.0` -- it
encoded bright-first as the contract, which is the bug written down as a test.
Corrected, with the reason in the comment.

### End to end, U-Net loaded, no warnings

    scene                        polys   oil km2    baseline km2
    caspian_baku_seeps               3     0.315          16.273
    gom_mc20_chronic_slick          29     5.149           0.409
    arabian_sea_mumbai_offshore      0     0.000           0.000

The control returns nothing and no suspects, which is the result that matters.
The Caspian drop from 16.3 to 0.315 km² is the segmenter declining to call a
whole seep field mineral oil where the threshold baseline called all of it oil;
worth showing both at demo time rather than only the flattering one.

Look-alikes remain 0 under the U-Net, as documented since entry 011: the
archive's class-1 masks are empty, so the head is unsupervised. The dB baseline
still separates them and the UI still draws them.

Checkpoint pushed to N-1ACE/tidetrace-oil-unet. 89 tests, ruff clean.

### What this cost, and the lesson

Three runs and roughly nine GPU-and-CPU hours were spent rebalancing a training
set to fix a symptom produced by a channel order. The rebalancing was still
worth doing -- 400 oil against 150 sea tiles with a 6.25x oil weight was
indefensible on its own terms, and `water_false_oil` is now a permanent guard.
But the evidence for the real cause was available from the first failed run, in
a two-line print of per-channel means, and I reached for the explanation I had
already been thinking about instead of the cheapest measurement that could
separate the hypotheses. Print the inputs before theorising about the model.


---

## Entry 018, 2026-09-05, Copernicus Marine currents, and the Caspian answer is no

CMEMS credentials supplied. Wired in, and the first thing worth recording is
that it does **not** do what I said it would.

### The Caspian is not a coverage gap that an account fixes

I had told the user twice that CMEMS would close clause b.1 and remove the
Caspian wind-only caveat. Checked it before building anything, which was the
right order:

    scene                          uo m/s     vo m/s   coverage
    gom_mc20_chronic_slick        -0.0834     0.0045   93.4% of 289 cells
    arabian_sea_mumbai_offshore    0.0024    -0.0721  100.0% of 289 cells
    caspian_baku_seeps            ALL NaN    ALL NaN     0.0% of 306 cells

The Caspian is endorheic, so the global ocean model carries it as land. Then
searched the catalogue: `cm.describe(contains=["Caspian"])` returns zero
products. There is no Copernicus Caspian product to point at. The gap is a fact
about what is modelled, not about what is configured, and no account closes it.

So b.1 stays partial, and the run keeps saying "wind driven only" over Baku.
Substituting a plausible-looking field there would be the same class of mistake
as inventing look-alike labels.

### What it did buy

Real eddy-resolving currents at 1/12 degree for the two scenes it covers, which
is the product an operational drift model would actually use:

    gom_mc20_chronic_slick        0.176 m/s mean
    arabian_sea_mumbai_offshore   0.065 m/s mean
    caspian_baku_seeps            none, wind only, warned

`--currents auto` tries Copernicus first and falls back to Open-Meteo, so the
build still works for anyone without an account and states which source it used
in the cube's `source` string.

Two choices worth writing down. `cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i`
over the daily mean, because six-hourly interpolated to the cube's hourly steps
is materially better for a 48 hour backtrack than a daily average. And the
`-cur` physics product over `merged-uv`, because the merged one folds in Stokes
drift while the drift model already adds a wind term at 3% of U10, taking both
would count the wave-driven part twice. The first hourly dataset id I tried did
not exist; listing the product's datasets gave the real ones rather than
guessing again.

Credentials are read from `COPERNICUSMARINE_SERVICE_USERNAME` and `_PASSWORD`
in the environment and are written nowhere, same as the Hugging Face, Kaggle
and aisstream keys.

### Ledger now

13 of 15 clauses met, 2 partial, 0 absent. The two partials are both limits of
what data exists: no Caspian current model anywhere, and optical present as
context rather than as a second detector.

89 tests, ruff clean.

### Addendum to 018, installing CMEMS broke the offline boundary

`test_importing_the_app_does_not_load_a_network_client` went red immediately
after the install. Importing `app.main` was pulling in boto3 and 40-odd urllib3
modules.

Traced it rather than assuming: `rasterio/session.py` imports boto3 on sight for
its optional AWS support. copernicusmarine depends on boto3, so installing it
put a network client into the process of anything that opens a GeoTIFF, which
is every process this product runs.

The tempting fix was to widen the exemption list next to the existing
`loaded -= {"httpx"}` and move on. That would have been hollow: the test exists
precisely to catch a network client arriving by a route nobody intended, and
this is that, exactly.

What was done instead. copernicusmarine is declared, commented out, under a new
"OFFLINE PREP, NOT ON THE DEMO LAPTOP" block, matching the convention already
used for torch and huggingface_hub. The sys.modules exemption is narrowed to the
four packages that can only arrive through rasterio's probe, with the reasoning
written next to it. And a new test,
`test_no_prep_only_package_is_a_runtime_requirement`, asserts that none of the
prep-only or training packages appears as an *active* line in requirements.txt.

That last one is what keeps the exemption honest, so it was checked by
uncommenting `copernicusmarine>=2.4` and confirming the test fails, then
restoring it and confirming it passes. An assertion nobody has watched fail is
not evidence.

90 tests, ruff clean.

### Addendum 2 to 018, the runtime claim was stale

The audit said "under 8 seconds a scene". That was measured on the dB baseline.
With the U-Net actually running, on a CPU laptop:

    caspian_baku_seeps            22.8 s total, DETECT 16.6 s
    gom_mc20_chronic_slick        24.4 s total, DETECT 16.4 s
    arabian_sea_mumbai_offshore   17.2 s total, DETECT 16.9 s

Inference over 25 tiles of 512 without a GPU is about 16.5 s of it, near
constant across scenes, which is what you would expect from a fixed tile count.
The baseline does the same scene in roughly a second.

Corrected in the ledger rather than left to be discovered by whoever timed it at
a demo. Both detectors remain available and the product still falls back to the
baseline automatically when no checkpoint is present, so the trade is the
operator's to make.

### Addendum 3 to 018, the corrected runtime was itself too flattering

Recorded 17-24 s in addendum 2. Re-measured on a fresh server, twice:

    cold    caspian 50.7 s   gom 48.1 s   arabian 35.0 s
    warm    caspian 28.6 s   gom 34.7 s   arabian 25.3 s

DETECT alone ranges 16 to 28 s across runs. Nothing else on the machine was
using CPU -- checked, all the stray processes were idle at 0 s of CPU over a
5 second window -- so this is just the honest variance of a 25-tile inference on
a shared laptop, not contention.

The first number came from a favourable moment and I wrote it down as if it were
the figure. The ledger now says 20 to 35 s with the 16 to 28 s inference range
stated, which covers everything actually observed. Quoting the best run as the
runtime is the same species of error as quoting IoU without the control scene.

Also worth noting for anyone tidying up: two uvicorn processes on port 8077 are
running from this project and were NOT started by me in this session. They were
left alone rather than killed, since a port I never used is not mine to reclaim.


---

## Entry 019, 2026-09-05, Closing the last two clauses properly

Asked whether every PS condition is strictly met, and to go and fix what is not.
Two were marked partial: EO used only as a display layer, and the Caspian having
no currents. Both are now addressed, one by finding the answer and one by
accepting it and building around it.

### The Caspian answer is still no, and now it is checked twice

Tested HYCOM the same way CMEMS was tested, because a second independent global
model is worth more than an opinion:

    HYCOM GLBy0.08, water_u at the Baku point   -30000.0   (its fill value)
    HYCOM GLBy0.08, water_u in the Gulf            -0.48 m/s

So HYCOM masks the Caspian as land too. Copernicus publishes no Caspian product.
The only Caspian-specific current dataset findable is a 2003-2005 research
reconstruction, wrong epoch for a 2023 scene and not operational. Two global
models and a catalogue search agree: the data does not exist.

That is a fact about the world, not a gap in the build, and no amount of further
searching changes it. What it *did* mean is that the demo set never showed the
full oceanographic capability on real data.

### So a fourth scene, with nothing simulated in it

`santa_barbara_seeps`, Santa Barbara Channel natural seeps, 2023-08-29T01:59.
Chosen because it is inside MarineCadastre NAIS coverage *and* inside the
Copernicus ocean model, so every input is real:

    SAR        Sentinel-1 RTC, 7.10 dB contrast, 93844 px dark blob, 100% valid
    currents   CMEMS 6-hourly, 0.073 m/s mean
    wind       ERA5, 4.58 m/s mean
    AIS        real MarineCadastre, 6 candidate vessels, zero simulated
    optical    Sentinel-2, 17 h after the radar pass, 0.0% obscured

That last line is the best optical pairing in the set by a wide margin. The AIS
store grew from 295,051 rows to 344,481 and now starts 2023-08-28.

### EO now does analysis, not decoration

New `app/eo/corroborate.py` and an `EO` stage in the pipeline. For each SAR
polygon it compares the optical inside against a ring around it. A film reads
darker than the surrounding water; a rig, sandbar or ship reads brighter; a
low-wind cell reads like nothing. Verdicts are `consistent`, `inconsistent`,
`neutral`, `obscured`, carried on the polygon properties, in the API, on the
verdict card and in the map popups.

It is corroboration and never a detector. Nothing is added, removed or
reweighted by it, and the caveat travels with every verdict: Sentinel-2 did not
observe this water when the radar did.

    santa_barbara_seeps    1 consistent, 1 neutral      17 h offset
    caspian_baku_seeps     3 consistent                 16.8 days offset
    gom_mc20               28 obscured, 1 neutral        4.3 days offset

The Caspian scoring 3 of 3 consistent is the kind of result worth pausing on
rather than celebrating: at a 16.8 day offset a discharge would be long gone, so
what it actually corroborates is that those are *persistent* features. Which is
correct -- they are natural seeps.

### The empty-image trap

The first run of this returned "neutral, no optical difference" for all 29 Gulf
polygons, with a delta of exactly 0.0 on every one. Exactly 0.0, twenty-nine
times, should never be believed, so I went and printed the pixels: the Gulf
optical chip is 0 across the entire area the detections sit in. A Sentinel-2
granule covers only part of a scene footprint and `cut_chip` fills the rest with
zero. Only saturated white counted as obscured, so nodata was being read as dark
water and reported as a confident finding.

Nodata now counts as obscured, the Gulf reports 28 obscured, and eight tests in
`tests/test_eo_corroboration.py` pin every branch including this one.

Also fixed while in there: `fetch_basemap.py --scene X` rewrote the manifest
naming only X, silently erasing the record of every scene cached before it. The
tiles stayed on disk and the manifest stopped mentioning them. It merges now.

98 tests, ruff clean. Version 1.3.0.

### Addendum to 019, and the same manifest bug bit once more

Fixed `fetch_basemap.py` to merge its manifest while the Santa Barbara tile
fetch was already running. The running process had the old code loaded, so it
finished and overwrote the manifest with only `santa_barbara_seeps` anyway --
exactly the bug I had just fixed, committed after the process that would hit it
had started. Re-ran `--all`, which rebuilt a correct four-scene manifest with
`saved 0 cached 3177`: every tile a cache hit, nothing re-downloaded, and the
merge verified in passing.

Final readiness:

    scene                          metocean  ais mode   acquired
    gom_mc20_chronic_slick         yes       real       2023-09-24
    arabian_sea_mumbai_offshore    yes       simulated  2024-03-13
    caspian_baku_seeps             yes       simulated  2023-10-14
    santa_barbara_seeps            yes       real       2023-08-29

    AIS store   441668 rows, 638 vessels, 2023-08-28 .. 2024-03-13
    basemap     6976 tiles, 19.2 MB, four layers, zoom 5-13
    detector    U-Net checkpoint present

Santa Barbara through the console: OIL DETECTED 0.37 km2, slick 1.1 km long,
origin 2023-08-28 14:59:18 +/- 8.3 km, 11.0 h drift proxy, offshore, detector
U-Net, **optical 1 agree 1 neutral (17 h after the radar pass)**, rank 1
FISH TWO at 67.0% from real MarineCadastre tracks. 11.6 s.

All fifteen PS clauses met. 98 tests, ruff clean, version 1.3.0.


---

## Entry 020, 2026-09-05, What the screenshots were actually showing

Sent two screenshots: oil polygons painted across the Santa Barbara mountains,
43 of them totalling 397 km2 on a 484 km2 chip, and a scene switch that left the
previous run's vessel marker floating over an empty graticule.

### The first one was a stale server, and that is my fault

The screenshot showed `IoU oil 0.8575` in the checkpoint panel. That is the
*quarantined* checkpoint's number. Checking what was listening:

    port 8077   version 1.1.0   started 04-09 15:53
    my server   version 1.3.0

1.1.0 predates the band-order fix entirely, and that process had loaded the
first broken checkpoint into memory before it was quarantined and never let go.
So the user was driving a build that was known-broken, on the port they happened
to be using, while I reported results from a different one.

I had seen those processes hours earlier, decided they might be the user's, and
left them running. "Might be theirs" was the wrong call for a server that
answers on the project's own name with a detector that cannot find water. All
TideTrace servers are now stopped and there is one current build.

Worth noting for future sessions: port 8000 is taken on this machine by an
unrelated project of the user's, `dell-ai-parts-inspector`. TideTrace was moved
to 8100 rather than reclaiming it.

### The second one was real, and there were two bugs behind it

**Scene switching kept the previous run.** The handler called `renderSceneInfo`
and `fitBounds` and nothing else: no layer clearing, no panel reset, no timeline
reset. So the map showed one sea while the panels described another, which is
worse than showing nothing. `resetRun()` now returns every panel to its
documented empty text, clears every layer except the graticule, stops playback
and resets the scrubber.

**Tiles vanished below zoom 5.** The layers set `maxNativeZoom` so zooming in
upscales, but no `minNativeZoom`, so zooming out past the cached range requested
tiles that were never downloaded and the map went blank. Both bounds now reuse
the nearest cached level. Panning away from every cached footprint says so in
words rather than leaving a bare grid to be read as a failure.

### And a third bug the screenshots led me to

The complaint was about mask accuracy, so I went looking at whether detection
masks land at all. It did not. Land was only ever used for the forecast cone's
coast check, so dark radar shadow behind a ridge was free to be reported as
mineral oil -- exactly what the screenshot showed, and a live risk on any chip
containing coastline.

The coastline file that existed was worse than no file. Tested against known
points it called the Santa Barbara mountains sea and open Caspian water land,
so it both missed real land and invented it.

Rebuilt from Natural Earth 1:10m, public domain, clipped to the scene footprints
with Sutherland-Hodgman. First attempt read 144 of 144 sample points across the
Caspian footprint as land, which would have deleted that scene entirely. Cause:
`_rings_of` returned only `coords[0]`, the exterior ring, and Natural Earth
models inland seas as **holes**. Dropping holes makes the Caspian part of
Eurasia. Holes are now emitted as water and win the point test.

    known points   8/8 correct, up from 6/8
    caspian          3/144 land, down from 144/144
    santa barbara   42/144 land, coast and ridge
    gom, arabian     0/144, both offshore

Detection now drops polygons whose centroid is ashore and reports the count. The
Caspian had two of them, on the Absheron peninsula: 3 polygons and 0.315 km2
becomes 1 polygon and 0.074 km2. The centroid test keeps a polygon straddling
the shoreline, which is the right way round to be wrong -- a slick coming ashore
is the case that must not be missed.

103 tests, ruff clean. Version 1.4.0.

### Addendum to 020, the reset I added broke the run

Reported: `Run failed: Cannot set properties of null (setting 'innerHTML')`,
reproducible on every run. Introduced by the `resetRun()` written earlier the
same day.

    <div class="card" id="verdict-card" hidden>
      <h2>Result <span id="verdict-time"></span></h2>
      <div class="body" id="verdict"></div>
    </div>

`resetRun()` cleared `verdict-card.innerHTML`, which deletes both children. The
renderer then looked up `verdict` by id, got null, and threw before drawing
anything. Two ids the script needs were simply gone from the document, which a
quick DOM query confirmed rather than reasoning about it:

    missing: ["verdict-time", "verdict"]

Fixed by clearing the card's *contents* -- `verdict.innerHTML = ""` and
re-setting `card.hidden = true`, which is also the card's authored initial
state -- instead of the card itself.

`tests/test_console_dom.py` now parses `index.html` for id nesting, extracts
every `$("id")` from `app.js`, and asserts that nothing the script empties
encloses an id the script writes into. Verified it bites by reintroducing the
bug: two tests fail, and pass again on revert. A third asserts every looked-up
id exists in the markup at all.

Browser verification of the fix, end to end:

    arabian sea    NO OIL DETECTED, clean-scene note, no errors
    santa barbara  OIL DETECTED 0.37 km2, coast REACHES LAND,
                   optical 1 agree 1 neutral, #1 FISH TWO 67.0%
    scene switch   map refit, every panel reset, verdict card hidden

    santa_barbara   2 polys  0.373 km2  ashore 0  coast True   6 suspects
    gom_mc20       29 polys  5.149 km2  ashore 0  coast False 10 suspects
    caspian         1 poly   0.074 km2  ashore 2  coast True   4 suspects
    arabian_sea     0 polys  0.000 km2  ashore 0  control, silent

Console clean apart from 404s on basemap tiles outside the cached footprints,
which are expected and swapped for a transparent tile.

106 tests, ruff clean. Version 1.4.1.
