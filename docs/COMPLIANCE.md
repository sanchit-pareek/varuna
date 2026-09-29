# SIH26143 compliance matrix

Every mandatory clause of the problem statement, mapped to the code that
implements it and the evidence you can check without trusting anyone.

Read this next to `README.md` (limitations) and `PROJECT_JOURNAL.md` (history).

---

## Clause (a) Detection and characterisation

| Requirement | Implementation | How to verify |
| --- | --- | --- |
| Detect oil slicks from satellite imagery (SAR) | `app/ml/infer.py` runs a 3-class U-Net over 512 tiles with cosine-tapered stitching. `app/ml/fallback.py` is the published dB baseline used when no checkpoint is present. | `POST /api/detect` returns `metrics.detector`, which is `unet` or `sigma0_threshold_baseline`. The UI prints whichever ran. |
| Reject look-alikes | Class 1 of the segmentation. The **dB baseline** separates oil from look-alike on local contrast and boundary regularity and is what produces class 1 today. The **trained model** cannot: see the measured note below. | Look-alikes are returned in a separate `lookalikes` array and are never passed to `run_attribution`. `tests/test_pipeline_offline.py` asserts the split exists. |
| Geometric properties | `app/geo/geometry.py`: area, perimeter, max Feret length, cross width, PCA orientation, compactness, centroid, bounding box, pixel count. All computed in a local azimuthal equidistant frame centred on the polygon. | `tests/test_geo_centroid.py` checks centroid to 1e-4 degrees, area to 1 percent, and orientation on a known bar. |
| Mask georeference | `app/geo/raster.py:load_mask_with_georef` copies the affine and CRS from the Sigma0 image onto the mask before any lat/lon maths. Zenodo masks routinely ship with none. | `test_mask_without_crs_inherits_the_sar_georeference`. |
| Age, if feasible | `app/pipeline.py` reports `age_hours_proxy`, the hours between the estimated origin and the acquisition. | The API field is named `age_hours_proxy`, the label reads "drift proxy, not lab age", and the UI repeats it. No chemistry model exists anywhere in the tree. |

## Clause (b) Oceanographic and meteorological drift

| Requirement | Implementation | How to verify |
| --- | --- | --- |
| Use oceanographic and meteorological data | `app/drift/fields.py` reads cached cubes of `u_current, v_current, u_wind10, v_wind10` on a regular time / lat / lon grid, bilinear in space and linear in time. | `GET /api/metocean` lists every cached cube with its real source string, licence, footprint and time range. |
| Data must be real, not mocked | `scripts/build_metocean_cache.py` fetches real ERA5 10 m wind and real marine currents from Open-Meteo, converts the direction conventions, and freezes npz. | The `source` field in each cube names the API and the caching date. If no cube matches, `synthetic` is `true`, the UI badge turns red, and the run carries a warning that clause (b) is not satisfied. |
| Trace the slick back to origin point and time | `app/drift/advection.py` runs the same RK2 integrator backwards. `app/drift/cone.py` builds the origin zone. | `test_backward_run_undoes_forward_run` proves the backward run is the forward physics reversed. |
| Physics | `V = U_current + 0.03 * U_wind10`, Stokes off, optional 15 degree leeway deflection, 50 particles, 1 hour step, fixed per-particle noise of 0.1 m/s on current and 1 m/s on wind. | `test_one_metre_per_second_for_one_hour_is_3_6_km` and `test_wind_contributes_exactly_alpha_of_its_speed`. |
| Origin is a zone, not a pin | 90 percent ensemble envelope at the chosen hour, buffered 2 km. Rule: first hour where the spread radius exceeds 8 km, clipped to 6 to 48 hours. | `drift.origin.spread_km` and `drift.origin.area_km2` are in every job document and printed in the sidebar. |
| Predict future flow | Forward ensemble for 36 hours, swept cone, hourly centroids, threatened bounding box. | `drift.cone_fwd`, `drift.forecast_hourly`, `drift.threatened_bbox`. |
| Coast impact flag | `app/drift/land.py` with a real Natural Earth coastline cached by `scripts/build_land_mask.py`. Still optional: skipped cleanly with a note when the file is absent. | `drift.coast.coast_flag` and the UI's Coast impact row. On the Caspian scene the cone overlaps land and it reports true; on MC20 it stays offshore and reports false. |
| OpenDrift not required | It is never imported at request time. The advection is 200 lines of numpy. | `grep -r opendrift app/` returns only comments. |

## Clause (c) AIS reconstruction, filtering, scoring, ranking

| Requirement | Implementation | How to verify |
| --- | --- | --- |
| MarineCadastre schema | `app/ais/ingest.py` implements all 16 columns: MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselName, IMO, CallSign, VesselType, Status, Length, Width, Draft, Cargo. | `test_sqlite_roundtrip_preserves_marinecadastre_columns` exports the store and compares the header. |
| Real AIS when it exists | `scripts/fetch_marinecadastre_ais.py` streams the national daily file, clips to the scene box while reading, ingests, then deletes the raw archive. | `app/scenes.py:in_nais_coverage` decides per scene from the footprint, and stores the decision as `ais_mode`. |
| Synthetic AIS only where allowed | `app/ais/synthetic.py` simulates traffic over the scene's real geobox and real time window. `build_synthetic_ais.py` refuses scenes inside real coverage unless forced. | The scene index records `ais_mode`, and the run warns how many candidate vessels came from simulated rows. |
| Historic reconstruction | `app/ais/interpolate.py` resamples each track to one minute, interpolating position in metres and course on the circle. | `test_course_interpolation_wraps_through_north`. |
| Filter irrelevant traffic | `app/ais/filter.py` keeps an MMSI only if some interpolated point is within the search radius of the origin zone inside the time window. | `attribution.funnel` reports considered, kept, dropped outside radius, dropped too few points, and the window. |
| Proximity score | `exp(-d_km / R)`, d from the closest approach to the estimated origin **point**, R the origin zone radius. Centre 1.00, zone edge 0.37. Measuring to the zone *boundary* and clamping to zero inside made this term a constant: an origin zone is routinely 200 km2 and 8 of 10 Gulf suspects scored exactly 1.00. | `test_proximity_does_not_saturate_across_a_wide_origin_zone`. |
| Temporal score | `exp(-|dt| / 1.5 h)` between the closest approach and the estimated origin time. The problem statement asks for spatio-temporal correlation; the plus or minus three hour window is a filter, and this is the score. | `test_time_offset_changes_the_score`. |
| Track confidence | `clamp(1 - 0.65 * dead_reckoned_fraction, 0.35, 1)`, pro-rated below 10 receptions, multiplied into the final percentage and reported beside it. Before it, the top Gulf suspect was a fishing boat with two AIS receptions on a 99.4 percent dead-reckoned track, and three vessels tied at 73.8 percent. | `test_a_two_ping_track_cannot_outrank_a_well_observed_one`, `test_confidence_is_reported_and_actually_applied`, `test_scores_are_distinct_enough_to_act_on`. |
| Trajectory score | 1.0 when the course at closest approach is within 45 degrees of the origin to slick bearing, else 0.4. | `suspect.detail.trajectory` shows course, drift bearing and the difference. |
| Vessel type score | Published prior over ITU-R M.1371 decoded types. | `GET /api/scoring` returns the whole table. `test_vessel_type_decoding_covers_the_ais_ranges`. |
| Behavioural anomalies | max of discharge speed band, course change, speed swing, and AIS gap. | `suspect.detail.behavior` shows each measured quantity, not just the verdict. |
| AIS gaps count as behaviour | A gap of 30 minutes or more whose dead reckoned segment passes within 5 km of the origin zone scores 0.95 and raises `NON-REPORTING` , but only on a vessel with at least 6 genuine receptions. Below that the gap scores 0.35 and is labelled `sparse_track_gap`, because a vessel seen twice in six hours has not demonstrably gone dark. | `test_ais_gap_across_the_origin_raises_the_reason_code`, the negative case beside it, and `test_a_two_ping_track_cannot_outrank_a_well_observed_one`. |
| Explainable scores | Every suspect carries components, weighted contributions and reason codes. | `test_every_suspect_carries_its_reasons`. |
| Rank culprit vessels | Absolute weighted sum, never batch-normalised. | `test_score_is_absolute_not_relative_to_the_batch`. |

## Product level requirements

| Requirement | Status | Evidence |
| --- | --- | --- |
| Every mandatory clause shipped | Yes | This document. |
| U-Net inference path verified | Yes | `tests/test_unet_path.py` loads a real checkpoint, checks its metadata round trips, asserts tiled stitching produces a valid distribution at every pixel, and confirms a corrupt checkpoint degrades to the baseline rather than crashing. |
| Runs fully offline on one laptop | Yes | `test_full_pipeline_runs_offline` blocks every outbound socket and DNS lookup and still completes all nine steps. |
| No Docker, cloud, ngrok, or runtime login | Yes | Network access exists only in `scripts/`, which run before the demo. |
| No LangChain, VLM, SAM2, vector DB, auth, React | Yes | `requirements.txt` and `app/static/` contain none of them. |
| Frozen stack | Yes | Python 3.11+, FastAPI, uvicorn, vanilla HTML plus Leaflet 1.9 from `app/static/vendor`. |
| Optional libraries are truly optional | Yes | rasterio, scipy, pyproj, shapely, Pillow, torch, smp and reportlab all have working fallbacks. `test_builtin_tiff_reader_matches_rasterio` checks the escape hatch agrees with the real thing. |
| Deterministic runs | Yes | `test_run_is_reproducible` compares two runs field by field. Seed is `TIDETRACE_SEED`, default 20260920. |
| No em dashes in UI copy | Yes | `grep` over `app/static/` returns none. |
| Dataset licences cited | Yes | README data table. |
| Demo works without CUDA | Yes | The whole system was built and verified on a CPU-only machine with no torch installed. |
| Training corpus stays off the demo laptop | Yes | Tiles and checkpoints live on Hugging Face. The laptop pulls one file under 80 MB via `scripts/hf_sync.py pull-model`. |
| Training survives the Kaggle session limit | Yes | The checkpoint is pushed on every IoU improvement, optimiser state after every epoch, and `--time-budget` stops cleanly before the kill. `--resume` continues from the Hub. |
| No credential is committed | Yes | `test_no_credentials_are_committed` scans the tree for token-shaped strings. Kaggle needs no credential at all: public repos, chained kernel outputs, and results returned through the Kaggle CLI. |
| Network clients cannot reach the request path | Yes | `test_offline_boundary.py` parses every module under `app/` and fails on a network import, then checks `sys.modules` after importing `app.main`. |

## Anti-slop checklist

The spec lists code patterns that should cause a rejection. Each one, and where
it is prevented:

| Rejected pattern | Prevented by |
| --- | --- |
| Hardcoded leaderboard or a fixed MMSI winner | Nothing in `app/` contains an MMSI literal. `test_a_closer_fishing_boat_still_beats_a_distant_tanker` proves geometry can overrule the type prior. |
| `/api/run` returning fixture JSON | The route calls `pipeline.run`, which calls detection, advection and the AIS join in order. `test_empty_ais_store_yields_no_suspects_not_a_crash` shows the output tracks the actual store contents. |
| Random or constant values labelled "currents" | `fields.synthetic_field` is the only constant field, it sets `synthetic: True`, the UI badge turns red and the run warns that clause (b) is unsatisfied. |
| Synthetic SAR blobs instead of real imagery | Synthetic rasters exist only in `scripts/make_selftest_scene.py`, are marked `selftest: true` and are hidden from `/api/scenes` by default. |
| UI that only filters a table | The console draws the SAR raster, both polygon classes, the backtrack, the origin zone, both cones, ranked AIS tracks, dead reckoned segments and a time slider. |
| OpenDrift, CMEMS or Open-Meteo as a required runtime import | None of them appear in `app/`. |
| Demo that dies without CUDA | Built and verified on CPU with no torch installed. |
| Guaranteed rank 1 tanker via score hacking | The simulator writes its ground truth to a file no scoring code reads, and gives distractors competitive geometry on purpose. |

## The look-alike class: a gap in the data, not the code

The spec sheet says to map a look-alike folder's mask value 1 to class 1. That
rule cannot be applied, because there is no mask value 1 to map.

Measured directly from Part II's `01_Train_Val_Lookalike_mask.7z`, which is
0.43 MB and so cheap to reproduce:

```
00000.tif  shape=(2048, 2048)  dtype=uint8  unique=[0]  positive=0 (0.00%)
... every sampled look-alike mask: 0 positive pixels
```

The Zenodo ground truth segments **mineral oil only**. A look-alike image is
dark water that is not oil, so an empty oil mask is the correct annotation for
it. There is no pixel-level look-alike label anywhere in the dataset.

Three options existed. Threshold the dark patches and call the result class 1,
which is inventing ground truth and is what the anti-slop rules forbid. Keep a
class-1 head that is never supervised, which is a dead output dressed as a
feature. Or use those chips for what they are genuinely good for.

**What was done:** look-alike chips are kept as **hard negatives**, labelled
sea. They are the most valuable negatives in the set, because the failure they
prevent is the one that matters: calling dark water oil. The tile index reports
`hard_negative_chips`; the training report lists `unsupervised_classes` and
prints an explanation rather than a bare NaN.

**What this means for the pitch:** the trained model is a binary oil segmenter
with strong hard negatives, and `IoU_lookalike` is undefined for it. The
look-alike concept still exists in the product: the dB baseline separates them
on contrast and compactness, the UI draws them yellow, and they are excluded
from attribution. Say that plainly rather than showing a NaN and hoping nobody
asks.

## The validation metric, and why IoU alone was not one

The first checkpoint scored **IoU oil 0.8575** on held-out tiles, beating the
dB baseline by +0.8575, and was useless. Run on the three demo scenes it
returned 483, 498 and 503 km² of oil on chips of about 484 km², including the
clean-water control, which must return nothing. It had learned to answer "oil".

It was not a threshold problem. On the control scene the oil softmax had median
0.953 and a 5th percentile of 0.838: saturated, with nothing to separate.

The cause was in the tile budget. Look-alike chips carry empty masks, so each
became a hard negative labelled sea, and because the look-alike folders are
walked first they consumed the entire 150-tile sea budget before `Images/No oil`
was opened. The run wrote 400 oil tiles, 150 look-alike-derived sea tiles and
zero tiles of ordinary open water, 73% oil, against roughly 0.1% in a real
scene. `CLASS_WEIGHTS = [0.4, 1.0, 2.5]` then added a 6.25× push toward oil on
top of that.

**The metric was the real defect.** IoU_oil is monotone in "predict more oil"
when it is measured only on tiles selected for containing oil, so ranking
epochs by it selects approximately the worst-calibrated epoch available. It was
not lying; it was answering a question nobody should have asked.

So the gate is now a different number:

    water_false_oil = (cm[0,2] + cm[1,2]) / (true sea + true look-alike)

the fraction of genuinely-water pixels called oil. `MAX_WATER_FALSE_OIL = 0.02`
is a hard ceiling: an epoch above it cannot become the shipped checkpoint
however good its IoU, and if no epoch clears it the run reports the closest
miss and **writes nothing**. `tests/test_training_smoke.py` covers the gate
firing and the metric itself; the plumbing-only smoke tests disable it by name
so the default cannot be weakened quietly.

Alongside it, `max_hard_negatives` stops look-alike chips starving the sea
budget, and `max_scene_negatives` keeps the open-water tiles cut from oil
scenes instead of discarding them, the best negatives available, differing
from the positives only in the thing being learned.

The broken checkpoint is kept in `models/quarantine/` with its report, so the
claim above can be checked rather than taken on trust.

## Known gaps, stated rather than hidden

1. **The shipped detector never draws a look-alike.** A checkpoint now ships
   and loads: `UnetPlusPlus` / `timm-efficientnet-b0`, `IoU_oil` 0.889 against a
   baseline of 0.000 on the same held-out tiles, `water_false_oil` 0.011 under a
   0.02 ceiling. But because the Zenodo ground truth labels oil only, class 1 is
   unsupervised, and the consequence was not previously carried through to this
   document. Measured across all four shipped scenes with the U-Net loaded:

   | scene | U-Net oil | U-Net look-alikes | baseline oil | baseline look-alikes |
   | --- | --- | --- | --- | --- |
   | gom_mc20_chronic_slick | 29 polys, 5.149 km2 | **0** | 4 polys, 0.409 km2 | 73 |
   | caspian_baku_seeps | 1 poly, 0.074 km2 | **0** | 11 polys, 15.409 km2 | 15 |
   | santa_barbara_seeps | 2 polys, 0.356 km2 | **0** | 4 polys, 82.073 km2 | 4 |
   | arabian_sea_mumbai_offshore | 0, clean water | **0** | 0, clean water | 0 |

   So clause a.2 holds on the baseline path and not on the shipped one. The
   yellow dashed polygons in the README and in the demo script are visible only
   when the baseline runs. Closing this needs look-alike labels that do not
   exist in the corpus; inventing them is what produced the first quarantined
   checkpoint, and it is not going to be repeated to make a table look better.

2. **The baseline has no measured skill, and the two detectors disagree by two
   orders of magnitude.** `IoU_oil` 0.000, `IoU_lookalike` 0.000, mIoU 0.288.
   On Santa Barbara it reports 82 km2 against the model's 0.36. It exists so
   that clauses (a), (b) and (c) still run with no checkpoint and no GPU, and
   that is the whole of its claim. "The baseline still produces a polygon" is
   true and was previously stated as reassurance; it produces the wrong one.

3. **A clean scene is now a finding rather than a blank.** The Arabian Sea chip
   returns nothing from either detector, and that is correct: its co-pol band
   spans 1.95 dB after speckle averaging against 7.0 dB on the Gulf chip, which
   has a visible slick. It is uniform wind-roughened water. Every run now
   measures and reports `sea_level_db` and `dynamic_range_db`, and a scene under
   the 3 dB floor is reported as clean water with the number attached, because
   an empty panel and a broken detector look identical otherwise.

4. **Scene radiometry is measured but not corrected.** Open water across the
   shipped scenes spans -17.4 dB to -22.9 dB, which means the frozen corpus
   normalisation is reading each scene at a different point on the model's
   response curve. A single clamped dB translation to fix that is implemented
   and flag-controlled, and it is **off by default**: the checkpoint records the
   corpus mean, not the corpus water level, and aligning to the mean brightened
   the Gulf chip until a correct 29-polygon detection became zero. The offset is
   reported on every run so an operator can see when a scene is out of range;
   correcting it properly needs a checkpoint that records `sea_level_db`.
5. **EO is used for corroboration, not as a second detector.** Every scene
   carries a cloud-masked Sentinel-2 L2A chip, and the pipeline has an `EO` stage
   that compares each SAR detection against it: inside the polygon versus a ring
   around it. A surface film reads darker than the water around it; a rig, a
   sandbar or a ship reads brighter; a low-wind cell reads like nothing at all.
   Each polygon carries a verdict of `consistent`, `inconsistent`, `neutral` or
   `obscured`, and the API, the verdict card and the polygon popups all show it.

   What it deliberately is not is a fused SAR+EO detector. There is no paired
   labelled dataset for that, and inventing supervision for one would repeat the
   mistake that produced the first quarantined checkpoint. The load-bearing
   caveat travels with every verdict: Sentinel-2 did not observe this water at
   the radar acquisition time -- offsets run from 17 hours to 17 days -- so
   agreement supports a detection and disagreement does not overturn one.
   **Nothing is added, removed or reweighted by the check.**

   The first version of this got it wrong in an instructive way. A Sentinel-2
   granule often covers only part of a scene footprint and the chip cutter fills
   the rest with zero. On the Gulf scene that emptiness sat exactly where the
   detections were, and because only saturated white counted as obscured, all 29
   polygons returned "neutral, no optical difference" with a delta of exactly
   0.0: a confident verdict computed from a black image. Nodata is now treated as
   obscured, the Gulf reports 28 obscured, and
   `tests/test_eo_corroboration.py` pins it.
6. **Ocean currents are unavailable in the Caspian, and Copernicus does not fix
   it.** Currents now come from Copernicus Marine's eddy-resolving 1/12 degree
   analysis, which is what an operational drift model would use, with Open-Meteo
   as the fallback. It covers the Gulf of Mexico at 0.176 m/s mean and the
   Arabian Sea at 0.065 m/s.

   It does **not** cover the Caspian. That sea is endorheic, so the global ocean
   model carries it as land: a query over the Baku footprint returns 306 cells
   and 306 NaNs, and Copernicus publishes no Caspian product either. This was
   worth checking rather than assuming an account would close the gap. Instead
   of letting a zero field pass silently, `has_currents` is computed and the run
   states that the drift there is wind driven only.
