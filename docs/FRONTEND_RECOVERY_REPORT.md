# Varuna Frontend Recovery Report

Date: 2026-10-01

> **Superseded in part.** This report records the state at the moment of the
> historical restore and is kept as an audit trail. The console has since moved
> on again: it now carries seven views (`investigate`, `drift`, `vessels`,
> `cases`, `method`, `data`, `operations`) and renders the `/api/cases/{id}/`
> claims properly rather than as raw JSON. Sections B, D, G and H below describe
> the six-mode branch at `3492cea`, not the current tree, and the 12 failures
> reported in section F are resolved. The audit in `FRONTEND_AUDIT.md` is
> written against the same earlier snapshot.

## A. Current Git State

The repository is on `feat/case-consoles-and-auditable-report` at `3492cea` (`Case consoles, auditable report, and honest uncertainty accounting`). Available history:

| Commit | Role |
| --- | --- |
| `e8743ab` | Initial TideTrace snapshot |
| `b22677a` (`main`, `origin/main`) | Rename-only Varuna baseline |
| `3492cea` (`HEAD`) | Case-console / auditable-report feature branch |

Before recovery, the worktree already had unrelated changes to README, backend, model, data, dependencies, and tests, plus untracked `workstation.css`, `recovered_style.css`, `recovered_app.js`, an earlier frontend audit, and a test. They were preserved. Recovery intentionally changed only `app/static/index.html`, `app/static/app.js`, and `app/static/style.css`; those files exactly match `b22677a` and remain uncommitted.

## B. Later-Branch Frontend Changes

`3492cea` changed the served frontend by 1,279 net lines: JavaScript +849/-39, markup +112/-62, CSS +318/-5. It replaced the original five-mode three-rail console (`Investigate`, `Drift`, `Vessels`, `Method`, `Data`) with six case-console modes (`Overview`, `Detection`, `Origin`, `Attribution`, `Sensitivity`, `Provenance`), a case rail, and new case-quality, uncertainty, sensitivity, calibration, and provenance surfaces. Associated backend/pipeline/report changes are retained.

The untracked recovery/workstation assets are not linked by the restored page. The existing `app/main.py` cache-stamp change that mentions `workstation.css` was left untouched.

## C. Restoration

A precise known-good Varuna frontend exists at `b22677a`. The three served static files were restored file-by-file from that commit. No reset, checkout, backend/API edit, data rewrite, or test rewrite occurred.

The source matches the repository's earlier screenshot-era layout. The referenced screenshot attachment was not separately available to the audit tooling, so this is source- and repository-screenshot-based validation, not a claim of pixel comparison to an unseen attachment.

## D. Functional Requirements Matrix

`Yes` means the recovered console exposes the feature and its current backend contract remains available. `Gap` means the feature exists only in the later case-console branch, so the exact historical recovery does not expose it.

| Feature | Recovered implementation / UI | Backend or data source | Preserve | Verification |
| --- | --- | --- | --- | --- |
| Scene selection | Left `Scenes` rows and select | `GET /api/scenes` | Yes | Four scenes loaded |
| Case switching | Not in historical baseline | `GET /api/cases`, `GET /api/cases/{id}` | Gap | Reported; not reconstructed |
| Map | Centre Leaflet work surface | Vendored Leaflet, local basemap | Yes | 940x579, 32 tiles |
| SAR imagery | `SAR image` layer | Detection overlays | Yes | Loaded after run |
| Optical imagery | `Optical (S2)` layer | Local optical JSON, EO result | Yes | Loader retained |
| Class mask | `Class mask` layer | Detection mask overlay | Yes | Loaded after run |
| Oil polygons | Layer and detection summary | `detection.polygons` | Yes | Four polygons returned |
| Look-alikes | Layer and excluded evidence count | `detection.lookalikes` | Yes | 30 returned |
| Backtrack | Layer and Drift view | `drift.hindcast_track` | Yes | Origin timeline shown |
| Hindcast cone | Layer | `drift.cone_back` | Yes | Renderer retained |
| Release zone | Layer and Drift facts | `drift.origin_zone` | Yes | 8.3 km radius shown |
| Forecast track | Layer | `drift.forecast_track` | Yes | Renderer retained |
| Forecast cone | Layer | `drift.cone_fwd` | Yes | Renderer retained |
| AIS tracks | Layer | Attributed vessel tracks | Yes | Renderer retained |
| Vessel markers | Layer and time scrubber | Attributed vessel tracks | Yes | Playback retained |
| Analysis controls | Left run-parameter rail | `POST /api/run` | Yes | Visible and active |
| Hindcast hours | `Hindcast h` | `hindcast_hours` | Yes | Posted by run |
| Forecast hours | `Forecast h` | `forecast_hours` | Yes | Posted by run |
| Search radius | `Search km` | `search_radius_km` | Yes | Posted by run |
| Window hours | `Window h` | `origin_window_hours` | Yes | Posted by run |
| Run analysis | Left primary button | `POST /api/run` | Yes | Full run completed in 3.9 s |
| Point probe | Checkbox + map click | `POST /api/demo/inject` | Yes | Event wiring retained |
| Case signal | Active/clean status and overview | Job status/detection/drift | Partial | Later quality band absent |
| Detection metrics | Investigate detection panel | `detection.metrics` | Yes | Geometry/confidence shown |
| Slick geometry | Detection/map data | Polygon properties | Yes | Area, dimensions, perimeter shown |
| Optical corroboration | Evidence + layer | `detection.eo` | Yes | Zero chips honestly shown |
| Detector provenance | Detection and sources | Metadata, health | Yes | Baseline caveat shown |
| Evidence | Investigate evidence block | Job document | Yes | Counts/trace rendered |
| Drift | Drift view + map | `job.drift` | Yes | Origin/metocean/spread shown |
| Vessel attribution | Vessels view/search | `job.attribution` | Yes | No culprit forced |
| Method | Method view | `GET /api/scoring` | Yes | Weights/formula shown |
| Data | Data view | Scene, health, job | Yes | System/provenance/export retained |
| Reports | Attribution note export | `GET /api/report/{id}` | Yes | Enabled after run |
| Job loading/reloading | Fragment loader | `GET /api/jobs/{id}` | Yes | Stored job reopened |
| Progress | Busy overlay + polling | `GET /api/jobs/{id}/progress` | Yes | Observed during run |
| Shareable job state | `#job=<id>&view=<mode>` | Stored job API | Yes | Validated |
| Real vs simulated AIS | Scene-row labels | `scene.ais_mode` | Yes | Both labels visible |
| Observed vs reconstructed AIS | Track-confidence explanation | Candidate track fields | Yes | Received/dead-reckoned labels retained |
| Offline operation | Local assets/runtime summary | Vendored assets, cache, health | Yes | Health and local assets loaded |

## E. Existing API Dependencies

The restored client directly uses `GET /api/health`, `GET /api/scenes`, `GET /api/scoring`, `POST /api/run`, `GET /api/jobs/{id}/progress`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/geojson`, `GET /api/report/{id}`, and `POST /api/demo/inject`. It reads local `/data/optical/`, `/data/cache/`, and `/data/basemap/` assets and vendored Leaflet files. Later `/api/cases/*` and evaluation APIs remain served but are not called by the exact baseline. No contracts changed.

## F. Missing Or Broken Functionality

- The exact baseline predates case switching, the explicit case-quality/safe-fail band, uncertainty chain, counter-evidence panel, sensitivity, ablation, calibration, and six renamed modes. This is intentional historical restoration, not a backend regression.
- `tests/test_console_dom.py` now asserts the later six-mode design. Full-suite result: **149 passed, 12 failed, 7 skipped**; all failures are in that redesign-specific DOM suite. Backend/scientific tests passed.
- Optional land and neighbouring basemap tiles are missing locally. The console handles this and reports coast impact unavailable without a land mask.
- At the narrow 614 px browser viewport, the legacy desktop grid clips/collapses the map area. The desktop validation viewport renders normally. Responsive repair is deferred.

## G. Screenshot And Baseline Difference

The restored baseline is the earlier screenshot-era Varuna: five navigation views, left scene/run rail, central map/layer control/pipeline strips, and right evidence rail. Pre-recovery HEAD instead had a Cases rail, six case-oriented views, case-quality/sensitivity/provenance surfaces, and moved pipeline/environment panels. Those later visual choices are absent by design.

## H. Future Redesign Scope

Future presentation work should initially be limited to:

- `app/static/index.html`
- `app/static/style.css`
- `app/static/app.js`

It can deliberately surface current case endpoints from `app/api/cases.py`, but should not modify `app/main.py`, scientific pipeline modules, schemas, or API contracts for visual work. The untracked `workstation.css`, `recovered_style.css`, and `recovered_app.js` require separate review rather than silent adoption.

