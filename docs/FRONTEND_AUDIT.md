# Frontend Feature Inventory

> **Superseded in part.** This inventory was taken before the workstation
> redesign and describes the five-view console at `b22677a`. The current tree has
> seven views and a Cases view that reads `/api/cases/{id}/*`. Kept as an audit
> trail; see `FRONTEND_RECOVERY_REPORT.md` for the same caveat.

Audit completed before the workstation redesign. The served application is a
vanilla HTML/CSS/JavaScript Leaflet console; it has no external runtime assets.

| Current feature | UI location | API/data source | Interaction | Preserve |
| --- | --- | --- | --- | --- |
| Scene and saved-case selection | Left case/acquisition rail | `GET /api/scenes`, `GET /api/cases`, `GET /api/jobs/{id}` | Select scene or reopen a case | Yes |
| Full analysis run and progress | Analysis controls | `POST /api/run`, `GET /api/jobs/{id}/progress` | Run, poll, render completed job | Yes |
| Point probe | Analysis controls and map | `POST /api/demo/inject` | Arm, click map, run drift/AIS only | Yes |
| SAR, mask, slick and look-alike layers | Leaflet map and layer control | Job overlays and detection GeoJSON | Toggle, inspect popups | Yes |
| Hindcast, origin zone and forecast | Leaflet map, Drift mode, timeline | Job drift document | Toggle, scrub frames, inspect envelopes | Yes |
| AIS tracks and directional vessel marks | Leaflet map, Vessels mode | Job attribution tracks | Toggle, select candidate, search MMSI/name | Yes |
| Candidate ranking and objections | Vessels mode | Job attribution/evidence | Select candidate, focus track, inspect supporting and counter-evidence | Yes |
| Evidence quality and uncertainty | Investigate mode | Job case-quality, chain and trace fields | Read stage certainty and limitations | Yes |
| Robustness and scoring method | Method mode | Job sensitivity/ablation/calibration, `GET /api/scoring` | Inspect scenarios and weighting | Yes |
| Provenance and system state | Data mode and left rail | Job provenance, `GET /api/health` | Inspect detector, AIS, metocean and offline state | Yes |
| Timeline and playback | Centre timeline | Job tracks/drift timestamps | Scrub or play vessel positions | Yes |
| Exports and report | Data mode and Report action | `GET /api/jobs/{id}`, `/geojson`, `GET /api/report/{id}` | Open JSON, GeoJSON or report | Yes |
| Shareable job URLs | Browser fragment | `#job=<id>&view=<mode>` | Reload stored job without recomputation | Yes |
| Theme and rail state | Header | Local browser storage | Toggle light/dark and fold rails | Yes |

## Backend capability not surfaced as a dedicated control

The console intentionally uses the full pipeline rather than duplicating its
stages. The following APIs remain available for integration or diagnostics but
are not independent front-end workflows: `POST /api/detect`,
`POST /api/detect/upload`, `POST /api/drift`, `POST /api/attribute`,
`GET /api/ais/track/{mmsi}`, `GET /api/ais/window`, `GET /api/config`,
`GET /api/metocean`, `GET /api/ais/stats`, and
`GET /api/evaluation/attribution/{job_id}`. No backend contract was changed by
the redesign.
