/* NTRO Oil Spill Attribution Console
   Vanilla JS, Leaflet only. No framework, no build step, no CDN at runtime.
   Every value rendered here comes from the job document the API returned. */

(function () {
  "use strict";

  var API = "";
  var state = {
    scenes: [],
    scene: null,
    job: null,
    config: null,
    health: null,
    suspects: [],
    selected: null,
    playing: false,
    timer: null,
    frames: [],
    frameIndex: 0
  };

  var layers = {};
  var map = null;
  var layerGroup = {};

  // ---------------------------------------------------------------- helpers
  function $(id) { return document.getElementById(id); }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  function fmt(v, digits) {
    if (v === null || v === undefined || v === "" || (typeof v === "number" && !isFinite(v))) return "n/a";
    if (typeof v === "number") return v.toFixed(digits === undefined ? 2 : digits);
    return String(v);
  }

  function utc(iso) {
    if (!iso) return "n/a";
    var d = new Date(iso);
    if (isNaN(d.getTime())) return String(iso);
    return d.toISOString().replace("T", " ").replace(/\.\d+Z?$/, "").replace("Z", "") + " UTC";
  }

  function hhmm(ts) {
    var d = new Date(ts * 1000);
    return d.toISOString().substring(0, 16).replace("T", " ") + " UTC";
  }

  function getJSON(path) {
    return fetch(API + path, { headers: { "Accept": "application/json" } })
      .then(function (r) {
        if (!r.ok) return r.text().then(function (t) { throw new Error(r.status + " " + t.slice(0, 300)); });
        return r.json();
      });
  }

  function postJSON(path, body) {
    return fetch(API + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    }).then(function (r) {
      if (!r.ok) return r.text().then(function (t) { throw new Error(r.status + " " + t.slice(0, 400)); });
      return r.json();
    });
  }

  // ------------------------------------------------------------------- map
  function initMap() {
    map = L.map("map", {
      // Zoom sits on the right. The layers panel owns the top-left corner, and
      // stacking a control under a panel makes the control unclickable.
      zoomControl: false,
      attributionControl: true,
      preferCanvas: true,
      worldCopyJump: false,
      // Leaflet fades tiles in over a few frames. If the map is re-fitted while
      // a fade is in flight -- which is exactly what happens when the scene
      // footprint is framed on load -- the fade stalls and the tiles are left
      // at opacity 0. They are decoded, positioned and invisible, so the
      // console showed a black map over perfectly good cached imagery. There is
      // nothing to fade for a local tile store anyway.
      fadeAnimation: false
    }).setView([20, 78], 4);

    map.attributionControl.setPrefix("");
    map.attributionControl.addAttribution("Varuna offline console");

    L.control.zoom({ position: "topright" }).addTo(map);
    L.control.scale({ imperial: false, position: "bottomright" }).addTo(map);

    // Exposed for diagnostics and for the browser checks that drive this map
    // during development. Read-only as far as the app is concerned.
    window.__varuna = { map: map, layers: layerGroup, state: state };

    // Leaflet stacks everything in `overlayPane` by DOM insertion order, and the
    // SAR backdrop is inserted at RUN time -- after the optical chip, which is
    // loaded when the scene is picked. So optical always ended up underneath it
    // and ticking the box appeared to do nothing. Explicit panes fix the order
    // properly: both sit below overlayPane (400), so the class mask and every
    // vector layer still draw above them.
    map.createPane("sarPane").style.zIndex = 350;
    map.createPane("opticalPane").style.zIndex = 360;

    // Order is z-order: later goes on top. Optical sits directly ABOVE the SAR
    // backdrop so that ticking it actually reveals something -- underneath a
    // 0.95-opacity radar image it would be invisible and the control would look
    // broken. The class mask and every vector layer stay above both.
    ["sar", "optical", "mask", "oil", "lookalike", "hindcast",
      "cone_back", "origin", "forecast", "cone_fwd", "tracks",
      "vessels"].forEach(function (name) {
        layerGroup[name] = L.layerGroup().addTo(map);
      });

    // Starts hidden, matching its unchecked box. It is context, not evidence.
    map.removeLayer(layerGroup.optical);

    addBasemaps();

    map.on("mousemove", function (e) {
      $("readout").textContent = "lat " + e.latlng.lat.toFixed(4) +
        "  lon " + e.latlng.lng.toFixed(4);
    });
  }

  /* A 1x1 transparent PNG. Outside the cached footprint there is simply no
     tile, and a missing tile should reveal the dark canvas and the graticule
     rather than a broken-image icon. */
  var BLANK = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";

  /* Basemaps are cached, not live.

     The console runs in airplane mode, so a normal tile layer pointed at a
     provider would go blank the moment the network did. Instead
     scripts/fetch_basemap.py downloads the tiles covering the scene footprints
     once, and they are served from data/basemap. Inside that footprint you get
     real imagery; outside it you get the graticule, which is honest: it shows
     exactly where there is data and where there is not. */
  // Shown when the current view intersects none of the cached footprints.
  function updateCoverageNote() {
    var note = document.getElementById("coverage-note");
    if (!note) {
      note = el("div", "coverage-note", "");
      note.id = "coverage-note";
      document.getElementById("mapwrap").appendChild(note);
    }
    var boxes = state.basemapBounds || {};
    var view = map.getBounds();
    var covered = Object.keys(boxes).some(function (k) {
      var b = boxes[k];   // [west, south, east, north]
      return view.intersects(L.latLngBounds([[b[1], b[0]], [b[3], b[2]]]));
    });
    note.textContent = covered ? "" :
      "Outside cached imagery. Satellite tiles were downloaded for the indexed " +
      "scene footprints only; elsewhere the chart base is drawn locally.";
    note.style.display = covered ? "none" : "block";
  }

  /* The chart base: a tile layer drawn in the browser, from nothing.

     Cached imagery only exists for the indexed scene footprints, so panning off
     one used to leave bare canvas with a few graticule lines over it. That reads
     as a broken map, and "the tiles you need were never downloaded" is not a
     thought a judge should have to have.

     This draws every tile on a canvas at request time: an ocean ground, a
     depth-of-field wash so the sea is not a flat fill, the graticule as part of
     the tile rather than as vectors laid over it, and the parallel and meridian
     stamped in the corner. It costs no network, no disk and no bundle, it is
     seamless at every zoom, and it covers the whole globe. Real imagery still
     draws on top wherever we actually have it.

     A chart with no soundings is still a chart. A blank rectangle is not. */
  function chartBaseLayer() {
    var Chart = L.GridLayer.extend({
      createTile: function (coords) {
        var size = this.getTileSize();
        var tile = document.createElement("canvas");
        tile.width = size.x;
        tile.height = size.y;
        var ctx = tile.getContext("2d");
        var w = size.x, h = size.y;

        // Ground. Two stops, so a wide view has some depth to it rather than
        // reading as one flat colour across the whole viewport.
        // Deep-water tone, pitched to sit close to the open ocean in the cached
        // satellite imagery. Where a tile is missing the join then reads as more
        // sea rather than as a hole, which is both better looking and more
        // truthful: offshore of the footprint there really is only more sea.
        var wash = ctx.createLinearGradient(0, 0, w, h);
        wash.addColorStop(0, "#0f1e2b");
        wash.addColorStop(1, "#132735");
        ctx.fillStyle = wash;
        ctx.fillRect(0, 0, w, h);

        // Continents. Without these, zooming out showed ocean, a graticule and
        // four islands of satellite imagery, which reads as a failed load.
        paintLand(ctx, coords, size, this._map);

        // Graticule, drawn into the tile. Leaflet gives us the tile's own
        // lat/lon corners, so the lines land on whole degrees rather than on
        // tile edges, and the interval opens up as you zoom out.
        var nw = this._map.unproject([coords.x * w, coords.y * h], coords.z);
        var se = this._map.unproject([(coords.x + 1) * w, (coords.y + 1) * h], coords.z);
        var stepChoices = [30, 10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01];
        var spanLon = Math.abs(se.lng - nw.lng);
        var step = stepChoices[0];
        for (var i = 0; i < stepChoices.length; i++) {
          if (spanLon / stepChoices[i] <= 4) { step = stepChoices[i]; break; }
        }

        ctx.strokeStyle = "rgba(148, 178, 198, 0.13)";
        ctx.lineWidth = 1;
        ctx.beginPath();

        var firstLon = Math.ceil(nw.lng / step) * step;
        for (var lon = firstLon; lon < se.lng; lon += step) {
          var px = ((lon - nw.lng) / (se.lng - nw.lng)) * w;
          ctx.moveTo(Math.round(px) + 0.5, 0);
          ctx.lineTo(Math.round(px) + 0.5, h);
        }
        var firstLat = Math.floor(nw.lat / step) * step;
        for (var lat = firstLat; lat > se.lat; lat -= step) {
          var py = ((nw.lat - lat) / (nw.lat - se.lat)) * h;
          ctx.moveTo(0, Math.round(py) + 0.5);
          ctx.lineTo(w, Math.round(py) + 0.5);
        }
        ctx.stroke();

        // The tile's own north-west corner, stamped like a chart margin.
        ctx.fillStyle = "rgba(148, 178, 198, 0.30)";
        ctx.font = "10px ui-monospace, Consolas, monospace";
        ctx.fillText(_dm(nw.lat, "NS") + "  " + _dm(nw.lng, "EW"), 6, 14);

        return tile;
      }
    });
    return new Chart({ minZoom: 0, maxZoom: 18, tileSize: 256, attribution: "" });
  }

  /* Degrees and decimal minutes, the way a chart margin writes them. */
  function _dm(v, hemis) {
    var hemi = hemis[v < 0 ? 1 : 0];
    var a = Math.abs(v);
    var d = Math.floor(a);
    var m = (a - d) * 60;
    return d + "°" + (m < 10 ? "0" : "") + m.toFixed(1) + "'" + hemi;
  }

  /* World coastline, drawn into the chart tiles.

     The satellite cache covers the four scene footprints and nothing else,
     which is the right trade for an offline demo but left zooming out looking
     like a broken tile pipeline: four imagery patches floating in an empty
     wash, with no continents anywhere. Natural Earth land, simplified to 79 KB
     coarse and 645 KB detailed, fills that in at every zoom with no network and
     no tiles.

     It is rasterised into each chart tile rather than added as a vector layer,
     because a vector layer re-paths every ring on every pan frame; a tile is
     drawn once and then moved by the browser. That is the difference between a
     map that drags smoothly and one that stutters. */
  var LAND = { coarse: null, detail: null, loading: false };

  function loadWorldLand() {
    if (LAND.loading) return;
    LAND.loading = true;
    var pending = 2;
    function done() { if (--pending === 0 && map) redrawChartBase(); }
    getJSON("/data/land/world_land_coarse.json")
      .then(function (d) { LAND.coarse = d.rings; }).catch(function () {}).then(done);
    getJSON("/data/land/world_land_detail.json")
      .then(function (d) { LAND.detail = d.rings; }).catch(function () {}).then(done);
  }

  function redrawChartBase() {
    if (state.chartLayer && state.chartLayer.redraw) state.chartLayer.redraw();
  }

  /* Paint the land rings that touch this tile. Rings are pre-simplified, so the
     only per-tile work is a bounds test and a path. */
  function paintLand(ctx, coords, size, map_) {
    var rings = (coords.z >= 6 ? LAND.detail : null) || LAND.coarse;
    if (!rings) return;

    var w = size.x, h = size.y;
    var nw = map_.unproject([coords.x * w, coords.y * h], coords.z);
    var se = map_.unproject([(coords.x + 1) * w, (coords.y + 1) * h], coords.z);
    var west = nw.lng, east = se.lng, north = nw.lat, south = se.lat;
    // A degree of slack, so a coastline that only clips the corner still draws.
    var pad = Math.max(1, (east - west) * 0.5);
    var originX = coords.x * w, originY = coords.y * h;

    ctx.beginPath();
    for (var i = 0; i < rings.length; i++) {
      var ring = rings[i];
      var minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9, j;
      for (j = 0; j < ring.length; j++) {
        var px = ring[j][0], py = ring[j][1];
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
      }
      if (maxX < west - pad || minX > east + pad ||
          maxY < south - pad || minY > north + pad) continue;

      for (j = 0; j < ring.length; j++) {
        var pt = map_.project([ring[j][1], ring[j][0]], coords.z);
        var x = pt.x - originX, y = pt.y - originY;
        if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
    }
    ctx.fillStyle = LAND_FILL;
    ctx.fill("evenodd");
    ctx.strokeStyle = LAND_EDGE;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  var LAND_FILL = "#1d2f3c";
  var LAND_EDGE = "rgba(126, 164, 188, 0.55)";

  function addBasemaps() {
    // Drawn before anything else and never removed, so there is always a chart
    // under the overlays no matter where the operator pans.
    state.chartLayer = chartBaseLayer();
    state.chartLayer.addTo(map);
    loadWorldLand();

    getJSON("/data/basemap/manifest.json").then(function (m) {
      var defs = [
        ["satellite", "Satellite", "Esri, Maxar, Earthstar Geographics"],
        ["ocean", "Nautical", "Esri, GEBCO, NOAA, National Geographic"]
      ];
      var bases = {};
      defs.forEach(function (d) {
        if (!m.layers || !m.layers[d[0]]) return;
        bases[d[1]] = L.tileLayer(
          "/data/basemap/" + d[0] + "/{z}/{x}/{y}." + m.layers[d[0]].ext, {
            minZoom: 1,
            // Tiles are cached for zoom 5 to 13 only. Without minNativeZoom,
            // zooming out past 5 requested tiles that were never downloaded and
            // the map went completely blank -- which reads as broken rather
            // than as "no data here". Both bounds now reuse the nearest cached
            // level, scaled, so there is always something under the overlays.
            minNativeZoom: m.min_zoom || 5,
            maxNativeZoom: m.max_zoom || 13,
            maxZoom: 18,
            errorTileUrl: BLANK,
            attribution: d[2],
            crossOrigin: false
          });
      });
      bases["Graticule only"] = L.layerGroup();

      var overlays = {};
      if (m.layers && m.layers.ocean_labels) {
        overlays["Place names"] = L.tileLayer(
          "/data/basemap/ocean_labels/{z}/{x}/{y}." + m.layers.ocean_labels.ext,
          { minNativeZoom: m.min_zoom || 5, maxNativeZoom: m.max_zoom || 13,
            maxZoom: 18, errorTileUrl: BLANK, opacity: 0.85 });
      }
      if (m.layers && m.layers.seamark) {
        overlays["Seamarks"] = L.tileLayer(
          "/data/basemap/seamark/{z}/{x}/{y}." + m.layers.seamark.ext,
          { minNativeZoom: m.min_zoom || 5, maxNativeZoom: m.max_zoom || 13,
            maxZoom: 18, errorTileUrl: BLANK,
            attribution: "OpenSeaMap, ODbL" });
      }

      // Satellite first if it exists; it is the one that makes open water read
      // as open water rather than as an empty screen.
      var first = bases["Satellite"] || bases["Nautical"] || bases["Graticule only"];
      first.addTo(map);
      state.basemap = first;

      L.control.layers(bases, overlays, { position: "topright", collapsed: true }).addTo(map);

      // Only the four scene footprints were ever cached. Panning away from them
      // leaves a correct but bare graticule, and a bare graticule looks like a
      // failure. Say which it is.
      state.basemapBounds = m.bounds || {};
      map.on("moveend zoomend", updateCoverageNote);
      updateCoverageNote();

      // Keep the drawn layers above the tiles.
      Object.keys(layerGroup).forEach(function (k) {
        if (layerGroup[k].bringToFront) layerGroup[k].bringToFront();
      });
      state.basemapAvailable = true;
    }).catch(function () {
      // No cache present. Say so once, in the layers card, instead of leaving
      // the operator wondering why the map is empty.
      state.basemapAvailable = false;
      var note = el("div", "notice",
        "No cached satellite imagery. The chart base is drawn locally. Run " +
        "scripts/fetch_basemap.py --all while online to cache imagery for the " +
        "scene areas; the demo stays offline afterwards.");
      var box = $("layers");
      if (box && box.parentNode) box.parentNode.appendChild(note);
    });
  }

  /* Graticule. Drawn under everything, and the only backdrop outside the
     cached basemap footprint. */
  /* The graticule used to be drawn here as vector polylines over a bare
     canvas. It is now part of the chart base tile, which is seamless, covers
     the whole globe and costs nothing to pan. */

  // Everything a run produces is cleared before the next one. The graticule and
  // the optical chip are not run output: they belong to the map and to the
  // scene. Clearing optical here wiped it on the first RUN and, since it only
  // reloads on scene change, it never came back.
  var KEEP_ON_RUN = { optical: true };

  function clearAll() {
    Object.keys(layerGroup).forEach(function (k) {
      if (!KEEP_ON_RUN[k]) layerGroup[k].clearLayers();
    });
  }

  function ringToLatLng(coords) {
    return coords.map(function (p) { return [p[1], p[0]]; });
  }

  // -------------------------------------------------------------- rendering
  function drawDetection(job) {
    var det = job.detection || {};
    var ov = det.overlays || {};

    if (ov.sar && ov.sar.url) {
      L.imageOverlay(ov.sar.url, ov.sar.bounds,
        { opacity: 0.95, interactive: false, pane: "sarPane" })
        .addTo(layerGroup.sar);
    }
    if (ov.mask && ov.mask.url) {
      L.imageOverlay(ov.mask.url, ov.mask.bounds, { opacity: 0.75, interactive: false })
        .addTo(layerGroup.mask);
    }

    (det.polygons || []).forEach(function (f) {
      if (!f.geometry) return;
      var p = f.properties;
      L.polygon(ringToLatLng(f.geometry.coordinates[0]), {
        color: MAPC.oil, weight: 2, fillColor: MAPC.oil, fillOpacity: 0.24
      }).bindPopup(
        "<b>" + p.polygon_id + " mineral oil</b><br>" +
        "area " + fmt(p.area_km2, 3) + " km2<br>" +
        "length " + fmt(p.length_km, 2) + " km, width " + fmt(p.width_km, 2) + " km<br>" +
        "perimeter " + fmt(p.perimeter_km, 2) + " km<br>" +
        "orientation " + fmt(p.orientation_deg, 0) + " deg<br>" +
        (p.eo_verdict ? "optical: " + p.eo_verdict + "<br>" : "") +
        "compactness " + fmt(p.compactness, 2) + "<br>" +
        "contrast " + fmt(p.contrast_db, 1) + " dB<br>" +
        "confidence " + fmt(p.confidence, 2) + "<br>" +
        "centroid " + fmt(p.centroid_lat, 4) + ", " + fmt(p.centroid_lon, 4)
      ).addTo(layerGroup.oil);
    });

    (det.lookalikes || []).forEach(function (f) {
      if (!f.geometry) return;
      var p = f.properties;
      L.polygon(ringToLatLng(f.geometry.coordinates[0]), {
        color: MAPC.lookalike, weight: 1.5, dashArray: "5,4",
        fillColor: MAPC.lookalike, fillOpacity: 0.08
      }).bindPopup(
        "<b>" + p.polygon_id + " look-alike</b><br>" +
        "area " + fmt(p.area_km2, 3) + " km2<br>" +
        "contrast " + fmt(p.contrast_db, 1) + " dB<br>" +
        "<i>excluded from AIS attribution</i>"
      ).addTo(layerGroup.lookalike);
    });
  }

  function drawDrift(job) {
    var d = job.drift;
    if (!d) return;

    if (d.cone_back && d.cone_back.geometry) {
      L.polygon(ringToLatLng(d.cone_back.geometry.coordinates[0]), {
        color: MAPC.hindEdge, weight: 1, dashArray: "3,4",
        fillColor: MAPC.hindFill, fillOpacity: 0.10
      }).bindPopup("Hindcast cone, swept 90 percent ensemble<br>area " +
        fmt(d.cone_back.properties.area_km2, 1) + " km2").addTo(layerGroup.cone_back);
    }

    if (d.hindcast_track && d.hindcast_track.geometry) {
      L.polyline(ringToLatLng(d.hindcast_track.geometry.coordinates), {
        color: MAPC.hind, weight: 2, dashArray: "2,6", opacity: 0.9
      }).bindPopup("Ensemble median backtrack").addTo(layerGroup.hindcast);
    }

    if (d.origin_zone && d.origin_zone.geometry) {
      L.polygon(ringToLatLng(d.origin_zone.geometry.coordinates[0]), {
        color: MAPC.origin, weight: 2, fillColor: MAPC.origin, fillOpacity: 0.18
      }).bindPopup(
        "<b>Origin zone</b><br>" +
        utc(d.origin.t) + "<br>" +
        "90 percent envelope, spread " + fmt(d.origin.spread_km, 1) + " km<br>" +
        "buffered " + fmt(d.origin.buffer_km, 1) + " km<br>" +
        "area " + fmt(d.origin.area_km2, 1) + " km2"
      ).addTo(layerGroup.origin);

      L.circleMarker([d.origin.lat, d.origin.lon], {
        radius: 4, color: "#ffffff", weight: 2, fillColor: MAPC.origin, fillOpacity: 1
      }).bindTooltip("origin estimate, zone centre").addTo(layerGroup.origin);
    }

    if (d.cone_fwd && d.cone_fwd.geometry) {
      L.polygon(ringToLatLng(d.cone_fwd.geometry.coordinates[0]), {
        color: MAPC.fore, weight: 1.5, fillColor: MAPC.fore, fillOpacity: 0.10
      }).bindPopup("Forecast cone " + fmt(d.cone_fwd.properties.hours, 0) +
        " h<br>area " + fmt(d.cone_fwd.properties.area_km2, 1) + " km2")
        .addTo(layerGroup.cone_fwd);
    }
    if (d.forecast_track && d.forecast_track.geometry) {
      L.polyline(ringToLatLng(d.forecast_track.geometry.coordinates), {
        color: MAPC.fore, weight: 2, opacity: 0.95, dashArray: "6,4"
      }).bindPopup("Ensemble median forecast").addTo(layerGroup.forecast);
    }
  }

  /* The map's own palette, kept in one place so it cannot drift away from the
     stylesheet. Amber is the case under investigation, cyan is the forecast,
     and everything else is slate. */
  var MAPC = {
    oil: "#e89550",
    lookalike: "#b9a05e",
    hind: "#e0b98d",
    hindEdge: "#c9a37c",
    hindFill: "#8a6a4c",
    origin: "#e8734a",
    fore: "#4fb8dd",
    gap: "#cc5b4e"
  };

  /* Rank 1 is amber like the release zone it is being connected to. Ranks 2-3
     step down through it, and everything below that is slate: the leaderboard
     already carries the ordering, and ten distinct hues on a chart is noise. */
  var RANK_COLORS = ["#e8734a", "#e89550", "#c9a06a", "#8fa6b5", "#8fa6b5",
    "#8fa6b5", "#8fa6b5", "#8fa6b5", "#8fa6b5", "#8fa6b5"];

  function rankColor(rank) {
    return RANK_COLORS[Math.min(rank - 1, RANK_COLORS.length - 1)] || "#8fa6b5";
  }

  function drawTracks(suspects) {
    layerGroup.tracks.clearLayers();
    suspects.forEach(function (s) {
      var color = rankColor(s.rank);
      var weight = s.rank === 1 ? 4 : (s.rank <= 3 ? 2.5 : 1.6);
      var gj = (s.track || {}).geojson;
      if (!gj) return;
      gj.features.forEach(function (f) {
        if (!f.geometry || !f.geometry.coordinates.length) return;
        var dr = f.properties.kind === "dead_reckoned";
        L.polyline(ringToLatLng(f.geometry.coordinates), {
          color: dr ? MAPC.gap : color,
          weight: dr ? Math.max(weight, 3) : weight,
          opacity: dr ? 0.95 : 0.8,
          dashArray: dr ? "7,5" : null
        }).bindPopup(
          "<b>" + (s.name || "UNKNOWN") + "</b><br>" +
          "MMSI " + s.mmsi + "<br>" +
          "rank " + s.rank + ", score " + fmt(s.score, 1) + "<br>" +
          "type " + s.type + "<br>" +
          (dr ? "<b style='color:" + MAPC.gap + "'>NON-REPORTING segment, " +
            fmt(f.properties.gap_minutes, 0) + " min</b><br>" : "") +
          s.reasons.join("<br>")
        ).addTo(layerGroup.tracks);
      });
    });
  }

  // ----------------------------------------------------------- time slider
  function buildFrames(job) {
    var suspects = ((job.attribution || {}).suspects) || [];
    var times = {};
    suspects.forEach(function (s) {
      (s.track.samples || []).forEach(function (p) { times[p.ts] = true; });
    });
    var list = Object.keys(times).map(Number).sort(function (a, b) { return a - b; });
    // Thin to at most 260 frames so the slider stays responsive.
    var stride = Math.max(1, Math.ceil(list.length / 260));
    state.frames = list.filter(function (_, i) { return i % stride === 0; });
    state.frameIndex = state.frames.length ? state.frames.length - 1 : 0;

    var slider = $("slider");
    slider.min = 0;
    slider.max = Math.max(0, state.frames.length - 1);
    slider.value = state.frameIndex;
    // resetRun disables the scrubber, and nothing re-enabled it: the first run
    // of a session worked, and every run after picking a different scene left
    // the slider greyed out and inert. `disabled` is a property set here rather
    // than only cleared, so the scrubber's state is owned in exactly one place
    // and always matches whether there are frames to scrub.
    slider.disabled = state.frames.length < 2;
    $("timebar").classList.toggle("on", state.frames.length > 1);
    renderFrame();
  }

  function nearestSample(samples, ts) {
    if (!samples.length) return null;
    var lo = 0, hi = samples.length - 1;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (samples[mid].ts < ts) lo = mid + 1; else hi = mid;
    }
    var a = samples[Math.max(0, lo - 1)], b = samples[lo];
    if (Math.abs(a.ts - ts) > 20 * 60 && Math.abs(b.ts - ts) > 20 * 60) return null;
    return Math.abs(a.ts - ts) <= Math.abs(b.ts - ts) ? a : b;
  }

  function renderFrame() {
    var g = layerGroup.vessels;
    g.clearLayers();
    if (!state.frames.length) return;
    var ts = state.frames[state.frameIndex];
    $("tlabel").textContent = hhmm(ts);

    state.suspects.forEach(function (s) {
      var p = nearestSample(s.track.samples || [], ts);
      if (!p) return;
      var color = rankColor(s.rank);

      // Directional glyph. A plain dot tells an operator where a vessel was
      // but not which way it was going, and the trajectory match is the whole
      // argument for a candidate, so heading belongs on the map rather than in
      // a tooltip that has to be hovered. An arrow rotated to the course over
      // ground, sized by rank, with a dead-reckoned sample hollowed out and
      // dashed-edged because a position inferred across a reporting gap is a
      // different kind of fact from a received one.
      var r = s.rank === 1 ? 11 : (s.rank <= 3 ? 9 : 7);
      var icon = L.divIcon({
        className: "vessel-glyph" + (p.dr ? " dr" : "") +
          (s.rank === 1 ? " lead" : ""),
        html: arrowGlyph(r, p.cog, color, p.dr),
        iconSize: [r * 2.4, r * 2.4],
        iconAnchor: [r * 1.2, r * 1.2]
      });
      L.marker([p.lat, p.lon], { icon: icon, interactive: true })
        .bindTooltip(
          "#" + s.rank + " " + (s.name || "UNKNOWN") + "  " +
          fmt(p.sog, 1) + " kn  heading " + fmt(p.cog, 0) + "°" +
          (p.dr ? "  DEAD RECKONED" : ""),
          { direction: "top" }
        )
        .addTo(g);

      if (s.rank === 1) {
        L.marker([p.lat, p.lon], {
          icon: L.divIcon({
            className: "vessel-label",
            html: (s.name || "UNKNOWN"),
            iconAnchor: [-13, 7]
          }),
          interactive: false
        }).addTo(g);
      }
    });
  }

  /* An arrowhead pointing along the course, drawn in the rank colour.

     Built as inline SVG rather than as a rotated image so it stays crisp at
     any zoom, recolours with the rank, and can be hollowed for a
     dead-reckoned fix without a second asset. */
  function arrowGlyph(r, cog, color, hollow) {
    var d = "M" + (r * 2.1) + " " + r +
            " L" + (r * 0.55) + " " + (r * 0.28) +
            " L" + (r * 0.95) + " " + r +
            " L" + (r * 0.55) + " " + (r * 1.72) +
            " L" + (r * 0.2) + " " + (r * 1.1) +
            " L" + (r * 0.72) + " " + r + " Z";
    var fill = hollow ? "none" : color;
    return '<svg width="' + (r * 2.4) + '" height="' + (r * 2.4) + '" ' +
      'viewBox="0 0 ' + (r * 2.4) + ' ' + (r * 2.4) + '">' +
      '<g transform="rotate(' + (Number(cog) || 0) + " " + r + " " + r + ')">' +
      '<path d="' + d + '" fill="' + fill + '" stroke="' + color + '" ' +
      'stroke-width="' + (hollow ? 1.6 : 1) + '"' +
      (hollow ? ' stroke-dasharray="2.5 1.8"' : "") +
      ' stroke-linejoin="round" opacity="' + (hollow ? 0.95 : 1) + '"/>' +
      "</g></svg>";
  }

  function play() {
    if (state.playing) { stop(); return; }
    if (state.frames.length < 2) return;
    state.playing = true;
    $("play").textContent = "PAUSE";
    state.timer = setInterval(function () {
      state.frameIndex = (state.frameIndex + 1) % state.frames.length;
      $("slider").value = state.frameIndex;
      renderFrame();
    }, 90);
  }

  function stop() {
    state.playing = false;
    $("play").textContent = "PLAY";
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  // ------------------------------------------------------------- sidebar
  /* The three-minute demo cannot afford scrolling to find the answer. This is
     the whole result in one card at the top: what detected, what it found,
     where it came from, and who is top of the leaderboard. */
  function renderVerdict(job) {
    var card = $("verdict-card");
    var box = $("verdict");
    box.innerHTML = "";
    if (card) card.hidden = false;
    $("verdict-time").textContent = job.total_ms ? fmt(job.total_ms / 1000, 1) + " s" : "";

    var det = job.detection || {};
    var m = det.metrics || {};
    var polys = det.polygons || [];
    var drift = job.drift;
    var top = ((job.attribution || {}).suspects || [])[0];

    // An empty result is a finding, and there are two different empties. Say
    // which one this is: water with no structure in it at all, or water with
    // structure the detector declined to call oil. An operator acts on those
    // two differently, and "nothing found" on its own reads like a crash.
    if (!polys.length) {
      var clean = job.clean_scene || {};
      var rad = clean.radiometry || m.radiometry || {};
      var line = el("div", "verdict-line clean");
      line.appendChild(document.createTextNode(clean.headline || "No slick detected."));
      box.appendChild(line);

      var cf = el("div", "verdict-facts");
      factRow(cf, "Water level", rad.sea_level_db != null ? fmt(rad.sea_level_db, 1) + " dB" : "n/a");
      factRow(cf, "Contrast span", rad.dynamic_range_db != null ? fmt(rad.dynamic_range_db, 2) + " dB" : "n/a");
      factRow(cf, "Look-alikes", String(m.lookalike_polygons_found || 0));
      factRow(cf, "Detector", m.detector === "unet" ? "U-Net" : "dB baseline");
      box.appendChild(cf);

      box.appendChild(el("div", "hint", clean.detail ||
        "Drift and AIS did not run: there is no slick to trace back."));
      return;
    }

    var p = polys[0].properties;
    var head = el("div", "verdict-line");
    head.appendChild(document.createTextNode("Slick detected, "));
    head.appendChild(el("span", "qty", fmt(m.oil_area_km2, 2) + " km²"));
    box.appendChild(head);

    var facts = el("div", "verdict-facts");
    factRow(facts, "Largest slick", fmt(p.length_km, 1) + " km × " + fmt(p.area_km2, 2) + " km²");
    factRow(facts, "Origin", drift ? utc(drift.origin.t).replace(" UTC", "") : "n/a");
    factRow(facts, "Zone radius", drift ? "± " + fmt(drift.origin.spread_km, 1) + " km" : "n/a");
    factRow(facts, "Age", drift ? fmt(job.age_hours_proxy, 1) + " h drift proxy" : "n/a");
    // Four facts, not six. Coast impact lives in the Drift tab where the cone
    // it describes is; the detector is already named in the header. A finding
    // panel that repeats what is on screen elsewhere is height the leaderboard
    // could have used.
    box.appendChild(facts);

    // No lead block here. The leaderboard sits directly below and its first row
    // is the lead; printing it twice cost a panel's worth of height and told
    // the operator nothing the next panel did not.
    if (!top) {
      box.appendChild(el("div", "notice",
        "No vessel passed the spatio-temporal filter for this origin. " +
        "Reporting nothing rather than forcing a culprit."));
    }
  }

  /* One labelled cell in the finding grid. */
  function factRow(parent, key, value) {
    var cell = el("div", "fact");
    cell.appendChild(el("span", "fact-k", key));
    cell.appendChild(el("span", "fact-v", value));
    parent.appendChild(cell);
  }

  /* The metrics card the spec asks for: the model against the published dB
     threshold baseline, on the same tiles. Only shown when a checkpoint ran. */
  function renderModelMetrics(job) {
    var m = (job.detection || {}).metrics || {};
    if (m.detector !== "unet") return null;
    var mm = (m.detector_detail || {}).metrics || {};
    if (!mm || mm.iou_oil === undefined) return null;

    var box = el("div", "notice ok");
    box.appendChild(el("div", null,
      "Checkpoint metrics, measured on the Zenodo validation tiles:"));
    [
      ["IoU oil", mm.iou_oil],
      ["IoU look-alike", mm.iou_lookalike],
      ["pixel accuracy", mm.pixel_accuracy]
    ].forEach(function (r) {
      if (r[1] === undefined) return;
      var line = el("div", "kv");
      line.appendChild(el("span", "k", r[0]));
      line.appendChild(el("span", "v", fmt(r[1], 4)));
      box.appendChild(line);
    });
    return box;
  }

  // The optical cross-check, rendered as its own panel. It is corroboration
  // and never changes a detection, so it reads as a count of agreements rather
  // than as a score, and it always carries the time offset: Sentinel-2 did not
  // see this water when the radar did, and a disagreement across a 26 hour gap
  // is weak evidence about anything.
  //
  // This used to be a `eoSummary()` string that nothing called, built from the
  // old pre-status fields. A cross-check an investigator cannot see is not a
  // cross-check, so it is now a panel, and it reports the pipeline's own status
  // vocabulary rather than a bar count that reads as a verdict.
  function renderOptical(job) {
    var box = $("optical");
    if (!box) return;
    box.innerHTML = "";
    if (!job) {
      box.appendChild(el("p", "hint",
        "Sentinel-2 corroboration. Never an input to detection."));
      return;
    }
    var eo = (job.detection || {}).eo;
    if (!eo) {
      box.appendChild(el("p", "hint", "No optical step ran for this job."));
      return;
    }
    if (!eo.available) {
      box.appendChild(el("div", "notice",
        "Optical not used: " + (eo.reason || "no chip cached for this scene") + "."));
      return;
    }

    var c = eo.counts || {};
    var status = eo.status || "unavailable";
    var head = el("div", "optical-status " + (status === "corroborated" ? "ok" :
      (status === "contradicted" ? "bad" : "")));
    head.appendChild(el("b", null, String(status).replace(/_/g, " ")));
    box.appendChild(head);

    var delta = eo.time_delta || {};
    var rows = [
      ["Acquired", utc(eo.acquired)],
      ["Offset from radar", delta.hours != null ? fmt(delta.hours, 1) + " h" : "unknown"],
      ["Cloud free", eo.cloud_percent != null ? fmt(eo.cloud_percent, 1) + "%" : "n/a"],
      ["Valid pixels", delta.valid_pixel_fraction != null
        ? fmt(delta.valid_pixel_fraction * 100, 1) + "%" : "n/a"],
      ["Oil px agreeing", String(c.consistent != null ? c.consistent : 0)],
      ["Oil px disagreeing", String(c.inconsistent != null ? c.inconsistent : 0)],
      ["Oil px neutral", String(c.neutral != null ? c.neutral : 0)],
      ["Obscured", String(c.obscured != null ? c.obscured : 0)]
    ];
    rows.forEach(function (row) {
      var line = el("div", "kv");
      line.appendChild(el("span", "k", row[0]));
      line.appendChild(el("span", "v", row[1]));
      box.appendChild(line);
    });

    // The whole point of the panel: state the limit, not just the result.
    box.appendChild(el("div", "hint",
      "Different sensor, different time, and over open ocean usually " +
      "different cloud. This is context for reading the map and never a " +
      "reason to accept or reject the radar detection."));
  }

  /* Detector provenance. A slick detection is only worth as much as the thing
     that drew the polygon, so the checkpoint, its threshold and any silent
     fallback are shown next to the result rather than buried in a log. */
  function renderDetectorMeta(job) {
    var box = $("detector-meta");
    if (!box) return;
    box.innerHTML = "";
    var meta = (((job && job.detection) || {}).metrics || {}).detector_metadata;
    if (!meta) {
      box.appendChild(el("p", "hint", "Detector metadata appears after a run."));
      return;
    }
    if (meta.is_trained_detector === false) {
      box.appendChild(el("div", "notice bad",
        "PHYSICS BASELINE, NOT THE TRAINED DETECTOR. " +
        (meta.fallback_reason || "no checkpoint available") + "."));
    }
    [
      ["Detector", meta.name || "unknown"],
      ["Trained", meta.is_trained_detector ? "yes" : "no"],
      ["Checkpoint", meta.checkpoint || "none"],
      ["Threshold", meta.threshold != null ? fmt(meta.threshold, 3) : "n/a"],
      ["Input", meta.input_shape ? (meta.input_shape.join(" x ")) : "n/a"],
      ["Tiled", meta.tiled ? "yes" : "no"],
      ["Benchmark", meta.benchmark_status || "not run"]
    ].forEach(function (row) {
      var line = el("div", "kv");
      line.appendChild(el("span", "k", row[0]));
      line.appendChild(el("span", "v", row[1]));
      box.appendChild(line);
    });
  }

  function renderDetection(job) {
    var det = job.detection || {};
    var m = det.metrics || {};
    var polys = det.polygons || [];
    var box = $("detection");
    box.innerHTML = "";
    $("det-method").textContent = m.detector === "unet" ? "U-Net" : "dB baseline";

    if (!polys.length) {
      box.appendChild(el("div", "notice", "No oil polygon above the area threshold. " +
        "Reporting an empty result rather than forcing one. Look-alike polygons found: " +
        (det.lookalikes || []).length + "."));
    }

    var p = polys.length ? polys[0].properties : null;
    var rows = [
      ["Oil polygons", polys.length],
      ["Look-alikes", (det.lookalikes || []).length],
      ["Total oil area", fmt(m.oil_area_km2, 3) + " km2"],
      ["Largest area", p ? fmt(p.area_km2, 3) + " km2" : "n/a"],
      ["Length", p ? fmt(p.length_km, 2) + " km" : "n/a"],
      ["Width", p ? fmt(p.width_km, 2) + " km" : "n/a"],
      ["Perimeter", p ? fmt(p.perimeter_km, 2) + " km" : "n/a"],
      ["Orientation", p ? fmt(p.orientation_deg, 0) + " deg" : "n/a"],
      ["Compactness", p ? fmt(p.compactness, 2) : "n/a"],
      ["Contrast", p ? fmt(p.contrast_db, 1) + " dB" : "n/a"],
      ["Confidence", p ? fmt(p.confidence, 2) : "n/a"],
      ["Centroid", p ? fmt(p.centroid_lat, 4) + ", " + fmt(p.centroid_lon, 4) : "n/a"]
    ];
    rows.forEach(function (r) {
      var line = el("div", "kv");
      line.appendChild(el("span", "k", r[0]));
      line.appendChild(el("span", "v", r[1]));
      box.appendChild(line);
    });

    var mm = renderModelMetrics(job);
    if (mm) box.appendChild(mm);

    if (m.accuracy_vs_truth) {
      var a = m.accuracy_vs_truth;
      var acc = el("div", "notice ok",
        "Against the ground truth mask: IoU oil " + fmt(a.iou_oil, 3) +
        ", IoU look-alike " + fmt(a.iou_lookalike, 3) +
        ", pixel accuracy " + fmt(a.pixel_accuracy, 3));
      box.appendChild(acc);
    }

    if (m.detector !== "unet") {
      box.appendChild(el("div", "notice",
        "Running the published " + fmt(m.detector_detail && m.detector_detail.threshold_db, 1) +
        " dB dark patch baseline. " + (m.fallback_reason || "")));
    }
  }

  function renderOrigin(job) {
    var box = $("origin");
    box.innerHTML = "";
    var d = job.drift;
    if (!d) {
      box.appendChild(el("div", "hint", "Drift was not run for this job."));
      return;
    }
    var tl = driftTimeline(job);
    if (tl) box.appendChild(tl);
    var rows = [
      ["Origin time", utc(d.origin.t)],
      ["Origin position", fmt(d.origin.lat, 4) + ", " + fmt(d.origin.lon, 4)],
      ["Zone radius", fmt(d.origin.spread_km, 1) + " km, buffered " + fmt(d.origin.buffer_km, 1) + " km"],
      ["Zone area", fmt(d.origin.area_km2, 1) + " km2"],
      ["Hours back", fmt(d.origin.index_hours_back, 0) + " h"],
      ["Age proxy", fmt(job.age_hours_proxy, 1) + " h"],
      ["Wind factor", fmt(d.physics.alpha_wind, 3) + " of 10 m wind"],
      ["Deflection", fmt(d.physics.deflection_deg, 0) + " deg right"],
      ["Particles", d.physics.n_particles],
      ["Forecast spread", fmt(d.forecast_hourly[d.forecast_hourly.length - 1].spread_km, 1) + " km"]
    ];
    rows.forEach(function (r) {
      var line = el("div", "kv");
      line.appendChild(el("span", "k", r[0]));
      line.appendChild(el("span", "v", r[1]));
      box.appendChild(line);
    });

    var coast = d.coast || {};
    var coastLine = el("div", "kv");
    coastLine.appendChild(el("span", "k", "Coast impact"));
    coastLine.appendChild(el("span", "v",
      coast.available === false ? "no land mask"
        : (coast.coast_flag ? "REACHES LAND" : "stays offshore")));
    if (coast.coast_flag) coastLine.querySelector(".v").style.color = "var(--warn)";
    box.appendChild(coastLine);

    var bb = d.threatened_bbox;
    if (bb) {
      var bbLine = el("div", "kv");
      bbLine.appendChild(el("span", "k", "Threatened box"));
      bbLine.appendChild(el("span", "v",
        fmt(bb.south, 2) + " to " + fmt(bb.north, 2) + " N, " +
        fmt(bb.west, 2) + " to " + fmt(bb.east, 2) + " E"));
      box.appendChild(bbLine);
    }

    box.appendChild(el("div", "hint",
      "Age is the drift time from the estimated origin to the acquisition. " +
      "It is a physical proxy, not a laboratory weathering age."));

    var mo = d.metocean || {};
    var moText = (mo.synthetic ? "NO CACHED METOCEAN. " : "Metocean: ") + (mo.source || "");
    if (mo.t_start) {
      moText += "  " + mo.t_start.substring(0, 16) + " to " + mo.t_end.substring(0, 16);
    }
    if (mo.has_currents === false && !mo.synthetic) {
      moText += "  |  wind only: no current data for this basin";
    } else if (mo.mean_current_ms !== undefined) {
      moText += "  |  mean current " + fmt(mo.mean_current_ms, 2) + " m/s, wind " +
        fmt(mo.mean_wind_ms, 1) + " m/s";
    }
    box.appendChild(el("div",
      mo.synthetic ? "notice bad" : (mo.has_currents === false ? "notice" : "notice ok"),
      moText));
  }

  function renderSuspects(job) {
    var wrap = $("suspects");
    wrap.innerHTML = "";
    var attr = job.attribution || {};
    var list = attr.suspects || [];
    state.suspects = list;
    $("susp-count").textContent = list.length ? list.length + " ranked" : "";

    // Where the candidate tracks came from. The scorer is blind to this by
    // design; the operator must not be.
    var used = attr.sources_used || (attr.funnel || {}).sources_used;
    if (used && Object.keys(used).length) {
      var parts = Object.keys(used).map(function (k) {
        return used[k] + " " + (k.indexOf("simulated") >= 0 ? "simulated" : k.replace(/_/g, " "));
      });
      var simulated = Object.keys(used).some(function (k) { return k.indexOf("simulated") >= 0; });
      var src = el("div", "body");
      src.appendChild(el("div", simulated ? "notice" : "notice ok",
        "AIS source for the candidates: " + parts.join(", ") + "." +
        (simulated ? " Simulated traffic over this scene's real geobox and time window."
                   : " Real recorded tracks.")));
      wrap.appendChild(src);
    }

    if (!list.length) {
      var b = el("div", "body");
      b.appendChild(el("div", "notice",
        "No vessel passed the spatio-temporal filter for this origin zone and window. " +
        "The pipeline reports nothing rather than inventing a culprit."));
      if (attr.funnel) {
        b.appendChild(el("div", "hint",
          "Considered " + attr.funnel.considered_vessels + " vessels in the box, " +
          "dropped " + attr.funnel.dropped_outside_radius + " outside the radius."));
      }
      wrap.appendChild(b);
      return;
    }

    list.forEach(function (s) { wrap.appendChild(suspectCard(s)); });
  }

  /* One ranked vessel, rendered the same way wherever it appears. */
  function suspectCard(s) {
    {
      var card = el("div", "suspect" + (s.rank === 1 ? " r1" : ""));
      card.dataset.mmsi = s.mmsi;

      var top = el("div", "top");
      top.appendChild(el("span", "rank", "#" + s.rank));
      top.appendChild(el("span", "nm", s.name || "UNKNOWN"));
      top.appendChild(el("span", "sc", fmt(s.score, 1) + "%"));
      card.appendChild(top);

      var d = s.detail || {};
      var dist = d.origin_distance_km != null
        ? fmt(d.origin_distance_km, 1) + " km"
        : fmt(d.min_distance_km, 1) + " km";
      var when = d.time_offset_minutes != null
        ? Math.abs(Math.round(d.time_offset_minutes)) + " min " +
          (d.time_offset_minutes < 0 ? "before" : "after")
        : null;
      card.appendChild(el("div", "meta",
        "MMSI " + s.mmsi + " · " + s.type.replace(/_/g, " ") +
        " · " + dist + (when ? " · " + when : "")));

      // One bar, two numbers. Solid to the ranked score; a hairline continues to
      // where the evidence alone would have put it. The gap is how much of the
      // case rests on positions that were dead reckoned rather than received.
      var evidence = s.score_before_confidence != null ? s.score_before_confidence : s.score;
      var bar = el("div", "bar");
      bar.style.setProperty("--score", Math.max(1, Math.min(100, s.score)) + "%");
      bar.style.setProperty("--evidence", Math.max(1, Math.min(100, evidence)) + "%");
      if (evidence - s.score < 0.5) bar.dataset.full = "1";
      bar.appendChild(el("i"));
      bar.appendChild(el("u"));
      bar.title = "Ranked " + fmt(s.score, 1) + "% · evidence alone " +
                  fmt(evidence, 1) + "% · track confidence " +
                  fmt((s.confidence != null ? s.confidence : 1) * 100, 0) + "%";
      card.appendChild(bar);

      // Only annotate a discount worth reading. A 3 percent haircut is visible
      // in the bar's ghost tail and does not need a sentence as well.
      if (s.confidence != null && s.confidence < 0.95) {
        var cn = el("div", "conf-note");
        cn.appendChild(document.createTextNode(fmt(evidence, 1) + "% on evidence, held to "));
        cn.appendChild(el("b", null, fmt(s.score, 1) + "%"));
        cn.appendChild(document.createTextNode(
          " (" + Math.round((d.dead_reckoned_fraction || 0) * 100) +
          "% of this track is dead reckoned from " + (d.raw_positions || 0) + " receptions)"));
        card.appendChild(cn);
      }

      var tags = el("div", "tags");
      s.reasons.forEach(function (r) {
        var cls = "tag";
        if (r.indexOf("ais_gap") === 0) cls += " gap";
        if (r.indexOf("type_") === 0) cls += " type";
        tags.appendChild(el("span", cls, r.replace(/_/g, " ")));
      });
      card.appendChild(tags);

      var det = el("div", "detail small");
      var c = s.components, w = s.weighted;
      [
        ["proximity", c.prox, w.prox],
        ["vessel type", c.type, w.type],
        ["trajectory", c.traj, w.traj],
        ["behaviour", c.beh, w.beh]
      ].forEach(function (row) {
        var line = el("div", "kv");
        line.appendChild(el("span", "k", row[0]));
        line.appendChild(el("span", "v", fmt(row[1], 2) + "  ->  " + fmt(row[2], 3)));
        det.appendChild(line);
      });
      [
        ["closest approach", utc(s.detail.closest_approach_utc)],
        ["at", fmt(s.detail.closest_lat, 4) + ", " + fmt(s.detail.closest_lon, 4)],
        ["course vs drift", fmt(s.detail.trajectory.course_deg, 0) + " vs " +
          fmt(s.detail.trajectory.drift_bearing_deg, 0) + " deg"],
        ["SOG at closest", fmt(s.detail.behavior.sog_at_closest_kn, 1) + " kn"],
        ["reported fixes", s.detail.raw_positions],
        ["gaps", s.detail.gaps.length]
      ].forEach(function (row) {
        var line = el("div", "kv");
        line.appendChild(el("span", "k", row[0]));
        line.appendChild(el("span", "v", row[1]));
        det.appendChild(line);
      });
      if (s.detail.behavior.non_reporting) {
        det.appendChild(el("div", "notice bad",
          "NON-REPORTING: " + fmt(s.detail.behavior.ais_gap_minutes, 0) +
          " min gap whose dead reckoned segment passes " +
          fmt(s.detail.behavior.ais_gap_min_distance_km, 1) + " km from the origin zone."));
      }
      card.appendChild(det);

      card.addEventListener("click", function () { selectSuspect(s.mmsi); });
      return card;
    }
  }

  function selectSuspect(mmsi) {
    state.selected = state.selected === mmsi ? null : mmsi;
    Array.prototype.forEach.call(document.querySelectorAll(".suspect"), function (n) {
      n.classList.toggle("sel", String(n.dataset.mmsi) === String(state.selected));
    });
    // Re-render either way. Deselecting must not leave the last vessel's
    // counter-evidence on screen attached to nothing, so the panel falls back
    // to rank 1 rather than to the vessel that was just dismissed.
    renderWhy(state.job);
    if (!state.selected) return;
    var s = state.suspects.filter(function (x) { return String(x.mmsi) === String(mmsi); })[0];
    if (!s) return;
    var pts = (s.track.samples || []).map(function (p) { return [p.lat, p.lon]; });
    if (pts.length) map.fitBounds(L.latLngBounds(pts).pad(0.25));
  }

  function renderTrace(job) {
    var box = $("trace");
    box.innerHTML = "";
    $("trace-total").textContent = job.total_ms ? fmt(job.total_ms / 1000, 1) + " s" : "";
    // Bars are scaled to the slowest step, so the shape of the run is legible:
    // where the seconds actually went, not just that they went.
    var slowest = (job.trace || []).reduce(function (a, st) {
      return Math.max(a, st.elapsed_ms || 0);
    }, 0);
    (job.trace || []).forEach(function (st) {
      var row = el("div", "step " + (st.status === "warn" ? "warn" : (st.status === "info" ? "info" : "")));
      row.appendChild(el("span", "n", st.step));
      var bits = [];
      Object.keys(st).forEach(function (k) {
        if (["step", "started_ms", "elapsed_ms", "status", "note"].indexOf(k) >= 0) return;
        bits.push(k + "=" + st[k]);
      });
      var d = el("span", "d", st.note || bits.join("  "));
      d.style.setProperty("--frac", (slowest ? (st.elapsed_ms || 0) / slowest * 100 : 0) + "%");
      d.title = (st.note ? st.note + "  " : "") + bits.join("  ");
      row.appendChild(d);
      row.appendChild(el("span", "ms", fmt(st.elapsed_ms, 0) + " ms"));
      box.appendChild(row);
    });
  }

  function renderNotices(job) {
    var box = $("notices");
    box.innerHTML = "";
    (job.warnings || []).forEach(function (w) {
      var cls = "notice";
      if (w.indexOf("NOT satisfied") >= 0 || w.indexOf("No cached metocean") >= 0) cls += " bad";
      box.appendChild(el("div", cls, w));
    });
  }

  function renderLayers() {
    var box = $("layers");
    box.innerHTML = "";
    // Each row carries the colour it controls, so the layer list is also the
    // map legend and there is only one of them to read.
    var defs = [
      ["optical", "Optical (S2)", false, null],
      ["sar", "SAR image", true, null],
      ["mask", "Class mask", true, null],
      ["oil", "Oil polygons", true, "var(--oil)"],
      ["lookalike", "Look-alikes", true, "dash:var(--lookalike)"],
      ["hindcast", "Backtrack", true, "dash:var(--hind)"],
      ["cone_back", "Hindcast cone", true, "var(--hind)"],
      ["origin", "Release zone", true, "var(--accent)"],
      ["forecast", "Forecast track", true, "dash:var(--fore)"],
      ["cone_fwd", "Forecast cone", true, "var(--fore)"],
      ["tracks", "AIS tracks", true, null],
      ["vessels", "Vessel marks", true, "var(--rank1)"]
    ];
    defs.forEach(function (d) {
      var lbl = el("label");
      var cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = d[2];
      cb.addEventListener("change", function () {
        if (cb.checked) map.addLayer(layerGroup[d[0]]);
        else map.removeLayer(layerGroup[d[0]]);
      });
      lbl.appendChild(cb);
      lbl.appendChild(document.createTextNode(d[1]));
      if (d[3]) {
        var dashed = d[3].indexOf("dash:") === 0;
        var col = dashed ? d[3].slice(5) : d[3];
        var key = el("span", "key" + (dashed ? " dash" : ""));
        if (dashed) key.style.color = col; else key.style.background = col;
        lbl.appendChild(key);
      }
      box.appendChild(lbl);
    });
  }

  function renderWeights(scoring) {
    var box = $("weights");
    box.innerHTML = "";
    var w = scoring.weights;
    Object.keys(w).forEach(function (k) {
      var line = el("div", "kv");
      line.appendChild(el("span", "k", k));
      line.appendChild(el("span", "v", fmt(w[k], 2)));
      box.appendChild(line);
    });
    box.appendChild(el("div", "hint", scoring.formula));
    // Every rule the server publishes is rendered, so adding one on the server
    // never leaves the sidebar quietly out of date.
    [["proximity", scoring.proximity],
     ["time", scoring.time],
     ["trajectory", scoring.trajectory],
     ["AIS gap", (scoring.behavior || {}).ais_gap],
     ["sparse gap", (scoring.behavior || {}).sparse_gap],
     ["confidence", scoring.confidence]].forEach(function (row) {
      if (!row[1]) return;
      box.appendChild(el("div", "hint", row[0] + ": " + row[1]));
    });
    box.appendChild(el("div", "notice ok", scoring.note));
  }

  // ------------------------------------------------ investigation panels
  /* Everything below renders one claim of the case and the limit on it.

     The panels are deliberately separate rather than one long "results"
     column, because a result and its caveat travel together. An investigator
     reading a rank needs the counter-evidence, the envelope containment and
     the quality band in the same viewport as the number, and the only way to
     guarantee that is to give each of them a fixed place. */

  /* Slick shape, which is what a trained eye actually uses first: a
     weathering slick is long and thin and loosely convoluted, and a
     biogenic or low-wind look-alike usually is not. Reporting elongation and
     boundary roughness turns "it looks like oil" into something checkable. */
  function renderShape(job) {
    var box = $("shape");
    if (!box) return;
    box.innerHTML = "";
    var p = job && (((job.detection || {}).polygons || [])[0] || {}).properties;
    var sh = p && p.shape_diagnostics;
    if (!sh) {
      box.appendChild(el("p", "hint", "Shape diagnostics appear after a run."));
      return;
    }
    var rows = [
      ["Elongation", fmt(sh.elongation, 2)],
      ["Solidity", fmt(sh.solidity, 3)],
      ["Irregularity", fmt(sh.irregularity, 3)],
      ["Boundary roughness", fmt(sh.boundary_roughness, 3)],
      ["Components", String(sh.n_components != null ? sh.n_components : "n/a")],
      ["Boundary complexity", fmt(p.boundary_complexity, 3)]
    ];
    rows.forEach(function (row) {
      var line = el("div", "kv");
      line.appendChild(el("span", "k", row[0]));
      line.appendChild(el("span", "v", row[1]));
      box.appendChild(line);
    });
    box.appendChild(el("div", "hint",
      "Elongation above about 3 and a solidity below about 0.4 is the usual " +
      "signature of a stretched weathering slick. These describe the polygon, " +
      "not the substance in it."));
  }

  /* The uncertainty chain, stage by stage. Each stage names what is uncertain
     and by how much, so "how sure is this case" has an answer that is not the
     candidate's score.

     `quantity` is certainty, not uncertainty: 1.0 means this stage's inputs
     are in good order and 0.0 means the stage could not be run. The bar is
     drawn that way round deliberately. A bar labelled "uncertainty" that filled
     up on a well-run stage would read backwards, and the level word beside it
     is what settles it. */
  function renderUncertainty(job) {
    var box = $("uncertainty");
    if (!box) return;
    box.innerHTML = "";
    var q = (job && job.case_quality) || {};
    var chain = q.uncertainty_chain;
    if (!chain || !chain.stages) {
      box.appendChild(el("p", "hint", "The uncertainty chain appears after a run."));
      return;
    }
    chain.stages.forEach(function (st) {
      var row = el("div", "ustage " + (st.level || "unknown").toLowerCase());
      var head = el("div", "uh");
      head.appendChild(el("span", "ul", st.label || st.stage));
      head.appendChild(el("span", "uv badge", (st.level || "unknown")));
      row.appendChild(head);

      if (st.quantity != null) {
        var bar = el("div", "ubar");
        bar.style.setProperty("--u", Math.max(0, Math.min(1, Number(st.quantity) || 0)) * 100 + "%");
        bar.title = "Stage certainty " + fmt(st.quantity, 2) + " of 1.0. " +
          "This is how well the inputs to this stage are in order, not a " +
          "probability that the case is correct.";
        row.appendChild(bar);
        row.appendChild(el("div", "uv-note", "stage certainty " + fmt(st.quantity, 2) + " of 1.0"));
      }
      var m = st.measurement || {};
      Object.keys(m).forEach(function (k) {
        if (m[k] === null || m[k] === undefined) return;
        row.appendChild(el("div", "uv-note", k.replace(/_/g, " ") + ": " + fmt(m[k], 2)));
      });
      if (st.note) row.appendChild(el("div", "uv-note", st.note));
      box.appendChild(row);
    });
    box.appendChild(el("div", "hint",
      "Case quality " + (q.overall || "unrated") + " is the weakest material " +
      "input, not an average and not a probability."));
  }

  /* The safe-fail state, and the conclusions the case is allowed to make.

     An empty result and a withheld conclusion are findings, and the pipeline
     distinguishes them deliberately: "no oil here" and "oil here, but no
     vessel is a defensible candidate" are different answers and an operator
     acts on them differently. */
  function renderSafeFail(job) {
    var box = $("safefail");
    if (!box) return;
    box.innerHTML = "";
    if (!job) {
      box.appendChild(el("p", "hint", "No case open."));
      return;
    }
    var q = job.case_quality || {};
    var sf = q.safe_fail || {};

    var head = el("div", "sf-state " + (sf.state || "none").replace(/\s+/g, "-").toLowerCase());
    head.appendChild(el("span", "sf-label", "Case state"));
    head.appendChild(el("b", null, sf.state || "REPORTED"));
    box.appendChild(head);
    if (sf.detail) box.appendChild(el("p", "hint", sf.detail));

    if (sf.reason) box.appendChild(el("div", "notice", sf.reason));

    var band = el("div", "sf-band " + (q.overall || "").toLowerCase());
    band.appendChild(el("span", "sf-label", "Evidence quality"));
    band.appendChild(el("b", null, (q.overall || "UNRATED") +
      (q.label ? " · " + q.label : "")));
    box.appendChild(band);

    (q.conclusions || []).forEach(function (c) {
      box.appendChild(el("div", "conclusion", c));
    });

    // `ordered_factors` is the weakest-first list of stage keys and `factors`
    // is the same information keyed by stage, carrying the level and the
    // measurement. Reading the order alone gave a column of bare internal keys
    // ("sar_detection", "origin_certainty") with no level beside them, which
    // is the one thing this panel exists to show.
    var ordered = q.ordered_factors || [];
    var factors = q.factors || {};
    if (ordered.length) {
      var w = el("div", "weakest");
      w.appendChild(el("div", "sf-label", "Weakest material inputs, in order"));
      ordered.slice(0, 5).forEach(function (key) {
        var f = factors[key] || {};
        var item = el("div", "witem " + String(f.level || "").toLowerCase());
        item.appendChild(el("span", "wn", f.label || f.stage || String(key)));
        var detail = f.level || "";
        if (f.quantity != null) detail += " · " + fmt(f.quantity, 2);
        item.appendChild(el("span", "wv", detail));
        if (f.note) item.title = f.note;
        w.appendChild(item);
      });
      box.appendChild(w);
    }
  }

  /* Why this vessel, and why not.

     The two halves are separated rather than interleaved because the second
     one is the one that stops a lead being read as a finding. A ranked list
     that puts "counter-evidence" under everything else is a ranked list that
     gets skimmed. Here the objections sit directly under the reason the vessel
     is on the list, at the same visual weight. */
  function renderWhy(job) {
    var box = $("why");
    if (!box) return;
    box.innerHTML = "";
    var list = ((job && job.attribution) || {}).suspects || [];
    if (!list.length) {
      box.appendChild(el("p", "hint", "No candidate to argue about."));
      return;
    }
    var s = list.filter(function (x) { return x.mmsi === state.selected; })[0] || list[0];
    if (!s) return;

    var ev = s.evidence || {};
    var head = el("div", "why-head");
    head.appendChild(el("span", "rank", "#" + s.rank));
    head.appendChild(el("span", "nm", s.name || "UNKNOWN"));
    head.appendChild(el("span", "sc", fmt(s.score, 1) + "%"));
    box.appendChild(head);

    // Opportunity and evidence are separate numbers on purpose. "Was nearby"
    // and "the evidence lines up" are different claims, and collapsing them
    // into one score is how a vessel that merely passed through gets
    // described as a source.
    var split = el("div", "split");
    [
      ["Opportunity", ev.opportunity_level, ev.opportunity_score,
       "Where it was and when. Says nothing about evidence."],
      ["Evidence", ev.evidence_level, ev.evidence_score,
       "How well the observation matches the modelled slick."]
    ].forEach(function (r) {
      var cell = el("div", "cell " + String(r[1] || "").toLowerCase());
      cell.appendChild(el("div", "cl", r[0]));
      cell.appendChild(el("div", "cv", r[1] || "UNRATED"));
      cell.appendChild(el("div", "cs", fmt(r[2], 1) + "%"));
      cell.appendChild(el("div", "cn", r[3]));
      split.appendChild(cell);
    });
    box.appendChild(split);

    var region = s.region || {};
    var win = s.release_window || {};
    var geom = el("div", "geom");
    [
      ["Inside 50% envelope", region.inside_50 != null ? String(region.inside_50) : "n/a"],
      ["Inside 90% envelope", region.inside_90 != null ? String(region.inside_90) : "n/a"],
      ["Observed track in zone", region.observed_track_fraction != null
        ? fmt(region.observed_track_fraction * 100, 0) + "%" : "n/a"],
      ["Release window", win.applicable === false ? "not applicable"
        : (win.inside != null ? String(win.inside) : "n/a")]
    ].forEach(function (row) {
      var line = el("div", "kv");
      line.appendChild(el("span", "k", row[0]));
      line.appendChild(el("span", "v", row[1]));
      geom.appendChild(line);
    });
    box.appendChild(geom);

    var pos = el("div", "why-col");
    pos.appendChild(el("div", "col-h good", "Why this vessel"));
    var positives = (ev.positive_evidence || ev.reasons || []);
    if (!positives.length) {
      pos.appendChild(el("p", "hint", "No supporting reason recorded."));
    } else {
      positives.forEach(function (r) {
        var it = el("div", "ev-item pos");
        it.appendChild(el("span", "cat", r.category || "opportunity"));
        it.appendChild(el("span", "txt", r.text || String(r)));
        pos.appendChild(it);
      });
    }
    box.appendChild(pos);

    var neg = el("div", "why-col");
    neg.appendChild(el("div", "col-h bad",
      "Why not · " + (s.counter_evidence_count || 0) +
      " objection" + (s.counter_evidence_count === 1 ? "" : "s")));
    var counter = s.counter_evidence || (ev.counter_evidence) || [];
    if (!counter.length) {
      neg.appendChild(el("p", "hint",
        "No counter-evidence recorded. That is the absence of a recorded " +
        "objection, not evidence that none exists."));
    } else {
      counter.forEach(function (r) {
        var it = el("div", "ev-item neg");
        it.appendChild(el("span", "cat", r.category || "counter_evidence"));
        it.appendChild(el("span", "txt", r.text || String(r)));
        neg.appendChild(it);
      });
    }
    box.appendChild(neg);

    var dq = ev.data_quality || [];
    var uq = ev.uncertainty || [];
    if (dq.length || uq.length) {
      var meta = el("div", "why-col");
      meta.appendChild(el("div", "col-h", "Track data quality"));
      dq.concat(uq).forEach(function (r) {
        var it = el("div", "ev-item meta-item");
        it.appendChild(el("span", "cat", r.kind || r.category || "data"));
        it.appendChild(el("span", "txt", r.text || String(r)));
        meta.appendChild(it);
      });
      box.appendChild(meta);
    }

    box.appendChild(el("div", "notice",
      "A lead for investigation. Not a finding of discharge, and not a " +
      "measurement of blame."));
  }

  /* Counterfactual sensitivity: which assumptions carry the ranking.

     The study is per candidate. For each one it removes or widens one declared
     assumption, re-ranks, and reports whether the vessel held its place. The
     point is not the new score, it is the stability label: a top candidate
     that only holds when the origin is exactly where the model put it is a
     weaker lead than one that holds across the window, and an operator can only
     see the difference if both are on the page.

     Every scenario is listed, including the ones that could not be evaluated.
     A scenario that is quietly dropped reads as "this assumption does not
     matter", which is the opposite of what a missing input means. */
  function renderSensitivity(job) {
    var box = $("sensitivity");
    if (!box) return;
    box.innerHTML = "";
    var rows = ((job && job.attribution) || {}).sensitivity || [];
    if (!rows.length) {
      box.appendChild(el("p", "hint", job
        ? "No counterfactual study was recorded for this case."
        : "Counterfactual scenarios appear after a run."));
      return;
    }

    rows.forEach(function (r) {
      var card = el("div", "scen");
      var h = el("div", "sh");
      h.appendChild(el("span", "sl", r.candidate_name || ("MMSI " + r.candidate_id)));
      var stability = r.stability || "NOT_EVALUATED";
      var badge = el("span", "badge " + (stability === "STABLE" ? "hold"
        : (stability === "NOT_EVALUATED" ? "na" : "flip")));
      badge.textContent = stability.replace(/_/g, " ");
      badge.title = stability === "STABLE"
        ? "The ranking held under every scenario that could be evaluated."
        : (stability === "UNSTABLE" ? "The rank changed under at least one scenario."
        : (stability === "SENSITIVE" ? "The score moved more than 15 points under at least one scenario."
        : "No scenario could be evaluated, so nothing is claimed either way."));
      h.appendChild(badge);
      card.appendChild(h);

      var sub = el("div", "uv-note",
        "baseline " + fmt(r.baseline_score, 1) + "% at rank " + r.baseline_rank +
        " · largest move " + fmt(r.max_score_delta, 1) + " points" +
        (r.primary_dependency ? " under " + String(r.primary_dependency).replace(/_/g, " ") : ""));
      card.appendChild(sub);

      var scen = el("div", "scenarios");
      (r.scenarios || []).forEach(function (s) {
        var line = el("div", "srow" + (s.applicable === false ? " na" : ""));
        if (s.applicable === false) {
          line.appendChild(el("span", "sn", s.label || s.name));
          line.appendChild(el("span", "sv2", "NOT EVALUATED"));
        } else {
          line.appendChild(el("span", "sn", s.label || s.name));
          line.appendChild(el("span", "sv2", fmt(s.score, 1) + "% at rank " + s.rank +
            (s.delta ? " (" + (s.delta > 0 ? "+" : "") + fmt(s.delta, 1) + ")" : "")));
        }
        if (s.question) line.title = s.question;
        if (s.reason) line.title = s.reason;
        scen.appendChild(line);
      });
      card.appendChild(scen);

      if (r.note) card.appendChild(el("div", "uv-note", r.note));
      box.appendChild(card);
    });

    box.appendChild(el("div", "hint",
      "Scenarios create no new evidence and are not probabilities. They answer " +
      "one question only: would a reviewer who disagreed with one of our stated " +
      "assumptions rank this differently?"));
  }

  /* Ablation: which stage of the pipeline is carrying the result.

     Rungs that could not be evaluated are listed as such. A row that is
     silently absent reads as "this stage did not matter", which is the exact
     opposite of what not having the data means. */
  function renderAblation(job) {
    var box = $("ablation");
    if (!box) return;
    box.innerHTML = "";
    var study = (job && job.ablation) || null;
    if (!study) {
      box.appendChild(el("p", "hint",
        "The ablation study appears after a run. It re-scores the same case " +
        "with one stage of the pipeline removed at a time."));
      return;
    }
    if (study.available === false) {
      box.appendChild(el("div", "notice bad",
        "Ablation not available: " + (study.reason || "no candidate set to re-rank.")));
      return;
    }
    var ladder = study.ladder || [];
    if (!ladder.length) {
      box.appendChild(el("p", "hint", "No ablation rungs were produced for this case."));
      return;
    }
    ladder.forEach(function (r) {
      var ok = r.status === "computed";
      var row = el("div", "rung" + (ok ? "" : " na"));
      row.appendChild(el("span", "rid", r.step || r.key));
      var name = el("span", "rname", r.label || r.key);
      name.title = r.description || r.question || "";
      row.appendChild(name);
      if (!ok) {
        row.appendChild(el("span", "rv na", "NOT EVALUATED"));
        row.title = r.reason || "This rung could not be evaluated for this case.";
      } else {
        var bits = [];
        var agree = r.agreement || {};
        if (agree.top1_match != null) {
          bits.push(agree.top1_match ? "top 1 held" : "top 1 changed");
        }
        if (r.spearman_vs_full != null) bits.push("rho " + fmt(r.spearman_vs_full, 3));
        row.appendChild(el("span", "rv", bits.join(" · ")));
      }
      box.appendChild(row);
    });
    if (study.caveat) box.appendChild(el("div", "hint", study.caveat));
    box.appendChild(el("div", "hint",
      "A rung marked NOT EVALUATED is a missing input, not a measured " +
      "insensitivity. It makes no claim either way."));
  }

  /* Calibration. For a single case this is almost always not estimable, and
     that is the correct answer rather than a gap in the feature: a calibration
     figure needs outcome labels, and an unconfirmed scene has none. */
  function renderCalibration(job) {
    var box = $("calibration");
    if (!box) return;
    box.innerHTML = "";
    var r = (job && job.calibration) || null;
    if (!r) {
      box.appendChild(el("p", "hint", "The calibration report appears after a run."));
      return;
    }
    var head = el("div", "cal-status " + (r.estimable ? "ok" : "na"));
    head.appendChild(el("b", null, (r.status || "unknown").replace(/_/g, " ")));
    box.appendChild(head);

    var rows = [
      ["Labelled samples", String(r.n != null ? r.n : 0) + " of " + (r.min_samples || 20)],
      ["Brier score", r.brier != null ? fmt(r.brier, 4) : "n/a"],
      ["Brier skill", r.brier_skill != null ? fmt(r.brier_skill, 3) : "n/a"],
      ["Log loss", r.log_loss != null ? fmt(r.log_loss, 4) : "n/a"],
      ["ECE", r.ece != null ? fmt(r.ece, 4) : "n/a"],
      ["AUC", r.auc != null ? fmt(r.auc, 3) : "n/a"]
    ];
    rows.forEach(function (row) {
      var line = el("div", "kv");
      line.appendChild(el("span", "k", row[0]));
      line.appendChild(el("span", "v", row[1]));
      box.appendChild(line);
    });
    if (r.reliability && r.reliability.length) {
      var rc = el("div", "reliability");
      rc.appendChild(el("div", "cal-lbl", "Reliability curve"));
      r.reliability.forEach(function (b) {
        var bar = el("div", "rel-row");
        bar.appendChild(el("span", "rb", String(b.bin != null ? b.bin : "")));
        bar.appendChild(el("span", "rv", "pred " + fmt(b.mean_predicted, 2)));
        bar.appendChild(el("span", "ro", "obs " + fmt(b.observed_frequency, 2)));
        bar.appendChild(el("span", "rn", "n=" + b.n));
        rc.appendChild(bar);
      });
      box.appendChild(rc);
    }
    box.appendChild(el("div", "notice " + (r.estimable ? "ok" : ""),
      r.message || "The investigative score is an uncalibrated evidence " +
      "index. It is not a probability that a vessel is responsible."));
  }

  /* Provenance: what went into the case, and what is missing from it.

     Completeness is shown as a count of absent inputs rather than a green
     tick, because "complete" and "we checked and there was nothing to find"
     are different states and only one of them is a result. */
  function renderProvenance(job) {
    var box = $("provenance");
    if (!box) return;
    box.innerHTML = "";
    var p = (job && job.provenance) || null;
    if (!p) {
      box.appendChild(el("p", "hint", "Provenance appears after a run."));
      return;
    }
    var c = p.completeness || {};
    var rows = [
      ["Case hash", p.case_hash || "n/a"],
      ["Software", (p.software || {}).version || "n/a"],
      ["Pipeline", (p.software || {}).pipeline_version || "n/a"],
      ["Seeds", JSON.stringify(p.seeds || {})],
      ["Inputs recorded", String((p.inputs || []).length)],
      ["Inputs absent", String((p.inputs_missing || []).length)],
      ["Complete", c.complete ? "yes" : "no"]
    ];
    rows.forEach(function (row) {
      var line = el("div", "kv");
      line.appendChild(el("span", "k", row[0]));
      line.appendChild(el("span", "v", row[1]));
      box.appendChild(line);
    });
    var missing = p.inputs_missing || [];
    if (missing.length) {
      var m = el("div", "missing");
      m.appendChild(el("div", "cal-lbl", "Absent inputs"));
      missing.forEach(function (x) {
        var name = typeof x === "string" ? x : (x.name || x.key || "input");
        var it = el("div", "mitem");
        it.appendChild(el("span", "mk", name));
        it.appendChild(el("span", "mw", (typeof x === "object" && x.reason) || "absent"));
        m.appendChild(it);
      });
      box.appendChild(m);
    }
    if (p.warnings && p.warnings.length) {
      var w = el("div", "missing");
      w.appendChild(el("div", "cal-lbl", "Provenance warnings"));
      p.warnings.forEach(function (x) {
        w.appendChild(el("div", "hint", x));
      });
      box.appendChild(w);
    }
  }

  /* The event timeline, ordered, with every entry tagged by what kind of claim
     it is. Mixing a modelled origin hour with a satellite acquisition on one
     unlabeled axis is how an inference ends up quoted as an observation. */
  function renderEvents(job) {
    var box = $("events");
    if (!box) return;
    box.innerHTML = "";
    if (!job) {
      box.appendChild(el("p", "hint", "The case timeline appears after a run."));
      return;
    }
    var events = buildEvents(job);
    if (!events.length) {
      box.appendChild(el("p", "hint", "No dated events for this case."));
      return;
    }
    events.forEach(function (ev) {
      var row = el("div", "evt " + ev.kind);
      row.appendChild(el("span", "ek", ev.kind));
      var b = el("div", "eb");
      b.appendChild(el("div", "el", ev.label));
      b.appendChild(el("div", "et", (ev.t || "").replace(" UTC", "")));
      b.appendChild(el("div", "ed", ev.detail));
      row.appendChild(b);
      box.appendChild(row);
    });
    box.appendChild(el("div", "hint",
      "Observed entries are measurements. Inferred entries are modelled and " +
      "carry their own uncertainty. Attributed entries are conclusions."));
  }

  /* Built here rather than fetched, because every field is already in the job
     document and a second round trip to re-read it would be a way for the
     timeline to disagree with the map beside it. */
  function buildEvents(job) {
    var out = [];
    var scene = job.scene || {};
    var det = job.detection || {};
    var m = det.metrics || {};
    var drift = job.drift || {};
    var origin = drift.origin || {};
    var eo = det.eo || {};

    function add(kind, t, label, detail) {
      out.push({ kind: kind, t: t || "", label: label, detail: detail || "" });
    }

    add("observed", scene.t_sat || job.created, "SAR acquisition",
      "The radar observed the water at this time. Everything downstream is " +
      "relative to it.");

    if (origin.t) {
      add("inferred", origin.t, "Estimated release",
        "Modelled: the first hour where the ensemble spread exceeded the " +
        "trigger. Not an observed discharge time. Zone " +
        fmt(origin.spread_km, 1) + " km.");
    }
    var iv = origin.release_time_interval || {};
    if (iv.start && iv.end) {
      add("inferred", iv.start, "Release window opens",
        "Earliest scenario origin across the drift ensemble.");
      add("inferred", iv.end, "Release window closes",
        "Latest scenario origin across the drift ensemble.");
    }
    if (eo.available && eo.acquired) {
      var dl = eo.time_delta || {};
      add("observed", eo.acquired, "Optical acquisition",
        "Sentinel-2 overpass, " + (dl.hours != null ? fmt(dl.hours, 1) : "?") +
        " h relative to the radar. " + (eo.status || "context only") + ".");
    }
    ((job.attribution || {}).suspects || []).slice(0, 10).forEach(function (s) {
      var d = s.detail || {};
      if (!d.closest_approach_utc) return;
      add(s.rank === 1 ? "attributed" : "inferred", d.closest_approach_utc,
        (s.name || s.mmsi) + " closest approach",
        "Rank " + s.rank + ", " + fmt(s.score, 1) + "%, " +
        fmt(d.origin_distance_km, 1) + " km from the origin estimate. " +
        (s.counter_evidence_count || 0) + " objection(s) recorded.");
    });

    out.sort(function (a, b) { return a.t < b.t ? -1 : (a.t > b.t ? 1 : 0); });
    return out;
  }

  /* Recent cases, from the case index rather than from anything held in
     memory, so the console opens on the work that already exists instead of
     an empty state. Each row carries the quality band and the safe-fail state
     for the same reason the header does: a list showing only candidate counts
     invites reading an empty case as a strong one. */
  function loadCaseIndex() {
    var box = $("case-list");
    if (!box) return Promise.resolve([]);
    return getJSON("/api/cases").then(function (r) {
      box.innerHTML = "";
      var list = (r && r.cases) || [];
      if (!list.length) {
        box.appendChild(el("p", "hint", "No cases run on this machine yet."));
        return list;
      }
      list.forEach(function (c) {
        var row = el("button", "case-row");
        row.type = "button";
        row.appendChild(el("span", "cband " + String(c.case_quality || "unrated").toLowerCase(),
          c.case_quality || "unrated"));
        var main = el("span", "cmain");
        main.appendChild(el("span", "cid", c.scene_id || c.job_id));
        var top = c.top_candidate;
        main.appendChild(el("span", "ctop", top
          ? ("#" + fmt(top.score, 1) + " " + (top.name || top.mmsi) +
             (top.counter_evidence_count ? " · " + top.counter_evidence_count + " obj" : ""))
          : (c.safe_fail_state || "no candidate")));
        row.appendChild(main);
        row.appendChild(el("span", "cdt", (c.created || "").substring(0, 10)));
        row.title = (c.safe_fail_state ? c.safe_fail_state + ". " : "") +
          c.oil_polygons + " polygon(s), " + c.suspects + " candidate(s). " +
          (c.case_hash ? "hash " + String(c.case_hash).substring(0, 12) : "");
        row.addEventListener("click", function () {
          getJSON("/api/jobs/" + c.job_id).then(showJob)
            .catch(function (e) {
              $("notices").appendChild(el("div", "notice bad",
                "Could not open that case: " + e.message));
            });
        });
        box.appendChild(row);
      });
      return list;
    }).catch(function (e) {
      box.innerHTML = "";
      box.appendChild(el("p", "hint", "Case index unavailable: " + e.message));
      return [];
    });
  }


  function fitToJob(job) {
    var pts = [];
    var det = job.detection || {};
    (det.polygons || []).concat(det.lookalikes || []).forEach(function (f) {
      if (f.geometry) f.geometry.coordinates[0].forEach(function (p) { pts.push([p[1], p[0]]); });
    });
    if (job.drift) {
      ["origin_zone", "cone_back", "cone_fwd"].forEach(function (k) {
        var f = job.drift[k];
        if (f && f.geometry) f.geometry.coordinates[0].forEach(function (p) { pts.push([p[1], p[0]]); });
      });
    }
    if (det.sar_bounds && det.sar_bounds.length === 4) {
      pts.push([det.sar_bounds[1], det.sar_bounds[0]]);
      pts.push([det.sar_bounds[3], det.sar_bounds[2]]);
    }
    if (pts.length) map.fitBounds(L.latLngBounds(pts).pad(0.08));
  }

  /* Everything the new rails show is derived from the job document, so one
     call site keeps them all in step. */
  function renderCase(job) {
    renderCaseHead(job);
    renderQuickStats(job);
    renderMetocean(job);
    renderSpreadChart(job);
    renderEvidence(job);
    renderVesselList(job);
  }

  function showJob(job) {
    state.job = job;
    clearAll();
    stop();
    drawDetection(job);
    drawDrift(job);
    var suspects = ((job.attribution || {}).suspects) || [];
    drawTracks(suspects);
    renderVerdict(job);
    renderDetection(job);
    renderOrigin(job);
    renderSuspects(job);
    renderTrace(job);
    renderNotices(job);
    renderCase(job);

    // The panels that answer "how sure is this" and "against what". Grouped
    // here rather than inside renderCase because they are the investigation
    // layer: renderCase is the run summary that also has to work before any
    // run exists, and these five are only meaningful after one.
    renderSafeFail(job);
    renderUncertainty(job);
    renderWhy(job);
    renderSensitivity(job);
    renderAblation(job);
    renderCalibration(job);
    renderProvenance(job);
    renderEvents(job);
    renderShape(job);
    renderOptical(job);
    renderDetectorMeta(job);

    buildFrames(job);
    fitToJob(job);
    loadCaseIndex();
    ["ex-json", "ex-geo", "ex-note"].forEach(function (id) { $(id).disabled = false; });
  }

  /* Detection alone is twenty-five to thirty seconds on a laptop CPU, and
     /api/run answers only when the whole thing is done. A spinner with no
     position is indistinguishable from a hang over that long, so the client
     names the run up front and polls it: the overlay shows which step is in
     flight and how long the run has taken so far. */
  function watchProgress(jobId, t0) {
    var steps = ["DETECT", "CHAR", "EO", "RENDER", "METOCEAN", "HINDCAST",
                 "FORECAST", "COAST", "AGE_PROXY", "AIS", "FILTER", "SCORE"];
    return setInterval(function () {
      var secs = Math.round((performance.now() - t0) / 1000);
      fetch("/api/jobs/" + jobId + "/progress", { cache: "no-store" })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (p) {
          if (!p || !p.running) {
            $("busytext").textContent = "Running analysis · " + secs + "s";
            return;
          }
          var i = steps.indexOf(p.step) + 1;
          $("busytext").textContent =
            (p.step || "Running").toLowerCase() +
            (i > 0 ? " · step " + i + " of " + steps.length : "") +
            " · " + secs + "s";
        })
        .catch(function () { /* progress is a courtesy, never a dependency */ });
    }, 700);
  }

  function runPipeline() {
    if (!state.scene) return;
    $("busy").classList.add("on");
    $("busytext").textContent = "Starting";
    $("run").disabled = true;
    var t0 = performance.now();
    var jobId = "job_ui" + Date.now().toString(36) +
                Math.random().toString(36).slice(2, 8);
    var poll = watchProgress(jobId, t0);

    postJSON("/api/run", {
      scene_id: state.scene.id,
      job_id: jobId,
      hindcast_hours: Number($("hind").value),
      forecast_hours: Number($("fore").value),
      search_radius_km: Number($("radius").value),
      origin_window_hours: Number($("window").value)
    }).then(function (job) {
      showJob(job);
      console.log("pipeline finished in " + Math.round(performance.now() - t0) + " ms", job);
    }).catch(function (err) {
      $("notices").innerHTML = "";
      $("notices").appendChild(el("div", "notice bad", "Run failed: " + err.message));
    }).then(function () {
      clearInterval(poll);
      $("busy").classList.remove("on");
      $("run").disabled = false;
    });
  }

  // Sentinel-2 rarely crosses the same water as Sentinel-1 at the same moment,
  // so the optical chip is loaded per scene, kept OFF by default, and always
  // carries its time offset from the radar pass. It is context for reading the
  // map -- coast, harbour, rigs, wakes -- and never an input to detection.
  function loadOptical(scene) {
    layerGroup.optical.clearLayers();
    var note = $("optical-note");
    if (note) { note.textContent = ""; note.className = "muted"; }
    if (!scene) return;

    fetch("/data/optical/" + encodeURIComponent(scene.id) + ".json", { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (m) {
        if (!note) return;
        if (!m) {
          note.textContent = "Optical: not cached. Run scripts/fetch_sentinel2_chip.py.";
          return;
        }
        if (m.status !== "ok") {
          note.textContent = "Optical: " + (m.reason || m.status) +
            ". Open ocean is usually cloud, which is why detection runs on radar.";
          return;
        }
        L.imageOverlay(m.url, m.bounds,
          { opacity: 0.95, interactive: false, pane: "opticalPane" })
          .addTo(layerGroup.optical);
        note.textContent = "Optical: Sentinel-2 " + utc(m.acquired) + ", " +
          m.offset_label + ", " + m.cloud_percent + "% obscured. Context only.";
      })
      .catch(function () {
        if (note) note.textContent = "Optical: unavailable.";
      });
  }

  // Switching scene used to leave the whole previous run on screen: the old
  // rank-1 vessel marker, the old drift cones, the old AIS provenance warning,
  // and a timeline still scrubbing through frames that belonged to a different
  // sea. The panels said one thing and the map showed another, which is worse
  // than showing nothing. Everything a run produced is cleared back to the
  // documented empty state; the scene's own optical chip is reloaded by
  // renderSceneInfo immediately afterwards.
  function resetRun() {
    stop();
    state.job = null;
    state.suspects = [];
    state.selected = null;
    state.frames = [];
    state.frameIndex = 0;

    Object.keys(layerGroup).forEach(function (k) {
      layerGroup[k].clearLayers();
    });

    var blanks = [
      ["detection", "No run yet. Pick a scene and run the analysis."],
      ["origin", "The hindcast has not run."],
      ["suspects", "Ranked likelihood for investigation. Not proof of discharge."],
      ["trace", "Step timings appear here after a run."],
      ["safefail", "No case open."],
      ["uncertainty", "The uncertainty chain appears after a run."],
      ["why", "No candidate to argue about."],
      ["sensitivity", "Counterfactual scenarios appear after a run."],
      ["ablation", "The ablation study appears after a run."],
      ["calibration", "The calibration report appears after a run."],
      ["provenance", "Provenance appears after a run."],
      ["events", "The case timeline appears after a run."],
      ["shape", "Shape diagnostics appear after a run."],
      ["optical", "Sentinel-2 corroboration. Never an input to detection."],
      ["detector-meta", "Detector metadata appears after a run."]
    ];
    blanks.forEach(function (b) {
      var n = $(b[0]);
      if (!n) return;
      n.innerHTML = "";
      n.appendChild(el("div", "hint", b[1]));
    });

    // Clear the CONTENTS of the verdict card, never the card itself. It owns
    // two elements the renderer looks up by id -- `verdict-time` and `verdict`
    // -- so `verdictCard.innerHTML = ""` deleted them, and the next run died on
    // "Cannot set properties of null (setting 'innerHTML')" before it could
    // draw anything. The card is also `hidden` until a run fills it.
    var v = $("verdict");
    if (v) {
      v.innerHTML = "";
      v.appendChild(el("p", "hint", "No run yet."));
    }
    renderCase(null);

    var notices = $("notices");
    if (notices) notices.innerHTML = "";

    [["susp-count", ""], ["trace-total", ""], ["det-method", ""],
     ["verdict-time", ""], ["tlabel", ""]].forEach(function (kv) {
      var n = $(kv[0]);
      if (n) n.textContent = kv[1];
    });

    var slider = $("slider");
    if (slider) { slider.value = 0; slider.max = 0; slider.disabled = true; }
  }

  function renderSceneInfo(scene) {
    var box = $("scene-info");
    box.innerHTML = "";
    if (!scene) { box.textContent = "No scenes indexed."; return; }
    box.appendChild(el("div", null, "Acquired " + utc(scene.t_sat)));
    box.appendChild(el("div", null, "Bounds " + scene.bounds.map(function (v) {
      return v.toFixed(3);
    }).join(", ")));
    box.appendChild(el("div", null, "CRS " + scene.crs + "  |  AIS " +
      (scene.ais_mode === "real" ? "real MarineCadastre" : "simulated traffic")));
    if (scene.source) box.appendChild(el("div", null, scene.source));
    if (scene.notes) box.appendChild(el("div", "muted", scene.notes));
    var on = el("div", "muted", "");
    on.id = "optical-note";
    box.appendChild(on);
    loadOptical(scene);
  }

  /* ---------------------------------------------------------------- theme
     Two themes, one stored preference, applied before first paint by a stub in
     the document head so a reload never flashes the wrong one. The map ground
     stays dark in both: a chart is imagery, and inverting it would make the
     slick harder to read, which is the one thing this page exists to show. */
  function applyTheme(mode) {
    document.documentElement.dataset.theme = mode;
    try { localStorage.setItem("varuna-theme", mode); } catch (e) { /* private mode */ }
    var b = $("theme");
    if (b) {
      b.title = mode === "dark" ? "Switch to light theme" : "Switch to dark theme";
      b.setAttribute("aria-label", b.title);
    }
    // The chart keeps one palette in both themes. Its ground is imagery and its
    // job is to make a slick legible; a light-theme variant meant near-white
    // continents against a dark ocean, which reads as an inverted map rather
    // than as a chart. Nothing to repaint, so nothing is repainted.
  }

  /* Either rail folds away so the chart can have the width. The choice is
     remembered, because an operator who works with the evidence rail closed
     wants it closed on the next run too. */
  function initRails() {
    [["rail-l", "no-l", "scene rail"], ["rail-r", "no-r", "evidence rail"]]
      .forEach(function (def) {
        var btn = $(def[0]);
        if (!btn) return;
        var key = "varuna-" + def[1];
        var hidden = false;
        try { hidden = localStorage.getItem(key) === "1"; } catch (e) { /* ignore */ }

        function apply() {
          document.getElementById("app").classList.toggle(def[1], hidden);
          btn.setAttribute("aria-pressed", hidden ? "true" : "false");
          btn.title = (hidden ? "Show the " : "Hide the ") + def[2];
          btn.setAttribute("aria-label", btn.title);
          try { localStorage.setItem(key, hidden ? "1" : "0"); } catch (e) { /* ignore */ }
          // Leaflet caches the container size, so it has to be told.
          if (map) setTimeout(function () { map.invalidateSize(); }, 210);
        }

        btn.addEventListener("click", function () { hidden = !hidden; apply(); });
        apply();
      });
  }

  function initTheme() {
    var stored = null;
    try { stored = localStorage.getItem("varuna-theme"); } catch (e) { /* ignore */ }
    applyTheme(stored === "light" ? "light" : "dark");
    var b = $("theme");
    if (b) {
      b.addEventListener("click", function () {
        applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
      });
    }
  }

  /* ------------------------------------------------------------------- nav
     The six destinations are modes over the same run, not separate pages, and
     they are ordered the way a case is actually worked: what happened, what
     was detected, where it came from, who is involved, how fragile is the
     ranking, and what the answer rests on. Everything they show comes from the
     job document already in memory, so switching mode never refetches and can
     never disagree with itself. */
  var selectView = null;

  function initNav() {
    var links = Array.prototype.slice.call(document.querySelectorAll(".navlink"));
    function select(view) {
      // A deep link can name a mode that no longer exists, and hiding every
      // panel at once is how a shared case URL ends up showing an empty rail
      // with no indication of why. Unknown names fall back to the overview.
      var known = links.map(function (l) { return l.dataset.view; });
      if (known.indexOf(view) < 0) view = known[0] || "overview";
      links.forEach(function (l) {
        l.setAttribute("aria-selected", l.dataset.view === view ? "true" : "false");
      });
      Array.prototype.forEach.call(document.querySelectorAll(".view"), function (v) {
        v.hidden = v.dataset.view !== view;
      });
      state.view = view;
      if (view === "attribution") renderVesselList(state.job);
    }
    links.forEach(function (l, i) {
      l.addEventListener("click", function () { select(l.dataset.view); });
      l.addEventListener("keydown", function (ev) {
        var step = ev.key === "ArrowRight" ? 1 : (ev.key === "ArrowLeft" ? -1 : 0);
        if (!step) return;
        ev.preventDefault();
        var next = links[(i + step + links.length) % links.length];
        select(next.dataset.view);
        next.focus();
      });
    });
    select("overview");
    selectView = select;
  }

  /* A case reference an analyst can say out loud: the basin, then the date of
     the pass. OS for oil spill, matching how a case file is named rather than
     how the job store keys its documents. */
  function caseRef(job) {
    var scene = job.scene || {};
    var id = (scene.id || "case").split("_");
    var tag = (id[0] || "xx").slice(0, 3).toUpperCase() +
              (id[1] ? id[1].slice(0, 2).toUpperCase() : "");
    var t = new Date(scene.t_sat || job.created);
    if (isNaN(t.getTime())) return "Case " + tag;
    var d = t.toISOString().slice(0, 10).replace(/-/g, "");
    return "OS-" + tag + "-" + d;
  }

  function showVesselsView() {
    var l = document.querySelector('.navlink[data-view="attribution"]');
    if (l) l.click();
  }

  /* --------------------------------------------------------- case header */
  function renderCaseHead(job) {
    var t = $("inc-title"), st = $("inc-status"), meta = $("inc-meta"), co = $("inc-coords");
    if (!t) return;
    if (!job) {
      t.textContent = "No run yet";
      st.textContent = "Idle";
      st.className = "pill";
      meta.textContent = "Pick a scene on the left and run the analysis.";
      co.textContent = "-";
      var qb0 = $("case-q");
      if (qb0) { qb0.textContent = ""; qb0.className = "cband"; }
      return;
    }
    var det = job.detection || {};
    var m = det.metrics || {};
    var scene = job.scene || {};
    var polys = det.polygons || [];
    var clean = !polys.length;

    t.textContent = caseRef(job);
    t.title = "Job " + (job.job_id || "");
    st.textContent = clean ? "Clean" : "Active";
    st.className = "pill " + (clean ? "clean" : "active");

    // The quality band rides in the header rather than only in the rail. It is
    // the single fact that changes how every other number should be read, so it
    // belongs where it is visible no matter which mode is open. It is labelled
    // inline so it cannot be read as a score.
    var qband = $("case-q");
    if (qband) {
      var q = job.case_quality || {};
      var sf = q.safe_fail || {};
      var parts = [];
      if (q.overall) parts.push("EVIDENCE " + q.overall);
      if (sf.state) parts.push(sf.state);
      qband.textContent = parts.join("  ·  ") || "";
      qband.className = "cband " + String(q.overall || "unrated").toLowerCase() +
        (sf.state ? " has-state" : "");
      qband.title = "Case quality describes the evidence available, not a " +
        "probability that anyone discharged oil.";
    }

    var first = (polys[0] || {}).properties;
    meta.textContent = [
      "Radar pass " + utc(scene.t_sat || job.created),
      "Sensor " + (scene.source ? scene.source.split(" via ")[0] : "Sentinel-1"),
      "Detector " + (m.detector === "unet" ? "U-Net" : "dB baseline"),
      first ? "Confidence " + fmt(first.confidence, 2) : null
    ].filter(Boolean).join("  ·  ");

    var lat = first ? first.centroid_lat : (scene.centroid ? scene.centroid[1] : null);
    var lon = first ? first.centroid_lon : (scene.centroid ? scene.centroid[0] : null);
    co.textContent = lat == null ? "-"
      : fmt(Math.abs(lat), 4) + "° " + (lat >= 0 ? "N" : "S") + ", " +
        fmt(Math.abs(lon), 4) + "° " + (lon >= 0 ? "E" : "W");
  }

  /* --------------------------------------------------------- run summary */
  function renderQuickStats(job) {
    var det = (job && job.detection) || {};
    var m = det.metrics || {};
    var sus = ((job && job.attribution) || {}).suspects || [];
    var set = function (id, v) { var n = $(id); if (n) n.textContent = v; };
    set("qs-polys", job ? String((det.polygons || []).length) : "-");
    set("qs-area", job ? fmt(m.oil_area_km2, 2) : "-");
    set("qs-vessels", job ? String(sus.length) : "-");
    set("qs-age", job && job.age_hours_proxy != null ? fmt(job.age_hours_proxy, 1) : "-");
  }

  /* ------------------------------------------------- environmental panel
     Wind and current are the two fields the drift model actually integrates,
     so those are the two reported. Wave height and sea surface temperature are
     not in the cached cube, and are not invented to fill the grid. */
  var ENV_ICONS = {
    wind: '<path d="M2 5h7a2.2 2.2 0 1 0-2.2-2.2"/><path d="M2 8h10a2.2 2.2 0 1 1-2.2 2.2"/><path d="M2 11h6"/>',
    current: '<path d="M1.5 5.5c2-2 3.5-2 5.5 0s3.5 2 5.5 0"/><path d="M1.5 9c2-2 3.5-2 5.5 0s3.5 2 5.5 0"/><path d="M11 12.5l2-1.5-2-1.5"/>',
    grid: '<rect x="2" y="2" width="12" height="12" rx="1"/><path d="M2 6h12M2 10h12M6 2v12M10 2v12"/>',
    clock: '<circle cx="8" cy="8" r="6"/><path d="M8 4.5V8l2.5 1.5"/>'
  };

  function renderMetocean(job) {
    var box = $("metocean");
    if (!box) return;
    box.innerHTML = "";
    var d = (job && job.drift) || {};
    var mo = d.metocean;
    if (!mo) {
      box.appendChild(el("p", "hint", job
        ? "This run did not reach the drift stage, so no metocean cube was read."
        : "Read from the cached metocean cube on the next run."));
      return;
    }

    var grid = el("div", "env");
    [
      ["wind", fmt(mo.mean_wind_ms, 1) + " m/s", "Mean 10 m wind"],
      ["current", mo.has_currents ? fmt(mo.mean_current_ms, 2) + " m/s" : "none",
        mo.has_currents ? "Mean surface current" : "No current model here"],
      ["grid", (mo.grid || []).join(" × ") + " pts", "Field resolution"],
      ["clock", (mo.n_times || 0) + " h", "Cube time span"]
    ].forEach(function (row) {
      var item = el("div", "env-item");
      var ic = el("div", "env-ic");
      ic.innerHTML = '<svg viewBox="0 0 16 16" aria-hidden="true">' +
                     (ENV_ICONS[row[0]] || "") + "</svg>";
      item.appendChild(ic);
      var txt = el("div");
      txt.appendChild(el("div", "env-v", row[1]));
      txt.appendChild(el("div", "env-l", row[2]));
      item.appendChild(txt);
      grid.appendChild(item);
    });
    box.appendChild(grid);

    var note = el("p", "hint", mo.source);
    note.style.marginTop = "10px";
    box.appendChild(note);
  }

  /* --------------------------------------------- hindcast spread chart
     The ensemble spread against hours before the radar pass. This is the curve
     the origin rule is read off: the backward run stops where the spread
     crosses its trigger, because past that point the physics has stopped
     narrowing anything down and the case belongs to AIS. Plotting it makes the
     size of the release zone an argument rather than an assertion.

     Every point is `drift.hindcast_hourly[i].spread_km` straight from the job
     document. An earlier version re-derived the radius in the browser from the
     envelope rings, which put the marker 9 percent off the origin's own
     reported spread -- a chart that disagrees with the number beside it is
     worse than no chart. */
  function svgNode(name, attrs, cls) {
    var n = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (cls) n.setAttribute("class", cls);
    return n;
  }

  function renderSpreadChart(job) {
    var box = $("spreadchart");
    if (!box) return;
    box.innerHTML = "";
    var d = (job && job.drift) || {};
    var hourly = d.hindcast_hourly;
    if (!hourly || hourly.length < 3) {
      box.appendChild(el("p", "hint", job
        ? "No hindcast ensemble for this run."
        : "Ensemble spread against hours before the radar pass."));
      return;
    }

    var pts = hourly.map(function (h, i) { return { h: i, r: Number(h.spread_km) || 0 }; });
    var W = 300, H = 96, L = 28, R = 8, T = 10, B = 16;
    var maxR = Math.max.apply(null, pts.map(function (q) { return q.r; })) || 1;
    var maxH = pts.length - 1;
    var x = function (h) { return L + (h / maxH) * (W - L - R); };
    var y = function (r) { return T + (1 - r / maxR) * (H - T - B); };

    var svg = svgNode("svg", { viewBox: "0 0 " + W + " " + H,
                               preserveAspectRatio: "none" }, "chart");

    [0, 0.5, 1].forEach(function (f) {
      svg.appendChild(svgNode("line",
        { x1: L, x2: W - R, y1: y(maxR * f), y2: y(maxR * f) }, "grid"));
      var tx = svgNode("text", { x: 2, y: y(maxR * f) + 3 }, "tick");
      tx.textContent = (maxR * f).toFixed(0);
      svg.appendChild(tx);
    });

    var dLine = pts.map(function (q, i) {
      return (i ? "L" : "M") + x(q.h).toFixed(1) + " " + y(q.r).toFixed(1);
    }).join(" ");
    svg.appendChild(svgNode("path",
      { d: dLine + " L" + x(maxH) + " " + y(0) + " L" + x(0) + " " + y(0) + " Z" }, "area"));
    svg.appendChild(svgNode("path", { d: dLine }, "line"));

    // Where the origin was taken, drawn on the point it was taken from.
    var oh = d.origin && d.origin.index_hours_back;
    if (oh != null && oh <= maxH) {
      var or_ = pts[Math.round(oh)].r;
      svg.appendChild(svgNode("line", { x1: x(oh), x2: x(oh), y1: T, y2: H - B }, "marker"));
      svg.appendChild(svgNode("circle", { cx: x(oh), cy: y(or_), r: 2.6 }, "originpt"));
      var lab = svgNode("text", { x: Math.min(x(oh) + 5, W - 74), y: T + 8 }, "mlabel");
      lab.textContent = "origin −" + oh + "h · " + or_.toFixed(1) + " km";
      svg.appendChild(lab);
    }

    svg.appendChild(svgNode("line", { x1: L, x2: W - R, y1: H - B, y2: H - B }, "axis"));
    [0, Math.round(maxH / 2), maxH].forEach(function (h) {
      var tx = svgNode("text", { x: Math.max(L - 6, x(h) - 9), y: H - 4 }, "tick");
      tx.textContent = "−" + h + "h";
      svg.appendChild(tx);
    });

    box.appendChild(svg);
    var cap = el("p", "hint",
      "Ensemble spread in km, from the job document, against hours before the pass.");
    cap.style.marginTop = "4px";
    box.appendChild(cap);
  }

  /* ------------------------------------------- drift reconstruction list
     Four moments with real times attached, in order, so a timeline here is
     information rather than ornament. */
  function driftTimeline(job) {
    var d = (job && job.drift) || {};
    if (!d.origin) return null;
    var tObs = (job.scene || {}).t_sat || job.created;
    var back = (job.input || {}).hindcast_hours || 48;
    var fwd = (job.input || {}).forecast_hours || 36;

    function shift(iso, hours) {
      return utc(new Date(new Date(iso).getTime() + hours * 3600e3).toISOString());
    }

    var ul = el("ul", "dtl");
    [
      ["Backtrack start", shift(tObs, -back), ""],
      ["Estimated release zone", utc(d.origin.t), "origin"],
      ["Observed by radar", utc(tObs), ""],
      ["Forecast horizon", shift(tObs, fwd), "fore"]
    ].forEach(function (row) {
      var li = el("li", row[2]);
      li.appendChild(el("div", "lbl", row[0]));
      li.appendChild(el("div", "tm", row[1].replace(" UTC", "")));
      ul.appendChild(li);
    });
    return ul;
  }

  /* ------------------------------------------------------ evidence summary */
  function renderEvidence(job) {
    var box = $("evidence");
    if (!box) return;
    box.innerHTML = "";
    if (!job) {
      box.appendChild(el("p", "hint", "Counts of what a finding rests on."));
      return;
    }
    var det = job.detection || {};
    var funnel = (job.attribution || {}).funnel || {};
    var rows = [
      ["Oil polygons", (det.polygons || []).length],
      ["Look-alikes excluded", (det.lookalikes || []).length],
      ["Optical chips compared", (det.eo && det.eo.available) ? 1 : 0],
      ["Vessels in the box", funnel.considered_vessels != null ? funnel.considered_vessels : "-"],
      ["Passed the filter", funnel.kept_vessels != null ? funnel.kept_vessels : "-"],
      ["Ensemble members", (job.config || {}).ensemble_n || "-"],
      ["Pipeline steps", (job.trace || []).length]
    ];
    rows.forEach(function (row) {
      var r = el("div", "ev-row");
      r.appendChild(el("span", "k", row[0]));
      r.appendChild(el("span", "v", String(row[1])));
      box.appendChild(r);
    });
  }

  /* ------------------------------------------------ data sources, system */
  function renderSources(h) {
    var box = $("sources");
    if (!box) return;
    box.innerHTML = "";
    [
      ["Detector", h.model_loaded ? "U-Net" : "dB baseline", !h.model_loaded],
      ["Metocean cubes", String(h.metocean_scenes), !h.metocean_scenes],
      ["AIS rows", Number(h.ais_rows).toLocaleString(), !h.ais_rows],
      ["AIS vessels", String(h.ais_vessels), !h.ais_vessels],
      ["Scenes indexed", String(h.scenes), !h.scenes],
      ["Run history", ((h.storage || {}).megabytes || 0) + " MB", false],
      ["Network", h.mode === "offline" ? "not used at runtime" : "online", false]
    ].forEach(function (row) {
      var r = el("div", "src" + (row[2] ? " bad" : ""));
      r.appendChild(el("span", "nm", row[0]));
      r.appendChild(el("span", "vl", row[1]));
      box.appendChild(r);
    });
  }

  function renderSysInfo(h) {
    var box = $("sysinfo");
    if (!box) return;
    box.innerHTML = "";
    var c = h.config || {};
    var md = h.model_detail || {};
    [
      ["Version", h.version],
      ["Mode", h.mode],
      ["Device", md.device || "cpu"],
      ["Checkpoint", md.checkpoint_mb ? md.checkpoint_mb + " MB" : "absent"],
      ["Wind factor", c.alpha_wind],
      ["Ensemble", c.ensemble_n],
      ["Seed", c.seed],
      ["Jobs kept", c.keep_jobs]
    ].forEach(function (row) {
      var kv = el("div", "kv");
      kv.appendChild(el("span", "k", row[0]));
      kv.appendChild(el("span", "v", String(row[1])));
      box.appendChild(kv);
    });
  }

  /* ------------------------------------------------------- vessels view */
  function renderVesselList(job) {
    var box = $("vessel-list");
    if (!box) return;
    box.innerHTML = "";
    var list = ((job && job.attribution) || {}).suspects || [];
    var cnt = $("vessels-count");
    if (cnt) cnt.textContent = list.length ? list.length + " ranked" : "";
    if (!list.length) {
      box.appendChild(el("p", "hint",
        "Run the analysis to reconstruct traffic around the origin."));
      return;
    }
    var q = (($("vsearch") || {}).value || "").trim().toLowerCase();
    var shown = list.filter(function (s) {
      return !q || String(s.mmsi).indexOf(q) >= 0 ||
        (s.name || "").toLowerCase().indexOf(q) >= 0;
    });
    if (!shown.length) {
      box.appendChild(el("p", "hint", "No vessel matches that search."));
      return;
    }
    shown.forEach(function (s) { box.appendChild(suspectCard(s)); });
  }

  function loadHealth() {
    return getJSON("/api/health").then(function (h) {
      state.health = h;
      renderSources(h);
      renderSysInfo(h);

      if (h.warnings && h.warnings.length) {
        var box = $("notices");
        h.warnings.forEach(function (w) {
          box.appendChild(el("div", "notice", w));
        });
      }
      return h;
    });
  }

  /* The scene picker is the case list. Each row carries the two facts that
     decide whether a run means anything: when the radar passed, and whether the
     AIS under it was recorded or simulated. The <select> stays in the DOM,
     hidden, because it is the element the rest of the app reads. */
  function renderSceneList(list) {
    var box = $("scene-list");
    if (!box) return;
    box.innerHTML = "";
    list.forEach(function (sc) {
      var row = el("button", "scene-row");
      row.type = "button";
      row.dataset.id = sc.id;
      row.setAttribute("aria-current", state.scene && state.scene.id === sc.id ? "true" : "false");
      row.appendChild(el("span", "nm", sc.title));
      var real = sc.ais_mode === "real";
      row.appendChild(el("span", "ais " + (real ? "real" : "sim"),
        real ? "recorded AIS" : "simulated AIS"));
      row.appendChild(el("span", "dt", utc(sc.t_sat).replace(" UTC", "") + " UTC"));
      row.addEventListener("click", function () { selectScene(sc.id); });
      box.appendChild(row);
    });
  }

  function selectScene(id) {
    var sel = $("scene");
    if (sel) sel.value = id;
    state.scene = state.scenes.filter(function (s) { return s.id === id; })[0] || null;
    Array.prototype.forEach.call(document.querySelectorAll(".scene-row"), function (r) {
      r.setAttribute("aria-current", r.dataset.id === id ? "true" : "false");
    });
    resetRun();
    renderSceneInfo(state.scene);
    loadOptical(state.scene);
    fitScene(state.scene);
  }

  function loadScenes() {
    return getJSON("/api/scenes").then(function (list) {
      state.scenes = list;
      var sel = $("scene");
      sel.innerHTML = "";
      list.forEach(function (s) {
        var o = document.createElement("option");
        o.value = s.id;
        o.textContent = s.title;
        sel.appendChild(o);
      });
      if (list.length) {
        state.scene = list[0];
        renderSceneList(list);
        renderSceneInfo(state.scene);
        fitScene(state.scene);
      } else {
        $("run").disabled = true;
        renderSceneList([]);
        renderSceneInfo(null);
      }
      return list;
    });
  }

  /* Frame the scene's footprint.

     Deferred by a frame and preceded by invalidateSize because the first call
     races the sidebar's layout: the scene list resolves while the grid is still
     settling, Leaflet measures a container that is not its final size, and the
     zoom it computes from that is wrong -- the console opened on a world view
     with the scene somewhere in it. */
  function fitScene(scene, attempt) {
    if (!scene || !scene.bounds) return;
    var box = [[scene.bounds[1], scene.bounds[0]], [scene.bounds[3], scene.bounds[2]]];
    attempt = attempt || 0;

    map.invalidateSize({ animate: false });
    var size = map.getSize();
    // A container that has not been laid out yet measures near zero, and
    // fitBounds against that returns the maximum zoom -- the console opened at
    // a 50 m scale bar somewhere inside the footprint. Wait for a real size.
    if ((size.x < 80 || size.y < 80) && attempt < 20) {
      setTimeout(function () { fitScene(scene, attempt + 1); }, 50);
      return;
    }
    map.fitBounds(box, { padding: [24, 24], animate: false });
  }

  function tickClock() {
    $("clock").textContent = "UTC " + new Date().toISOString().substring(11, 19);
  }

  /* Diagnostic probe. Runs clause (b) and (c) at a point the operator picks, so
     the physics and the AIS join can be exercised on open water. It skips
     detection entirely and the response says so. It is not the judged path. */
  function runProbe(latlng) {
    $("busy").classList.add("on");
    $("busytext").textContent = "PROBE: DRIFT AND AIS ONLY";
    postJSON("/api/demo/inject", {
      lat: latlng.lat,
      lon: latlng.lng,
      hindcast_hours: Number($("hind").value),
      forecast_hours: Number($("fore").value),
      radius_km: Number($("radius").value),
      window_h: Number($("window").value)
    }).then(showJob).catch(function (err) {
      $("notices").appendChild(el("div", "notice bad", "Probe failed: " + err.message));
    }).then(function () {
      $("busy").classList.remove("on");
    });
  }

  function wire() {
    initTheme();
    initRails();
    initNav();
    $("run").addEventListener("click", runPipeline);

    $("probe").addEventListener("change", function (e) {
      document.getElementById("map").classList.toggle("probing", e.target.checked);
    });
    map.on("click", function (e) {
      if ($("probe").checked) runProbe(e.latlng);
    });
    // The visible picker is the case list; the <select> stays for keyboard and
    // for anything that sets it programmatically. Both land in selectScene, so
    // there is one code path for changing scene rather than two that drift.
    $("scene").addEventListener("change", function (e) {
      selectScene(e.target.value);
    });

    // Filtering the ranked list is a read, so it applies as you type.
    $("vsearch").addEventListener("input", function () {
      renderVesselList(state.job);
      if (state.view !== "attribution") showVesselsView();
    });
    $("slider").addEventListener("input", function (e) {
      stop();
      state.frameIndex = Number(e.target.value);
      renderFrame();
    });
    $("play").addEventListener("click", play);
    $("toEnd").addEventListener("click", function () {
      stop();
      state.frameIndex = Math.max(0, state.frames.length - 1);
      $("slider").value = state.frameIndex;
      renderFrame();
    });
    $("ex-json").addEventListener("click", function () {
      if (state.job) window.open("/api/jobs/" + state.job.job_id, "_blank");
    });
    $("ex-geo").addEventListener("click", function () {
      if (state.job) window.open("/api/jobs/" + state.job.job_id + "/geojson", "_blank");
    });
    $("ex-note").addEventListener("click", function () {
      if (state.job) window.open("/api/report/" + state.job.job_id, "_blank");
    });
  }

  function bindKeys() {
    document.addEventListener("keydown", function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      var tag = (e.target.tagName || "").toLowerCase();
      var typing = tag === "input" || tag === "select" || tag === "textarea";
      if (e.key === "Enter" && !$("run").disabled) {
        e.preventDefault();
        runPipeline();
      } else if (!typing && (e.key === " " || e.key === "k")) {
        if (state.frames.length > 1) { e.preventDefault(); play(); }
      }
    });
  }

  /* A finished run is a case, and a case should be linkable. The fragment
     #job=<id>&view=vessels reopens a stored run at the view worth arguing
     about, without recomputing it. */
  function linkParams() {
    var out = {};
    (location.hash || "").replace(/^#/, "").split("&").forEach(function (pair) {
      var i = pair.indexOf("=");
      if (i > 0) out[pair.slice(0, i)] = decodeURIComponent(pair.slice(i + 1));
    });
    return out;
  }

  function openLinkedJob() {
    var q = linkParams();
    if (q.theme === "light" || q.theme === "dark") applyTheme(q.theme);
    if (q.view && selectView) selectView(q.view);
    if (!/^[A-Za-z0-9_-]+$/.test(q.job || "")) return;
    return getJSON("/api/jobs/" + q.job).then(function (job) {
      showJob(job);
      if (q.view && selectView) selectView(q.view);
    }).catch(function (e) {
      $("notices").appendChild(el("div", "notice bad", "No such run: " + e.message));
    });
  }

  function boot() {
    initMap();
    renderLayers();
    wire();
    bindKeys();
    tickClock();
    setInterval(tickClock, 1000);
    loadHealth().catch(function (e) { console.error(e); });
    loadScenes().then(openLinkedJob).catch(function (e) {
      $("notices").appendChild(el("div", "notice bad", "Could not load scenes: " + e.message));
    });
    getJSON("/api/scoring").then(renderWeights).catch(function (e) { console.error(e); });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
