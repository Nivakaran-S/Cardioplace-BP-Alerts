/* Cardioplace BP Alerts — dashboard behaviour and the SBP chart.
   No external libraries: the chart is plain SVG so the page works offline and
   inside a HuggingFace Space with no CDN access. */

(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var SVG_NS = "http://www.w3.org/2000/svg";

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function el(tag, attrs, text) {
    var node = document.createElementNS(SVG_NS, tag);
    for (var k in attrs) { if (attrs[k] !== null) node.setAttribute(k, attrs[k]); }
    if (text !== undefined) node.textContent = text;
    return node;
  }

  /* ------------------------------------------------------------- theme */

  var THEME_KEY = "cardioplace-theme";
  function applyTheme(mode) {
    if (mode) { document.documentElement.setAttribute("data-theme", mode); }
    else { document.documentElement.removeAttribute("data-theme"); }
  }
  applyTheme(localStorage.getItem(THEME_KEY));

  $("theme-toggle").addEventListener("click", function () {
    var current = document.documentElement.getAttribute("data-theme");
    var prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    var next = current ? (current === "dark" ? "light" : "dark")
                       : (prefersDark ? "light" : "dark");
    applyTheme(next);
    localStorage.setItem(THEME_KEY, next);
    if (LAST) render(LAST);
  });

  /* ------------------------------------------------------- sample data */

  // Deterministic pseudo-history: 3x/week cadence, AR(1)-ish drift upward so the
  // forecast and the early-warning detector both have something to react to.
  function sampleReadings() {
    var out = [], seed = 20260728, sbp = 138, dbp = 76;
    function rand() { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; }
    var d = new Date(Date.UTC(2026, 0, 5));
    var gaps = [2, 2, 3];
    for (var i = 0; i < 36; i++) {
      sbp = 0.72 * sbp + 0.28 * (140 + i * 0.55) + (rand() - 0.5) * 13;
      dbp = 0.70 * dbp + 0.30 * (75 + i * 0.18) + (rand() - 0.5) * 7;
      out.push(d.toISOString().slice(0, 10) + ", " + Math.round(sbp) + ", " + Math.round(dbp));
      d = new Date(d.getTime() + gaps[i % 3] * 86400000);
    }
    return out.join("\n");
  }

  /* ------------------------------------------------------------ parsing */

  function parseReadings(raw) {
    var lines = raw.split("\n"), rows = [], errors = [];
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i].trim();
      if (!line) continue;
      var parts = line.split(/[,\t;]+/).map(function (s) { return s.trim(); });
      if (parts.length < 3) { errors.push("line " + (i + 1) + ": expected date, SBP, DBP"); continue; }
      var sbp = Number(parts[1]), dbp = Number(parts[2]);
      if (!/^\d{4}-\d{2}-\d{2}$/.test(parts[0])) { errors.push("line " + (i + 1) + ": date must be YYYY-MM-DD"); continue; }
      if (!isFinite(sbp) || !isFinite(dbp)) { errors.push("line " + (i + 1) + ": SBP/DBP must be numeric"); continue; }
      var row = { ts: parts[0], sbp: sbp, dbp: dbp };
      if (parts.length > 3 && parts[3] !== "" && isFinite(Number(parts[3]))) row.weight = Number(parts[3]);
      if (parts.length > 4 && parts[4] !== "" && isFinite(Number(parts[4]))) row.idwg = Number(parts[4]);
      rows.push(row);
    }
    return { rows: rows, errors: errors };
  }

  function updateCount() {
    var p = parseReadings($("readings").value);
    var n = p.rows.length;
    $("reading-count").textContent = n + (n === 1 ? " reading" : " readings")
      + (n < 7 ? " — below the 7-reading cold-start floor, no forecast will be issued" : "");
  }

  /* ------------------------------------------------------------- chart */

  var LAST = null;

  function render(data) {
    var host = $("chart");
    host.innerHTML = "";

    var hist = data.history || [];
    if (!hist.length) return;

    var s1 = cssVar("--series-1"), s2 = cssVar("--series-2");
    var grid = cssVar("--grid"), axis = cssVar("--axis");
    var muted = cssVar("--text-muted"), secondary = cssVar("--text-secondary");
    var warning = cssVar("--status-warning"), critical = cssVar("--status-critical");
    var surface = cssVar("--surface-1");

    // forecast points, ordered by horizon
    var fc = (data.forecast && data.forecast.sbp) || {};
    var fcPts = Object.keys(fc).map(function (k) {
      return { key: k, h: fc[k].steps_ahead, point: fc[k].point,
               lo: fc[k].lo80, hi: fc[k].hi80, days: fc[k].days_ahead_est };
    }).sort(function (a, b) { return a.h - b.h; });

    var thr = data.personalisation ? data.personalisation.threshold : null;
    var floor = data.emergency_floor_mmHg;

    var W = 900, H = 340;
    var m = { t: 18, r: 78, b: 34, l: 46 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;

    var n = hist.length, total = n + fcPts.length;
    var vals = hist.map(function (d) { return d.sbp; });
    fcPts.forEach(function (p) {
      vals.push(p.point);
      if (p.lo != null) vals.push(p.lo);
      if (p.hi != null) vals.push(p.hi);
    });
    if (thr != null) vals.push(thr);
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    if (floor != null && hi > floor - 12) vals.push(floor);
    lo = Math.min.apply(null, vals); hi = Math.max.apply(null, vals);
    var pad = Math.max(6, (hi - lo) * 0.12);
    var yMin = Math.floor((lo - pad) / 10) * 10, yMax = Math.ceil((hi + pad) / 10) * 10;

    var X = function (i) { return m.l + (total <= 1 ? 0 : (i / (total - 1)) * iw); };
    var Y = function (v) { return m.t + ih - ((v - yMin) / (yMax - yMin)) * ih; };

    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "xMidYMid meet" });

    // ---- gridlines + y axis (recessive)
    var step = (yMax - yMin) <= 60 ? 10 : 20;
    for (var v = yMin; v <= yMax; v += step) {
      svg.appendChild(el("line", { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v),
                                   stroke: grid, "stroke-width": 1 }));
      svg.appendChild(el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end",
                                   fill: muted, "font-size": 11 }, String(v)));
    }
    svg.appendChild(el("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih,
                                 stroke: axis, "stroke-width": 1 }));

    // ---- x labels: first, a few middles, last observed, and the last forecast
    var ticks = [0, Math.floor(n / 3), Math.floor((2 * n) / 3), n - 1];
    if (fcPts.length) ticks.push(total - 1);
    ticks.filter(function (t, i, a) { return a.indexOf(t) === i && t >= 0; }).forEach(function (i) {
      var label = i < n ? hist[i].ts.slice(5)
                        : "+" + fcPts[i - n].h + " sess";
      svg.appendChild(el("text", { x: X(i), y: m.t + ih + 20, "text-anchor": "middle",
                                   fill: muted, "font-size": 11 }, label));
    });

    // ---- threshold reference lines (status colour + always a text label)
    function refLine(value, colour, label) {
      if (value == null || value < yMin || value > yMax) return;
      svg.appendChild(el("line", { x1: m.l, x2: m.l + iw, y1: Y(value), y2: Y(value),
                                   stroke: colour, "stroke-width": 2,
                                   "stroke-dasharray": "5 4", opacity: 0.9 }));
      svg.appendChild(el("text", { x: m.l + iw + 6, y: Y(value) + 4,
                                   fill: secondary, "font-size": 11 }, label));
    }
    refLine(thr, warning, "threshold " + thr);
    refLine(floor, critical, "floor " + floor);

    // ---- forecast interval band
    var band = fcPts.filter(function (p) { return p.lo != null && p.hi != null; });
    if (band.length) {
      band.forEach(function (p) {
        var i = n + fcPts.indexOf(p);
        svg.appendChild(el("line", { x1: X(i), x2: X(i), y1: Y(p.hi), y2: Y(p.lo),
                                     stroke: s2, "stroke-width": 8, opacity: 0.20,
                                     "stroke-linecap": "round" }));
      });
    }

    // ---- observed line (2px)
    var dPath = hist.map(function (p, i) { return (i ? "L" : "M") + X(i) + " " + Y(p.sbp); }).join(" ");
    svg.appendChild(el("path", { d: dPath, fill: "none", stroke: s1, "stroke-width": 2,
                                 "stroke-linejoin": "round", "stroke-linecap": "round" }));

    // ---- forecast line, from the last observed point
    if (fcPts.length) {
      var fPath = "M" + X(n - 1) + " " + Y(hist[n - 1].sbp) + " "
        + fcPts.map(function (p, i) { return "L" + X(n + i) + " " + Y(p.point); }).join(" ");
      svg.appendChild(el("path", { d: fPath, fill: "none", stroke: s2, "stroke-width": 2,
                                   "stroke-dasharray": "6 4", "stroke-linecap": "round" }));
      fcPts.forEach(function (p, i) {
        // 2px surface ring so overlapping marks stay separable
        svg.appendChild(el("circle", { cx: X(n + i), cy: Y(p.point), r: 5,
                                       fill: s2, stroke: surface, "stroke-width": 2 }));
      });
      // one direct label on the final horizon, not a number on every point
      var lastP = fcPts[fcPts.length - 1];
      svg.appendChild(el("text", { x: X(total - 1), y: Y(lastP.point) - 12,
                                   "text-anchor": "middle", fill: secondary,
                                   "font-size": 11, "font-weight": 600 },
                         Math.round(lastP.point) + ""));
    }

    // last observed marker
    svg.appendChild(el("circle", { cx: X(n - 1), cy: Y(hist[n - 1].sbp), r: 4.5,
                                   fill: s1, stroke: surface, "stroke-width": 2 }));

    // ---- hover layer: crosshair + tooltip
    var cross = el("line", { y1: m.t, y2: m.t + ih, stroke: axis, "stroke-width": 1,
                             opacity: 0, "pointer-events": "none" });
    var dot = el("circle", { r: 4.5, fill: s1, stroke: surface, "stroke-width": 2,
                             opacity: 0, "pointer-events": "none" });
    svg.appendChild(cross); svg.appendChild(dot);

    var hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" });
    svg.appendChild(hit);
    host.appendChild(svg);

    var tip = document.createElement("div");
    tip.className = "tooltip";
    host.appendChild(tip);

    hit.addEventListener("mousemove", function (ev) {
      var box = svg.getBoundingClientRect();
      var scale = W / box.width;
      var px = (ev.clientX - box.left) * scale;
      var idx = Math.round(((px - m.l) / iw) * (total - 1));
      idx = Math.max(0, Math.min(total - 1, idx));

      var isObs = idx < n;
      var val = isObs ? hist[idx].sbp : fcPts[idx - n].point;
      var colour = isObs ? s1 : s2;

      cross.setAttribute("x1", X(idx)); cross.setAttribute("x2", X(idx));
      cross.setAttribute("opacity", 0.7);
      dot.setAttribute("cx", X(idx)); dot.setAttribute("cy", Y(val));
      dot.setAttribute("fill", colour); dot.setAttribute("opacity", 1);

      var rows = '<div class="tt-date">'
        + (isObs ? hist[idx].ts : "forecast +" + fcPts[idx - n].h + " sessions ("
            + fcPts[idx - n].days + " d)") + "</div>"
        + '<div class="tt-row"><span class="tt-dot" style="background:' + colour + '"></span>'
        + "SBP <span class=\"tt-val\">" + Math.round(val) + "</span></div>";
      if (isObs) {
        rows += '<div class="tt-row"><span class="tt-dot" style="background:' + muted
          + '"></span>DBP <span class="tt-val">' + Math.round(hist[idx].dbp) + "</span></div>";
      }
      tip.innerHTML = rows;
      tip.setAttribute("data-show", "1");
      var left = (X(idx) / scale) + 14;
      if (left + tip.offsetWidth > box.width) left = (X(idx) / scale) - tip.offsetWidth - 14;
      tip.style.left = left + "px";
      tip.style.top = Math.max(0, (Y(val) / scale) - 12) + "px";
    });

    hit.addEventListener("mouseleave", function () {
      cross.setAttribute("opacity", 0);
      dot.setAttribute("opacity", 0);
      tip.setAttribute("data-show", "0");
    });

    // ---- legend (always present for >= 2 series)
    var legend = $("legend");
    legend.innerHTML = "";
    function legendItem(colour, label, dashed) {
      var d = document.createElement("span");
      d.className = "legend-item";
      d.innerHTML = '<span class="legend-swatch' + (dashed ? " dashed" : "")
        + '" style="' + (dashed ? "color:" : "background:") + colour + '"></span>' + label;
      legend.appendChild(d);
    }
    legendItem(s1, "Observed");
    if (fcPts.length) legendItem(s2, "Forecast", true);
    if (thr != null) legendItem(warning, "Personalised threshold", true);
    if (floor != null) legendItem(critical, "Emergency floor", true);

    // ---- accessible description
    $("chart-desc").textContent = "Systolic blood pressure over " + n
      + " dialysis sessions, from " + hist[0].sbp + " to " + hist[n - 1].sbp + " mmHg"
      + (fcPts.length ? ", with " + fcPts.length + " forecast horizons reaching "
          + Math.round(fcPts[fcPts.length - 1].point) + " mmHg." : ".")
      + (thr != null ? " Personalised threshold " + thr + " mmHg." : "");

    // ---- table view
    var tb = $("chart-table").querySelector("tbody");
    tb.innerHTML = "";
    hist.forEach(function (d) {
      var tr = document.createElement("tr");
      tr.innerHTML = "<td>" + d.ts + "</td><td>" + Math.round(d.sbp) + "</td><td>"
        + Math.round(d.dbp) + "</td><td>Observed</td>";
      tb.appendChild(tr);
    });
    fcPts.forEach(function (p) {
      var tr = document.createElement("tr");
      tr.innerHTML = "<td>+" + p.h + " sessions</td><td>" + Math.round(p.point)
        + "</td><td>–</td><td>Forecast</td>";
      tb.appendChild(tr);
    });
  }

  /* ------------------------------------------------------------ results */

  function paint(data) {
    LAST = data;
    $("empty-state").hidden = true;
    $("output").hidden = false;

    var pers = data.personalisation || {};
    $("v-threshold").textContent = pers.threshold != null ? pers.threshold : "–";
    $("v-threshold-note").textContent = pers.offset != null
      ? ("offset " + (pers.offset > 0 ? "+" : "") + pers.offset + " mmHg vs population"
         + (pers.capped ? " — bound by the governance cap" : "")
         + " · cohort " + (pers.cohort_key || ""))
      : "";

    var tierLabel = { cold_start: "Cold start", bootstrapping: "Bootstrapping", steady: "Steady" };
    $("v-tier").textContent = tierLabel[data.confidence_tier] || data.confidence_tier;
    $("v-tier-note").textContent = (data.n_observations || 0) + " readings"
      + (data.note ? " — " + data.note : "");

    var ew = data.early_warning;
    var chip = $("v-ew-chip");
    if (!ew) {
      chip.textContent = "not issued";
      chip.className = "chip chip-muted";
      $("v-ew-note").textContent = "below the cold-start floor";
    } else {
      chip.textContent = ew.flagged ? "⚠ Flagged" : "✓ Not flagged";
      chip.className = "chip " + (ew.flagged ? "chip-critical" : "chip-good");
      $("v-ew-note").textContent = "score " + ew.score + " vs cut " + ew.cut
        + " · est. lead " + ew.est_lead_days + " d · " + ew.event_definition;
    }

    // forecast table
    var fc = (data.forecast && data.forecast.sbp) || {};
    var tb = $("forecast-table").querySelector("tbody");
    tb.innerHTML = "";
    Object.keys(fc).sort(function (a, b) { return fc[a].steps_ahead - fc[b].steps_ahead; })
      .forEach(function (k) {
        var f = fc[k], thr = pers.threshold;
        var band = (f.lo80 != null && f.hi80 != null) ? f.lo80 + " – " + f.hi80 : "–";
        var delta = thr != null ? (f.point - thr) : null;
        var verdict = delta == null ? "–"
          : (delta >= 0 ? "▲ " + delta.toFixed(1) + " over" : "▼ " + Math.abs(delta).toFixed(1) + " under");
        var tr = document.createElement("tr");
        tr.innerHTML = "<td>+" + f.steps_ahead + " sessions</td><td>" + f.point
          + "</td><td>" + band + "</td><td>" + f.days_ahead_est + "</td><td>" + verdict + "</td>";
        tb.appendChild(tr);
      });
    var withBand = Object.keys(fc).map(function (k) { return fc[k]; })
      .filter(function (f) { return f.interval_basis; })[0];
    $("interval-basis").textContent = withBand
      ? "Interval basis: " + withBand.interval_basis
      : "No conformal interval on these horizons.";

    $("v-floor").textContent = (data.emergency_floor_mmHg || "–") + " mmHg";
    $("v-pop").textContent = (POP_THRESHOLD != null ? POP_THRESHOLD + " mmHg" : "–");
    $("v-version").textContent = data.model_version || "–";

    render(data);
  }

  /* --------------------------------------------------------------- wire */

  var POP_THRESHOLD = null;

  function showError(msg) {
    var box = $("form-error");
    box.textContent = msg;
    box.hidden = !msg;
  }

  async function predict() {
    showError("");
    var parsed = parseReadings($("readings").value);
    if (parsed.errors.length) { showError(parsed.errors.slice(0, 3).join(" · ")); return; }
    if (!parsed.rows.length) { showError("Enter at least one reading."); return; }

    var btn = $("btn-predict");
    btn.disabled = true; btn.textContent = "Working…";
    document.querySelector(".results").setAttribute("aria-busy", "true");

    try {
      var res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: $("patient-id").value || "demo",
          age: Number($("age").value) || 65,
          is_male: Number($("sex").value),
          is_dm: Number($("dm").value),
          readings: parsed.rows
        })
      });
      var body = await res.json();
      if (!res.ok) { showError(body.detail || ("Request failed (" + res.status + ")")); return; }
      paint(body);
    } catch (e) {
      showError("Could not reach the API: " + e.message);
    } finally {
      btn.disabled = false; btn.textContent = "Get advisory";
      document.querySelector(".results").setAttribute("aria-busy", "false");
    }
  }

  $("btn-predict").addEventListener("click", predict);
  $("btn-sample").addEventListener("click", function () {
    $("readings").value = sampleReadings();
    updateCount();
    showError("");
  });
  $("readings").addEventListener("input", updateCount);

  // model status chip
  (async function () {
    var chip = $("model-chip");
    try {
      var h = await (await fetch("/api/health")).json();
      if (!h.model_loaded) {
        chip.textContent = "no model loaded";
        chip.className = "chip chip-critical";
        showError(h.detail || "No trained model is available. Run the training pipeline first.");
        return;
      }
      var m = await (await fetch("/api/model")).json();
      POP_THRESHOLD = m.governance ? m.governance.population_threshold_mmHg : null;
      chip.textContent = m.model_version;
      chip.className = "chip chip-good";
      chip.title = "features " + m.n_features + " · selected " + JSON.stringify(m.selected_family);
    } catch (e) {
      chip.textContent = "API unreachable";
      chip.className = "chip chip-critical";
    }
  })();

  $("readings").value = sampleReadings();
  updateCount();
})();
