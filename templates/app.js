/* Cardioplace BP Alerts — dashboard.
   All charts are plain SVG: a Space has no CDN access at render time.
   Series colours are categorical slots 1 (blue) and 2 (orange) with the reserved
   status ramp for thresholds; the pair was validated in both light and dark modes. */

(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var SVG = "http://www.w3.org/2000/svg";
  var LAST = null, SCHEMA = null, POP_THRESHOLD = null;

  function cssVar(n) {
    return getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  }
  function el(tag, attrs, text) {
    var n = document.createElementNS(SVG, tag);
    for (var k in attrs) { if (attrs[k] !== null) n.setAttribute(k, attrs[k]); }
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function esc(s) {
    return String(s === null || s === undefined ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function pretty(s) {
    return String(s || "").replace(/^RULE_/, "").replace(/_/g, " ").toLowerCase()
      .replace(/^./, function (c) { return c.toUpperCase(); });
  }

  /* ------------------------------------------------------------- theme */

  var THEME_KEY = "cardioplace-theme";
  function applyTheme(m) {
    if (m) document.documentElement.setAttribute("data-theme", m);
    else document.documentElement.removeAttribute("data-theme");
  }
  applyTheme(localStorage.getItem(THEME_KEY));
  $("theme-toggle").addEventListener("click", function () {
    var cur = document.documentElement.getAttribute("data-theme");
    var dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    var next = cur ? (cur === "dark" ? "light" : "dark") : (dark ? "light" : "dark");
    applyTheme(next); localStorage.setItem(THEME_KEY, next);
    if (LAST) paint(LAST);
  });

  /* ------------------------------------------------------- input groups */

  // Symptoms that carry an emergency/red-flag meaning get the status colour when set,
  // so a Stage-A symptom does not look like an ordinary checkbox.
  var RED = ["severeHeadache", "visualChanges", "alteredMentalStatus",
             "chestPainOrDyspnea", "focalNeuroDeficit", "severeEpigastricPain",
             "faceSwelling", "throatTightness", "syncope"];
  var LABELS = {
    severe_headache: "Severe headache", visual_changes: "Visual changes",
    altered_mental: "Altered mental state", chest_pain_dyspnea: "Chest pain / dyspnea",
    focal_neuro: "Focal neuro deficit", severe_epigastric: "Severe epigastric pain",
    new_onset_headache: "New-onset headache", ruq_pain: "RUQ pain", edema: "Edema",
    dizziness: "Dizziness", syncope: "Syncope", palpitations: "Palpitations",
    leg_swelling: "Leg swelling", fatigue: "Fatigue", sob: "Shortness of breath",
    dry_cough: "Dry cough", nsaid_use: "NSAID use", face_swelling: "Face swelling",
    throat_tightness: "Throat tightness",
    has_hf: "Heart failure", has_afib: "AFib", has_cad: "CAD", has_hcm: "HCM",
    has_dcm: "DCM", has_as: "Aortic stenosis", has_tachy: "Tachycardia",
    has_brady: "Bradycardia", dx_htn: "Hypertension",
    on_ace: "ACE inhibitor", on_arb: "ARB", on_bb: "Beta blocker",
    on_dhp_ccb: "DHP CCB", on_ndhp_ccb: "Non-DHP CCB", on_loop: "Loop diuretic",
    on_thiazide: "Thiazide", on_mra: "MRA", on_nitrate: "Nitrate",
    on_antiarr: "Antiarrhythmic", on_nsaid: "NSAID"
  };
  function label(k) { return LABELS[k] || k.replace(/_/g, " "); }

  function buildChecks(hostId, keys, counterId) {
    var host = $(hostId);
    host.innerHTML = "";
    keys.forEach(function (k) {
      var l = document.createElement("label");
      l.className = "chk" + (RED.indexOf(k) >= 0 ? " red-capable" : "");
      l.innerHTML = '<input type="checkbox" value="' + esc(k) + '">' + esc(label(k));
      l.querySelector("input").addEventListener("change", function (e) {
        l.classList.toggle("on", e.target.checked);
        if (RED.indexOf(k) >= 0) l.classList.toggle("red", e.target.checked);
        if (counterId) {
          $(counterId).textContent = host.querySelectorAll("input:checked").length;
        }
      });
      host.appendChild(l);
    });
  }
  function checked(hostId) {
    return Array.prototype.slice
      .call($(hostId).querySelectorAll("input:checked"))
      .map(function (i) { return i.value; });
  }

  /* ------------------------------------------------------- sample data */

  function sampleReadings() {
    var out = [], seed = 20260728, sbp = 138, dbp = 76, w = 74.5;
    function rnd() { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; }
    var d = new Date(Date.UTC(2026, 0, 5)), gaps = [2, 2, 3];
    for (var i = 0; i < 34; i++) {
      sbp = 0.72 * sbp + 0.28 * (140 + i * 0.75) + (rnd() - 0.5) * 13;
      dbp = 0.70 * dbp + 0.30 * (75 + i * 0.2) + (rnd() - 0.5) * 7;
      w += (rnd() - 0.45) * 0.4;
      out.push(d.toISOString().slice(0, 10) + ", " + Math.round(sbp) + ", " + Math.round(dbp)
        + ", " + Math.round(70 + (rnd() - 0.5) * 16) + ", " + w.toFixed(1));
      d = new Date(d.getTime() + gaps[i % 3] * 86400000);
    }
    return out.join("\n");
  }

  function parseReadings(raw) {
    var rows = [], errors = [];
    raw.split("\n").forEach(function (line, i) {
      line = line.trim();
      if (!line) return;
      var p = line.split(/[,\t;]+/).map(function (s) { return s.trim(); });
      if (p.length < 3) { errors.push("line " + (i + 1) + ": expected date, SBP, DBP"); return; }
      if (!/^\d{4}-\d{2}-\d{2}$/.test(p[0])) {
        errors.push("line " + (i + 1) + ": date must be YYYY-MM-DD"); return; }
      var sbp = Number(p[1]), dbp = Number(p[2]);
      if (!isFinite(sbp) || !isFinite(dbp)) {
        errors.push("line " + (i + 1) + ": SBP/DBP must be numeric"); return; }
      var r = { ts: p[0], sbp: sbp, dbp: dbp, symptoms: [] };
      if (p[3] && isFinite(Number(p[3]))) r.pulse = Number(p[3]);
      if (p[4] && isFinite(Number(p[4]))) r.weight = Number(p[4]);
      rows.push(r);
    });
    return { rows: rows, errors: errors };
  }

  function updateCount() {
    var n = parseReadings($("readings").value).rows.length;
    $("reading-count").textContent = n + (n === 1 ? " reading" : " readings")
      + (n < 7 ? " — below the 7-reading cold-start floor; the engine still fires, "
                 + "the forecaster does not" : "");
  }

  /* ==================================================== generic chart bits */

  function axes(svg, m, iw, ih, yMin, yMax, fmt, ticks) {
    var grid = cssVar("--grid"), axis = cssVar("--axis"), muted = cssVar("--text-muted");
    var Y = function (v) { return m.t + ih - ((v - yMin) / (yMax - yMin)) * ih; };
    for (var k = 0; k <= (ticks || 4); k++) {
      var v = yMin + (k / (ticks || 4)) * (yMax - yMin);
      svg.appendChild(el("line", { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v),
                                   stroke: grid, "stroke-width": 1 }));
      svg.appendChild(el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end",
                                   fill: muted, "font-size": 11 }, fmt(v)));
    }
    svg.appendChild(el("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih,
                                 stroke: axis, "stroke-width": 1 }));
    return Y;
  }

  function legendInto(hostId, items) {
    var lg = $(hostId); lg.innerHTML = "";
    items.forEach(function (it) {
      var d = document.createElement("span");
      d.className = "legend-item";
      d.innerHTML = '<span class="legend-swatch' + (it[2] ? " dashed" : "") + '" style="'
        + (it[2] ? "color:" : "background:") + it[0] + '"></span>' + esc(it[1]);
      lg.appendChild(d);
    });
  }

  function hoverLayer(svg, host, m, iw, ih, n, X, Y, valueAt, render) {
    var axis = cssVar("--axis"), surface = cssVar("--surface-1");
    var cross = el("line", { y1: m.t, y2: m.t + ih, stroke: axis, "stroke-width": 1, opacity: 0 });
    var dot = el("circle", { r: 4.5, stroke: surface, "stroke-width": 2, opacity: 0 });
    svg.appendChild(cross); svg.appendChild(dot);
    var hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" });
    svg.appendChild(hit);
    var tip = document.createElement("div");
    tip.className = "tooltip"; host.appendChild(tip);

    hit.addEventListener("mousemove", function (ev) {
      var box = svg.getBoundingClientRect(), sc = 900 / box.width;
      var i = Math.round((((ev.clientX - box.left) * sc - m.l) / iw) * (n - 1));
      i = Math.max(0, Math.min(n - 1, i));
      var v = valueAt(i);
      cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i));
      cross.setAttribute("opacity", 0.7);
      dot.setAttribute("cx", X(i)); dot.setAttribute("cy", Y(v.y));
      dot.setAttribute("fill", v.colour); dot.setAttribute("opacity", 1);
      tip.innerHTML = render(i);
      tip.setAttribute("data-show", "1");
      var left = (X(i) / sc) + 14;
      if (left + tip.offsetWidth > box.width) left = (X(i) / sc) - tip.offsetWidth - 14;
      tip.style.left = left + "px";
      tip.style.top = Math.max(0, (Y(v.y) / sc) - 12) + "px";
    });
    hit.addEventListener("mouseleave", function () {
      cross.setAttribute("opacity", 0); dot.setAttribute("opacity", 0);
      tip.setAttribute("data-show", "0");
    });
  }

  /* ======================================= Model 1: forecast chart */

  function renderForecast(data) {
    var host = $("chart"); host.innerHTML = "";
    var hist = data.history || []; if (!hist.length) return;

    var s1 = cssVar("--series-1"), s2 = cssVar("--series-2");
    var muted = cssVar("--text-muted"), secondary = cssVar("--text-secondary");
    var warning = cssVar("--status-warning"), critical = cssVar("--status-critical");
    var surface = cssVar("--surface-1");

    var fc = (data.forecast && data.forecast.sbp) || {};
    var pts = Object.keys(fc).map(function (k) {
      return { h: fc[k].steps_ahead, point: fc[k].point, lo: fc[k].lo80, hi: fc[k].hi80 };
    }).sort(function (a, b) { return a.h - b.h; });

    var thr = data.personalisation ? data.personalisation.threshold : null;
    var floor = data.emergency_floor_mmHg;

    var W = 900, H = 320, m = { t: 18, r: 78, b: 34, l: 46 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var n = hist.length, total = n + pts.length;

    var vals = hist.map(function (d) { return d.sbp; });
    pts.forEach(function (p) { vals.push(p.point);
      if (p.lo != null) vals.push(p.lo); if (p.hi != null) vals.push(p.hi); });
    if (thr != null) vals.push(thr);
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    if (floor != null && hi > floor - 12) vals.push(floor);
    lo = Math.min.apply(null, vals); hi = Math.max.apply(null, vals);
    var pad = Math.max(6, (hi - lo) * 0.12);
    var yMin = Math.floor((lo - pad) / 10) * 10, yMax = Math.ceil((hi + pad) / 10) * 10;

    var X = function (i) { return m.l + (total <= 1 ? 0 : (i / (total - 1)) * iw); };
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "xMidYMid meet" });
    var Y = axes(svg, m, iw, ih, yMin, yMax, function (v) { return String(Math.round(v)); },
                 Math.min(6, Math.round((yMax - yMin) / 10)));

    [0, Math.floor(n / 3), Math.floor(2 * n / 3), n - 1].concat(pts.length ? [total - 1] : [])
      .filter(function (t, i, a) { return a.indexOf(t) === i && t >= 0; })
      .forEach(function (i) {
        svg.appendChild(el("text", { x: X(i), y: m.t + ih + 20, "text-anchor": "middle",
          fill: muted, "font-size": 11 },
          i < n ? hist[i].ts.slice(5) : "+" + pts[i - n].h + " sess"));
      });

    function ref(v, colour, text) {
      if (v == null || v < yMin || v > yMax) return;
      svg.appendChild(el("line", { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v),
        stroke: colour, "stroke-width": 2, "stroke-dasharray": "5 4", opacity: .9 }));
      svg.appendChild(el("text", { x: m.l + iw + 6, y: Y(v) + 4, fill: secondary,
        "font-size": 11 }, text));
    }
    ref(thr, warning, "threshold " + thr);
    ref(floor, critical, "floor " + floor);

    pts.forEach(function (p, i) {
      if (p.lo == null || p.hi == null) return;
      svg.appendChild(el("line", { x1: X(n + i), x2: X(n + i), y1: Y(p.hi), y2: Y(p.lo),
        stroke: s2, "stroke-width": 8, opacity: .2, "stroke-linecap": "round" }));
    });

    svg.appendChild(el("path", { d: hist.map(function (p, i) {
      return (i ? "L" : "M") + X(i) + " " + Y(p.sbp); }).join(" "),
      fill: "none", stroke: s1, "stroke-width": 2, "stroke-linejoin": "round",
      "stroke-linecap": "round" }));

    if (pts.length) {
      svg.appendChild(el("path", { d: "M" + X(n - 1) + " " + Y(hist[n - 1].sbp) + " "
        + pts.map(function (p, i) { return "L" + X(n + i) + " " + Y(p.point); }).join(" "),
        fill: "none", stroke: s2, "stroke-width": 2, "stroke-dasharray": "6 4",
        "stroke-linecap": "round" }));
      pts.forEach(function (p, i) {
        svg.appendChild(el("circle", { cx: X(n + i), cy: Y(p.point), r: 5, fill: s2,
          stroke: surface, "stroke-width": 2 }));
      });
      var last = pts[pts.length - 1];
      svg.appendChild(el("text", { x: X(total - 1), y: Y(last.point) - 12,
        "text-anchor": "middle", fill: secondary, "font-size": 11, "font-weight": 600 },
        String(Math.round(last.point))));
    }
    svg.appendChild(el("circle", { cx: X(n - 1), cy: Y(hist[n - 1].sbp), r: 4.5, fill: s1,
      stroke: surface, "stroke-width": 2 }));
    host.appendChild(svg);

    hoverLayer(svg, host, m, iw, ih, total, X, Y,
      function (i) { return i < n ? { y: hist[i].sbp, colour: s1 }
                                  : { y: pts[i - n].point, colour: s2 }; },
      function (i) {
        if (i < n) return '<div class="tt-date">' + hist[i].ts + "</div>"
          + '<div class="tt-row"><span class="tt-dot" style="background:' + s1
          + '"></span>SBP <span class="tt-val">' + Math.round(hist[i].sbp) + "</span></div>"
          + '<div class="tt-row"><span class="tt-dot" style="background:' + muted
          + '"></span>DBP <span class="tt-val">' + Math.round(hist[i].dbp) + "</span></div>";
        var p = pts[i - n];
        return '<div class="tt-date">forecast +' + p.h + " sessions</div>"
          + '<div class="tt-row"><span class="tt-dot" style="background:' + s2
          + '"></span>SBP <span class="tt-val">' + Math.round(p.point) + "</span></div>";
      });

    var items = [[s1, "Observed", false]];
    if (pts.length) items.push([s2, "Forecast", true]);
    if (thr != null) items.push([warning, "Personalised threshold", true]);
    if (floor != null) items.push([critical, "Emergency floor", true]);
    legendInto("legend", items);

    $("chart-desc").textContent = "Systolic blood pressure across " + n + " sessions"
      + (pts.length ? ", with " + pts.length + " forecast horizons." : ".");
  }

  /* ======================================= backtest: predicted vs actual */

  function renderBacktest(bt) {
    var panel = $("backtest-panel"), host = $("backtest-chart");
    host.innerHTML = "";
    if (!bt || !bt.horizons || !bt.horizons.length) { panel.hidden = true; return; }
    panel.hidden = false;

    var first = bt.horizons[0];
    var rows = bt.series["h" + first.horizon] || [];
    if (rows.length < 2) { panel.hidden = true; return; }

    var s1 = cssVar("--series-1"), s2 = cssVar("--series-2");
    var muted = cssVar("--text-muted"), surface = cssVar("--surface-1");

    var W = 900, H = 280, m = { t: 18, r: 20, b: 32, l: 46 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b, n = rows.length;
    var vals = rows.map(function (r) { return r.actual; })
      .concat(rows.map(function (r) { return r.predicted; }));
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    var pad = Math.max(5, (hi - lo) * 0.12);
    var yMin = Math.floor((lo - pad) / 10) * 10, yMax = Math.ceil((hi + pad) / 10) * 10;

    var X = function (i) { return m.l + (n <= 1 ? 0 : (i / (n - 1)) * iw); };
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "xMidYMid meet" });
    var Y = axes(svg, m, iw, ih, yMin, yMax, function (v) { return String(Math.round(v)); }, 4);

    [0, Math.floor(n / 3), Math.floor(2 * n / 3), n - 1]
      .filter(function (t, i, a) { return a.indexOf(t) === i; })
      .forEach(function (i) {
        svg.appendChild(el("text", { x: X(i), y: m.t + ih + 19, "text-anchor": "middle",
          fill: muted, "font-size": 11 }, rows[i].ts.slice(5)));
      });

    svg.appendChild(el("path", { d: rows.map(function (r, i) {
      return (i ? "L" : "M") + X(i) + " " + Y(r.actual); }).join(" "),
      fill: "none", stroke: s1, "stroke-width": 2, "stroke-linejoin": "round" }));
    svg.appendChild(el("path", { d: rows.map(function (r, i) {
      return (i ? "L" : "M") + X(i) + " " + Y(r.predicted); }).join(" "),
      fill: "none", stroke: s2, "stroke-width": 2, "stroke-dasharray": "6 4",
      "stroke-linejoin": "round" }));

    // worst miss, directly labelled -- the number a reviewer looks for
    var worst = rows.reduce(function (a, b) {
      return Math.abs(b.error) > Math.abs(a.error) ? b : a; }, rows[0]);
    var wi = rows.indexOf(worst);
    svg.appendChild(el("circle", { cx: X(wi), cy: Y(worst.actual), r: 4.5,
      fill: s1, stroke: surface, "stroke-width": 2 }));
    svg.appendChild(el("text", { x: X(wi), y: Y(worst.actual) - 11, "text-anchor": "middle",
      fill: cssVar("--text-secondary"), "font-size": 11, "font-weight": 600 },
      (worst.error > 0 ? "+" : "") + worst.error));

    host.appendChild(svg);
    hoverLayer(svg, host, m, iw, ih, n, X, Y,
      function (i) { return { y: rows[i].actual, colour: s1 }; },
      function (i) {
        var r = rows[i];
        return '<div class="tt-date">' + r.ts + "</div>"
          + '<div class="tt-row"><span class="tt-dot" style="background:' + s1
          + '"></span>actual <span class="tt-val">' + r.actual + "</span></div>"
          + '<div class="tt-row"><span class="tt-dot" style="background:' + s2
          + '"></span>predicted <span class="tt-val">' + r.predicted + "</span></div>"
          + '<div class="tt-row"><span class="tt-dot" style="background:' + muted
          + '"></span>miss <span class="tt-val">' + (r.error > 0 ? "+" : "") + r.error
          + "</span></div>";
      });

    legendInto("backtest-legend", [[s1, "Actual", false],
      [s2, "Predicted " + first.days_ahead + " days earlier", true]]);
    $("backtest-sub").textContent = "Each point is the forecast made " + first.horizon
      + " session" + (first.horizon === 1 ? "" : "s") + " (~" + first.days_ahead
      + " days) before that reading — the features behind it never saw the day they predict.";
    $("backtest-desc").textContent = "Actual versus previously forecast SBP over "
      + n + " sessions, mean absolute error " + first.mae + " mmHg.";

    var tb = $("backtest-table").querySelector("tbody"); tb.innerHTML = "";
    bt.horizons.forEach(function (h) {
      var tr = document.createElement("tr");
      tr.innerHTML = "<td>+" + h.horizon + " sessions</td><td>" + h.days_ahead
        + "</td><td>" + h.n + "</td><td>" + h.mae + "</td><td>"
        + Math.round(h.within_10 * 100) + "%</td>";
      tb.appendChild(tr);
    });
  }

  /* ======================================= Model 3: anomaly chart */

  function renderAnomaly(an) {
    var host = $("anomaly-chart"), panel = $("anomaly-panel");
    host.innerHTML = "";
    if (!an || !an.points || an.points.length < 2) { panel.hidden = true; return; }
    panel.hidden = false;

    var pts = an.points, cut = an.cut;
    var s1 = cssVar("--series-1"), muted = cssVar("--text-muted");
    var secondary = cssVar("--text-secondary"), warning = cssVar("--status-warning");
    var critical = cssVar("--status-critical"), surface = cssVar("--surface-1");

    var W = 900, H = 250, m = { t: 16, r: 74, b: 32, l: 52 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b, n = pts.length;

    // The cut is always inside the domain: the question is how much headroom is left,
    // which is unreadable if the line autoscales to itself and the cut sits off-canvas.
    var vals = pts.map(function (p) { return p.score; }).concat([cut]);
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    var pad = Math.max((hi - lo) * 0.15, 0.01);
    var yMin = lo - pad, yMax = hi + pad;

    var X = function (i) { return m.l + (n <= 1 ? 0 : (i / (n - 1)) * iw); };
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "xMidYMid meet" });
    var Y = axes(svg, m, iw, ih, yMin, yMax, function (v) { return v.toFixed(2); }, 4);

    var warm = pts.filter(function (p) { return p.warmup; }).length;
    if (warm > 0) {
      svg.appendChild(el("rect", { x: m.l, y: m.t, width: Math.max(X(warm - 1) - m.l, 2),
        height: ih, fill: muted, opacity: .09 }));
      svg.appendChild(el("text", { x: m.l + 5, y: m.t + 13, fill: muted,
        "font-size": 10 }, "warm-up"));
    }
    svg.appendChild(el("line", { x1: m.l, x2: m.l + iw, y1: Y(cut), y2: Y(cut),
      stroke: warning, "stroke-width": 2, "stroke-dasharray": "5 4" }));
    svg.appendChild(el("text", { x: m.l + iw + 6, y: Y(cut) + 4, fill: secondary,
      "font-size": 11 }, "cut " + cut.toFixed(3)));

    [0, Math.floor(n / 3), Math.floor(2 * n / 3), n - 1]
      .filter(function (t, i, a) { return a.indexOf(t) === i; })
      .forEach(function (i) {
        svg.appendChild(el("text", { x: X(i), y: m.t + ih + 19, "text-anchor": "middle",
          fill: muted, "font-size": 11 }, pts[i].ts.slice(5)));
      });

    svg.appendChild(el("path", { d: pts.map(function (p, i) {
      return (i ? "L" : "M") + X(i) + " " + Y(p.score); }).join(" "),
      fill: "none", stroke: s1, "stroke-width": 2, "stroke-linejoin": "round" }));
    pts.forEach(function (p, i) {
      if (!p.flagged) return;
      svg.appendChild(el("circle", { cx: X(i), cy: Y(p.score), r: 5, fill: critical,
        stroke: surface, "stroke-width": 2 }));
    });
    var peak = pts.reduce(function (a, b) { return b.score > a.score ? b : a; }, pts[0]);
    var pi = pts.indexOf(peak);
    svg.appendChild(el("text", { x: X(pi), y: Y(peak.score) - 11, "text-anchor": "middle",
      fill: secondary, "font-size": 11, "font-weight": 600 }, peak.score.toFixed(3)));

    host.appendChild(svg);
    hoverLayer(svg, host, m, iw, ih, n, X, Y,
      function (i) { return { y: pts[i].score, colour: pts[i].flagged ? critical : s1 }; },
      function (i) {
        var p = pts[i];
        return '<div class="tt-date">' + p.ts + (p.warmup ? " · warm-up" : "") + "</div>"
          + '<div class="tt-row"><span class="tt-dot" style="background:'
          + (p.flagged ? critical : s1) + '"></span>score <span class="tt-val">'
          + p.score.toFixed(4) + "</span></div>"
          + '<div class="tt-row"><span class="tt-dot" style="background:' + warning
          + '"></span>' + (p.score >= cut ? "over" : "under") + ' by <span class="tt-val">'
          + Math.abs(p.score - cut).toFixed(4) + "</span></div>";
      });

    legendInto("anomaly-legend", [[s1, "Detector score", false],
      [warning, "Alert cut", true], [critical, "Flagged", false]]);
    $("anomaly-sub").textContent = an.event_definition;
    $("anomaly-headroom").textContent = an.n_flagged > 0
      ? an.n_flagged + " of " + an.n_settled + " settled sessions crossed the cut. Peak "
        + peak.score.toFixed(3) + " on " + peak.ts + "."
      : "No session crossed the cut. Peak " + peak.score.toFixed(3) + " on " + peak.ts
        + ", leaving " + (cut - peak.score).toFixed(3) + " of headroom at the "
        + an.budget_pct + "% alert budget.";
    $("anomaly-desc").textContent = "Detector score across " + n + " sessions against a cut of "
      + cut.toFixed(3) + "; " + an.n_flagged + " flagged.";
  }

  /* ======================================= rule engine + combined */

  function renderEngine(eng) {
    if (!eng || eng.error) {
      $("engine-note").textContent = eng ? ("Rule engine unavailable: " + eng.error) : "";
      return;
    }
    var cur = eng.current || {};
    var chip = $("v-engine-tier");
    chip.textContent = cur.tier ? pretty(cur.tier) : "No alert";
    chip.className = "chip " + (cur.is_emergency ? "chip-critical"
      : cur.fired ? "chip-warning" : "chip-good");
    $("v-engine-note").textContent = cur.rule_id
      ? pretty(cur.rule_id) + (cur.mode === "PERSONALIZED" ? " · personalised" : "")
      : (cur.gate_reason && cur.gate_reason !== "NONE"
          ? "gated: " + pretty(cur.gate_reason) : "nothing fired on this reading");

    var tl = $("engine-timeline"); tl.innerHTML = "";
    (eng.history || []).forEach(function (h) {
      var d = document.createElement("span");
      d.className = "tl" + (h.tier ? " tl-" + h.tier : "");
      d.title = h.ts + (h.tier ? " · " + pretty(h.tier) + " · " + pretty(h.rule_id) : " · no alert");
      tl.appendChild(d);
    });

    $("engine-count").textContent = eng.fired_count + " of "
      + (eng.history || []).length + " readings fired";

    var tb = $("engine-table").querySelector("tbody"); tb.innerHTML = "";
    (eng.history || []).slice().reverse().filter(function (h) { return h.fired; })
      .slice(0, 12).forEach(function (h) {
        var tr = document.createElement("tr");
        tr.innerHTML = "<td>" + esc(h.ts) + "</td><td>" + esc(pretty(h.tier))
          + "</td><td>" + esc(pretty(h.rule_id)) + "</td>";
        tb.appendChild(tr);
      });

    var counts = Object.keys(eng.tier_counts || {}).map(function (k) {
      return pretty(k) + " " + eng.tier_counts[k]; }).join(" · ");
    var blocked = SCHEMA && SCHEMA.rules ? SCHEMA.rules.blocked_on_corpus : null;
    $("engine-note").textContent = (counts ? counts + ". " : "")
      + (blocked ? blocked + " of " + SCHEMA.rules.total_slots + " rule slots are "
          + "BLOCKED_ON_INPUTS against the training corpus, which carries no symptom, "
          + "medication or condition columns. The engine is deterministic, so whatever "
          + "you enter above is evaluated here immediately." : "");
  }

  function renderPredicted(pa) {
    var host = $("predicted-alerts"); host.innerHTML = "";
    if (!pa || !pa.horizons || !pa.horizons.length) {
      host.innerHTML = '<p class="hint">No forecast issued — below the cold-start floor.</p>';
      $("predicted-note").textContent = pa ? pa.symptom_note : "";
      return;
    }
    pa.horizons.forEach(function (h) {
      var card = document.createElement("div");
      card.className = "horizon";
      var chipCls = h.is_emergency ? "chip-critical" : h.fired ? "chip-warning" : "chip-good";
      card.innerHTML =
        '<p class="horizon-when">in ~' + esc(h.days_ahead) + ' days · +'
          + esc(h.steps_ahead) + ' sessions</p>'
        + '<p class="horizon-bp">' + esc(h.sbp)
          + (h.dbp != null ? ' / ' + esc(h.dbp) : '') + '<span class="unit">mmHg</span></p>'
        + '<p class="horizon-band">' + (h.lo80 != null
            ? '80% ' + esc(h.lo80) + ' – ' + esc(h.hi80) : 'no interval') + '</p>'
        + '<span class="chip ' + chipCls + '">' + esc(h.tier ? pretty(h.tier) : "No alert")
          + '</span>'
        + (h.rule_id ? '<p class="horizon-rule">' + esc(pretty(h.rule_id)) + '</p>' : '');
      host.appendChild(card);
    });
    $("predicted-note").textContent = pa.basis + ". " + pa.symptom_note;
  }

  /* ------------------------------------------------------------ banner */

  function renderBanner(d) {
    var pers = d.personalisation || {}, ew = d.early_warning || {};
    var eng = (d.rule_engine || {}).current || {};
    var pa = (d.predicted_alert || {}).horizons || [];
    var firstFire = pa.filter(function (h) { return h.fired; })[0];

    var cls, icon, title, detail;
    if (eng.is_emergency) {
      cls = "banner-critical"; icon = "⚠";
      title = "Rule engine fired an emergency alert on the latest reading";
      detail = pretty(eng.rule_id) + " — the emergency floor is never personalised and "
        + "the engine is authoritative here.";
    } else if (ew.flagged) {
      cls = "banner-critical"; icon = "⚠";
      title = "Early-warning detector flagged this patient";
      detail = "Score " + ew.score + " at or above the " + ew.budget_pct + "% cut of "
        + ew.cut + ", about " + ew.est_lead_days + " days of lead time.";
    } else if (firstFire) {
      cls = "banner-watch"; icon = "◆";
      title = pretty(firstFire.tier) + " forecast in about " + firstFire.days_ahead + " days";
      detail = "Predicted SBP " + firstFire.sbp + " mmHg would fire "
        + pretty(firstFire.rule_id) + " if the trend holds.";
    } else if (eng.fired) {
      cls = "banner-watch"; icon = "◆";
      title = pretty(eng.tier) + " on the latest reading";
      detail = pretty(eng.rule_id) + ". No further breach forecast in the next horizons.";
    } else if (d.confidence_tier === "cold_start") {
      cls = ""; icon = "○"; title = "Cold start — no forecast issued";
      detail = d.note || "";
    } else {
      cls = "banner-good"; icon = "✓"; title = "Nothing due";
      detail = "The engine fired nothing on the latest reading, the forecast stays below "
        + "the personalised threshold of " + pers.threshold + " mmHg, and the detector is "
        + "below its cut.";
    }
    $("banner").className = "banner " + cls;
    $("banner-icon").textContent = icon;
    $("banner-title").textContent = title;
    $("banner-detail").textContent = detail;
    $("banner-meta").textContent = (d.n_observations || 0) + " readings · "
      + (d.latency_ms != null ? d.latency_ms + " ms" : "");
  }

  /* ------------------------------------------------------------ results */

  function paint(d) {
    LAST = d;
    $("empty-state").hidden = true;
    $("output").hidden = false;

    var pers = d.personalisation || {};
    $("v-threshold").textContent = pers.threshold != null ? pers.threshold : "–";
    $("v-threshold-note").textContent = pers.offset != null
      ? "offset " + (pers.offset > 0 ? "+" : "") + pers.offset + " mmHg"
        + (pers.capped ? " — bound by the governance cap" : "")
        + " · cohort " + (pers.cohort_key || "") : "";

    var tl = { cold_start: "Cold start", bootstrapping: "Bootstrapping", steady: "Steady" };
    $("v-tier").textContent = tl[d.confidence_tier] || d.confidence_tier;
    $("v-tier-note").textContent = (d.n_observations || 0) + " readings"
      + (d.note ? " — " + d.note : "");

    var ew = d.early_warning, chip = $("v-ew-chip"), meter = $("v-ew-meter");
    if (!ew) {
      chip.textContent = "not issued"; chip.className = "chip chip-muted";
      meter.hidden = true; $("v-ew-note").textContent = "below the cold-start floor";
    } else {
      chip.textContent = ew.flagged ? "⚠ Flagged" : "✓ Not flagged";
      chip.className = "chip " + (ew.flagged ? "chip-critical" : "chip-good");
      meter.hidden = false;
      var pct = Math.max(2, Math.min(100, (ew.score / ew.cut) * 80));
      var fill = $("v-ew-fill");
      fill.style.width = pct + "%";
      fill.className = "meter-fill" + (ew.flagged ? " over" : "");
      $("v-ew-score").textContent = "score " + ew.score;
      $("v-ew-cut").textContent = ew.cut;
      $("v-ew-note").textContent = "est. lead " + ew.est_lead_days + " d";
    }

    var fc = (d.forecast && d.forecast.sbp) || {};
    var tb = $("forecast-table").querySelector("tbody"); tb.innerHTML = "";
    Object.keys(fc).sort(function (a, b) { return fc[a].steps_ahead - fc[b].steps_ahead; })
      .forEach(function (k) {
        var f = fc[k], thr = pers.threshold;
        var delta = thr != null ? (f.point - thr) : null;
        var tr = document.createElement("tr");
        tr.innerHTML = "<td>+" + f.steps_ahead + " sessions</td><td>" + f.point + "</td><td>"
          + (f.lo80 != null ? f.lo80 + " – " + f.hi80 : "–") + "</td><td>"
          + f.days_ahead_est + "</td><td>"
          + (delta == null ? "–" : (delta >= 0 ? "▲ " + delta.toFixed(1) + " over"
              : "▼ " + Math.abs(delta).toFixed(1) + " under")) + "</td>";
        tb.appendChild(tr);
      });
    var band = Object.keys(fc).map(function (k) { return fc[k]; })
      .filter(function (f) { return f.interval_basis; })[0];
    $("interval-basis").textContent = band ? "Interval basis: " + band.interval_basis
      : "No conformal interval on these horizons.";

    $("v-floor").textContent = (d.emergency_floor_mmHg || "–") + " mmHg";
    $("v-pop").textContent = POP_THRESHOLD != null ? POP_THRESHOLD + " mmHg" : "–";
    $("v-version").textContent = d.model_version || "–";

    renderBanner(d);
    renderPredicted(d.predicted_alert);
    renderEngine(d.rule_engine);
    renderForecast(d);
    renderBacktest(d.backtest);
    renderAnomaly(d.anomaly);
  }

  /* --------------------------------------------------------------- wire */

  function showError(msg) {
    var b = $("form-error"); b.textContent = msg; b.hidden = !msg;
  }

  async function predict() {
    showError("");
    var parsed = parseReadings($("readings").value);
    if (parsed.errors.length) { showError(parsed.errors.slice(0, 3).join(" · ")); return; }
    if (!parsed.rows.length) { showError("Enter at least one reading."); return; }

    // symptoms and position attach to the most recent reading
    var rows = parsed.rows;
    rows[rows.length - 1].symptoms = checked("symptoms");
    rows[rows.length - 1].position = $("position").value;
    rows[rows.length - 1].n_meas = Number($("n-meas").value) || 2;

    var pt = $("provider-target").value;
    var body = {
      patient_id: $("patient-id").value || "demo",
      profile: {
        age: Number($("age").value) || 65,
        is_male: Number($("sex").value),
        is_dm: Number($("dm").value),
        is_pregnant: Number($("pregnant").value),
        hf_type: $("hf-type").value,
        conditions: checked("conditions"),
        medications: checked("medications"),
        provider_target: pt === "" ? null : Number(pt),
        missed_3d: Number($("missed-3d").value) || 0,
        adherence_7d: Number($("adherence").value)
      },
      readings: rows
    };
    if (body.profile.hf_type !== "NONE"
        && body.profile.conditions.indexOf("has_hf") < 0) {
      body.profile.conditions.push("has_hf");
    }

    var btn = $("btn-predict");
    btn.disabled = true; btn.textContent = "Working…";
    document.querySelector(".results").setAttribute("aria-busy", "true");
    try {
      var res = await fetch("/api/predict", { method: "POST",
        headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      var out = await res.json();
      if (!res.ok) { showError(out.detail || ("Request failed (" + res.status + ")")); return; }
      paint(out);
    } catch (e) {
      showError("Could not reach the API: " + e.message);
    } finally {
      btn.disabled = false; btn.textContent = "Get advisory";
      document.querySelector(".results").setAttribute("aria-busy", "false");
    }
  }

  /* ---------------------------------------------------------- training */

  var pollTimer = null;
  function renderTrain(s) {
    var chip = $("train-state");
    if (s.running) {
      chip.textContent = "running"; chip.className = "chip chip-warning";
      $("train-note").textContent = "Started " + (s.started_at || "") + ". A full run "
        + "streams 250 MB and does a forward-chained search — expect tens of minutes.";
    } else if (s.returncode === 0) {
      chip.textContent = "finished"; chip.className = "chip chip-good";
      $("train-note").textContent = "Completed " + (s.finished_at || "")
        + ". The new model was loaded automatically.";
    } else if (s.returncode != null) {
      chip.textContent = "failed (" + s.returncode + ")";
      chip.className = "chip chip-critical";
      $("train-note").textContent = "Pipeline exited non-zero — see the log below.";
    } else {
      chip.textContent = "idle"; chip.className = "chip chip-muted";
      $("train-note").textContent = "";
    }
    $("train-log").textContent = (s.tail || []).join("\n");
    $("train-log").scrollTop = $("train-log").scrollHeight;
  }

  async function pollTrain() {
    try {
      var s = await (await fetch("/api/train/status")).json();
      renderTrain(s);
      if (!s.running && pollTimer) { clearInterval(pollTimer); pollTimer = null; refreshModel(); }
    } catch (e) { /* transient */ }
  }

  $("btn-train").addEventListener("click", async function () {
    $("train-panel").hidden = false;
    if (!confirm("Run the full training pipeline?\n\nThis streams the 250 MB corpus and "
        + "runs a forward-chained random search. It takes tens of minutes and is heavy "
        + "for a free Space — training locally is usually the better choice.")) return;
    try {
      var res = await fetch("/api/train", { method: "POST" });
      var out = await res.json();
      if (!res.ok) { renderTrain({ running: false, tail: [out.detail || "failed to start"] }); return; }
      if (!pollTimer) pollTimer = setInterval(pollTrain, 3000);
      pollTrain();
    } catch (e) {
      renderTrain({ running: false, tail: ["could not start: " + e.message] });
    }
  });
  $("train-close").addEventListener("click", function () { $("train-panel").hidden = true; });

  /* -------------------------------------------------------------- boot */

  async function refreshModel() {
    var chip = $("model-chip");
    try {
      var h = await (await fetch("/api/health")).json();
      if (h.training && h.training.running) {
        $("train-panel").hidden = false;
        renderTrain(h.training);
        if (!pollTimer) pollTimer = setInterval(pollTrain, 3000);
      }
      if (!h.model_loaded) {
        chip.textContent = "no model loaded"; chip.className = "chip chip-critical";
        showError((h.detail || "No trained model is available.")
          + " Use Train model, or run python main.py locally.");
        return;
      }
      var m = await (await fetch("/api/model")).json();
      POP_THRESHOLD = m.governance ? m.governance.population_threshold_mmHg : null;
      chip.textContent = m.model_version;
      chip.className = "chip chip-good";
      chip.title = "features " + m.n_features + " · " + JSON.stringify(m.selected_family);
    } catch (e) {
      chip.textContent = "API unreachable"; chip.className = "chip chip-critical";
    }
  }

  (async function boot() {
    try {
      SCHEMA = await (await fetch("/api/schema")).json();
      buildChecks("symptoms", SCHEMA.symptoms, null);
      buildChecks("conditions", SCHEMA.conditions, "cond-count");
      buildChecks("medications", SCHEMA.medications, "med-count");
    } catch (e) {
      $("symptoms").innerHTML = '<p class="hint">Could not load the field vocabulary.</p>';
    }
    $("readings").value = sampleReadings();
    updateCount();
    refreshModel();
  })();

  $("btn-predict").addEventListener("click", predict);
  $("btn-sample").addEventListener("click", function () {
    $("readings").value = sampleReadings(); updateCount(); showError("");
  });
  $("readings").addEventListener("input", updateCount);
})();
