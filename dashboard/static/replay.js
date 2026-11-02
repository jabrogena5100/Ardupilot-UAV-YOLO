/* replay.js — browser side of dashboard/replay_server.py.
   Every number drawn comes from a saved file via /api/run. Where a file is missing the
   view says so and draws nothing: no interpolation, no defaults. */
"use strict";

const ARM_COLOR = { baseline: "#9aa7b4", A0: "#4a9eff", B2s0: "#c77dff" };
const ARM_ORDER = ["baseline", "A0", "B2s0"];
const $ = (id) => document.getElementById(id);

const S = {
  list: null, mode: "replay", sel: null, cmp: null, // cmp = {pilot, scenario}
  cache: new Map(), t: 0, playing: false, tmax: 600,
};

async function api(url) {
  const E = window.__REPLAY_EMBEDDED__;  // set only by dashboard/replay_export.py (standalone file)
  if (E) {
    const u = new URL(url, "http://x");
    if (u.pathname === "/api/runs") return E.runs;
    const d = E.run[u.searchParams.get("id")];
    if (!d) throw new Error("run not embedded in this file: " + u.searchParams.get("id"));
    return d;
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " -> " + r.status);
  return r.json();
}
async function getRun(id) {
  if (!S.cache.has(id)) S.cache.set(id, await api("/api/run?id=" + encodeURIComponent(id)));
  return S.cache.get(id);
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (x, d = 1) => (x === null || x === undefined ? null : Number(x).toFixed(d));
const na = (why) => `<td class="na" title="${esc(why)}">n/a</td>`;

/* ---------------- run list ---------------- */

function runName(r) {
  if (r.diagnostic) return "post-fix baseline flight (diagnostic)";
  return (r.label || r.tag) + " · " + (r.scenario ? "scenario " + r.scenario : "");
}

function renderList() {
  const L = S.list, el = $("runlist");
  let h = "";
  const sections = [["completed", "Completed"], ["interrupted", "Interrupted / incomplete"]];
  for (const [st, title] of sections) {
    const rs = L.runs.filter((r) => r.status === st);
    h += `<h3>${title} (${rs.length})</h3>`;
    for (const p of Object.keys(L.pilots)) {
      const g = rs.filter((r) => r.pilot === p);
      if (!g.length) continue;
      h += `<div class="grp" title="${esc(L.pilots[p].note)}">${esc(L.pilots[p].label)}</div>`;
      if (st === "completed") {
        const scns = [...new Set(g.filter((r) => !r.diagnostic).map((r) => r.scenario))];
        for (const s of scns) h += `<button class="run" data-cmp="${p}|${s}">▸ compare arms, scenario ${s}</button>`;
      }
      for (const r of g) {
        const sub = st === "completed"
          ? (r.arm === "baseline" || r.diagnostic ? "no detector" : `TP ${r.det_true} · FP ${r.det_false}`) +
            (r.has_telemetry ? "" : " · no telemetry copy")
          : esc(r.why || "");
        h += `<button class="run ${S.sel === r.id ? "sel" : ""}" data-id="${esc(r.id)}">${esc(runName(r))}
          <span class="tag ${st === "completed" ? "ok" : "bad"}">${st === "completed" ? "done" : "partial"}</span>
          <span class="sub">${sub}</span></button>`;
      }
    }
  }
  h += `<h3>Not run yet (${L.planned.length})</h3>`;
  for (const q of L.planned)
    h += `<div class="run gone">${esc(q.label)} · scenario ${esc(q.scenario)}<span class="tag warn">pending</span>
      <span class="sub">${esc(L.pilots[q.pilot].label)}: in matrix, no data exists</span></div>`;
  if (L.discarded.length) {
    h += `<h3>Discarded attempts</h3>`;
    for (const d of L.discarded) h += `<div class="run gone">${esc(d.path.split("/").pop())}<span class="sub">${esc(d.why)}</span></div>`;
  }
  el.innerHTML = h;
  el.querySelectorAll("button.run[data-id]").forEach((b) => (b.onclick = () => { setMode("replay"); select(b.dataset.id); }));
  el.querySelectorAll("button.run[data-cmp]").forEach((b) => (b.onclick = () => {
    const [pilot, scenario] = b.dataset.cmp.split("|");
    S.cmp = { pilot, scenario }; setMode("compare"); refresh();
  }));
}

function setMode(m) {
  S.mode = m;
  $("tab-replay").classList.toggle("on", m === "replay");
  $("tab-compare").classList.toggle("on", m === "compare");
}

async function select(id) { S.sel = id; renderList(); await refresh(); }

/* ---------------- what to show ---------------- */

// [{key, title, run: data | null, why}]
async function panels() {
  if (S.mode === "replay") {
    if (!S.sel) return [];
    const d = await getRun(S.sel);
    return [{ key: S.sel, title: runName(d.desc), run: d, arm: d.meta.arm }];
  }
  if (!S.cmp) return [];
  const { pilot, scenario } = S.cmp, out = [];
  for (const arm of ARM_ORDER) {
    const title = S.list.arm_label[arm];
    const cands = S.list.runs.filter((r) => r.pilot === pilot && r.scenario === scenario && r.arm === arm && !r.diagnostic);
    const ok = cands.find((r) => r.status === "completed");
    if (ok) { out.push({ key: ok.id, title, run: await getRun(ok.id), arm }); continue; }
    const bad = cands[0], planned = S.list.planned.find((q) => q.pilot === pilot && q.scenario === scenario && q.arm === arm);
    out.push({ key: pilot + arm, title, run: null, arm,
      why: bad ? "interrupted: " + bad.why : planned ? "not run: " + planned.why : "no run found on disk" });
  }
  return out;
}

async function refresh() {
  const ps = await panels();
  const have = ps.filter((p) => p.run);
  S.tmax = have.length ? Math.max(...have.map((p) => Number(p.run.meta.max_seconds) || 600)) : 600;
  $("slider").max = S.tmax;
  if (S.t > S.tmax) S.t = S.tmax;
  S.panels = ps;
  buildDom(ps);
  draw();
}

/* ---------------- DOM ---------------- */

function buildDom(ps) {
  // missing-data notices
  let n = "";
  for (const p of ps) {
    if (!p.run) { n += `<div class="miss"><b>${esc(p.title)}:</b> ${esc(p.why)}. Nothing is shown for it.</div>`; continue; }
    for (const [k, v] of Object.entries(p.run.missing))
      n += `<div class="miss"><b>${esc(p.title)} · ${esc(k)}:</b> ${esc(v)}</div>`;
    const m = p.run.meta;
    if (m.status !== "completed") n += `<div class="miss"><b>${esc(p.title)}:</b> ${esc(m.why)}</div>`;
    if (m.pilot === "pilot") n += `<div class="miss"><b>${esc(p.title)}:</b> from the superseded pilot 1 (pre-fix); kept for the record, not comparable to pilot 2.</div>`;
    const tj = p.run.trajectory;
    if (tj && tj.align_qc) {
      const q = tj.align_qc;
      n += `<div class="miss" style="border-color:var(--line)"><b>${esc(p.title)} · path timing check:</b> t=0 is the start-gate moment re-derived from the tlog;
        against fog's own first-visit times the median offset is ${q.median_s} s over ${q.cells_compared} cells
        (${q.n_over_10s} cell(s) differ by more than 10 s, max ${q.max_abs_s} s; not explained, so read path timing as ±a few seconds).</div>`;
    }
  }
  $("notice").innerHTML = n;

  $("maps").innerHTML = ps.map((p, k) => p.run
    ? `<div class="mapbox"><h2>${esc(p.title)}</h2><canvas id="map${k}" width="560" height="560"></canvas></div>`
    : `<div class="mapbox"><h2>${esc(p.title)}</h2><div class="nodata">NO DATA<br>${esc(p.why)}</div></div>`).join("");
  if (!ps.length) $("maps").innerHTML = `<div class="panel" style="padding:20px">Pick a run on the left.</div>`;

  renderSummary(ps);
  renderCharts(ps);
}

function firstT(dets, pred) {
  if (!dets) return null;
  const xs = dets.filter(pred).map((d) => d.t);
  return xs.length ? Math.min(...xs) : null;
}

function renderSummary(ps) {
  const rows = ps.filter((p) => p.run);
  if (!rows.length) { $("summary").innerHTML = ""; return; }
  const cols = [
    ["Run", (d, p) => `<td>${esc(p.title)}${S.mode === "replay" ? "" : ""}</td>`],
    ["Model / thr", (d) => d.meta.detector_enabled ? `<td>${esc(d.meta.model_tag || "?")} @ ${d.meta.conf_threshold ?? "?"}</td>` : `<td>none</td>`],
    ["TP", (d) => d.meta.detector_enabled === false ? na("detector disabled") : d.end ? `<td>${d.end.det_true}</td>` : na("experiment_end missing")],
    ["FP", (d) => d.meta.detector_enabled === false ? na("detector disabled") : d.end ? `<td>${d.end.det_false}</td>` : na("experiment_end missing")],
    ["precision", (d) => d.meta.detector_enabled === false ? na("detector disabled") : d.end && +d.end.det_total > 0 ? `<td>${fmt(d.end.precision, 3)}</td>` : na("no detections")],
    ["first TP (s)", (d) => { if (d.meta.detector_enabled === false) return na("detector disabled"); const x = firstT(d.detections, (q) => q.correct === 1); return x === null ? na("no true detection in this run") : `<td>${fmt(x, 1)}</td>`; }],
    ["first FP (s)", (d) => { if (d.meta.detector_enabled === false) return na("detector disabled"); const x = firstT(d.detections, (q) => q.correct === 0); return x === null ? `<td>none</td>` : `<td>${fmt(x, 1)}</td>`; }],
    ["visited", (d) => d.end ? `<td>${fmt(100 * d.end.visited_frac, 0)}%</td>` : na("experiment_end missing")],
    ["fire cells visited", (d) => d.fire ? `<td>${d.fire.filter((c) => c.first_visit_t !== null).length} / ${d.fire.length}</td>` : na("fire_cells missing")],
    ["burnt", (d) => d.end ? `<td>${fmt(100 * d.end.burnt_frac, 0)}%</td>` : na("experiment_end missing")],
  ];
  $("summary").innerHTML = `<div class="panel" style="padding:8px 12px"><table class="sum"><tr>${cols.map((c) => `<th>${c[0]}</th>`).join("")}</tr>` +
    rows.map((p) => `<tr>${cols.map((c) => c[1](p.run, p)).join("")}</tr>`).join("") +
    `</table><div class="key" style="font-size:11px;color:var(--ink-dim);margin-top:6px">n = 1 run per cell and ~20 s SITL timing jitter (methodology sec. 13): differences are illustrative, not a performance claim. "n/a" = not recorded for that run.</div></div>`;
}

/* ---------------- map drawing ---------------- */

function lastAtOrBefore(arr, t, key = 0) {
  let lo = 0, hi = arr.length - 1, r = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (arr[m][key] <= t) { r = m; lo = m + 1; } else hi = m - 1; }
  return r;
}

function drawMap(cv, d, t) {
  const g = cv.getContext("2d"), W = cv.width, N = d.meta.N, C = W / N, side = d.meta.side_m, half = side / 2;
  g.fillStyle = "#05080c"; g.fillRect(0, 0, W, W);
  const cellXY = (i, j) => [i * C, (N - 1 - j) * C];

  if ($("l-cov").checked && d.cells)
    for (const c of d.cells) if (c.first_t !== null && c.first_t <= t) {
      const [x, y] = cellXY(c.i, c.j); g.fillStyle = "rgba(45,111,181,0.30)"; g.fillRect(x, y, C, C);
    }
  if ($("l-fire").checked && d.fire && d.burn_s !== null)
    for (const c of d.fire) if (c.ignite_t !== null && c.ignite_t <= t) {
      const [x, y] = cellXY(c.i, c.j), burnt = t >= c.ignite_t + d.burn_s;
      g.fillStyle = burnt ? "rgba(90,97,105,0.60)" : "rgba(255,87,34,0.80)"; g.fillRect(x, y, C, C);
    }
  g.strokeStyle = "rgba(255,255,255,0.08)"; g.lineWidth = 1;
  for (let k = 0; k <= N; k++) { g.beginPath(); g.moveTo(k * C, 0); g.lineTo(k * C, W); g.stroke(); g.beginPath(); g.moveTo(0, k * C); g.lineTo(W, k * C); g.stroke(); }
  for (const ig of d.meta.ignition || []) {  // ignition cell
    const [x, y] = cellXY(ig[0], ig[1]); g.strokeStyle = "#ffc61a"; g.lineWidth = 2; g.strokeRect(x + 2, y + 2, C - 4, C - 4);
  }
  if ($("l-det").checked && d.detections) {
    // one marker per (cell, correctness); radius grows with count, alpha with the best confidence so far
    const agg = new Map();
    for (const q of d.detections) if (q.t <= t) {
      const k = q.i + "," + q.j + "," + q.correct, a = agg.get(k) || { n: 0, best: 0, last: -1e9, q };
      a.n++; a.best = Math.max(a.best, q.conf || 0); a.last = Math.max(a.last, q.t); agg.set(k, a);
    }
    for (const a of agg.values()) {
      const [x, y] = cellXY(a.q.i, a.q.j), cx = x + C / 2, cy = y + C / 2, r = Math.min(C * 0.42, 5 + 2.2 * Math.sqrt(a.n));
      const fp = a.q.correct === 0, col = fp ? "244,86,74" : "53,208,127", fresh = t - a.last < 15;
      g.fillStyle = `rgba(${col},${0.25 + 0.55 * a.best})`; g.beginPath(); g.arc(cx, cy, r, 0, 7); g.fill();
      g.lineWidth = fresh ? 3 : 1; g.strokeStyle = `rgb(${col})`; g.stroke();
      g.fillStyle = "#fff"; g.font = "11px ui-monospace,monospace"; g.textAlign = "center";
      g.fillText(a.best.toFixed(2), cx, cy + r + 11);
    }
  }
  const tj = d.trajectory; let uav = "";
  if ($("l-trail").checked && tj) {
    const P = (s) => [((s[2] + half) / side) * W, (1 - (s[1] + half) / side) * W];
    const k = lastAtOrBefore(tj.samples, t);
    if (k >= 0) {
      g.strokeStyle = "rgba(255,255,255,0.55)"; g.lineWidth = 1.5; g.beginPath();
      let pen = false;
      for (let m = 0; m <= k; m++) {
        const s = tj.samples[m], [x, y] = P(s), gap = m > 0 && s[0] - tj.samples[m - 1][0] > 3;
        if (!pen || gap) { g.moveTo(x, y); pen = true; } else g.lineTo(x, y);
      }
      g.stroke();
      const s = tj.samples[k];
      if (t - s[0] <= 3) {
        const [x, y] = P(s); g.fillStyle = "#fff"; g.beginPath(); g.arc(x, y, 6, 0, 7); g.fill();
        if (s[4] !== null) { const a = (s[4] * Math.PI) / 180; g.strokeStyle = "#fff"; g.lineWidth = 2; g.beginPath(); g.moveTo(x, y); g.lineTo(x + 16 * Math.sin(a), y - 16 * Math.cos(a)); g.stroke(); }
        uav = `UAV ${s[3]} m AGL, ${s[5]} m/s`;
      } else uav = "UAV: telemetry gap here";
    } else uav = "UAV: before first telemetry sample";
  } else if ($("l-trail").checked && !tj) {
    g.fillStyle = "#f4564a"; g.font = "12px sans-serif"; g.textAlign = "left"; g.fillText("no UAV path: " + (d.missing.trajectory || "n/a"), 8, W - 8);
  }
  if (uav) { g.fillStyle = "#e6edf3"; g.font = "12px ui-monospace,monospace"; g.textAlign = "left"; g.fillText(uav, 8, 16); }
}

/* ---------------- charts ---------------- */

function plot(cv, series, o) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr);
  const L = 36, R = 8, T = 8, B = 20, xmax = o.xmax, ymax = o.ymax || 1;
  const X = (x) => L + (x / xmax) * (w - L - R), Y = (y) => h - B - (y / ymax) * (h - T - B);
  g.strokeStyle = "#2b3540"; g.fillStyle = "#8b98a5"; g.font = "10px sans-serif"; g.lineWidth = 1;
  for (let k = 0; k <= 4; k++) { const y = (ymax * k) / 4; g.beginPath(); g.moveTo(L, Y(y)); g.lineTo(w - R, Y(y)); g.stroke(); g.textAlign = "right"; g.fillText(+y.toFixed(2), L - 4, Y(y) + 3); }
  for (let x = 0; x <= xmax; x += 100) { g.textAlign = "center"; g.fillText(x, X(x), h - 6); }
  for (const s of series) {
    g.strokeStyle = g.fillStyle = s.color; g.lineWidth = 1.6; g.setLineDash(s.dash || []);
    if (s.mode === "points") {
      g.setLineDash([]);
      s.x.forEach((x, k) => { if (s.y[k] === null) return; const px = X(x), py = Y(s.y[k]);
        if (s.cross) { g.beginPath(); g.moveTo(px - 3, py - 3); g.lineTo(px + 3, py + 3); g.moveTo(px + 3, py - 3); g.lineTo(px - 3, py + 3); g.stroke(); }
        else { g.globalAlpha = 0.6; g.beginPath(); g.arc(px, py, 2.5, 0, 7); g.fill(); g.globalAlpha = 1; } });
    } else if (s.mode === "hline") {
      g.beginPath(); g.moveTo(L, Y(s.y[0])); g.lineTo(w - R, Y(s.y[0])); g.stroke();
    } else {
      g.beginPath(); let prev = null;
      s.x.forEach((x, k) => { if (s.y[k] === null) return; const px = X(x), py = Y(s.y[k]);
        if (prev === null) g.moveTo(px, py); else { if (s.mode === "step") g.lineTo(px, prev); g.lineTo(px, py); } prev = py; });
      g.stroke();
    }
    g.setLineDash([]);
  }
  g.strokeStyle = "#e6edf3"; g.lineWidth = 1; g.beginPath(); g.moveTo(X(S.t), T); g.lineTo(X(S.t), h - B); g.stroke();
}

function cum(dets, pred) { const xs = [0], ys = [0]; let n = 0; for (const d of dets.filter(pred)) { n++; xs.push(d.t, d.t); ys.push(n - 1, n); } return [xs, ys]; }

let chartSpecs = [];
function renderCharts(ps) {
  const have = ps.filter((p) => p.run);
  chartSpecs = [];
  if (!have.length) { $("charts").innerHTML = ""; return; }
  const col = (p) => (S.mode === "compare" ? ARM_COLOR[p.arm] || "#fff" : "#4a9eff");
  const det = have.filter((p) => p.run.detections && p.run.meta.detector_enabled !== false);
  const maxCnt = Math.max(5, ...det.map((p) => p.run.detections.length));
  const key = (arr) => arr.map((p) => `<i style="background:${col(p)}"></i>${esc(p.title)}`).join("");
  chartSpecs.push({ title: "Coverage (fraction of cells visited)", ymax: 0.5, key: key(have),
    series: have.filter((p) => p.run.coverage_t).map((p) => ({ x: p.run.coverage_t.t, y: p.run.coverage_t.visited_frac, color: col(p) })) });
  chartSpecs.push({ title: "Cumulative detections: solid = true, dashed = false positive", ymax: maxCnt, key: key(det),
    series: det.flatMap((p) => { const a = cum(p.run.detections, (d) => d.correct === 1), b = cum(p.run.detections, (d) => d.correct === 0);
      return [{ x: a[0], y: a[1], color: col(p), mode: "step" }, { x: b[0], y: b[1], color: col(p), mode: "step", dash: [4, 3] }]; }) });
  chartSpecs.push({ title: "Detection confidence: dot = true, × = false positive; dashed = run threshold", ymax: 1, key: key(det),
    series: det.flatMap((p) => { const d = p.run.detections, tp = d.filter((q) => q.correct === 1), fp = d.filter((q) => q.correct === 0);
      const s = [{ x: tp.map((q) => q.t), y: tp.map((q) => q.conf), color: col(p), mode: "points" }, { x: fp.map((q) => q.t), y: fp.map((q) => q.conf), color: col(p), mode: "points", cross: true }];
      if (p.run.meta.conf_threshold) s.push({ x: [0], y: [Number(p.run.meta.conf_threshold)], color: col(p), mode: "hline", dash: [2, 4] });
      return s; }) });
  chartSpecs.push({ title: "Fire truth: cells burning + burnt", ymax: 40, key: key(have),
    series: have.filter((p) => p.run.fire_t).map((p) => ({ x: p.run.fire_t.t, y: p.run.fire_t.t.map((_, k) => p.run.fire_t.burning_cnt[k] + p.run.fire_t.burnt_cnt[k]), color: col(p) })) });
  $("charts").innerHTML = chartSpecs.map((c, k) => `<div class="chart"><h2>${esc(c.title)}</h2><canvas id="ch${k}"></canvas><div class="key">${c.key}${c.series.length ? "" : " &nbsp; no data"}</div></div>`).join("");
  if (S.mode === "compare" && !det.length) $("charts").insertAdjacentHTML("beforeend", "");
}

function drawCharts() { chartSpecs.forEach((c, k) => { const cv = $("ch" + k); if (cv) plot(cv, c.series, { xmax: S.tmax, ymax: c.ymax }); }); }

function renderDets() {
  const el = $("dets");
  if (S.mode !== "replay" || !S.panels || !S.panels[0] || !S.panels[0].run) { el.hidden = true; return; }
  const d = S.panels[0].run; el.hidden = false;
  if (d.meta.detector_enabled === false) { el.innerHTML = "<b>Detections:</b> this arm has no detector, so there are none to list."; return; }
  if (!d.detections) { el.innerHTML = "<b>Detections:</b> csv missing."; return; }
  const rows = d.detections.filter((q) => q.t <= S.t).slice(-80).reverse();
  el.innerHTML = `<b>Detections up to t = ${S.t.toFixed(0)} s</b> (newest first, last 80 of ${d.detections.filter((q) => q.t <= S.t).length}; truth = fire within ±1 cell)<table><tr><th>t (s)</th><th>cell</th><th>conf</th><th>verdict</th></tr>` +
    rows.map((q) => `<tr class="${q.correct ? "tp" : "fp"}"><td>${q.t.toFixed(1)}</td><td>${q.i},${q.j}</td><td>${q.conf}</td><td>${q.correct ? "true" : "FALSE POSITIVE"}</td></tr>`).join("") + "</table>";
}

/* ---------------- loop ---------------- */

function draw() {
  $("clock").textContent = "t = " + Math.round(S.t) + " s";
  $("slider").value = S.t;
  (S.panels || []).forEach((p, k) => { if (p.run) drawMap($("map" + k), p.run, S.t); });
  drawCharts(); renderDets();
}

let timer = null;
function setPlaying(on) {
  S.playing = on; $("play").textContent = on ? "Pause" : "Play";
  clearInterval(timer);
  if (on) timer = setInterval(() => { S.t += (+$("speed").value) * 0.2; if (S.t >= S.tmax) { S.t = S.tmax; setPlaying(false); } draw(); }, 200);
}

async function init() {
  S.list = await api("/api/runs");
  renderList();
  $("tab-replay").onclick = () => { setMode("replay"); refresh(); };
  $("tab-compare").onclick = () => {
    if (!S.cmp) { const r = S.list.runs.find((r) => r.status === "completed" && !r.diagnostic && r.pilot === "pilot2"); if (r) S.cmp = { pilot: r.pilot, scenario: r.scenario }; }
    setMode("compare"); refresh();
  };
  $("play").onclick = () => setPlaying(!S.playing);
  $("slider").oninput = (e) => { S.t = +e.target.value; draw(); };
  ["l-cov", "l-fire", "l-det", "l-trail"].forEach((id) => ($(id).onchange = draw));
  window.addEventListener("resize", drawCharts);
  const first = S.list.runs.find((r) => r.status === "completed" && r.pilot === "pilot2" && r.arm === "A0") || S.list.runs.find((r) => r.status === "completed");
  if (first) await select(first.id); else refresh();
}
init().catch((e) => { $("notice").innerHTML = `<div class="miss"><b>Failed to load:</b> ${esc(e.message)}</div>`; });
