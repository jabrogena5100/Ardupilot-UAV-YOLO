/* replay.js: browser side of dashboard/replay_server.py.
   Every number drawn comes from a saved file via /api/run. Where a file is missing the
   view says so and draws nothing: no interpolation, no defaults, no simulated stand-ins. */
"use strict";

const ARM_COLOR = { baseline: "#9aa7b4", A: "#4a9eff", B: "#c77dff" };
const $ = (id) => document.getElementById(id);
const famOf = (arm) => (arm === "baseline" ? "baseline" : arm && arm[0] === "A" ? "A" : arm && arm[0] === "B" ? "B" : "?");
const FAM_TITLE = { baseline: "Baseline: patrol, no detector", A: "Model A: trained on synthetic images", B: "Model B: trained on real images" };

const S = {
  list: null, mode: "compare", dataset: "final", scenario: null, seedA: "A0", seedB: "B2s0", arm: "baseline",
  cache: new Map(), t: 0, playing: false, tmax: 600, panels: [],
};

/* ---------------- pure helpers (also used by the tests through the data they draw) ---------------- */

// Map a local north/east offset (metres from the grid centre) to canvas pixels. North is up, east is right.
function nePixel(n, e, side, W) { return [((e + side / 2) / side) * W, (1 - (n + side / 2) / side) * W]; }
// Map a grid cell (i = east index, j = north index) to the canvas pixel of its top-left corner.
function cellXY(i, j, N, C) { return [i * C, (N - 1 - j) * C]; }
// Metres offset -> cell index (same rule as dashboard/replay_build.py).
function neCell(n, e, side, N) { const c = side / N; return [Math.floor((e + side / 2) / c), Math.floor((n + side / 2) / c)]; }

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
const na = (why, txt = "n/a") => `<td class="na" title="${esc(why)}">${txt}</td>`;

/* ---------------- run selection ---------------- */

const runsOf = (ds) => S.list.runs.filter((r) => r.pilot === ds && !r.diagnostic);
const findRun = (ds, scn, arm) => runsOf(ds).find((r) => r.scenario === scn && r.arm === arm && r.status === "completed");
const scenariosOf = (ds) => [...new Set(runsOf(ds).filter((r) => r.status === "completed").map((r) => r.scenario))].sort();
const armsOf = (ds, fam) => [...new Set(runsOf(ds).filter((r) => r.status === "completed" && famOf(r.arm) === fam).map((r) => r.arm))].sort();

function renderList() {
  const L = S.list, el = $("runlist"), ds = S.dataset;
  const scns = scenariosOf(ds);
  if (!scns.includes(S.scenario)) S.scenario = scns[0] || null;
  const seedsA = armsOf(ds, "A"), seedsB = armsOf(ds, "B");
  if (!seedsA.includes(S.seedA)) S.seedA = seedsA[0] || null;
  if (!seedsB.includes(S.seedB)) S.seedB = seedsB[0] || null;
  const arms = ["baseline", ...seedsA, ...seedsB].filter((a) => findRun(ds, S.scenario, a));
  if (!arms.includes(S.arm)) S.arm = arms[0] || null;
  const opt = (v, cur, label) => `<option value="${esc(v)}" ${v === cur ? "selected" : ""}>${esc(label)}</option>`;
  let h = `<h3>1 · Experiment</h3>
    <select id="sel-ds" aria-label="experiment set">${Object.keys(L.pilots).map((p) => opt(p, ds, L.pilots[p].label)).join("")}</select>
    <div class="hint">${esc(L.pilots[ds].note)}</div>`;
  if (ds !== "final") h += `<div class="hint warn">Pilot data: n = 1 per cell and not part of the final results. Kept for the record.</div>`;
  h += `<h3>2 · Scenario</h3>
    <select id="sel-scn" aria-label="scenario">${scns.map((s) => opt(s, S.scenario, "scenario " + s)).join("")}</select>
    <div class="hint">Same ignition cell and fire configuration for every arm in a scenario (matched).</div>`;
  if (S.mode === "compare") {
    h += `<h3>3 · Detector seeds to compare</h3>
      <label class="sel-l">Model A seed <select id="sel-a">${seedsA.map((a) => opt(a, S.seedA, "seed " + a.slice(1))).join("")}</select></label>
      <label class="sel-l">Model B seed <select id="sel-b">${seedsB.map((a) => opt(a, S.seedB, "seed " + a.slice(-1))).join("")}</select></label>
      <div class="hint">Baseline, A and B are shown side by side on one shared clock. Each seed is a separately trained detector; the report uses all three.</div>`;
  } else {
    h += `<h3>3 · Arm</h3><select id="sel-arm" aria-label="arm">${arms.map((a) => opt(a, S.arm, L.arm_label[a] || a)).join("")}</select>`;
  }
  const bad = runsOf(ds).filter((r) => r.status !== "completed"), planned = L.planned.filter((q) => q.pilot === ds);
  if (bad.length || planned.length) {
    h += `<h3>Not available</h3>` + bad.map((r) => `<div class="run gone">${esc(L.arm_label[r.arm] || r.tag)} · ${esc(r.scenario)}<span class="tag bad">partial</span><span class="sub">${esc(r.why || "")}</span></div>`).join("") +
      planned.map((q) => `<div class="run gone">${esc(q.label)} · ${esc(q.scenario)}<span class="tag warn">pending</span><span class="sub">in matrix, no data exists</span></div>`).join("");
  }
  el.innerHTML = h;
  const on = (id, fn) => { const e = $(id); if (e) e.onchange = fn; };
  on("sel-ds", (e) => { S.dataset = e.target.value; S.scenario = null; renderList(); refresh(); });
  on("sel-scn", (e) => { S.scenario = e.target.value; renderList(); refresh(); });
  on("sel-a", (e) => { S.seedA = e.target.value; refresh(); });
  on("sel-b", (e) => { S.seedB = e.target.value; refresh(); });
  on("sel-arm", (e) => { S.arm = e.target.value; refresh(); });
}

function setMode(m) {
  S.mode = m;
  $("tab-replay").classList.toggle("on", m === "replay");
  $("tab-compare").classList.toggle("on", m === "compare");
  renderList();
}

/* ---------------- what to show ---------------- */

async function panels() {
  const ds = S.dataset, scn = S.scenario, out = [];
  if (!scn) return out;
  const want = S.mode === "compare" ? ["baseline", S.seedA, S.seedB] : [S.arm];
  for (const arm of want) {
    if (!arm) continue;
    const title = S.mode === "compare" ? FAM_TITLE[famOf(arm)] + (famOf(arm) === "baseline" ? "" : " (seed " + arm.slice(-1) + ")") : (S.list.arm_label[arm] || arm);
    const ok = findRun(ds, scn, arm);
    if (ok) { out.push({ key: ok.id, title, run: await getRun(ok.id), arm }); continue; }
    const bad = runsOf(ds).find((r) => r.scenario === scn && r.arm === arm);
    const planned = S.list.planned.find((q) => q.pilot === ds && q.scenario === scn && q.arm === arm);
    out.push({ key: ds + arm, title, run: null, arm, why: bad ? "interrupted: " + bad.why : planned ? "not run: " + planned.why : "no run found on disk" });
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
  const have = ps.filter((p) => p.run);
  // banner: what is being shown, and the standing scientific caveats
  const ds = S.dataset, ig = have.length && have[0].run.meta.ignition ? have[0].run.meta.ignition.map((c) => `(${c[0]},${c[1]})`).join(", ") : null;
  $("banner").innerHTML = `<div class="panel banner"><b>${esc(S.list.pilots[ds].label)}</b> · scenario ${esc(S.scenario || "?")}` +
    (ig ? ` · fire ignites at cell ${esc(ig)} (i = east, j = north; N is up)` : "") +
    ` · single UAV · ${ds === "final" ? "oracle off: the UAV learns about the fire only through its detector" : "pilot run"}</div>`;

  // data notes (collapsed): missing files, status, timing check, fire-clock caveat
  let n = "";
  const seenQC = new Set();
  for (const p of ps) {
    if (!p.run) { n += `<div class="miss"><b>${esc(p.title)}:</b> ${esc(p.why)}. Nothing is shown for it.</div>`; continue; }
    for (const [k, v] of Object.entries(p.run.missing)) n += `<div class="miss"><b>${esc(p.title)} · ${esc(k)}:</b> ${esc(v)}</div>`;
    const m = p.run.meta;
    if (m.status !== "completed") n += `<div class="miss"><b>${esc(p.title)}:</b> ${esc(m.why)}</div>`;
    if (m.pilot === "pilot") n += `<div class="miss"><b>${esc(p.title)}:</b> from the superseded pilot 1 (pre-fix); not comparable to later runs.</div>`;
    const q = p.run.trajectory && p.run.trajectory.align_qc;
    if (q && !seenQC.has(p.key)) {
      seenQC.add(p.key);
      n += `<div class="miss dim"><b>${esc(p.title)} · path timing:</b> t = 0 is the start-gate moment re-derived from the saved MAVLink log. Against the fog tracker's own first-visit times the median offset is ${q.median_s} s over ${q.cells_compared} cells (${q.n_over_10s} cell(s) differ by more than 10 s, max ${q.max_abs_s} s). Read UAV path timing as ± a few seconds.</div>`;
    }
  }
  if (ds === "final") {
    n += `<div class="miss dim"><b>Matched scenarios:</b> the initial fire is identical across arms, but the fire simulator steps on the wall clock, so the realized spread differs slightly between runs (more so in runs executed 2 at a time). See docs/experiments/level3_final.md section 2a.</div>`;
    n += `<div class="miss dim"><b>Untimed investigation targets:</b> the agent log has no timestamps, so investigation targets are a whole-run summary. “Fire nearby” is judged in hindsight (within 1 cell of any cell that burned at any time); the UAV did not know that when it chose the target.</div>`;
    n += `<div class="miss dim"><b>Capped times:</b> a value shown as “≥ 600 s” means the event was not observed inside the 600 s window; the true time is at least that.</div>`;
  }
  $("notice").innerHTML = n;
  $("notes").querySelector("summary").textContent = "Data notes and caveats";

  $("maps").innerHTML = ps.map((p, k) => p.run
    ? `<div class="mapbox"><h2 style="border-left:4px solid ${ARM_COLOR[famOf(p.arm)] || "#fff"};padding-left:8px">${esc(p.title)}</h2>
         <canvas id="map${k}" width="560" height="560" aria-label="${esc(p.title)} map"></canvas><div class="live" id="live${k}"></div></div>`
    : `<div class="mapbox"><h2>${esc(p.title)}</h2><div class="nodata">NO DATA<br>${esc(p.why)}</div></div>`).join("");
  if (!ps.length) $("maps").innerHTML = `<div class="panel" style="padding:20px">No completed runs for this selection.</div>`;

  renderSummary(ps);
  renderCharts(ps);
  renderJumps(ps);
}

function firstT(dets, pred) {
  if (!dets) return null;
  const xs = dets.filter(pred).map((d) => d.t);
  return xs.length ? Math.min(...xs) : null;
}

/* Recorded metrics (final: analysis/per_run.csv; otherwise derived from the run's own files). */
function metricsOf(d) {
  const det = d.meta.detector_enabled !== false;
  const rec = d.recorded;
  const falseTargets = d.investigation ? d.investigation.targets.filter((t) => t.near_fire === false) : null;
  return {
    det,
    firstTP: rec ? (rec.first_true_det_na ? { na: "no detector" } : rec.first_true_det_censored ? { cap: rec.first_true_det_capped_s } : { v: rec.first_true_det_s })
                 : (!det ? { na: "no detector" } : (() => { const x = firstT(d.detections, (q) => q.correct === 1); return x === null ? { cap: d.meta.max_seconds || 600 } : { v: x }; })()),
    ignVisit: rec ? (rec.ign_first_visit_censored ? { cap: rec.ign_first_visit_capped_s } : { v: rec.ign_first_visit_s })
                  : (() => { const ig = d.fire && d.fire.find((c) => c.ignite_t !== null && c.ignite_t <= 0.5); return !ig ? { na: "no ignition cell" } : ig.first_visit_t === null ? { cap: d.meta.max_seconds || 600 } : { v: ig.first_visit_t }; })(),
    coverage: rec && rec.visited_frac !== null ? rec.visited_frac : d.end ? Number(d.end.visited_frac) : null,
    fireVisited: rec ? [rec.fire_cells_visited, rec.fire_cells] : d.fire ? [d.fire.filter((c) => c.first_visit_t !== null).length, d.fire.length] : null,
    tp: rec ? rec.det_true : d.end ? Number(d.end.det_true) : null,
    fp: rec ? rec.det_false : d.end ? Number(d.end.det_false) : null,
    invTrue: rec ? rec.inv_true_s : null, invFalse: rec ? rec.inv_false_s : null,
    falseTargets: falseTargets ? falseTargets.length : null,
    nTargets: d.investigation ? d.investigation.targets.length : null,
  };
}

function renderSummary(ps) {
  const rows = ps.filter((p) => p.run);
  if (!rows.length) { $("summary").innerHTML = ""; return; }
  const cap = (x, unit = " s") => x.na ? na(x.na, "n/a") : x.cap !== undefined ? `<td class="cap" title="not observed within the run window; the true time is at least this">≥ ${fmt(x.cap, 0)}${unit}<sup>capped</sup></td>` : `<td>${fmt(x.v, 1)}${unit}</td>`;
  const cols = [
    ["Arm", (m, d, p) => `<td>${esc(p.title)}</td>`],
    ["First true detection", (m) => cap(m.firstTP)],
    ["Ignition cell first reached by UAV", (m) => cap(m.ignVisit)],
    ["Coverage (cells visited)", (m) => m.coverage === null ? na("not recorded") : `<td>${fmt(100 * m.coverage, 0)}%</td>`],
    ["Fire cells overflown", (m) => m.fireVisited && m.fireVisited[0] !== null ? `<td>${m.fireVisited[0]} / ${m.fireVisited[1]}</td>` : na("fire_cells csv missing")],
    ["True / false detection observations", (m, d) => !m.det ? na("no detector in this arm", "n/a") : m.tp === null ? na("experiment_end missing") : `<td>${m.tp} / ${m.fp}</td>`],
    ["Investigation time on fire-adjacent / false targets", (m, d) => !m.det ? na("no detector, so no investigations") : m.invTrue === null ? na("only recorded for the final runs") : `<td>≈${m.invTrue} / ≈${m.invFalse} s</td>`],
    ["False-priority diversions (distinct cells)", (m, d) => !m.det ? na("no detector, so no investigations") : m.falseTargets === null ? na("agent log not saved") : `<td>${m.falseTargets} of ${m.nTargets}</td>`],
  ];
  $("summary").innerHTML = `<div class="panel" style="padding:8px 12px"><h2 class="sumh">Recorded results for this scenario</h2><div class="tscroll"><table class="sum"><tr>${cols.map((c) => `<th>${c[0]}</th>`).join("")}</tr>` +
    rows.map((p) => { const m = metricsOf(p.run); return `<tr>${cols.map((c) => c[1](m, p.run, p)).join("")}</tr>`; }).join("") +
    `</table></div><div class="key" style="font-size:11px;color:var(--ink-dim);margin-top:6px">` +
    (S.dataset === "final" ? `Values come from the saved run files and the recorded analysis table (analysis/per_run.csv). ` : `Pilot run, n = 1: illustrative only. `) +
    `<sup>capped</sup> = not observed in the 600 s window; shown as a lower bound. “Investigation time” counts agent HUD lines (about 1 per second, approximate). A true detection means fire within 1 cell at the time of the observation; detection counts are repeated observations, not incidents. “n/a” = not recorded or not applicable.</div></div>`;
}

function renderJumps(ps) {
  const items = [];
  for (const p of ps) if (p.run) {
    const m = metricsOf(p.run), who = famOf(p.arm) === "baseline" ? "baseline" : "Model " + famOf(p.arm);
    if (m.firstTP.v !== undefined) items.push([m.firstTP.v, `first true detection · ${who}`, "g-det"]);
    if (m.ignVisit.v !== undefined) items.push([m.ignVisit.v, `UAV reaches ignition cell · ${who}`, "g-uav"]);
  }
  items.sort((a, b) => a[0] - b[0]);
  $("jumps").innerHTML = items.length ? "Jump to: " + items.map((it) => `<button class="jump ${it[2]}" data-t="${it[0]}">${esc(it[1])} (${it[0].toFixed(0)} s)</button>`).join("") : "";
  $("jumps").querySelectorAll("button.jump").forEach((b) => (b.onclick = () => { S.t = Math.min(S.tmax, +b.dataset.t); draw(); }));
}

/* ---------------- map drawing ---------------- */

function lastAtOrBefore(arr, t, key = 0) {
  let lo = 0, hi = arr.length - 1, r = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (arr[m][key] <= t) { r = m; lo = m + 1; } else hi = m - 1; }
  return r;
}

function drawMap(cv, d, t) {
  const g = cv.getContext("2d"), W = cv.width, N = d.meta.N, C = W / N, side = d.meta.side_m;
  g.fillStyle = "#05080c"; g.fillRect(0, 0, W, W);
  const cxy = (i, j) => cellXY(i, j, N, C);

  if ($("l-cov").checked && d.cells)
    for (const c of d.cells) if (c.first_t !== null && c.first_t <= t) {
      const [x, y] = cxy(c.i, c.j); g.fillStyle = "rgba(45,111,181,0.38)"; g.fillRect(x, y, C, C);
    }
  if ($("l-fire").checked && d.fire && d.burn_s !== null)
    for (const c of d.fire) if (c.ignite_t !== null && c.ignite_t <= t) {
      const [x, y] = cxy(c.i, c.j), burnt = t >= c.ignite_t + d.burn_s;
      g.fillStyle = burnt ? "rgba(90,97,105,0.60)" : "rgba(255,87,34,0.80)"; g.fillRect(x, y, C, C);
    }
  g.strokeStyle = "rgba(255,255,255,0.08)"; g.lineWidth = 1;
  for (let k = 0; k <= N; k++) { g.beginPath(); g.moveTo(k * C, 0); g.lineTo(k * C, W); g.stroke(); g.beginPath(); g.moveTo(0, k * C); g.lineTo(W, k * C); g.stroke(); }
  if ($("l-fire").checked)
    for (const ig of d.meta.ignition || []) {
      const [x, y] = cxy(ig[0], ig[1]); g.strokeStyle = "#ffc61a"; g.lineWidth = 2; g.strokeRect(x + 2, y + 2, C - 4, C - 4);
    }
  if ($("l-inv").checked && d.investigation) {
    g.lineWidth = 2; g.setLineDash([5, 3]); g.font = "bold 11px sans-serif"; g.textAlign = "left";
    for (const tg of d.investigation.targets) {
      const [x, y] = cxy(tg.i, tg.j), col = !$("l-fire").checked ? "#e8e8e8" : (tg.near_fire === false ? "#e040fb" : "#3de0e0");  // truth hidden: no hindsight colouring
      g.strokeStyle = col; g.strokeRect(x + 5, y + 5, C - 10, C - 10);
      g.setLineDash([]); g.fillStyle = col; g.fillText(String(tg.order), x + 8, y + 17); g.setLineDash([5, 3]);
    }
    g.setLineDash([]);
  }
  if ($("l-det").checked && d.detections) {
    // one marker per (cell, correctness); radius grows with count, alpha with the best confidence so far
    const agg = new Map();
    for (const q of d.detections) if (q.t <= t) {
      const k = q.i + "," + q.j + "," + q.correct, a = agg.get(k) || { n: 0, best: 0, last: -1e9, q };
      a.n++; a.best = Math.max(a.best, q.conf || 0); a.last = Math.max(a.last, q.t); agg.set(k, a);
    }
    for (const a of agg.values()) {
      const [x, y] = cxy(a.q.i, a.q.j), cx = x + C / 2, cy = y + C / 2, r = Math.min(C * 0.42, 5 + 2.2 * Math.sqrt(a.n));
      const fp = a.q.correct === 0, col = !$("l-fire").checked ? "232,200,90" : (fp ? "244,86,74" : "53,208,127"), fresh = t - a.last < 15;  // truth hidden: the UAV cannot tell true from false
      g.fillStyle = `rgba(${col},${0.25 + 0.55 * a.best})`; g.beginPath(); g.arc(cx, cy, r, 0, 7); g.fill();
      g.lineWidth = fresh ? 3 : 1; g.strokeStyle = `rgb(${col})`; g.stroke();
      g.fillStyle = "#fff"; g.font = "11px ui-monospace,monospace"; g.textAlign = "center";
      g.fillText(a.best.toFixed(2), cx, cy + r + 11);
    }
  }
  const tj = d.trajectory; let uav = "";
  if ($("l-trail").checked && tj) {
    const P = (s) => nePixel(s[1], s[2], side, W);
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
  g.fillStyle = "rgba(230,237,243,.7)"; g.font = "bold 12px sans-serif"; g.textAlign = "right"; g.fillText("N ↑", W - 8, 16);
}

function liveText(d, t) {
  const m = d.meta, N2 = m.N * m.N;
  const cov = d.cells ? d.cells.filter((c) => c.first_t !== null && c.first_t <= t).length : null;
  const burning = d.fire && d.burn_s !== null ? d.fire.filter((c) => c.ignite_t !== null && c.ignite_t <= t && t < c.ignite_t + d.burn_s).length : null;
  let s = `<span class="g-uav">covered ${cov === null ? "n/a" : (100 * cov / N2).toFixed(0) + "%"}</span>`;
  if (burning !== null && $("l-fire").checked) s += ` · <span class="g-truth">${burning} cells burning (truth)</span>`;
  if (m.detector_enabled === false) s += ` · <span class="dim">no detector</span>`;
  else if (d.detections) {
    const so = d.detections.filter((q) => q.t <= t), tp = so.filter((q) => q.correct === 1).length;
    s += $("l-fire").checked ? ` · <span class="g-det">detections so far: ${tp} true / ${so.length - tp} false</span>` : ` · <span class="g-det">detections so far: ${so.length} (true/false is judged against the hidden fire)</span>`;
  }
  return s;
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
  const col = (p) => (S.mode === "compare" ? ARM_COLOR[famOf(p.arm)] || "#fff" : "#4a9eff");
  const det = have.filter((p) => p.run.detections && p.run.meta.detector_enabled !== false);
  const maxCnt = Math.max(5, ...det.map((p) => p.run.detections.length));
  const key = (arr) => arr.map((p) => `<i style="background:${col(p)}"></i>${esc(p.title)}`).join("");
  chartSpecs.push({ title: "Coverage (fraction of cells visited)", ymax: 0.5, key: key(have),
    series: have.filter((p) => p.run.coverage_t).map((p) => ({ x: p.run.coverage_t.t, y: p.run.coverage_t.visited_frac, color: col(p) })) });
  chartSpecs.push({ title: "Cumulative detections: solid = true, dashed = false positive", ymax: maxCnt, key: key(det),
    series: det.flatMap((p) => { const a = cum(p.run.detections, (d) => d.correct === 1), b = cum(p.run.detections, (d) => d.correct === 0);
      return [{ x: a[0], y: a[1], color: col(p), mode: "step" }, { x: b[0], y: b[1], color: col(p), mode: "step", dash: [4, 3] }]; }) });
  chartSpecs.push({ title: "Detection confidence: dot = true, × = false positive; dotted line = run threshold", ymax: 1, key: key(det),
    series: det.flatMap((p) => { const d = p.run.detections, tp = d.filter((q) => q.correct === 1), fp = d.filter((q) => q.correct === 0);
      const s = [{ x: tp.map((q) => q.t), y: tp.map((q) => q.conf), color: col(p), mode: "points" }, { x: fp.map((q) => q.t), y: fp.map((q) => q.conf), color: col(p), mode: "points", cross: true }];
      if (p.run.meta.conf_threshold) s.push({ x: [0], y: [Number(p.run.meta.conf_threshold)], color: col(p), mode: "hline", dash: [2, 4] });
      return s; }) });
  chartSpecs.push({ title: "Fire ground truth: cells burning + burnt", ymax: 40, key: key(have),
    series: have.filter((p) => p.run.fire_t).map((p) => ({ x: p.run.fire_t.t, y: p.run.fire_t.t.map((_, k) => p.run.fire_t.burning_cnt[k] + p.run.fire_t.burnt_cnt[k]), color: col(p) })) });
  $("charts").innerHTML = chartSpecs.map((c, k) => `<div class="chart"><h2>${esc(c.title)}</h2><canvas id="ch${k}"></canvas><div class="key">${c.key}${c.series.length ? "" : " &nbsp; no data"}</div></div>`).join("");
}

function drawCharts() { chartSpecs.forEach((c, k) => { const cv = $("ch" + k); if (cv) plot(cv, c.series, { xmax: S.tmax, ymax: c.ymax }); }); }

function renderDets() {
  const el = $("dets");
  if (S.mode !== "replay" || !S.panels || !S.panels[0] || !S.panels[0].run) { el.hidden = true; return; }
  const d = S.panels[0].run; el.hidden = false;
  if (d.meta.detector_enabled === false) { el.innerHTML = "<b>Detections:</b> this arm has no detector, so there are none to list."; return; }
  if (!d.detections) { el.innerHTML = "<b>Detections:</b> csv missing."; return; }
  const rows = d.detections.filter((q) => q.t <= S.t).slice(-80).reverse();
  el.innerHTML = `<b>Detections up to t = ${S.t.toFixed(0)} s</b> (newest first, last 80 of ${d.detections.filter((q) => q.t <= S.t).length}; true = fire within ±1 cell at that moment)<table><tr><th>t (s)</th><th>cell</th><th>conf</th><th>verdict</th></tr>` +
    rows.map((q) => `<tr class="${q.correct ? "tp" : "fp"}"><td>${q.t.toFixed(1)}</td><td>${q.i},${q.j}</td><td>${q.conf}</td><td>${q.correct ? "true" : "FALSE POSITIVE"}</td></tr>`).join("") + "</table>";
}

/* ---------------- loop ---------------- */

function draw() {
  $("clock").textContent = "t = " + Math.round(S.t) + " s / " + S.tmax;
  $("slider").value = S.t;
  (S.panels || []).forEach((p, k) => { if (p.run) { drawMap($("map" + k), p.run, S.t); $("live" + k).innerHTML = liveText(p.run, S.t); } });
  drawCharts(); renderDets();
}

let timer = null, last = 0;
function setPlaying(on) {
  S.playing = on; $("play").textContent = on ? "Pause" : "Play";
  clearInterval(timer);
  if (on) {
    if (S.t >= S.tmax) S.t = 0;
    last = performance.now();
    timer = setInterval(() => {
      const now = performance.now(); S.t += (+$("speed").value) * (now - last) / 1000; last = now;
      if (S.t >= S.tmax) { S.t = S.tmax; setPlaying(false); }
      draw();
    }, 100);
  }
}

function setLayers(truth) { $("l-fire").checked = truth; draw(); }

async function init() {
  S.list = await api("/api/runs");
  const ds = Object.keys(S.list.pilots);
  S.dataset = ds.includes("final") && runsOf("final").some((r) => r.status === "completed") ? "final" : ds[0];
  renderList();
  $("tab-replay").onclick = () => { setMode("replay"); refresh(); };
  $("tab-compare").onclick = () => { setMode("compare"); refresh(); };
  $("play").onclick = () => setPlaying(!S.playing);
  $("restart").onclick = () => { S.t = 0; setPlaying(true); draw(); };
  $("slider").oninput = (e) => { S.t = +e.target.value; draw(); };
  ["l-cov", "l-fire", "l-det", "l-trail", "l-inv"].forEach((id) => ($(id).onchange = draw));
  $("pre-uav").onclick = () => setLayers(false);
  $("pre-all").onclick = () => { ["l-cov", "l-fire", "l-det", "l-trail", "l-inv"].forEach((id) => ($(id).checked = true)); draw(); };
  window.addEventListener("resize", drawCharts);
  window.addEventListener("keydown", (e) => {
    if (e.target && /^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(e.target.tagName)) return;  // native behaviour for focused controls
    if (e.code === "Space") { e.preventDefault(); setPlaying(!S.playing); }
    else if (e.code === "Home") { S.t = 0; draw(); }
    else if (e.code === "ArrowRight") { S.t = Math.min(S.tmax, S.t + 5); draw(); }
    else if (e.code === "ArrowLeft") { S.t = Math.max(0, S.t - 5); draw(); }
  });
  await refresh();
}
init().catch((e) => { $("notice").innerHTML = `<div class="miss"><b>Failed to load:</b> ${esc(e.message)}</div>`; $("notes").open = true; });
