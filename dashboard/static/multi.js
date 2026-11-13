/* dashboard/static/multi.js: the four-UAV demonstration page. Reads /api/state (the same read-only feed as the main dashboard)
 * and /api/demo (static metadata written by demo_multi.py). Nothing here invents data: positions come from MAVLink via the agents'
 * POSE packets, coverage and fire from fog, trails are only the positions received so far, regions are the agents' configured partitions. */
const el = (id) => document.getElementById(id);
const cv = el("map"), g = cv.getContext("2d");
let META = null, STATE = null;
const trails = new Map();            // sys -> [[x, y], ...] in canvas px, only positions actually received
const lastSeen = new Map();          // sys -> last_t of the last sample stored
const MAXTRAIL = 4000;

function bitmapToSet(hex, N) {
  const out = new Set(); if (!hex) return out;
  let b; try { b = BigInt("0x" + hex); } catch (e) { return out; }
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if ((b >> BigInt(i * N + j)) & 1n) out.add(i + "," + j);
  return out;
}
const fireAt = (s, N, i, j) => { const c = s ? s.charCodeAt(i * N + j) : NaN; return isNaN(c) ? 0 : c - 48; };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function toPx(lat, lon, N) {   // same flat-earth projection as mission/geo.py; grid is centred on the origin, (0,0) = south-west
  const R = 6378137, d2r = Math.PI / 180, side = META.miles * 1609.344;
  const n = (lat - META.origin.lat) * d2r * R, e = (lon - META.origin.lon) * d2r * R * Math.cos(META.origin.lat * d2r);
  return [((e + side / 2) / side) * cv.width, cv.height - ((n + side / 2) / side) * cv.height];
}

function ingestTrails(state, N) {
  for (const [sid, a] of Object.entries(state.agents || {})) {
    if (a.lat == null || a.lon == null || a.stale) continue;
    if (lastSeen.get(sid) === a.last_t) continue;
    lastSeen.set(sid, a.last_t);
    const p = toPx(a.lat, a.lon, N), t = trails.get(sid) || [];
    const q = t[t.length - 1];
    if (!q || Math.hypot(q[0] - p[0], q[1] - p[1]) > 1.5) { t.push(p); if (t.length > MAXTRAIL) t.shift(); }
    trails.set(sid, t);
  }
}

function draw(state) {
  const w = state.world || {}, N = w.N || META.N, S = cv.width / N;
  g.fillStyle = "#05080c"; g.fillRect(0, 0, cv.width, cv.height);
  const vis = bitmapToSet(w.vis, N);
  for (const u of META.uavs) {                                       // region tint
    g.fillStyle = u.color + "1c";
    for (const [i, j] of u.cells) g.fillRect(i * S, (N - 1 - j) * S, S, S);
  }
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const x = i * S, y = (N - 1 - j) * S;
    if (vis.has(i + "," + j)) { g.fillStyle = "rgba(70,130,200,0.45)"; g.fillRect(x, y, S, S); }
    const f = fireAt(w.fire, N, i, j);
    if (f === 1) { g.fillStyle = "rgba(255,87,34,0.85)"; g.fillRect(x, y, S, S); }
    else if (f === 2) { g.fillStyle = "rgba(90,97,105,0.65)"; g.fillRect(x, y, S, S); }
  }
  g.strokeStyle = "rgba(255,255,255,0.08)"; g.lineWidth = 1;
  for (let k = 0; k <= N; k++) { g.beginPath(); g.moveTo(k * S, 0); g.lineTo(k * S, cv.height); g.stroke(); g.beginPath(); g.moveTo(0, k * S); g.lineTo(cv.width, k * S); g.stroke(); }
  // region borders: draw each owned cell's outer edges in the drone's colour
  g.lineWidth = 3; g.setLineDash([9, 6]);
  for (const u of META.uavs) {
    const own = new Set(u.cells.map((c) => c[0] + "," + c[1]));
    g.strokeStyle = u.color; g.beginPath();
    for (const [i, j] of u.cells) {
      const x = i * S, y = (N - 1 - j) * S;
      if (!own.has(i + "," + (j + 1))) { g.moveTo(x, y); g.lineTo(x + S, y); }
      if (!own.has(i + "," + (j - 1))) { g.moveTo(x, y + S); g.lineTo(x + S, y + S); }
      if (!own.has((i - 1) + "," + j)) { g.moveTo(x, y); g.lineTo(x, y + S); }
      if (!own.has((i + 1) + "," + j)) { g.moveTo(x + S, y); g.lineTo(x + S, y + S); }
    }
    g.stroke();
  }
  g.setLineDash([]);
  const col = (sid) => (META.uavs.find((u) => String(u.sys) === String(sid)) || { color: "#fff" }).color;
  for (const [sid, t] of trails) {                                   // trajectories
    if (t.length < 2) continue;
    g.strokeStyle = col(sid); g.globalAlpha = 0.8; g.lineWidth = 2; g.beginPath();
    t.forEach((p, k) => (k ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.stroke(); g.globalAlpha = 1;
  }
  for (const [sid, a] of Object.entries(state.agents || {})) {       // the cell each UAV is currently flying to (its own claim)
    if (!a.claim || a.stale) continue;
    g.strokeStyle = col(sid); g.lineWidth = 3; g.strokeRect(a.claim[0] * S + 4, (N - 1 - a.claim[1]) * S + 4, S - 8, S - 8);
  }
  for (const [sid, a] of Object.entries(state.agents || {})) {       // UAVs at their reported lat/lon
    if (a.lat == null || a.lon == null) continue;
    const u = META.uavs.find((q) => String(q.sys) === String(sid)); if (!u) continue;
    const [x, y] = toPx(a.lat, a.lon, N);
    g.globalAlpha = a.stale ? 0.4 : 1;
    g.fillStyle = u.color; g.strokeStyle = "#05080c"; g.lineWidth = 3;
    g.beginPath(); g.arc(x, y, 15, 0, 7); g.fill(); g.stroke();
    if (a.hdg != null) { const r = ((a.hdg - 90) * Math.PI) / 180; g.beginPath(); g.moveTo(x, y); g.lineTo(x + Math.cos(r) * 28, y + Math.sin(r) * 28); g.strokeStyle = u.color; g.lineWidth = 3; g.stroke(); }
    g.fillStyle = "#05080c"; g.font = "bold 16px sans-serif"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(String(u.n), x, y + 1);
    g.fillStyle = "#fff"; g.font = "bold 13px sans-serif"; g.fillText(u.label, x, y - 25);
    g.globalAlpha = 1;
  }
}

function render(state) {
  const w = state.world || {}, N = w.N || META.N, agents = state.agents || {};
  const vis = bitmapToSet(w.vis, N);
  const live = META.uavs.filter((u) => agents[u.sys] && !agents[u.sys].stale).length;
  const conn = el("conn");
  conn.textContent = state.ended ? "run ended: " + state.ended : (live === META.uavs.length ? "4/4 UAVs reporting" : live + "/" + META.uavs.length + " UAVs reporting");
  conn.className = "badge " + (live === META.uavs.length && !state.ended ? "badge-ok" : "badge-bad");
  const t = w.t != null ? w.t : null;
  el("stats").innerHTML = [
    ["Elapsed sim time", t == null ? "—" : Math.floor(t / 60) + ":" + String(Math.floor(t % 60)).padStart(2, "0")],
    ["Coverage", w.visited_frac != null ? (100 * w.visited_frac).toFixed(0) + "%" : "—"],
    ["Cells visited", w.visited_frac != null ? Math.round(w.visited_frac * N * N) + " / " + N * N : "—"],
    ["Burning / burnt", (w.burning_cnt ?? "—") + " / " + (w.burnt_cnt ?? "—")],
  ].map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  let st = "";
  if (!state.fog_connected && !state.ended) st = live ? "Waiting for the ground station (fog)…" : "Starting up: ArduPilot SITL instances are booting and the UAVs are connecting. This takes about two minutes.";
  else if (t !== null && t < 1 && !state.ended && w.burning_cnt === 0) st = "Ground station is waiting for all four UAVs to reach flying altitude before it starts the clock and the fire.";
  el("status").className = "status" + (st ? " bad" : ""); el("status").textContent = st;
  el("fleet").innerHTML = META.uavs.map((u) => {
    const a = agents[u.sys] || {}, own = u.cells.filter((c) => vis.has(c[0] + "," + c[1])).length;
    const spd = a.vx != null && a.vy != null ? Math.hypot(a.vx, a.vy).toFixed(1) + " m/s" : "—";
    return `<div class="uav${a.stale ? " stale" : ""}"><span class="dot" style="background:${u.color}"></span><div><div class="name">${esc(u.label)} <span style="color:${u.color}">●</span> <span class="kv">${esc(u.region)}</span></div>
      <div class="kv">${a.lat != null ? "SITL position " + a.lat.toFixed(5) + ", " + a.lon.toFixed(5) : "not yet connected"} · alt ${a.alt != null ? a.alt.toFixed(0) + " m" : "—"} · ${spd} · mode ${esc(a.mode ?? "—")}</div>
      <div class="kv">region cells visited (by any UAV) ${own} / ${u.cells.length} · MAVLink sysid ${u.sys} · last packet ${a.age_s != null ? a.age_s + " s ago" : "—"}</div></div></div>`;
  }).join("");
  const dets = state.detections || [];
  el("dets").innerHTML = dets.length
    ? "<b>Reported by the running system:</b><br>" + dets.slice(-8).reverse().map((d) => `UAV ${esc(d.sys)} · cell (${d.i},${d.j}) · ${esc(d.cls)} ${Number(d.conf).toFixed(2)}`).join("<br>")
    : (META.perception ? "No detections received yet." : "None. This demonstration runs <b>no smoke detector</b> (no neural network), so nothing is shown here; a detection would appear only if the system produced one.");
  el("truth").innerHTML = META.truth.map((t) => `<p style="margin:0 0 6px"><b>${esc(t[0])}</b> ${esc(t[1])}</p>`).join("");
}

async function tick() {
  try {
    STATE = await (await fetch("/api/state", { cache: "no-store" })).json();
    ingestTrails(STATE, (STATE.world && STATE.world.N) || META.N);
    draw(STATE); render(STATE);
  } catch (e) { el("conn").textContent = "dashboard unreachable"; el("conn").className = "badge badge-bad"; }
}
(async () => {
  try { META = await (await fetch("/api/demo")).json(); } catch (e) { META = null; }
  if (!META || !META.uavs) { document.body.innerHTML = "<p style='padding:20px'>This page needs <code>python3 demo_multi.py</code> (the launcher supplies the demo metadata).</p>"; return; }
  try {                                // history so far (positions the dashboard has actually received), for a page opened mid-run
    const h = await (await fetch("/api/trails", { cache: "no-store" })).json();
    for (const [sid, pts] of Object.entries(h)) trails.set(sid, pts.map((q) => toPx(q[1], q[2])));
  } catch (e) { /* no history: the trail simply starts now */ }
  if (META.banner) { el("banner").innerHTML = esc(META.banner[0]) + " <span>" + esc(META.banner[1]) + "</span>"; }
  setInterval(tick, 250); tick();
})();
