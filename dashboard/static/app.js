/* dashboard/static/app.js
 *
 * Polls /api/state and draws. Deliberately dependency-free: no build step, no
 * CDN, works on a laptop with no internet in a demo room.
 *
 * Grid convention matches the Python side: i = east (x), j = north (y), and
 * (0,0) is the south-west corner. Canvas y grows downward, so every draw
 * flips j -> (N-1-j). Getting this wrong is the classic wildfire-sim bug
 * where the fire spreads the right shape in the wrong corner.
 */

const POLL_MS = 250;

const el = (id) => document.getElementById(id);
const canvas = el("map");
const ctx = canvas.getContext("2d");

let latest = null;
let frozen = false;

/* ---------------- helpers ---------------- */

function pct(x) {
  return ((x || 0) * 100).toFixed(1) + "%";
}

function fmt(v, digits, suffix) {
  if (v === null || v === undefined || v === "") return "—";
  const n = Number(v);
  if (!isFinite(n)) return "—";
  return n.toFixed(digits === undefined ? 1 : digits) + (suffix || "");
}

/* Fog packs coverage as a hex bitmap with bit (i*N + j). BigInt because a
 * 12x12 grid already needs 144 bits and JS numbers top out at 53. */
function bitmapToSet(hex, N) {
  const out = new Set();
  if (!hex) return out;
  let bits;
  try {
    bits = BigInt("0x" + hex);
  } catch (e) {
    return out;
  }
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if ((bits >> BigInt(i * N + j)) & 1n) out.add(i + "," + j);
    }
  }
  return out;
}

/* Fire arrives as one N*N digit string, row-major over i then j. */
function fireAt(str, N, i, j) {
  if (!str) return 0;
  const c = str.charCodeAt(i * N + j);
  return isNaN(c) ? 0 : c - 48;
}

function headingFrom(a) {
  if (a.hdg !== null && a.hdg !== undefined) return Number(a.hdg);
  // fall back to course over ground from the NED velocity
  if (
    a.vx === null ||
    a.vx === undefined ||
    a.vy === null ||
    a.vy === undefined
  )
    return null;
  if (Math.hypot(a.vx, a.vy) < 0.3) return null; // hovering: heading is noise
  return ((Math.atan2(a.vy, a.vx) * 180) / Math.PI + 360) % 360;
}

/* ---------------- drawing ---------------- */

function draw(state) {
  const w = state.world || {};
  const N = w.N || 10;
  const S = canvas.width / N;

  const showCoverage = el("layer-coverage").checked;
  const showFire = el("layer-fire").checked;
  const showBelief = el("layer-belief").checked;
  const showGrid = el("layer-grid").checked;

  ctx.fillStyle = "#05080c";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const visited = bitmapToSet(w.vis, N);
  const done = bitmapToSet(w.done, N);
  // belief arrives sparse: [[i, j, p], ...]
  const belief = new Map();
  for (const [bi, bj, p] of w.belief || []) belief.set(bi + "," + bj, p);

  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const x = i * S;
      const y = (N - 1 - j) * S; // flip: canvas y grows downward
      const key = i + "," + j;

      if (showCoverage && visited.has(key)) {
        ctx.fillStyle = done.has(key)
          ? "rgba(74,158,255,0.34)"
          : "rgba(45,111,181,0.20)";
        ctx.fillRect(x, y, S, S);
      }

      if (showBelief && belief.has(key)) {
        ctx.fillStyle =
          "rgba(255,198,26," + Math.min(0.72, belief.get(key)) + ")";
        ctx.fillRect(x + S * 0.18, y + S * 0.18, S * 0.64, S * 0.64);
      }

      if (showFire) {
        const s = fireAt(w.fire, N, i, j);
        if (s === 1) {
          ctx.fillStyle = "rgba(255,87,34,0.85)";
          ctx.fillRect(x, y, S, S);
        } else if (s === 2) {
          ctx.fillStyle = "rgba(90,97,105,0.55)";
          ctx.fillRect(x, y, S, S);
        }
      }
    }
  }

  if (showGrid) {
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    for (let k = 0; k <= N; k++) {
      ctx.beginPath();
      ctx.moveTo(k * S, 0);
      ctx.lineTo(k * S, canvas.height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, k * S);
      ctx.lineTo(canvas.width, k * S);
      ctx.stroke();
    }
  }

  // claimed target cells, drawn under the drones
  const agents = state.agents || {};
  ctx.strokeStyle = "rgba(53,208,127,0.45)";
  ctx.setLineDash([4, 3]);
  for (const a of Object.values(agents)) {
    if (!a.claim || a.stale) continue;
    const [ci, cj] = a.claim;
    ctx.strokeRect(ci * S + 2, (N - 1 - cj) * S + 2, S - 4, S - 4);
  }
  ctx.setLineDash([]);

  // drones
  for (const [sid, a] of Object.entries(agents)) {
    if (!a.cell) continue;
    const [ci, cj] = a.cell;
    const cx = (ci + 0.5) * S;
    const cy = (N - 1 - cj + 0.5) * S;
    const r = Math.max(5, S * 0.22);

    ctx.fillStyle = a.stale ? "rgba(244,86,74,0.85)" : "rgba(53,208,127,0.95)";
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.fill();

    const hdg = headingFrom(a);
    if (hdg !== null) {
      const rad = ((hdg - 90) * Math.PI) / 180; // 0 deg = north = up
      ctx.strokeStyle = "#0d1117";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(rad) * r * 1.9, cy + Math.sin(rad) * r * 1.9);
      ctx.stroke();
    }

    ctx.fillStyle = "#05080c";
    ctx.font = "bold " + Math.max(9, S * 0.26) + "px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(a.name || sid, cx, cy);
  }
}

/* ---------------- panels ---------------- */

function renderStats(state) {
  const w = state.world || {};
  const cells = [
    ["coverage", pct(w.visited_frac)],
    ["traversed", pct(w.traverse_frac)],
    ["burning", w.burning_cnt ?? "—"],
    ["burnt", w.burnt_cnt ?? "—"],
    ["detections", (w.det_true ?? 0) + " / " + (w.det_total ?? 0)],
    ["precision", w.det_total ? pct(w.precision) : "—"],
    ["lag", w.detection_lag_s != null ? fmt(w.detection_lag_s, 1, " s") : "—"],
  ];
  el("stats").innerHTML = cells
    .map(
      ([k, v]) =>
        `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`,
    )
    .join("");
}

function renderPerception(state) {
  const dets = state.detections || [];
  const recent = dets.slice(-25);
  const best = { smoke: null, fire: null };
  for (const d of recent) {
    const c = (d.cls || "").toLowerCase();
    if (c in best && (best[c] === null || d.conf > best[c].conf)) best[c] = d;
  }
  const top =
    [best.smoke, best.fire]
      .filter(Boolean)
      .sort((a, b) => b.conf - a.conf)[0] || null;

  const line = (label, d, cls) => {
    const v = d ? pct(d.conf) : "—";
    const colour = d ? cls : "dim";
    const width = d ? Math.round(d.conf * 100) : 0;
    const bar = `<div class="bar"><i style="width:${width}%;background:var(--${cls === "fire" ? "burning" : "belief"})"></i></div>`;
    return `<div class="big"><div><div class="label">${label}</div>${bar}</div>
            <div class="value ${colour}">${v}</div></div>`;
  };

  let html =
    line("Smoke", best.smoke, "smoke") + line("Fire", best.fire, "fire");
  html += `<div class="big"><span class="label">Status</span>
           <span class="value ${top ? "fire" : "dim"}" style="font-size:16px">
           ${top ? "DETECTED" : "Clear"}</span></div>`;
  if (top) {
    html += `<div class="big"><span class="label">Cell</span>
             <span class="value dim" style="font-size:16px">${top.i}, ${top.j}</span></div>`;
    html += `<div class="big"><span class="label">Reported by</span>
             <span class="value dim" style="font-size:16px">UAV ${top.sys}</span></div>`;
    html += `<div class="big"><span class="label">Model</span>
             <span class="value dim" style="font-size:14px">${top.model || "unknown"}</span></div>`;
  }
  el("perception").innerHTML = html;
}

function renderFeed(state) {
  const w = state.world || {};
  const scored = w.recent_det || [];
  if (!scored.length) {
    el("feed").innerHTML =
      '<div class="empty">Waiting for the network to report.</div>';
    return;
  }
  el("feed").innerHTML = scored
    .slice()
    .reverse()
    .map((d) => {
      const ok = d.correct
        ? '<span class="tick">&#10003; true</span>'
        : '<span class="cross">&#10007; false</span>';
      return `<div class="row"><span class="t">${fmt(d.t, 1, "s")}</span>
            <span>UAV ${d.sys}</span><span>${d.cls} ${pct(d.conf)}</span>
            <span>(${d.i},${d.j})</span>${ok}</div>`;
    })
    .join("");
}

function renderFleet(state) {
  const agents = state.agents || {};
  const ids = Object.keys(agents).sort((a, b) => Number(a) - Number(b));
  if (!ids.length) {
    el("fleet").innerHTML = '<div class="empty">No UAVs on the bus.</div>';
    return;
  }
  el("fleet").innerHTML = ids
    .map((sid) => {
      const a = agents[sid];
      const hdg = headingFrom(a);
      const det = a.det
        ? `<span class="pill">${a.det.cls} ${pct(a.det.conf)}</span>`
        : "";
      return `<div class="fleet-row ${a.stale ? "stale" : ""}">
      <div class="name"><span>UAV ${a.name || sid}</span>${det}</div>
      <div class="kv">
        <span>cell <b>${a.cell ? a.cell.join(",") : "—"}</b></span>
        <span>alt <b>${fmt(a.alt, 1, " m")}</b></span>
        <span>hdg <b>${hdg === null ? "—" : fmt(hdg, 0, "&deg;")}</b></span>
        <span>batt <b>${a.batt == null || a.batt < 0 ? "—" : a.batt + "%"}</b></span>
        <span>mode <b>${a.mode || "—"}</b></span>
        <span>mission <b>${a.mission || "—"}</b></span>
        <span>target <b>${a.claim ? a.claim.join(",") : "—"}</b></span>
        <span>age <b>${fmt(a.age_s, 1, " s")}</b></span>
      </div></div>`;
    })
    .join("");
}

function renderHeader(state) {
  const conn = el("conn");
  if (state.fog_connected) {
    conn.textContent = "live";
    conn.className = "badge badge-ok";
  } else {
    conn.textContent = state.packets ? "fog offline" : "waiting for fog";
    conn.className = "badge badge-bad";
  }
  const w = state.world || {};
  el("clock").textContent = "t = " + fmt(w.t, 1, " s");
  el("oracle-badge").hidden = !w.oracle;

  if (state.ended && !frozen) {
    frozen = true;
    const box = el("ended");
    box.hidden = false;
    box.textContent = "Experiment ended: " + state.ended;
  }
}

/* ---------------- loop ---------------- */

async function tick() {
  try {
    const r = await fetch("/api/state", { cache: "no-store" });
    latest = await r.json();
    renderHeader(latest);
    draw(latest);
    renderStats(latest);
    renderPerception(latest);
    renderFeed(latest);
    renderFleet(latest);
  } catch (e) {
    const conn = el("conn");
    conn.textContent = "server down";
    conn.className = "badge badge-bad";
  }
}

for (const id of [
  "layer-coverage",
  "layer-fire",
  "layer-belief",
  "layer-grid",
]) {
  el(id).addEventListener("change", () => {
    if (latest) draw(latest);
  });
}

tick();
setInterval(tick, POLL_MS);
