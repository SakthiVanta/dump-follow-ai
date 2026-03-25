"""
Review & Active Learning UI.

Routes
------
GET  /review                     → HTML dashboard (frame review table)
GET  /api/v1/review/frames       → JSON list of all frames
GET  /api/v1/review/pending      → JSON list of frames needing review
GET  /api/v1/review/frames/{id}  → JSON single frame
GET  /api/v1/review/image/{id}   → JPEG image of the frame patch
POST /api/v1/review/frames/{id}/label  → Confirm / correct label
POST /api/v1/review/frames/{id}/skip   → Skip (mark reviewed, no label)
POST /api/v1/review/train        → Force retrain tiny NN now
GET  /api/v1/review/stats        → Collection + model stats
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from robot.learning.tiny_nn import ACTIONS

router = APIRouter()

# ── Request models ──────────────────────────────────────────────────────────


class LabelRequest(BaseModel):
    action: str  # must be in ACTIONS


# ── HTML Dashboard ──────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Tiny AI Robot — Active Learning Review</title>
<style>
  :root{--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;
        --muted:#8b949e;--accent:#58a6ff;--green:#3fb950;--red:#f85149;
        --yellow:#d29922;--purple:#bc8cff}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:monospace;font-size:13px;padding:16px}
  h1{color:var(--accent);font-size:18px;margin-bottom:16px}
  h2{color:var(--muted);font-size:13px;margin-bottom:8px;text-transform:uppercase;letter-spacing:.08em}
  .stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}
  .stat{background:var(--surface);border:1px solid var(--border);border-radius:6px;
        padding:10px 16px;min-width:100px}
  .stat .val{font-size:22px;font-weight:700;color:var(--accent)}
  .stat .lbl{color:var(--muted);font-size:11px;margin-top:2px}
  .toolbar{display:flex;gap:8px;margin-bottom:14px;align-items:center;flex-wrap:wrap}
  button{background:var(--surface);color:var(--text);border:1px solid var(--border);
         border-radius:4px;padding:5px 12px;cursor:pointer;font-size:12px;font-family:monospace}
  button:hover{background:var(--border)}
  button.primary{background:var(--accent);color:#000;border-color:var(--accent)}
  button.primary:hover{opacity:.85}
  button.danger{background:var(--red);color:#fff;border-color:var(--red)}
  button.success{background:var(--green);color:#000;border-color:var(--green)}
  .badge{display:inline-block;border-radius:3px;padding:1px 6px;font-size:11px;font-weight:600}
  .badge-low{background:#5c2e2e;color:var(--red)}
  .badge-med{background:#4a3800;color:var(--yellow)}
  .badge-high{background:#1a3a1a;color:var(--green)}
  .badge-done{background:#1e2a3a;color:var(--accent)}
  table{width:100%;border-collapse:collapse;background:var(--surface)}
  th{background:#1c2128;color:var(--muted);text-align:left;padding:8px 10px;
     border-bottom:1px solid var(--border);position:sticky;top:0;z-index:1}
  td{padding:7px 10px;border-bottom:1px solid var(--border);vertical-align:middle}
  tr:hover td{background:#1c2128}
  img.thumb{width:56px;height:56px;object-fit:cover;border-radius:3px;border:1px solid var(--border)}
  .action-sel{background:var(--bg);color:var(--text);border:1px solid var(--border);
              border-radius:3px;padding:3px 6px;font-family:monospace;font-size:12px}
  .row-done td{opacity:.45}
  .filter-bar{display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
  input[type=text]{background:var(--surface);color:var(--text);border:1px solid var(--border);
                   border-radius:4px;padding:4px 8px;font-family:monospace;font-size:12px}
  .model-box{background:var(--surface);border:1px solid var(--border);border-radius:6px;
             padding:10px 14px;margin-bottom:16px;display:flex;gap:24px;align-items:center;flex-wrap:wrap}
  .model-box .acc{font-size:20px;font-weight:700;color:var(--purple)}
  .pagination{display:flex;gap:6px;align-items:center;margin-top:12px}
  #toast{position:fixed;bottom:20px;right:20px;background:var(--green);color:#000;
         border-radius:4px;padding:8px 16px;font-size:13px;display:none;z-index:999}
  #loading{color:var(--muted);margin:20px 0}
</style>
</head>
<body>
<h1>&#129302; Tiny AI Robot — Active Learning Review</h1>

<div class="model-box" id="modelBox">
  <div><div class="acc" id="modelAcc">—</div><div style="color:var(--muted);font-size:11px">Model Accuracy</div></div>
  <div><span id="modelRetrains" style="color:var(--purple);font-size:16px;font-weight:700">0</span><div style="color:var(--muted);font-size:11px">Retrains</div></div>
  <div><span id="modelNewLabels" style="color:var(--yellow);font-size:16px;font-weight:700">0</span><div style="color:var(--muted);font-size:11px">New labels until retrain</div></div>
  <button class="primary" onclick="triggerRetrain()" id="retrainBtn">&#9889; Retrain Now</button>
  <span id="retrainStatus" style="color:var(--muted);font-size:11px"></span>
</div>

<div class="stats" id="statsRow"></div>

<div class="toolbar">
  <button onclick="loadFrames('pending')" id="btnPending">&#128681; Pending Review</button>
  <button onclick="loadFrames('all')" id="btnAll">&#128196; All Frames</button>
  <button onclick="loadFrames('pending')" title="Refresh">&#8635; Refresh</button>
  <label style="color:var(--muted);font-size:11px">
    <input type="checkbox" id="autoRefresh" onchange="toggleAutoRefresh()"> Auto-refresh 5s
  </label>
</div>

<div class="filter-bar">
  <input type="text" id="filterText" placeholder="Filter by action..." oninput="applyFilter()"/>
  <span style="color:var(--muted)" id="filteredCount"></span>
</div>

<div id="loading" style="display:none">Loading…</div>
<div style="overflow-x:auto">
<table id="framesTable">
  <thead><tr>
    <th>ID</th><th>Time</th><th>Patch</th>
    <th>Predicted</th><th>Confidence</th>
    <th>Features</th><th>Status</th><th>Action</th>
  </tr></thead>
  <tbody id="framesBody"></tbody>
</table>
</div>
<div class="pagination" id="pagination"></div>

<div id="toast"></div>

<script>
const ACTIONS = ["LEFT","RIGHT","FORWARD","STOP"];
let currentMode = "pending";
let allRows = [];
let autoTimer = null;

async function api(path, opts){
  const r = await fetch(path, opts);
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}

function confBadge(c){
  const pct = Math.round(c*100);
  let cls = c>=0.7?"badge-high":c>=0.5?"badge-med":"badge-low";
  return `<span class="badge ${cls}">${pct}%</span>`;
}

function timeFmt(ts){
  if(!ts) return "—";
  const d = new Date(ts);
  return d.toLocaleTimeString()+' '+d.toLocaleDateString();
}

function actionSelect(frameId, current){
  const opts = ACTIONS.map(a=>
    `<option value="${a}" ${a===current?"selected":""}>${a}</option>`
  ).join("");
  return `<select class="action-sel" id="sel_${frameId}">${opts}</select>`;
}

function toast(msg, color="#3fb950"){
  const el = document.getElementById("toast");
  el.textContent = msg; el.style.background = color;
  el.style.display = "block";
  setTimeout(()=>{el.style.display="none"}, 2500);
}

async function loadStats(){
  try{
    const s = await api("/api/v1/review/stats");
    const db = s.db || {};
    document.getElementById("statsRow").innerHTML = [
      ["Total",    db.total||0],
      ["Pending",  db.pending||0,  "badge-low"],
      ["Reviewed", db.reviewed||0, "badge-done"],
      ["Labeled",  db.labeled||0,  "badge-high"],
      ["Avg Conf", db.avg_confidence ? (db.avg_confidence*100).toFixed(0)+"%" : "—"],
    ].map(([l,v,c])=>`
      <div class="stat">
        <div class="val ${c||""}" style="${c?"color:inherit":""}">${v}</div>
        <div class="lbl">${l}</div>
      </div>`
    ).join("");

    const m = s.model || {};
    document.getElementById("modelAcc").textContent =
      (m.last_accuracy > 0) ? (m.last_accuracy*100).toFixed(1)+"%" : "untrained";
    document.getElementById("modelRetrains").textContent = m.retrain_count||0;
    const need = Math.max(0,(m.retrain_threshold||20) - (m.new_labels_since_last_train||0));
    document.getElementById("modelNewLabels").textContent = need+" more";
  } catch(e){ console.error(e); }
}

async function loadFrames(mode){
  currentMode = mode;
  document.getElementById("btnPending").style.fontWeight = mode==="pending"?"bold":"normal";
  document.getElementById("btnAll").style.fontWeight    = mode==="all"?"bold":"normal";
  document.getElementById("loading").style.display = "block";
  document.getElementById("framesTable").style.display = "none";
  try{
    const endpoint = mode==="pending"
      ? "/api/v1/review/pending"
      : "/api/v1/review/frames";
    const data = await api(endpoint);
    allRows = data.frames || data;
    renderTable(allRows);
    applyFilter();
    await loadStats();
  } catch(e){ toast("Error: "+e.message,"#f85149"); }
  finally{
    document.getElementById("loading").style.display = "none";
    document.getElementById("framesTable").style.display = "table";
  }
}

function renderTable(rows){
  const tbody = document.getElementById("framesBody");
  tbody.innerHTML = rows.map(f=>{
    const done = f.is_reviewed===1;
    const feats = f.features_json ? JSON.parse(f.features_json) : null;
    const featStr = feats
      ? `<span style="color:var(--muted);font-size:10px">xc=${feats[0].toFixed(2)} yc=${feats[1].toFixed(2)}<br>area=${feats[2].toFixed(3)} dx=${feats[3].toFixed(3)} dy=${feats[4].toFixed(3)}</span>`
      : "<span style='color:var(--muted)'>—</span>";
    const statusBadge = done
      ? `<span class="badge badge-done">&#10003; ${f.confirmed_action||"skipped"}</span>`
      : `<span class="badge badge-low">&#128681; pending</span>`;
    const actionCol = done
      ? `<button onclick="undoFrame(${f.id})" style="font-size:10px;padding:2px 6px">undo</button>`
      : `<div style="display:flex;gap:4px;align-items:center;flex-wrap:wrap">
          ${actionSelect(f.id, f.predicted_action)}
          <button class="success" onclick="confirmLabel(${f.id})">&#10003; Confirm</button>
          <button onclick="skipFrame(${f.id})">skip</button>
        </div>`;
    return `<tr id="row_${f.id}" class="${done?"row-done":""}">
      <td>${f.id}</td>
      <td style="white-space:nowrap">${timeFmt(f.timestamp)}</td>
      <td><img class="thumb" src="/api/v1/review/image/${f.id}" loading="lazy" onerror="this.style.opacity='.2'"/></td>
      <td><span class="badge" style="background:#2a2a3a;color:var(--purple)">${f.predicted_action||"—"}</span>
          <br><span style="color:var(--muted);font-size:10px">${f.source||""}</span></td>
      <td>${confBadge(f.confidence||0)}</td>
      <td>${featStr}</td>
      <td>${statusBadge}</td>
      <td>${actionCol}</td>
    </tr>`;
  }).join("");
  document.getElementById("filteredCount").textContent = rows.length+" rows";
}

function applyFilter(){
  const q = document.getElementById("filterText").value.toLowerCase();
  const filtered = q ? allRows.filter(f=>
    (f.predicted_action||"").toLowerCase().includes(q) ||
    (f.confirmed_action||"").toLowerCase().includes(q)
  ) : allRows;
  renderTable(filtered);
}

async function confirmLabel(frameId){
  const sel = document.getElementById(`sel_${frameId}`);
  const action = sel ? sel.value : null;
  if(!action){ toast("Select an action first","#f85149"); return; }
  try{
    await api(`/api/v1/review/frames/${frameId}/label`,{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({action})
    });
    toast(`&#10003; Frame ${frameId} labeled: ${action}`);
    loadFrames(currentMode);
  } catch(e){ toast("Error: "+e.message,"#f85149"); }
}

async function skipFrame(frameId){
  try{
    await api(`/api/v1/review/frames/${frameId}/skip`,{method:"POST"});
    toast(`Frame ${frameId} skipped`,"#8b949e");
    loadFrames(currentMode);
  } catch(e){ toast("Error: "+e.message,"#f85149"); }
}

async function undoFrame(frameId){
  // Re-open a reviewed frame (just mark unreviewed)
  try{
    await api(`/api/v1/review/frames/${frameId}/undo`,{method:"POST"});
    toast(`Frame ${frameId} re-opened`,"#d29922");
    loadFrames(currentMode);
  } catch(e){ toast("Error: "+e.message,"#f85149"); }
}

async function triggerRetrain(){
  const btn = document.getElementById("retrainBtn");
  const status = document.getElementById("retrainStatus");
  btn.disabled = true; btn.textContent = "Training…";
  status.textContent = "";
  try{
    const r = await fetch("/api/v1/review/train",{method:"POST"});
    if(!r.ok){
      const err = await r.json().catch(()=>({detail:r.statusText}));
      const msg = err.detail || r.statusText;
      toast(msg, "#f85149");
      status.textContent = msg;
      return;
    }
    const data = await r.json();
    const acc = (data.accuracy*100).toFixed(1)+"%";
    toast(`&#9889; Retrained! Accuracy: ${acc}`,"#bc8cff");
    status.textContent = `accuracy: ${acc}  samples: ${data.samples}`;
    loadStats();
  } catch(e){ toast("Train error: "+e.message,"#f85149"); }
  finally{ btn.disabled=false; btn.textContent="&#9889; Retrain Now"; }
}

function toggleAutoRefresh(){
  if(autoTimer){ clearInterval(autoTimer); autoTimer=null; return; }
  autoTimer = setInterval(()=>loadFrames(currentMode), 5000);
}

// ── Boot ────────────────────────────────────────────────────────────────────
loadFrames("pending");
loadStats();
</script>
</body>
</html>
"""


# ── API Routes ──────────────────────────────────────────────────────────────

@router.get("/review", response_class=HTMLResponse, include_in_schema=False)
async def review_ui():
    return HTMLResponse(_HTML)


@router.get("/api/v1/review/frames")
async def list_all_frames(request: Request, limit: int = 200, offset: int = 0):
    store = request.app.state.robot.label_store
    frames = store.get_all_frames(limit=limit, offset=offset)
    return {"frames": frames, "total": len(frames)}


@router.get("/api/v1/review/pending")
async def list_pending(request: Request):
    store = request.app.state.robot.label_store
    frames = store.get_pending_review(limit=200)
    return {"frames": frames, "total": len(frames)}


@router.get("/api/v1/review/frames/{frame_id}")
async def get_frame(frame_id: int, request: Request):
    store = request.app.state.robot.label_store
    frame = store.get_frame(frame_id)
    if not frame:
        raise HTTPException(status_code=404, detail="Frame not found")
    return frame


@router.get("/api/v1/review/image/{frame_id}")
async def get_image(frame_id: int, request: Request):
    """Serve the JPEG patch for a frame."""
    store = request.app.state.robot.label_store
    frame = store.get_frame(frame_id)
    if not frame:
        raise HTTPException(status_code=404, detail="Frame not found")
    path = Path(frame["image_path"])
    if not path.exists():
        # Return a tiny grey placeholder
        raise HTTPException(status_code=404, detail="Image file not found")
    return Response(content=path.read_bytes(), media_type="image/jpeg")


@router.post("/api/v1/review/frames/{frame_id}/label")
async def label_frame(frame_id: int, body: LabelRequest, request: Request):
    if body.action not in ACTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action '{body.action}'. Must be one of {ACTIONS}",
        )
    store = request.app.state.robot.label_store
    ok = store.confirm_label(frame_id, body.action)
    if not ok:
        raise HTTPException(status_code=404, detail="Frame not found or invalid action")

    # Opportunistic retrain check
    trainer = getattr(request.app.state.robot, "active_trainer", None)
    if trainer:
        trainer.check_and_retrain()

    return {"status": "labeled", "frame_id": frame_id, "action": body.action}


@router.post("/api/v1/review/frames/{frame_id}/skip")
async def skip_frame(frame_id: int, request: Request):
    store = request.app.state.robot.label_store
    store.skip_frame(frame_id)
    return {"status": "skipped", "frame_id": frame_id}


@router.post("/api/v1/review/frames/{frame_id}/undo")
async def undo_frame(frame_id: int, request: Request):
    """Re-open a reviewed frame for re-labelling."""
    store = request.app.state.robot.label_store
    store._con.execute(
        "UPDATE frames SET is_reviewed=0, needs_review=1, confirmed_action=NULL WHERE id=?",
        (frame_id,),
    )
    store._con.commit()
    return {"status": "reopened", "frame_id": frame_id}


@router.post("/api/v1/review/train")
async def trigger_retrain(request: Request):
    """Force an immediate retraining run."""
    trainer = getattr(request.app.state.robot, "active_trainer", None)
    if trainer is None:
        raise HTTPException(status_code=503, detail="Active trainer not initialised")

    store = request.app.state.robot.label_store
    X, y = store.get_training_data()

    if len(X) < 4:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough labeled samples to train ({len(X)} confirmed, need ≥ 4). "
                   "Go to /review, confirm some frame labels first.",
        )

    model = trainer.force_retrain()
    acc = model.accuracy(X, y)
    return {
        "status": "trained",
        "accuracy": round(acc, 4),
        "samples": len(X),
        "retrain_count": trainer.retrain_count,
    }


@router.get("/api/v1/review/stats")
async def review_stats(request: Request):
    store = request.app.state.robot.label_store
    trainer = getattr(request.app.state.robot, "active_trainer", None)
    collector = getattr(request.app.state.robot, "collector", None)

    return {
        "db": store.get_stats(),
        "model": trainer.stats() if trainer else {},
        "collector": {
            "total_collected": collector.total_collected if collector else 0,
            "confidence_threshold": collector.confidence_threshold if collector else None,
            "sample_rate": collector.sample_rate if collector else None,
        },
    }
