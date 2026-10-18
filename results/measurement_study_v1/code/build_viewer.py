#!/usr/bin/env python3
"""Builds the offline review viewers.

Stage A (public)  : one candidate image at a time, blinded.
Stage B (PRIVATE) : matched original reference shown beside each candidate.

Design constraints that drove the implementation:

* Must run from `file://` after unzipping, with no account, no network and no server.
  Therefore the item manifest is EMBEDDED in the HTML: `fetch()` of a sibling JSON is
  blocked by CORS under `file://` in Chrome, so a fetch-based viewer would silently fail
  for reviewers. Images are plain <img src="images/..."> which do work on file://.
* No external CSS/JS/fonts. Everything is inline.
* Nothing in the public bundle may reveal arm, method, prompt, seed, source filename or
  any machine score. Item ids are neutral and opaque.
* Reviewer order is derived from the reviewer's own alias, independently of any private
  arm mapping, and is reproducible; three named variants are also offered.
"""
import os, sys, json, hashlib, argparse, shutil, html

ROOT = "/data/bijaypandey/cuig_pilot/measurement_study_v1"

CSS = """
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:#16181d;background:#f4f5f7}
header{position:sticky;top:0;z-index:20;background:#fff;border-bottom:1px solid #d8dce2;padding:10px 16px}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.grow{flex:1 1 auto}
h1{font-size:16px;margin:0;font-weight:650}
.sub{color:#5b6472;font-size:13px}
main{max-width:1100px;margin:0 auto;padding:16px}
.card{background:#fff;border:1px solid #d8dce2;border-radius:10px;padding:16px;margin-bottom:16px}
.imgwrap{background:#111;border-radius:8px;display:flex;align-items:center;justify-content:center;
  overflow:auto;max-height:72vh}
.imgwrap img{max-width:100%;max-height:72vh;display:block;transition:none}
.imgwrap.zoom{align-items:flex-start;justify-content:flex-start}
.imgwrap.zoom img{max-width:none;max-height:none;cursor:grab}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:760px){.pair{grid-template-columns:1fr}}
.pane h3{margin:0 0 6px;font-size:13px;color:#5b6472;font-weight:650;text-transform:uppercase;letter-spacing:.04em}
fieldset{border:1px solid #d8dce2;border-radius:8px;margin:14px 0 0;padding:12px}
legend{font-weight:650;font-size:14px;padding:0 6px}
label.opt{display:block;padding:9px 10px;border:1px solid #d8dce2;border-radius:7px;margin:6px 0;cursor:pointer;background:#fff}
label.opt:hover{background:#f0f3f8}
label.opt input{margin-right:9px}
label.opt.sel{background:#e8f0fe;border-color:#5b8def}
textarea{width:100%;min-height:62px;padding:8px;border:1px solid #d8dce2;border-radius:7px;font:inherit;resize:vertical}
button{font:inherit;padding:8px 14px;border-radius:7px;border:1px solid #c3c9d2;background:#fff;cursor:pointer}
button:hover{background:#f0f3f8}
button.primary{background:#1b62d6;border-color:#1b62d6;color:#fff}
button.primary:hover{background:#1853b8}
button:disabled{opacity:.45;cursor:not-allowed}
.bar{height:7px;background:#e3e7ee;border-radius:4px;overflow:hidden}
.bar>div{height:100%;background:#1b62d6;width:0}
.pill{font-size:12px;padding:2px 8px;border-radius:99px;background:#eef1f6;color:#4a5363;border:1px solid #dde2ea}
.pill.warn{background:#fff4e5;border-color:#f0cf9b;color:#8a5a00}
.pill.ok{background:#e7f6ec;border-color:#a8dcb9;color:#1d6b35}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.grid figure{margin:0;background:#fff;border:1px solid #d8dce2;border-radius:8px;overflow:hidden}
.grid img{width:100%;display:block;background:#111}
.grid figcaption{padding:7px 8px;font-size:12px;color:#444}
.note{background:#fff8e6;border:1px solid #f0d79b;border-radius:8px;padding:11px;font-size:14px}
.err{background:#fdeceb;border:1px solid #f2b5b1;border-radius:8px;padding:11px;font-size:14px;color:#8a1c14}
code{background:#eef1f6;padding:1px 5px;border-radius:4px;font-size:13px}
.hide{display:none!important}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border:1px solid #dde2ea;padding:6px 8px;text-align:left}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
"""

JS = r"""
// ---------- deterministic helpers (no library, no network) ----------
function h32(s){let h=2166136261>>>0;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)>>>0;}return h>>>0;}
function rng(seed){let x=seed>>>0||1;return function(){x^=x<<13;x>>>=0;x^=x>>17;x^=x<<5;x>>>=0;return x/4294967296;};}
function shuffle(arr,seed){const a=arr.slice(),r=rng(seed);for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}

const CFG = window.__CFG__;
let S = {alias:"", variant:"", exposure:"", items:[], idx:0, answers:{}, started:null, practiceSeen:false};

function storeKey(){ return "msv1::"+CFG.stage+"::"+CFG.package_id+"::"+(S.alias||"_"); }
function save(){ try{ localStorage.setItem(storeKey(), JSON.stringify({alias:S.alias,variant:S.variant,
  exposure:S.exposure,idx:S.idx,answers:S.answers,started:S.started,practiceSeen:S.practiceSeen})); }catch(e){} }
function load(){ try{ const r=localStorage.getItem(storeKey()); if(!r)return false; const d=JSON.parse(r);
  S.variant=d.variant||S.variant; S.exposure=d.exposure||""; S.idx=d.idx||0; S.answers=d.answers||{};
  S.started=d.started||null; S.practiceSeen=!!d.practiceSeen; return true; }catch(e){ return false; } }

// order is a pure function of (alias, variant) and NEVER of any arm/condition
function buildOrder(){
  const ids = CFG.items.map(it=>it.id);
  const vseed = {A:1013904223,B:1664525,C:22695477}[S.variant] || 1013904223;
  return shuffle(ids, (h32(S.alias)^vseed)>>>0);
}

function el(id){return document.getElementById(id);}
function show(id,on){el(id).classList.toggle("hide",!on);}

function startSession(){
  const a=el("alias").value.trim();
  if(!a){ el("gateErr").textContent="Please enter the alias your coordinator gave you."; show("gateErr",true); return; }
  if(!/^[A-Za-z0-9_-]{2,32}$/.test(a)){ el("gateErr").textContent="Alias must be 2-32 chars: letters, digits, - or _ only."; show("gateErr",true); return; }
  const ex=document.querySelector('input[name="exposure"]:checked');
  if(!ex){ el("gateErr").textContent="Please answer the prior-exposure question."; show("gateErr",true); return; }
  show("gateErr",false);
  S.alias=a; S.exposure=ex.value;
  const chosenVariant=el("variant").value;
  const had=load();
  // BUG FIX (found in browser validation): load() restored the STORED variant over the one
  // just selected at the gate, so a reviewer assigned variant B or C silently kept variant A
  // on every resume. The coordinator-assigned selection must win; answers are keyed by item
  // id, so they survive a variant change, only the presentation order moves.
  S.variant=chosenVariant; S.exposure=ex.value;
  if(!had){ S.started=new Date().toISOString(); S.answers={}; S.idx=0; }
  S.items=buildOrder(); save();
  show("gate",false); show("app",true);
  if(CFG.items.length===0){
    show("emptyPanel",true); show("review",false);
    el("btnPrev").disabled=true; el("btnNext").disabled=true;
    renderHeader(); return;
  }
  render();
}

function renderHeader(){
  el("who").textContent = S.alias ? ("alias "+S.alias+" · order "+S.variant) : "";
  const n=CFG.items.length, done=Object.keys(S.answers).filter(k=>isComplete(S.answers[k])).length;
  el("prog").textContent = n? (done+" / "+n+" answered") : "0 study items in this package";
  el("barfill").style.width = n? (100*done/n)+"%" : "0%";
  el("btnFinal").disabled = !(n>0 && done===n);
}

function isComplete(a){ if(!a)return false; if(CFG.stage==="A") return !!a.q1 && !!a.q2; return !!a.q1; }

function render(){
  renderHeader();
  const id=S.items[S.idx]; const it=CFG.items.find(x=>x.id===id);
  el("pos").textContent = "Item "+(S.idx+1)+" of "+S.items.length;
  el("neutralId").textContent = it.id;
  el("targetName").textContent = it.target;
  el("targetDef").textContent = CFG.definitions[it.target] || "";
  if(CFG.stage==="A"){
    el("imgA").src = "images/"+it.file;
    el("imgA").alt = "item "+it.id;
  } else {
    el("imgRef").src = "images/"+it.ref_file; el("imgCand").src="images/"+it.file;
  }
  zoomOff();
  const a=S.answers[id]||{};
  document.querySelectorAll('input[name="q1"]').forEach(r=>{r.checked=(a.q1===r.value); r.parentNode.classList.toggle("sel",a.q1===r.value);});
  document.querySelectorAll('input[name="q2"]').forEach(r=>{r.checked=(a.q2===r.value); r.parentNode.classList.toggle("sel",a.q2===r.value);});
  el("why").value = a.why||"";
  el("btnPrev").disabled = S.idx===0;
  el("btnNext").disabled = S.idx>=S.items.length-1;
}

function recordAndSave(){
  const id=S.items[S.idx]; const a=S.answers[id]||{};
  const q1=document.querySelector('input[name="q1"]:checked'); if(q1)a.q1=q1.value;
  const q2=document.querySelector('input[name="q2"]:checked'); if(q2)a.q2=q2.value;
  const w=el("why").value.trim(); if(w)a.why=w; else delete a.why;
  a.answered_at=new Date().toISOString();
  S.answers[id]=a; save(); renderHeader();
}

function go(d){ recordAndSave(); S.idx=Math.max(0,Math.min(S.items.length-1,S.idx+d)); save(); render(); }

// ---------- zoom / full resolution ----------
let zoomed=false;
function zoomOff(){ zoomed=false; document.querySelectorAll(".imgwrap").forEach(w=>w.classList.remove("zoom")); el("btnZoom").textContent="Zoom to full resolution"; }
function toggleZoom(){ zoomed=!zoomed; document.querySelectorAll(".imgwrap").forEach(w=>w.classList.toggle("zoom",zoomed));
  el("btnZoom").textContent = zoomed? "Fit to window" : "Zoom to full resolution"; }

// ---------- export / import ----------
function buildExport(kind){
  recordAndSave();
  const responses = CFG.items.map(it=>{
    const a=S.answers[it.id]||{};
    const rec={neutral_id:it.id, target:it.target, position:S.items.indexOf(it.id)+1};
    if(CFG.stage==="A"){
      rec.target_presence = a.q1||null;
      rec.scene_quality  = a.q2||null;
    } else {
      rec.deformation_vs_reference = a.q1||null;
    }
    rec.explanation = a.why||null;
    rec.answered = isComplete(a);
    return rec;
  });
  return {
    schema_id: CFG.schema_id,
    schema_version: CFG.schema_version,
    package_id: CFG.package_id,
    protocol_id: CFG.protocol_id,
    stage: CFG.stage,
    export_kind: kind,
    reviewer_alias: S.alias,
    order_variant: S.variant,
    prior_exposure: S.exposure,
    session_started_utc: S.started,
    exported_utc: new Date().toISOString(),
    n_items: CFG.items.length,
    n_answered: responses.filter(r=>r.answered).length,
    n_missing: responses.filter(r=>!r.answered).length,
    responses: responses
  };
}
function download(kind){
  const data=buildExport(kind);
  if(kind==="final" && data.n_missing>0){
    if(!confirm("There are still "+data.n_missing+" unanswered item(s). Export as FINAL anyway?")) return;
  }
  const blob=new Blob([JSON.stringify(data,null,1)],{type:"application/json"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob);
  a.download="msv1_"+CFG.stage+"_"+(S.alias||"anon")+"_"+kind+".json";
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  setTimeout(()=>URL.revokeObjectURL(a.href),4000);
}
function importFile(ev){
  const f=ev.target.files[0]; if(!f)return;
  const rd=new FileReader();
  rd.onload=function(){
    let d; const problems=[];
    try{ d=JSON.parse(rd.result); }catch(e){ showImport("Could not parse that file as JSON.",true); ev.target.value=""; return; }
    if(d.stage!==CFG.stage) problems.push("file is for stage "+(d.stage||"?")+", this viewer is stage "+CFG.stage);
    if(d.package_id!==CFG.package_id) problems.push("package id mismatch ("+(d.package_id||"?")+")");
    if(d.schema_version!==CFG.schema_version) problems.push("schema version mismatch ("+(d.schema_version||"?")+")");
    if(!Array.isArray(d.responses)) problems.push("no responses array");
    if(problems.length){ showImport("Refused: "+problems.join("; ")+".",true); ev.target.value=""; return; }
    const known=new Set(CFG.items.map(i=>i.id));
    let applied=0, unknown=0, dup=0; const seen=new Set();
    d.responses.forEach(r=>{
      if(!known.has(r.neutral_id)){ unknown++; return; }
      if(seen.has(r.neutral_id)){ dup++; return; }
      seen.add(r.neutral_id);
      const a={};
      if(CFG.stage==="A"){ if(r.target_presence)a.q1=r.target_presence; if(r.scene_quality)a.q2=r.scene_quality; }
      else { if(r.deformation_vs_reference)a.q1=r.deformation_vs_reference; }
      if(r.explanation)a.why=r.explanation;
      if(Object.keys(a).length){ S.answers[r.neutral_id]=a; applied++; }
    });
    if(d.reviewer_alias && d.reviewer_alias!==S.alias) problems.push("note: file alias '"+d.reviewer_alias+"' differs from this session's alias");
    save(); render();
    showImport("Imported "+applied+" answer(s)."+(unknown?" Ignored "+unknown+" unknown id(s).":"")+
      (dup?" Ignored "+dup+" duplicate(s).":"")+(problems.length?" "+problems.join("; ")+".":""), false);
    ev.target.value="";
  };
  rd.readAsText(f);
}
function showImport(msg,isErr){ const n=el("importMsg"); n.textContent=msg; n.className=isErr?"err":"note"; show("importMsg",true); }

function resetSession(){
  if(!confirm("Clear ALL saved answers for alias '"+S.alias+"' in this package? This cannot be undone."))return;
  try{ localStorage.removeItem(storeKey()); }catch(e){}
  S.answers={}; S.idx=0; save(); render(); showImport("Session cleared.",false);
}

window.addEventListener("DOMContentLoaded",function(){
  el("btnStart").onclick=startSession;
  el("btnPrev").onclick=()=>go(-1);
  el("btnNext").onclick=()=>go(1);
  el("btnZoom").onclick=toggleZoom;
  el("btnDraft").onclick=()=>download("draft");
  el("btnFinal").onclick=()=>download("final");
  el("importer").onchange=importFile;
  el("btnReset").onclick=resetSession;
  el("btnPractice").onclick=function(){ const on=el("practice").classList.contains("hide");
    show("practice",on); S.practiceSeen=S.practiceSeen||on; save(); this.textContent=on?"Hide practice examples":"Show practice examples"; };
  document.querySelectorAll('input[name="q1"],input[name="q2"]').forEach(r=>{
    r.addEventListener("change",function(){
      document.getElementsByName(this.name).forEach(o=>o.parentNode.classList.toggle("sel",o.checked));
      recordAndSave();
    });
  });
  // BUG FIX (found in browser validation): saving only on blur lost a reviewer's typed
  // note if they closed the tab while the textarea still had focus. Save on input too,
  // debounced, and once more on blur / page hide.
  let whyT=null;
  el("why").addEventListener("input",function(){ clearTimeout(whyT); whyT=setTimeout(recordAndSave,300); });
  el("why").addEventListener("blur",recordAndSave);
  window.addEventListener("pagehide",function(){ try{recordAndSave();}catch(e){} });
  document.addEventListener("visibilitychange",function(){ if(document.hidden){ try{recordAndSave();}catch(e){} } });
  document.addEventListener("keydown",e=>{
    if(e.target.tagName==="TEXTAREA"||e.target.tagName==="INPUT")return;
    if(e.key==="ArrowLeft"&&!el("btnPrev").disabled)go(-1);
    if(e.key==="ArrowRight"&&!el("btnNext").disabled)go(1);
  });
  el("pkgid").textContent=CFG.package_id;
  el("schemaid").textContent=CFG.schema_version;
});
"""


def q_block(stage):
    if stage == "A":
        return """
      <fieldset><legend>1. Is the target object present in this image?</legend>
        <label class="opt"><input type="radio" name="q1" value="present">Present — I can see the target object</label>
        <label class="opt"><input type="radio" name="q1" value="absent">Absent — the target object is not in this image</label>
        <label class="opt"><input type="radio" name="q1" value="uncertain">Uncertain — I genuinely cannot tell</label>
      </fieldset>
      <fieldset><legend>2. Is this a coherent scene?</legend>
        <label class="opt"><input type="radio" name="q2" value="coherent">Coherent scene</label>
        <label class="opt"><input type="radio" name="q2" value="severe_rendering_failure">Severe rendering failure</label>
        <label class="opt"><input type="radio" name="q2" value="unable_to_judge">Unable to judge</label>
      </fieldset>
      <fieldset><legend>3. Anything you want to add? (optional)</legend>
        <textarea id="why" placeholder="Optional. A short note on what made this easy or hard."></textarea>
      </fieldset>"""
    return """
      <fieldset><legend>1. Compared with the reference on the left, the candidate on the right shows…</legend>
        <label class="opt"><input type="radio" name="q1" value="less">Less structural deformation</label>
        <label class="opt"><input type="radio" name="q1" value="about_same">About the same structural deformation</label>
        <label class="opt"><input type="radio" name="q1" value="more">More structural deformation</label>
        <label class="opt"><input type="radio" name="q1" value="cannot_compare">Cannot fairly compare</label>
      </fieldset>
      <fieldset><legend>2. Anything you want to add? (optional)</legend>
        <textarea id="why" placeholder="Optional."></textarea>
      </fieldset>"""


def image_block(stage):
    if stage == "A":
        return '<div class="imgwrap"><img id="imgA" alt=""></div>'
    return ('<div class="pair">'
            '<div class="pane"><h3>Reference (original model)</h3><div class="imgwrap"><img id="imgRef" alt=""></div></div>'
            '<div class="pane"><h3>Candidate</h3><div class="imgwrap"><img id="imgCand" alt=""></div></div>'
            '</div>')


def guidance(stage):
    if stage == "A":
        return """
      <p>You will see one image at a time. For each, say whether a named <b>target object</b>
      is present, and whether the image is a coherent scene. Please judge only what you can
      see in the image itself.</p>
      <ul>
        <li><b>Uncertain</b> is a real answer. Use it when you genuinely cannot tell; please do
            not force a guess.</li>
        <li>A <b>severe rendering failure</b> means the image is essentially unusable: blank,
            near-uniform colour, noise or static, or a garbled texture with no depicted scene.</li>
        <li>An image is still <b>coherent</b> even if it is unusual, heavily stylised, oddly
            composed, or simply does not contain the target object.</li>
        <li>Local distortion — for example an awkward hand or face — is <b>not</b> by itself a
            whole-frame rendering failure.</li>
      </ul>
      <p>You will not be told which model or method produced any image, and there is no
      expected or desired answer. Please work independently and do not discuss items with
      other reviewers.</p>"""
    return """
      <p>Each screen shows a <b>reference image from the original model</b> on the left and a
      <b>candidate</b> on the right. They were produced from the same prompt and the same
      starting noise. Judge only <b>structural deformation</b>: whether things are built
      correctly, not whether you like the picture.</p>
      <ul>
        <li>The reference is <b>not ground truth</b>. It may itself contain defects. If it does,
            say so with "about the same" or "cannot fairly compare".</li>
        <li>A <b>coherent substitution</b> — the candidate shows something else entirely, built
            correctly — is <b>not</b> deformation.</li>
        <li>A <b>changed composition</b>, the <b>absence of a particular object</b>, and
            <b>painting style alone</b> are <b>not</b> rendering breakdown.</li>
        <li><b>Local anatomical distortion</b> is not automatically a whole-frame failure.</li>
      </ul>
      <p>This stage deliberately shows you which image is the original reference.</p>"""


def build(stage, items, definitions, practice, pkg_id, protocol_id, schema_id,
          schema_version, out_dir, title):
    os.makedirs(out_dir, exist_ok=True)
    cfg = {"stage": stage, "package_id": pkg_id, "protocol_id": protocol_id,
           "schema_id": schema_id, "schema_version": schema_version,
           "items": items, "definitions": definitions}
    pr = "".join(
        f'<figure><img src="practice/{p["file"]}" alt="practice {p["practice_id"]}" loading="lazy">'
        f'<figcaption><b>{p["practice_id"]}</b> — {html.escape(p["purpose"])}</figcaption></figure>'
        for p in practice)
    empty_banner = ("" if items else
                    '<div id="emptyPanel" class="card err"><b>This package contains no study '
                    'items.</b><br>The study panel is blocked pending an eligible data source '
                    '(see the published status note). The viewer, practice examples, questions '
                    'and export format below are complete and validated, so the package is ready '
                    'to be filled once a source exists.</div>')
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style></head><body>
<header><div class="row">
  <div class="grow"><h1>{html.escape(title)}</h1>
  <div class="sub">Package <span class="mono" id="pkgid"></span> · schema <span class="mono" id="schemaid"></span>
   · <span id="who"></span></div></div>
  <span class="pill" id="pos"></span><span class="pill" id="prog"></span>
</div><div class="bar" style="margin-top:8px"><div id="barfill"></div></div></header>
<main>

<section id="gate">
  <div class="card">
    <h2 style="margin-top:0">Before you begin</h2>
    {guidance(stage)}
    <div class="note">Your answers are saved in this browser only. Nothing is uploaded and no
      account is needed. When you are done, press <b>Export final</b> and send the downloaded
      JSON file back to your coordinator.</div>
    <fieldset><legend>Your alias</legend>
      <p class="sub">Use the short alias your coordinator gave you. Please do not type your name,
      email or any other identifying detail.</p>
      <input id="alias" maxlength="32" placeholder="e.g. R1" style="padding:8px;border:1px solid #d8dce2;border-radius:7px;font:inherit;width:220px">
      <label style="margin-left:14px">Order variant
        <select id="variant" style="padding:8px;border:1px solid #d8dce2;border-radius:7px;font:inherit">
          <option value="A">A</option><option value="B">B</option><option value="C">C</option>
        </select></label>
    </fieldset>
    <fieldset><legend>Prior exposure (please answer honestly — it is not a disqualification)</legend>
      <label class="opt"><input type="radio" name="exposure" value="none">I have not seen images from this study before</label>
      <label class="opt"><input type="radio" name="exposure" value="some">I have previously seen some related images or results</label>
      <label class="opt"><input type="radio" name="exposure" value="involved">I have been involved in producing or analysing this material</label>
    </fieldset>
    <div id="gateErr" class="err hide"></div>
    <p><button class="primary" id="btnStart">Start</button>
       <button id="btnPractice">Show practice examples</button></p>
    <div id="practice" class="hide">
      <div class="note" style="margin-bottom:10px"><b>Practice examples only.</b> These are
        instructional aids. They are not study items, they are not scored, and they are not a
        test you can pass or fail. They use a practice object (<b>bicycle</b>) that is not a
        study target.</div>
      <div class="grid">{pr}</div>
    </div>
  </div>
</section>

<section id="app" class="hide">
  {empty_banner}
  <div id="review" class="card">
    <div class="row" style="margin-bottom:10px">
      <div class="grow"><span class="pill">Item <span class="mono" id="neutralId"></span></span>
        <span class="pill">Target: <b id="targetName"></b></span></div>
      <button id="btnZoom">Zoom to full resolution</button>
    </div>
    <p class="sub" id="targetDef" style="margin-top:0"></p>
    {image_block(stage)}
    {q_block(stage)}
  </div>
  <div class="card">
    <div class="row">
      <button id="btnPrev">&larr; Previous</button>
      <button id="btnNext">Next &rarr;</button>
      <span class="grow"></span>
      <button id="btnDraft">Export draft</button>
      <button class="primary" id="btnFinal">Export final</button>
    </div>
  </div>
  <div class="card">
    <h3 style="margin-top:0">Resume, or restore from a file</h3>
    <p class="sub">Your progress is saved automatically in this browser. To continue on another
      machine, export a draft here and import it there.</p>
    <div class="row">
      <input type="file" id="importer" accept="application/json">
      <span class="grow"></span>
      <button id="btnReset">Clear my saved answers</button>
    </div>
    <div id="importMsg" class="note hide" style="margin-top:10px"></div>
  </div>
</section>
</main>
<script>window.__CFG__={json.dumps(cfg)};</script>
<script>{JS}</script>
</body></html>"""
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(doc)
    return os.path.join(out_dir, "index.html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", default=None, help="JSON file of study items (default: none)")
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--protocol-id", required=True)
    ap.add_argument("--schema-version", required=True)
    ap.add_argument("--practice", default=None, help="override practice manifest (validation only)")
    ap.add_argument("--defs", default=None, help="override target definitions (validation only)")
    a = ap.parse_args()
    items = json.load(open(a.items)) if a.items else []
    practice = json.load(open(a.practice or f"{ROOT}/assets/practice/practice_manifest.json"))["items"]
    defs = json.load(open(a.defs or f"{ROOT}/code/target_definitions.json"))
    title = ("Image review — Task A" if a.stage == "A" else
             "Image review — Task B (matched reference)")
    p = build(a.stage, items, defs, practice, a.pkg, a.protocol_id,
              "msv1-response-schema", a.schema_version, a.out, title)
    print("wrote", p, os.path.getsize(p), "bytes")
