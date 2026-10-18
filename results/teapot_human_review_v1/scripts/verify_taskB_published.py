#!/usr/bin/env python3
"""Fresh-directory verification of the published Task-B bundle."""
import hashlib, json, os, re, shutil, subprocess, sys, urllib.request, zipfile
ROOT="/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
FRESH=os.path.join(ROOT,"verify","fresh_taskB")
TAG="teapot-human-review-v1-taskB"; ZIPNAME=TAG+".zip"; ZROOT=TAG
IDENT=json.load(open(os.path.join(ROOT,"publishB","TASKB_ZIP_IDENTITY.json")))
def sha(p,c=1<<22):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(c),b""): h.update(b)
    return h.hexdigest()
def main():
    shutil.rmtree(FRESH,ignore_errors=True); os.makedirs(FRESH)
    rq=urllib.request.Request(f"https://api.github.com/repos/bijay299/XXResearch/releases/tags/{TAG}",
        headers={"User-Agent":"thrB","Accept":"application/vnd.github+json"})
    rel=json.loads(urllib.request.urlopen(rq,timeout=120).read())
    a=next(x for x in rel["assets"] if x["name"]==ZIPNAME)
    dst=os.path.join(FRESH,ZIPNAME)
    r=urllib.request.Request(a["browser_download_url"],
        headers={"User-Agent":"thrB","Accept":"application/octet-stream"})
    with urllib.request.urlopen(r,timeout=900) as resp, open(dst,"wb") as o:
        shutil.copyfileobj(resp,o,1<<22)
    rep={"tag":rel["tag_name"],"release_target_commitish":rel["target_commitish"],
         "release_html_url":rel["html_url"],"download_url":a["browser_download_url"],
         "n_assets_on_release":len(rel["assets"]),
         "asset_names":sorted(x["name"] for x in rel["assets"]),
         "expected_sha256":IDENT["sha256"],"got_sha256":sha(dst),
         "expected_bytes":IDENT["bytes"],"got_bytes":os.path.getsize(dst)}
    rep["sha256_match"]=rep["got_sha256"]==rep["expected_sha256"]
    rep["bytes_match"]=rep["got_bytes"]==rep["expected_bytes"]
    ex=os.path.join(FRESH,"extracted")
    with zipfile.ZipFile(dst) as z:
        names=[n for n in z.namelist() if not n.endswith("/")]; z.extractall(ex)
    base=os.path.join(ex,ZROOT); idir=os.path.join(base,"images")
    pngs=sorted(f for f in os.listdir(idir) if f.endswith(".png"))
    pair_ids=sorted({f.rsplit("_",1)[0] for f in pngs})
    # source-image hashes from the private B mapping (aggregate check only)
    priv=json.load(open(os.path.join(ROOT,"private","id_mapping_private.json")))["taskB_mapping"]
    ok_hash=0; role_ok=0; bad=[]
    for pid in pair_ids:
        m=priv.get(pid)
        if not m: bad.append(pid+":unmapped"); continue
        if m["reference_condition"]=="m0" and m["candidate_condition"]=="edited": role_ok+=1
        for role,exp in (("reference",m["reference_sha256"]),("candidate",m["candidate_sha256"])):
            f=os.path.join(idir,f"{pid}_{role}.png")
            if os.path.isfile(f) and sha(f)==exp: ok_hash+=1
            else: bad.append(f"{pid}/{role}")
    rep["inventory"]={"n_members":len(names),"n_images":len(pngs),"n_pairs":len(pair_ids),
      "pair_ids_contiguous":pair_ids==[f"B-{i:02d}" for i in range(1,21)],
      "two_images_per_pair":all(sum(1 for f in pngs if f.startswith(p+"_"))==2 for p in pair_ids),
      "source_image_hashes_match":f"{ok_hash}/40","role_assignment_ok":f"{role_ok}/20",
      "problems":bad,
      "support_files":sorted(os.path.basename(n) for n in names if "/images/" not in n)}
    need={"index.html","app.js","items.js","style.css","items.json","INSTRUCTIONS.md","README.txt"}
    rep["support_complete"]=set(rep["inventory"]["support_files"])>=need
    # private-association audit on downloaded bytes
    # roster comes from the PRIVATE record; never a literal in published code
    _rp=os.path.join(ROOT,"private","taskA_closure_private.json")
    roster=[r["alias"] for r in json.load(open(_rp))["roster"]] if os.path.isfile(_rp) else []
    badf=[]
    for n in names:
        if n.endswith(".png"): continue
        txt=open(os.path.join(ex,n),"r",errors="ignore").read()
        for a in roster:
            if re.search(r"\b"+re.escape(a)+r"\b",txt): badf.append({"member":n,"kind":"reviewer alias"})
        for label,rx in [("case id",re.compile(r"\bCASE-\d{2}\b")),
                         ("seed",re.compile(r"\bseed ?7\d{3}\b",re.I)),
                         ("source cond",re.compile(r"_(m0|edited)_seed")),
                         ("A item id",re.compile(r"\bA-\d{3}\b")),
                         ("crosswalk",re.compile(r'"(reference_filename|candidate_filename|case_id|source_filename)"')),
                         ("answer key",re.compile(r'"(answer_key|ground_truth|gold)"')),
                         ("test export",re.compile(r'"_fixture"|SYNTHETIC QA FIXTURE'))]:
            if rx.search(txt): badf.append({"member":n,"kind":label})
    rep["private_audit"]={"findings":badf,"clean":not badf,
                          "alias_scan_performed":bool(roster)}
    # browser QA on the downloaded bytes
    qout=os.path.join(ROOT,"qa","qa_taskB_downloaded_report.json")
    subprocess.run([sys.executable,os.path.join(ROOT,"scripts","qa_taskB.py"),
        "--bundle",base,"--out",qout,"--label","DOWNLOADED release asset "+ZIPNAME],
        capture_output=True,text=True)
    try:
        q=json.load(open(qout))
        rep["browser_qa_on_downloaded_bytes"]={"n_checks":q["n_checks"],"n_pass":q["n_pass"],
          "n_fail":q["n_fail"],"all_pass":q["all_pass"],"engines":q["engines"],
          "browser_versions":q["browser_versions"],
          "network_requests_attempted":q["network_requests_attempted"]}
    except Exception as e:
        rep["browser_qa_on_downloaded_bytes"]={"all_pass":False,"error":str(e)}
    inv=rep["inventory"]
    ok=(rep["sha256_match"] and rep["bytes_match"] and inv["n_images"]==40
        and inv["n_pairs"]==20 and inv["pair_ids_contiguous"] and inv["two_images_per_pair"]
        and inv["source_image_hashes_match"]=="40/40" and inv["role_assignment_ok"]=="20/20"
        and not inv["problems"] and rep["support_complete"]
        and rep["private_audit"]["clean"]
        and rep["browser_qa_on_downloaded_bytes"].get("all_pass"))
    rep["verdict"]="TASKB_PUBLISHED_PACKAGE_VERIFIED" if ok else "TASKB_PUBLISHED_PACKAGE_PROBLEM"
    json.dump(rep,open(os.path.join(ROOT,"relB","taskB_published_verification.json"),"w"),indent=1)
    for k in ("tag","release_target_commitish","n_assets_on_release","sha256_match",
              "bytes_match","support_complete","verdict"): print(f"{k:34} {rep[k]}")
    print(f"{'pairs / images':34} {inv['n_pairs']} / {inv['n_images']}  contiguous={inv['pair_ids_contiguous']}")
    print(f"{'source hashes / roles':34} {inv['source_image_hashes_match']} / {inv['role_assignment_ok']}")
    print(f"{'private audit clean':34} {rep['private_audit']['clean']}")
    b=rep["browser_qa_on_downloaded_bytes"]
    print(f"{'browser QA on downloaded bytes':34} {b.get('n_pass')}/{b.get('n_checks')} pass={b.get('all_pass')}")
    sys.exit(0 if ok else 7)
if __name__=="__main__": main()
