#!/usr/bin/env python3
"""Focused synthetic browser QA for the Task-B bundle.

Every clicked choice is SYNTHETIC, drawn from a fixed rotation, and is unrelated
to image content. Nothing here is a human judgement or reviewer evidence, and no
actual participant data is used.

Asserts actual restored state (alias, declarations, order, raw answers), not
success messages. Rejection cases assert that neither memory nor localStorage
changed. Includes A/B isolation using SYNTHETIC Task-A state only.

Usage: qa_taskB.py [--bundle DIR] [--out REPORT.json] [--label TEXT]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
VALIDATOR = os.path.join(ROOT, "scripts", "validate_export.py")

CMPS = ["less", "about_the_same", "more", "cannot_fairly_compare"]
PKG, PROTO = "THR-V1-B", "TEAPOT-HR-PROTO-1.0.0"
SCHEMA, LEGACY = "THR-EXPORT-SCHEMA-1.1.0", "THR-EXPORT-SCHEMA-1.0.0"

results = []
ENGINE = {"name": "?"}


def check(name, ok, detail=""):
    results.append({"engine": ENGINE["name"], "check": name,
                    "pass": bool(ok), "detail": str(detail)[:400]})
    print(f"  [{'PASS' if ok else 'FAIL'}] [{ENGINE['name']}] {name}"
          f"{'' if ok else '  <- ' + str(detail)[:200]}")
    return bool(ok)


def new_page(ctx, attempts):
    page = ctx.new_page()

    def route(r):
        if not r.request.url.startswith("file://"):
            attempts.append(r.request.url)
            return r.abort()
        return r.continue_()
    page.route("**/*", route)
    return page


def storage_snapshot(page):
    return page.evaluate(
        "JSON.stringify(Object.keys(localStorage).sort()"
        ".map(k=>[k,localStorage.getItem(k)]))")


def session_state(page):
    return json.loads(page.evaluate(
        "(()=>{var a=document.getElementById('aliasout');"
        "a=a&&a.textContent?a.textContent.trim():null;"
        "if(a){var v=localStorage.getItem('THR::THR-V1-B::B::'+a);if(v)return v;}"
        "var k=Object.keys(localStorage).filter(x=>x.indexOf('THR::THR-V1-B::')===0);"
        "return k.length?localStorage.getItem(k.sort()[0]):'null';})()"))


def gate_start(page, alias, variant="ALIAS", exposure=("yes", "no", "no"), note=""):
    page.fill("#alias", alias)
    page.select_option("#ordervar", variant)
    for n, v in zip(("exp_a_done", "exp_cond", "exp_coach"), exposure):
        page.check(f'input[name="{n}"][value="{v}"]')
    if note:
        page.fill("#exp_other", note)
    page.click("#startbtn")
    page.wait_for_selector("#app:not([hidden])", timeout=15000)


def open_export(page):
    page.eval_on_selector(".exportbox", "e=>e.open=true")
    page.wait_for_timeout(80)


def answer_n(page, n):
    for i in range(n):
        page.check(f'input[name="cmp"][value="{CMPS[i % 4]}"]')
        if i < n - 1:
            page.click("#nextbtn")


def do_import(page, path, where="session"):
    sel = "#importfile" if where == "session" else "#gateimport"
    msg = "#importmsg" if where == "session" else "#gateimportmsg"
    if where == "session":
        open_export(page)
    else:
        page.eval_on_selector_all("details", "els=>els.forEach(e=>e.open=true)")
    page.set_input_files(sel, path)
    page.wait_for_timeout(700)
    return page.inner_text(msg)


def base(ids, alias="synthetic-b", kind="draft", n=None, variant="ALIAS",
         exposure=None, schema=SCHEMA, proto=PROTO, pkg=PKG, stage="B"):
    exposure = exposure or {
        "declared": True,
        "task_a_final_exported_before_task_b": True,
        "prior_access_to_condition_labels": False,
        "image_specific_coaching_before_task_a": False,
        "reference_identity_disclosed_in_this_stage": True,
        "other_exposure_note": ""}
    rs = []
    for i, x in enumerate(ids):
        if n is not None and i >= n:
            rs.append({"pair_id": x, "comparison": None, "reason": ""})
        else:
            rs.append({"pair_id": x, "comparison": CMPS[i % 4], "reason": ""})
    return {"_fixture": "SYNTHETIC QA FIXTURE - not a human judgement",
            "package_id": pkg, "protocol_id": proto, "schema_id": schema,
            "app_version": "1.1.0", "app_build": "taskB-b1", "stage": stage,
            "export_kind": kind, "reviewer_alias": alias,
            "order_variant": variant, "exposure": exposure,
            "exported_iso": "2026-10-18T00:00:00Z",
            "item_count_expected": len(ids), "responses": rs,
            "completion": {"answered_items": 999, "unanswered_items": 0,
                           "self_reported": True}}


def write(tmp, name, obj):
    p = os.path.join(tmp, name)
    with open(p, "w") as fh:
        fh.write(obj) if isinstance(obj, str) else json.dump(obj, fh, indent=1)
    return p


def run_engine(engine_name, launch, url, ids, tmp):
    ENGINE["name"] = engine_name
    from playwright.sync_api import sync_playwright
    attempts = []
    with sync_playwright() as p:
        br = launch(p)

        # ---------- A. display, labels, zoom, navigation ----------
        ctx = br.new_context(viewport={"width": 1400, "height": 1000},
                             accept_downloads=True)
        page = new_page(ctx, attempts)
        page.goto(url, wait_until="load")
        check("bundle opens offline from file://", page.is_visible("#gate h1"))
        gate = page.inner_text("#gate")
        check("instructions state the stage is only partially masked",
              "partially" in gate and "tells you which image is which" in gate)
        check("instructions carry the four neutral caveats",
              all(x in gate for x in ["may already contain defects",
                                      "not automatically deterioration",
                                      "missing teapot",
                                      "matched noise does not guarantee"]))
        check("instructions tell reviewers to leave Task A unchanged",
              "leave your Task A answers exactly as you submitted" in gate)

        gate_start(page, "qa-b-main", "ALIAS")
        seen = set()
        for i in range(len(ids)):
            pid = page.inner_text("#itemid")
            seen.add(pid)
            ref = page.get_attribute("#imgref", "src")
            cand = page.get_attribute("#imgcand", "src")
            if i == 0:
                check("reference labelled as the original model",
                      "REFERENCE" in page.inner_text(".pairwrap"))
                check("candidate labelled as the modified model",
                      "CANDIDATE" in page.inner_text(".pairwrap"))
                check("both images render at native 512",
                      page.eval_on_selector("#imgref", "e=>e.naturalWidth") == 512
                      and page.eval_on_selector("#imgcand", "e=>e.naturalWidth") == 512)
                check("no answer is preselected",
                      page.eval_on_selector_all('input[name="cmp"]:checked',
                                                "e=>e.length") == 0)
            if not (ref.endswith(pid + "_reference.png")
                    and cand.endswith(pid + "_candidate.png")):
                check(f"correct pairing for {pid}", False, f"{ref} / {cand}")
                break
            if i < len(ids) - 1:
                page.click("#nextbtn")
        check("all 20 pair IDs displayed, each correctly paired",
              len(seen) == 20 and seen == set(ids), sorted(seen))

        page.click("#zoomrefin"); page.click("#zoomrefin")
        rw = page.eval_on_selector("#imgref", "e=>e.clientWidth")
        cw = page.eval_on_selector("#imgcand", "e=>e.clientWidth")
        check("reference zooms independently of candidate", rw > 512 and cw == 512,
              f"ref={rw} cand={cw}")
        page.click("#zoomcandin")
        check("candidate zooms independently",
              page.eval_on_selector("#imgcand", "e=>e.clientWidth") > 512)
        page.click("#zoomrefreset"); page.click("#zoomcandreset")
        check("zoom reset returns both to native 512",
              page.eval_on_selector("#imgref", "e=>e.clientWidth") == 512
              and page.eval_on_selector("#imgcand", "e=>e.clientWidth") == 512)

        open_export(page)
        check("Final disabled before all pairs answered",
              page.eval_on_selector("#finalbtn", "e=>e.disabled"))
        ctx.close()

        # ---------- B. cannot_fairly_compare counts complete ----------
        ctx = br.new_context(viewport={"width": 1400, "height": 1000},
                             accept_downloads=True)
        page = new_page(ctx, attempts)
        page.goto(url, wait_until="load")
        gate_start(page, "qa-b-cfc", "FIXED")
        for i in range(len(ids)):
            page.check('input[name="cmp"][value="cannot_fairly_compare"]')
            if i < len(ids) - 1:
                page.click("#nextbtn")
        open_export(page)
        check("all cannot_fairly_compare counts COMPLETE and enables Final",
              not page.eval_on_selector("#finalbtn", "e=>e.disabled"))
        ctx.close()

        # ---------- C. draft export -> fresh import -> reload/resume ----------
        ctx = br.new_context(viewport={"width": 1400, "height": 1000},
                             accept_downloads=True)
        page = new_page(ctx, attempts)
        page.goto(url, wait_until="load")
        gate_start(page, "qa-b-draft", "ALIAS", ("yes", "yes", "no"), "synthetic")
        answer_n(page, 6)
        page.click("#savebtn")
        src = session_state(page)
        open_export(page)
        with page.expect_download() as dl:
            page.click("#draftbtn")
        dpath = os.path.join(tmp, f"b_draft_{engine_name}.json")
        dl.value.save_as(dpath)
        ctx.close()
        draft = json.load(open(dpath))
        check("draft export is partial and marked draft",
              draft["export_kind"] == "draft"
              and sum(1 for r in draft["responses"] if r["comparison"]) == 6)
        check("export records reference-identity disclosure",
              draft["exposure"]["reference_identity_disclosed_in_this_stage"] is True)
        check("export carries stage/package/schema provenance",
              draft["stage"] == "B" and draft["package_id"] == PKG
              and draft["schema_id"] == SCHEMA and draft["protocol_id"] == PROTO)
        check("export contains no case id, seed, prompt or Task-A data",
              not any(k in json.dumps(draft) for k in
                      ["CASE-", "seed7", "item_id", "taskA", "_m0_", "_edited_"]))

        ctx = br.new_context(viewport={"width": 1400, "height": 1000},
                             accept_downloads=True)
        page = new_page(ctx, attempts)
        page.goto(url, wait_until="load")
        check("fresh session starts with empty storage", storage_snapshot(page) == "[]")
        do_import(page, dpath, where="gate")
        got = session_state(page)
        check("draft restored into a fresh browser without starting a session",
              got is not None and page.is_visible("#itemid"))
        if got:
            check("restored alias equals the file's", got["reviewer_alias"] == "qa-b-draft")
            check("restored declarations equal the file's",
                  got["exposure"] == draft["exposure"])
            check("restored order variant and order equal the trusted reconstruction",
                  got["order_variant"] == "ALIAS" and got["order"] == src["order"])
            check("restored raw answers equal the exported answers",
                  {k: v for k, v in got["responses"].items() if v.get("comparison")}
                  == {k: v for k, v in src["responses"].items() if v.get("comparison")})
        page.goto(url, wait_until="load")
        page.fill("#alias", "qa-b-draft"); page.wait_for_timeout(250)
        check("resume offered after reload", page.eval_on_selector("#resumebtn", "e=>!e.hidden"))
        page.click("#resumebtn"); page.wait_for_selector("#app:not([hidden])")
        got2 = session_state(page)
        check("resumed state still equals the imported state",
              got2["responses"] == got["responses"]
              and got2["exposure"] == got["exposure"])
        ctx.close()

        # ---------- D. complete Final -> fresh re-import -> validator ----------
        ctx = br.new_context(viewport={"width": 1400, "height": 1000},
                             accept_downloads=True)
        page = new_page(ctx, attempts)
        page.goto(url, wait_until="load")
        gate_start(page, "qa-b-final", "ALIAS")
        answer_n(page, len(ids))
        page.click("#savebtn")
        fin_state = session_state(page)
        open_export(page)
        check("Final enabled once all 20 pairs answered",
              not page.eval_on_selector("#finalbtn", "e=>e.disabled"))
        with page.expect_download() as dl:
            page.click("#finalbtn")
        fpath = os.path.join(tmp, f"b_final_{engine_name}.json")
        dl.value.save_as(fpath)
        ctx.close()
        fin = json.load(open(fpath))
        check("final has exactly 20 answered pairs",
              len(fin["responses"]) == 20
              and all(r["comparison"] in CMPS for r in fin["responses"]))

        ctx = br.new_context(viewport={"width": 1400, "height": 1000},
                             accept_downloads=True)
        page = new_page(ctx, attempts)
        page.goto(url, wait_until="load")
        do_import(page, fpath, where="gate")
        gf = session_state(page)
        check("FINAL re-imported into a fresh session", gf is not None)
        if gf:
            check("re-imported final: answers equal", gf["responses"] == fin_state["responses"])
            check("re-imported final: metadata equal",
                  gf["reviewer_alias"] == fin["reviewer_alias"]
                  and gf["exposure"] == fin["exposure"]
                  and gf["order_variant"] == fin["order_variant"])
        page.goto(url, wait_until="load")
        page.fill("#alias", "qa-b-final"); page.wait_for_timeout(250)
        page.click("#resumebtn"); page.wait_for_selector("#app:not([hidden])")
        check("re-imported final survives reload/resume",
              session_state(page)["responses"] == gf["responses"])
        ctx.close()

        v = subprocess.run([sys.executable, VALIDATOR, fpath, "--expect-stage", "B"],
                           capture_output=True, text=True)
        try:
            vr = json.loads(v.stdout)["results"][0]
        except Exception:
            vr = {"valid": False, "errors": ["validator output unparsable"]}
        check("companion validator accepts the browser B final", vr["valid"],
              vr.get("errors"))

        # ---------- E. rejections must not mutate state ----------
        bad = {}
        bad["wrong schema (legacy)"] = base(ids, schema=LEGACY)
        bad["wrong protocol"] = base(ids, proto="OTHER-PROTOCOL")
        bad["wrong package"] = base(ids, pkg="THR-V1-A")
        bad["wrong stage"] = base(ids, stage="A")
        d = base(ids); d["responses"].append(dict(d["responses"][0]))
        bad["duplicate pair id"] = d
        d = base(ids); d["responses"].append(
            {"pair_id": "B-99", "comparison": "less", "reason": ""})
        bad["unknown pair id"] = d
        d = base(ids); d["responses"][2]["comparison"] = "slightly_worse"
        bad["invalid enum"] = d
        d = base(ids); d["responses"][1]["comparison"] = []
        bad["wrong type comparison"] = d
        d = base(ids); d["reviewer_alias"] = 7
        bad["wrong type alias"] = d
        d = base(ids); del d["exposure"]["task_a_final_exported_before_task_b"]
        bad["missing declaration"] = d
        d = base(ids); d["exposure"]["prior_access_to_condition_labels"] = "no"
        bad["non-boolean declaration"] = d
        d = base(ids); d["exposure"]["reference_identity_disclosed_in_this_stage"] = False
        bad["disclosure flag not true"] = d
        d = base(ids, kind="final"); d["responses"] = []
        bad["empty final"] = d
        bad["partial final"] = base(ids, kind="final", n=19)
        d = base(ids); d["responses"] = {"not": "a list"}
        bad["responses not a list"] = d
        bad["malformed JSON"] = "{ broken ,,,"

        ctx = br.new_context(viewport={"width": 1400, "height": 1000},
                             accept_downloads=True)
        page = new_page(ctx, attempts)
        page.goto(url, wait_until="load")
        gate_start(page, "qa-b-reject", "ALIAS")
        answer_n(page, 3)
        page.click("#savebtn")
        before_store = storage_snapshot(page)
        before_state = session_state(page)
        for label, obj in bad.items():
            pth = write(tmp, f"bad_{abs(hash(label))}.json", obj)
            msg = do_import(page, pth, where="session")
            ok = ("Rejected" in msg
                  and storage_snapshot(page) == before_store
                  and session_state(page) == before_state)
            check(f"rejected without mutating state: {label}", ok, msg[:90])
        check("viewer still usable after all rejections", page.is_visible("#itemid"))
        ctx.close()

        # ---------- F. A/B isolation, using SYNTHETIC Task-A state only ----------
        ctx = br.new_context(viewport={"width": 1400, "height": 1000},
                             accept_downloads=True)
        page = new_page(ctx, attempts)
        page.goto(url, wait_until="load")
        synth_a = {"package_id": "THR-V1-A", "stage": "A",
                   "reviewer_alias": "qa-shared", "order_variant": "V1",
                   "responses": {"A-001": {"target": "present",
                                           "rendering": "coherent", "reason": ""}},
                   "_synthetic": "QA fixture, not a real Task A session"}
        page.evaluate("s=>localStorage.setItem('THR::THR-V1-A::A::qa-shared',s)",
                      json.dumps(synth_a))
        a_before = page.evaluate(
            "localStorage.getItem('THR::THR-V1-A::A::qa-shared')")
        gate_start(page, "qa-shared", "ALIAS")
        answer_n(page, 4)
        page.click("#savebtn")
        a_after = page.evaluate(
            "localStorage.getItem('THR::THR-V1-A::A::qa-shared')")
        check("Task B never alters an existing Task A session (same alias)",
              a_before == a_after)
        keys = page.evaluate("Object.keys(localStorage).sort()")
        check("A and B sessions coexist under separate namespaced keys",
              keys == ["THR::THR-V1-A::A::qa-shared", "THR::THR-V1-B::B::qa-shared"],
              keys)
        a_export = {"package_id": "THR-V1-A", "protocol_id": PROTO,
                    "schema_id": SCHEMA, "stage": "A", "export_kind": "final",
                    "reviewer_alias": "qa-shared", "order_variant": "V1",
                    "exposure": {"declared": True}, "responses": [],
                    "_synthetic": "QA fixture"}
        ap = write(tmp, "synthetic_a_export.json", a_export)
        before_store = storage_snapshot(page)
        msg = do_import(page, ap, where="session")
        check("Task B rejects a Task A export with a clear message",
              "Rejected" in msg and "Task A" in msg, msg[:120])
        check("rejecting the Task A export mutated nothing",
              storage_snapshot(page) == before_store)
        ctx.close()

        # ---------- G. mobile ----------
        mctx = br.new_context(viewport={"width": 390, "height": 844},
                              device_scale_factor=2)
        mp = new_page(mctx, attempts)
        mp.goto(url, wait_until="load")
        gate_start(mp, "qa-b-mobile", "ALIAS")
        ow = mp.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
        check("no horizontal overflow at 390px", ow[0] <= ow[1] + 1, ow)
        check("comparison controls reachable on mobile",
              mp.is_visible('input[name="cmp"][value="less"]'))
        check("both images present on mobile",
              mp.is_visible("#imgref") and mp.is_visible("#imgcand"))
        mctx.close()
        br.close()

    check("no network requests attempted (all non-file:// aborted)",
          not attempts, attempts[:3])
    return attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(ROOT, "private", "taskB_package"))
    ap.add_argument("--out", default=os.path.join(ROOT, "qa", "qa_taskB_report.json"))
    ap.add_argument("--label", default="local private build")
    a = ap.parse_args()

    ids = json.load(open(os.path.join(a.bundle, "items.json")))["item_ids"]
    url = "file://" + os.path.join(os.path.abspath(a.bundle), "index.html")
    tmp = tempfile.mkdtemp(prefix="qab_", dir=os.path.join(ROOT, "qa"))

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        versions = {}
        for n, L in (("chromium", lambda q: q.chromium.launch()),
                     ("firefox", lambda q: q.firefox.launch())):
            b = L(p); versions[n] = b.version; b.close()

    engines = [
        ("chromium", lambda q: q.chromium.launch()),
        ("firefox", lambda q: q.firefox.launch()),
        ("firefox-strict-file-origin",
         lambda q: q.firefox.launch(firefox_user_prefs={
             "privacy.file_unique_origin": True,
             "security.fileuri.strict_origin_policy": True})),
    ]
    allatt = []
    for name, launch in engines:
        allatt += run_engine(name, launch, url, ids, tmp) or []

    n_pass = sum(1 for r in results if r["pass"])
    rep = {
        "suite": "focused synthetic browser QA (Task B)",
        "tested_artifact": os.path.abspath(a.bundle),
        "artifact_label": a.label,
        "stage": "B", "package_id": PKG, "schema_id": SCHEMA,
        "app_version": "1.1.0", "app_build": "taskB-b1",
        "browser_versions": versions, "engines": [e[0] for e in engines],
        "fixtures_are_synthetic": True,
        "contains_human_judgements": False,
        "contains_reviewer_evidence": False,
        "real_participant_data_used": 0,
        "taskA_state_used": "synthetic fixture only; no real Task A session or response was read",
        "n_checks": len(results), "n_pass": n_pass,
        "n_fail": len(results) - n_pass, "all_pass": n_pass == len(results),
        "network_requests_attempted": allatt,
        "checks": results,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{n_pass}/{len(results)} checks pass  ->  {a.out}")
    sys.exit(0 if rep["all_pass"] else 4)


if __name__ == "__main__":
    main()
