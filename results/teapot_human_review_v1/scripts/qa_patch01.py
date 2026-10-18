#!/usr/bin/env python3
"""Focused synthetic browser QA for the corrected Task-A bundle (patch01).

Every clicked choice is SYNTHETIC, drawn from a fixed rotation, and is unrelated
to image content. Nothing produced here is a human judgement or reviewer
evidence.

Asserts actual state equality -- restored alias, declarations, order and raw
answers -- not merely that a success message appeared. Rejection cases assert
that neither in-memory state nor localStorage changed.

Usage: qa_patch01.py [--bundle DIR] [--out REPORT.json] [--label TEXT]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
VALIDATOR = os.path.join(ROOT, "scripts", "validate_export.py")
ITEMS_PATH = os.path.join(ROOT, "protocol", "taskA_items.json")

TARGETS = ["present", "absent", "uncertain"]
RENDERS = ["coherent", "localized_deformation",
           "whole_frame_breakdown", "unable_to_judge"]
PKG, PROTO = "THR-V1-A", "TEAPOT-HR-PROTO-1.0.0"
SCHEMA, LEGACY = "THR-EXPORT-SCHEMA-1.1.0", "THR-EXPORT-SCHEMA-1.0.0"

results = []
ENGINE = {"name": "?"}


def check(name, ok, detail=""):
    results.append({"engine": ENGINE["name"], "check": name,
                    "pass": bool(ok), "detail": str(detail)[:400]})
    print(f"  [{'PASS' if ok else 'FAIL'}] [{ENGINE['name']}] {name}"
          f"{'' if ok else '  <- ' + str(detail)[:200]}")
    return bool(ok)


# ---------------- page helpers ----------------
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


def gate_start(page, alias, variant, exposure=("no", "no", "no"), note=""):
    page.fill("#alias", alias)
    page.select_option("#ordervar", variant)
    for n, v in zip(("exp_pair", "exp_cond", "exp_coach"), exposure):
        page.check(f'input[name="{n}"][value="{v}"]')
    if note:
        page.fill("#exp_other", note)
    page.click("#startbtn")
    page.wait_for_selector("#app:not([hidden])", timeout=15000)


def open_export(page):
    page.eval_on_selector(".exportbox", "e=>e.open=true")
    page.wait_for_timeout(80)


def answer_n(page, n, offset=0):
    """Answer the first n items with synthetic rotating choices."""
    for i in range(n):
        page.check(f'input[name="target"][value="{TARGETS[(i+offset) % 3]}"]')
        page.check(f'input[name="rendering"][value="{RENDERS[(i+offset) % 4]}"]')
        if i < n - 1:
            page.click("#nextbtn")


def session_state(page):
    """Read the ACTIVE session straight out of localStorage.

    Keyed by the alias the viewer is currently showing -- several aliases can
    legitimately coexist in storage, so picking an arbitrary key would read the
    wrong session."""
    raw = page.evaluate(
        "(()=>{var a=document.getElementById('aliasout');"
        "a=a&&a.textContent?a.textContent.trim():null;"
        "if(a){var v=localStorage.getItem('THR::THR-V1-A::A::'+a);if(v)return v;}"
        "var k=Object.keys(localStorage).filter(x=>x.indexOf('THR::')===0);"
        "return k.length?localStorage.getItem(k.sort()[0]):null;})()")
    return json.loads(raw) if raw else None


def do_import(page, path, where="session"):
    sel = "#importfile" if where == "session" else "#gateimport"
    msg = "#importmsg" if where == "session" else "#gateimportmsg"
    if where == "session":
        open_export(page)
    else:
        page.eval_on_selector_all(
            "details", "els=>els.forEach(e=>e.open=true)")
    page.set_input_files(sel, path)
    page.wait_for_timeout(700)
    return page.inner_text(msg)


# ---------------- synthetic export fixtures ----------------
def base_export(ids, alias="synthetic-a", kind="draft", n=None, variant="V1",
                exposure=None, schema=SCHEMA, proto=PROTO, pkg=PKG, stage="A"):
    exposure = exposure or {
        "declared": True,
        "prior_access_to_source_pairings": False,
        "prior_access_to_condition_labels": False,
        "image_specific_coaching_before_task_a": False,
        "other_exposure_note": ""}
    rs = []
    for i, x in enumerate(ids):
        if n is not None and i >= n:
            rs.append({"item_id": x, "target": None, "rendering": None, "reason": ""})
        else:
            rs.append({"item_id": x, "target": TARGETS[i % 3],
                       "rendering": RENDERS[i % 4], "reason": ""})
    return {"_fixture": "SYNTHETIC QA FIXTURE - not a human judgement",
            "package_id": pkg, "protocol_id": proto, "schema_id": schema,
            "app_version": "1.1.0", "stage": stage, "export_kind": kind,
            "reviewer_alias": alias, "order_variant": variant,
            "exposure": exposure, "exported_iso": "2026-10-18T00:00:00Z",
            "item_count_expected": len(ids), "responses": rs,
            "completion": {"answered_items": 999, "unanswered_items": 0,
                           "self_reported": True}}


def write(tmp, name, obj):
    p = os.path.join(tmp, name)
    with open(p, "w") as fh:
        if isinstance(obj, str):
            fh.write(obj)
        else:
            json.dump(obj, fh, indent=1)
    return p


# ---------------- the suite ----------------
def run_engine(engine_name, launch, url, ids, tmp):
    ENGINE["name"] = engine_name
    from playwright.sync_api import sync_playwright
    attempts = []
    with sync_playwright() as p:
        br = launch(p)

        # ---------- A. incomplete draft: export -> fresh session -> import ----------
        ctx = br.new_context(viewport={"width": 1280, "height": 1000},
                             accept_downloads=True)
        page = new_page(ctx, attempts)
        page.goto(url, wait_until="load")
        check("bundle opens offline from file://", page.is_visible("#gate h1"))
        check("exposure wording covers same-day exposure",
              "Before starting this Task A review" in page.inner_text("#gate")
              and "Before today" not in page.inner_text("#gate"))
        check("eligibility explanation shown",
              "ineligible for the primary" in page.inner_text("#gate"))

        gate_start(page, "qa-draft", "V2", ("yes", "no", "no"), "synthetic note")
        answer_n(page, 5)
        page.click("#savebtn")
        src_state = session_state(page)
        open_export(page)
        with page.expect_download() as dl:
            page.click("#draftbtn")
        draft_path = os.path.join(tmp, f"draft_{engine_name}.json")
        dl.value.save_as(draft_path)
        ctx.close()

        draft = json.load(open(draft_path))
        check("draft export is partial and marked draft",
              draft["export_kind"] == "draft" and
              sum(1 for r in draft["responses"]
                  if r["target"] and r["rendering"]) == 5)

        # fresh browser context = fresh storage
        ctx = br.new_context(viewport={"width": 1280, "height": 1000},
                             accept_downloads=True)
        page = new_page(ctx, attempts)
        page.goto(url, wait_until="load")
        check("fresh session starts with empty storage",
              storage_snapshot(page) == "[]")
        do_import(page, draft_path, where="gate")
        got = session_state(page)
        check("draft restored into a fresh browser without starting a session",
              got is not None and page.is_visible("#itemid"))
        if got:
            check("restored alias equals the file's alias",
                  got["reviewer_alias"] == "qa-draft", got.get("reviewer_alias"))
            check("restored declarations equal the file's declarations",
                  got["exposure"] == draft["exposure"], got.get("exposure"))
            check("restored order variant equals the file's",
                  got["order_variant"] == "V2", got.get("order_variant"))
            check("restored order equals the trusted reconstruction",
                  got["order"] == src_state["order"])
            check("restored raw answers equal the exported answers",
                  {k: v for k, v in got["responses"].items() if v.get("target")}
                  == {k: v for k, v in src_state["responses"].items()
                      if v.get("target")})
        # reload + resume preserves it
        page.goto(url, wait_until="load")
        page.fill("#alias", "qa-draft")
        page.wait_for_timeout(250)
        check("resume offered after reload", page.eval_on_selector("#resumebtn", "e=>!e.hidden"))
        page.click("#resumebtn")
        page.wait_for_selector("#app:not([hidden])")
        got2 = session_state(page)
        check("resumed state still equals the imported state",
              got2["responses"] == got["responses"] and
              got2["exposure"] == got["exposure"] and
              got2["reviewer_alias"] == got["reviewer_alias"])
        ctx.close()

        # ---------- B. complete Final: export -> fresh import -> validator ----------
        ctx = br.new_context(viewport={"width": 1280, "height": 1000},
                             accept_downloads=True)
        page = new_page(ctx, attempts)
        page.goto(url, wait_until="load")
        gate_start(page, "qa-final", "V1")
        open_export(page)
        check("Final disabled while answers are missing",
              page.eval_on_selector("#finalbtn", "e=>e.disabled"))
        answer_n(page, len(ids))
        page.click("#savebtn")
        open_export(page)
        check("Final enabled once all 40 are answered",
              not page.eval_on_selector("#finalbtn", "e=>e.disabled"))
        fin_state = session_state(page)
        with page.expect_download() as dl:
            page.click("#finalbtn")
        final_path = os.path.join(tmp, f"final_{engine_name}.json")
        dl.value.save_as(final_path)
        ctx.close()

        fin = json.load(open(final_path))
        check("final export has 40 complete answers",
              len(fin["responses"]) == 40 and
              all(r["target"] and r["rendering"] for r in fin["responses"]))

        ctx = br.new_context(viewport={"width": 1280, "height": 1000},
                             accept_downloads=True)
        page = new_page(ctx, attempts)
        page.goto(url, wait_until="load")
        do_import(page, final_path, where="gate")
        gf = session_state(page)
        check("FINAL re-imported into a fresh session (not just validated)",
              gf is not None)
        if gf:
            check("re-imported final: answers equal",
                  gf["responses"] == fin_state["responses"])
            check("re-imported final: metadata equal",
                  gf["reviewer_alias"] == fin["reviewer_alias"] and
                  gf["exposure"] == fin["exposure"] and
                  gf["order_variant"] == fin["order_variant"])
        page.goto(url, wait_until="load")
        page.fill("#alias", "qa-final"); page.wait_for_timeout(250)
        page.click("#resumebtn"); page.wait_for_selector("#app:not([hidden])")
        gf2 = session_state(page)
        check("re-imported final survives reload/resume",
              gf2["responses"] == gf["responses"])
        open_export(page)
        check("Final still enabled after re-import",
              not page.eval_on_selector("#finalbtn", "e=>e.disabled"))
        ctx.close()

        v = subprocess.run([sys.executable, VALIDATOR, final_path,
                            "--expect-stage", "A", "--items", ITEMS_PATH],
                           capture_output=True, text=True)
        try:
            vr = json.loads(v.stdout)["results"][0]
        except Exception:
            vr = {"valid": False, "errors": ["validator output unparsable"]}
        check("companion validator accepts the browser final",
              vr["valid"], vr.get("errors"))

        # ---------- C. uncertainty counts complete ----------
        ctx = br.new_context(viewport={"width": 1280, "height": 1000},
                             accept_downloads=True)
        page = new_page(ctx, attempts)
        page.goto(url, wait_until="load")
        gate_start(page, "qa-unc", "V3")
        for i in range(len(ids)):
            page.check('input[name="target"][value="uncertain"]')
            page.check('input[name="rendering"][value="unable_to_judge"]')
            if i < len(ids) - 1:
                page.click("#nextbtn")
        open_export(page)
        check("all-uncertain counts as complete and enables Final",
              not page.eval_on_selector("#finalbtn", "e=>e.disabled"))
        ctx.close()

        # ---------- D. rejections must not mutate state ----------
        bad = {
            "wrong protocol (same package)":
                base_export(ids, proto="SOME-OTHER-PROTOCOL"),
            "unsupported legacy schema":
                base_export(ids, schema=LEGACY),
            "wrong package":
                base_export(ids, pkg="THR-V1-B"),
            "wrong stage":
                base_export(ids, stage="B"),
            "duplicate id":
                None, "unknown id": None, "invalid enum": None,
            "wrong types (alias)":
                None, "missing declaration": None,
            "non-boolean declaration": None,
            "empty final": None, "partial final": None,
            "responses not a list": None,
            "malformed JSON": None,
        }
        d = base_export(ids); d["responses"].append(dict(d["responses"][0]))
        bad["duplicate id"] = d
        d = base_export(ids)
        d["responses"].append({"item_id": "A-999", "target": "present",
                               "rendering": "coherent", "reason": ""})
        bad["unknown id"] = d
        d = base_export(ids); d["responses"][2]["target"] = "definitely"
        bad["invalid enum"] = d
        d = base_export(ids); d["reviewer_alias"] = 12345
        bad["wrong types (alias)"] = d
        d = base_export(ids); del d["exposure"]["prior_access_to_source_pairings"]
        bad["missing declaration"] = d
        d = base_export(ids); d["exposure"]["prior_access_to_condition_labels"] = "no"
        bad["non-boolean declaration"] = d
        d = base_export(ids, kind="final"); d["responses"] = []
        bad["empty final"] = d
        bad["partial final"] = base_export(ids, kind="final", n=39)
        d = base_export(ids); d["responses"] = {"not": "a list"}
        bad["responses not a list"] = d
        bad["malformed JSON"] = "{ this is not json ,,,"

        ctx = br.new_context(viewport={"width": 1280, "height": 1000},
                             accept_downloads=True)
        page = new_page(ctx, attempts)
        page.goto(url, wait_until="load")
        gate_start(page, "qa-reject", "V1")
        answer_n(page, 3)
        page.click("#savebtn")
        before_store = storage_snapshot(page)
        before_state = session_state(page)
        for label, obj in bad.items():
            pth = write(tmp, f"bad_{abs(hash(label))}.json", obj)
            msg = do_import(page, pth, where="session")
            after_store = storage_snapshot(page)
            after_state = session_state(page)
            ok = ("Rejected" in msg and after_store == before_store
                  and after_state == before_state)
            check(f"rejected without mutating state: {label}", ok,
                  f"msg={msg[:90]!r} storage_changed={after_store != before_store}")
        check("viewer still usable after all rejections", page.is_visible("#itemid"))
        ctx.close()

        # ---------- E. conflicting provenance follows the documented policy ----------
        ctx = br.new_context(viewport={"width": 1280, "height": 1000},
                             accept_downloads=True)
        page = new_page(ctx, attempts)
        page.goto(url, wait_until="load")
        gate_start(page, "alias-one", "V1", ("no", "no", "no"))
        answer_n(page, 2)
        page.click("#savebtn")
        a1_state = session_state(page)
        # import a file belonging to a DIFFERENT alias with different exposure/order
        other = base_export(ids, alias="alias-two", variant="V3", n=4, exposure={
            "declared": True,
            "prior_access_to_source_pairings": True,
            "prior_access_to_condition_labels": False,
            "image_specific_coaching_before_task_a": False,
            "other_exposure_note": "synthetic"})
        op = write(tmp, "other_alias.json", other)
        do_import(page, op, where="session")
        now = session_state(page)
        check("import of another alias restores THAT alias, never merges",
              now["reviewer_alias"] == "alias-two" and
              now["exposure"]["prior_access_to_source_pairings"] is True and
              now["order_variant"] == "V3")
        check("the original alias session is left intact",
              json.loads(page.evaluate(
                  "localStorage.getItem('THR::THR-V1-A::A::alias-one')"
              ))["responses"] == a1_state["responses"])
        check("imported answers are not reassigned to the active alias",
              sum(1 for v in now["responses"].values() if v.get("target")) == 4)
        ctx.close()

        # ---------- F. isolation, navigation, zoom, mobile ----------
        ctx = br.new_context(viewport={"width": 1280, "height": 1000})
        page = new_page(ctx, attempts)
        page.goto(url, wait_until="load")
        gate_start(page, "iso-a", "V1")
        first = page.inner_text("#itemid")
        page.check('input[name="target"][value="present"]')
        page.check('input[name="rendering"][value="coherent"]')
        page.click("#savebtn")
        page.click("#nextbtn")
        check("navigation advances", page.inner_text("#itemid") != first)
        page.click("#prevbtn")
        check("navigation returns and keeps the answer",
              page.inner_text("#itemid") == first and
              page.is_checked('input[name="target"][value="present"]'))
        page.click("#zoomin"); page.click("#zoomin")
        check("zoom enlarges beyond native",
              page.eval_on_selector("#theimage", "e=>e.clientWidth") > 512)
        page.click("#zoomreset")
        check("zoom reset returns to native 512",
              page.eval_on_selector("#theimage", "e=>e.clientWidth") == 512 and
              page.eval_on_selector("#theimage", "e=>e.naturalWidth") == 512)
        page.goto(url, wait_until="load")
        gate_start(page, "iso-b", "V2")
        check("different alias starts clean (isolation)",
              page.eval_on_selector_all('input[name="target"]:checked',
                                        "e=>e.length") == 0)
        keys = page.evaluate("Object.keys(localStorage)")
        check("storage namespaced by package::stage::alias",
              all(k.startswith("THR::THR-V1-A::A::") for k in keys) and
              len(keys) == 2, keys)
        ctx.close()

        mctx = br.new_context(viewport={"width": 390, "height": 844},
                              device_scale_factor=2)
        mp = new_page(mctx, attempts)
        mp.goto(url, wait_until="load")
        gate_start(mp, "qa-mobile", "V1")
        ow = mp.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
        check("no horizontal overflow at 390px", ow[0] <= ow[1] + 1, ow)
        check("controls reachable on mobile",
              mp.is_visible('input[name="target"][value="present"]'))
        mctx.close()
        br.close()

    check("no network requests attempted (all non-file:// aborted)",
          not attempts, attempts[:3])
    return attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(ROOT, "taskA"))
    ap.add_argument("--out", default=os.path.join(ROOT, "qa",
                                                  "qa_patch01_report.json"))
    ap.add_argument("--label", default="local build")
    a = ap.parse_args()

    ids = json.load(open(ITEMS_PATH))["item_ids"]
    url = "file://" + os.path.join(os.path.abspath(a.bundle), "index.html")
    tmp = tempfile.mkdtemp(prefix="thrqa_", dir=os.path.join(ROOT, "qa"))

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
    all_attempts = []
    for name, launch in engines:
        all_attempts += run_engine(name, launch, url, ids, tmp) or []

    n_pass = sum(1 for r in results if r["pass"])
    rep = {
        "suite": "focused synthetic browser QA (patch01)",
        "tested_artifact": os.path.abspath(a.bundle),
        "artifact_label": a.label,
        "app_version": "1.1.0",
        "schema_id": SCHEMA,
        "browser_versions": versions,
        "engines": [e[0] for e in engines],
        "fixtures_are_synthetic": True,
        "contains_human_judgements": False,
        "contains_reviewer_evidence": False,
        "scope_note": ("Task-A only. These checks do NOT certify the private "
                       "Task-B bundle, which was not browser-tested."),
        "n_checks": len(results), "n_pass": n_pass,
        "n_fail": len(results) - n_pass,
        "all_pass": n_pass == len(results),
        "network_requests_attempted": all_attempts,
        "checks": results,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(f"\n{n_pass}/{len(results)} checks pass  ->  {a.out}")
    sys.exit(0 if rep["all_pass"] else 4)


if __name__ == "__main__":
    main()
