#!/usr/bin/env python3
"""Real-browser QA for the Task-A reviewer bundle (Playwright + Firefox).

Mechanical checks only. Nothing here constitutes a human judgement about any
image: every choice this script clicks comes from a fixed rotation and is
discarded. No output of this suite may be used as reviewer evidence.

Tests, as required:
  offline opening (file://, with the network hard-blocked)
  save / resume
  alias + package + stage storage isolation
  draft export, draft import round-trip
  complete Final round-trip (and that the validator accepts it)
  missing answers (Final must stay unavailable)
  malformed import (must be rejected, not crash)
  zoom
  mobile readability (no horizontal overflow at 390px)
  navigation
"""
import json
import os
import re
import sys
import tempfile

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
A = os.path.join(ROOT, "taskA")
SHOTS = os.path.join(ROOT, "qa", "screenshots")
URL = "file://" + os.path.join(A, "index.html")

TARGETS = ["present", "absent", "uncertain"]
RENDERS = ["coherent", "localized_deformation", "whole_frame_breakdown", "unable_to_judge"]

results = []
network_attempts = []


def check(name, ok, detail=""):
    results.append({"check": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail if not ok else ''}")
    return ok


def setup_page(ctx):
    page = ctx.new_page()
    # hard-block every non-file request: proves the viewer needs no network
    def route(r):
        u = r.request.url
        if not u.startswith("file://"):
            network_attempts.append(u)
            return r.abort()
        return r.continue_()
    page.route("**/*", route)
    page.on("pageerror", lambda e: results.append(
        {"check": "no page JS errors", "pass": False, "detail": str(e)[:300]}))
    return page


def fill_gate(page, alias, variant="V1"):
    page.fill("#alias", alias)
    page.select_option("#ordervar", variant)
    page.check('input[name="exp_pair"][value="no"]')
    page.check('input[name="exp_cond"][value="no"]')
    page.check('input[name="exp_coach"][value="no"]')
    page.click("#startbtn")
    page.wait_for_selector("#app:not([hidden])", timeout=15000)


def answer_current(page, i):
    page.check(f'input[name="target"][value="{TARGETS[i % 3]}"]')
    page.check(f'input[name="rendering"][value="{RENDERS[i % 4]}"]')


def open_export(page):
    """The export/import controls sit inside a collapsed <details>; open it."""
    page.eval_on_selector(".exportbox", "e=>e.open=true")
    page.wait_for_timeout(100)


def main():
    from playwright.sync_api import sync_playwright
    os.makedirs(SHOTS, exist_ok=True)
    n_items = len(json.load(open(os.path.join(A, "items.json")))["item_ids"])

    with sync_playwright() as p:
        br = p.firefox.launch()

        # ---------------- offline opening + first render ----------------
        ctx = br.new_context(viewport={"width": 1280, "height": 1000},
                             accept_downloads=True)
        page = setup_page(ctx)
        page.goto(URL, wait_until="load")
        check("offline opening from file:// renders the gate",
              page.is_visible("#gate h1"))
        check("instructions show the frozen target definition",
              "brew or serve tea" in page.inner_text("#targetdef"))
        check("gate blocks start without required declarations",
              (page.fill("#alias", "qa-gate"), page.click("#startbtn"),
               page.is_visible("#gateerr"))[2])
        page.screenshot(path=os.path.join(SHOTS, "A1_instructions.png"), full_page=True)

        # ---------------- start, navigation, zoom ----------------
        fill_gate(page, "qa-alpha", "V1")
        first_id = page.inner_text("#itemid")
        check("item ID is neutral (A-nnn only)", bool(re.fullmatch(r"A-\d{3}", first_id)),
              first_id)
        check("image renders at native 512 at 100%",
              page.eval_on_selector("#theimage", "e=>e.naturalWidth") == 512 and
              page.eval_on_selector("#theimage", "e=>e.clientWidth") == 512)
        check("only one image element is shown (no paired counterpart)",
              page.eval_on_selector_all("#app img", "els=>els.length") == 1)
        page.screenshot(path=os.path.join(SHOTS, "A2_item_view.png"), full_page=True)

        page.click("#zoomin"); page.click("#zoomin")
        check("zoom in enlarges the image",
              page.eval_on_selector("#theimage", "e=>e.clientWidth") > 512,
              page.eval_on_selector("#theimage", "e=>e.clientWidth"))
        page.click("#zoomreset")
        check("zoom reset returns to native 512",
              page.eval_on_selector("#theimage", "e=>e.clientWidth") == 512)

        answer_current(page, 0)
        page.click("#nextbtn")
        second_id = page.inner_text("#itemid")
        check("navigation advances to a different item", second_id != first_id)
        page.click("#prevbtn")
        check("previous returns to the first item",
              page.inner_text("#itemid") == first_id)
        check("answers persist across navigation",
              page.is_checked(f'input[name="target"][value="{TARGETS[0]}"]'))

        # ---------------- missing answers: final must stay unavailable ----
        open_export(page)
        check("Final export disabled while answers are missing",
              page.get_attribute("#finalbtn", "disabled") is not None or
              page.eval_on_selector("#finalbtn", "e=>e.disabled"))

        # ---------------- save / resume ----------------
        page.click("#savebtn")
        page.goto(URL, wait_until="load")
        page.fill("#alias", "qa-alpha")
        page.wait_for_timeout(200)
        check("resume offered for a known alias",
              page.eval_on_selector("#resumebtn", "e=>!e.hidden"))
        page.click("#resumebtn")
        page.wait_for_selector("#app:not([hidden])")
        resumed = page.eval_on_selector_all(
            'input[name="target"]:checked', "e=>e.length")
        check("resumed session restores a saved answer", resumed == 1)

        # ---------------- storage isolation across alias ----------------
        page.goto(URL, wait_until="load")
        fill_gate(page, "qa-beta", "V2")
        check("different alias starts clean (alias isolation)",
              page.eval_on_selector_all('input[name="target"]:checked', "e=>e.length") == 0)
        keys = page.evaluate("Object.keys(localStorage)")
        check("storage keys namespaced by package+stage+alias",
              all(k.startswith("THR::THR-V1-A::A::") for k in keys) and
              len([k for k in keys if k.endswith("qa-alpha")]) == 1 and
              len([k for k in keys if k.endswith("qa-beta")]) == 1,
              keys)
        check("order variant V2 differs from V1",
              page.inner_text("#itemid") != first_id)

        # ---------------- complete final round-trip ----------------
        page.goto(URL, wait_until="load")
        fill_gate(page, "qa-final", "V1")
        for i in range(n_items):
            answer_current(page, i)
            if i < n_items - 1:
                page.click("#nextbtn")
        page.click("#savebtn")
        open_export(page)
        check("Final enabled once all items are answered",
              not page.eval_on_selector("#finalbtn", "e=>e.disabled"))
        with page.expect_download() as dl:
            page.click("#finalbtn")
        fpath = os.path.join(ROOT, "qa", "fixtures", "browser_final_roundtrip.json")
        dl.value.save_as(fpath)
        exported = json.load(open(fpath))
        check("exported final has the expected shape",
              exported["stage"] == "A" and exported["export_kind"] == "final"
              and len(exported["responses"]) == n_items
              and exported["package_id"] == "THR-V1-A")
        check("export carries explicit exposure declarations",
              exported["exposure"]["declared"] is True and
              isinstance(exported["exposure"]["prior_access_to_source_pairings"], bool))
        check("export contains no condition/pairing/seed fields",
              not re.search(r'"(condition|case_id|eval_seed|source_filename|prompt)"',
                            json.dumps(exported)))

        # validator must accept it
        import subprocess
        v = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "validate_export.py"),
                            fpath, "--expect-stage", "A"],
                           capture_output=True, text=True)
        try:
            vr = json.loads(v.stdout)["results"][0]
        except Exception:
            vr = {"valid": False, "errors": ["validator output unparsable"]}
        check("validator accepts the browser-produced final",
              vr["valid"], vr.get("errors"))

        # ---------------- draft export + import round-trip ----------------
        page.goto(URL, wait_until="load")
        fill_gate(page, "qa-draft", "V3")
        answer_current(page, 1)
        page.click("#nextbtn"); answer_current(page, 2)
        open_export(page)
        with page.expect_download() as dl2:
            page.click("#draftbtn")
        dpath = os.path.join(ROOT, "qa", "fixtures", "browser_draft_roundtrip.json")
        dl2.value.save_as(dpath)
        draft = json.load(open(dpath))
        check("draft export is marked draft and may be partial",
              draft["export_kind"] == "draft" and
              sum(1 for r in draft["responses"] if r["target"] and r["rendering"]) == 2)

        page.goto(URL, wait_until="load")
        fill_gate(page, "qa-import", "V3")
        open_export(page)
        page.set_input_files("#importfile", dpath)
        page.wait_for_timeout(600)
        msg = page.inner_text("#importmsg")
        check("draft import restores items", "Restored" in msg, msg)

        # ---------------- malformed / wrong-stage import ----------------
        bad = os.path.join(tempfile.gettempdir(), "thr_bad.json")
        open(bad, "w").write("{ not json ,,,")
        page.set_input_files("#importfile", bad)
        page.wait_for_timeout(500)
        check("malformed import rejected without crashing",
              "Rejected" in page.inner_text("#importmsg"),
              page.inner_text("#importmsg"))
        wrong = os.path.join(tempfile.gettempdir(), "thr_wrongstage.json")
        json.dump({"package_id": "THR-V1-B", "stage": "B", "responses": []}, open(wrong, "w"))
        page.set_input_files("#importfile", wrong)
        page.wait_for_timeout(500)
        check("wrong-package/stage import rejected",
              "different package or stage" in page.inner_text("#importmsg"),
              page.inner_text("#importmsg"))
        check("viewer still usable after bad imports", page.is_visible("#itemid"))
        ctx.close()

        # ---------------- mobile readability ----------------
        mctx = br.new_context(viewport={"width": 390, "height": 844},
                              device_scale_factor=2)
        mp = setup_page(mctx)
        mp.goto(URL, wait_until="load")
        mp.screenshot(path=os.path.join(SHOTS, "A3_mobile_instructions.png"), full_page=True)
        fill_gate(mp, "qa-mobile", "V1")
        ow = mp.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
        check("no horizontal page overflow at 390px", ow[0] <= ow[1] + 1, ow)
        check("question controls reachable on mobile",
              mp.is_visible('input[name="target"][value="present"]'))
        mp.screenshot(path=os.path.join(SHOTS, "A4_mobile_item.png"), full_page=True)
        mctx.close()
        br.close()

    check("viewer made no network requests (all non-file:// blocked)",
          not network_attempts, network_attempts[:5])

    n_pass = sum(1 for r in results if r["pass"])
    rep = {
        "suite": "real-browser QA (Playwright + Firefox, headless)",
        "browser": "Firefox (Playwright standalone build, not the system snap)",
        "url_scheme": "file:// (offline; every non-file request aborted)",
        "scope": ("Mechanical UI behaviour only. Contains no human judgement about "
                  "any image and no image-specific result. Clicked choices come "
                  "from a fixed rotation and are discarded."),
        "contains_reviewer_evidence": False,
        "n_checks": len(results), "n_pass": n_pass, "n_fail": len(results) - n_pass,
        "all_pass": n_pass == len(results),
        "network_requests_attempted": network_attempts,
        "screenshots": sorted(os.listdir(SHOTS)) if os.path.isdir(SHOTS) else [],
        "checks": results,
    }
    json.dump(rep, open(os.path.join(ROOT, "qa", "browser_qa_report.json"), "w"), indent=1)
    print(f"\n{n_pass}/{len(results)} browser checks pass")
    sys.exit(0 if rep["all_pass"] else 4)


if __name__ == "__main__":
    main()
