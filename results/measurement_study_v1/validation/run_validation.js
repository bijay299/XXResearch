// Real-browser validation of the measurement_study_v1 review viewers.
// Uses SYNTHETIC FIXTURES ONLY. No human responses are simulated as data; the scripted
// clicks exist to exercise the instrument, and their outputs are discarded.
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path'), os = require('os');

const ROOT = '/data/bijaypandey/cuig_pilot/measurement_study_v1';
const FIXA = 'file://' + ROOT + '/validation/fixture_pkg/index.html';
const FIXB = 'file://' + ROOT + '/validation/fixture_pkg_B/index.html';
const REAL = 'file://' + ROOT + '/public_build/taskA_pkg/index.html';

const results = [];
function check(name, pass, detail) {
  results.push({ name, pass: !!pass, detail: detail || '' });
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
}

async function newCtx(browser, opts) {
  const ctx = await browser.newContext(Object.assign({ acceptDownloads: true }, opts || {}));
  // Any attempt to reach the NETWORK is a failure of the offline requirement.
  // file:// must pass through untouched, otherwise we would block the page itself.
  const trap = r => { netHits.push(r.request().url()); r.abort(); };
  await ctx.route('http://**', trap);
  await ctx.route('https://**', trap);
  return ctx;
}
let netHits = [];

async function start(page, alias, variant, exposure) {
  await page.fill('#alias', alias);
  if (variant) await page.selectOption('#variant', variant);
  await page.check(`input[name="exposure"][value="${exposure || 'none'}"]`);
  await page.click('#btnStart');
  await page.waitForTimeout(120);
}

async function answerAll(page, q1, q2) {
  const n = await page.evaluate(() => window.__CFG__.items.length);
  for (let i = 0; i < n; i++) {
    await page.check(`input[name="q1"][value="${q1}"]`);
    if (q2) await page.check(`input[name="q2"][value="${q2}"]`);
    const dis = await page.isDisabled('#btnNext');
    if (!dis) await page.click('#btnNext');
    await page.waitForTimeout(40);
  }
}

async function grabDownload(page, sel) {
  const [dl] = await Promise.all([page.waitForEvent('download'), page.click(sel)]);
  const p = path.join(os.tmpdir(), 'msv1_' + Date.now() + '.json');
  await dl.saveAs(p);
  const j = JSON.parse(fs.readFileSync(p, 'utf8'));
  fs.unlinkSync(p);
  return j;
}

(async () => {
  const browser = await chromium.launch();

  // ---------------------------------------------------------------- 1. gate validation
  {
    const ctx = await newCtx(browser); const page = await ctx.newPage();
    await page.goto(FIXA);
    await page.click('#btnStart');
    check('gate rejects empty alias', await page.isVisible('#gateErr'));
    await page.fill('#alias', 'bad alias!!');
    await page.check('input[name="exposure"][value="none"]');
    await page.click('#btnStart');
    check('gate rejects malformed alias', await page.isVisible('#gateErr'));
    await page.fill('#alias', 'R1');
    await page.goto(FIXA); await page.fill('#alias', 'R1'); await page.click('#btnStart');
    check('gate requires prior-exposure answer', await page.isVisible('#gateErr'));
    await ctx.close();
  }

  // ------------------------------------------- 2. no preselected answers; required fields
  {
    const ctx = await newCtx(browser); const page = await ctx.newPage();
    await page.goto(FIXA); await start(page, 'R1', 'A', 'none');
    const pre = await page.evaluate(() =>
      document.querySelectorAll('input[name="q1"]:checked,input[name="q2"]:checked').length);
    check('no answer is preselected or inferred', pre === 0, `${pre} preselected`);
    const finalDisabled = await page.isDisabled('#btnFinal');
    check('final export blocked until every item answered', finalDisabled);
    const d = await grabDownload(page, '#btnDraft');
    check('draft export allowed with nothing answered', d.export_kind === 'draft' && d.n_answered === 0,
      `answered=${d.n_answered} missing=${d.n_missing}`);
    check('draft marks every unanswered item missing (null, not guessed)',
      d.responses.every(r => r.target_presence === null && r.scene_quality === null && !r.answered));
    await ctx.close();
  }

  // ------------------------------------------------- 3. autosave / resume across reloads
  {
    const ctx = await newCtx(browser); const page = await ctx.newPage();
    await page.goto(FIXA); await start(page, 'R2', 'B', 'some');
    await page.check('input[name="q1"][value="present"]');
    await page.check('input[name="q2"][value="coherent"]');
    await page.click('#btnNext');
    await page.check('input[name="q1"][value="uncertain"]');
    await page.check('input[name="q2"][value="unable_to_judge"]');
    await page.fill('#why', 'fixture note');
    await page.waitForTimeout(100);
    const idBefore = await page.textContent('#neutralId');
    await page.reload(); await start(page, 'R2', 'B', 'some');
    const idAfter = await page.textContent('#neutralId');
    const q1 = await page.inputValue('#why');
    const checked = await page.evaluate(() =>
      document.querySelector('input[name="q1"]:checked')?.value);
    check('resume restores position after reload', idBefore === idAfter, `${idBefore} -> ${idAfter}`);
    check('resume restores saved answer', checked === 'uncertain', `got ${checked}`);
    check('resume restores free-text explanation', q1 === 'fixture note');
    check('uncertainty is preserved as a real answer', checked === 'uncertain');
    await ctx.close();
  }

  // ------------------------------------------------------ 4. order stability and variants
  {
    const ctx = await newCtx(browser); const page = await ctx.newPage();
    const ord = async (alias, variant) => {
      await page.goto(FIXA); await start(page, alias, variant, 'none');
      return page.evaluate(() => S.items.join(','));
    };
    const a1 = await ord('RA', 'A'), a2 = await ord('RA', 'A');
    const b1 = await ord('RB', 'A'), v2 = await ord('RA', 'B'), v3 = await ord('RA', 'C');
    check('order is stable for the same alias+variant', a1 === a2);
    check('order differs between reviewers', a1 !== b1);
    check('three order variants give three different orders',
      new Set([a1, v2, v3]).size === 3);
    check('order is a pure function of alias+variant, not of any arm mapping',
      await page.evaluate(() => !JSON.stringify(window.__CFG__.items).includes('arm')));
    await ctx.close();
  }

  // ------------------------------------- 5. full pass, final export, clean-session import
  let finalExport = null;
  {
    const ctx = await newCtx(browser); const page = await ctx.newPage();
    await page.goto(FIXA); await start(page, 'R3', 'A', 'none');
    await answerAll(page, 'present', 'coherent');
    await page.waitForTimeout(150);
    check('final export enabled once all answered', !(await page.isDisabled('#btnFinal')));
    finalExport = await grabDownload(page, '#btnFinal');
    check('final export has stage/package/protocol/schema ids',
      finalExport.stage === 'A' && finalExport.package_id === 'msv1-FIXTURE-A' &&
      finalExport.protocol_id === 'msv1-protocol-v1' && finalExport.schema_version === '1.0.0');
    check('final export carries alias, order variant and exposure declaration',
      finalExport.reviewer_alias === 'R3' && finalExport.order_variant === 'A' &&
      finalExport.prior_exposure === 'none');
    check('final export records neutral ids and positions',
      finalExport.responses.every(r => /^X\d{3}$/.test(r.neutral_id) && r.position >= 1));
    check('final export contains no prompt/seed/file/arm field',
      !/prompt|seed|arm|method|file|sha/i.test(JSON.stringify(finalExport)));
    check('final export counts answered/missing', finalExport.n_answered === 8 && finalExport.n_missing === 0);
    await ctx.close();
  }
  {
    const ctx = await newCtx(browser); const page = await ctx.newPage();   // clean session
    await page.goto(FIXA); await start(page, 'R9', 'C', 'none');
    const p = path.join(os.tmpdir(), 'imp.json');
    fs.writeFileSync(p, JSON.stringify(finalExport));
    await page.setInputFiles('#importer', p);
    await page.waitForTimeout(200);
    const msg = await page.textContent('#importMsg');
    const done = await page.textContent('#prog');
    check('import round-trip into a clean session restores answers',
      /Imported 8 answer/.test(msg) && /8 \/ 8/.test(done), `${msg} | ${done}`);
    fs.unlinkSync(p);
    await ctx.close();
  }

  // ------------------------------------------------------------ 6. malformed/bad imports
  {
    const ctx = await newCtx(browser); const page = await ctx.newPage();
    await page.goto(FIXA); await start(page, 'R4', 'A', 'none');
    // NOTE: each case needs its OWN file written lazily; an earlier version of this suite
    // wrote every case to one path eagerly, so only the last survived.
    let n = 0;
    const tmp = f => { const p = path.join(os.tmpdir(), 'bad' + (n++) + '.json'); fs.writeFileSync(p, f); return p; };
    const cases = [
      ['malformed JSON', () => tmp('{not json'), /Could not parse/],
      ['wrong stage', () => tmp(JSON.stringify(Object.assign({}, finalExport, { stage: 'B' }))), /stage/],
      ['wrong package id', () => tmp(JSON.stringify(Object.assign({}, finalExport, { package_id: 'other' }))), /package id/],
      ['wrong schema version', () => tmp(JSON.stringify(Object.assign({}, finalExport, { schema_version: '9.9.9' }))), /schema version/],
      ['no responses array', () => tmp(JSON.stringify(Object.assign({}, finalExport, { responses: null }))), /responses/],
    ];
    for (const [nm, mk, re] of cases) {
      const p = mk();
      await page.setInputFiles('#importer', p);
      await page.waitForTimeout(150);
      const m = await page.textContent('#importMsg');
      check(`import rejects ${nm}`, re.test(m), m.slice(0, 90));
      fs.unlinkSync(p);
    }
    const mixed = JSON.parse(JSON.stringify(finalExport));
    mixed.responses.push(Object.assign({}, mixed.responses[0]));              // duplicate
    mixed.responses.push(Object.assign({}, mixed.responses[0], { neutral_id: 'ZZZ999' })); // unknown
    const p2 = tmp(JSON.stringify(mixed));
    await page.setInputFiles('#importer', p2); await page.waitForTimeout(200);
    const m2 = await page.textContent('#importMsg');
    check('import reports unknown ids and duplicates instead of silently accepting',
      /unknown id/.test(m2) && /duplicate/.test(m2), m2.slice(0, 120));
    fs.unlinkSync(p2);
    await ctx.close();
  }

  // --------------------------------------------------- 7. per-reviewer storage isolation
  {
    const ctx = await newCtx(browser); const page = await ctx.newPage();
    await page.goto(FIXA); await start(page, 'ISO1', 'A', 'none');
    await page.check('input[name="q1"][value="absent"]');
    await page.check('input[name="q2"][value="coherent"]');
    await page.waitForTimeout(120);
    await page.goto(FIXA); await start(page, 'ISO2', 'A', 'none');
    const other = await page.evaluate(() =>
      document.querySelector('input[name="q1"]:checked')?.value || null);
    const keys = await page.evaluate(() => Object.keys(localStorage).filter(k => k.startsWith('msv1::')));
    check('a second reviewer alias starts clean on the same browser', other === null, `saw ${other}`);
    check('storage is namespaced per stage/package/alias',
      keys.some(k => k.includes('ISO1')) && keys.some(k => k.includes('ISO2')), keys.join(' | '));
    await ctx.close();
  }

  // --------------------------------------------- 8. zoom, navigation, mobile readability
  {
    const ctx = await newCtx(browser); const page = await ctx.newPage();
    await page.goto(FIXA); await start(page, 'R5', 'A', 'none');
    await page.click('#btnZoom');
    const zoomed = await page.evaluate(() => document.querySelector('.imgwrap').classList.contains('zoom'));
    const label = await page.textContent('#btnZoom');
    check('zoom to full resolution toggles', zoomed && /Fit to window/.test(label));
    await page.click('#btnZoom');
    check('zoom toggles back to fit', !(await page.evaluate(() => document.querySelector('.imgwrap').classList.contains('zoom'))));
    check('previous is disabled on the first item', await page.isDisabled('#btnPrev'));
    await page.click('#btnNext'); await page.waitForTimeout(80);
    check('navigation advances and enables previous', !(await page.isDisabled('#btnPrev')));
    await ctx.close();

    const mctx = await newCtx(browser, { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
    const mp = await mctx.newPage();
    await mp.goto(FIXA); await start(mp, 'R6', 'A', 'none');
    const overflow = await mp.evaluate(() =>
      document.documentElement.scrollWidth - document.documentElement.clientWidth);
    check('no horizontal overflow at 390px width', overflow <= 1, `overflow ${overflow}px`);
    // measure a VISIBLE answer option inside the review card (an earlier version of this
    // suite measured the first label.opt in the DOM, which sits in the hidden gate section)
    const tap = await mp.evaluate(() => {
      const r = document.querySelector('#review label.opt').getBoundingClientRect(); return r.height;
    });
    check('answer targets are tappable on mobile (>=36px)', tap >= 36, `${Math.round(tap)}px`);
    await mctx.close();
  }

  // ------------------------------------------------------- 9. stage B is a distinct stage
  {
    const ctx = await newCtx(browser); const page = await ctx.newPage();
    await page.goto(FIXB); await start(page, 'R7', 'A', 'none');
    const hasRef = await page.isVisible('#imgRef');
    const q = await page.evaluate(() => Array.from(document.querySelectorAll('input[name="q1"]')).map(i => i.value));
    check('stage B shows a reference pane beside the candidate', hasRef);
    check('stage B asks the deformation-comparison question',
      JSON.stringify(q) === JSON.stringify(['less', 'about_same', 'more', 'cannot_compare']), q.join(','));
    await page.check('input[name="q1"][value="about_same"]');
    await page.waitForTimeout(80);
    const d = await grabDownload(page, '#btnDraft');
    check('stage B export is tagged stage B with its own field',
      d.stage === 'B' && 'deformation_vs_reference' in d.responses[0]);
    const p = path.join(os.tmpdir(), 'astage.json');
    fs.writeFileSync(p, JSON.stringify(finalExport));              // a stage-A file
    await page.setInputFiles('#importer', p); await page.waitForTimeout(150);
    check('stage B viewer refuses a stage-A export', /stage/.test(await page.textContent('#importMsg')));
    fs.unlinkSync(p);
    await ctx.close();
  }

  // ------------------------- 10. the REAL public bundle: empty panel, no Task-B leakage
  {
    const ctx = await newCtx(browser); const page = await ctx.newPage();
    const errs = []; page.on('pageerror', e => errs.push(String(e)));
    await page.goto(REAL);
    await page.click('#btnPractice');
    const npr = await page.evaluate(() => document.querySelectorAll('#practice .grid figure').length);
    check('public bundle shows the 12 practice items', npr === 12, `${npr} figures`);
    await start(page, 'R8', 'A', 'none');
    check('public bundle states plainly that it holds no study items',
      await page.isVisible('#emptyPanel'));
    check('public bundle raises no JavaScript errors', errs.length === 0, errs.join('; ').slice(0, 120));
    const cfg = await page.evaluate(() => JSON.stringify(window.__CFG__));
    check('public bundle carries no Task-B reference associations', !/ref_file/.test(cfg));
    check('public bundle carries no arm/method/seed/prompt keys',
      !/\b(arm|method|seed|prompt|source|sha256|erased|original)\b/i.test(cfg));
    const d = await grabDownload(page, '#btnDraft');
    check('public bundle still exports a well-formed (empty) response file',
      d.stage === 'A' && d.n_items === 0 && Array.isArray(d.responses) && d.responses.length === 0);
    await ctx.close();
  }

  check('no network request was made by any viewer (offline)', netHits.length === 0,
    netHits.slice(0, 3).join(', '));

  await browser.close();
  const pass = results.filter(r => r.pass).length;
  fs.writeFileSync(ROOT + '/validation/validation_results.json', JSON.stringify({
    generated_utc: new Date().toISOString(),
    browser: 'Chromium (Playwright 1.49.0, headless)',
    fixtures: 'synthetic only; no human responses simulated as data',
    total: results.length, passed: pass, failed: results.length - pass, results
  }, null, 1));
  console.log(`\n${pass}/${results.length} checks passed`);
  process.exit(pass === results.length ? 0 : 1);
})();
