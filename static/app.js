/* ==========================================================================
   Dermora AI — front end
   Vanilla ES2020. No framework, no CDN, no build step: the whole interface is
   this one file plus styles.css, so it runs offline on any machine that can
   reach the local server.
   ========================================================================== */

const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const RISK_COLOR = {
  CRITICAL: '#ff3b5c', HIGH: '#ff7a45', MODERATE: '#ffc53d', LOW: '#3ddc97',
};
const BAND_RISK = { URGENT: 'CRITICAL', PRIORITY: 'HIGH', REVIEW: 'MODERATE', ROUTINE: 'LOW' };

const state = {
  current: null,        // active case payload
  view: 'analyse',
  modelReady: false,
  imageTab: 'attention',
  cases: [],
  batchSort: { key: 'band', dir: 1 },
  batchRows: [],
};

/* ────────────────────────────── utilities ───────────────────────────── */

const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/** Inline custom properties for a risk colour, including pre-computed alpha
 *  variants. Avoids relying on CSS color-mix(), which older browsers ignore
 *  entirely — and an ignored background is an invisible badge. */
function riskVars(hex) {
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  const a = (v) => `rgba(${r},${g},${b},${v})`;
  return `--risk:${hex};--risk-a09:${a(0.09)};--risk-a16:${a(0.16)};` +
         `--risk-a24:${a(0.24)};--risk-a30:${a(0.3)};--risk-a38:${a(0.38)}`;
}

function toast(msg, isError = false) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.toggle('err', isError);
  el.classList.add('on');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove('on'), 3600);
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not json */ }
    throw new Error(detail);
  }
  return res.json();
}

/* ────────────────────────────── navigation ──────────────────────────── */

function showView(name) {
  state.view = name;
  $$('.nav-item').forEach((b) => b.classList.toggle('active', b.dataset.view === name));
  $$('.view').forEach((v) => v.classList.toggle('active', v.id === `view-${name}`));
  $('#topnav')?.classList.remove('open');
  window.scrollTo(0, 0);
  if (name === 'cases') loadCases();
  if (name === 'model') loadModelCard();
  if (name === 'home') populateTypeGrid();
  if (name === 'learn') populateLearnTaxo();
}

// Any element carrying data-view switches the page — nav links, the hero CTA,
// footer links, the brand mark. Not just the nav bar itself.
$$('[data-view]').forEach((b) => b.addEventListener('click', () => showView(b.dataset.view)));

// data-scroll elements smooth-scroll to an in-page anchor (hero "see how it
// works", the Learn table of contents) rather than switching views.
$$('[data-scroll]').forEach((b) => b.addEventListener('click', (ev) => {
  ev.preventDefault();
  document.getElementById(b.dataset.scroll)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}));

$('#burger')?.addEventListener('click', () => $('#topnav').classList.toggle('open'));

populateFAQ();

/* ──────────────────────────── model status ──────────────────────────── */

async function pollStatus() {
  try {
    const h = await api('/api/health');
    const dot = $('#status .dot');
    state.modelReady = h.model_status === 'ready';
    dot.className = 'dot ' + (h.model_status === 'ready' ? 'ready'
      : h.model_status === 'error' ? 'error' : 'loading');
    $('#statusText').textContent = h.model_status === 'ready' ? 'Model ready'
      : h.model_status === 'error' ? 'Model failed' : 'Loading model';
    $('#statusSub').textContent = h.model_status === 'ready'
      ? `yolov8n · ${h.imgsz}px · ${h.tta ? '4-view TTA' : 'single view'}`
      : (h.error ? h.error.slice(0, 42) : 'reading checkpoint');
    $('#caseCount').textContent = h.cases_in_session;
    if (h.model_status === 'loading') setTimeout(pollStatus, 900);
    if (h.model_status === 'error') toast(h.error, true);
  } catch {
    $('#statusText').textContent = 'Server unreachable';
    $('#status .dot').className = 'dot error';
    setTimeout(pollStatus, 2500);
  }
}

/* ───────────────────────────── file intake ──────────────────────────── */

const dropzone = $('#dropzone');
const fileInput = $('#fileInput');

['dragenter', 'dragover'].forEach((e) =>
  dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.add('over'); }));
['dragleave', 'drop'].forEach((e) =>
  dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.remove('over'); }));

dropzone.addEventListener('drop', (ev) => {
  const f = ev.dataTransfer.files[0];
  if (f) analyseFile(f);
});

// The <input type="file"> sits inside the dropzone, so the synthetic click we
// dispatch on it bubbles straight back up to the dropzone's own handler. That
// re-entrant loop makes the browser refuse to open the file chooser at all
// ("File chooser dialog can only be shown with a user activation"), which looks
// exactly like a dead button. Stop the input's clicks at the source.
fileInput.addEventListener('click', (ev) => ev.stopPropagation());

function openFilePicker() {
  fileInput.value = '';          // re-selecting the same file must still fire change
  fileInput.click();
}

dropzone.addEventListener('click', (ev) => {
  if (ev.target === fileInput) return;
  openFilePicker();
});
$('#btnPick').addEventListener('click', openFilePicker);
fileInput.addEventListener('change', () => { if (fileInput.files[0]) analyseFile(fileInput.files[0]); });
$('#btnSamples').addEventListener('click', () => {
  dropzone.classList.remove('hidden');
  $('#result').classList.add('hidden');
  $('#dzSamples').scrollIntoView({ block: 'center' });
});

async function loadSamples() {
  try {
    const { samples } = await api('/api/samples');
    if (!samples.length) return;
    $('#dzSamples').innerHTML = samples.map((s) => `
      <button class="chip-sample" data-sample="${esc(s.name)}">
        <img src="${esc(s.url)}" alt=""> ${esc(s.name)}
      </button>`).join('');
    $$('#dzSamples .chip-sample').forEach((b) =>
      b.addEventListener('click', (ev) => { ev.stopPropagation(); analyseSample(b.dataset.sample); }));
  } catch { /* samples are optional */ }
}

/* ───────────────────────── progress choreography ────────────────────── */

const STEPS = ['quality', 'infer', 'cam', 'morph'];

function startProgress(label) {
  dropzone.classList.add('hidden');
  $('#result').classList.add('hidden');
  const box = $('#progress');
  box.classList.remove('hidden');
  $('#progFile').textContent = label;
  $$('.steps li').forEach((li) => li.className = '');
  $('#progress .prog-bar span').style.width = '4%';

  let i = 0;
  const advance = () => {
    if (i > 0) $(`.steps li[data-step="${STEPS[i - 1]}"]`).className = 'done';
    if (i < STEPS.length) {
      $(`.steps li[data-step="${STEPS[i]}"]`).className = 'run';
      $('#progress .prog-bar span').style.width = `${12 + i * 26}%`;
      i += 1;
      startProgress._t = setTimeout(advance, i === 2 ? 700 : 380);
    }
  };
  advance();
}

function finishProgress() {
  clearTimeout(startProgress._t);
  $$('.steps li').forEach((li) => li.className = 'done');
  $('#progress .prog-bar span').style.width = '100%';
  setTimeout(() => $('#progress').classList.add('hidden'), 260);
}

function abortProgress() {
  clearTimeout(startProgress._t);
  $('#progress').classList.add('hidden');
  dropzone.classList.remove('hidden');
}

/* ─────────────────────────────── analysis ───────────────────────────── */

async function analyseFile(file) {
  if (!file.type.startsWith('image/') && !/\.(jpe?g|png|bmp|webp|tiff?)$/i.test(file.name)) {
    return toast('That file is not an image.', true);
  }
  // Fail loudly rather than letting the request 503 with no visible cause.
  if (!state.modelReady) {
    return toast('The model is not ready yet — check the status indicator in the header.', true);
  }
  startProgress(file.name);
  const fd = new FormData();
  fd.append('file', file);
  try {
    const data = await api('/api/analyze', { method: 'POST', body: fd });
    finishProgress();
    renderCase(data);
  } catch (err) {
    abortProgress();
    toast(err.message, true);
  }
}

async function analyseSample(name) {
  startProgress(name);
  try {
    const data = await api(`/api/analyze-sample?name=${encodeURIComponent(name)}`, { method: 'POST' });
    finishProgress();
    renderCase(data);
  } catch (err) {
    abortProgress();
    toast(err.message, true);
  }
}

/* ───────────────────────────── result render ────────────────────────── */

function renderCase(d) {
  state.current = d;
  const p = d.prediction;
  const risk = d.triage.risk;
  const color = RISK_COLOR[risk] || RISK_COLOR.LOW;
  const box = $('#result');

  box.innerHTML = `
    <div class="result-actions">
      <div class="result-actions-text">
        <b>Take this to a doctor</b>
        <span>Download a complete PDF case report — imagery, full probability breakdown,
          ABCD morphometry and management guidance — to bring to your appointment.</span>
      </div>
      <div class="result-actions-btns">
        <button class="btn" id="btnPdfTop">
          <svg viewBox="0 0 24 24"><path d="M12 4v11M8 11l4 4 4-4M5 19h14"/></svg>
          Download PDF for your doctor
        </button>
        <button class="btn ghost sm" id="btnAnotherTop">Analyse another</button>
      </div>
    </div>
    <div class="col-left">
      ${viewerPanel(d)}
      ${morphPanel(d)}
    </div>
    <div class="col-right">
      ${diagnosisPanel(d, color)}
      ${detailPanel(d)}
    </div>`;

  box.classList.remove('hidden');
  wireResult(d);

  // Animate the bars and the ring after paint so the transitions actually run.
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $$('.prow-bar i', box).forEach((el) => { el.style.width = el.dataset.w; });
    const ring = $('.ring .fill', box);
    if (ring) ring.style.strokeDashoffset = ring.dataset.off;
    const marker = $('.tds-scale i', box);
    if (marker) marker.style.left = marker.dataset.left;
  }));

  loadCases();
  if (p.degraded) toast('Running in degraded mode — raw head unavailable on this build.', true);
}

function viewerPanel(d) {
  const imgs = d.images || {};
  const tabs = [
    ['attention', 'Attention', '1', 'Eigen-CAM over the P3/P4/P5 neck features. Warm regions drove the prediction.'],
    ['original', 'Original', '2', 'The submitted image, unmodified.'],
    ['outline', 'Segmentation', '3', 'Lesion boundary from Otsu thresholding after DullRazor hair removal. All morphometry is computed inside this contour.'],
    ['roi', 'Detector ROI', '4', 'The highest-scoring anchor box. Training labels covered the whole frame, so this is expected to be near-full-frame — it is shown for transparency, not localisation.'],
  ].filter(([k]) => imgs[k]);

  const active = imgs[state.imageTab] ? state.imageTab : (tabs[0]?.[0] || 'original');

  return `
  <div class="panel">
    <div class="viewer">
      ${tabs.map(([k]) => `<img src="${imgs[k]}" data-img="${k}" class="${k === active ? 'on' : ''}" alt="">`).join('')}
    </div>
    <div class="tabs">
      ${tabs.map(([k, label, key]) => `
        <button class="tab ${k === active ? 'on' : ''}" data-tab="${k}">${label}<kbd>${key}</kbd></button>`).join('')}
    </div>
    <p class="viewer-note" id="viewerNote">${esc(tabs.find(([k]) => k === active)?.[3] || '')}</p>
  </div>`;
}

function diagnosisPanel(d, color) {
  const p = d.prediction;
  const C = 2 * Math.PI * 40;
  const off = C * (1 - p.confidence);

  const q = d.quality || {};
  const qCls = q.score >= 75 ? 'good' : q.score >= 45 ? 'warn' : 'bad';
  const aCls = p.tta_agreement >= 1 ? 'good' : p.tta_agreement >= 0.75 ? 'warn' : 'bad';
  const mCls = p.margin >= 0.3 ? 'good' : p.margin >= 0.1 ? 'warn' : 'bad';

  return `
  <div class="panel" style="${riskVars(color)}">
    <div class="dx">
      <div class="dx-top">
        <div>
          <span class="pill">${esc(d.triage.band)} · ${esc(d.lesion.category)}</span>
          <div class="dx-name" style="margin-top:10px">${esc(p.top_full_name)}</div>
          <div class="dx-code">${esc(p.top_class)} · case ${esc(d.case_id)} · ${esc(d.filename)}</div>
        </div>
        <svg class="ring" viewBox="0 0 100 100">
          <circle class="track" cx="50" cy="50" r="40"></circle>
          <circle class="fill" cx="50" cy="50" r="40"
                  stroke-dasharray="${C.toFixed(1)}" stroke-dashoffset="${C.toFixed(1)}"
                  data-off="${off.toFixed(1)}" transform="rotate(-90 50 50)"></circle>
          <text x="50" y="49">${(p.confidence * 100).toFixed(0)}<tspan font-size="11">%</tspan></text>
          <text x="50" y="62" class="sub">CONFIDENCE</text>
        </svg>
      </div>
      <div class="dx-triage">
        <b>${esc(d.triage.headline)}</b>
        <p>${esc(d.triage.detail)}</p>
      </div>
    </div>

    <div class="metrics">
      <div class="metric ${d.concern >= 0.3 ? 'bad' : 'good'}">
        <b>${pct(d.concern, 0)}</b><span>malignant +<br>pre-malignant</span></div>
      <div class="metric ${mCls}"><b>${(p.margin * 100).toFixed(0)}</b><span>top-two<br>margin (pts)</span></div>
      <div class="metric ${aCls}"><b>${pct(p.tta_agreement, 0)}</b><span>TTA view<br>agreement</span></div>
      <div class="metric ${qCls}"><b>${(q.score ?? 0).toFixed(0)}</b><span>image<br>quality</span></div>
    </div>

    <div class="panel-body">
      <div class="probs">
        ${p.ranked.map((r, i) => `
          <div class="prow ${i === 0 ? 'lead' : ''}" style="--c:${r.color}">
            <div class="prow-code">${esc(r.code)}</div>
            <div class="prow-bar"><i data-w="${Math.max(r.prob * 100, 0.6)}%"></i><span>${esc(r.name)}</span></div>
            <div class="prow-val">${(r.prob * 100).toFixed(1)}</div>
          </div>`).join('')}
      </div>
    </div>
  </div>`;
}

function detailPanel(d) {
  const p = d.prediction;
  const les = d.lesion;
  const q = d.quality || {};

  const caveats = (d.caveats || []).length ? `
    <div class="caveats">
      ${d.caveats.map((c) => `
        <div class="caveat">
          <svg viewBox="0 0 24 24"><path d="M12 9v5M12 17.5v.5M10.3 4.3 2.6 18a2 2 0 0 0 1.7 3h15.4a2 2 0 0 0 1.7-3L13.7 4.3a2 2 0 0 0-3.4 0z"/></svg>
          <span>${esc(c)}</span>
        </div>`).join('')}
    </div>` : `<p class="prose">No quality or consistency flags were raised for this case.</p>`;

  return `
  <div class="panel">
    <div class="subtabs">
      <button class="subtab on" data-pane="interp">Interpretation</button>
      <button class="subtab" data-pane="diff">Differentials</button>
      <button class="subtab" data-pane="flags">Flags${(d.caveats || []).length ? ` · ${d.caveats.length}` : ''}</button>
      <button class="subtab" data-pane="tech">Technical</button>
    </div>

    <div class="subpane on" data-pane="interp">
      <p class="prose">${esc(d.narrative)}</p>
      <p class="prose" style="margin-top:14px"><strong>About this class.</strong> ${esc(les.summary)}</p>
      <h4 class="prose" style="margin:16px 0 9px;color:var(--ink);font-size:12.5px;font-weight:650">
        Dermoscopic features to confirm on examination</h4>
      <ul class="feature-list">${les.dermoscopy.map((f) => `<li>${esc(f)}</li>`).join('')}</ul>
      <p class="prose" style="margin-top:15px"><strong>Suggested management.</strong> ${esc(les.management)}</p>
      <p class="prose"><strong>Urgency.</strong> ${esc(les.urgency)}</p>
      <div style="margin-top:16px;display:flex;gap:9px;flex-wrap:wrap">
        <button class="btn sm" id="btnPdf">
          <svg viewBox="0 0 24 24"><path d="M12 4v11M8 11l4 4 4-4M5 19h14"/></svg>Download report
        </button>
        <button class="btn sm ghost" id="btnAnother">Analyse another</button>
      </div>
    </div>

    <div class="subpane" data-pane="diff">
      <p class="prose" style="margin-bottom:13px">Classes most often confused with
        <strong>${esc(les.name)}</strong>, with this model's probability for each.</p>
      <div class="diffs">
        ${d.differentials.map((x) => `
          <div class="diff" style="${riskVars(RISK_COLOR[x.risk])}">
            <b>${esc(x.name)} <span style="color:var(--ink-3);font-family:var(--mono);font-size:11px">${esc(x.code)}</span></b>
            <span class="p">${(x.prob * 100).toFixed(1)}%</span>
            <em>Discriminating feature — ${esc(x.distinguisher)}</em>
          </div>`).join('')}
      </div>
    </div>

    <div class="subpane" data-pane="flags">${caveats}</div>

    <div class="subpane" data-pane="tech">
      <dl class="kv">
        <dt>Posterior entropy</dt><dd>${p.entropy.toFixed(3)}</dd>
        <dt>Peak detector score</dt><dd>${p.detector_score.toFixed(4)}</dd>
        <dt>Pooled raw score (top)</dt><dd>${(p.per_class_raw[p.top_class] ?? 0).toFixed(4)}</dd>
        <dt>Augmented views</dt><dd>${p.views}</dd>
        <dt>Inference time</dt><dd>${p.inference_ms.toFixed(0)} ms</dd>
        <dt>Total pipeline</dt><dd>${d.elapsed_ms.toFixed(0)} ms</dd>
        <dt>Focus (Laplacian var)</dt><dd>${(q.focus ?? 0).toFixed(1)}</dd>
        <dt>Brightness / contrast</dt><dd>${(q.brightness ?? 0).toFixed(0)} / ${(q.contrast ?? 0).toFixed(0)}</dd>
        <dt>Skin-tone fraction</dt><dd>${pct(q.skin_fraction ?? 0, 0)}</dd>
        <dt>Resolution</dt><dd>${(q.resolution || []).join(' × ')}</dd>
        <dt>Attention concentration</dt><dd>${d.attention?.available ? pct(d.attention.concentration, 0) : '—'}</dd>
      </dl>
      ${d.attention?.available ? `<p class="prose" style="margin-top:13px">${esc(d.attention.interpretation)}</p>` : ''}
    </div>
  </div>

  <div class="disclaimer"><b>Not a diagnosis.</b> ${esc(d.disclaimer)}</div>`;
}

function morphPanel(d) {
  const m = d.morphology;
  if (!m || !m.available) {
    return `<div class="panel"><div class="panel-head"><h3>Morphometry</h3></div>
      <div class="empty">Morphometric analysis was not available for this image.</div></div>`;
  }
  const abcd = m.abcd;
  const risk = abcd.band === 'HIGH' ? RISK_COLOR.CRITICAL
    : abcd.band === 'SUSPICIOUS' ? RISK_COLOR.MODERATE : RISK_COLOR.LOW;
  const markerLeft = Math.min(100, Math.max(0, (abcd.tds / 8.9) * 100));

  const rows = [
    ['A', 'Asymmetry', `${pct(m.asymmetry.index)} non-overlap under reflection about both principal axes`,
      `${abcd.components.A} / 2`],
    ['B', 'Border', `${abcd.components.B} of 8 sectors cut off abruptly · compactness ${m.border.compactness.toFixed(2)} · convexity ${m.border.convexity.toFixed(2)}`,
      `${abcd.components.B} / 8`],
    ['C', 'Colour', (m.color.present.join(', ') || 'single colour') + ` · variegation ${m.color.variegation.toFixed(2)}`,
      `${abcd.components.C} / 6`],
    ['D', 'Structures', `${m.structures.clusters} distinct colour-texture regions · heterogeneity ${m.structures.heterogeneity.toFixed(2)} (approximation)`,
      `${abcd.components.D} / 5`],
  ];

  return `
  <div class="panel" style="${riskVars(risk)}">
    <div class="panel-head">
      <h3>Morphometry · ABCD rule of dermoscopy</h3>
      <span style="font:500 10.5px var(--mono);color:var(--ink-3)">Stolz 1994 · computed independently of the network</span>
    </div>
    <div class="panel-body">
      <div class="abcd">
        ${rows.map(([k, name, desc, val]) => `
          <div class="abcd-row">
            <div class="lbl"><b>${k}</b>${esc(name)}</div>
            <div class="desc">${esc(desc)}</div>
            <div class="val">${esc(val)}</div>
          </div>`).join('')}
      </div>

      <div class="tds-box" style="margin-top:15px">
        <div class="num">${abcd.tds}<small>TOTAL DERMOSCOPY SCORE</small></div>
        <div class="tds-scale"><i data-left="${markerLeft.toFixed(1)}%" style="left:0"></i></div>
        <div class="tds-ticks"><span>0 benign</span><span>4.75</span><span>5.45</span><span>8.9 melanoma</span></div>
        <p>${esc(abcd.verdict)}</p>
      </div>

      <div class="swatches">
        ${Object.entries(m.color.shares).slice(0, 6).map(([name, share]) => `
          <span class="swatch"><i style="background:${swatchColor(name)}"></i>${esc(name)} ${pct(share, 0)}</span>`).join('')}
      </div>

      <p class="prose" style="margin-top:14px;font-size:12px">
        Lesion occupies ${pct(m.geometry.area_fraction, 1)} of the frame, maximum diameter
        ${m.geometry.diameter_px.toFixed(0)} px (${pct(m.geometry.frame_fraction, 0)} of width).
        ${esc(m.geometry.note)} Hair covering ${pct(m.hair_fraction, 1)} of the field was inpainted before measurement.
      </p>
      <p class="prose" style="font-size:12px;margin-top:8px;color:var(--ink-3)">${esc(m.evolution_note)}</p>
    </div>
  </div>`;
}

function swatchColor(name) {
  return {
    white: '#f2f4f6', red: '#cc3333', 'light-brown': '#b8865c',
    'dark-brown': '#5e3c22', 'blue-grey': '#6e7d96', black: '#231e1e',
  }[name] || '#888';
}

function wireResult(d) {
  $$('#result .tab').forEach((btn) => btn.addEventListener('click', () => setImageTab(btn.dataset.tab)));
  $$('#result .subtab').forEach((btn) => btn.addEventListener('click', () => {
    $$('#result .subtab').forEach((b) => b.classList.toggle('on', b === btn));
    $$('#result .subpane').forEach((p) => p.classList.toggle('on', p.dataset.pane === btn.dataset.pane));
  }));
  const openPdf = () => window.open(`/api/case/${d.case_id}/report.pdf`, '_blank');
  $('#btnPdf')?.addEventListener('click', openPdf);
  $('#btnPdfTop')?.addEventListener('click', openPdf);

  const resetToUpload = () => {
    $('#result').classList.add('hidden');
    dropzone.classList.remove('hidden');
    fileInput.value = '';
  };
  $('#btnAnother')?.addEventListener('click', resetToUpload);
  $('#btnAnotherTop')?.addEventListener('click', resetToUpload);
}

function setImageTab(key) {
  state.imageTab = key;
  $$('#result .viewer img').forEach((i) => i.classList.toggle('on', i.dataset.img === key));
  $$('#result .tab').forEach((t) => t.classList.toggle('on', t.dataset.tab === key));
  const notes = {
    attention: 'Eigen-CAM over the P3/P4/P5 neck features. Warm regions drove the prediction.',
    original: 'The submitted image, unmodified.',
    outline: 'Lesion boundary from Otsu thresholding after DullRazor hair removal. All morphometry is computed inside this contour.',
    roi: 'The highest-scoring anchor box. Training labels covered the whole frame, so this is expected to be near-full-frame — it is shown for transparency, not localisation.',
  };
  const note = $('#viewerNote');
  if (note) note.textContent = notes[key] || '';
}

/* ─────────────────────────────── batch ──────────────────────────────── */

const batchInput = $('#batchInput');
const batchDrop = $('#batchDrop');

$('#btnBatchPick').addEventListener('click', () => batchInput.click());
batchDrop.addEventListener('click', () => batchInput.click());
batchInput.addEventListener('change', () => runBatch([...batchInput.files]));
['dragenter', 'dragover'].forEach((e) =>
  batchDrop.addEventListener(e, (ev) => { ev.preventDefault(); batchDrop.classList.add('over'); }));
['dragleave', 'drop'].forEach((e) =>
  batchDrop.addEventListener(e, (ev) => { ev.preventDefault(); batchDrop.classList.remove('over'); }));
batchDrop.addEventListener('drop', (ev) => runBatch([...ev.dataTransfer.files]));

async function runBatch(files) {
  files = files.filter((f) => f.type.startsWith('image/'));
  if (!files.length) return toast('No images in that selection.', true);
  if (files.length > 40) { files = files.slice(0, 40); toast('Limited to the first 40 images.'); }

  const box = $('#batchState');
  box.classList.remove('hidden');
  box.innerHTML = `Analysing ${files.length} image${files.length > 1 ? 's' : ''}…
    <div class="prog-bar" style="margin-top:10px"><span style="width:35%"></span></div>`;
  $('#batchResult').innerHTML = '';

  const fd = new FormData();
  files.forEach((f) => fd.append('files', f));
  try {
    const data = await api('/api/batch', { method: 'POST', body: fd });
    state.batchRows = data.results;
    box.innerHTML = `<b>${data.count}</b> analysed · ordered by clinical concern
      ${data.errors.length ? ` · <span style="color:var(--critical)">${data.errors.length} failed</span>` : ''}`;
    renderBatch();
    loadCases();
  } catch (err) {
    box.innerHTML = `<span style="color:var(--critical)">${esc(err.message)}</span>`;
  }
}

function renderBatch() {
  const rank = { URGENT: 0, PRIORITY: 1, REVIEW: 2, ROUTINE: 3 };
  const { key, dir } = state.batchSort;
  const rows = [...state.batchRows].sort((a, b) => {
    const va = key === 'band' ? rank[a.band] : a[key];
    const vb = key === 'band' ? rank[b.band] : b[key];
    return (va > vb ? 1 : va < vb ? -1 : 0) * dir;
  });

  $('#batchResult').innerHTML = `
    <div class="table-wrap"><table>
      <thead><tr>
        <th style="width:60px"></th>
        <th data-sort="filename">Image</th>
        <th data-sort="band">Triage</th>
        <th data-sort="top_class">Leading class</th>
        <th data-sort="confidence">Confidence</th>
        <th data-sort="concern">Concern</th>
        <th data-sort="tds">TDS</th>
        <th data-sort="quality">Quality</th>
      </tr></thead>
      <tbody>
        ${rows.map((r) => `
          <tr data-case="${r.case_id}" style="${riskVars(RISK_COLOR[r.risk])}">
            <td><img class="th-img" src="${r.thumb}" alt=""></td>
            <td><b>${esc(r.filename)}</b><br>
                <span style="font:500 10px var(--mono);color:var(--ink-3)">${esc(r.case_id)}</span></td>
            <td><span class="band">${esc(r.band)}</span></td>
            <td>${esc(r.top_name)} <span style="font-family:var(--mono);color:var(--ink-3);font-size:11px">${esc(r.top_class)}</span></td>
            <td>${pct(r.confidence, 1)}</td>
            <td><div class="mini-bar"><i style="width:${(r.concern * 100).toFixed(0)}%"></i></div>
                <span style="font:500 10.5px var(--mono);color:var(--ink-3)">${pct(r.concern, 0)}</span></td>
            <td>${r.tds ?? '—'}</td>
            <td>${r.quality.toFixed(0)}</td>
          </tr>`).join('')}
      </tbody>
    </table></div>`;

  $$('#batchResult th[data-sort]').forEach((th) => th.addEventListener('click', () => {
    const k = th.dataset.sort;
    state.batchSort = { key: k, dir: state.batchSort.key === k ? -state.batchSort.dir : 1 };
    renderBatch();
  }));
  $$('#batchResult tbody tr').forEach((tr) => tr.addEventListener('click', () => openCase(tr.dataset.case)));
}

/* ─────────────────────────────── cases ──────────────────────────────── */

async function loadCases() {
  try {
    const { cases } = await api('/api/cases');
    state.cases = cases;
    $('#caseCount').textContent = cases.length;
    $('#casesList').innerHTML = cases.length ? cases.map((c) => `
      <article class="case-card" data-case="${c.case_id}" style="${riskVars(RISK_COLOR[c.risk])}">
        <div class="cc-img">${c.thumb ? `<img src="${c.thumb}" alt="">` : ''}
          <span class="cc-band band">${esc(c.band)}</span></div>
        <div class="cc-body">
          <b>${esc(c.top_name)}</b>
          <div class="cc-meta">${esc(c.case_id)} · ${pct(c.confidence, 1)} confidence<br>
            ${esc(c.filename)}<br>${esc(c.timestamp.replace('T', ' '))}</div>
        </div>
      </article>`).join('') : `<div class="empty">No cases yet. Analyse an image to begin.</div>`;

    $$('#casesList .case-card').forEach((el) =>
      el.addEventListener('click', () => openCase(el.dataset.case)));
  } catch (err) { toast(err.message, true); }
}

async function openCase(id) {
  try {
    const data = await api(`/api/case/${id}`);
    showView('analyse');
    dropzone.classList.add('hidden');
    renderCase(data);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  } catch (err) { toast(err.message, true); }
}

$('#btnClearCases').addEventListener('click', async () => {
  await fetch('/api/cases', { method: 'DELETE' });
  toast('Session cases cleared.');
  loadCases();
});

$('#btnExportCsv').addEventListener('click', () => {
  if (!state.cases.length) return toast('Nothing to export.', true);
  const cols = ['case_id', 'timestamp', 'filename', 'top_class', 'top_name', 'confidence',
    'concern', 'band', 'risk', 'quality', 'tds'];
  const csv = [cols.join(','), ...state.cases.map((c) =>
    cols.map((k) => `"${String(c[k] ?? '').replace(/"/g, '""')}"`).join(','))].join('\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  const a = document.createElement('a');
  a.href = url; a.download = `dermora_cases_${Date.now()}.csv`; a.click();
  URL.revokeObjectURL(url);
});

/* ───────────────────────────── model card ───────────────────────────── */

let modelCardCache = null;
async function getModelCard() {
  if (!modelCardCache) modelCardCache = await api('/api/model-card');
  return modelCardCache;
}

let modelLoaded = false;

async function loadModelCard() {
  if (modelLoaded) return;
  try {
    const m = await getModelCard();
    modelLoaded = true;
    const met = m.metrics || {};
    const hp = m.hyperparameters || {};

    $('#modelCard').innerHTML = `
      <div class="tiles">
        <div class="tile"><b>${((met.mAP50 ?? 0) * 100).toFixed(1)}%</b><span>mAP@50</span>
          <small>Validation, whole-frame boxes</small></div>
        <div class="tile"><b>${((met.mAP50_95 ?? 0) * 100).toFixed(1)}%</b><span>mAP@50-95</span>
          <small>Strict IoU sweep</small></div>
        <div class="tile"><b>${((met.recall ?? 0) * 100).toFixed(1)}%</b><span>Recall</span>
          <small>Sensitivity — the number that matters in cancer triage</small></div>
        <div class="tile"><b>${((met.precision ?? 0) * 100).toFixed(1)}%</b><span>Precision</span>
          <small>False positives are the cost of that sensitivity</small></div>
        <div class="tile"><b>${((met.f1 ?? 0) * 100).toFixed(1)}%</b><span>F1</span>
          <small>Harmonic mean</small></div>
        <div class="tile"><b>${m.epochs_completed ?? '—'}</b><span>Epochs</span>
          <small>${esc(hp.model || 'yolov8n.pt')} · ${hp.imgsz ?? '—'}px · batch ${hp.batch ?? '—'} · ${esc(hp.device || 'cpu')}</small></div>
      </div>

      <div class="panel">
        <div class="panel-head"><h3>Training history</h3>
          <span style="font:500 10.5px var(--mono);color:var(--ink-3)">${esc(m.trained || '')}</span></div>
        ${trainingChart(m.history || {})}
      </div>

      <div class="panel">
        <div class="panel-head"><h3>Checkpoint &amp; hyper-parameters</h3></div>
        <div class="panel-body">
          <dl class="kv" style="grid-template-columns:auto 1fr">
            <dt>File</dt><dd>${esc(m.file)} · ${m.size_mb} MB</dd>
            <dt>Architecture</dt><dd>YOLOv8n detection head · ${m.nc ?? 7} classes</dd>
            <dt>Ultralytics</dt><dd>${esc(m.ultralytics_version || '—')}</dd>
            <dt>Trained</dt><dd>${esc(m.trained || '—')}</dd>
            <dt>Dataset</dt><dd>${esc(hp.data || '—')} · ISIC 2018 Task 3</dd>
            <dt>Optimiser</dt><dd>${esc(hp.optimizer || '—')} · lr0 ${hp.lr0 ?? '—'} · momentum ${hp.momentum ?? '—'}</dd>
            <dt>Weight decay</dt><dd>${hp.weight_decay ?? '—'}</dd>
            <dt>Augmentation</dt><dd>mosaic ${hp.mosaic ?? '—'} · fliplr ${hp.fliplr ?? '—'} · hsv ${hp.hsv_h ?? '—'}/${hp.hsv_s ?? '—'}/${hp.hsv_v ?? '—'} · erasing ${hp.erasing ?? '—'}</dd>
            <dt>Loss weights</dt><dd>box ${hp.box ?? '—'} · cls ${hp.cls ?? '—'} · dfl ${hp.dfl ?? '—'}</dd>
            <dt>Runtime</dt><dd>${m.runtime.tta_views}-view TTA · ${m.runtime.inference_size}px · ${esc(m.runtime.device)}</dd>
          </dl>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head"><h3>Taxonomy · ISIC 2018 Task 3</h3></div>
        <div class="panel-body"><div class="taxo">
          ${(m.taxonomy || []).map((t) => `
            <div class="taxo-item" style="${riskVars(t.color)}">
              <div class="taxo-head">
                <span class="code">${esc(t.code)}</span>
                <span class="nm"><b>${esc(t.full_name)}</b><span>${esc(t.category)} · ${esc(t.prevalence)}</span></span>
                <span class="band">${esc(t.risk)}</span>
              </div>
              <div class="taxo-body">
                <p class="prose">${esc(t.summary)}</p>
                <ul class="feature-list" style="margin-top:11px">${t.dermoscopy.map((f) => `<li>${esc(f)}</li>`).join('')}</ul>
                <p class="prose" style="margin-top:11px"><strong>Management.</strong> ${esc(t.management)}</p>
              </div>
            </div>`).join('')}
        </div></div>
      </div>

      <div class="panel">
        <div class="panel-head"><h3>Declared limitations</h3></div>
        <div class="panel-body"><div class="limits">
          ${(m.limitations || []).map((l, i) => `<div class="limit"><i>0${i + 1}</i><span>${esc(l)}</span></div>`).join('')}
        </div></div>
      </div>`;

    $$('#modelCard .taxo-head').forEach((h) =>
      h.addEventListener('click', () => h.parentElement.classList.toggle('open')));
  } catch (err) { toast(err.message, true); }
}

function trainingChart(hist) {
  const series = [
    { key: 'mAP50', color: '#14e0d0', label: 'mAP@50', axis: 'unit' },
    { key: 'mAP50_95', color: '#7c6cff', label: 'mAP@50-95', axis: 'unit' },
    { key: 'recall', color: '#3ddc97', label: 'Recall', axis: 'unit' },
    { key: 'precision', color: '#ffc53d', label: 'Precision', axis: 'unit' },
  ].filter((s) => Array.isArray(hist[s.key]) && hist[s.key].length > 1);

  if (!series.length) return `<div class="empty">No per-epoch history stored in this checkpoint.</div>`;

  const W = 900, H = 250, P = { t: 16, r: 16, b: 26, l: 34 };
  const n = Math.max(...series.map((s) => hist[s.key].length));
  const x = (i) => P.l + (i / (n - 1)) * (W - P.l - P.r);
  const y = (v) => P.t + (1 - v) * (H - P.t - P.b);

  const gridlines = [0, 0.25, 0.5, 0.75, 1].map((v) => `
    <line class="grid-line" x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}"/>
    <text class="axis" x="${P.l - 7}" y="${y(v) + 3}" text-anchor="end">${v.toFixed(2)}</text>`).join('');

  const ticks = [0, Math.floor(n / 2), n - 1].map((i) => `
    <text class="axis" x="${x(i)}" y="${H - 8}" text-anchor="middle">${i + 1}</text>`).join('');

  const paths = series.map((s) => {
    const d = hist[s.key].map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(Math.max(0, Math.min(1, v))).toFixed(1)}`).join('');
    return `<path class="series" d="${d}" stroke="${s.color}"/>`;
  }).join('');

  return `
    <svg class="chart" viewBox="0 0 ${W} ${H}">
      ${gridlines}${ticks}${paths}
    </svg>
    <div class="legend">
      ${series.map((s) => `<span><i style="background:${s.color}"></i>${s.label}
        <b style="font-family:var(--mono);color:var(--ink-3);font-weight:500"> ${hist[s.key].at(-1).toFixed(3)}</b></span>`).join('')}
      <span style="color:var(--ink-3);font-size:11px">epoch →</span>
    </div>`;
}

/* ───────────────────────── home & learn content ──────────────────────── */

let typeGridLoaded = false;
async function populateTypeGrid() {
  if (typeGridLoaded) return;
  const el = $('#typeGrid');
  if (!el) return;
  try {
    const m = await getModelCard();
    typeGridLoaded = true;
    el.innerHTML = (m.taxonomy || []).map((t) => `
      <article class="type-card" style="${riskVars(t.color)}" data-code="${esc(t.code)}">
        <div class="tc-top">
          <span class="tc-code">${esc(t.code)}</span>
          <span class="band">${esc(t.risk)}</span>
        </div>
        <h3>${esc(t.full_name)}</h3>
        <div class="tc-cat">${esc(t.category)} &middot; ${esc(t.prevalence)}</div>
        <p>${esc(t.summary)}</p>
        <div class="tc-more">
          <ul class="feature-list">${t.dermoscopy.slice(0, 3).map((f) => `<li>${esc(f)}</li>`).join('')}</ul>
          <p style="margin-top:10px;font-size:12.2px;color:var(--ink-2)"><strong style="color:var(--ink)">Management.</strong> ${esc(t.management)}</p>
        </div>
      </article>`).join('');
    $$('.type-card', el).forEach((card) =>
      card.addEventListener('click', () => card.classList.toggle('open')));
  } catch (err) { toast(err.message, true); }
}

function renderTaxoAccordion(taxonomy) {
  return (taxonomy || []).map((t) => `
    <div class="taxo-item" style="${riskVars(t.color)}">
      <div class="taxo-head">
        <span class="code">${esc(t.code)}</span>
        <span class="nm"><b>${esc(t.full_name)}</b><span>${esc(t.category)} · ${esc(t.prevalence)}</span></span>
        <span class="band">${esc(t.risk)}</span>
      </div>
      <div class="taxo-body">
        <p class="prose">${esc(t.summary)}</p>
        <ul class="feature-list" style="margin-top:11px">${t.dermoscopy.map((f) => `<li>${esc(f)}</li>`).join('')}</ul>
        <p class="prose" style="margin-top:11px"><strong>Management.</strong> ${esc(t.management)}</p>
        <p class="prose"><strong>Urgency.</strong> ${esc(t.urgency)}</p>
      </div>
    </div>`).join('');
}

let learnTaxoLoaded = false;
async function populateLearnTaxo() {
  if (learnTaxoLoaded) return;
  const el = $('#learnTaxo');
  if (!el) return;
  try {
    const m = await getModelCard();
    learnTaxoLoaded = true;
    el.innerHTML = renderTaxoAccordion(m.taxonomy);
    $$('.taxo-head', el).forEach((h) => h.addEventListener('click', () => h.parentElement.classList.toggle('open')));
  } catch (err) { toast(err.message, true); }
}

const FAQ_ITEMS = [
  ['Is Dermora AI a diagnosis?',
   'No. It is decision support, not a diagnostic device. Every result is a probability distribution, not a verdict, and the interface says so on every screen. Only a clinician who has examined the lesion in person — and where needed, a biopsy — can give you an actual diagnosis.'],
  ['How accurate is it?',
   'The underlying model reports 87% recall and 59% precision on its validation split, deliberately biased towards sensitivity: in cancer screening, a false alarm is far cheaper than a missed melanoma. Open the Technology page for the full metrics, training curve and stated limitations.'],
  ['Does my photo get uploaded anywhere?',
   'No. Analysis runs entirely on this machine — nothing is sent to a server, a cloud account or a third party. Case history lives in memory for this session only and disappears when the app is closed.'],
  ['What kind of photo works best?',
   'The model was trained on dermoscopic images — taken through a magnifying, polarised-light dermatoscope, the tool a dermatologist uses. An ordinary phone photo will still produce a result, but the built-in quality gate will usually flag it as less reliable.'],
  ['What do the seven classes mean?',
   'Melanoma and basal cell carcinoma are malignant; actinic keratosis is pre-malignant; melanocytic nevus, benign keratosis, dermatofibroma and vascular lesion are benign. See the "What it detects" section above, or the Learn page, for the full detail on each.'],
  ['Why does it disagree with itself sometimes?',
   'It doesn’t, quite — the network’s posterior and the classical ABCD morphometry are computed by two completely independent methods on purpose. When they disagree, the app flags it rather than silently picking one, because that disagreement is itself useful information for a clinician.'],
  ['What should I actually do with a concerning result?',
   'Book an appointment with a doctor or dermatologist and bring the downloaded PDF report. Do not wait for a second opinion from this tool, and do not let a reassuring result talk you out of a visit if the lesion is changing or worries you.'],
];

function populateFAQ() {
  const el = $('#faq');
  if (!el) return;
  el.innerHTML = FAQ_ITEMS.map(([q, a], i) => `
    <div class="faq-item${i === 0 ? ' open' : ''}">
      <button class="faq-q"><span>${esc(q)}</span><i></i></button>
      <div class="faq-a">${esc(a)}</div>
    </div>`).join('');
  $$('.faq-item', el).forEach((item) =>
    item.querySelector('.faq-q').addEventListener('click', () => item.classList.toggle('open')));
}

/* ────────────────────────────── shortcuts ───────────────────────────── */

document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, textarea')) return;
  const map = { 1: 'attention', 2: 'original', 3: 'outline', 4: 'roi' };
  if (map[e.key] && state.current) return setImageTab(map[e.key]);
  if (e.key === 'u' || e.key === 'U') { showView('analyse'); openFilePicker(); }
  if ((e.key === 'r' || e.key === 'R') && state.current) {
    window.open(`/api/case/${state.current.case_id}/report.pdf`, '_blank');
  }
  if (e.key === 'Escape') { $('#result').classList.add('hidden'); dropzone.classList.remove('hidden'); }
});

/* ──────────────────────────────── boot ──────────────────────────────── */

showView('home');
pollStatus();
loadSamples();
loadCases();
