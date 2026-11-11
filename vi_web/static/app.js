'use strict';
// Review analysis page. Every review string is inserted with textContent (el/svgEl never
// parse HTML), the page only calls same-origin paths, and nothing is stored in the browser.

const LABELS = ['negative', 'neutral', 'positive'];
const NAMES = { negative: 'Negative', neutral: 'Neutral', positive: 'Positive' };
const SVG_NS = 'http://www.w3.org/2000/svg';
const $ = (id) => document.getElementById(id);
const num = (n) => Number(n).toLocaleString('en-US');
const pct = (x) => `${(x * 100).toFixed(x >= 0.1 || x === 0 ? 0 : 1)}%`;
const plural = (n, w) => `${num(n)} ${w}${n === 1 ? '' : 's'}`;

// ---------- DOM helpers ----------
function el(tag, props, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) e.append(kid);
  return e;
}
function svgEl(tag, attrs, ...kids) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
  for (const kid of kids) if (kid != null) n.append(kid);
  return n;
}
function svgText(x, y, text, cls, anchor) {
  const t = svgEl('text', { x, y, class: cls, 'text-anchor': anchor || 'start' });
  t.textContent = text;
  return t;
}

// ---------- network ----------
async function api(method, path, body) {
  try {
    const res = await fetch(path, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    let data = null;
    try { data = await res.json(); } catch (e) { /* empty body */ }
    return { ok: res.ok, status: res.status, data };
  } catch (e) {
    return { ok: false, status: 0, data: { detail: 'Could not reach the server. Is it still running?' } };
  }
}
function errorText(r) {
  const d = r.data && r.data.detail;
  if (typeof d === 'string') return d;
  return r.status === 422 ? 'That request was not accepted.' : 'Something went wrong. Try again.';
}

// ---------- tooltip ----------
const tip = $('tip');
function showTip(target, title, rows, evt) {
  tip.replaceChildren(
    el('div', { class: 'tip-title', text: title }),
    ...rows.map((r) => el('div', { class: 'tip-row' },
      r.label ? el('span', { class: `key f-${r.label}` }) : null,
      el('strong', { text: r.value }),
      el('span', { class: 'tip-name', text: r.name }))));
  tip.hidden = false;
  const box = tip.getBoundingClientRect();
  let x; let y;
  if (evt) { x = evt.clientX + 14; y = evt.clientY + 14; }
  else { const b = target.getBoundingClientRect(); x = b.left + 8; y = b.bottom + 8; }
  x = Math.min(Math.max(8, x), window.innerWidth - box.width - 8);
  y = Math.min(Math.max(8, y), window.innerHeight - box.height - 8);
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
function hideTip() { tip.hidden = true; }
function bindTip(node, title, rows) {
  node.addEventListener('pointermove', (e) => showTip(node, title, rows, e));
  node.addEventListener('pointerleave', hideTip);
  node.addEventListener('focus', () => showTip(node, title, rows));
  node.addEventListener('blur', hideTip);
}

// ---------- small builders ----------
function legend(labels) {
  return el('ul', { class: 'legend' },
    ...labels.map((l) => el('li', null, el('span', { class: `sw f-${l}` }), NAMES[l])));
}
function dataTable(caption, headers, rows) {
  return el('details', { class: 'tbl' },
    el('summary', { text: 'Show as table' }),
    el('table', null,
      el('caption', { text: caption }),
      el('thead', null, el('tr', null, ...headers.map((h) => el('th', { scope: 'col', text: h })))),
      el('tbody', null, ...rows.map((r) => el('tr', null,
        ...r.map((c, i) => (i === 0 ? el('th', { scope: 'row', text: String(c) }) : el('td', { text: String(c) }))))))));
}
function niceTicks(max) {
  const raw = Math.max(max, 1) / 4;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = Math.max(1, [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw));
  const ticks = [];
  for (let t = 0; ; t += step) { ticks.push(t); if (t >= max) break; }
  return ticks;
}
// Bars grow from a square baseline; only the data end is rounded (4px).
function barPath(x, y, w, h, round) {
  const r = Math.min(4, w / 2, h / 2);
  if (round === 'right') return `M${x},${y}H${x + w - r}A${r},${r} 0 0 1 ${x + w},${y + r}V${y + h - r}A${r},${r} 0 0 1 ${x + w - r},${y + h}H${x}Z`;
  if (round === 'top') return `M${x},${y + h}V${y + r}A${r},${r} 0 0 1 ${x + r},${y}H${x + w - r}A${r},${r} 0 0 1 ${x + w},${y + r}V${y + h}Z`;
  return `M${x},${y}H${x + w}V${y + h}H${x}Z`;
}
const sum = (c) => LABELS.reduce((a, l) => a + c[l], 0);

// ---------- charts ----------
function renderShare(res) {
  const box = $('share-chart');
  box.replaceChildren();
  const c = res.sentiment_counts;
  const total = sum(c);
  if (!total) { box.append(el('p', { class: 'empty', text: 'No predicted reviews to chart.' })); return; }
  const bar = el('div', { class: 'segbar', role: 'group', 'aria-label': 'Share of predicted reviews by sentiment' });
  for (const l of LABELS) {
    if (!c[l]) continue;
    const share = c[l] / total;
    const seg = el('span', { class: `seg ${l}`, tabindex: '0', 'aria-label': `${NAMES[l]}: ${plural(c[l], 'review')}, ${pct(share)}` },
      el('span', { class: 'seglabel', text: pct(share) }));
    seg.style.flexGrow = String(c[l]);
    bindTip(seg, NAMES[l], [{ label: l, value: num(c[l]), name: `reviews (${pct(share)})` }]);
    bar.append(seg);
  }
  box.append(bar, legend(LABELS.filter((l) => c[l])),
    dataTable('Predicted reviews by sentiment', ['Sentiment', 'Reviews', 'Share'],
      LABELS.map((l) => [NAMES[l], num(c[l]), pct(c[l] / total)])));
  // A label only stays when it fits its block with padding on both sides.
  requestAnimationFrame(() => bar.querySelectorAll('.seg').forEach((seg) => {
    const lab = seg.firstChild;
    if (lab.offsetWidth + 16 > seg.clientWidth) lab.hidden = true;
  }));
}

function renderStars(res) {
  const box = $('stars-chart');
  box.replaceChildren();
  const rows = [5, 4, 3, 2, 1].map((st) => ({ st, c: res.stars_by_label[String(st)] }));
  const totals = rows.map((r) => sum(r.c));
  const max = Math.max(0, ...totals);
  if (!max) { box.append(el('p', { class: 'empty', text: 'No predicted reviews to chart.' })); return; }
  const W = Math.max(box.clientWidth, 280);
  const left = 62; const right = W < 520 ? 118 : 170;
  const plotW = Math.max(W - left - right, 80);
  const ticks = niceTicks(max);
  const scale = plotW / ticks[ticks.length - 1];
  const rowH = 34; const barH = 20; const top = 6;
  const H = top + rows.length * rowH + 26;
  const svg = svgEl('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'group', 'aria-label': 'Reviews by star rating and predicted sentiment' });
  for (const t of ticks) {
    const x = left + t * scale;
    svg.append(svgEl('line', { x1: x, x2: x, y1: top, y2: top + rows.length * rowH, class: 'grid' }),
      svgText(x, top + rows.length * rowH + 16, num(t), 'axis', 'middle'));
  }
  const disagreeAt = (st, c) => (st >= 4 ? c.negative : st <= 2 ? c.positive : 0);
  rows.forEach((r, i) => {
    const y = top + i * rowH; const by = y + (rowH - barH) / 2;
    const sl = r.st === 1 ? '1 star' : `${r.st} stars`;
    svg.append(svgText(left - 10, by + barH / 2 + 4, sl, 'axis', 'end'));
    const drawn = LABELS.filter((l) => r.c[l] > 0);
    let x = left;
    drawn.forEach((l, k) => {
      const w = r.c[l] * scale; const last = k === drawn.length - 1;
      svg.append(svgEl('path', { d: barPath(x, by, Math.max(w - (last ? 0 : 2), 1), barH, last ? 'right' : null), class: `f-${l}` }));
      x += w;
    });
    const dis = disagreeAt(r.st, r.c);
    svg.append(svgText(x + 8, by + barH / 2 + 4, num(totals[i]) + (dis ? ` · ${num(dis)} disagree` : ''), 'value', 'start'));
    const hit = svgEl('rect', {
      x: 0, y, width: W, height: rowH, class: 'hit', tabindex: '0',
      'aria-label': `${sl}: ${LABELS.map((l) => `${num(r.c[l])} ${NAMES[l]}`).join(', ')}${dis ? `; ${num(dis)} disagree with the stars` : ''}`,
    });
    bindTip(hit, sl, [
      ...LABELS.map((l) => ({ label: l, value: num(r.c[l]), name: NAMES[l] })),
      ...(dis ? [{ value: num(dis), name: 'disagree with the star rating' }] : [])]);
    svg.append(hit);
  });
  box.append(svg, legend(LABELS),
    dataTable('Reviews by star rating and predicted sentiment', ['Stars', 'Negative', 'Neutral', 'Positive', 'Disagree'],
      rows.map((r) => [r.st, num(r.c.negative), num(r.c.neutral), num(r.c.positive), num(disagreeAt(r.st, r.c))])));
}

function fillMonths(list) {
  if (!list.length) return [];
  const by = new Map(list.map((m) => [m.month, m]));
  const [y0, m0] = list[0].month.split('-').map(Number);
  const [y1, m1] = list[list.length - 1].month.split('-').map(Number);
  const out = [];
  let y = y0; let m = m0;
  while ((y < y1 || (y === y1 && m <= m1)) && out.length < 600) {
    const key = `${y}-${String(m).padStart(2, '0')}`;
    out.push(by.get(key) || { month: key, negative: 0, neutral: 0, positive: 0 });
    m += 1;
    if (m > 12) { m = 1; y += 1; }
  }
  return out;
}

function renderMonths(res) {
  const box = $('months-chart');
  box.replaceChildren();
  const data = fillMonths(res.by_month);
  if (!data.length) { box.append(el('p', { class: 'empty', text: 'No dated reviews to chart.' })); return; }
  const totals = data.map(sum);
  const ticks = niceTicks(Math.max(...totals));
  const W = Math.max(box.clientWidth, 280);
  const left = 46; const right = 8; const top = 8; const plotH = 170; const H = top + plotH + 28;
  const baseY = top + plotH; const scale = plotH / ticks[ticks.length - 1];
  const band = (W - left - right) / data.length; const bw = Math.min(24, band * 0.7);
  const step = Math.max(1, Math.ceil(64 / band));
  const svg = svgEl('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'group', 'aria-label': 'Reviews per month by predicted sentiment' });
  for (const t of ticks) {
    const y = baseY - t * scale;
    svg.append(svgEl('line', { x1: left, x2: W - right, y1: y, y2: y, class: 'grid' }), svgText(left - 6, y + 4, num(t), 'axis', 'end'));
  }
  data.forEach((d, i) => {
    const x = left + i * band + (band - bw) / 2;
    const drawn = LABELS.filter((l) => d[l] > 0);
    let cum = 0;
    drawn.forEach((l, k) => {
      const h = d[l] * scale; const topY = baseY - (cum + d[l]) * scale;
      svg.append(svgEl('path', { d: barPath(x, topY, bw, Math.max(h - (k === 0 ? 0 : 2), 1), k === drawn.length - 1 ? 'top' : null), class: `f-${l}` }));
      cum += d[l];
    });
    if (i % step === 0) svg.append(svgText(left + i * band + band / 2, baseY + 16, d.month, 'axis', 'middle'));
    const hit = svgEl('rect', {
      x: left + i * band, y: top, width: band, height: plotH, class: 'hit', tabindex: '0',
      'aria-label': `${d.month}: ${LABELS.map((l) => `${num(d[l])} ${NAMES[l]}`).join(', ')}`,
    });
    bindTip(hit, d.month, [...LABELS.map((l) => ({ label: l, value: num(d[l]), name: NAMES[l] })), { value: num(totals[i]), name: 'in total' }]);
    svg.append(hit);
  });
  box.append(svg, legend(LABELS),
    dataTable('Predicted reviews per month', ['Month', 'Negative', 'Neutral', 'Positive'],
      data.map((d) => [d.month, num(d.negative), num(d.neutral), num(d.positive)])));
}

// ---------- summary and notices ----------
function renderSummary(res) {
  const s = res.summary;
  const tiles = [
    ['Reviews found', s.reviews_found], ['With text', s.with_text], ['Stars only, cannot be predicted', s.stars_only],
    ['Not predicted', s.not_predicted], ['Duplicates', s.duplicates], ['Repeated words', s.repetitive],
    ['Disagree with stars', s.disagreements],
  ];
  $('summary').replaceChildren(...tiles.map(([l, v]) => el('div', { class: 'tile' }, el('div', { class: 'v', text: num(v) }), el('div', { class: 'l', text: l }))));
  const notes = [];
  if (s.training_data_warning) notes.push('This product is one of the products the model was trained on, so these predictions are not a fair test. Open it from the saved list to see held-out predictions, or try a product the model has not seen.');
  if (s.mode === 'saved') {
    notes.push('Held-out predictions: the model did not see this product when it made them.');
    if (s.not_predicted) notes.push(`${plural(s.not_predicted, 'review')} repeat a review from another product, have no held-out prediction, and show as not predicted.`);
  }
  if (s.date_range) notes.push(`Reviews from ${s.date_range[0]} to ${s.date_range[1]}.`);
  const box = $('notice');
  box.replaceChildren(...notes.map((n) => el('p', { text: n })));
  box.hidden = !notes.length;
}

// ---------- review browser ----------
const state = { result: null, label: '', stars: '', flagged: false, disagree: false, q: '', page: 0, size: 25, filtered: [] };

function chip(label) {
  return el('span', { class: 'chip' }, label ? el('span', { class: `dot f-${label}` }) : null, label ? NAMES[label] : 'Not predicted');
}
function flagsOf(r) {
  const f = [];
  if (r.duplicate) f.push('Duplicate');
  if (r.repetitive) f.push('Repeated words');
  if (r.disagrees) f.push('Disagrees with stars');
  return f;
}
function applyFilters() {
  const s = state; const q = s.q.trim().toLowerCase();
  s.filtered = s.result.reviews.filter((r) => {
    if (s.label === 'none') { if (r.label !== null) return false; } else if (s.label && r.label !== s.label) return false;
    if (s.stars && r.stars !== Number(s.stars)) return false;
    if (s.flagged && !(r.duplicate || r.repetitive)) return false;
    if (s.disagree && !r.disagrees) return false;
    if (q && !(r.text.toLowerCase().includes(q) || r.cleaned.toLowerCase().includes(q))) return false;
    return true;
  });
  s.page = 0;
  renderTable();
}
function renderTable() {
  const s = state; const n = s.filtered.length;
  const pages = Math.max(1, Math.ceil(n / s.size));
  s.page = Math.min(s.page, pages - 1);
  const slice = s.filtered.slice(s.page * s.size, (s.page + 1) * s.size);
  $('count').textContent = n ? `Showing ${num(s.page * s.size + 1)} to ${num(s.page * s.size + slice.length)} of ${plural(n, 'review')}` : 'No reviews match these filters.';
  $('rows').replaceChildren(...slice.map((r) => {
    const snippet = r.text.replace(/\s+/g, ' ');
    const btn = el('button', { type: 'button', class: 'rowbtn', 'aria-label': `Open review ${r.n}: ${snippet.slice(0, 60)}`, onclick: () => openDetail(r) },
      snippet.length > 180 ? `${snippet.slice(0, 180)}…` : snippet);
    return el('tr', null,
      el('td', { class: 'stars', text: String(r.stars) }),
      el('td', { class: 'pred' }, chip(r.label)),
      el('td', { class: 'text' }, btn),
      el('td', { class: 'flags' }, ...flagsOf(r).map((f) => el('span', { class: 'flag', text: f }))));
  }));
  $('pageinfo').textContent = `Page ${num(s.page + 1)} of ${num(pages)}`;
  $('prev').disabled = s.page === 0;
  $('next').disabled = s.page >= pages - 1;
}
function openDetail(r) {
  $('d-title').textContent = `Review ${r.n}`;
  const kv = el('dl', { class: 'kv' },
    el('dt', { text: 'Stars' }), el('dd', { text: `${r.stars} of 5` }),
    el('dt', { text: 'Predicted' }), el('dd', null, chip(r.label)),
    ...(r.date ? [el('dt', { text: 'Date' }), el('dd', { text: r.date })] : []),
    ...(r.variation ? [el('dt', { text: 'Variation' }), el('dd', { text: r.variation })] : []),
    ...(flagsOf(r).length ? [el('dt', { text: 'Flags' }), el('dd', { text: flagsOf(r).join(', ') })] : []));
  const parts = [kv];
  if (r.probs) {
    parts.push(el('div', { class: 'label-small', text: 'Probabilities' }),
      el('div', { class: 'probs' }, ...LABELS.map((l) => {
        const p = r.probs[l] || 0;
        const fill = el('span', { class: `fill ${l}` });
        fill.style.width = `${(p * 100).toFixed(1)}%`;
        return el('div', { class: 'prob' }, el('span', { text: NAMES[l] }), el('span', { class: 'track' }, fill), el('span', { class: 'pv', text: pct(p) }));
      })));
  }
  parts.push(el('div', { class: 'label-small', text: 'Review' }), el('div', { class: 'text-box', text: r.text }),
    el('div', { class: 'label-small', text: 'Cleaned text (what the pipeline changed)' }), el('div', { class: 'text-box', text: r.cleaned || '(empty after cleaning)' }));
  $('d-body').replaceChildren(...parts);
  $('detail').showModal();
}

// ---------- results ----------
function renderCharts(res) { renderShare(res); renderStars(res); renderMonths(res); }
function render(res) {
  state.result = res;
  $('main').hidden = false;
  renderSummary(res);
  renderCharts(res);
  for (const id of ['f-label', 'f-stars']) $(id).value = '';
  $('f-q').value = ''; $('f-flagged').checked = false; $('f-disagree').checked = false;
  Object.assign(state, { label: '', stars: '', flagged: false, disagree: false, q: '' });
  applyFilters();
  $('results-title').focus();
}
function clearResults() { state.result = null; $('main').hidden = true; hideTip(); }

// ---------- jobs ----------
let busy = false; let timer = null;
function setBusy(b) { busy = b; $('go').disabled = b; $('saved').disabled = b; }
function showStatus(msg) { $('status').textContent = msg; }
function showError(msg) { const e = $('error'); e.textContent = msg; e.hidden = !msg; }
function finish() { clearTimeout(timer); setBusy(false); $('ready').hidden = true; }

async function poll(id) {
  const tick = async () => {
    const r = await api('GET', `/analyze/jobs/${id}`);
    if (!r.ok) { finish(); showStatus(''); showError(errorText(r)); return; }
    const j = r.data;
    $('ready').hidden = j.state !== 'waiting_login';
    if (j.state === 'waiting_login') showStatus('A Chrome window opened with Shopee. If it asks you to log in, do that there, then press "I\'m ready".');
    else if (j.state === 'crawling') showStatus(`Collecting reviews: ${num(j.collected)} so far…`);
    else if (j.state === 'analyzing') showStatus('Analyzing reviews…');
    if (j.state === 'error') { finish(); showStatus(''); showError((j.error && j.error.message) || 'The analysis failed.'); return; }
    if (j.state === 'done') {
      const res = await api('GET', `/analyze/jobs/${id}/result`);
      finish(); showStatus('');
      if (res.ok) render(res.data); else showError(errorText(res));
      return;
    }
    timer = setTimeout(tick, 1000);
  };
  tick();
}

let currentJob = null;
$('link-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (busy) return;
  const url = $('url').value.trim();
  showError('');
  if (!url) { showError('Paste a Shopee product link first.'); return; }
  clearResults(); setBusy(true); showStatus('Starting…');
  const r = await api('POST', '/analyze/link', { url });
  if (!r.ok) { finish(); showStatus(''); showError(errorText(r)); return; }
  currentJob = r.data.job_id;
  poll(currentJob);
});
$('ready').addEventListener('click', async () => {
  $('ready').hidden = true; showStatus('Starting to collect reviews…');
  const r = await api('POST', `/analyze/jobs/${currentJob}/ready`);
  if (!r.ok) showError(errorText(r));
});
$('saved').addEventListener('change', async () => {
  const file = $('saved').value;
  if (!file || busy) return;
  showError(''); clearResults(); setBusy(true); showStatus('Analyzing the saved product…');
  const r = await api('POST', '/analyze/saved', { file });
  finish(); showStatus('');
  if (r.ok) render(r.data); else showError(errorText(r));
});

// ---------- wiring ----------
$('f-label').addEventListener('change', (e) => { state.label = e.target.value; applyFilters(); });
$('f-stars').addEventListener('change', (e) => { state.stars = e.target.value; applyFilters(); });
$('f-flagged').addEventListener('change', (e) => { state.flagged = e.target.checked; applyFilters(); });
$('f-disagree').addEventListener('change', (e) => { state.disagree = e.target.checked; applyFilters(); });
let searchTimer = null;
$('f-q').addEventListener('input', (e) => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.q = e.target.value; applyFilters(); }, 200); });
$('prev').addEventListener('click', () => { state.page -= 1; renderTable(); });
$('next').addEventListener('click', () => { state.page += 1; renderTable(); });
$('d-close').addEventListener('click', () => $('detail').close());

let lastWidth = 0; let resizeTimer = null;
new ResizeObserver(() => {
  const w = $('main').clientWidth;
  if (w === lastWidth || !state.result) { lastWidth = w; return; }
  lastWidth = w;
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => renderCharts(state.result), 150);
}).observe($('main'));

(async () => {
  const r = await api('GET', '/analyze/saved');
  if (!r.ok) return;
  for (const p of r.data) $('saved').append(el('option', { value: p.file, text: `${p.name.replace(/_/g, ' ')} (${num(p.reviews)} reviews)` }));
})();
