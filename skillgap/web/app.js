/* Keep documents private: no remote assets or browser persistence. All model
   and user text uses textContent, preventing HTML injection and remote images. */
const $ = id => document.getElementById(id);
let selectedFile = null, report = null, busy = false;
function el(tag, text, cls) { const n = document.createElement(tag); if (text != null) n.textContent = text; if (cls) n.className = cls; return n; }
function jobsCount() { return $('jobs').value.split(/^\s*---\s*$/m).filter(s => s.trim()).length; }
function update() { $('job-count').textContent = `${jobsCount()} ROLE${jobsCount() === 1 ? '' : 'S'}`; $('analyze').disabled = busy || !selectedFile || !jobsCount(); }
function invalidate() { report = null; $('results').hidden = true; $('empty-state').hidden = false; $('error').hidden = true; update(); }
function choose(file) { selectedFile = file; $('file-label').textContent = file ? file.name : 'Drop your resume here'; $('file-detail').textContent = file ? `${(file.size / 1024).toFixed(1)} KB · ready to analyze` : 'or choose a file · up to 5 MB'; invalidate(); }
function fail(message) { $('error').textContent = message; $('error').hidden = false; }
$('resume').addEventListener('change', e => choose(e.target.files[0]));
$('jobs').addEventListener('input', invalidate);
['dragover','dragleave','drop'].forEach(event => $('dropzone').addEventListener(event, e => { e.preventDefault(); $('dropzone').classList.toggle('dragover', event === 'dragover'); if (event === 'drop' && !busy) choose(e.dataTransfer.files[0]); }));
$('settings-toggle').onclick = () => { $('model-settings').hidden = !$('model-settings').hidden; };
$('provider').onchange = () => { const external = $('provider').value === 'api'; $('api-settings').hidden = !external; $('privacy-label').textContent = external ? 'EXTERNAL API MODE' : '● LOCAL MODE'; $('api-consent').checked = false; invalidate(); };
['model','api-model','api-key','api-consent'].forEach(id => $(id).addEventListener('input', invalidate));
$('sample').onclick = async () => { try { const response = await fetch('/api/sample'); if (!response.ok) throw Error('Sample could not load.'); const data = await response.json(); $('jobs').value = data.jobs; choose(new File([data.resume], 'sample-resume.txt', {type:'text/plain'})); } catch (error) { fail(error.message); } };
fetch('/api/config').then(r => { if (!r.ok) throw Error(); return r.json(); }).then(c => { $('model').value = c.model; }).catch(() => fail('Local server configuration could not load.'));
const inputs = ['resume','jobs','sample','provider','model','api-model','api-key','api-consent','report-file'];
// Report import is a local file read, not a new model call. Clear input fields
// so a saved result is never presented as an analysis of the current inputs.
$('report-file').onchange = async e => {
  try {
    const file = e.target.files[0]; if (!file) return;
    if (file.size > 5000000) throw Error('Report exceeds the 5 MB limit.');
    const data = JSON.parse(await file.text());
    if (!Number.isInteger(data.job_count) || data.job_count < 1 || !Array.isArray(data.warnings) || !Array.isArray(data.ranked_gaps) || !Array.isArray(data.recommendations)) throw Error('Invalid SkillGap report structure.');
    for (const category of ['covered','partial','missing']) {
      if (!Array.isArray(data[category])) throw Error('Invalid evidence columns.');
      for (const item of data[category]) if (typeof item.text !== 'string' || !Number.isFinite(item.score) || !Array.isArray(item.candidates) || item.candidates.some(c => !Number.isFinite(c.score) || typeof c.line !== 'string')) throw Error('Invalid requirement evidence.');
    }
    if (data.recommendations.some(r => typeof r.kind !== 'string' || !Array.isArray(r.requirement_ids))) throw Error('Invalid recommendations.');
    $('jobs').value = ''; $('resume').value = ''; choose(null); report = data; render();
    $('status').textContent = `Saved report: ${file.name}. Edit inputs to start a new analysis.`;
  } catch (error) { report = null; $('results').hidden = true; fail(error.message || 'Could not open this report.'); }
  finally { e.target.value = ''; }
};
$('analyze').onclick = async () => {
  if ($('provider').value === 'api' && !$('api-consent').checked) { fail('Confirm external processing in Model settings before analyzing.'); return; }
  busy = true; invalidate(); $('empty-state').hidden = true;
  // Freeze the input snapshot until inference ends; avoid stale report labels.
  inputs.forEach(id => $(id).disabled = true);
  $('status').textContent = 'Comparing evidence and planning next steps…'; $('analyze').textContent = 'Analyzing…';
  const body = new FormData(); body.append('resume', selectedFile); body.append('jobs', $('jobs').value);
  body.append('provider', $('provider').value); body.append('model', $('model').value);
  body.append('api_model', $('api-model').value); body.append('api_key', $('api-key').value); body.append('api_consent', $('api-consent').checked);
  try { const response = await fetch('/api/analyze', {method:'POST', body}); const data = await response.json(); if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'Check the upload and model settings.'); report = data; render(); $('status').textContent = data.incomplete ? 'Analysis needs review. See the details below.' : 'Analysis ready. Follow the evidence.'; }
  catch (error) { fail(error.message || 'Could not reach the local server.'); $('empty-state').hidden = false; $('status').textContent = 'Analysis stopped. Review the message below.'; }
  finally { busy = false; inputs.forEach(id => $(id).disabled = false); $('analyze').textContent = 'Find my skill gaps ↗'; update(); }
};
function evidenceCard(item) {
  const card = el('article', null, 'evidence-card'); card.append(el('div', `ROLE ${item.job_index + 1} · ${item.importance} · SIM ${item.score.toFixed(3)}`, 'card-meta'), el('h3', item.text), el('p', item.explanation));
  card.append(item.citation ? el('div', item.citation, 'quote') : el('p', 'No supporting resume citation.'));
  const detail = el('details'); detail.append(el('summary', 'Inspect source & candidates'), el('div', `Job requirement: ${item.source_quote}`, 'quote'));
  for (const c of item.candidates) detail.append(el('div', `${c.score.toFixed(3)} · ${c.line}`, 'candidate'));
  card.append(detail); return card;
}
function renderEvidence() {
  $('evidence').replaceChildren(); const filter = $('job-filter').value;
  for (const category of ['covered','partial','missing']) { const items = report[category].filter(i => filter === 'all' || String(i.job_index) === filter); const col = el('div', null, `evidence-column ${category}`), heading = el('div', null, 'column-heading'); heading.append(el('span', `${category === 'covered' ? '●' : category === 'partial' ? '◐' : '○'}  ${category[0].toUpperCase() + category.slice(1)}`), el('span', items.length, 'count')); col.append(heading); items.forEach(i => col.append(evidenceCard(i))); if (!items.length) col.append(el('p', 'No requirements in this category.', 'report-note')); $('evidence').append(col); }
}
function render() {
  $('results').hidden = false; $('empty-state').hidden = true; $('warnings').replaceChildren();
  if (report.incomplete) $('warnings').append(el('div', 'Some analysis steps could not be completed. Review the details below; the available results are provisional.', 'warning'));
  report.warnings.forEach(w => $('warnings').append(el('div', w, 'warning')));
  $('job-filter').replaceChildren(new Option('All roles','all')); for (let i = 0; i < report.job_count; i++) $('job-filter').add(new Option(`Role ${i + 1}`, String(i))); renderEvidence();
  $('gap-list').replaceChildren(); report.ranked_gaps.forEach((gap, i) => { const row = el('div', null, 'gap-row'), content = el('div'); content.append(el('h3', gap.text), el('p', `${gap.frequency} role${gap.frequency === 1 ? '' : 's'} · ${gap.importance}`)); row.append(el('span', String(i + 1).padStart(2,'0'), 'gap-number'), content); $('gap-list').append(row); });
  if (!report.ranked_gaps.length) $('gap-list').append(el('p', 'No gaps ranked. Check extraction warnings before interpreting this result.', 'report-note'));
  const requirements = Object.fromEntries(['covered','partial','missing'].flatMap(c => report[c]).map(i => [i.id,i]));
  $('recommendations').replaceChildren(); for (const rec of report.recommendations) { const card = el('article', null, 'recommendation'); card.append(el('div', `${rec.kind.toUpperCase()} / ABOUT ${rec.estimated_hours} HOURS`, 'meta'), el('h3', rec.title), el('p', rec.what_to_do), el('p', `Why it helps: ${rec.why_it_helps}`), el('p', `Resource: ${rec.resource_type}`), el('div', 'AFTER COMPLETION, ON YOUR RESUME', 'label'), el('div', rec.resume_example, 'quote')); const detail = el('details'); detail.append(el('summary', 'Evidence behind this next step')); for (const id of rec.requirement_ids) { const item = requirements[id]; if (item) detail.append(el('p', `Role ${item.job_index + 1} · ${item.category}: ${item.text}`), el('div', item.citation || 'No supporting resume line found.', 'quote')); } card.append(detail); $('recommendations').append(card); }
}
$('job-filter').onchange = renderEvidence;
$('download').onclick = () => { if (!report) return; const url = URL.createObjectURL(new Blob([JSON.stringify(report,null,2)], {type:'application/json'})); const link = el('a'); link.href = url; link.download = 'skillgap-report.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); };
