// ── State ──
let isStreaming = false;
let papers = [];        // current search results (Papers tab)
let selectedPool = [];  // papers added to selection pool (Papers tab)
let currentDetail = null;

// Tech Learning state
let authorPapers = [];       // author search results
let techAnalysisPool = [];   // selected for analysis
let lastAnalysisText = '';   // for download

// Paper analysis & QA state
let paperQAHistory = [];

// Relations state
let currentGraphData = null;   // { nodes, edges, stats }
let currentNetworkType = 'coauthor';

// ── DOM helpers ──
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
// (messagesDiv and input removed — Chat tab replaced by Note Workshop)

// ── Tabs ──
$$('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    $$('.tab').forEach(t => t.classList.remove('active'));
    $$('.tab-content').forEach(tc => tc.classList.remove('active'));
    tab.classList.add('active');
    $(`#tab-${tab.dataset.tab}`).classList.add('active');
  });
});

// ── Discover Sub-tabs ──
$$('.discover-subtab').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.discover-subtab').forEach(b => b.classList.remove('active'));
    $$('.discover-subcontent').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    $(`#subtab-${btn.dataset.subtab}`).classList.add('active');
  });
});

// ── Sliders ──
$('#max-results').addEventListener('input', e => {
  $('#max-val').textContent = e.target.value;
});
$('#tl-max-results').addEventListener('input', e => {
  $('#tl-max-val').textContent = e.target.value;
});

// ── Selection mode switching ──
$$('input[name="sel-mode"]').forEach(radio => {
  radio.addEventListener('change', () => {
    const mode = document.querySelector('input[name="sel-mode"]:checked').value;
    $('#sel-prompt').style.display = mode === 'prompt' ? 'block' : 'none';
    $('#btn-auto-select').style.display = (mode === 'auto' || mode === 'prompt') ? 'block' : 'none';
  });
});

// ── Markdown rendering with LaTeX ──
function renderMarkdown(text) {
  const blocks = [];
  text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
    blocks.push({ type: 'block', math });
    return `%%MATHBLOCK${blocks.length - 1}%%`;
  });
  text = text.replace(/\$([^\$\n]+?)\$/g, (_, math) => {
    blocks.push({ type: 'inline', math });
    return `%%MATHBLOCK${blocks.length - 1}%%`;
  });
  let html = marked.parse(text);
  blocks.forEach((b, i) => {
    try {
      const rendered = katex.renderToString(b.math.trim(), {
        displayMode: b.type === 'block',
        throwOnError: false,
      });
      html = html.replace(`%%MATHBLOCK${i}%%`, rendered);
    } catch {
      html = html.replace(`%%MATHBLOCK${i}%%`, `<code>${b.math}</code>`);
    }
  });
  return html;
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

// Helper: read SSE stream
async function readSSE(resp, onText, onDone) {
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let fullText = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);
      if (data === '[DONE]') continue;
      let evt;
      try { evt = JSON.parse(data); } catch { continue; }
      if (evt.type === 'text') {
        fullText = evt.content;
        onText(fullText);
      } else if (evt.type === 'done') {
        if (evt.content) fullText = evt.content;
        if (onDone) onDone(fullText);
      }
    }
  }
  return fullText;
}


// ════════════════════════════════════════
// Papers Tab
// ════════════════════════════════════════

async function searchPapers() {
  const query = $('#search-query').value.trim();
  if (!query) return;

  const maxResults = parseInt($('#max-results').value);
  const sortBy = $('#sort-by').value;

  $('#paper-list').innerHTML = '<div class="paper-list-loading">Searching arXiv...</div>';
  $('#papers-status').textContent = '';
  $('#btn-search').disabled = true;

  try {
    const resp = await fetch('/api/papers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, max_results: maxResults, sort_by: sortBy }),
    });
    const data = await resp.json();
    papers = data.papers || [];
    renderPaperList();
    $('#papers-status').textContent = `${papers.length} results`;
  } catch (err) {
    $('#paper-list').innerHTML = `<div class="paper-list-empty"><p>Error: ${err.message}</p></div>`;
  }
  $('#btn-search').disabled = false;
}

$('#search-query').addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); searchPapers(); }
});

function renderPaperList() {
  if (!papers.length) {
    $('#paper-list').innerHTML = '<div class="paper-list-empty"><p>No papers found</p></div>';
    return;
  }
  const html = papers.map((p, i) => {
    const isSelected = selectedPool.some(s => s.arxiv_id === p.arxiv_id);
    const authors = (p.authors || []).slice(0, 3).join(', ') + (p.authors?.length > 3 ? ' et al.' : '');
    const date = p.published ? p.published.split('T')[0] : '';
    const cats = (p.categories || []).join(', ');
    return `
      <div class="paper-card ${isSelected ? 'selected' : ''}" data-index="${i}">
        <input type="checkbox" ${isSelected ? 'checked' : ''} onclick="togglePaper(event, ${i})">
        <div class="paper-card-body" onclick="openDetail(${i})">
          <div class="paper-card-title">${escapeHtml(p.title)}</div>
          <div class="paper-card-meta">
            <span class="authors">${escapeHtml(authors)}</span>
            <span>${date}</span>
            <span class="categories">${escapeHtml(cats)}</span>
          </div>
        </div>
      </div>`;
  }).join('');
  $('#paper-list').innerHTML = html;
}

function togglePaper(event, index) {
  event.stopPropagation();
  const paper = papers[index];
  const idx = selectedPool.findIndex(s => s.arxiv_id === paper.arxiv_id);
  if (idx >= 0) selectedPool.splice(idx, 1);
  else selectedPool.push(paper);
  renderPaperList();
  renderPool();
  updateGenerateBtn();
}

function openDetail(index) {
  const p = papers[index];
  currentDetail = p;
  $('#detail-title').textContent = p.title;
  $('#detail-date').textContent = p.published ? p.published.split('T')[0] : '';
  $('#detail-categories').textContent = (p.categories || []).join(', ');
  $('#detail-authors').textContent = (p.authors || []).join(', ');
  $('#detail-abstract').textContent = p.abstract || '';
  $('#detail-pdf').href = p.pdf_url || '#';
  $('#detail-url').href = p.url || '#';
  const inPool = selectedPool.some(s => s.arxiv_id === p.arxiv_id);
  $('#btn-add-pool').textContent = inPool ? '- Remove from Pool' : '+ Add to Pool';
  // Reset analysis & QA state
  paperQAHistory = [];
  $('#qa-messages').innerHTML = '';
  $('#paper-analysis-output').style.display = 'none';
  $('#paper-analysis-content').innerHTML = '';
  $('#btn-analyze-paper').disabled = false;
  $('#btn-analyze-paper').textContent = 'Analyze Paper';

  $('#paper-detail').style.display = 'block';
  $('#paper-detail').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function closeDetail() {
  $('#paper-detail').style.display = 'none';
  currentDetail = null;
}

function renderPool() {
  const container = $('#selection-pool');
  $('#pool-count').textContent = selectedPool.length;
  if (!selectedPool.length) {
    container.innerHTML = '<p class="pool-empty">Select papers from the list</p>';
    return;
  }
  container.innerHTML = selectedPool.map((p, i) => `
    <div class="pool-item">
      <span class="pool-item-title" title="${escapeHtml(p.title)}">${escapeHtml(p.title)}</span>
      <button onclick="removeFromPool(${i})">&#10005;</button>
    </div>
  `).join('');
}

function removeFromPool(index) {
  selectedPool.splice(index, 1);
  renderPaperList();
  renderPool();
  updateGenerateBtn();
}

function addToPoolFromDetail() {
  if (!currentDetail) return;
  const idx = selectedPool.findIndex(s => s.arxiv_id === currentDetail.arxiv_id);
  if (idx >= 0) selectedPool.splice(idx, 1);
  else selectedPool.push(currentDetail);
  renderPaperList();
  renderPool();
  updateGenerateBtn();
  const stillIn = selectedPool.some(s => s.arxiv_id === currentDetail.arxiv_id);
  $('#btn-add-pool').textContent = stillIn ? '- Remove from Pool' : '+ Add to Pool';
}

function updateGenerateBtn() {
  $('#btn-generate').disabled = selectedPool.length === 0;
}

async function analyzePaper() {
  if (!currentDetail) return;
  const btn = $('#btn-analyze-paper');
  btn.disabled = true;
  btn.textContent = 'Analyzing...';
  $('#paper-analysis-output').style.display = 'block';
  $('#paper-analysis-content').innerHTML = '<p style="color:#888">Generating analysis...</p>';

  try {
    const resp = await fetch('/api/analyze-paper', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paper: currentDetail }),
    });
    await readSSE(resp,
      text => { $('#paper-analysis-content').innerHTML = renderMarkdown(text); },
      () => { btn.textContent = 'Re-analyze'; btn.disabled = false; }
    );
  } catch (err) {
    $('#paper-analysis-content').innerHTML = `<p style="color:#f66">Error: ${err.message}</p>`;
    btn.textContent = 'Analyze Paper';
    btn.disabled = false;
  }
}

async function askPaperQuestion() {
  if (!currentDetail) return;
  const input = $('#qa-input');
  const question = input.value.trim();
  if (!question) return;
  input.value = '';

  const messagesDiv = $('#qa-messages');

  // Add user message
  const userDiv = document.createElement('div');
  userDiv.className = 'qa-msg user';
  userDiv.textContent = question;
  messagesDiv.appendChild(userDiv);

  // Add assistant placeholder
  const assistantDiv = document.createElement('div');
  assistantDiv.className = 'qa-msg assistant';
  assistantDiv.innerHTML = '<span style="color:#888">Thinking...</span>';
  messagesDiv.appendChild(assistantDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;

  const btn = $('#btn-qa-send');
  btn.disabled = true;

  try {
    const resp = await fetch('/api/paper-qa', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paper: currentDetail, question, history: paperQAHistory }),
    });
    paperQAHistory.push({ role: 'user', content: question });
    const fullText = await readSSE(resp,
      text => {
        assistantDiv.innerHTML = renderMarkdown(text);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
      },
      () => {}
    );
    paperQAHistory.push({ role: 'assistant', content: fullText });
  } catch (err) {
    assistantDiv.innerHTML = `<span style="color:#f66">Error: ${err.message}</span>`;
  }
  btn.disabled = false;
  input.focus();
}

$('#qa-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); askPaperQuestion(); }
});

async function autoSelect() {
  if (!papers.length) { alert('Search for papers first'); return; }
  const mode = document.querySelector('input[name="sel-mode"]:checked').value;
  const prompt = mode === 'prompt' ? $('#sel-prompt').value.trim() : '';

  $('#btn-auto-select').disabled = true;
  $('#btn-auto-select').textContent = 'Selecting...';

  try {
    const resp = await fetch('/api/auto-select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ papers, prompt }),
    });
    const data = await resp.json();
    selectedPool = [];
    for (const id of (data.selected || [])) {
      const paper = papers.find(p => p.arxiv_id === id);
      if (paper) selectedPool.push(paper);
    }
    renderPaperList();
    renderPool();
    updateGenerateBtn();
  } catch (err) {
    alert('Auto-select failed: ' + err.message);
  }

  $('#btn-auto-select').disabled = false;
  $('#btn-auto-select').textContent = 'Let Agent Select';
}

async function generateIdeas() {
  if (!selectedPool.length) return;
  const userPrompt = $('#idea-prompt').value.trim();
  const outputDiv = $('#idea-output');
  const contentDiv = $('#idea-content');

  outputDiv.style.display = 'block';
  contentDiv.innerHTML = '<div class="paper-list-loading">Generating ideas...</div>';
  $('#btn-generate').disabled = true;

  try {
    const resp = await fetch('/api/generate-ideas', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ papers: selectedPool, prompt: userPrompt }),
    });
    await readSSE(resp,
      text => { contentDiv.innerHTML = renderMarkdown(text); },
      text => { contentDiv.innerHTML = renderMarkdown(text); }
    );
    outputDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    contentDiv.innerHTML = `<p style="color:#ef4444">Error: ${err.message}</p>`;
  }
  $('#btn-generate').disabled = selectedPool.length === 0;
}


// ════════════════════════════════════════
// Tech Learning Tab
// ════════════════════════════════════════

$('#tl-author').addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); searchAuthorPapers(); }
});

async function searchAuthorPapers() {
  const author = $('#tl-author').value.trim();
  if (!author) return;

  const maxResults = parseInt($('#tl-max-results').value);
  $('#tl-paper-list').innerHTML = '<div class="paper-list-loading">Searching arXiv...</div>';
  $('#tl-papers-status').textContent = '';
  $('#btn-author-search').disabled = true;

  try {
    const resp = await fetch('/api/author-papers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ author, max_results: maxResults }),
    });
    const data = await resp.json();

    if (data.error) {
      $('#tl-paper-list').innerHTML = `<div class="paper-list-empty"><p>${escapeHtml(data.error)}</p></div>`;
      $('#btn-author-search').disabled = false;
      return;
    }

    authorPapers = data.papers || [];
    if (author) {
      $('#tl-papers-title').textContent = `Papers by ${author}`;
    }
    renderAuthorPaperList();
    $('#tl-papers-status').textContent = `${authorPapers.length} papers found`;
  } catch (err) {
    $('#tl-paper-list').innerHTML = `<div class="paper-list-empty"><p>Error: ${err.message}</p></div>`;
  }
  $('#btn-author-search').disabled = false;
}

function renderAuthorPaperList() {
  if (!authorPapers.length) {
    $('#tl-paper-list').innerHTML = '<div class="paper-list-empty"><p>No papers found</p></div>';
    return;
  }
  const html = authorPapers.map((p, i) => {
    const isSelected = techAnalysisPool.some(s => s.arxiv_id === p.arxiv_id);
    const authors = (p.authors || []).slice(0, 3).join(', ') + (p.authors?.length > 3 ? ' et al.' : '');
    const date = p.published ? p.published.split('T')[0] : '';
    const cats = (p.categories || []).join(', ');
    return `
      <div class="paper-card ${isSelected ? 'selected' : ''}" data-index="${i}">
        <input type="checkbox" ${isSelected ? 'checked' : ''} onclick="toggleTechPaper(event, ${i})">
        <div class="paper-card-body">
          <div class="paper-card-title">${escapeHtml(p.title)}</div>
          <div class="paper-card-meta">
            <span class="authors">${escapeHtml(authors)}</span>
            <span>${date}</span>
            <span class="categories">${escapeHtml(cats)}</span>
          </div>
        </div>
      </div>`;
  }).join('');
  $('#tl-paper-list').innerHTML = html;
}

function toggleTechPaper(event, index) {
  event.stopPropagation();
  const paper = authorPapers[index];
  const idx = techAnalysisPool.findIndex(s => s.arxiv_id === paper.arxiv_id);
  if (idx >= 0) techAnalysisPool.splice(idx, 1);
  else techAnalysisPool.push(paper);
  renderAuthorPaperList();
  renderTechPool();
  updateAnalyzeBtn();
}

function renderTechPool() {
  const container = $('#tl-analysis-pool');
  $('#tl-pool-count').textContent = techAnalysisPool.length;
  if (!techAnalysisPool.length) {
    container.innerHTML = '<p class="pool-empty">Select papers from the list</p>';
    return;
  }
  container.innerHTML = techAnalysisPool.map((p, i) => `
    <div class="pool-item">
      <span class="pool-item-title" title="${escapeHtml(p.title)}">${escapeHtml(p.title)}</span>
      <button onclick="removeTechPaper(${i})">&#10005;</button>
    </div>
  `).join('');
}

function removeTechPaper(index) {
  techAnalysisPool.splice(index, 1);
  renderAuthorPaperList();
  renderTechPool();
  updateAnalyzeBtn();
}

function updateAnalyzeBtn() {
  $('#btn-analyze').disabled = techAnalysisPool.length === 0;
  $('#btn-tl-download').disabled = !lastAnalysisText;
}

async function analyzeTechniques() {
  if (!techAnalysisPool.length) return;
  const author = $('#tl-author').value.trim();
  const outputDiv = $('#tl-analysis-output');
  const contentDiv = $('#tl-analysis-content');

  outputDiv.style.display = 'block';
  contentDiv.innerHTML = '<div class="paper-list-loading">Analyzing techniques...</div>';
  $('#btn-analyze').disabled = true;

  try {
    const resp = await fetch('/api/analyze-techniques', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ author, papers: techAnalysisPool }),
    });
    lastAnalysisText = await readSSE(resp,
      text => { contentDiv.innerHTML = renderMarkdown(text); },
      text => { contentDiv.innerHTML = renderMarkdown(text); }
    );
    outputDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    $('#btn-tl-download').disabled = false;
  } catch (err) {
    contentDiv.innerHTML = `<p style="color:#ef4444">Error: ${err.message}</p>`;
  }
  $('#btn-analyze').disabled = techAnalysisPool.length === 0;
}

function downloadTechAnalysis() {
  if (!lastAnalysisText) return;
  const author = $('#tl-author').value.trim();
  downloadJSON({
    author,
    papers: techAnalysisPool,
    analysis: lastAnalysisText,
    generated_at: new Date().toISOString(),
  }, `tech_analysis_${author.replace(/\s+/g, '_')}.json`);
}


// ════════════════════════════════════════
// Relations Tab
// ════════════════════════════════════════

async function buildCoauthorNetwork() {
  const author = $('#rel-author').value.trim();
  if (!author) { alert('Enter an author name'); return; }

  // Try to use authorPapers if same author, otherwise fetch
  let papersToUse = authorPapers;
  if (!papersToUse.length || !authorPapers.some(p => (p.authors || []).some(a => a.toLowerCase().includes(author.toLowerCase())))) {
    $('#network-display').style.display = 'block';
    $('#network-display').innerHTML = '<div class="paper-list-loading">Fetching author\'s papers...</div>';
    try {
      const resp = await fetch('/api/author-papers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ author, max_results: 20 }),
      });
      const data = await resp.json();
      papersToUse = data.papers || [];
    } catch (err) {
      $('#network-display').style.display = 'block';
      $('#network-display').innerHTML = `<p style="color:#ef4444">Error: ${err.message}</p>`;
      return;
    }
  }

  if (!papersToUse.length) {
    $('#network-display').style.display = 'block';
    $('#network-display').innerHTML = '<div class="paper-list-empty"><p>No papers found for this author</p></div>';
    return;
  }

  $('#network-display').style.display = 'block';
  $('#network-display').innerHTML = '<div class="paper-list-loading">Building network...</div>';

  try {
    const resp = await fetch('/api/coauthor-network', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ author, papers: papersToUse }),
    });
    currentGraphData = await resp.json();
    currentNetworkType = 'coauthor';
    renderNetwork();
  } catch (err) {
    $('#network-display').innerHTML = `<p style="color:#ef4444">Error: ${err.message}</p>`;
  }
}

async function buildCitationNetwork() {
  // Use papers from selectedPool or techAnalysisPool
  const poolPapers = selectedPool.length ? selectedPool : techAnalysisPool;
  if (!poolPapers.length) {
    alert('Select papers in Papers or Tech Learning tab first');
    return;
  }

  const arxivIds = poolPapers.map(p => p.arxiv_id).filter(Boolean);
  $('#network-display').style.display = 'block';
  $('#network-display').innerHTML = `<div class="paper-list-loading">Fetching citation data (${arxivIds.length} papers, ~${arxivIds.length * 3}s)...</div>`;

  try {
    const resp = await fetch('/api/citation-network', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ arxiv_ids: arxivIds }),
    });
    currentGraphData = await resp.json();
    currentNetworkType = 'citation';
    renderNetwork();
  } catch (err) {
    $('#network-display').innerHTML = `<p style="color:#ef4444">Error: ${err.message}</p>`;
  }
}

async function addPapersToNetwork() {
  const arxivId = $('#rel-add-arxiv').value.trim();
  if (!arxivId) return;
  if (!currentGraphData) { alert('Build a network first'); return; }

  $('#network-display').style.display = 'block';
  $('#network-display').innerHTML = '<div class="paper-list-loading">Adding papers to network...</div>';

  try {
    const resp = await fetch('/api/add-papers-to-network', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        arxiv_ids: [arxivId],
        existing_graph: currentGraphData,
        network_type: currentNetworkType,
      }),
    });
    currentGraphData = await resp.json();
    renderNetwork();
    $('#rel-add-arxiv').value = '';
  } catch (err) {
    $('#network-display').innerHTML = `<p style="color:#ef4444">Error: ${err.message}</p>`;
  }
}

function renderNetwork() {
  if (!currentGraphData || !currentGraphData.nodes?.length) {
    $('#network-display').style.display = 'block';
    $('#network-display').innerHTML = '<div class="paper-list-empty"><p>No data to display</p></div>';
    return;
  }

  // Hide placeholder, show canvas
  $('#network-display').style.display = 'none';
  const canvas = $('#network-canvas');
  canvas.style.display = 'block';
  canvas._nodePositions = null; // reset for fresh layout
  drawNetworkCanvas(currentGraphData);

  // Stats
  const stats = currentGraphData.stats || {};
  $('#network-stats').style.display = 'block';
  $('#stats-nodes').textContent = `${stats.total_nodes || 0} nodes`;
  $('#stats-edges').textContent = `${stats.total_edges || 0} edges`;
  $('#stats-top').textContent = stats.most_connected || '-';

  // Graph data JSON
  $('#graph-data-section').style.display = 'block';
  $('#graph-data-json').textContent = JSON.stringify(currentGraphData, null, 2);

  // Enable download
  $('#btn-rel-download').disabled = false;
}

function drawNetworkCanvas(data) {
  const canvas = $('#network-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;

  // Clear
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#fafbfc';
  ctx.fillRect(0, 0, W, H);

  if (!data.nodes.length) return;

  // Map normalized coords [-1,1] to canvas with padding
  const pad = 60;
  const xs = data.nodes.map(n => n.x);
  const ys = data.nodes.map(n => n.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  function toCanvas(nx, ny) {
    return [
      pad + ((nx - minX) / rangeX) * (W - 2 * pad),
      pad + ((ny - minY) / rangeY) * (H - 2 * pad),
    ];
  }

  // Pre-compute canvas positions (preserve dragged positions)
  const existing = canvas._nodePositions || {};
  const nodePositions = {};
  data.nodes.forEach(n => {
    if (existing[n.id]) {
      nodePositions[n.id] = existing[n.id];
      nodePositions[n.id].node = n;
    } else {
      const [cx, cy] = toCanvas(n.x, n.y);
      nodePositions[n.id] = { cx, cy, node: n };
    }
  });

  // Draw edges
  ctx.strokeStyle = '#c4b5fd';
  data.edges.forEach(e => {
    const from = nodePositions[e.source];
    const to = nodePositions[e.target];
    if (!from || !to) return;
    ctx.lineWidth = Math.min(e.weight || 1, 4);
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    ctx.moveTo(from.cx, from.cy);
    ctx.lineTo(to.cx, to.cy);
    ctx.stroke();

    // Arrow for directed
    if (e.directed) {
      const angle = Math.atan2(to.cy - from.cy, to.cx - from.cx);
      const r = 12;
      const arrowX = to.cx - Math.cos(angle) * 14;
      const arrowY = to.cy - Math.sin(angle) * 14;
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.moveTo(arrowX, arrowY);
      ctx.lineTo(arrowX - r * Math.cos(angle - 0.4), arrowY - r * Math.sin(angle - 0.4));
      ctx.lineTo(arrowX - r * Math.cos(angle + 0.4), arrowY - r * Math.sin(angle + 0.4));
      ctx.closePath();
      ctx.fillStyle = '#aaa';
      ctx.fill();
    }
  });
  ctx.globalAlpha = 1;

  // Draw nodes
  data.nodes.forEach(n => {
    const { cx, cy } = nodePositions[n.id];
    const radius = n.is_center ? 14 : Math.max(7, Math.min(11, 4 + n.paper_count));

    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = n.is_center ? '#4338ca' : '#7c3aed';
    ctx.fill();
    ctx.strokeStyle = n.is_center ? '#312e81' : '#5b21b6';
    ctx.lineWidth = n.is_center ? 2.5 : 1.5;
    ctx.stroke();

    // Label
    ctx.fillStyle = '#1a1a2e';
    ctx.font = n.is_center ? 'bold 11px sans-serif' : '10px sans-serif';
    ctx.textAlign = 'center';
    const label = n.name.length > 20 ? n.name.substring(0, 18) + '...' : n.name;
    ctx.fillText(label, cx, cy + radius + 13);
  });

  // Store positions for click/drag detection
  canvas._nodePositions = nodePositions;
}

// ── Canvas drag & click handling ──
(function() {
  const canvas = $('#network-canvas');
  let dragNode = null;
  let dragStartX = 0, dragStartY = 0;
  let hasDragged = false;

  function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    return [
      (e.clientX - rect.left) * (canvas.width / rect.width),
      (e.clientY - rect.top) * (canvas.height / rect.height),
    ];
  }

  function findNodeAt(mx, my) {
    const positions = canvas._nodePositions;
    if (!positions) return null;
    for (const id of Object.keys(positions)) {
      const { cx, cy, node } = positions[id];
      const r = (node.is_center ? 14 : 11) + 4; // slightly larger hit area
      if (Math.hypot(mx - cx, my - cy) <= r) return id;
    }
    return null;
  }

  canvas.addEventListener('mousedown', function(e) {
    const [mx, my] = getMousePos(e);
    const id = findNodeAt(mx, my);
    if (id) {
      dragNode = id;
      dragStartX = mx;
      dragStartY = my;
      hasDragged = false;
      canvas.style.cursor = 'grabbing';
      e.preventDefault();
    }
  });

  canvas.addEventListener('mousemove', function(e) {
    const [mx, my] = getMousePos(e);
    if (dragNode) {
      hasDragged = true;
      const pos = canvas._nodePositions[dragNode];
      if (pos) {
        pos.cx = mx;
        pos.cy = my;
        drawNetworkCanvas(currentGraphData);
      }
    } else {
      // Hover cursor
      const id = findNodeAt(mx, my);
      canvas.style.cursor = id ? 'grab' : 'default';
    }
  });

  canvas.addEventListener('mouseup', function(e) {
    if (dragNode && !hasDragged) {
      // It was a click, not a drag
      const pos = canvas._nodePositions[dragNode];
      if (pos) showNodeInfo(pos.node);
    }
    dragNode = null;
    canvas.style.cursor = 'default';
  });

  canvas.addEventListener('mouseleave', function() {
    dragNode = null;
    canvas.style.cursor = 'default';
  });

  // Click on empty area — hide panel
  canvas.addEventListener('click', function(e) {
    if (hasDragged) return;
    const [mx, my] = getMousePos(e);
    const id = findNodeAt(mx, my);
    if (!id) $('#node-info').style.display = 'none';
  });
})();

function showNodeInfo(node) {
  $('#node-info').style.display = 'block';
  $('#node-info-name').textContent = node.name;
  $('#node-info-stats').textContent = `${node.paper_count} paper(s) in this network`;

  const papersHtml = (node.papers || []).map(p => `
    <div class="node-paper-item">
      ${escapeHtml(p.title)} ${p.published ? `(${p.published})` : ''}
      ${p.arxiv_id ? ` <a href="https://arxiv.org/abs/${p.arxiv_id}" target="_blank" style="color:#667eea;font-size:0.8rem">[arXiv]</a>` : ''}
    </div>
  `).join('');
  $('#node-info-papers').innerHTML = papersHtml || '<p style="color:#aaa">No paper details available</p>';

  $('#node-info').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function downloadNetworkJSON() {
  if (!currentGraphData) return;
  const type = currentNetworkType === 'coauthor' ? 'coauthor' : 'citation';
  downloadJSON(currentGraphData, `${type}_network.json`);
}


// ════════════════════════════════════════
// Note Workshop Tab
// ════════════════════════════════════════

let noteLatex = '';
let noteChatHistory = [];

// Sync context from Discover tab when switching to Note Workshop
function syncNoteContext() {
  // Papers pool
  const countEl = $('#note-pool-count');
  const listEl = $('#note-papers-list');
  if (countEl) countEl.textContent = selectedPool.length;
  if (listEl) {
    if (!selectedPool.length) {
      listEl.innerHTML = '<p class="pool-empty">Select papers in Discover tab</p>';
    } else {
      listEl.innerHTML = selectedPool.map((p, i) => `
        <div class="pool-item">
          <span class="pool-item-title" title="${escapeHtml(p.title)}">${escapeHtml(p.title)}</span>
        </div>
      `).join('');
    }
  }

  // Ideas
  const ideasEl = $('#note-ideas-preview');
  if (ideasEl) {
    const ideaContent = $('#idea-content');
    if (ideaContent && ideaContent.textContent.trim()) {
      ideasEl.innerHTML = `<div style="max-height:120px;overflow-y:auto;font-size:0.82rem;line-height:1.5">${ideaContent.innerHTML}</div>`;
    } else {
      ideasEl.innerHTML = '<p class="pool-empty">Generate ideas in Discover tab first</p>';
    }
  }
}

// Hook into tab switching
$$('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    if (tab.dataset.tab === 'chat') {
      setTimeout(syncNoteContext, 50);
    }
  });
});

async function generateNote() {
  const instruction = $('#note-instruction').value.trim();
  const ideasText = $('#idea-content') ? $('#idea-content').textContent.trim() : '';

  if (!selectedPool.length && !ideasText && !instruction) {
    alert('Add papers to pool or generate ideas in Discover tab, or provide instructions.');
    return;
  }

  const editor = $('#note-editor');
  const btn = $('#btn-generate-note');
  btn.disabled = true;
  btn.textContent = 'Generating...';
  editor.value = '';
  noteLatex = '';
  noteChatHistory = [];
  $('#note-chat-messages').innerHTML = '';

  try {
    const resp = await fetch('/api/generate-note', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        papers: selectedPool,
        ideas: ideasText,
        instruction: instruction,
      }),
    });

    await readSSE(resp,
      text => {
        // Strip markdown code fences if LLM wraps output
        let clean = text.replace(/^```(?:latex)?\n?/gm, '').replace(/\n?```$/gm, '');
        editor.value = clean;
        noteLatex = clean;
        editor.scrollTop = editor.scrollHeight;
      },
      text => {
        let clean = text.replace(/^```(?:latex)?\n?/gm, '').replace(/\n?```$/gm, '');
        editor.value = clean;
        noteLatex = clean;
      }
    );

    $('#btn-preview-note').disabled = false;
    $('#btn-download-tex').disabled = false;
  } catch (err) {
    editor.value = `% Error: ${err.message}`;
  }
  btn.disabled = false;
  btn.textContent = 'Generate Note';
}

async function sendNoteEdit() {
  const inputEl = $('#note-chat-input');
  const instruction = inputEl.value.trim();
  if (!instruction) return;
  if (!noteLatex.trim()) { alert('Generate a note first'); return; }

  inputEl.value = '';
  const messagesDiv = $('#note-chat-messages');

  // User message
  const userDiv = document.createElement('div');
  userDiv.className = 'qa-msg user';
  userDiv.textContent = instruction;
  messagesDiv.appendChild(userDiv);

  // Assistant placeholder
  const assistantDiv = document.createElement('div');
  assistantDiv.className = 'qa-msg assistant';
  assistantDiv.innerHTML = '<span style="color:#888">Editing...</span>';
  messagesDiv.appendChild(assistantDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;

  const btn = $('#btn-note-chat');
  btn.disabled = true;

  try {
    const resp = await fetch('/api/edit-note', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        latex: noteLatex,
        instruction: instruction,
        history: noteChatHistory,
      }),
    });

    noteChatHistory.push({ role: 'user', content: instruction });

    const editor = $('#note-editor');
    const fullText = await readSSE(resp,
      text => {
        let clean = text.replace(/^```(?:latex)?\n?/gm, '').replace(/\n?```$/gm, '');
        editor.value = clean;
        noteLatex = clean;
        assistantDiv.innerHTML = '<span style="color:#10b981">Updated LaTeX source</span>';
      },
      text => {
        let clean = text.replace(/^```(?:latex)?\n?/gm, '').replace(/\n?```$/gm, '');
        editor.value = clean;
        noteLatex = clean;
      }
    );

    noteChatHistory.push({ role: 'assistant', content: 'Done. Updated the LaTeX source.' });
    assistantDiv.innerHTML = '<span style="color:#10b981">Updated LaTeX source</span>';
  } catch (err) {
    assistantDiv.innerHTML = `<span style="color:#f66">Error: ${err.message}</span>`;
  }

  btn.disabled = false;
  inputEl.focus();
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

$('#note-chat-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); sendNoteEdit(); }
});

function previewNote() {
  const latex = $('#note-editor').value.trim();
  if (!latex) return;

  const previewSection = $('#note-preview-section');
  const previewDiv = $('#note-preview');
  previewSection.style.display = 'block';
  previewDiv.innerHTML = renderLaTeXPreview(latex);
  previewSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderLaTeXPreview(latex) {
  let html = '';

  // Extract title
  const titleMatch = latex.match(/\\title\{([^}]*)\}/);
  if (titleMatch) html += `<div class="latex-title">${renderLatexText(titleMatch[1])}</div>`;

  // Extract author
  const authorMatch = latex.match(/\\author\{([^}]*)\}/);
  if (authorMatch) html += `<div class="latex-author">${renderLatexText(authorMatch[1])}</div>`;

  // Extract abstract
  const absMatch = latex.match(/\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}/);
  if (absMatch) {
    html += `<div class="latex-abstract"><div class="latex-abstract-title">Abstract</div>${renderLatexText(absMatch[1].trim())}</div>`;
  }

  // Get body: everything after \maketitle (or \begin{document}) until \end{document}
  let body = latex;
  const maketitleIdx = latex.indexOf('\\maketitle');
  if (maketitleIdx >= 0) {
    body = latex.substring(maketitleIdx + 10);
  } else {
    const beginDocIdx = latex.indexOf('\\begin{document}');
    if (beginDocIdx >= 0) body = latex.substring(beginDocIdx + 16);
  }
  const endDocIdx = body.indexOf('\\end{document}');
  if (endDocIdx >= 0) body = body.substring(0, endDocIdx);

  // Remove abstract (already rendered)
  body = body.replace(/\\begin\{abstract\}[\s\S]*?\\end\{abstract\}/, '');

  // Process bibliography
  let bibHtml = '';
  const bibMatch = body.match(/\\begin\{thebibliography\}[\s\S]*?\\end\{thebibliography\}/);
  if (bibMatch) {
    body = body.replace(bibMatch[0], '');
    const bibItems = bibMatch[0].match(/\\bibitem\{([^}]*)\}\s*([\s\S]*?)(?=\\bibitem|\\end\{thebibliography\})/g) || [];
    if (bibItems.length) {
      bibHtml = '<div class="latex-bib"><h2>References</h2>';
      bibItems.forEach(item => {
        const m = item.match(/\\bibitem\{([^}]*)\}\s*([\s\S]*)/);
        if (m) bibHtml += `<div class="latex-bib-item">[${m[1]}] ${renderLatexText(m[2].trim())}</div>`;
      });
      bibHtml += '</div>';
    }
  }

  // Process sections
  body = body.replace(/\\section\*?\{([^}]*)\}/g, (_, t) => `</p><h2>${renderLatexText(t)}</h2><p>`);
  body = body.replace(/\\subsection\*?\{([^}]*)\}/g, (_, t) => `</p><h3>${renderLatexText(t)}</h3><p>`);

  // Process theorem-like environments
  const envNames = ['theorem', 'lemma', 'proposition', 'corollary', 'definition', 'remark', 'proof'];
  envNames.forEach(env => {
    const label = env.charAt(0).toUpperCase() + env.slice(1);
    const re = new RegExp(`\\\\begin\\{${env}\\}(\\[([^\\]]*)\\])?([\\s\\S]*?)\\\\end\\{${env}\\}`, 'g');
    body = body.replace(re, (_, _opt, optName, content) => {
      const nameStr = optName ? ` (${renderLatexText(optName)})` : '';
      const qed = env === 'proof' ? '<span style="float:right">□</span>' : '';
      return `</p><div class="latex-env ${env}"><div class="latex-env-label">${label}${nameStr}.</div>${renderLatexText(content.trim())}${qed}</div><p>`;
    });
  });

  // Process lists
  body = body.replace(/\\begin\{enumerate\}([\s\S]*?)\\end\{enumerate\}/g, (_, content) => {
    const items = content.split('\\item').filter(s => s.trim());
    return '<ol>' + items.map(i => `<li>${renderLatexText(i.trim())}</li>`).join('') + '</ol>';
  });
  body = body.replace(/\\begin\{itemize\}([\s\S]*?)\\end\{itemize\}/g, (_, content) => {
    const items = content.split('\\item').filter(s => s.trim());
    return '<ul>' + items.map(i => `<li>${renderLatexText(i.trim())}</li>`).join('') + '</ul>';
  });

  // Render remaining body text
  html += '<div>' + renderLatexText(body) + '</div>';
  html += bibHtml;

  // Clean up empty <p> tags
  html = html.replace(/<p>\s*<\/p>/g, '');

  return html;
}

function renderLatexText(text) {
  if (!text) return '';

  // Display math: $$...$$ and \[...\]
  text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
    try { return katex.renderToString(math.trim(), { displayMode: true, throwOnError: false }); }
    catch { return `<div><code>${math}</code></div>`; }
  });
  text = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => {
    try { return katex.renderToString(math.trim(), { displayMode: true, throwOnError: false }); }
    catch { return `<div><code>${math}</code></div>`; }
  });

  // Equation environments
  text = text.replace(/\\begin\{(?:equation|align)\*?\}([\s\S]*?)\\end\{(?:equation|align)\*?\}/g, (_, math) => {
    try { return katex.renderToString(math.trim(), { displayMode: true, throwOnError: false }); }
    catch { return `<div><code>${math}</code></div>`; }
  });

  // Inline math: $...$
  text = text.replace(/\$([^\$\n]+?)\$/g, (_, math) => {
    try { return katex.renderToString(math.trim(), { displayMode: false, throwOnError: false }); }
    catch { return `<code>${math}</code>`; }
  });

  // Text formatting
  text = text.replace(/\\textbf\{([^}]*)\}/g, '<strong>$1</strong>');
  text = text.replace(/\\textit\{([^}]*)\}/g, '<em>$1</em>');
  text = text.replace(/\\emph\{([^}]*)\}/g, '<em>$1</em>');
  text = text.replace(/\\underline\{([^}]*)\}/g, '<u>$1</u>');
  text = text.replace(/\\texttt\{([^}]*)\}/g, '<code>$1</code>');

  // References
  text = text.replace(/\\cite\{([^}]*)\}/g, '[$1]');
  text = text.replace(/\\ref\{([^}]*)\}/g, '($1)');
  text = text.replace(/\\label\{[^}]*\}/g, '');

  // Line breaks and spacing
  text = text.replace(/\\\\/g, '<br>');
  text = text.replace(/\\\\$/gm, '<br>');
  text = text.replace(/\\(quad|qquad)/g, '&emsp;');
  text = text.replace(/\\,/g, '&thinsp;');
  text = text.replace(/~/g, '&nbsp;');

  // Clean up remaining LaTeX commands (not already handled)
  text = text.replace(/\\(?:noindent|medskip|bigskip|smallskip|vspace\{[^}]*\}|hspace\{[^}]*\}|newpage|clearpage|maketitle)/g, '');

  // Paragraphs (double newlines)
  text = text.replace(/\n\n+/g, '</p><p>');

  return text;
}

function downloadTeX() {
  const latex = $('#note-editor').value;
  if (!latex) return;
  const blob = new Blob([latex], { type: 'application/x-tex' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'research_note.tex';
  a.click();
  URL.revokeObjectURL(a.href);
}


// ════════════════════════════════════════
// Reports Tab
// ════════════════════════════════════════

async function refreshLog() {
  const resp = await fetch('/api/experiments');
  const data = await resp.json();
  $('#log-summary').textContent = `Total: ${data.total} | Success rate: ${(data.success_rate * 100).toFixed(0)}%`;

  let html = '<table style="width:100%;font-size:0.8rem;border-collapse:collapse">';
  html += '<tr><th>Time</th><th>Method</th><th>Domain</th><th>Result</th><th>Status</th></tr>';
  for (const row of data.rows) {
    html += `<tr><td>${row[0]}</td><td>${row[1]}</td><td>${row[2]}</td><td>${row[3]?.substring(0,60)}...</td><td>${row[4]}</td></tr>`;
  }
  html += '</table>';
  $('#experiment-table').innerHTML = html;
}

async function genReport() {
  const fmt = document.querySelector('input[name="fmt"]:checked')?.value || 'markdown';
  const resp = await fetch(`/api/report?fmt=${fmt}`);
  const data = await resp.json();
  $('#report-preview').innerHTML = renderMarkdown(data.content || 'No content');
  if (data.filepath) {
    $('#report-download').innerHTML = `<a href="/api/download?path=${encodeURIComponent(data.filepath)}">Download report</a>`;
  }
}

async function genNotebook() {
  const resp = await fetch('/api/notebook');
  const data = await resp.json();
  if (data.filepath) {
    $('#report-download').innerHTML = `<a href="/api/download?path=${encodeURIComponent(data.filepath)}">Download notebook</a>`;
  }
}
