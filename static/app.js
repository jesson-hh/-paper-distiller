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

// ── Persistence Helpers ──
function saveState(key, value) {
  try { localStorage.setItem('mathagent:' + key, JSON.stringify(value)); } catch {}
}
function loadState(key, fallback) {
  try {
    const v = localStorage.getItem('mathagent:' + key);
    return v !== null ? JSON.parse(v) : fallback;
  } catch { return fallback; }
}

// ── Theme Toggle ──
function toggleTheme() {
  document.documentElement.classList.toggle('dark');
  const isDark = document.documentElement.classList.contains('dark');
  localStorage.setItem('mathagent:theme', isDark ? 'dark' : 'light');
}
// Apply saved theme on load
(function() {
  const saved = localStorage.getItem('mathagent:theme');
  if (saved === 'dark') document.documentElement.classList.add('dark');
})();

// ── Sidebar Navigation ──
$$('.nav-item').forEach(item => {
  item.addEventListener('click', () => {
    $$('.nav-item').forEach(t => t.classList.remove('active'));
    $$('.tab-content').forEach(tc => tc.classList.remove('active'));
    item.classList.add('active');
    $(`#tab-${item.dataset.tab}`).classList.add('active');
    saveState('activeTab', item.dataset.tab);

    // Show/hide sub-menu for Paper Search
    const sub = $('#nav-sub-discover');
    if (item.dataset.tab === 'discover') {
      sub.classList.remove('collapsed');
    } else {
      sub.classList.add('collapsed');
    }
  });
});

// ── Discover Sub-navigation ──
$$('.nav-subitem').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.nav-subitem').forEach(b => b.classList.remove('active'));
    $$('.discover-subcontent').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    $(`#subtab-${btn.dataset.subtab}`).classList.add('active');
    saveState('activeSubTab', btn.dataset.subtab);
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
  saveState('lastSearchQuery', query);

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
  saveState('selectedPool', selectedPool);
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
  $('#paper-analysis-content').innerHTML = '<p style="color:var(--text-3)">Generating analysis...</p>';

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
    $('#paper-analysis-content').innerHTML = `<p style="color:var(--red)">Error: ${err.message}</p>`;
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
  assistantDiv.innerHTML = '<span style="color:var(--text-3)">Thinking...</span>';
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
    assistantDiv.innerHTML = `<span style="color:var(--red)">Error: ${err.message}</span>`;
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
      text => {
        contentDiv.innerHTML = renderMarkdown(text);
        saveState('lastIdeas', text);
      }
    );
    outputDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    contentDiv.innerHTML = `<p style="color:var(--red)">Error: ${err.message}</p>`;
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
  saveState('lastAuthorQuery', author);

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
  saveState('techAnalysisPool', techAnalysisPool);
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
    contentDiv.innerHTML = `<p style="color:var(--red)">Error: ${err.message}</p>`;
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
      $('#network-display').innerHTML = `<p style="color:var(--red)">Error: ${err.message}</p>`;
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
    $('#network-display').innerHTML = `<p style="color:var(--red)">Error: ${err.message}</p>`;
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
    $('#network-display').innerHTML = `<p style="color:var(--red)">Error: ${err.message}</p>`;
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
    $('#network-display').innerHTML = `<p style="color:var(--red)">Error: ${err.message}</p>`;
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
      ${p.arxiv_id ? ` <a href="https://arxiv.org/abs/${p.arxiv_id}" target="_blank" style="color:var(--primary);font-size:0.8rem">[arXiv]</a>` : ''}
    </div>
  `).join('');
  $('#node-info-papers').innerHTML = papersHtml || '<p style="color:var(--text-3)">No paper details available</p>';

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
let pendingNoteLatex = '';  // pending edit awaiting confirmation

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
$$('.nav-item').forEach(tab => {
  tab.addEventListener('click', () => {
    if (tab.dataset.tab === 'chat') {
      setTimeout(syncNoteContext, 50);
    }
  });
});

async function generateNote() {
  const instruction = $('#note-instruction').value.trim();
  const ideasText = $('#idea-content') ? $('#idea-content').textContent.trim() : '';
  const template = $('#note-template') ? $('#note-template').value : 'research';
  const templatePrompt = NOTE_TEMPLATES[template] || '';

  if (!selectedPool.length && !ideasText && !instruction) {
    alert('Add papers to pool or generate ideas in Discover tab, or provide instructions.');
    return;
  }

  const fullInstruction = templatePrompt ? `${templatePrompt}\n\n${instruction}` : instruction;

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
        instruction: fullInstruction,
      }),
    });

    await readSSE(resp,
      text => {
        // Strip markdown code fences if LLM wraps output
        let clean = text.replace(/^```(?:latex)?\n?/gm, '').replace(/\n?```$/gm, '');
        editor.value = clean;
        noteLatex = clean;
        editor.scrollTop = editor.scrollHeight;
        syncEditorHighlight();
      },
      text => {
        let clean = text.replace(/^```(?:latex)?\n?/gm, '').replace(/\n?```$/gm, '');
        editor.value = clean;
        noteLatex = clean;
        saveState('noteLatex', noteLatex);
        syncEditorHighlight();
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
  const chatMessages = $('#note-chat-messages');

  // User message
  const userDiv = document.createElement('div');
  userDiv.className = 'qa-msg user';
  userDiv.textContent = instruction;
  chatMessages.appendChild(userDiv);

  // Assistant placeholder
  const assistantDiv = document.createElement('div');
  assistantDiv.className = 'qa-msg assistant';
  assistantDiv.innerHTML = '<span style="color:var(--text-3)">Editing...</span>';
  chatMessages.appendChild(assistantDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  const btn = $('#btn-note-chat');
  btn.disabled = true;
  pendingNoteLatex = '';

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

    const fullText = await readSSE(resp,
      text => {
        let clean = text.replace(/^```(?:latex)?\n?/gm, '').replace(/\n?```$/gm, '');
        pendingNoteLatex = clean;
        assistantDiv.innerHTML = '<span style="color:var(--amber)">Changes ready — review and accept/reject</span>';
      },
      text => {
        let clean = text.replace(/^```(?:latex)?\n?/gm, '').replace(/\n?```$/gm, '');
        pendingNoteLatex = clean;
      }
    );

    noteChatHistory.push({ role: 'assistant', content: 'Proposed changes ready for review.' });

    // Show pending preview in editor with visual indicator
    const editor = $('#note-editor');
    editor.value = pendingNoteLatex;
    editor.classList.add('pending-edit');

    // Show confirm bar
    $('#note-confirm-bar').style.display = 'flex';

    // Auto-preview the pending version
    if ($('#note-preview-section').style.display !== 'none') {
      $('#note-preview').innerHTML = renderLaTeXPreview(pendingNoteLatex);
    }

    assistantDiv.innerHTML = '<span style="color:var(--amber)">Changes ready — accept or reject below</span>';
  } catch (err) {
    assistantDiv.innerHTML = `<span style="color:var(--red)">Error: ${err.message}</span>`;
  }

  btn.disabled = false;
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function acceptNoteEdit() {
  if (!pendingNoteLatex) return;
  noteLatex = pendingNoteLatex;
  pendingNoteLatex = '';
  saveState('noteLatex', noteLatex);
  $('#note-editor').classList.remove('pending-edit');
  $('#note-confirm-bar').style.display = 'none';

  // Add confirmation to chat
  const chatMessages = $('#note-chat-messages');
  const div = document.createElement('div');
  div.className = 'qa-msg assistant';
  div.innerHTML = '<span style="color:var(--success)">Changes accepted</span>';
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  // Refresh preview if visible
  if ($('#note-preview-section').style.display !== 'none') {
    previewNote();
  }
}

function rejectNoteEdit() {
  pendingNoteLatex = '';
  $('#note-editor').value = noteLatex;  // restore original
  $('#note-editor').classList.remove('pending-edit');
  $('#note-confirm-bar').style.display = 'none';

  // Add rejection to chat
  const chatMessages = $('#note-chat-messages');
  const div = document.createElement('div');
  div.className = 'qa-msg assistant';
  div.innerHTML = '<span style="color:var(--red)">Changes rejected — reverted to previous version</span>';
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  // Restore preview
  if ($('#note-preview-section').style.display !== 'none') {
    previewNote();
  }
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

  // Attach click handlers for preview→source navigation
  previewDiv.querySelectorAll('.latex-nav').forEach(el => {
    el.style.cursor = 'pointer';
    el.title = 'Click to jump to source';
    el.addEventListener('click', () => {
      const searchText = el.dataset.search;
      if (!searchText) return;
      const editor = $('#note-editor');
      const src = editor.value;
      // Find the raw section command in source
      const re = new RegExp(searchText.replace(/\\\\/g, '\\'));
      const match = re.exec(src);
      if (match) {
        editor.focus();
        editor.setSelectionRange(match.index, match.index + match[0].length);
        // Scroll to selection
        const linesBefore = src.substring(0, match.index).split('\n').length;
        const lineHeight = 19; // approximate
        editor.scrollTop = Math.max(0, (linesBefore - 3) * lineHeight);
      }
    });
  });

  previewSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Source→Preview navigation: Ctrl+Click on section in editor jumps to preview
$('#note-editor').addEventListener('dblclick', function() {
  const editor = this;
  const pos = editor.selectionStart;
  const text = editor.value;

  // Find the line at cursor
  const lineStart = text.lastIndexOf('\n', pos - 1) + 1;
  const lineEnd = text.indexOf('\n', pos);
  const line = text.substring(lineStart, lineEnd >= 0 ? lineEnd : text.length);

  // Check if it's a section/subsection line
  const secMatch = line.match(/\\(sub)?section\*?\{([^}]*)\}/);
  if (!secMatch) return;

  const sectionTitle = secMatch[2];
  const previewDiv = $('#note-preview');
  if (!previewDiv) return;

  // Find matching heading in preview
  const headings = previewDiv.querySelectorAll('.latex-nav');
  for (const h of headings) {
    if (h.textContent.trim().includes(sectionTitle.trim())) {
      h.scrollIntoView({ behavior: 'smooth', block: 'center' });
      h.style.transition = 'background 0.3s';
      h.style.background = 'var(--primary-bg)';
      setTimeout(() => { h.style.background = ''; }, 1500);
      break;
    }
  }
});

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

  // Process sections — add data-src for source↔preview sync
  let sectionCounter = 0;
  body = body.replace(/\\section\*?\{([^}]*)\}/g, (match, t) => {
    const id = `sec-${sectionCounter++}`;
    return `</p><h2 class="latex-nav" data-search="\\\\section{${t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}}" data-id="${id}" id="pv-${id}">${renderLatexText(t)}</h2><p>`;
  });
  body = body.replace(/\\subsection\*?\{([^}]*)\}/g, (match, t) => {
    const id = `sec-${sectionCounter++}`;
    return `</p><h3 class="latex-nav" data-search="\\\\subsection{${t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}}" data-id="${id}" id="pv-${id}">${renderLatexText(t)}</h3><p>`;
  });

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

let lastExperimentPlan = '';
let lastCodePlan = '';

// Sync context from other tabs when switching to Reports
function syncReportContext() {
  const ideasEl = $('#report-ideas-preview');
  if (ideasEl) {
    const ideaContent = $('#idea-content');
    if (ideaContent && ideaContent.textContent.trim()) {
      ideasEl.innerHTML = `<div style="max-height:110px;overflow-y:auto;font-size:0.82rem;line-height:1.5">${ideaContent.innerHTML}</div>`;
    } else {
      ideasEl.innerHTML = '<p class="pool-empty">Generate ideas in Discover tab</p>';
    }
  }
  const noteEl = $('#report-note-preview');
  if (noteEl) {
    if (noteLatex && noteLatex.trim()) {
      const preview = noteLatex.length > 500 ? noteLatex.substring(0, 500) + '...' : noteLatex;
      noteEl.innerHTML = `<pre style="max-height:110px;overflow-y:auto;font-size:0.75rem;white-space:pre-wrap;word-break:break-all;line-height:1.4">${escapeHtml(preview)}</pre>`;
    } else {
      noteEl.innerHTML = '<p class="pool-empty">Generate a note in Note Workshop</p>';
    }
  }
}

// Hook tab switching for Reports
$$('.nav-item').forEach(tab => {
  tab.addEventListener('click', () => {
    if (tab.dataset.tab === 'reports') {
      setTimeout(syncReportContext, 50);
    }
  });
});

async function designExperiment() {
  const instruction = $('#exp-instruction').value.trim();
  const ideasText = $('#idea-content') ? $('#idea-content').textContent.trim() : '';

  if (!ideasText && !noteLatex && !instruction) {
    alert('Generate ideas or a note first, or enter instructions.');
    return;
  }

  const btn = $('#btn-design-exp');
  btn.disabled = true;
  btn.textContent = 'Designing...';
  const output = $('#experiment-plan-output');
  output.style.display = 'block';
  output.innerHTML = '<p style="color:var(--text-3)">Designing experiments...</p>';

  try {
    const resp = await fetch('/api/design-experiment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ideas: ideasText,
        note_latex: noteLatex,
        instruction: instruction,
      }),
    });

    const fullText = await readSSE(resp,
      text => { output.innerHTML = renderMarkdown(text); },
      text => { output.innerHTML = renderMarkdown(text); }
    );
    lastExperimentPlan = fullText;
    $('#btn-gen-code').disabled = false;
  } catch (err) {
    output.innerHTML = `<p style="color:var(--red)">Error: ${err.message}</p>`;
  }
  btn.disabled = false;
  btn.textContent = 'Design';
}

$('#exp-instruction').addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); designExperiment(); }
});

async function generateCodePlan() {
  if (!lastExperimentPlan) { alert('Design an experiment first'); return; }

  const instruction = $('#code-instruction').value.trim();
  const btn = $('#btn-gen-code');
  btn.disabled = true;
  btn.textContent = 'Generating...';

  const outputDiv = $('#code-plan-output');
  const contentDiv = $('#code-plan-content');
  outputDiv.style.display = 'block';
  contentDiv.innerHTML = '<p style="color:var(--text-3)">Generating code...</p>';

  try {
    const resp = await fetch('/api/generate-code-plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        experiment_plan: lastExperimentPlan,
        note_latex: noteLatex,
        instruction: instruction,
      }),
    });

    const fullText = await readSSE(resp,
      text => { contentDiv.innerHTML = renderMarkdown(text); },
      text => { contentDiv.innerHTML = renderMarkdown(text); }
    );
    lastCodePlan = fullText;
    $('#btn-download-py').disabled = false;
    $('#btn-download-all').disabled = false;
  } catch (err) {
    contentDiv.innerHTML = `<p style="color:var(--red)">Error: ${err.message}</p>`;
  }
  btn.disabled = false;
  btn.textContent = 'Generate Code';
}

$('#code-instruction').addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); generateCodePlan(); }
});

function extractCodeBlocks(text) {
  const blocks = [];
  // Match ### filename followed by ```lang ... ```
  const re = /###\s+(\S+)\s*\n```[\w]*\n([\s\S]*?)```/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    blocks.push({ filename: m[1], content: m[2].trim() });
  }
  // Fallback: if no ### headers, extract all code blocks
  if (!blocks.length) {
    const fallback = /```(?:python)?\n([\s\S]*?)```/g;
    let idx = 0;
    while ((m = fallback.exec(text)) !== null) {
      blocks.push({ filename: idx === 0 ? 'experiment.py' : `file_${idx}.py`, content: m[1].trim() });
      idx++;
    }
  }
  return blocks;
}

function downloadCodeFile() {
  if (!lastCodePlan) return;
  const blocks = extractCodeBlocks(lastCodePlan);
  // Download the main .py file (first python block)
  const pyBlock = blocks.find(b => b.filename.endsWith('.py')) || blocks[0];
  if (!pyBlock) { alert('No code found'); return; }
  const blob = new Blob([pyBlock.content], { type: 'text/x-python' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = pyBlock.filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function downloadAllCode() {
  if (!lastCodePlan) return;
  const blocks = extractCodeBlocks(lastCodePlan);
  if (!blocks.length) { alert('No code found'); return; }

  if (blocks.length === 1) {
    downloadCodeFile();
    return;
  }

  // Download each file individually
  blocks.forEach(block => {
    const type = block.filename.endsWith('.py') ? 'text/x-python' : 'text/plain';
    const blob = new Blob([block.content], { type });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = block.filename;
    a.click();
    URL.revokeObjectURL(a.href);
  });
}

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

// ════════════════════════════════════════
// Auto Research Tab
// ════════════════════════════════════════

let researchRunning = false;

$('#research-max-iter').addEventListener('input', e => {
  $('#research-iter-val').textContent = e.target.value;
});
$('#research-max-time').addEventListener('input', e => {
  $('#research-time-val').textContent = e.target.value;
});

function switchResearchOutput(tab) {
  $$('.research-output-tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  $('#research-text-output').style.display = tab === 'text' ? 'block' : 'none';
  $('#research-scratchpad-output').style.display = tab === 'scratchpad' ? 'block' : 'none';
}

async function startAutonomousResearch() {
  const goal = $('#research-goal').value.trim();
  if (!goal) { alert('Please enter a research goal.'); return; }

  const domain = $('#research-domain').value;
  const maxIter = parseInt($('#research-max-iter').value);
  const maxTime = parseInt($('#research-max-time').value) * 60;

  researchRunning = true;
  $('#btn-start-research').disabled = true;
  $('#btn-stop-research').disabled = false;
  $('#research-progress').style.display = 'block';
  $('#research-tasks').style.display = 'block';
  $('#research-output').style.display = 'block';
  $('#research-text-output').innerHTML = '';
  $('#research-scratchpad-output').innerHTML = '';
  $('#research-task-list').textContent = '';
  $('#research-phase').textContent = 'STARTING';
  $('#research-status').textContent = 'Initializing...';
  $('#research-progress-fill').style.width = '0%';

  let iterationCount = 0;

  try {
    const resp = await fetch('/api/autonomous', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ goal, domain, max_iterations: maxIter, max_time: maxTime }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') { researchRunning = false; break; }

        try {
          const evt = JSON.parse(raw);

          if (evt.type === 'phase') {
            $('#research-phase').textContent = evt.phase;
            $('#research-status').textContent = evt.status || '';
            if (evt.phase === 'EXECUTING') {
              iterationCount++;
              const pct = Math.min((iterationCount / maxIter) * 100, 100);
              $('#research-progress-fill').style.width = pct + '%';
            } else if (evt.phase === 'COMPLETE') {
              $('#research-progress-fill').style.width = '100%';
            }
          }

          if (evt.type === 'plan') {
            $('#research-task-list').textContent = evt.content;
          }

          if (evt.type === 'text') {
            $('#research-text-output').innerHTML = renderMarkdown(evt.content);
            $('#research-text-output').scrollTop = $('#research-text-output').scrollHeight;
          }

          if (evt.type === 'scratchpad') {
            $('#research-scratchpad-output').textContent = evt.content;
            $('#research-scratchpad-output').scrollTop = $('#research-scratchpad-output').scrollHeight;
          }

          if (evt.type === 'image') {
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${evt.content}`;
            img.style.maxWidth = '100%';
            img.style.borderRadius = 'var(--radius-sm)';
            img.style.margin = '8px 0';
            $('#research-text-output').appendChild(img);
          }

          if (evt.type === 'done') {
            $('#research-phase').textContent = 'COMPLETE';
            $('#research-status').textContent = 'Research complete';
            $('#research-progress-fill').style.width = '100%';
          }
        } catch {}
      }
    }
  } catch (err) {
    $('#research-status').textContent = `Error: ${err.message}`;
  }

  researchRunning = false;
  $('#btn-start-research').disabled = false;
  $('#btn-stop-research').disabled = true;
}

async function stopAutonomousResearch() {
  try {
    await fetch('/api/stop-research', { method: 'POST' });
    $('#research-status').textContent = 'Stopping...';
  } catch {}
}

// ════════════════════════════════════════
// LaTeX Syntax Highlighting
// ════════════════════════════════════════

function highlightLatex(text) {
  let html = escapeHtml(text);
  // Comments (must come first to avoid inner matches)
  html = html.replace(/(%.*)$/gm, '<span class="hl-comment">$1</span>');
  // \begin{...} and \end{...}
  html = html.replace(/(\\(?:begin|end))\{([^}]*)\}/g, '<span class="hl-env">$1{$2}</span>');
  // Math: $$...$$ and $...$
  html = html.replace(/(\$\$[\s\S]*?\$\$)/g, '<span class="hl-math">$1</span>');
  html = html.replace(/(\$[^$\n]+?\$)/g, '<span class="hl-math">$1</span>');
  // Commands: \word
  html = html.replace(/(\\[a-zA-Z]+)/g, '<span class="hl-cmd">$1</span>');
  return html;
}

function syncEditorHighlight() {
  const editor = $('#note-editor');
  const highlight = $('#editor-highlight-code');
  if (!editor || !highlight) return;
  highlight.innerHTML = highlightLatex(editor.value) + '\n';
  // Sync scroll
  const pre = $('#editor-highlight');
  pre.scrollTop = editor.scrollTop;
  pre.scrollLeft = editor.scrollLeft;
}

// Hook editor events
(function() {
  const editor = $('#note-editor');
  if (!editor) return;
  editor.addEventListener('input', syncEditorHighlight);
  editor.addEventListener('scroll', () => {
    const pre = $('#editor-highlight');
    pre.scrollTop = editor.scrollTop;
    pre.scrollLeft = editor.scrollLeft;
  });
})();

// ════════════════════════════════════════
// BibTeX Generation
// ════════════════════════════════════════

function generateBibTeX() {
  if (!selectedPool.length) {
    alert('No papers in selection pool. Add papers in the Paper Search tab first.');
    return;
  }

  const entries = selectedPool.map(p => {
    const firstAuthor = (p.authors && p.authors[0]) ? p.authors[0].split(' ').pop().toLowerCase() : 'unknown';
    const year = p.published ? p.published.substring(0, 4) : '2024';
    const titleWord = p.title.split(/\s+/).find(w => w.length > 4)?.toLowerCase() || 'paper';
    const key = `${firstAuthor}${year}${titleWord}`;
    const authors = (p.authors || []).join(' and ');
    const title = p.title || '';
    const arxivId = p.arxiv_id || '';

    return `@article{${key},
  title  = {${title}},
  author = {${authors}},
  year   = {${year}},
  eprint = {${arxivId}},
  archivePrefix = {arXiv}
}`;
  });

  const bibtex = entries.join('\n\n');

  // Insert into editor or download
  const editor = $('#note-editor');
  if (editor.value.trim()) {
    // Append to existing note
    editor.value += '\n\n% ── BibTeX ──\n% ' + bibtex.split('\n').join('\n% ');
    noteLatex = editor.value;
    saveState('noteLatex', noteLatex);
    syncEditorHighlight();
  }

  // Also offer download
  const blob = new Blob([bibtex], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'references.bib';
  a.click();
  URL.revokeObjectURL(a.href);
}

// ════════════════════════════════════════
// Note Template Support
// ════════════════════════════════════════

const NOTE_TEMPLATES = {
  'research': '',
  'survey': 'Generate a SURVEY/REVIEW style note. Include: an introduction summarizing the field, a comparison table of different approaches, per-paper summaries organized by theme, a discussion section, and open problems.',
  'problem-set': 'Generate a PROBLEM SET style note. Format as a series of numbered problems/exercises with solutions. Include hints, difficulty levels, and connections to the papers.',
  'proof-sketch': 'Generate a PROOF SKETCH style note. Focus on key theorems and their proof ideas. Use concise notation, include lemma statements, and outline proof strategies without full details.',
};

// ════════════════════════════════════════
// Restore Persisted State on Page Load
// ════════════════════════════════════════
(function restoreState() {
  // Restore search queries
  const lastQuery = loadState('lastSearchQuery', '');
  if (lastQuery) $('#search-query').value = lastQuery;
  const lastAuthor = loadState('lastAuthorQuery', '');
  if (lastAuthor) $('#tl-author').value = lastAuthor;

  // Restore paper pools
  const savedPool = loadState('selectedPool', []);
  if (savedPool.length) {
    selectedPool = savedPool;
    renderPool();
    renderPaperList();
    updateGenerateBtn();
  }
  const savedTechPool = loadState('techAnalysisPool', []);
  if (savedTechPool.length) {
    techAnalysisPool = savedTechPool;
    renderTechPool();
  }

  // Restore note
  const savedNote = loadState('noteLatex', '');
  if (savedNote) {
    noteLatex = savedNote;
    $('#note-editor').value = savedNote;
    $('#btn-preview-note').disabled = false;
    $('#btn-download-tex').disabled = false;
    syncEditorHighlight();
  }

  // Restore ideas
  const savedIdeas = loadState('lastIdeas', '');
  if (savedIdeas) {
    $('#idea-output').style.display = 'block';
    $('#idea-content').innerHTML = renderMarkdown(savedIdeas);
  }

  // Restore active tab
  const savedTab = loadState('activeTab', 'discover');
  if (savedTab !== 'discover') {
    const tabBtn = $(`.nav-item[data-tab="${savedTab}"]`);
    if (tabBtn) tabBtn.click();
  }

  // Restore active subtab
  const savedSubTab = loadState('activeSubTab', 'papers');
  if (savedSubTab !== 'papers') {
    const subBtn = $(`.nav-subitem[data-subtab="${savedSubTab}"]`);
    if (subBtn) subBtn.click();
  }
})();
