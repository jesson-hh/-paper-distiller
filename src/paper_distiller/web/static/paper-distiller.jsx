// paper-distiller — web frontend (wired to backend)
const { useState, useEffect, useRef, useCallback } = React;

// ───────────────────────────────────────────────────────────
// VAULT PATH — read from <meta name="vault-path"> injected by server.py

function getVaultPath() {
  const meta = document.querySelector('meta[name="vault-path"]');
  return meta ? meta.getAttribute("content") || "" : "";
}

const VAULT_PATH = getVaultPath();

// ───────────────────────────────────────────────────────────
// SEED PROMPTS (static — UI-only)

const SEED_PROMPTS = [
  { tag: "蒸馏", text: "找几篇 Transformer 注意力机制的核心论文,蒸馏 3 篇" },
  { tag: "问答", text: "解释一下 Attention Is All You Need 里 √d_k 的作用" },
  { tag: "审查", text: "把开放的这篇证明跑一遍审查,找最值得手工核对的步骤" },
  { tag: "深度研究", text: "亚二次复杂度的 attention 路线有哪些?" }
];

// ───────────────────────────────────────────────────────────
// HELPERS

const uid = () => Math.random().toString(36).slice(2, 9);

// ───────────────────────────────────────────────────────────
// API helpers

async function apiFetch(path, params = {}) {
  const url = new URL(path, window.location.origin);
  if (VAULT_PATH) url.searchParams.set("vault_path", VAULT_PATH);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null) url.searchParams.set(k, String(v));
  }
  const r = await fetch(url.toString());
  if (!r.ok) throw new Error(`API error ${r.status}: ${await r.text()}`);
  return r.json();
}

// ───────────────────────────────────────────────────────────
// Minimal markdown renderer
// Splits on "## " headings; treats content as paragraphs with RichText $...$

function renderMarkdown(body) {
  if (!body) return [];
  const lines = body.split("\n");
  const sections = [];
  let current = null;

  for (const line of lines) {
    if (line.startsWith("## ")) {
      if (current) sections.push(current);
      current = { heading: line.slice(3).trim(), paras: [] };
    } else if (line.startsWith("### ")) {
      if (!current) current = { heading: null, paras: [] };
      current.paras.push({ type: "h3", text: line.slice(4).trim() });
    } else if (line.trim()) {
      if (!current) current = { heading: null, paras: [] };
      current.paras.push({ type: "p", text: line.trim() });
    }
  }
  if (current) sections.push(current);
  return sections;
}

// ───────────────────────────────────────────────────────────
// MATH — KaTeX-backed

function TeX({ tex, display = false }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    const render = () => {
      if (!window.katex) { setTimeout(render, 30); return; }
      try {
        window.katex.render(tex, ref.current, {
          displayMode: display,
          throwOnError: false,
          strict: "ignore",
          output: "html"
        });
      } catch (e) {
        ref.current.textContent = tex;
      }
    };
    render();
  }, [tex, display]);
  return <span ref={ref} className={display ? "tex-display" : "tex-inline"} />;
}

// Display equation block — rule bar + (optional) tag.
function Equation({ tex, tag, small }) {
  return (
    <div className={"eq-block" + (small ? " eq-small" : "")}>
      <span className="eq-rule"></span>
      <div className="eq-body">
        <TeX tex={tex} display />
      </div>
      {tag && <span className="eq-tag">({tag})</span>}
    </div>
  );
}

// Renders a CJK string with `$...$` segments rendered as inline KaTeX.
function RichText({ children }) {
  const str = String(children);
  const out = [];
  let i = 0;
  let key = 0;
  while (i < str.length) {
    if (str[i] === "$") {
      const end = str.indexOf("$", i + 1);
      if (end > i) {
        out.push(<TeX key={key++} tex={str.slice(i + 1, end)} />);
        i = end + 1;
        continue;
      }
    }
    const next = str.indexOf("$", i);
    const stop = next === -1 ? str.length : next;
    out.push(<React.Fragment key={key++}>{str.slice(i, stop)}</React.Fragment>);
    i = stop;
  }
  return <>{out}</>;
}

// ───────────────────────────────────────────────────────────
// TOPBAR

function TopBar({ cost, busy, dark, setDark }) {
  return (
    <header className="topbar">
      <div className="brand">paper—<em>distiller</em></div>
      <div className="topbar-right">
        <div className={"cost-chip" + (busy ? " busy" : "")}>
          <span className="dot"></span>
          <span>{busy ? "running" : "ready"}</span>
          <span style={{ opacity: 0.35 }}>·</span>
          <span>¥{cost.toFixed(2)}</span>
        </div>
        <button className="icon-btn" onClick={() => setDark(d => !d)} title="theme">
          {dark ? "☀" : "◐"}
        </button>
      </div>
    </header>
  );
}

// ───────────────────────────────────────────────────────────
// CHAT — empty state with seed prompts

function EmptyChat({ onSeed }) {
  return (
    <div className="empty">
      <h1>What are we <em>reading</em> today?</h1>
      <p>用自然语言告诉它你想读什么。它会去 arXiv 找论文、写中文笔记、建证明图,并把所有材料沉淀到本地 vault。</p>
      <div className="seeds">
        {SEED_PROMPTS.map((p, i) => (
          <button key={i} className="seed" onClick={() => onSeed(p.text)}>
            <span className="seed-tag">{p.tag}</span>
            {p.text}
          </button>
        ))}
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// MESSAGES

function UserMsg({ text }) {
  return <div className="msg-user">{text}</div>;
}

function AsstMsg({ text, streaming, children }) {
  return (
    <div className="msg-asst">
      {typeof text === "string" ? <RichText>{text}</RichText> : text}
      {streaming && <span className="caret"></span>}
      {children}
    </div>
  );
}

function ToolCard({ name, status, args, results, onPick }) {
  const statusText = {
    running: "running",
    done: results ? `done · ${results.length} 条` : "done"
  }[status] || status;
  return (
    <div className="tool">
      <div className="tool-head">
        <div className="tool-name">{name}</div>
        <div className={"tool-status " + status}>{statusText}</div>
      </div>
      {args && (
        <div className="tool-body">
          {Object.entries(args).map(([k, v]) => (
            <div key={k} className="tool-arg">
              <span className="k">{k}</span>
              <span className="v">{typeof v === "object" ? JSON.stringify(v) : String(v)}</span>
            </div>
          ))}
        </div>
      )}
      {results && (
        <div className="tool-result">
          {results.map((r, i) => (
            <div
              key={i}
              className={"tool-result-row" + (onPick ? " clickable" : "")}
              onClick={onPick ? () => onPick(r) : undefined}
            >
              <span className="idx">{r.idx || String(i + 1)}</span>
              <span>{r.title}</span>
              <span className="meta">{r.meta}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// INPUT

function InputBox({ value, setValue, onSend, disabled }) {
  const ref = useRef(null);
  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && value.trim()) onSend();
    }
  };
  useEffect(() => {
    if (ref.current) {
      ref.current.style.height = "auto";
      ref.current.style.height = Math.min(ref.current.scrollHeight, 160) + "px";
    }
  }, [value]);
  return (
    <div className="input-wrap">
      <div className="input-box">
        <textarea
          ref={ref}
          rows={1}
          value={value}
          placeholder={disabled ? "agent 正在工作中…" : "键入消息,或按 / 触发斜杠命令…"}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKey}
          disabled={disabled}
        />
        <button
          className="input-send"
          onClick={onSend}
          disabled={disabled || !value.trim()}
          title="send"
        >↗</button>
      </div>
      <div className="input-meta">
        <span>⏎ 发送 · ⇧⏎ 换行 · / 命令</span>
        <span>vault: {VAULT_PATH || "(none)"}</span>
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// WORKSPACE — welcome view (wired to /vault/stats + /vault/recent)

function WelcomeView({ onPick, onOpenArticle }) {
  const [stats, setStats] = useState(null);
  const [recent, setRecent] = useState([]);

  useEffect(() => {
    apiFetch("/vault/stats").then(setStats).catch(() => setStats({}));
    apiFetch("/vault/recent", { limit: 10 }).then(d => setRecent(d.recent || [])).catch(() => {});
  }, []);

  const fmtNum = (n) => {
    if (n === undefined || n === null) return "—";
    if (n >= 1000) return (n / 1000).toFixed(1) + "k";
    return String(n);
  };

  return (
    <div className="welcome-wrap">
      <div className="greeting">YOUR VAULT · {VAULT_PATH || "~/research"}</div>
      <h2>欢迎回来。<br /><em>已经读过</em>这些论文。</h2>
      <p className="lead">左边问点什么开始,或者直接打开一篇最近的笔记。所有内容都存在本地 — 可以用 Obsidian 打开。</p>

      <div className="stat-grid">
        <div className="stat">
          <div className="stat-k">articles</div>
          <div className="stat-v"><em>{stats ? fmtNum(stats.articles) : "…"}</em></div>
        </div>
        <div className="stat">
          <div className="stat-k">surveys</div>
          <div className="stat-v"><em>{stats ? fmtNum(stats.surveys) : "…"}</em></div>
        </div>
        <div className="stat">
          <div className="stat-k">proof nodes</div>
          <div className="stat-v"><em>{stats ? fmtNum(stats.proof_nodes) : "…"}</em></div>
        </div>
      </div>

      <div className="recent-h">Recent</div>
      <div className="recent-list">
        {recent.length === 0 && <div style={{ color: "var(--ink-3)", fontSize: 13 }}>暂无最近文章 — 先蒸馏几篇试试</div>}
        {recent.map((r, i) => (
          <div key={i} className="recent-item" onClick={() => onOpenArticle && onOpenArticle(r.slug, r.category)}>
            <span className="recent-title">{r.title}</span>
            <span className="recent-meta">{r.arxiv_id}{r.updated ? " · " + r.updated.slice(0, 10) : ""}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// WORKSPACE — article view (wired to /vault/article/{category}/{slug})

function ArticleView({ slug, category, articleFlash, onOpenGraph, onOpenPaper }) {
  const [article, setArticle] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!slug) return;
    setArticle(null);
    setError(null);
    apiFetch(`/vault/article/${category || "articles"}/${slug}`)
      .then(setArticle)
      .catch(e => setError(String(e)));
  }, [slug, category]);

  if (!slug) return <div className="article-wrap"><p style={{ padding: 32 }}>打开一篇文章查看详情。</p></div>;
  if (error) return <div className="article-wrap"><p style={{ padding: 32, color: "var(--accent)" }}>加载失败: {error}</p></div>;
  if (!article) return <div className="article-wrap"><p style={{ padding: 32 }}>加载中…</p></div>;

  const tags = Array.isArray(article.tags) ? article.tags : [];
  const sections = renderMarkdown(article.body);
  const ps = article.proof_stats || {};

  return (
    <div className="article-wrap">
      <div className="article-eyebrow">
        <span>ARTICLE · DISTILLED</span>
        <span className="arxiv">{article.arxiv_id}</span>
        {article.arxiv_id && (
          <a
            href={`https://arxiv.org/abs/${article.arxiv_id}`}
            target="_blank"
            rel="noreferrer"
            className="eyebrow-cta"
            style={{ textDecoration: "none" }}
          >
            <span>↗</span> view original
          </a>
        )}
      </div>
      <h1>{article.title}</h1>
      {article.frontmatter && (
        <p className="article-authors">
          {[article.frontmatter.authors, article.frontmatter.venue].filter(Boolean).join(" · ")}
        </p>
      )}
      <div className="article-tags">
        {tags.map(t => <span key={t} className="tag">{t}</span>)}
      </div>

      {sections.map((sec, i) => (
        <div key={i} data-section={sec.heading} className={"article-section" + (articleFlash === sec.heading ? " sec-flash" : "")}>
          {sec.heading && <h3><span>{sec.heading}</span></h3>}
          {sec.paras.map((p, j) => {
            if (p.type === "h3") return <h4 key={j}>{p.text}</h4>;
            return <p key={j}><RichText>{p.text}</RichText></p>;
          })}
        </div>
      ))}

      {(ps.nodes > 0) && (
        <div className="notice" style={{ marginTop: 32 }}>
          <span className="nicon">⌗</span>
          <p>
            这篇有 <span style={{ color: "var(--accent)" }}>{ps.nodes} 个证明节点</span>
            {ps.suspicious > 0 && (
              <span>,其中 <span style={{ color: "var(--warn)" }}>{ps.suspicious} 个 review 时被标可疑</span></span>
            )}
            。
            {article.arxiv_id && (
              <span className="wiki" onClick={() => onOpenGraph && onOpenGraph(article.arxiv_id)}
                style={{ color: "var(--accent-2)", borderBottom: "1px solid var(--accent-2)", cursor: "pointer", marginLeft: 6 }}>
                打开证明图谱 →
              </span>
            )}
          </p>
        </div>
      )}
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// PAPER VIEW — Phase 2 placeholder (PDF rendering deferred)

function PaperView({ arxivId }) {
  const id = arxivId || "";
  return (
    <div className="paper-wrap">
      <div className="paper-toolbar">
        <div className="paper-tb-left">
          <span className="paper-pageinfo"><i>PDF 视图</i></span>
        </div>
        <div className="paper-tb-right">
          {id && <span className="paper-tb-src">arxiv.org / {id}</span>}
        </div>
      </div>
      <div className="paper-scroll" style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: 320 }}>
        <div style={{ textAlign: "center", padding: 40, color: "var(--ink-3)" }}>
          <div style={{ fontSize: 32, marginBottom: 12 }}>📄</div>
          <p style={{ fontFamily: "var(--serif)", fontSize: 15, marginBottom: 12 }}>
            PDF 视图正在 Phase 2 中,先看 arXiv 原文 ↗
          </p>
          {id && (
            <a
              href={`https://arxiv.org/abs/${id}`}
              target="_blank"
              rel="noreferrer"
              style={{ color: "var(--accent)", fontFamily: "var(--mono)", fontSize: 13 }}
            >
              arxiv.org/abs/{id}
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// GRAPH VIEW (wired to /vault/graph/{paper_arxiv_id})

const _KIND_ROW = {
  assumption: 0, definition: 0, axiom: 0,
  lemma: 1, step: 1, claim: 1,
  theorem: 2, proposition: 2, corollary: 2
};

function GraphView({ arxivId, onJumpArticle }) {
  const [graphData, setGraphData] = useState(null);
  const [error, setError] = useState(null);
  const [focusId, setFocusId] = useState(null);
  const [verifyOpen, setVerifyOpen] = useState(false);

  useEffect(() => {
    if (!arxivId) return;
    setGraphData(null);
    setError(null);
    apiFetch(`/vault/graph/${arxivId}`)
      .then(setGraphData)
      .catch(e => setError(String(e)));
  }, [arxivId]);

  if (!arxivId) return <div className="graph-wrap"><p style={{ padding: 32 }}>打开一篇文章的证明图谱。</p></div>;
  if (error) return <div className="graph-wrap"><p style={{ padding: 32, color: "var(--accent)" }}>加载图谱失败: {error}</p></div>;
  if (!graphData) return <div className="graph-wrap"><p style={{ padding: 32 }}>加载中…</p></div>;

  const nodes = graphData.nodes || [];
  const edges = graphData.edges || [];
  const stats = graphData.stats || {};

  if (nodes.length === 0) {
    return <div className="graph-wrap"><p style={{ padding: 32 }}>该论文暂无证明图谱节点。先用 review_proof 跑一遍。</p></div>;
  }

  // Build node center map for edge rendering
  const nodeCenters = {};
  nodes.forEach(n => {
    nodeCenters[n.id] = { x: (n.x || 0) + 75, y: (n.y || 0) + 28 };
  });

  const focused = nodes.find(n => n.id === focusId);
  const totalNodes = nodes.length;
  const suspectCount = (stats.by_status || {}).suspicious || 0;

  const _statusCls = (s) => {
    if (s === "suspicious") return "suspect";
    if (s === "gap") return "suspect";
    if (s === "ok") return "";
    return "";
  };

  return (
    <div className="graph-wrap">
      <div className="graph-toolbar">
        <span className="label">filter</span>
        <span className="pill on">all</span>
        <span className="pill">theorem</span>
        <span className="pill">lemma</span>
        <span className="pill">step</span>
        {suspectCount > 0 && <span className="pill" style={{ color: "var(--accent)" }}>suspect ({suspectCount})</span>}
        <span style={{ flex: 1 }}></span>
        <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--ink-3)" }}>
          {totalNodes} nodes · {arxivId}
        </span>
      </div>

      <div className="graph-canvas">
        <svg className="edges">
          <defs>
            <marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" fill="currentColor" />
            </marker>
          </defs>
          {edges.map((e, i) => {
            const sa = nodeCenters[e.src_id];
            const sb = nodeCenters[e.dst_id];
            if (!sa || !sb) return null;
            const isDash = e.cross_paper > 0;
            return (
              <line
                key={i}
                x1={sa.x} y1={sa.y} x2={sb.x} y2={sb.y}
                stroke={isDash ? "var(--accent)" : "var(--ink)"}
                strokeWidth={1.5}
                strokeDasharray={isDash ? "4 4" : null}
                markerEnd="url(#ah)"
                color={isDash ? "var(--accent)" : "var(--ink)"}
              />
            );
          })}
        </svg>

        {nodes.map(n => (
          <div
            key={n.id}
            className={"gnode " + _statusCls(n.status) + (focusId === n.id ? " focused" : "")}
            style={{ left: n.x || 0, top: n.y || 0 }}
            onClick={() => setFocusId(n.id)}
          >
            <span className="kind">{n.kind}</span>
            <span>{n.label || n.text.slice(0, 40)}</span>
          </div>
        ))}

        {focused && (
          <div className="graph-detail">
            <div className="gd-kind">{focused.kind} · {focused.status}</div>
            <h4 className="gd-title">{focused.label || focused.text.slice(0, 60)}</h4>
            {focused.text && <p className="gd-text">{focused.text.slice(0, 200)}</p>}
            {focused.source_quote && (
              <div className="gd-quote">"{focused.source_quote.slice(0, 200)}"</div>
            )}
            {focused.techniques && focused.techniques.length > 0 && (
              <div style={{ fontSize: 12, color: "var(--ink-3)", marginTop: 6 }}>
                技术: {focused.techniques.join(", ")}
              </div>
            )}
            <div className="gd-actions">
              {focused.status === "suspicious" && (
                <button className="gd-verify" onClick={() => setVerifyOpen(true)}>⌗ open verification</button>
              )}
              <button className="gd-jump" onClick={() => onJumpArticle && onJumpArticle()}>↗ jump to article</button>
            </div>
          </div>
        )}

        {verifyOpen && focused && (
          <VerificationPanel
            node={focused}
            onClose={() => setVerifyOpen(false)}
            onJumpArticle={onJumpArticle}
          />
        )}
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// VERIFICATION PANEL — manual review workspace (kept as-is per plan)

function VerificationPanel({ node, onClose, onJumpArticle }) {
  const [verdict, setVerdict] = useState(null);
  const [step, setStep] = useState(0);
  const STEPS = [
    {
      title: "Re-state the assumption",
      body: <>The bound assumes <TeX tex="Q_i, K_i" /> are <em>independent</em> with mean 0 and variance 1. In practice they are produced by learned projections — independence is at best approximate.</>,
      check: "assume i.i.d. unit-variance"
    },
    {
      title: "Re-derive the variance",
      body: <><TeX tex="\mathrm{Var}(QK^{\top}) = \sum_{i=1}^{d_k} \mathrm{Var}(Q_i K_i) = d_k" display />Stable. The variance step itself is sound.</>,
      check: "variance computation"
    },
    {
      title: "Inspect the tail bound",
      body: <>The paper appeals to Hoeffding-style concentration to claim <TeX tex="QK^{\top}" /> is sub-Gaussian. <strong>This needs boundedness or strict sub-Gaussianity</strong> — for learned projections only an asymptotic statement holds. Confidence cap: <TeX tex="\le 0.7" />.</>,
      check: "concentration bound",
      warn: true
    },
    {
      title: "Try a counter-example",
      body: <>Heavy-tailed inputs (e.g. <TeX tex="Q_i \sim t_{2.5}" />, infinite kurtosis) violate sub-Gaussian; empirically softmax still concentrates because of post-LayerNorm. Bound is <em>practically tight</em>, theoretically <em>looser than claimed</em>.</>,
      check: "counter-example"
    }
  ];
  const cur = STEPS[step];
  return (
    <div className="vrf-overlay" onClick={onClose}>
      <div className="vrf-panel" onClick={(e) => e.stopPropagation()}>
        <div className="vrf-head">
          <div>
            <div className="vrf-eyebrow">VERIFICATION · {node?.kind?.toUpperCase() || "NODE"}</div>
            <h3 className="vrf-title">{node?.label || node?.text?.slice(0, 60) || "node"}</h3>
          </div>
          <button className="vrf-close" onClick={onClose}>×</button>
        </div>

        <div className="vrf-stepper">
          {STEPS.map((s, i) => (
            <button
              key={i}
              className={"vrf-pip" + (i === step ? " cur" : "") + (i < step ? " past" : "")}
              onClick={() => setStep(i)}
            >
              <span className="vrf-pip-n">{String(i + 1).padStart(2, "0")}</span>
              <span className="vrf-pip-l">{s.check}</span>
            </button>
          ))}
        </div>

        <div className={"vrf-step" + (cur.warn ? " warn" : "")}>
          <div className="vrf-step-num">step {step + 1} of {STEPS.length}</div>
          <h4 className="vrf-step-title">{cur.title}</h4>
          <div className="vrf-step-body">{cur.body}</div>
        </div>

        <div className="vrf-foot">
          <button
            className="vrf-nav"
            onClick={() => setStep(s => Math.max(0, s - 1))}
            disabled={step === 0}
          >‹ prev</button>
          {step < STEPS.length - 1 ? (
            <button className="vrf-nav next" onClick={() => setStep(s => s + 1)}>next ›</button>
          ) : (
            <div className="vrf-verdict">
              <button
                className={"vrf-btn ok" + (verdict === "ok" ? " on" : "")}
                onClick={() => setVerdict("ok")}
              >✓ accept</button>
              <button
                className={"vrf-btn still" + (verdict === "still" ? " on" : "")}
                onClick={() => setVerdict("still")}
              >⚑ still suspect</button>
              <button
                className="vrf-jump"
                onClick={() => { onJumpArticle && onJumpArticle(); onClose(); }}
              >↗ go to article</button>
            </div>
          )}
        </div>

        {verdict && (
          <div className={"vrf-banner " + verdict}>
            {verdict === "ok"
              ? "Marked verified. Confidence raised to 0.95."
              : "Marked still-suspect. Confidence held at 0.7. Added to review queue."}
          </div>
        )}
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// DASHBOARD — Phase 1b placeholder (keeps demo counters, marked MOCK)

function DashboardView() {
  const [t, setT] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setT(x => x + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const phase = (t % 120) / 120;
  const elapsedMin = 194 + Math.floor(t / 6);
  const elapsedH = Math.floor(elapsedMin / 60);
  const elapsedM = elapsedMin % 60;
  const elapsedPct = Math.min(99, 54 + phase * 6).toFixed(0);
  const spent = 2.18 + phase * 0.42;
  const spentPct = Math.min(99, 22 + phase * 4).toFixed(0);
  const papers = 14 + Math.floor(phase * 3);
  const papersPct = Math.min(99, 47 + phase * 10).toFixed(0);
  const coverage = phase > 0.6 ? 4 : 3;
  const coveragePct = phase > 0.6 ? 80 : 60;
  const stage3Pct = Math.min(98, 50 + phase * 48).toFixed(0);
  const stage3Spent = (0.54 + phase * 0.18).toFixed(2);
  const stage3Cur = Math.min(4, 2 + Math.floor(phase * 3));

  return (
    <div className="dash-wrap">
      <div className="dash-head">
        <div className="label" style={{ display: "flex", alignItems: "center", gap: 8 }}>
          RESEARCH SESSION · S_1419
          <span style={{ background: "var(--accent)", color: "#fff", borderRadius: 4, padding: "2px 7px", fontSize: 10, fontFamily: "var(--mono)", letterSpacing: 1 }}>MOCK</span>
        </div>
        <p style={{ color: "var(--ink-3)", fontFamily: "var(--mono)", fontSize: 12, marginTop: 4, marginBottom: 8 }}>
          深度研究仪表盘 Phase 1b 接入 — 真实 research session 数据将在 Phase 1b 中对接。当前数据为演示占位。
        </p>
        <h2>"在 Transformer 之后,有哪些路线能把 attention 的 O(n²) 复杂度降到亚二次,而又不显著损失质量?"</h2>
      </div>

      <div className="kpis">
        <div className="kpi">
          <div className="k">elapsed</div>
          <div className="v">{elapsedH}h <em>{String(elapsedM).padStart(2, "0")}m</em></div>
          <div className="sub">of 6h budget</div>
          <div className="bar" style={{ "--p": elapsedPct + "%" }}></div>
        </div>
        <div className="kpi">
          <div className="k">spent</div>
          <div className="v">¥ <em>{spent.toFixed(2)}</em></div>
          <div className="sub">of ¥10.00</div>
          <div className="bar" style={{ "--p": spentPct + "%" }}></div>
        </div>
        <div className="kpi">
          <div className="k">papers</div>
          <div className="v"><em>{papers}</em> / 30</div>
          <div className="sub">distilled · graph_depth=step</div>
          <div className="bar" style={{ "--p": papersPct + "%" }}></div>
        </div>
        <div className="kpi">
          <div className="k">coverage</div>
          <div className="v"><em>{coverage}</em> / 5</div>
          <div className="sub">themes converged</div>
          <div className="bar" style={{ "--p": coveragePct + "%" }}></div>
        </div>
      </div>

      <div className="pipeline">
        <div className="stage done">
          <span className="stage-num">1 · SEARCH</span>
          <span className="stage-name">42 papers</span>
          <span className="stage-stat">¥0.02</span>
        </div>
        <div className="stage done">
          <span className="stage-num">2 · DISTILL</span>
          <span className="stage-name">14 notes</span>
          <span className="stage-stat">¥1.62</span>
        </div>
        <div className="stage running">
          <span className="stage-num">3 · EXPAND</span>
          <span className="stage-name">{stage3Cur} of 4</span>
          <span className="stage-stat">
            <span className="stage-spin"></span>
            running · ¥{stage3Spent}
          </span>
          <div className="stage-progress" style={{ "--p": stage3Pct + "%" }}></div>
        </div>
        <div className="stage queued">
          <span className="stage-num">4 · CLUSTER</span>
          <span className="stage-name">queued</span>
          <span className="stage-stat">—</span>
        </div>
        <div className="stage queued">
          <span className="stage-num">5 · SYNTH</span>
          <span className="stage-name">queued</span>
          <span className="stage-stat">—</span>
        </div>
      </div>

      <div className="live-log">
        <div className="live-log-head">
          <span>● live</span> expanding paper {stage3Cur}/4 — <em>Linformer</em>
        </div>
        <div className="live-log-line">
          [{String(elapsedH).padStart(2, "0")}:{String(elapsedM).padStart(2, "0")}:{String(t % 60).padStart(2, "0")}] tracing references → found {3 + (t % 5)} new arxiv ids
        </div>
      </div>

      <div className="themes">
        <div className="theme">
          <div className="theme-tag">THEME A</div>
          <h4><em>Sparse</em></h4>
          <p>滑动窗 · 全局+随机 · 块对角</p>
          <div className="papers">Longformer · BigBird · Sparse-T</div>
        </div>
        <div className="theme">
          <div className="theme-tag">THEME B</div>
          <h4><em>Low-rank</em></h4>
          <p>投影到 k 维 · 核函数化</p>
          <div className="papers">Linformer · Performer</div>
        </div>
        <div className="theme">
          <div className="theme-tag">THEME C</div>
          <h4><em>State-space</em></h4>
          <p>替代路线 · 非 attention</p>
          <div className="papers">S4 · Mamba</div>
        </div>
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────
// SSE CHAT STREAM — drives the real agent loop

async function* readSSEStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop(); // keep incomplete last line
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const payload = line.slice(6).trim();
        if (payload && payload !== "[DONE]") {
          try { yield JSON.parse(payload); } catch (_) {}
        }
      }
    }
  }
}

// ───────────────────────────────────────────────────────────
// MAIN APP

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [cost, setCost] = useState(0);
  const [tab, setTab] = useState("welcome");
  const [currentSlug, setCurrentSlug] = useState(null);
  const [currentCategory, setCurrentCategory] = useState("articles");
  const [currentArxivId, setCurrentArxivId] = useState(null);
  const [hasDashboard, setHasDashboard] = useState(false);
  const [articleFlash, setArticleFlash] = useState(null);
  const [history, setHistory] = useState([]);  // conversation history (stateless: sent with each turn)
  const [dark, setDark] = useState(() => localStorage.getItem("pd-theme") === "dark");
  const feedRef = useRef(null);
  const lastSearchTopicRef = useRef("");

  useEffect(() => {
    document.body.classList.toggle("dark", dark);
    localStorage.setItem("pd-theme", dark ? "dark" : "light");
  }, [dark]);

  useEffect(() => {
    if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [messages]);

  const addMsg = (m) => setMessages(prev => [...prev, { ...m, id: m.id || uid() }]);
  const updateMsg = (id, patch) => setMessages(prev =>
    prev.map(m => m.id === id ? { ...m, ...(typeof patch === "function" ? patch(m) : patch) } : m)
  );

  const openArticle = (slug, cat = "articles", arxivId = null) => {
    setCurrentSlug(slug);
    setCurrentCategory(cat || "articles");
    if (arxivId) setCurrentArxivId(arxivId);
    setTab("article");
  };

  const openGraph = (arxivId) => {
    if (arxivId) setCurrentArxivId(arxivId);
    setTab("graph");
  };

  // Pick a paper from search results — synthesize a chat message to distill it
  const pickPaper = (paper) => {
    if (busy) return;
    const topic = lastSearchTopicRef.current || paper.title || "";
    const msg = `蒸馏这篇: ${paper.arxiv || paper.id || ""} (关于 ${topic})`;
    sendText(msg);
  };

  // Main sendText — posts to /chat/stream and drives SSE into messages state
  const sendText = async (text) => {
    if (busy) return;
    if (!text.trim()) return;

    addMsg({ role: "user", text });
    setInput("");
    setBusy(true);

    // Build the streaming assistant message
    const asstId = uid();
    addMsg({ id: asstId, role: "asst", text: "", streaming: true });

    const toolMsgIds = {};  // tool_call_id -> message id

    try {
      const resp = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history, vault_path: VAULT_PATH }),
      });

      if (!resp.ok) {
        updateMsg(asstId, { text: `请求失败 (${resp.status})`, streaming: false });
        setBusy(false);
        return;
      }

      for await (const event of readSSEStream(resp)) {
        switch (event.type) {
          case "text":
            updateMsg(asstId, prev => ({ text: (prev.text || "") + event.delta, streaming: true }));
            break;

          case "tool_call_start": {
            const tmId = uid();
            toolMsgIds[event.id] = tmId;
            // Extract search topic for pickPaper
            if (event.name === "search" && event.args && event.args.topic) {
              lastSearchTopicRef.current = event.args.topic;
            }
            addMsg({ id: tmId, type: "tool", name: event.name, status: "running", args: event.args, results: null });
            break;
          }

          case "tool_call_done": {
            const tmId = toolMsgIds[event.id];
            if (tmId) {
              // Format search results for ToolCard display
              let results = null;
              if (event.result && event.result.candidates) {
                results = event.result.candidates.map((c, i) => ({
                  idx: ["①","②","③","④","⑤","⑥","⑦","⑧","⑨","⑩"][i] || String(i + 1),
                  title: c.title || c.id || "",
                  meta: [c.authors && c.authors.slice(0,2).join(", "), c.year].filter(Boolean).join(" · "),
                  arxiv: c.id || "",
                  id: c.id || "",
                }));
              }
              // For distill_by_id: if articles were distilled, open the first one
              if (event.result && event.result.distilled && event.result.distilled.length > 0) {
                const first = event.result.distilled[0];
                openArticle(first.slug, first.category || "articles");
              }
              updateMsg(tmId, { status: "done", results });
            }
            break;
          }

          case "cost":
            setCost(prev => prev + (event.cny || 0));
            break;

          case "done":
            setHistory(event.history || []);
            updateMsg(asstId, { streaming: false });
            // If research tool was used, show dashboard
            if (event.history && event.history.some(m =>
              m.role === "tool" && m.content && m.content.includes("research_session_id")
            )) {
              setHasDashboard(true);
              setTab("dashboard");
            }
            break;

          case "error":
            updateMsg(asstId, prev => ({ text: (prev.text ? prev.text + "\n" : "") + `错误: ${event.message}`, streaming: false }));
            break;

          default:
            break;
        }
      }
    } catch (err) {
      updateMsg(asstId, { text: `连接错误: ${err.message}`, streaming: false });
    }

    updateMsg(asstId, prev => ({ ...prev, streaming: false }));
    setBusy(false);
  };

  const onSend = () => {
    if (input.trim()) sendText(input.trim());
  };
  const onSeed = (text) => sendText(text);

  const openArticleOpen = currentSlug !== null;

  return (
    <div className="app">
      <TopBar cost={cost} busy={busy} dark={dark} setDark={setDark} />
      <div className="body">
        {/* CHAT */}
        <section className="chat">
          <div className="chat-feed" ref={feedRef}>
            {messages.length === 0 ? (
              <EmptyChat onSeed={onSeed} />
            ) : (
              messages.map(m => {
                if (m.role === "user") return <UserMsg key={m.id} text={m.text} />;
                if (m.role === "asst") return <AsstMsg key={m.id} text={m.text} streaming={m.streaming} />;
                if (m.type === "tool") return (
                  <ToolCard
                    key={m.id}
                    name={m.name}
                    status={m.status}
                    args={m.args}
                    results={m.results}
                    onPick={m.name === "search" && m.status === "done" ? pickPaper : null}
                  />
                );
                return null;
              })
            )}
          </div>
          <InputBox value={input} setValue={setInput} onSend={onSend} disabled={busy} />
        </section>

        {/* WORKSPACE */}
        <section className="work">
          <div className="work-tabs">
            <button className={"work-tab" + (tab === "welcome" ? " active" : "")} onClick={() => setTab("welcome")}>
              Welcome
            </button>
            <button
              className={"work-tab" + (tab === "article" ? " active" : "")}
              onClick={() => openArticleOpen && setTab("article")}
              disabled={!openArticleOpen}
            >
              Article{openArticleOpen && currentSlug && <span className="badge">{currentSlug.slice(0, 12)}</span>}
            </button>
            <button
              className={"work-tab" + (tab === "graph" ? " active" : "")}
              onClick={() => (openArticleOpen || currentArxivId) && setTab("graph")}
              disabled={!openArticleOpen && !currentArxivId}
            >
              Graph{currentArxivId && <span className="badge">{currentArxivId}</span>}
            </button>
            <button
              className={"work-tab" + (tab === "paper" ? " active" : "")}
              onClick={() => openArticleOpen && setTab("paper")}
              disabled={!openArticleOpen}
            >
              Paper
            </button>
            <button
              className={"work-tab" + (tab === "dashboard" ? " active" : "")}
              onClick={() => hasDashboard && setTab("dashboard")}
              disabled={!hasDashboard}
            >
              Research{hasDashboard && <span className="badge">mock</span>}
            </button>
          </div>
          <div className="work-body">
            {tab === "welcome" && (
              <WelcomeView
                onPick={() => openArticleOpen && setTab("article")}
                onOpenArticle={(slug, cat) => openArticle(slug, cat)}
              />
            )}
            {tab === "article" && (
              <ArticleView
                slug={currentSlug}
                category={currentCategory}
                articleFlash={articleFlash}
                onOpenGraph={(arxivId) => openGraph(arxivId)}
                onOpenPaper={() => setTab("paper")}
              />
            )}
            {tab === "graph" && (
              <GraphView
                arxivId={currentArxivId}
                onJumpArticle={() => currentSlug && setTab("article")}
              />
            )}
            {tab === "paper" && (
              <PaperView arxivId={currentArxivId} />
            )}
            {tab === "dashboard" && hasDashboard && <DashboardView />}
          </div>
        </section>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
