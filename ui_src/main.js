import {
  App,
  applyDocumentTheme,
  applyHostStyleVariables,
} from "@modelcontextprotocol/ext-apps";

const state = {
  items: [],
  selectedIndex: -1,
  generated: null,
  hostCaps: {},
};

const statusEl = document.getElementById("host-status");
const queryInput = document.getElementById("query-input");
const minScoreEl = document.getElementById("min-score");
const topKEl = document.getElementById("top-k");
const searchBtn = document.getElementById("search-btn");
const resultsList = document.getElementById("results-list");
const resultsCount = document.getElementById("results-count");
const inspectorTitle = document.getElementById("inspector-title");
const syntaxBadge = document.getElementById("syntax-badge");
const inspectorBody = document.getElementById("inspector-body");

const app = new App({ name: "sand3d-rag-explorer", version: "1.2.0" }, {});

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function highlightPython(code) {
  const escaped = escapeHtml(code || "");
  const tokenRe =
    /(#[^\n]*)|(&quot;(?:\\.|[^&]|&(?!quot;))*?&quot;|&#39;.*?&#39;|'(?:\\.|[^'\\])*')|\b(import|from|as|def|return|for|in|if|elif|else|with|try|except|raise|True|False|None|and|or|not|pass|lambda)\b|\b(s3d\.[A-Za-z_]\w*)\b|\b(\d+(?:\.\d+)?)\b/g;

  return escaped.replace(tokenRe, (m, comment, str, kw, s3dCall, num) => {
    if (comment) return `<span class="tok-com">${comment}</span>`;
    if (str) return `<span class="tok-str">${str}</span>`;
    if (kw) return `<span class="tok-kw">${kw}</span>`;
    if (s3dCall) return `<span class="tok-api">${s3dCall}</span>`;
    if (num) return `<span class="tok-num">${num}</span>`;
    return m;
  });
}

function applyHostContext(ctx) {
  if (!ctx) return;
  if (ctx.theme) {
    applyDocumentTheme(ctx.theme);
    if (ctx.theme === "light") {
      document.documentElement.setAttribute("data-theme", "light");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
  }
  if (ctx.styles && ctx.styles.variables) {
    applyHostStyleVariables(ctx.styles.variables);
  }
}

function setStatus(text, tone = "neutral") {
  statusEl.textContent = text;
  statusEl.dataset.tone = tone;
}

function renderResults() {
  resultsCount.textContent = `${state.items.length} assets`;
  if (!state.items.length) {
    resultsList.innerHTML = `<div class="empty-state">No matching 3D assets found. Try another Persian or English query.</div>`;
    return;
  }

  resultsList.innerHTML = state.items
    .map((item, idx) => {
      const active = idx === state.selectedIndex ? "active" : "";
      const thumb = item.thumbnail_data_uri
        ? `<img src="${item.thumbnail_data_uri}" alt="${escapeHtml(item.subject || "3D render")}" />`
        : `<div class="thumb-placeholder">${escapeHtml(item.family || "3D")}</div>`;
      const ranksText = item.ranks
        ? Object.entries(item.ranks)
            .map(([k, v]) => `${k}:#${v}`)
            .join(" · ")
        : "curated";
      const scoreBadge =
        item.rrf_score !== undefined && item.rrf_score !== null
          ? `<span class="tag tag-accent">RRF ${Number(item.rrf_score).toFixed(4)}</span>`
          : `<span class="tag tag-accent">score ${Number(item.semantic_score || 0).toFixed(2)}</span>`;

      return `
        <button type="button" class="card ${active}" data-idx="${idx}">
          <div class="thumb-wrap">${thumb}</div>
          <div class="card-body">
            <div class="card-meta">
              ${scoreBadge}
              <span class="tag">${escapeHtml(item.family || "")}</span>
            </div>
            <div class="card-prompt" dir="auto">${escapeHtml(item.prompt)}</div>
            <div class="card-ranks">${escapeHtml(ranksText)} · ${Number(item.triangles || 0).toLocaleString()} tris</div>
          </div>
        </button>
      `;
    })
    .join("");

  resultsList.querySelectorAll(".card").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const idx = Number(btn.getAttribute("data-idx"));
      state.selectedIndex = idx;
      state.generated = null;
      renderResults();
      renderInspector();
      const selected = state.items[idx];
      if (selected && !selected.program) {
        await fetchFullAssetDetails(selected);
      }
    });
  });
}

async function fetchFullAssetDetails(item) {
  setStatus(`LOADING #${item.id}...`);
  try {
    const res = await app.callServerTool({
      name: "app_get_asset_details",
      arguments: { point_id: item.id },
    });
    const sc = res && (res.structuredContent || res.structured_content);
    if (sc && sc.program) {
      item.program = sc.program;
      if (sc.thumbnail_data_uri) item.thumbnail_data_uri = sc.thumbnail_data_uri;
      renderInspector();
      renderResults();
    }
    setStatus("READY", "ok");
  } catch (err) {
    setStatus(`ERR: ${err.message || err}`, "err");
  }
}

function buildAgentPromptText(item) {
  const userNote = queryInput.value.trim();
  const taskLine = userNote
    ? `Create a new sand3d Python script for: "${userNote}", using reference asset #${item.id} (${item.entry}) below.`
    : `Write a new sand3d Python script inspired by reference asset #${item.id} (${item.entry}) below.`;
  return (
    `${taskLine}\n` +
    `Use \`get_sand3d_api_reference\` if you need extra API signatures, write the Python code yourself, ` +
    `and validate/save it with \`save_sand3d_program\`.\n\n` +
    `Prompt: ${item.prompt}\n` +
    `Taxonomy: ${item.domain} / ${item.family} / ${item.subject}\n\n` +
    "```python\n" +
    (item.program || "") +
    "\n```"
  );
}

function renderInspector() {
  if (state.generated) {
    const gen = state.generated;
    inspectorTitle.textContent = gen.saved_path ? `Saved: ${gen.saved_path}` : "Generated Script";
    syntaxBadge.innerHTML = gen.syntax_valid
      ? `<span class="pill pill-ok">AST VALID</span>`
      : `<span class="pill pill-err">SYNTAX ERROR</span>`;
    inspectorBody.innerHTML = `
      <div class="actions-row">
        <button type="button" class="btn btn-primary" id="copy-code-btn">Copy Python Code</button>
        <button type="button" class="btn" id="back-ref-btn">Back to Reference</button>
      </div>
      <pre class="code-box"><code>${highlightPython(gen.code || "")}</code></pre>
    `;
    document.getElementById("copy-code-btn").onclick = () => copyText(gen.code || "", "COPIED CODE");
    document.getElementById("back-ref-btn").onclick = () => {
      state.generated = null;
      renderInspector();
    };
    return;
  }

  const item = state.items[state.selectedIndex];
  if (!item) {
    inspectorTitle.textContent = "Program Inspector";
    syntaxBadge.textContent = "";
    inspectorBody.innerHTML = `<div class="empty-state">Select a 3D asset card on the left to inspect its procedural sand3d script or send it to your AI agent.</div>`;
    return;
  }

  const canMessage = Boolean(state.hostCaps && state.hostCaps.message);
  const canAttach = Boolean(state.hostCaps && state.hostCaps.updateModelContext);

  const primaryAction = canMessage
    ? `<button type="button" class="btn btn-primary" id="agent-remix-btn">Remix with Agent</button>`
    : `<button type="button" class="btn btn-primary" id="copy-agent-prompt-btn">Copy Prompt + Code for Agent</button>`;

  const attachBtn = canAttach
    ? `<button type="button" class="btn" id="attach-ctx-btn">Attach to Chat</button>`
    : "";

  inspectorTitle.textContent = item.entry || "sand3d program";
  syntaxBadge.innerHTML = `<span class="pill">#${item.id} · ${escapeHtml(item.split || "train")}</span>`;
  inspectorBody.innerHTML = `
    <div class="prompt-banner" dir="auto">${escapeHtml(item.prompt)}</div>
    <div class="metrics-strip">
      <div class="metric">
        <span class="metric-label">Domain</span>
        <span class="metric-val">${escapeHtml(item.domain)}</span>
      </div>
      <div class="metric">
        <span class="metric-label">Family</span>
        <span class="metric-val">${escapeHtml(item.family)}</span>
      </div>
      <div class="metric">
        <span class="metric-label">Subject</span>
        <span class="metric-val">${escapeHtml(item.subject)}</span>
      </div>
      <div class="metric">
        <span class="metric-label">Triangles</span>
        <span class="metric-val">${Number(item.triangles || 0).toLocaleString()}</span>
      </div>
    </div>
    <div class="actions-row">
      ${primaryAction}
      ${attachBtn}
      <button type="button" class="btn" id="copy-code-btn">Copy Code</button>
      <button type="button" class="btn btn-subtle" id="gen-ollama-btn" title="Optional offline draft using local Ollama">Local Ollama Draft</button>
    </div>
    <pre class="code-box"><code>${highlightPython(item.program || "# Loading full Python program...")}</code></pre>
  `;

  document.getElementById("copy-code-btn").onclick = () => copyText(item.program || "", "COPIED CODE");
  document.getElementById("gen-ollama-btn").onclick = () => runLocalOllamaDraft(item);

  if (canMessage) {
    document.getElementById("agent-remix-btn").onclick = () => sendToAgent(item);
  } else {
    document.getElementById("copy-agent-prompt-btn").onclick = () =>
      copyText(buildAgentPromptText(item), "COPIED PROMPT FOR AGENT");
  }
  if (canAttach) {
    document.getElementById("attach-ctx-btn").onclick = () => attachToChat(item);
  }
}

async function copyText(text, okLabel = "COPIED") {
  try {
    await navigator.clipboard.writeText(text);
    setStatus(okLabel, "ok");
  } catch {
    setStatus("COPY BLOCKED BY SANDBOX", "err");
  }
}

async function sendToAgent(item) {
  setStatus("SENDING TO AGENT...");
  try {
    await app.sendMessage({
      content: [{ type: "text", text: buildAgentPromptText(item) }],
    });
    setStatus("SENT TO AGENT", "ok");
  } catch (err) {
    setStatus(`SEND ERR: ${err.message || err}`, "err");
  }
}

async function attachToChat(item) {
  try {
    await app.updateModelContext({
      content: [{ type: "text", text: buildAgentPromptText(item) }],
    });
    setStatus("ATTACHED TO NEXT MESSAGE", "ok");
  } catch (err) {
    setStatus(`ATTACH ERR: ${err.message || err}`, "err");
  }
}

function applyToolResultPayload(result) {
  if (!result) return;
  const sc = result.structuredContent || result.structured_content;
  if (sc) {
    if (sc.query && !queryInput.value) {
      queryInput.value = sc.query;
    }
    if (Array.isArray(sc.results)) {
      state.items = sc.results;
      state.selectedIndex = state.items.length ? 0 : -1;
      state.generated = null;
      renderResults();
      renderInspector();
      return;
    }
    if (typeof sc.code === "string") {
      state.generated = sc;
      renderInspector();
      return;
    }
  }
  const firstText = result.content && result.content[0] && result.content[0].text;
  if (firstText) {
    try {
      const parsed = JSON.parse(firstText);
      if (Array.isArray(parsed)) {
        state.items = parsed;
        state.selectedIndex = parsed.length ? 0 : -1;
        renderResults();
        renderInspector();
      } else if (parsed && typeof parsed.code === "string") {
        state.generated = parsed;
        renderInspector();
      }
    } catch {
      // Plain non-JSON response
    }
  }
}

async function runSearch(e) {
  if (e) e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;
  searchBtn.disabled = true;
  setStatus("SEARCHING...");
  try {
    const res = await app.callServerTool({
      name: "search_sand3d_assets",
      arguments: {
        query,
        top_k: Number(topKEl.value),
        min_score: Number(minScoreEl.value),
        include_program: true,
      },
    });
    applyToolResultPayload(res);
    setStatus("READY", "ok");
  } catch (err) {
    setStatus(`ERR: ${err.message || err}`, "err");
  } finally {
    searchBtn.disabled = false;
  }
}

async function runLocalOllamaDraft(item) {
  const prompt = queryInput.value.trim() || item.prompt || "mossy rock";
  const btn = document.getElementById("gen-ollama-btn");
  if (btn) btn.disabled = true;
  setStatus("DRAFTING VIA OLLAMA...");
  try {
    const res = await app.callServerTool({
      name: "generate_sand3d_program",
      arguments: {
        prompt,
        top_k: 3,
        model: "gemma4:12b",
        output_filename: "ollama_draft.py",
      },
    });
    applyToolResultPayload(res);
    setStatus("OLLAMA DRAFT READY", "ok");
  } catch (err) {
    setStatus(`OLLAMA ERR: ${err.message || err}`, "err");
  } finally {
    if (btn) btn.disabled = false;
  }
}

// Register lifecycle handlers BEFORE connect() so no host notification is missed.
app.ontoolinput = ({ arguments: args }) => {
  if (args && args.query) {
    queryInput.value = args.query;
  }
  setStatus("RUNNING TOOL...");
};

app.ontoolresult = (result) => {
  applyToolResultPayload(result);
  setStatus("READY", "ok");
};

app.ontoolcancelled = () => {
  setStatus("CANCELLED");
};

app.onhostcontextchanged = (ctx) => {
  applyHostContext(ctx);
};

document.getElementById("search-form").addEventListener("submit", runSearch);

app
  .connect()
  .then(() => {
    state.hostCaps = app.getHostCapabilities() || {};
    applyHostContext(app.getHostContext());
    setStatus("CONNECTED", "ok");
    if (state.selectedIndex >= 0) renderInspector();
  })
  .catch(() => {
    setStatus("STANDALONE");
  });
