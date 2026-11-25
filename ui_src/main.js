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

const app = new App({ name: "sand3d-rag-explorer", version: "1.1.0" }, {});

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
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

function renderResults() {
  resultsCount.textContent = `${state.items.length} items`;
  if (!state.items.length) {
    resultsList.innerHTML = `<div class="empty-state">No matching assets found.</div>`;
    return;
  }
  resultsList.innerHTML = state.items
    .map((item, idx) => {
      const active = idx === state.selectedIndex ? "active" : "";
      const thumb = item.thumbnail_data_uri
        ? `<img src="${item.thumbnail_data_uri}" alt="asset render" />`
        : `<div class="thumb-placeholder">${escapeHtml(item.family || "3D")}</div>`;
      const ranksText = item.ranks
        ? Object.entries(item.ranks)
            .map(([k, v]) => `${k}:#${v}`)
            .join(" · ")
        : "featured";
      const scoreBadge =
        item.rrf_score !== undefined && item.rrf_score !== null
          ? `<span class="tag tag-score">RRF ${Number(item.rrf_score).toFixed(4)}</span>`
          : `<span class="tag tag-score">CURATED</span>`;
      return `
        <button type="button" class="card ${active}" data-idx="${idx}">
          <div class="thumb-wrap">${thumb}</div>
          <div>
            <div class="card-meta">
              ${scoreBadge}
              <span class="tag">${escapeHtml(item.family || "")}</span>
              <span class="tag">score ${Number(item.semantic_score || 0).toFixed(2)}</span>
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
  statusEl.textContent = `LOADING #${item.id}...`;
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
    statusEl.textContent = "READY";
  } catch (err) {
    statusEl.textContent = `DETAIL ERR: ${err.message || err}`;
  }
}

function renderInspector() {
  if (state.generated) {
    const gen = state.generated;
    inspectorTitle.textContent = `Generated: ${gen.saved_path || "script.py"}`;
    syntaxBadge.innerHTML = gen.syntax_valid
      ? `<span class="badge-ok">AST VALID</span>`
      : `<span class="badge-err">SYNTAX ERROR</span>`;
    inspectorBody.innerHTML = `
      <div class="actions-row">
        <button type="button" class="btn" id="copy-code-btn">COPY CODE</button>
        <button type="button" class="btn" id="back-ref-btn">BACK TO SELECTED REF</button>
      </div>
      <pre class="code-box"><code>${escapeHtml(gen.code || "")}</code></pre>
    `;
    document.getElementById("copy-code-btn").onclick = () => copyText(gen.code || "");
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
    inspectorBody.innerHTML = `<div class="empty-state">Select an asset card to inspect its sand3d Python script.</div>`;
    return;
  }

  const canAttach = Boolean(state.hostCaps && state.hostCaps.updateModelContext);
  const attachBtnHtml = canAttach
    ? `<button type="button" class="btn" id="attach-ctx-btn">ATTACH TO CHAT</button>`
    : "";

  inspectorTitle.textContent = `Entry: ${item.entry || "program"}`;
  syntaxBadge.textContent = `#${item.id}`;
  inspectorBody.innerHTML = `
    <div class="detail-grid">
      <div class="metric">
        <div class="metric-label">Domain</div>
        <div class="metric-val">${escapeHtml(item.domain)}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Subject</div>
        <div class="metric-val">${escapeHtml(item.subject)}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Triangles</div>
        <div class="metric-val">${Number(item.triangles || 0).toLocaleString()}</div>
      </div>
    </div>
    <div class="actions-row">
      <button type="button" class="btn" id="copy-code-btn">COPY CODE</button>
      ${attachBtnHtml}
      <button type="button" class="btn btn-primary" id="gen-ollama-btn">GENERATE NEW WITH OLLAMA</button>
    </div>
    <pre class="code-box"><code>${escapeHtml(item.program || "Loading full Python program...")}</code></pre>
  `;
  document.getElementById("copy-code-btn").onclick = () => copyText(item.program || "");
  document.getElementById("gen-ollama-btn").onclick = () => runGenerate(item);
  if (canAttach) {
    document.getElementById("attach-ctx-btn").onclick = () => attachToChat(item);
  }
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    statusEl.textContent = "COPIED TO CLIPBOARD";
  } catch {
    statusEl.textContent = "COPY FAILED IN SANDBOX";
  }
}

async function attachToChat(item) {
  try {
    await app.updateModelContext({
      content: [
        {
          type: "text",
          text:
            `Reference sand3d asset #${item.id} (${item.domain}/${item.family}/${item.subject}):\n` +
            `Prompt: ${item.prompt}\n` +
            `Entry: ${item.entry}\n\n` +
            "```python\n" +
            (item.program || "") +
            "\n```",
        },
      ],
    });
    statusEl.textContent = "ATTACHED TO NEXT MESSAGE";
  } catch (err) {
    statusEl.textContent = `ATTACH ERR: ${err.message || err}`;
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
  statusEl.textContent = "SEARCHING...";
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
    statusEl.textContent = "READY";
  } catch (err) {
    statusEl.textContent = `ERR: ${err.message || err}`;
  } finally {
    searchBtn.disabled = false;
  }
}

async function runGenerate(item) {
  const prompt = queryInput.value.trim() || item.prompt || "mossy rock";
  const btn = document.getElementById("gen-ollama-btn");
  if (btn) btn.disabled = true;
  statusEl.textContent = "GENERATING VIA OLLAMA...";
  try {
    const res = await app.callServerTool({
      name: "generate_sand3d_program",
      arguments: {
        prompt,
        top_k: 3,
        model: "gemma4:12b",
        output_filename: "mcp_app_generated.py",
      },
    });
    applyToolResultPayload(res);
    statusEl.textContent = "GENERATED";
  } catch (err) {
    statusEl.textContent = `GEN ERR: ${err.message || err}`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

// Register lifecycle handlers BEFORE connect() so no host notification is missed.
app.ontoolinput = ({ arguments: args }) => {
  if (args && args.query) {
    queryInput.value = args.query;
  }
  statusEl.textContent = "RUNNING TOOL...";
};

app.ontoolresult = (result) => {
  applyToolResultPayload(result);
  statusEl.textContent = "READY";
};

app.ontoolcancelled = () => {
  statusEl.textContent = "CANCELLED";
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
    statusEl.textContent = "MCP APP CONNECTED";
  })
  .catch(() => {
    statusEl.textContent = "STANDALONE PREVIEW";
  });
