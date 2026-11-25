import fs from "node:fs";
import path from "node:path";
import { build } from "esbuild";

const root = path.resolve(import.meta.dirname, "..");
const bundle = await build({
  entryPoints: [path.resolve(import.meta.dirname, "main.js")],
  bundle: true,
  minify: true,
  format: "iife",
  write: false,
});

const jsCode = bundle.outputFiles[0].text;
const template = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Sand3D Nature — Hybrid RAG Explorer</title>
  <style>
    :root {
      --bg: var(--color-background-primary, #0d1117);
      --panel: var(--color-background-secondary, #161b22);
      --panel-alt: var(--color-background-tertiary, #1c222b);
      --border: var(--color-border-primary, #2a313c);
      --ink: var(--color-text-primary, #e6edf3);
      --muted: var(--color-text-secondary, #8b949e);
      --accent: #f97316;
      --accent-ink: #0d1117;
      --ok: var(--color-text-success, #3fb950);
      --err: var(--color-text-danger, #f85149);
      --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
      --sans: var(--font-sans, "Vazirmatn", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif);
    }

    :root[data-theme="light"] {
      --bg: var(--color-background-primary, #f6f8fa);
      --panel: var(--color-background-secondary, #ffffff);
      --panel-alt: var(--color-background-tertiary, #f0f3f6);
      --border: var(--color-border-primary, #d0d7de);
      --ink: var(--color-text-primary, #1f2328);
      --muted: var(--color-text-secondary, #656d76);
      --accent: #ea580c;
      --accent-ink: #ffffff;
      --ok: var(--color-text-success, #1a7f37);
      --err: var(--color-text-danger, #cf222e);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      padding: 12px;
      background: var(--bg);
      color: var(--ink);
      font-family: var(--sans);
      font-size: 13px;
      line-height: 1.45;
    }

    .shell {
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 10px;
      max-width: 1120px;
      margin: 0 auto;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 12px;
      background: var(--panel);
      border: 1px solid var(--border);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 8px;
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }

    .brand-mark {
      width: 8px;
      height: 8px;
      background: var(--accent);
      display: inline-block;
    }

    .status-pill {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--muted);
      border: 1px solid var(--border);
      padding: 2px 8px;
      background: var(--panel-alt);
    }

    .controls {
      display: grid;
      grid-template-columns: 1fr auto auto auto;
      gap: 8px;
      padding: 10px;
      background: var(--panel);
      border: 1px solid var(--border);
    }

    input, select, button {
      font-family: inherit;
      font-size: 13px;
      border-radius: 2px;
    }

    input[type="text"], select {
      background: var(--bg);
      color: var(--ink);
      border: 1px solid var(--border);
      padding: 7px 10px;
      outline: none;
    }

    input[type="text"]:focus, select:focus {
      border-color: var(--accent);
    }

    .btn {
      background: var(--panel-alt);
      color: var(--ink);
      border: 1px solid var(--border);
      padding: 7px 12px;
      cursor: pointer;
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 600;
    }

    .btn:hover { border-color: var(--ink); }

    .btn-primary {
      background: var(--accent);
      color: var(--accent-ink);
      border-color: var(--accent);
    }

    .btn-primary:hover { opacity: 0.92; }
    .btn:disabled { opacity: 0.45; cursor: not-allowed; }

    .workspace {
      display: grid;
      grid-template-columns: 1.05fr 1fr;
      gap: 10px;
      min-height: 440px;
    }

    @media (max-width: 780px) {
      .controls { grid-template-columns: 1fr 1fr; }
      .controls input[type="text"] { grid-column: 1 / -1; }
      .workspace { grid-template-columns: 1fr; }
    }

    .pane {
      background: var(--panel);
      border: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      min-height: 0;
    }

    .pane-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 12px;
      border-bottom: 1px solid var(--border);
      background: var(--panel-alt);
      font-family: var(--mono);
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .results-list {
      overflow-y: auto;
      max-height: 520px;
      display: flex;
      flex-direction: column;
    }

    .card {
      display: grid;
      grid-template-columns: 88px 1fr;
      gap: 10px;
      padding: 10px;
      border-bottom: 1px solid var(--border);
      cursor: pointer;
      text-align: left;
      background: transparent;
      color: inherit;
      border-left: 3px solid transparent;
    }

    .card:hover { background: var(--panel-alt); }
    .card.active {
      background: var(--panel-alt);
      border-left-color: var(--accent);
    }

    .thumb-wrap {
      width: 88px;
      height: 88px;
      background: var(--bg);
      border: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      flex-shrink: 0;
    }

    .thumb-wrap img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }

    .thumb-placeholder {
      font-family: var(--mono);
      font-size: 10px;
      color: var(--muted);
      text-align: center;
      padding: 4px;
    }

    .card-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 4px;
      font-family: var(--mono);
      font-size: 11px;
    }

    .tag {
      border: 1px solid var(--border);
      padding: 1px 6px;
      color: var(--muted);
      background: var(--bg);
    }

    .tag-score {
      color: var(--accent);
      border-color: var(--accent);
    }

    .card-prompt {
      font-size: 12.5px;
      color: var(--ink);
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }

    .card-ranks {
      margin-top: 5px;
      font-family: var(--mono);
      font-size: 10.5px;
      color: var(--muted);
    }

    .inspector-body {
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      flex: 1;
      min-height: 0;
    }

    .detail-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      font-family: var(--mono);
      font-size: 11px;
    }

    .metric {
      border: 1px solid var(--border);
      background: var(--bg);
      padding: 6px 8px;
    }

    .metric-label {
      color: var(--muted);
      font-size: 10px;
      text-transform: uppercase;
    }

    .metric-val {
      color: var(--ink);
      font-weight: 600;
      margin-top: 2px;
      word-break: break-all;
    }

    .actions-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }

    pre.code-box {
      margin: 0;
      padding: 10px;
      background: var(--bg);
      border: 1px solid var(--border);
      font-family: var(--mono);
      font-size: 11.5px;
      line-height: 1.5;
      overflow: auto;
      max-height: 360px;
      white-space: pre;
      tab-size: 4;
      flex: 1;
    }

    .empty-state {
      padding: 28px 16px;
      text-align: center;
      color: var(--muted);
      font-family: var(--mono);
      font-size: 12px;
    }

    .badge-ok { color: var(--ok); }
    .badge-err { color: var(--err); }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div class="brand">
        <span class="brand-mark"></span>
        <span>SAND3D-RAG // HYBRID MULTIMODAL EXPLORER</span>
      </div>
      <div class="status-pill" id="host-status">CONNECTING...</div>
    </header>

    <form class="controls" id="search-form">
      <input
        type="text"
        id="query-input"
        dir="auto"
        placeholder="Search in English or Persian (e.g. صخره خزه‌دار با قارچ / spiral shell)..."
      />
      <select id="min-score" title="Minimum semantic score">
        <option value="0.9">score >= 0.90</option>
        <option value="0.8">score >= 0.80</option>
        <option value="0.0">all scores</option>
      </select>
      <select id="top-k" title="Results count">
        <option value="5">top 5</option>
        <option value="8">top 8</option>
        <option value="12">top 12</option>
      </select>
      <button type="submit" class="btn btn-primary" id="search-btn">SEARCH</button>
    </form>

    <main class="workspace">
      <section class="pane">
        <div class="pane-head">
          <span>Retrieved References</span>
          <span id="results-count">0 items</span>
        </div>
        <div class="results-list" id="results-list">
          <div class="empty-state">Waiting for tool result or search query...</div>
        </div>
      </section>

      <section class="pane">
        <div class="pane-head">
          <span id="inspector-title">Program Inspector</span>
          <span id="syntax-badge"></span>
        </div>
        <div class="inspector-body" id="inspector-body">
          <div class="empty-state">Select an asset card to inspect its sand3d Python script or synthesize a new variation via Ollama.</div>
        </div>
      </section>
    </main>
  </div>

  <script>${jsCode}</script>
</body>
</html>
`;

const outPath = path.join(root, "ui", "explorer.html");
fs.writeFileSync(outPath, template, "utf8");
console.log(`Bundled self-contained MCP App HTML (${(template.length / 1024).toFixed(1)} KiB) -> ${outPath}`);
