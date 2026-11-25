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
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link
    href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&family=Vazirmatn:wght@400;500;600;700&display=swap"
    rel="stylesheet"
  />
  <style>
    :root {
      --bg: var(--color-background-primary, #0b0e14);
      --panel: var(--color-background-secondary, #111620);
      --panel-alt: var(--color-background-tertiary, #171e2b);
      --border: var(--color-border-primary, rgba(255, 255, 255, 0.09));
      --border-strong: rgba(255, 255, 255, 0.16);
      --ink: var(--color-text-primary, #eef2f7);
      --muted: var(--color-text-secondary, #8d99ae);
      --accent: #f97316;
      --accent-soft: rgba(249, 115, 22, 0.12);
      --accent-ink: #0b0e14;
      --ok: var(--color-text-success, #34d399);
      --err: var(--color-text-danger, #f87171);
      --tok-kw: #fb923c;
      --tok-str: #86efac;
      --tok-com: #64748b;
      --tok-api: #93c5fd;
      --tok-num: #fcd34d;
      --sans: "Space Grotesk", "Vazirmatn", "Inter Display", "Inter", "Adwaita Sans", system-ui, -apple-system, sans-serif;
      --mono: "JetBrains Mono", "JetBrainsMono Nerd Font", "FiraCode Nerd Font Mono", "Adwaita Mono", "Liberation Mono", monospace;
    }

    :root[data-theme="light"] {
      --bg: var(--color-background-primary, #f6f8fb);
      --panel: var(--color-background-secondary, #ffffff);
      --panel-alt: var(--color-background-tertiary, #eef2f7);
      --border: var(--color-border-primary, rgba(15, 23, 42, 0.1));
      --border-strong: rgba(15, 23, 42, 0.2);
      --ink: var(--color-text-primary, #0f172a);
      --muted: var(--color-text-secondary, #475569);
      --accent: #ea580c;
      --accent-soft: rgba(234, 88, 12, 0.1);
      --accent-ink: #ffffff;
      --ok: var(--color-text-success, #059669);
      --err: var(--color-text-danger, #dc2626);
      --tok-kw: #c2410c;
      --tok-str: #15803d;
      --tok-com: #64748b;
      --tok-api: #1d4ed8;
      --tok-num: #b45309;
    }

    * { box-sizing: border-box; }

    html, body {
      margin: 0;
      padding: 0;
      max-width: 100%;
      overflow-x: hidden;
      background: var(--bg);
      color: var(--ink);
      font-family: var(--sans);
      font-size: 13.5px;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    body {
      padding: 14px;
    }

    .shell {
      display: flex;
      flex-direction: column;
      gap: 12px;
      width: 100%;
      max-width: 1240px;
      margin: 0 auto;
      min-width: 0;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 14px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }

    .brand-mark {
      width: 9px;
      height: 9px;
      border-radius: 2px;
      background: var(--accent);
      flex-shrink: 0;
    }

    .brand-title {
      font-weight: 600;
      font-size: 14px;
      letter-spacing: -0.01em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .brand-sub {
      color: var(--muted);
      font-size: 12px;
      font-weight: 400;
    }

    .status-pill {
      font-family: var(--mono);
      font-size: 11px;
      font-weight: 500;
      color: var(--muted);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 3px 10px;
      background: var(--panel-alt);
      white-space: nowrap;
      flex-shrink: 0;
    }

    .status-pill[data-tone="ok"] {
      color: var(--ok);
      border-color: color-mix(in srgb, var(--ok) 35%, transparent);
    }

    .status-pill[data-tone="err"] {
      color: var(--err);
      border-color: color-mix(in srgb, var(--err) 35%, transparent);
    }

    .controls {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }

    input, select, button {
      font-family: inherit;
      font-size: 13px;
      border-radius: 6px;
    }

    input[type="text"] {
      flex: 1 1 260px;
      min-width: 0;
      background: var(--bg);
      color: var(--ink);
      border: 1px solid var(--border);
      padding: 8px 12px;
      outline: none;
      transition: border-color 140ms ease;
    }

    select {
      background: var(--bg);
      color: var(--ink);
      border: 1px solid var(--border);
      padding: 8px 10px;
      outline: none;
      font-family: var(--mono);
      font-size: 12px;
    }

    input[type="text"]:focus, select:focus {
      border-color: var(--accent);
    }

    .btn {
      background: var(--panel-alt);
      color: var(--ink);
      border: 1px solid var(--border);
      padding: 8px 13px;
      cursor: pointer;
      font-family: var(--sans);
      font-size: 12.5px;
      font-weight: 600;
      white-space: nowrap;
      transition: border-color 140ms ease, opacity 140ms ease, background 140ms ease;
    }

    .btn:hover { border-color: var(--border-strong); }

    .btn-primary {
      background: var(--accent);
      color: var(--accent-ink);
      border-color: var(--accent);
    }

    .btn-primary:hover { opacity: 0.92; }

    .btn-subtle {
      background: transparent;
      color: var(--muted);
      font-weight: 500;
    }

    .btn-subtle:hover { color: var(--ink); }
    .btn:disabled { opacity: 0.45; cursor: not-allowed; }

    .workspace {
      display: grid;
      grid-template-columns: minmax(280px, 340px) minmax(0, 1fr);
      gap: 12px;
      align-items: start;
      min-width: 0;
    }

    @media (max-width: 760px) {
      .workspace {
        grid-template-columns: minmax(0, 1fr);
      }
    }

    .pane {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      display: flex;
      flex-direction: column;
      min-width: 0;
      overflow: hidden;
    }

    .pane-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 9px 13px;
      border-bottom: 1px solid var(--border);
      background: var(--panel-alt);
      font-family: var(--mono);
      font-size: 11px;
      font-weight: 500;
      color: var(--muted);
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }

    .results-list {
      overflow-y: auto;
      max-height: 620px;
      display: flex;
      flex-direction: column;
    }

    .card {
      display: grid;
      grid-template-columns: 76px minmax(0, 1fr);
      gap: 11px;
      padding: 11px 12px;
      border: 0;
      border-bottom: 1px solid var(--border);
      border-left: 3px solid transparent;
      border-radius: 0;
      cursor: pointer;
      text-align: left;
      background: transparent;
      color: inherit;
      width: 100%;
      transition: background 120ms ease;
    }

    .card:last-child { border-bottom: 0; }
    .card:hover { background: var(--panel-alt); }
    .card.active {
      background: var(--panel-alt);
      border-left-color: var(--accent);
    }

    .thumb-wrap {
      width: 76px;
      height: 76px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
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

    .card-body {
      min-width: 0;
    }

    .card-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-bottom: 5px;
      font-family: var(--mono);
      font-size: 10.5px;
      font-variant-numeric: tabular-nums;
    }

    .tag {
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 1px 6px;
      color: var(--muted);
      background: var(--bg);
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .tag-accent {
      color: var(--accent);
      background: var(--accent-soft);
      border-color: color-mix(in srgb, var(--accent) 40%, transparent);
    }

    .card-prompt {
      font-size: 12.5px;
      font-weight: 500;
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
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .inspector-body {
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 11px;
      min-width: 0;
    }

    .prompt-banner {
      padding: 10px 12px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 13px;
      color: var(--ink);
    }

    .metrics-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      min-width: 0;
    }

    .metric {
      flex: 1 1 130px;
      min-width: 0;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--bg);
      padding: 7px 10px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    .metric-label {
      font-family: var(--mono);
      color: var(--muted);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .metric-val {
      font-family: var(--mono);
      color: var(--ink);
      font-size: 12px;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .actions-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }

    pre.code-box {
      margin: 0;
      padding: 12px 14px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.55;
      overflow: auto;
      max-height: 430px;
      white-space: pre;
      tab-size: 4;
      width: 100%;
    }

    .tok-kw { color: var(--tok-kw); font-weight: 500; }
    .tok-str { color: var(--tok-str); }
    .tok-com { color: var(--tok-com); font-style: italic; }
    .tok-api { color: var(--tok-api); }
    .tok-num { color: var(--tok-num); }

    .empty-state {
      padding: 32px 18px;
      text-align: center;
      color: var(--muted);
      font-size: 13px;
    }

    .pill {
      font-family: var(--mono);
      font-size: 10.5px;
      padding: 2px 7px;
      border-radius: 4px;
      border: 1px solid var(--border);
      color: var(--muted);
    }

    .pill-ok { color: var(--ok); border-color: color-mix(in srgb, var(--ok) 40%, transparent); }
    .pill-err { color: var(--err); border-color: color-mix(in srgb, var(--err) 40%, transparent); }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div class="brand">
        <span class="brand-mark"></span>
        <span class="brand-title">Sand3D Explorer <span class="brand-sub">// Hybrid Multimodal RAG</span></span>
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
      <button type="submit" class="btn btn-primary" id="search-btn">Search</button>
    </form>

    <main class="workspace">
      <section class="pane">
        <div class="pane-head">
          <span>References</span>
          <span id="results-count">0 assets</span>
        </div>
        <div class="results-list" id="results-list">
          <div class="empty-state">Loading references...</div>
        </div>
      </section>

      <section class="pane">
        <div class="pane-head">
          <span id="inspector-title">Program Inspector</span>
          <span id="syntax-badge"></span>
        </div>
        <div class="inspector-body" id="inspector-body">
          <div class="empty-state">Select a 3D asset card to inspect its procedural sand3d script.</div>
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
