import { renderMarkdown } from "./markdown.js";

// Static list -- this server has no directory-listing API, so new docs
// need a line added here. Order matches reading order (project overview,
// then phases in sequence, then follow-on optimisation work).
const DOCS = ["PROJECT-SUMMARY.md", "PHASE-0-1-SUMMARY.md", "PHASE-2-SUMMARY.md", "PHASE-3-SUMMARY.md", "PHASE-4-SUMMARY.md", "OPTIMISATIONS.md"];

// claude.ai Artifacts published from work on this project (charts,
// interactive companions to the .md docs above -- e.g. OPTIMISATIONS.md's
// "companion version ... with rendered charts" note). No API to list these
// from a static file server, so this is a manually-maintained pointer list
// -- update it when a new artifact is published or an old one is replaced.
const ARTIFACTS = [
  { title: "WGSL Inference Optimisations", url: "https://claude.ai/code/artifact/61cb79b4-48ae-4f60-9560-2cbd9abbeef7", updated: "2026-08-13", note: "Charted companion to OPTIMISATIONS.md -- per-stage speedup and per-layer timing breakdowns." },
  { title: "Reading Frame 1650", url: "https://claude.ai/code/artifact/c2e2e6a8-ccb5-42d1-a353-5728b9581ded", updated: "2026-08-13", note: "Visual walkthrough of a single held-out frame across pipeline stages." },
];

const navLinksEl = document.querySelector("#navLinks");
const contentEl = document.querySelector("#content");
const titleCache = new Map();

function titleFromMarkdown(text, fallbackFile) {
  const m = text.match(/^#\s+(.+)$/m);
  return m ? m[1].trim() : fallbackFile;
}

function renderArtifactsPage() {
  const items = ARTIFACTS.map(
    (a) => `
    <li>
      <a href="${a.url}" target="_blank" rel="noopener">${a.title}</a>
      <div class="meta">updated ${a.updated} — ${a.note}</div>
    </li>`
  ).join("");
  contentEl.innerHTML = `
    <h1>Published Artifacts</h1>
    <p>Interactive/visual companions published to claude.ai alongside the written docs on the left. These are private by default -- links only work for whoever they've been shared with.</p>
    <ul class="artifact-list">${items}</ul>
  `;
  document.title = "Published Artifacts — Neural Upscaler Docs";
}

async function loadDoc(file) {
  contentEl.innerHTML = '<div id="status">loading…</div>';
  let text;
  try {
    const res = await fetch(`./${file}`);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    text = await res.text();
  } catch (err) {
    contentEl.innerHTML = `<div id="status">Failed to load ${file}: ${err.message}</div>`;
    return;
  }
  contentEl.innerHTML = renderMarkdown(text);
  document.title = `${titleFromMarkdown(text, file)} — Neural Upscaler Docs`;
}

function currentRouteFromHash() {
  const hash = decodeURIComponent(location.hash.slice(1));
  if (hash === "artifacts") return "artifacts";
  return DOCS.includes(hash) ? hash : DOCS[0];
}

async function route() {
  const current = currentRouteFromHash();
  for (const a of navLinksEl.children) {
    a.classList.toggle("active", a.dataset.route === current);
  }
  if (current === "artifacts") {
    renderArtifactsPage();
  } else {
    await loadDoc(current);
  }
}

async function buildNav() {
  for (const file of DOCS) {
    const a = document.createElement("a");
    a.href = `#${file}`;
    a.dataset.route = file;
    a.textContent = file; // replaced with the real title once fetched below
    navLinksEl.appendChild(a);
  }
  const divider = document.createElement("div");
  divider.className = "nav-divider";
  divider.textContent = "Artifacts";
  navLinksEl.appendChild(divider);
  const artifactsLink = document.createElement("a");
  artifactsLink.href = "#artifacts";
  artifactsLink.dataset.route = "artifacts";
  artifactsLink.textContent = "Published Artifacts";
  navLinksEl.appendChild(artifactsLink);

  // Fetch every file once up front just for its title -- cheap (docs are
  // small, kilobytes each) and means the sidebar shows real names
  // immediately instead of raw filenames while browsing.
  await Promise.all(
    DOCS.map(async (file) => {
      try {
        const res = await fetch(`./${file}`);
        const text = await res.text();
        titleCache.set(file, titleFromMarkdown(text, file));
      } catch {
        titleCache.set(file, file);
      }
    })
  );
  for (const a of navLinksEl.children) {
    if (a.dataset.route && DOCS.includes(a.dataset.route)) {
      a.textContent = titleCache.get(a.dataset.route) ?? a.dataset.route;
    }
  }
}

window.addEventListener("hashchange", route);

async function main() {
  await buildNav();
  await route();
}

main();
