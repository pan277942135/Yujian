export const CARD_ROLES = ["HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL"];
export const ALL_ROLES = ["COVER_LIST", "COVER_HERO", "TRANSPARENT_MAIN", "TRANSPARENT_ALT", ...CARD_ROLES];
export const LABEL = {
  COVER_LIST: "列表封面", COVER_HERO: "详情主视觉", TRANSPARENT_MAIN: "透明主体",
  TRANSPARENT_ALT: "透明备选", HERO: "鱼种概览卡", IDENTIFICATION: "识别特征卡",
  ECO: "生态习性卡", GEAR: "装备记录卡", SKILL: "垂钓技巧卡",
};
export let speciesId = "";
export let speciesRows = [];
export let workspace = null;

export function byId(id) { return document.getElementById(id); }
export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[char]);
}
export function message(value, error = false) {
  const node = byId("notice");
  if (!node) return;
  node.textContent = value;
  node.classList.toggle("error", error);
  node.style.display = "block";
}
export async function api(path, options = {}) {
  const response = await fetch(path, {credentials:"same-origin", ...options});
  const raw = await response.text();
  let body;
  try { body = JSON.parse(raw); } catch { body = raw; }
  if (!response.ok) {
    const detail = body?.detail;
    const msg = typeof detail === "string" ? detail : (detail?.message || body?.message || JSON.stringify(detail || body));
    const error = new Error(msg);
    error.code = detail?.code || body?.error;
    error.detail = detail;
    error.publicationCommitted = detail?.publication_committed === true;
    error.contentCommitted = detail?.content_committed === true;
    throw error;
  }
  return body;
}
export async function loadSpecies(initialId = "") {
  speciesRows = await api("/api/v1/admin/fish/species");
  const select = byId("speciesSelect");
  select.innerHTML = '<option value="">选择鱼种…</option>' + speciesRows.map(row =>
    `<option value="${escapeHtml(row.id)}">${escapeHtml(row.name_cn)} · ${escapeHtml(row.id)}</option>`
  ).join("");
  const queryId = new URLSearchParams(location.search).get("species_id") || "";
  speciesId = initialId || queryId || speciesRows[0]?.id || "";
  select.value = speciesId;
  updateLinks();
  select.addEventListener("change", () => {
    const section = document.body.dataset.section;
    const nextId = select.value;
    location.href = nextId ? `/fish-knowledge/${encodeURIComponent(nextId)}/${section}` : `/fish-knowledge/${section}`;
  });
  byId("speciesStatus").textContent = speciesRows.find(row => row.id === speciesId)?.name_cn || "";
}
export function updateLinks() {
  for (const link of document.querySelectorAll("[data-section-link]")) {
    const section = link.dataset.sectionLink;
    link.href = speciesId ? `/fish-knowledge/${encodeURIComponent(speciesId)}/${section}` : `/fish-knowledge/${section}`;
  }
  for (const link of document.querySelectorAll("[data-workspace-link]")) {
    const section = link.dataset.workspaceLink;
    link.href = speciesId ? `/fish-knowledge/${encodeURIComponent(speciesId)}/${section}` : `/fish-knowledge/${section}`;
  }
}
export async function loadWorkspace(selectedVersionId = null) {
  if (!speciesId) return null;
  const suffix = selectedVersionId ? `?selected_version_id=${encodeURIComponent(selectedVersionId)}` : "";
  workspace = await api(`/api/v1/admin/fish/assets/species/${encodeURIComponent(speciesId)}/workspace${suffix}`);
  return workspace;
}
export async function startBase(initialId = "") {
  try { await loadSpecies(initialId); return true; }
  catch (error) { message(error.message, true); return false; }
}
export const jsonHeaders = {"Content-Type":"application/json"};
