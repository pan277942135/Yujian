import {ALL_ROLES, LABEL, api, byId, escapeHtml, loadWorkspace, message, startBase, speciesId} from "./common.js";

function render(workspace) {
  byId("assetSlots").innerHTML = ALL_ROLES.map(role => {
    const slot = workspace.roles[role];
    const active = slot.active_versions.map(v => `<div><span class="status active">ACTIVE v${v.version}</span> · ${escapeHtml(v.source_sha256 || "-")}</div>`).join("") || '<span class="muted">无 ACTIVE</span>';
    const legacy = slot.legacy_active?.length ? `<div><span class="status">LEGACY ACTIVE · 只读</span> · ${slot.legacy_active.length} 条</div>` : "";
    const drafts = slot.draft_versions.map(v => `<div><span class="status draft">DRAFT v${v.version}</span> · #${v.id} · ${escapeHtml(v.source_sha256 || "-")}</div>`).join("") || '<span class="muted">无 DRAFT</span>';
    return `<article class="tile"><h3>${LABEL[role]} <span class="muted">${role}</span></h3><div>${active}${legacy}</div><hr><div>${drafts}</div><p class="muted">历史版本：${slot.history.versions.length}</p></article>`;
  }).join("");
}
export async function start(initialId) {
  if (!await startBase(initialId) || !speciesId) return;
  byId("assetRole").innerHTML = ALL_ROLES.map(role => `<option value="${role}">${LABEL[role]} · ${role}</option>`).join("");
  try { render(await loadWorkspace()); } catch (error) { message(error.message, true); }
  byId("uploadForm").addEventListener("submit", async event => {
    event.preventDefault();
    const file = byId("assetFile").files[0];
    if (!file) return message("请选择图片文件", true);
    const data = new FormData(); data.append("species_id", speciesId); data.append("asset_role", byId("assetRole").value); data.append("file", file, file.name);
    try {
      const result = await api("/api/v1/admin/fish/assets/single-upload", {method:"POST", body:data});
      message(result.needs_warning_confirmation ? "素材已扫描；存在警告，请到发布中心确认并完成 QA。" : `已创建 DRAFT 版本 #${result.version?.id || ""}`);
      render(await loadWorkspace());
    } catch (error) { message(error.message, true); }
  });
}
