import {api, byId, escapeHtml, message, startBase, speciesId, speciesRows} from "./common.js";

export async function start(initialId) {
  if (!await startBase(initialId) || !speciesId) return;
  try {
    const row = speciesRows.find(item => item.id === speciesId);
    const detail = await api(`/api/v1/admin/fish/species/${encodeURIComponent(speciesId)}`);
    const workspace = await api(`/api/v1/admin/fish/assets/species/${encodeURIComponent(speciesId)}/workspace`);
    const primaryRoles = ["COVER_HERO", "HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL"];
    const active = primaryRoles.filter(role => workspace.roles[role].active_versions.length === 1).length;
    const draft = primaryRoles.reduce((sum, role) => sum + workspace.roles[role].draft_versions.length, 0);
    byId("overviewSummary").innerHTML = [
      ["鱼种状态", row?.status || "UNKNOWN"], ["知识卡 ACTIVE", `${detail.cards?.length || 0} / 5`],
      ["目标角色已发布", `${active} / 6`], ["待编辑 DRAFT", String(draft)],
    ].map(([label,value]) => `<div class="tile"><h3>${escapeHtml(label)}</h3><strong>${escapeHtml(value)}</strong></div>`).join("");
  } catch (error) { message(error.message, true); }
}
