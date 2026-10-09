import {api, byId, escapeHtml, message, startBase, speciesId, speciesRows} from "./common.js";

export async function start(initialId) {
  if (!await startBase(initialId) || !speciesId) return;
  try {
    const row = speciesRows.find(item => item.id === speciesId);
    const detail = await api(`/api/v1/admin/fish/species/${encodeURIComponent(speciesId)}`);
    const workspace = await api(`/api/v1/admin/fish/assets/species/${encodeURIComponent(speciesId)}/workspace`);
    const active = Object.values(workspace.roles).filter(slot => slot.publication_status === "ACTIVE").length;
    const draft = Object.values(workspace.roles).reduce((sum, slot) => sum + slot.draft_versions.length, 0);
    byId("overviewSummary").innerHTML = [
      ["鱼种状态", row?.status || "UNKNOWN"], ["知识卡 ACTIVE", `${detail.cards?.length || 0} / 5`],
      ["资产角色已发布", `${active} / 9`], ["待编辑 DRAFT", String(draft)],
    ].map(([label,value]) => `<div class="tile"><h3>${escapeHtml(label)}</h3><strong>${escapeHtml(value)}</strong></div>`).join("");
  } catch (error) { message(error.message, true); }
}
