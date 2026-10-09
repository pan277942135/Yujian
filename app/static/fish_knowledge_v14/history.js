import {ALL_ROLES, LABEL, api, byId, escapeHtml, message, startBase, speciesId} from "./common.js";
export async function start(initialId) {
  if (!await startBase(initialId) || !speciesId) return;
  try {
    const data = await api(`/api/v1/admin/fish/assets/species/${encodeURIComponent(speciesId)}/workspace`);
    byId("historyRows").innerHTML = ALL_ROLES.map(role => {
      const slot = data.roles[role];
      const versions = slot.history.versions.map(v => `<tr><td>${escapeHtml(v.status)}</td><td>v${v.version} · #${v.id}</td><td>${escapeHtml(v.source_sha256 || "-")}</td><td>${escapeHtml(v.created_at || "-")}</td><td>${v.frozen ? "FROZEN" : ""}</td></tr>`).join("");
      const audits = slot.history.publication_audits.map(a => `<li>发布 #${a.version_id} · ${escapeHtml(a.status)} · ${escapeHtml(a.created_at || "")} · ${escapeHtml(a.actor)}</li>`).join("");
      const revisions = slot.history.content_revisions.map(r => `<li>内容 revision ${r.content_revision} · 卡片 #${r.card_id} · ${escapeHtml(r.created_at || "")}</li>`).join("");
      return `<section class="panel"><h2>${LABEL[role]} <span class="muted">${role}</span></h2><table class="table"><thead><tr><th>状态</th><th>版本</th><th>来源 SHA-256</th><th>创建时间</th><th>冻结</th></tr></thead><tbody>${versions || '<tr><td colspan="5">暂无资产版本</td></tr>'}</tbody></table><h3>发布审计</h3><ul>${audits || "<li>暂无发布记录</li>"}</ul><h3>内容修订</h3><ul>${revisions || "<li>暂无内容修订</li>"}</ul></section>`;
    }).join("");
  } catch (error) { message(error.message, true); }
}
