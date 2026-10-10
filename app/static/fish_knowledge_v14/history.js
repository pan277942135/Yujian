import {ALL_ROLES, LABEL, api, byId, escapeHtml, message, startBase, speciesId} from "./common.js";
export async function start(initialId) {
  if (!await startBase(initialId) || !speciesId) return;
  try {
    const data = await api(`/api/v1/admin/fish/assets/species/${encodeURIComponent(speciesId)}/workspace`);
    byId("historyRows").innerHTML = ALL_ROLES.map(role => {
      const slot = data.roles[role];
      const versions = slot.history.versions.map(v => `<tr><td>${escapeHtml(v.status)}</td><td>v${v.version} · version_id #${v.id}</td><td>${escapeHtml(v.created_at || "-")}</td><td>${v.frozen ? "FROZEN" : ""}</td><td><details><summary>来源与存储证据</summary><p>SHA-256：<code>${escapeHtml(v.source_sha256 || "-")}</code></p><p>GCS generation：<code>${escapeHtml(v.object_generation || "-")}</code></p><p>GCS 对象：<code>${escapeHtml(v.object_name || "-")}</code></p></details></td></tr>`).join("");
      const audits = slot.history.publication_audits.map(a => { const check=a.validation?.public_api_check; return `<li>发布 #${a.version_id} · ${escapeHtml(a.status)} · ${escapeHtml(a.created_at || "")} · ${escapeHtml(a.actor)}${check ? ` · 公共 API ${escapeHtml(check.status)} · ${escapeHtml(check.checked_at || "")} · ${escapeHtml(check.reviewer || "")}` : " · 公共 API 实测待执行"}</li>`; }).join("");
      const revisions = slot.history.content_revisions.map(r => `<li>内容 revision ${r.content_revision} · 卡片 #${r.card_id} · ${escapeHtml(r.created_at || "")}</li>`).join("");
      const qa = (slot.history.qa_audits || []).map(item => `<li>${escapeHtml(item.stage)} QA · ${escapeHtml(item.result)} · version #${item.version_id} · ${escapeHtml(item.reviewer)} · ${escapeHtml(item.created_at || "")} · 内容 revision ${escapeHtml(item.content_revision ?? "—")} · ${escapeHtml(item.evidence_note || "无备注")} <details><summary>核验快照</summary><p>SHA-256 <code>${escapeHtml(item.source_sha256)}</code> · GCS generation <code>${escapeHtml(item.object_generation)}</code> · ${escapeHtml(item.object_name)}</p></details></li>`).join("");
      return `<section class="panel"><h2>${LABEL[role]} <span class="muted">${role}</span></h2><table class="table"><thead><tr><th>状态</th><th>版本</th><th>创建时间</th><th>冻结</th><th>证明</th></tr></thead><tbody>${versions || '<tr><td colspan="5">暂无资产版本</td></tr>'}</tbody></table><h3>QA 阶段审计</h3><ul>${qa || "<li>暂无 QA 记录</li>"}</ul><h3>发布审计</h3><ul>${audits || "<li>暂无发布记录</li>"}</ul><h3>内容修订</h3><ul>${revisions || "<li>暂无内容修订</li>"}</ul></section>`;
    }).join("");
  } catch (error) { message(error.message, true); }
}
