import {ALL_ROLES, LABEL, api, byId, escapeHtml, loadWorkspace, message, startBase, speciesId} from "./common.js";

function render(workspace) {
  byId("publicationSlots").innerHTML = ALL_ROLES.map(role => {
    const slot = workspace.roles[role];
    const active = slot.active_versions.map(v => `<p><span class="status active">线上 ACTIVE v${v.version}</span> · #${v.id}</p>`).join("") || '<p class="muted">无 ACTIVE 版本</p>';
    const drafts = slot.draft_versions.map(v => `<div class="tile" data-version-id="${v.id}">
      <p><span class="status draft">待审核 DRAFT v${v.version}</span> · #${v.id}</p>
      <p class="muted">来源 SHA-256：${escapeHtml(v.source_sha256 || "-")}</p>
      <p class="muted">视觉 QA：${escapeHtml(v.review?.visual_qa_result || "PENDING")} · 内容 QA：${escapeHtml(v.review?.content_qa_result || "PENDING")}</p>
      <div class="row"><button class="qa-pass">标记 QA PASS</button><button class="primary publish">发布此版本</button></div>
    </div>`).join("") || '<p class="muted">无待发布 DRAFT</p>';
    return `<article class="tile publication-role" data-role="${role}"><h3>${LABEL[role]} <span class="muted">${role}</span></h3>${active}${drafts}</article>`;
  }).join("");
  for (const card of document.querySelectorAll(".publication-role [data-version-id]")) {
    const versionId = Number(card.dataset.versionId);
    const role = card.closest(".publication-role").dataset.role;
    card.querySelector(".qa-pass").addEventListener("click", async () => {
      const slot = workspace.roles[role];
      const version = slot.draft_versions.find(v => v.id === versionId);
      if (!version?.batch_id) return message("该素材未关联可审核批次", true);
      try {
        await api(`/api/v1/admin/fish/assets/versions/${versionId}/review`, {
          method:"PUT", headers:{"Content-Type":"application/json"},
          body:JSON.stringify({batch_id:version.batch_id,visual_qa_result:"PASS",content_qa_result:"PASS",review_note:"v1.4 发布工作区 QA 确认",reviewer:"cms-operator"}),
        });
        message(`${role} 版本 #${versionId} QA 已记录`);
        render(await loadWorkspace());
      } catch (error) { message(error.message, true); }
    });
    card.querySelector(".publish").addEventListener("click", async () => {
      try {
        const result = await api(`/api/v1/admin/fish/assets/versions/${versionId}/activate`, {method:"POST"});
        message(`${role} v${result.version_id} 已发布 · FishCard #${result.card_id ?? "无"}`);
        render(await loadWorkspace());
      } catch (error) { message(error.message, true); }
    });
  }
}
export async function start(initialId) {
  if (!await startBase(initialId) || !speciesId) return;
  try { render(await loadWorkspace()); } catch (error) { message(error.message, true); }
}
