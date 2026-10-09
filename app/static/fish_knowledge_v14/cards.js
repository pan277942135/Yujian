import {CARD_ROLES, LABEL, api, byId, escapeHtml, jsonHeaders, loadWorkspace, message, startBase, speciesId} from "./common.js";

function render(workspace) {
  byId("cardSlots").innerHTML = CARD_ROLES.map(role => {
    const slot = workspace.roles[role];
    const versions = [...slot.active_versions, ...slot.draft_versions];
    const selected = slot.selected_version;
    const options = versions.map(v => `<option value="${v.id}" ${selected?.id === v.id ? "selected" : ""}>${v.status} v${v.version} · #${v.id}</option>`).join("");
    const active = slot.active_version;
    const preview = selected?.preview_url || selected?.image_url || "";
    const content = JSON.stringify(slot.structured_content || {}, null, 2);
    const disabled = !selected || selected.status !== "DRAFT" || slot.binding_status === "MISSING";
    return `<article class="tile card-editor" data-role="${role}">
      <h3>${LABEL[role]} <span class="muted">${role}</span></h3>
      <p>${active ? `<span class="status active">线上 ACTIVE v${active.version}</span> · #${active.id}` : '<span class="status">无版本化线上 ACTIVE</span>'}</p>
      <label>编辑目标版本<select class="wide version-select">${options || '<option value="">暂无版本</option>'}</select></label>
      ${preview ? `<img class="preview" src="${escapeHtml(preview)}" alt="${LABEL[role]} 版本预览">` : '<div class="preview empty">选择素材版本查看预览</div>'}
      <p class="muted">绑定状态：${escapeHtml(slot.binding_status)} · revision ${escapeHtml(slot.content_revision ?? "-")}</p>
      <label>卡片标题<input class="wide card-title" value="${escapeHtml(slot.card_title || "")}" placeholder="卡片标题"></label>
      <label>结构化内容 JSON<textarea class="card-content">${escapeHtml(content)}</textarea></label>
      <button class="primary save-content" ${disabled ? "disabled" : ""}>保存到此 DRAFT 版本</button>
    </article>`;
  }).join("");
  for (const card of document.querySelectorAll(".card-editor")) {
    const role = card.dataset.role;
    card.querySelector(".version-select").addEventListener("change", async event => {
      try { render(await loadWorkspace(Number(event.target.value))); }
      catch (error) { message(error.message, true); }
    });
    card.querySelector(".save-content").addEventListener("click", async () => {
      const selectedId = Number(card.querySelector(".version-select").value);
      const slot = workspace.roles[role];
      const selected = [...slot.active_versions, ...slot.draft_versions].find(v => v.id === selectedId);
      if (!selected || selected.status !== "DRAFT") return message("请选择具体 DRAFT 图片版本", true);
      let content;
      try { content = JSON.parse(card.querySelector(".card-content").value); }
      catch { return message(`${role} 内容不是有效 JSON`, true); }
      try {
        await api(`/api/v1/admin/fish/assets/species/${encodeURIComponent(speciesId)}/roles/${role}/content`, {
          method:"PUT", headers:jsonHeaders,
          body:JSON.stringify({version_id:selectedId,title:card.querySelector(".card-title").value,structured_content:content}),
        });
        message(`${LABEL[role]} 内容已保存到版本 #${selectedId}`);
        render(await loadWorkspace(selectedId));
      } catch (error) { message(error.message, true); }
    });
  }
}
export async function start(initialId) {
  if (!await startBase(initialId) || !speciesId) return;
  try { render(await loadWorkspace()); } catch (error) { message(error.message, true); }
}
