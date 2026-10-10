import {CARD_ROLES, LABEL, api, byId, escapeHtml, jsonHeaders, loadWorkspace, message, startBase, speciesId} from "./common.js";

const FIELDS = {
  HERO: [["tag", "概览标签", "text"], ["description", "简介", "long"], ["rarity", "稀有度（数值）", "number"], ["power", "力量（数值）", "number"], ["challenge", "挑战度（数值）", "number"]],
  IDENTIFICATION: [["features", "识别特征（每行一项）", "list"], ["similar", "易混淆鱼种（每行一项）", "list"]],
  ECO: [["habitat", "栖息地", "text"], ["water_layer", "主要水层", "text"], ["season", "活动季节", "text"], ["behavior", "行为特征", "long"], ["diet", "食性", "text"]],
  GEAR: [["method", "推荐钓法", "text"], ["rod", "鱼竿", "text"], ["line", "钓线", "text"], ["hook", "鱼钩", "text"], ["bait", "饵料", "text"]],
  SKILL: [["find", "找鱼", "long"], ["attract", "诱鱼", "long"], ["action", "操作", "long"], ["tip", "注意事项", "long"]],
};

function inputFor([key, label, kind], value) {
  const normalized = Array.isArray(value) ? value.join("\n") : value ?? "";
  const tag = kind === "long" || kind === "list" ? "textarea" : "input";
  const type = kind === "number" ? "number" : "text";
  const attr = tag === "textarea" ? "" : `type="${type}"`;
  return `<label>${label}${tag === "textarea" ? `<textarea class="content-field" data-key="${key}" data-kind="${kind}">${escapeHtml(normalized)}</textarea>` : `<input class="content-field wide" data-key="${key}" data-kind="${kind}" ${attr} value="${escapeHtml(normalized)}">`}</label>`;
}

function versionPreview(version, heading, role) {
  return `<div class="tile"><p>${heading}</p>${version ? `<p><span class="status ${version.status === "ACTIVE" ? "active" : "draft"}">${version.status} v${version.version}</span> · #${version.id}</p><img class="preview" src="${escapeHtml(version.preview_url)}" alt="${LABEL[role]} ${heading}">` : '<p class="muted">暂无版本化图片</p>'}</div>`;
}

function render(workspace) {
  byId("cardSlots").innerHTML = CARD_ROLES.map(role => {
    const slot = workspace.roles[role];
    const versions = [...slot.active_versions, ...slot.draft_versions];
    const selected = slot.selected_version;
    const active = slot.active_version || (slot.legacy_active?.[0] ? {
      id:slot.legacy_active[0].card_id, version:"legacy", status:"ACTIVE", preview_url:slot.legacy_active[0].image_url, legacy:true,
    } : null);
    const disabled = !selected || selected.status !== "DRAFT" || slot.binding_status === "MISSING";
    const content = slot.structured_content || {};
    const options = versions.map(v => `<option value="${v.id}" ${selected?.id === v.id ? "selected" : ""}>${v.status} v${v.version} · version #${v.id}</option>`).join("");
    const fields = (FIELDS[role] || []).map(field => inputFor(field, content[field[0]])).join("");
    return `<article class="tile card-editor" data-role="${role}">
      <h3>${LABEL[role]} <span class="muted">${role}</span></h3>
      <p>精确绑定：species_id <code>${escapeHtml(speciesId)}</code> · asset_role <code>${role}</code></p>
      <label>编辑目标素材版本<select class="wide version-select" ${versions.length ? "" : "disabled"}>${options || '<option value="">先上传此角色图片</option>'}</select></label>
      <div class="grid">${versionPreview(active, "当前线上图片", role)}${versionPreview(selected?.status === "DRAFT" ? selected : slot.draft_version, "正在编辑的 DRAFT 图片", role)}</div>
      <p class="muted">绑定状态：${escapeHtml(slot.binding_status)}${active?.legacy ? ` · 兼容只读 FishCard #${active.id}` : ""} · 内容 revision ${escapeHtml(slot.content_revision ?? "—")} · QA：视觉 ${escapeHtml(slot.visual_qa)} / 内容 ${escapeHtml(slot.content_qa)}</p>
      <label>卡片标题<input class="wide card-title" value="${escapeHtml(slot.card_title || "")}" ${disabled ? "disabled" : ""}></label>
      <div class="stack structured-fields">${fields}</div>
      <details><summary>高级模式：原始结构化 JSON</summary><p class="muted">只保存到页面所示的精确素材版本 ID。高级编辑会与上方常用字段合并。</p><textarea class="card-content" ${disabled ? "disabled" : ""}>${escapeHtml(JSON.stringify(content, null, 2))}</textarea></details>
      <button class="primary save-content" ${disabled ? "disabled" : ""}>保存知识卡草稿</button>
    </article>`;
  }).join("");

  for (const card of document.querySelectorAll(".card-editor")) {
    const role = card.dataset.role;
    card.querySelector(".version-select").addEventListener("change", async event => {
      const id = Number(event.target.value);
      if (!id) return;
      const dirty = [...card.querySelectorAll("input:not(.version-select), textarea")].some(input => input.value !== input.defaultValue);
      if (dirty && !window.confirm("切换素材版本会重新载入内容。放弃当前未保存输入并继续吗？")) { event.target.value = workspace.roles[role].selected_version_id; return; }
      try { render(await loadWorkspace(id)); }
      catch (error) { message(error.message, true); }
    });
    card.querySelector(".save-content").addEventListener("click", async () => {
      const selectedId = Number(card.querySelector(".version-select").value);
      const slot = workspace.roles[role];
      const target = [...slot.active_versions, ...slot.draft_versions].find(v => v.id === selectedId);
      if (!target || target.status !== "DRAFT") return message("请选择要编辑的具体 DRAFT 图片版本", true);
      let content;
      try {
        content = JSON.parse(card.querySelector(".card-content").value || "{}");
        if (!content || Array.isArray(content) || typeof content !== "object") throw new Error("结构化内容必须是 JSON 对象");
      } catch (error) { return message(error.message || `${role} 原始 JSON 无效`, true); }
      for (const field of card.querySelectorAll(".content-field")) {
        const key = field.dataset.key, kind = field.dataset.kind, value = field.value.trim();
        if (kind === "list") content[key] = value ? value.split(/\r?\n/).map(item => item.trim()).filter(Boolean) : [];
        else if (kind === "number") { if (value) content[key] = Number(value); else delete content[key]; }
        else if (value) content[key] = value;
        else if (Object.hasOwn(content, key)) content[key] = "";
      }
      if (!Object.keys(content).length) return message("请填写至少一个结构化字段", true);
      const button = card.querySelector(".save-content");
      button.disabled = true;
      try {
        const saved = await api(`/api/v1/admin/fish/assets/species/${encodeURIComponent(speciesId)}/roles/${role}/content`, {
          method:"PUT", headers:jsonHeaders,
          body:JSON.stringify({version_id:selectedId,title:card.querySelector(".card-title").value,structured_content:content}),
        });
        const readback = await loadWorkspace(selectedId);
        const refreshed = readback.roles[role];
        if (refreshed.card_id !== saved.card_id || refreshed.content_revision !== saved.content_revision || refreshed.selected_version_id !== selectedId || refreshed.content_qa !== "PENDING") {
          throw new Error("保存请求已返回，但内容 revision / 精确版本绑定回读不一致；请先刷新核对后再重试");
        }
        message(`${LABEL[role]} 草稿已保存到 version #${selectedId}，revision ${saved.content_revision}；内容 QA 已重置为 PENDING，线上 ACTIVE 内容未改变。`);
        render(readback);
      } catch (error) { message(error.message, true); }
      finally { button.disabled = false; }
    });
  }
}

export async function start(initialId) {
  if (!await startBase(initialId) || !speciesId) return;
  try { render(await loadWorkspace()); } catch (error) { message(error.message, true); }
}
