import {ALL_ROLES, CARD_ROLES, LABEL, api, byId, escapeHtml, loadWorkspace, message, startBase, speciesId} from "./common.js";

const PRIMARY_ROLES = ["COVER_HERO", ...CARD_ROLES];
const EXTRA_ROLES = ["COVER_LIST", "TRANSPARENT_MAIN", "TRANSPARENT_ALT"];
let preflight = null;

function versionTile(version, tone) {
  if (!version) return '<p class="muted">暂无版本</p>';
  return `<div class="tile">
    <p><span class="status ${tone}">${escapeHtml(version.status)} ${version.legacy ? "兼容卡" : `v${version.version}`}</span> · ${version.legacy ? `FishCard #${version.id}` : `version #${version.id}`}</p>
    <img class="preview" src="${escapeHtml(version.preview_url)}" alt="${escapeHtml(version.asset_role)} ${escapeHtml(version.status)} v${version.version}" loading="lazy">
    <p class="muted">${escapeHtml(version.source_filename || "已上传图片")}</p>
  </div>`;
}

function roleTile(role, slot) {
  const versionedActive = slot.active_version || slot.active_versions?.[0] || null;
  const legacyActive = slot.legacy_active?.[0];
  const active = versionedActive || (legacyActive ? {id:legacyActive.card_id,version:"legacy",status:"ACTIVE",source_filename:"旧 ACTIVE FishCard",preview_url:legacyActive.image_url,legacy:true,asset_role:role} : null);
  const drafts = slot.draft_versions || [];
  return `<article class="tile asset-slot" data-role="${role}">
    <h3>${LABEL[role]} <span class="muted">${role}</span></h3>
    <div class="grid">
      <div><p>当前线上图片</p>${versionTile(active, "active")}</div>
      <div><p>编辑中的草稿${drafts.length > 1 ? `（${drafts.length} 个）` : ""}</p>${drafts.map(v => versionTile(v, "draft")).join("") || '<p class="muted">暂无 DRAFT</p>'}</div>
    </div>
    <p><button type="button" data-upload-role="${role}">上传此角色的新版本</button></p>
  </article>`;
}

function render(workspace) {
  byId("assetSlots").innerHTML = PRIMARY_ROLES.map(role => roleTile(role, workspace.roles[role])).join("");
  byId("otherAssetSlots").innerHTML = EXTRA_ROLES.map(role => roleTile(role, workspace.roles[role])).join("");
  document.querySelectorAll("[data-upload-role]").forEach(button => button.addEventListener("click", () => {
    byId("assetRole").value = button.dataset.uploadRole;
    byId("assetFile").focus();
    byId("uploadForm").scrollIntoView({behavior:"smooth", block:"center"});
  }));
}

function showPreflight(result) {
  preflight = result;
  const box = byId("uploadReview");
  const item = result.item || {};
  const warnings = item.validation_warnings || [];
  const errors = item.validation_errors || [];
  const isInvalid = result.validation_status === "INVALID" || item.validation_status === "INVALID";
  const isWarning = result.validation_status === "WARNING" || warnings.length > 0;
  const duplicateBlocked = errors.some(value => /DUPLICATE/.test(value.code || ""));
  const alreadyExists = warnings.some(value => value.code === "ASSET_ALREADY_EXISTS");
  const state = isInvalid ? (duplicateBlocked ? "DUPLICATE" : "INVALID") : isWarning ? (alreadyExists ? "ALREADY_EXISTS" : "WARNING") : "VALID";
  const messages = (isInvalid ? errors : warnings).map(value => `<li><b>${escapeHtml(value.code || "校验信息")}</b> · ${escapeHtml(value.message || JSON.stringify(value))}</li>`).join("") || "<li>图片预检通过</li>";
  box.hidden = false;
  box.innerHTML = `<h3>预检状态：<span class="status ${isInvalid ? "error" : isWarning ? "" : "active"}">${state}</span></h3>
    <p>鱼种：${escapeHtml(speciesId)} · 角色：${escapeHtml(item.asset_role || byId("assetRole").value)} · 文件：${escapeHtml(item.source_filename || "")}</p>
    <div class="grid"><div><p>源图片预览</p>${item.source_preview_url ? `<img class="preview" src="${escapeHtml(item.source_preview_url)}" alt="上传原图预览">` : '<p class="muted">预览地址不可用</p>'}</div>
      <div><p>预检结果</p><ul>${messages}</ul><p class="muted">${escapeHtml(item.width || "?")} × ${escapeHtml(item.height || "?")} · ${escapeHtml(item.file_size || "?")} bytes</p></div></div>
    ${isInvalid ? '<p class="status error">此文件被阻止，不会创建 DRAFT，也不会进入发布列表。</p><button type="button" data-dismiss-preflight>返回修改</button>' : `
      <p>${alreadyExists ? "同鱼种、同角色和同 SHA 的版本已存在。继续将回用该版本，不会重复写入图片对象。" : isWarning ? "请检查以上警告。只有明确确认后才会执行导入并创建或关联版本。" : "预检通过。查看预览后确认，才会创建 DRAFT。"}</p>
      <div class="row"><button type="button" class="primary" data-confirm-upload>${alreadyExists ? "确认关联已有版本" : isWarning ? "确认警告并继续" : "确认上传为 DRAFT"}</button><button type="button" data-dismiss-preflight>返回修改</button></div>`}`;
  box.querySelector("[data-dismiss-preflight]")?.addEventListener("click", () => { preflight = null; box.hidden = true; });
  box.querySelector("[data-confirm-upload]")?.addEventListener("click", executePreflight);
}

async function executePreflight() {
  if (!preflight?.batch_id) return message("预检批次不可用，请重新预检", true);
  const button = byId("uploadReview").querySelector("[data-confirm-upload]");
  if (button) { button.disabled = true; button.textContent = "UPLOADING…"; }
  try {
    const batchPath = `/api/v1/admin/fish/assets/import-batches/${encodeURIComponent(preflight.batch_id)}`;
    let batch = await api(batchPath);
    if (batch.status === "READY") {
      batch = await api(`${batchPath}/execute`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({allow_warnings:preflight.validation_status === "WARNING" || preflight.needs_warning_confirmation === true}),
      });
    } else if (batch.status === "FAILED" || batch.status === "COMPLETED") {
      await api(`${batchPath}/retry`, {method:"POST"});
      batch = await api(batchPath);
    } else if (batch.status === "IMPORTING" || batch.status === "SCANNING") {
      throw new Error(`批次当前状态为 ${batch.status}，请稍后回读批次与角色版本；不要重新选图重复上传。`);
    } else throw new Error(`当前批次状态 ${batch.status} 不支持执行，请在批量导入诊断中核查。`);
    const row = (batch.items || []).find(item => item.species_id === speciesId && item.asset_role === byId("assetRole").value);
    if (!row?.version_id) {
      const state = (row?.validation_errors || []).some(value => /DUPLICATE/.test(value.code || "")) ? "DUPLICATE" : "UPLOAD_FAILED";
      byId("uploadReview").insertAdjacentHTML("beforeend", `<p class="status error">${state} · ${escapeHtml((row?.validation_errors || []).map(x => x.message || x.code).join("；") || "导入没有返回素材版本")}</p>`);
      if (button) { button.disabled = false; button.textContent = "重试上传"; }
      return;
    }
    const persisted = await loadWorkspace(Number(row.version_id));
    const slot = persisted.roles[byId("assetRole").value];
    const saved = [...slot.active_versions, ...slot.draft_versions].find(version => version.id === Number(row.version_id));
    if (!saved || !saved.preview_url || !["DRAFT", "ACTIVE"].includes(saved.status)) {
      const error = new Error("上传结果无法从 CMS 数据库回读；未确认版本状态，请先刷新或联系管理员核查。");
      error.code = "UPLOAD_READBACK_FAILED";
      throw error;
    }
    const isExisting = (row.validation_warnings || []).some(value => value.code === "ASSET_ALREADY_EXISTS");
    let previewReadable = false;
    try { const previewResponse = await fetch(saved.preview_url, {credentials:"same-origin"}); previewReadable = previewResponse.ok; } catch { previewReadable = false; }
    const finalState = isExisting ? "ALREADY_EXISTS" : saved.status === "DRAFT" ? "DRAFT_CREATED" : "ALREADY_EXISTS";
    byId("uploadReview").insertAdjacentHTML("beforeend", `<p class="status ${previewReadable ? "active" : "error"}">${finalState} · 真实 version_id #${saved.id} · ${escapeHtml(saved.status)}${previewReadable ? " · 预览可读" : " · 图片无法访问"}</p>`);
    byId("assetFile").value = "";
    preflight = null;
    render(persisted);
    message(`${finalState}：version_id #${saved.id}，数据库状态 ${saved.status}${previewReadable ? "，预览地址可读" : "，但图片预览读取失败"}`, !previewReadable);
  } catch (error) {
    if (button) { button.disabled = false; button.textContent = "重试或核对版本状态"; }
    byId("uploadReview").insertAdjacentHTML("beforeend", `<p class="status error">${error.code === "UPLOAD_READBACK_FAILED" ? "上传状态未核实" : "UPLOAD_FAILED"} · ${escapeHtml(error.message)}。文件仍保留，可重试；若服务器已提交，请先检查角色历史避免重复上传。</p>`);
    message(error.message, true);
  }
}

export async function start(initialId) {
  if (!await startBase(initialId) || !speciesId) return;
  byId("uploadSpecies").value = speciesId;
  byId("batchImportLink").href = `/fish-knowledge/${encodeURIComponent(speciesId)}/batch-import`;
  byId("extensionsLink").href = `/fish-knowledge/${encodeURIComponent(speciesId)}/extensions`;
  byId("assetRole").innerHTML = `<optgroup label="主视觉与知识卡">${PRIMARY_ROLES.map(role => `<option value="${role}">${LABEL[role]} · ${role}</option>`).join("")}</optgroup><optgroup label="其他保留角色">${EXTRA_ROLES.map(role => `<option value="${role}">${LABEL[role]} · ${role}</option>`).join("")}</optgroup>`;
  try { render(await loadWorkspace()); } catch (error) { message(error.message, true); }
  byId("uploadForm").addEventListener("submit", async event => {
    event.preventDefault();
    const file = byId("assetFile").files[0];
    if (!file) return message("请选择图片文件", true);
    const button = byId("preflightButton");
    button.disabled = true; button.textContent = "UPLOADING / PREFLIGHT…";
    message("UPLOADING：正在上传到预检批次，尚未创建资产 DRAFT。");
    const data = new FormData();
    data.append("species_id", speciesId);
    data.append("asset_role", byId("assetRole").value);
    data.append("file", file, file.name);
    data.append("preflight_only", "true");
    try {
      const result = await api("/api/v1/admin/fish/assets/single-upload", {method:"POST", body:data});
      showPreflight(result);
      if (result.validation_status === "INVALID") {
        const duplicate = (result.item?.validation_errors || []).some(value => /DUPLICATE/.test(value.code || ""));
        message(`${duplicate ? "DUPLICATE" : "INVALID"}：${(result.item?.validation_errors || []).map(x => x.message || x.code).join("；") || "文件不符合要求"}`, true);
      }
      else if (result.validation_status === "WARNING") message("WARNING：已展示预览和警告。确认前不会执行导入或创建 DRAFT。");
      else message("VALID：预检通过，查看原图预览后再确认创建 DRAFT。");
    } catch (error) {
      byId("uploadReview").hidden = false;
      byId("uploadReview").innerHTML = `<h3>UPLOAD_FAILED</h3><p>${escapeHtml(error.message)}</p><p>已保留所选文件，可重试预检。</p>`;
      message(error.message, true);
    } finally { button.disabled = false; button.textContent = "重新预检"; }
  });
}
