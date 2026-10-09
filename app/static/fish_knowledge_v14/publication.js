import {ALL_ROLES, CARD_ROLES, LABEL, api, byId, escapeHtml, loadWorkspace, message, startBase, speciesId} from "./common.js";

const apiCheckState = new Map();

function preview(version, title) {
  return version ? `<div class="tile"><p>${title}</p><p><span class="status ${version.status === "ACTIVE" ? "active" : "draft"}">${version.status} v${version.version}</span> · version_id #${version.id}</p><img class="preview" src="${escapeHtml(version.preview_url)}" alt="${escapeHtml(title)}"></div>` : `<div class="tile"><p>${title}</p><p class="muted">无版本</p></div>`;
}

function qaPanel(stage, version, allow, current) {
  const label = stage === "VISUAL" ? "视觉 QA" : "内容 QA";
  const key = stage.toLowerCase();
  const resultField = `${key}_qa_result`;
  const noteField = `${key}_qa_note`;
  const reviewerField = `${key}_qa_reviewer`;
  const stampField = `${key}_qa_reviewed_at`;
  const blockedValue = stage === "VISUAL" ? "BLOCKED_VISUAL_QA" : "BLOCKED_CONTENT_MISMATCH";
  const currentResult = current === blockedValue ? blockedValue : current === "PASS" ? "PASS" : "";
  return `<section class="tile">
    <h4>${label} · ${escapeHtml(current || "PENDING")}</h4>
    <p class="muted">${current === "PASS" ? `${escapeHtml(version.review?.[reviewerField] || "未知审核人")} · ${escapeHtml(version.review?.[stampField] || "无时间记录")}` : stage === "CONTENT" ? "核对本 DRAFT 版本绑定的结构化内容、鱼种与角色。" : "对照 ACTIVE 与 DRAFT 图片，记录清晰度、主体、裁切和规范核验。"}</p>
    <label>审核结果<select class="qa-result" data-stage="${stage}" ${allow ? "" : "disabled"}><option value="" ${currentResult ? "" : "selected"}>选择结果…</option><option value="PASS" ${currentResult === "PASS" ? "selected" : ""}>PASS</option><option value="${blockedValue}" ${currentResult === blockedValue ? "selected" : ""}>BLOCKED</option></select></label>
    <label>审核人<input class="qa-reviewer wide" data-stage="${stage}" value="${escapeHtml(version.review?.[reviewerField] || "")}" placeholder="必填"></label>
    <label>审核备注与证据<textarea class="qa-note" data-stage="${stage}" placeholder="记录判断依据及发现，例如所核版本、画面或内容字段。" ${allow ? "" : "disabled"}>${escapeHtml(version.review?.[noteField] || "")}</textarea></label>
    <button type="button" class="save-qa" data-stage="${stage}" ${allow ? "" : "disabled"}>保存${label}</button>
  </section>`;
}

function render(workspace) {
  byId("publicationSlots").innerHTML = ALL_ROLES.map(role => {
    const slot = workspace.roles[role];
    const active = slot.active_versions || [];
    const drafts = slot.draft_versions || [];
    const legacyCards = slot.legacy_active || [];
    const activeSummary = active.map(v => `<p><span class="status active">已发布至数据库 ACTIVE · v${v.version}</span> · version_id #${v.id}</p>`).join("") || (legacyCards.length ? legacyCards.map(card => `<p><span class="status active">兼容 ACTIVE FishCard</span> · #${card.card_id}</p>`).join("") : '<p class="muted">无版本化 ACTIVE</p>');
    const draftMarkup = drafts.map(version => {
      const review = version.review || {};
      const validation = review.validation_result || "PENDING";
      const visual = review.visual_qa_result || "PENDING";
      const content = review.content_qa_result || "PENDING";
      const bindingStatus = slot.binding_status || "ROLE_ONLY";
      const hasContentBinding = !CARD_ROLES.includes(role) || (bindingStatus === "BOUND_DRAFT" && Boolean(slot.card_id));
      const publishReady = validation === "PASS" && visual === "PASS" && content === "PASS" && hasContentBinding;
      const current = active[0] || (legacyCards[0] ? {id:legacyCards[0].card_id,version:"legacy",status:"ACTIVE",preview_url:legacyCards[0].image_url,legacy:true} : null);
      const expected = role === "COVER_HERO"
        ? `species list + detail: cover_hero_image, cover_hero_version_id=#${version.id}, cover_hero_status=ACTIVE`
        : CARD_ROLES.includes(role)
          ? `detail: knowledge_assets[${role}] and cards[] both version_id=#${version.id}, same species / role / image URL`
          : `detail: cover_assets[${role}] version_id=#${version.id}, status=ACTIVE`;
      const legacy = (slot.legacy_active || []).map(card => `旧 ACTIVE FishCard #${card.card_id} 将与新绑定在同一事务中切换为 DRAFT。`).join(" ");
      return `<div class="tile publication-draft" data-role="${role}" data-version-id="${version.id}">
        <h4>待审 DRAFT v${version.version} · version_id #${version.id}</h4>
        <div class="grid">${preview(current, "当前 ACTIVE 线上图片")}${preview(version, "待发布 DRAFT 预览")}</div>
        <p><b>素材校验：</b>${escapeHtml(validation)} · <b>视觉 QA：</b>${escapeHtml(visual)} · <b>内容 QA：</b>${escapeHtml(content)}</p>
        <p><b>绑定：</b>${escapeHtml(bindingStatus)}${slot.card_id ? ` · FishCard #${slot.card_id}` : ""}${slot.content_revision ? ` · revision ${slot.content_revision}` : ""} · species_id ${escapeHtml(speciesId)} · role ${role}</p>
        <details><summary>来源、GCS 与内容证据</summary><p>来源 SHA-256：<code>${escapeHtml(version.source_sha256 || "—")}</code></p><p>GCS generation：<code>${escapeHtml(version.object_generation || "—")}</code></p><p>GCS 对象：<code>${escapeHtml(version.object_name || "—")}</code></p><p>审核记录：${escapeHtml(review.visual_qa_reviewer || "—")} / ${escapeHtml(review.visual_qa_reviewed_at || "未做视觉 QA")}；${escapeHtml(review.content_qa_reviewer || "—")} / ${escapeHtml(review.content_qa_reviewed_at || "未做内容 QA")}</p><p>视觉证据：${escapeHtml(review.visual_qa_note || "—")}</p><p>内容证据：${escapeHtml(review.content_qa_note || "—")}</p></details>
        <div class="grid">${qaPanel("VISUAL", version, validation === "PASS", visual)}${qaPanel("CONTENT", version, validation === "PASS" && visual === "PASS", content)}</div>
        <section class="tile"><h4>发布前核对</h4><ul><li>鱼种：${escapeHtml(slot.species_name_cn || workspace.species_name_cn || speciesId)} (${escapeHtml(speciesId)})</li><li>角色：${LABEL[role]} (${role})</li><li>当前 ACTIVE：${current ? current.legacy ? `兼容 FishCard #${current.id}` : `version_id #${current.id} · v${current.version}` : "无"}</li><li>目标 DRAFT：version_id #${version.id} · v${version.version}</li><li>图片：上方同时预览新旧版本；发布前核对来源 SHA 与 GCS generation。</li><li>内容绑定：${escapeHtml(bindingStatus)}${slot.content_revision ? ` · revision ${slot.content_revision}` : ""}</li><li>公共 API 预计字段：${escapeHtml(expected)}</li><li>${escapeHtml(legacy)}</li></ul>
          <label><input class="prepublish-check" type="checkbox" ${publishReady ? "" : "disabled"}> 我已核对鱼种、角色、ACTIVE / DRAFT 版本、图片、来源、generation、绑定、QA 和 API 影响</label>
          <button type="button" class="primary publish" ${publishReady ? "disabled" : "disabled"}>确认发布 version_id #${version.id}</button>
        </section>
      </div>`;
    }).join("") || '<p class="muted">无待审核 DRAFT</p>';
    const apiCheck = active.map(v => {
      const audit = (slot.history.publication_audits || []).find(item => item.version_id === v.id);
      const recorded = audit?.validation?.public_api_check;
      const client = audit?.validation?.client_acceptance;
      const recordedLabel = recorded?.status === "PUBLIC_API_OK"
        ? `已发布至公共 API · PUBLIC_API_OK · ${recorded.checked_at || ""} · ${recorded.reviewer || ""}`
        : recorded?.status === "IMAGE_UNREADABLE"
          ? `图片无法访问 · ${recorded.detail || ""}`
          : recorded?.status === "API_MISMATCH"
            ? `公共 API 校验不通过 · ${recorded.detail || ""}`
            : "尚未执行公共 API 实测";
      const readbackEvidence = recorded ? `<details><summary>公共 API 与图片响应证据</summary><p>核验 version_id：${escapeHtml(recorded.observed_version_id ?? "—")} · DB version_id：${v.id}</p><p>公共图片 SHA-256：<code>${escapeHtml(recorded.public_image_sha256 || "—")}</code></p><p>目标版本预览 SHA-256：<code>${escapeHtml(recorded.preview_image_sha256 || "—")}</code></p><p>检查人 / 时间：${escapeHtml(recorded.reviewer || "—")} · ${escapeHtml(recorded.checked_at || "—")}</p></details>` : "";
      const clientLabel = client?.status === "CLIENT_PASSED" ? `客户端实测通过 · ${client.checked_at || ""} · ${client.reviewer || ""}` : client?.status === "CLIENT_FAILED" ? `客户端实测未通过 · ${client.detail || ""}` : "等待 Android 客户端确认";
      const clientEvidence = client ? `<details><summary>客户端实测记录</summary><p>检查 version_id：${escapeHtml(client.observed_version_id ?? "—")} · 审核人：${escapeHtml(client.reviewer || "—")} · 时间：${escapeHtml(client.checked_at || "—")}</p><p>${escapeHtml(client.detail || "无补充说明")}</p></details>` : "";
      return `<div class="tile"><p>API / 图片生效检查：${escapeHtml(apiCheckState.get(String(v.id)) || recordedLabel)}</p>${readbackEvidence}<label>API 检查人<input class="api-reviewer wide" data-version-id="${v.id}" value="${escapeHtml(recorded?.reviewer || "")}" placeholder="必填"></label><button type="button" class="verify-api" data-version-id="${v.id}" data-role="${role}">检查公共 API 与图片</button><hr><p>Android 客户端：<span class="status">${escapeHtml(clientLabel)}</span></p>${clientEvidence}<label>客户端实测人<input class="client-reviewer wide" data-version-id="${v.id}" value="${escapeHtml(client?.reviewer || "")}" placeholder="由独立客户端验收人填写"></label><label>客户端实测说明<textarea class="client-detail" data-version-id="${v.id}" placeholder="记录设备/应用版本与展示结果">${escapeHtml(client?.detail || "")}</textarea></label><button type="button" class="verify-client" data-version-id="${v.id}" data-result="CLIENT_PASSED">记录客户端实测通过</button><button type="button" class="verify-client" data-version-id="${v.id}" data-result="CLIENT_FAILED">记录客户端未通过</button></div>`;
    }).join("");
    return `<article class="tile publication-role" data-role="${role}"><h3>${LABEL[role]} <span class="muted">${role}</span></h3>${activeSummary}${draftMarkup}${apiCheck}</article>`;
  }).join("");

  for (const card of document.querySelectorAll(".publication-draft")) {
    const versionId = Number(card.dataset.versionId);
    const role = card.dataset.role;
    for (const button of card.querySelectorAll(".save-qa")) button.addEventListener("click", async () => {
      const stage = button.dataset.stage;
      const result = card.querySelector(`.qa-result[data-stage="${stage}"]`).value;
      const reviewer = card.querySelector(`.qa-reviewer[data-stage="${stage}"]`).value.trim();
      const note = card.querySelector(`.qa-note[data-stage="${stage}"]`).value.trim();
      if (!result || !reviewer || !note) return message("请填写此阶段的结果、审核人及备注/证据", true);
      const slot = workspace.roles[role];
      const version = slot.draft_versions.find(value => value.id === versionId);
      if (!version?.batch_id) return message("该素材没有审核批次，无法保存 QA", true);
      const payload = {batch_id:version.batch_id};
      if (stage === "VISUAL") Object.assign(payload, {visual_qa_result:result, visual_qa_note:note, visual_qa_reviewer:reviewer});
      else Object.assign(payload, {content_qa_result:result, content_qa_note:note, content_qa_reviewer:reviewer});
      button.disabled = true;
      try {
        await api(`/api/v1/admin/fish/assets/versions/${versionId}/review`, {method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
        message(`${stage === "VISUAL" ? "视觉" : "内容"} QA 已单独记录：${role} version #${versionId}`);
        render(await loadWorkspace());
      } catch (error) { message(error.message, true); }
      finally { button.disabled = false; }
    });
    const check = card.querySelector(".prepublish-check");
    const publish = card.querySelector(".publish");
    check?.addEventListener("change", () => { publish.disabled = !check.checked; });
    publish?.addEventListener("click", async () => {
      const slot = workspace.roles[role];
      const current = slot.active_version;
      const confirmation = `发布确认\n鱼种：${slot.species_name_cn || workspace.species_name_cn || speciesId} (${speciesId})\n角色：${role}\n当前 ACTIVE version_id：${current?.id ?? "无"}\n目标 DRAFT version_id：${versionId}\n\n确认只发布 version_id ${versionId} 吗？`;
      if (!window.confirm(confirmation)) return;
      publish.disabled = true;
      try {
        const result = await api(`/api/v1/admin/fish/assets/versions/${versionId}/activate`, {method:"POST"});
        if (!result.success || result.version_id !== versionId || result.publication_status !== "ACTIVE") throw new Error("发布响应与指定 version_id 不一致，请核对历史记录");
        message(`${role} version_id #${versionId} 已写入 ACTIVE。接下来仍需单独检查公共 API 与图片访问；Android 客户端保持等待验收。`);
        render(await loadWorkspace());
      } catch (error) {
        if (error.code === "PUBLICATION_COMMITTED_READBACK_FAILED" || error.detail?.publication_committed) {
          message(`数据库发布已提交，但公共详情后置核验失败。目标 version_id #${versionId} 状态需立即复核；不得按“回滚未发布”处理。${error.message}`, true);
        } else message(error.message, true);
        try { render(await loadWorkspace()); } catch { /* keep the explicit server result visible */ }
      } finally { publish.disabled = false; }
    });
  }
  for (const button of document.querySelectorAll(".verify-api")) button.addEventListener("click", () => verifyPublicApi(Number(button.dataset.versionId), button.dataset.role, button));
  for (const button of document.querySelectorAll(".verify-client")) button.addEventListener("click", () => recordClientAcceptance(Number(button.dataset.versionId), button.dataset.result, button));
}

async function sha256(blob) {
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, "0")).join("");
}

async function recordClientAcceptance(versionId, result, button) {
  const panel = button.closest(".tile");
  const reviewer = panel?.querySelector(`.client-reviewer[data-version-id="${versionId}"]`)?.value.trim() || "";
  const detail = panel?.querySelector(`.client-detail[data-version-id="${versionId}"]`)?.value.trim() || "";
  if (!reviewer || !detail) return message("请填写客户端实测人及设备/应用版本和展示结果", true);
  button.disabled = true;
  try {
    const saved = await api(`/api/v1/admin/fish/assets/versions/${versionId}/client-check`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({result, reviewer, observed_version_id:versionId, detail}),
    });
    message(`${saved.status} 已记录：${saved.asset_role} version_id #${versionId} · ${saved.reviewer}`);
    render(await loadWorkspace(versionId));
  } catch (error) { message(`客户端实测结果未核实或未保存：${error.message}`, true); }
  finally { button.disabled = false; }
}

async function verifyPublicApi(versionId, role, button) {
  button.disabled = true; button.textContent = "正在检查…";
  const reviewer = button.closest(".tile")?.querySelector(`.api-reviewer[data-version-id="${versionId}"]`)?.value.trim() || "";
  if (!reviewer) { button.disabled = false; button.textContent = "检查公共 API 与图片"; return message("请填写检查人以保存 API / 图片核验审计", true); }
  let responseChecksPassed = false;
  let imageCheckStarted = false;
  try {
    const workspace = await loadWorkspace(versionId);
    const active = workspace.roles[role].active_versions.find(value => value.id === versionId);
    if (!active || active.status !== "ACTIVE") throw new Error("数据库未回读到目标 ACTIVE 版本");
    const [listRows, detail] = await Promise.all([
      api("/api/v1/fish/species"),
      api(`/api/v1/fish/species/${encodeURIComponent(speciesId)}/detail`),
    ]);
    if (role === "COVER_HERO") {
      const list = listRows.find(item => item.id === speciesId);
      if (!list || list.cover_hero_status !== "ACTIVE" || list.cover_hero_version_id !== versionId || list.cover_hero_image !== active.image_url || detail.cover_hero_status !== "ACTIVE" || detail.cover_hero_version_id !== versionId || detail.cover_hero_image !== active.image_url) {
        throw new Error("公共 API 列表与详情的 COVER_HERO 版本/URL/ACTIVE 状态不一致");
      }
    } else if (CARD_ROLES.includes(role)) {
      const asset = detail.knowledge_assets?.[role];
      const card = (detail.cards || []).find(item => item.card_type === role);
      if (!asset || asset.asset_status !== "ACTIVE" || asset.version_id !== versionId || asset.asset_role !== role || asset.image_url !== active.image_url || !card || card.species_id !== speciesId || card.card_type !== role || card.status !== "ACTIVE" || card.asset_version_id !== versionId || card.image_url !== active.image_url) {
        throw new Error(`公共 API knowledge_assets[${role}] 与 cards[] 没有对应同一 ACTIVE version_id`);
      }
    } else {
      const asset = detail.cover_assets?.[role];
      if (!asset || asset.asset_status !== "ACTIVE" || asset.version_id !== versionId || asset.image_url !== active.image_url) throw new Error(`公共 API 没有返回 ${role} 的指定 ACTIVE 版本`);
    }
    imageCheckStarted = true;
    const [publicResponse, previewResponse] = await Promise.all([
      fetch(active.image_url, {credentials:"same-origin"}),
      fetch(active.preview_url, {credentials:"same-origin"}),
    ]);
    if (!publicResponse.ok || !previewResponse.ok) throw new Error(`图片无法访问：公共 URL ${publicResponse.status}，CMS 目标预览 ${previewResponse.status}`);
    const [publicImage, previewImage] = await Promise.all([publicResponse.blob(), previewResponse.blob()]);
    if (!publicImage.size || !previewImage.size) throw new Error("图片响应为空");
    const [publicHash, previewHash] = await Promise.all([sha256(publicImage), sha256(previewImage)]);
    if (publicHash !== previewHash) throw new Error("公共图片响应与目标 version_id 的 CMS 预览字节不同");
    responseChecksPassed = true;
    const saved = await api(`/api/v1/admin/fish/assets/versions/${versionId}/public-api-check`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({status:"PUBLIC_API_OK", reviewer, observed_version_id:versionId, public_image_sha256:publicHash, preview_image_sha256:previewHash}),
    });
    apiCheckState.set(String(versionId), `PUBLIC_API_OK · ${saved.checked_at} · version_id #${versionId} · image SHA-256 ${publicHash}`);
    message(`${role} PUBLIC_API_OK 已写入发布审计：公共 API、ACTIVE 状态和图片字节均对应 version_id #${versionId}。Android 客户端仍等待独立验收。`);
    try { render(await loadWorkspace(versionId)); }
    catch (error) { message(`PUBLIC_API_OK 已保存；发布审核页面回读失败，请刷新页面。${error.message}`, true); }
  } catch (error) {
    const serverProjectionMismatch = ["PUBLIC_API_IMAGE_MISMATCH", "PUBLIC_API_PROJECTION_MISMATCH"].includes(error.code);
    if (responseChecksPassed && !serverProjectionMismatch) {
      const state = `API 与图片已核验，但审计保存失败 · ${error.message}`;
      apiCheckState.set(String(versionId), state);
      message(state, true);
      try { render(await loadWorkspace(versionId)); } catch { /* retain the unsaved status in the current page */ }
    } else {
      const imageFailure = imageCheckStarted && !serverProjectionMismatch;
      const state = imageFailure ? "图片无法访问" : "API 校验不通过";
      const status = imageFailure ? "IMAGE_UNREADABLE" : "API_MISMATCH";
      let recorded = false;
      try {
        await api(`/api/v1/admin/fish/assets/versions/${versionId}/public-api-check`, {
          method:"POST", headers:{"Content-Type":"application/json"},
          body:JSON.stringify({status, reviewer, detail:error.message}),
        });
        recorded = true;
      } catch { /* keep the failure visible if the audit endpoint is also unavailable */ }
      apiCheckState.set(String(versionId), `${state} · ${error.message}${recorded ? " · 已记入发布审计" : " · 审计未保存"}`);
      message(`${state}：${error.message}${recorded ? "。结果已记入发布审计。" : "。发布审计服务不可用，结果未保存。"}`, true);
      try { render(await loadWorkspace(versionId)); } catch { /* retain status in current page */ }
    }
  } finally { button.disabled = false; button.textContent = "检查公共 API 与图片"; }
}

export async function start(initialId) {
  if (!await startBase(initialId) || !speciesId) return;
  try { render(await loadWorkspace()); } catch (error) { message(error.message, true); }
}
