(() => {
  "use strict";

  const PROCESSING_MS = 2800;
  const RR09_MS = 850;
  const PROCESSING_FRAMES = [
    { id: "P0", at: 0 }, { id: "P1", at: 350 }, { id: "P2", at: 950 },
    { id: "P3", at: 1200 }, { id: "P4", at: 1550 }, { id: "P5", at: 2050 },
    { id: "P6", at: 2600 }, { id: "P7", at: 2800 },
  ];
  const RR09_FRAMES = [
    { id: "R0", at: 0 }, { id: "R0.5", at: 100 }, { id: "R1", at: 200 },
    { id: "R2", at: 380 }, { id: "R3", at: 550 }, { id: "R4", at: 680 },
    { id: "R5", at: 850 },
  ];
  const MAX_FILE_BYTES = 15 * 1024 * 1024;
  const ACCEPTED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
  const app = document.getElementById("catchRecognitionApp");
  if (!app) return;

  const $ = (id) => document.getElementById(id);
  const core = window.YujianCatch;
  const state = new core.RecognitionController();

  const nodes = {
    status: $("catchLiveStatus"), grid: $("catchStageGrid"), input: $("catchFileInput"), photoStage: $("catchPhotoStage"),
    drop: $("catchDrop"), start: $("catchStartButton"), clear: $("catchClearButton"),
    uploadPreview: $("catchUploadPreview"), uploadPhoto: $("catchUploadPhoto"),
    fileMeta: $("catchFileMeta"), fileName: $("catchFileName"), fileSize: $("catchFileSize"),
    uploadBadge: $("catchUploadBadge"), processingPanel: document.querySelector('[data-stage="processing"]'),
    resultPanel: document.querySelector('[data-stage="result"]'), processingMedia: $("catchProcessingMedia"),
    processingPhoto: $("catchProcessingPhoto"), processingEmpty: $("catchProcessingEmpty"),
    processingBadge: $("catchProcessingBadge"), businessMessage: $("catchBusinessMessage"),
    dimensions: $("catchPhotoDimensions"), taskList: $("catchTaskList"), diagnostic: $("catchDiagnostic"),
    bbox: $("catchBBoxOverlay"), regionGlow: $("catchRegionGlow"), resultBadge: $("catchResultBadge"),
    resultEmpty: $("catchResultEmpty"), resultContent: $("catchResultContent"), resultMedia: $("catchResultMedia"),
    resultPhoto: $("catchResultPhoto"), resultBackdrop: $("catchResultBackdrop"), speciesLabel: $("catchSpeciesLabel"), speciesName: $("catchSpeciesName"),
    confidence: $("catchConfidence"), candidates: $("catchCandidates"), speciesSelect: $("catchSpeciesSelect"),
    saveHint: $("catchSaveHint"), saveButton: $("catchSaveButton"), memoryButton: $("catchMemoryButton"),
    timingNote: $("catchTimingNote"), loginBackdrop: $("catchLoginBackdrop"), loginForm: $("catchLoginForm"),
    loginError: $("catchLoginError"), loginCancel: $("catchLoginCancel"), loginSubmit: $("catchLoginSubmit"),
  };

  const taskNames = {
    input_validation: ["input_validation", "image_load"],
    detector: ["detector", "boundary_gate"],
    crop: ["crop"],
    classifier: ["classifier"],
  };
  const swosRenderer = core.createSwosRenderer({
    media: nodes.processingMedia, host: $("catchPhotoEffects"), focus: nodes.regionGlow,
  });

  function setStatus(message, kind = "") {
    nodes.status.textContent = message;
    nodes.status.className = `catch-live-status${kind ? ` ${kind}` : ""}`;
  }

  function isChineseName(value) {
    return typeof value === "string" && /[\u3400-\u9fff]/u.test(value.trim());
  }

  function isReducedMotion() {
    return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
  }

  function delay(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, Math.max(0, ms)));
  }

  function updateStageNav(activeStage, completed = null) {
    const finished = completed || (state.inference ? ["upload", "processing"] : state.busy ? ["upload"] : []);
    app.dataset.activeStage = activeStage;
    core.renderJourney(app, state);
    document.querySelectorAll("[data-stage-step]").forEach((node) => {
      const stage = node.getAttribute("data-stage-step");
      node.classList.toggle("active", stage === activeStage);
      if (stage === activeStage) node.setAttribute("aria-current", "step"); else node.removeAttribute("aria-current");
      node.classList.toggle("done", finished.includes(stage));
      node.disabled = stage === "processing" ? !state.file : stage === "result" ? !(state.inference || state.phase === "ERROR") : false;
    });
    document.querySelectorAll(".catch-panel[data-stage]").forEach((panel) => {
      const stage = panel.getAttribute("data-stage");
      panel.classList.toggle("is-active", stage === activeStage);
      panel.classList.toggle("is-ready", finished.includes(stage));
      const mobile = window.matchMedia("(max-width: 767px)").matches;
      const handoffSource = stage === "processing" && nodes.grid.classList.contains("handoff");
      panel.hidden = mobile && stage !== activeStage && !handoffSource;
    });
  }

  function setBadge(node, label, tone = "neutral") {
    const toneClass = {
      success: "status-success", running: "status-running", failed: "status-failed",
      review: "status-review", neutral: "status-neutral",
    }[tone] || "status-neutral";
    node.className = `status-tag ${toneClass}`;
    node.textContent = label;
  }

  function formatFileSize(bytes) {
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function clearTasks() {
    nodes.taskList.querySelectorAll("[data-task]").forEach((item) => {
      item.classList.remove("done", "failed", "running");
    });
  }

  function markTasksFromResult(result) {
    const stages = new Map((result.timing?.stages || []).map((stage) => [stage.name, stage]));
    Object.entries(taskNames).forEach(([task, sourceNames]) => {
      const rows = sourceNames.map((name) => stages.get(name)).filter(Boolean);
      const target = nodes.taskList.querySelector(`[data-task="${task}"]`);
      if (!target || rows.length === 0) return;
      const status = rows.some((row) => row.status === "ERROR")
        ? "failed"
        : rows.some((row) => row.status === "SUCCESS")
          ? "done"
          : rows.some((row) => row.status === "RUNNING") ? "running" : "";
      if (status) target.classList.add(status);
    });
  }

  function setInProgressTasks() {
    clearTasks();
    // The current API does not stream individual stage events. Keep the list
    // unclaimed while it runs and show only the real terminal timings after it
    // responds; do not invent a percentage or a per-stage completion order.
  }

  function hidePreviousOutcome() {
    nodes.resultContent.classList.remove("visible");
    nodes.resultContent.hidden = true;
    nodes.resultEmpty.hidden = false;
    nodes.resultEmpty.innerHTML = '<div><span class="catch-empty-icon" aria-hidden="true"><svg viewBox="0 0 32 32"><path d="M4 17c5-7 12-10 21-7l5 7-5 7C16 27 9 24 4 17Zm20-1h.1M9 12 5 8m4 14-4 4" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"/></svg></span><strong>结果会显示在这里</strong><br>识别状态由真实模型响应决定。</div>';
    nodes.resultPhoto.hidden = true;
    nodes.resultBadge.textContent = "等待识别";
    nodes.resultBadge.className = "status-tag status-neutral";
    nodes.speciesName.textContent = "";
    nodes.speciesName.style.visibility = "visible";
    delete nodes.resultPanel.dataset.revealState;
    nodes.saveButton.textContent = "保存本次鱼获";
    nodes.confidence.textContent = "";
    nodes.confidence.className = "catch-confidence";
    nodes.candidates.hidden = true;
    nodes.candidates.replaceChildren();
    nodes.bbox.classList.remove("visible");
    swosRenderer.stop();
    nodes.speciesSelect.innerHTML = '<option value="">等待真实识别结果</option>';
    nodes.speciesSelect.disabled = true;
    nodes.saveButton.disabled = true;
    nodes.memoryButton.disabled = true;
    nodes.timingNote.textContent = "名义 Processing 2800ms + RR09 850ms = 3650ms；真实识别较慢时等待真实结果，不提前揭晓。";
    nodes.resultPanel.dataset.businessState = "idle";
    nodes.resultPanel.removeAttribute("data-result-state");
    nodes.resultPanel.classList.remove("revealing");
    state.transitionState = "idle";
    state.revealState = "idle";
    state.inference = null;
    state.bbox = null;
    if (state.photo) { state.photo.bbox = null; state.photo.result = null; }
    state.saved = false;
    state.selectedByUser = false;
    $("catchLength").value = "";
    $("catchWeight").value = "";
    $("catchLocation").value = "";
    $("catchStory").value = "";
    setCapturedAtNow();
  }

  function setCapturedAtNow() {
    const now = new Date();
    now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
    $("catchCapturedAt").value = now.toISOString().slice(0, 16);
  }

  function setPhotoUrl(url) {
    [nodes.uploadPhoto, nodes.processingPhoto, nodes.resultPhoto].forEach((image) => {
      image.src = url;
    });
    nodes.resultBackdrop.src = url;
    nodes.resultBackdrop.hidden = false;
    nodes.photoStage.hidden = false;
    nodes.processingPhoto.hidden = false;
    nodes.processingEmpty.hidden = true;
    nodes.resultPhoto.hidden = false;
  }

  function clearSelectedPhoto({keepStatus = false} = {}) {
    if (state.saveInProgress) { setStatus("鱼获正在保存，请稍候再更换照片。"); return; }
    state.invalidate();
    nodes.input.disabled = false;
    nodes.grid.classList.remove("handoff");
    $("catchCancelButton").hidden = true;
    closeLogin();
    state.objectUrl = null;
    state.file = null;
    nodes.input.value = "";
    nodes.uploadPreview.classList.remove("visible");
    nodes.uploadPhoto.removeAttribute("src");
    nodes.fileMeta.hidden = true;
    nodes.start.disabled = true;
    nodes.clear.disabled = true;
    nodes.processingPhoto.removeAttribute("src");
    nodes.processingPhoto.hidden = true;
    nodes.photoStage.hidden = true;
    nodes.processingEmpty.hidden = false;
    nodes.resultPhoto.removeAttribute("src");
    nodes.resultBackdrop.removeAttribute("src");
    nodes.resultBackdrop.hidden = true;
    swosRenderer.stop();
    state.bbox = null;
    nodes.regionGlow.classList.remove("visible");
    nodes.bbox.classList.remove("visible");
    nodes.processingMedia.classList.remove("processing");
    nodes.processingPanel.dataset.businessState = "idle";
    nodes.businessMessage.textContent = "等待真实推理";
    nodes.dimensions.textContent = "—";
    nodes.diagnostic.textContent = "尚无推理数据。";
    clearTasks();
    setBadge(nodes.uploadBadge, "待上传", "neutral");
    setBadge(nodes.processingBadge, "等待照片", "neutral");
    hidePreviousOutcome();
    updateStageNav("upload");
    if (!keepStatus) setStatus("照片已清除。选择一张真实鱼获照片即可重新开始。");
  }

  async function chooseFile(file) {
    if (!file) return;
    if (state.saveInProgress) { setStatus("鱼获正在保存，请稍候再更换照片。"); return; }
    clearSelectedPhoto({keepStatus: true});
    const selectionSequence = state.requestSequence;
    if (file.size <= 0) {
      setStatus("这张照片为空，请重新选择。", "error");
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      setStatus("照片超过 15 MB，请压缩后重新选择。", "error");
      return;
    }
    if (file.type && !ACCEPTED_TYPES.has(file.type.toLowerCase())) {
      setStatus("仅支持 JPEG、PNG、WEBP 图片。", "error");
      return;
    }
    let decoded;
    let imageWidth = 0;
    let imageHeight = 0;
    try {
      decoded = await createImageBitmap(file, {imageOrientation: "from-image"});
      if (!state.current(selectionSequence)) { decoded.close(); return; }
      imageWidth = decoded.width;
      imageHeight = decoded.height;
      if (imageWidth * imageHeight > 40_000_000) {
        decoded.close();
        setStatus("照片分辨率过高，请压缩后重新选择。", "error");
        return;
      }
      decoded.close();
    } catch (_error) {
      if (!state.current(selectionSequence)) return;
      setStatus("无法解码这张照片。请选择有效的 JPEG、PNG 或 WEBP 文件。", "error");
      return;
    }
    if (!state.current(selectionSequence)) return;
    state.select(file, imageWidth, imageHeight);
    app.dataset.sessionId = state.photo.id;
    nodes.uploadPhoto.src = state.objectUrl;
    nodes.uploadPreview.classList.add("visible");
    nodes.fileMeta.hidden = false;
    nodes.fileName.textContent = file.name || "鱼获照片";
    nodes.fileSize.textContent = formatFileSize(file.size);
    nodes.start.disabled = false;
    nodes.clear.disabled = false;
    setPhotoUrl(state.objectUrl);
    nodes.processingPhoto.decode().then(updatePhotoGeometry).catch(() => {});
    nodes.dimensions.textContent = `${imageWidth || "原图"} × ${imageHeight || "待校验"}`;
    setBadge(nodes.uploadBadge, "已选择照片", "success");
    setBadge(nodes.processingBadge, "待开始", "neutral");
    nodes.processingPanel.dataset.businessState = "idle";
    hidePreviousOutcome();
    updateStageNav("upload");
    setStatus("照片已在本页预览；点击“开始识别”后，才会发送到本次识别请求。", "success");
  }

  function renderDiagnostics(result) {
    const top1 = result.classifier?.top1;
    const bbox = result.detector?.detections?.[0]?.bbox || null;
    const details = {
      request_id: result.request_id,
      result_state: result.result_state,
      detector_version: result.detector?.model_version,
      bbox_coordinate_space: result.detector?.coordinate_space,
      top1,
      top3: result.classifier?.top3 || [],
      primary_bbox: bbox,
      quality_gate: result.detector?.quality_gate,
      crop: result.crop,
      classifier_version: result.classifier?.model_version,
      timings: result.timing,
      segmentation: result.segmentation,
    };
    nodes.diagnostic.textContent = JSON.stringify(details, null, 2);
  }

  function updateResultTasks(result) {
    clearTasks();
    markTasksFromResult(result);
    const segmentation = nodes.taskList.querySelector('[data-task="segmentation"]');
    if (segmentation) segmentation.classList.add("failed");
  }

  function resultLabel(resultState) {
    return ({
      RESULT_HIGH: ["高置信度", "success", "结果鱼种"],
      RESULT_MEDIUM: ["中置信度 · 请确认", "review", "候选鱼种"],
      RESULT_LOW: ["低置信度 · 手动确认", "failed", "暂未确认鱼种"],
      ERROR_NO_FISH: ["没有检测到鱼", "failed", "需要重新选择照片"],
      ERROR_IMAGE_QUALITY: ["照片需要调整", "review", "当前无法可靠识别"],
      TECHNICAL_FAILURE: ["识别服务暂不可用", "failed", "本次识别未完成"],
    })[resultState] || ["识别状态未知", "failed", "本次识别未完成"];
  }

  function catalogKeyForPrediction(prediction) {
    if (!prediction) return null;
    const byKey = state.species.find((item) => item.species_key === prediction.species_key);
    if (byKey) return byKey.species_key;
    const name = String(prediction.species || "").trim();
    return state.species.find((item) => item.common_name_zh === name)?.species_key || null;
  }

  function populateSpecies(result, resultState) {
    const top3 = result.classifier?.top3 || [];
    const allSpecies = state.species;
    const top1 = result.classifier?.top1;
    const preferredKey = (resultState === "RESULT_HIGH" && top1)
      ? catalogKeyForPrediction(top1)
      : "";
    const seen = new Set();
    nodes.speciesSelect.innerHTML = '<option value="">请选择鱼种</option>';
    const candidates = [];
    top3.forEach((prediction) => {
      const match = catalogKeyForPrediction(prediction);
      if (match && !seen.has(match)) {
        seen.add(match);
        candidates.push({key: match, label: state.species.find((item) => item.species_key === match)?.common_name_zh || prediction.species});
      }
    });
    const candidateKeys = new Set(candidates.map((item) => item.key));
    [...candidates, ...allSpecies.filter((item) => !candidateKeys.has(item.species_key)).map((item) => ({key: item.species_key, label: item.common_name_zh}))]
      .forEach((option) => {
        const node = document.createElement("option");
        node.value = option.key;
        node.textContent = option.label;
        nodes.speciesSelect.appendChild(node);
      });
    nodes.speciesSelect.disabled = allSpecies.length === 0;
    nodes.speciesSelect.value = preferredKey || "";
    state.selectedByUser = false;
    renderCandidates(top3);
  }

  function renderCandidates(top3) {
    nodes.candidates.replaceChildren();
    const rows = top3.filter((row) => isChineseName(row.species));
    if (!rows.length) {
      nodes.candidates.hidden = true;
      return;
    }
    rows.forEach((row) => {
      const card = document.createElement("div");
      card.className = "catch-candidate";
      const name = document.createElement("strong");
      name.textContent = row.species;
      const score = document.createElement("small");
      score.textContent = `${(Number(row.confidence) * 100).toFixed(1)}%`;
      card.append(name, score);
      nodes.candidates.appendChild(card);
    });
    nodes.candidates.hidden = false;
  }

  function currentSpeciesChoice() {
    return state.species.find((item) => item.species_key === nodes.speciesSelect.value) || null;
  }

  function updateSaveAvailability() {
    const result = state.inference;
    const goodState = result && ["RESULT_HIGH", "RESULT_MEDIUM", "RESULT_LOW"].includes(result.result_state);
    const choice = currentSpeciesChoice();
    const needsManualChoice = result && ["RESULT_MEDIUM", "RESULT_LOW"].includes(result.result_state);
    const canSave = Boolean(goodState && choice && (!needsManualChoice || state.selectedByUser) && !state.saved && !state.saveInProgress && state.file);
    nodes.saveButton.disabled = !canSave;
    if (state.saved) {
      nodes.saveHint.textContent = "鱼获记录已由保存服务确认。";
      nodes.saveHint.className = "catch-form-hint";
    } else if (!result || !goodState) {
      nodes.saveHint.textContent = "当前识别状态不支持创建鱼获记录。";
    } else if (!choice) {
      nodes.saveHint.textContent = state.species.length
        ? "请选择当前活动鱼种目录中的鱼种后保存。"
        : "活动鱼种目录暂不可用，无法安全保存分类结果。";
    } else if (needsManualChoice && !state.selectedByUser) {
      nodes.saveHint.textContent = "请先手动确认鱼种，再保存本次鱼获。";
    } else if (!state.appToken) {
      nodes.saveHint.textContent = "保存前需登录鱼获账户；模型工厂登录不代表鱼获账户登录。";
    } else {
      nodes.saveHint.textContent = "点击保存后，照片会上传到你的鱼获记录。";
    }
  }

  function showRecovery(result) {
    const [badge, tone, label] = resultLabel(result.result_state);
    nodes.resultEmpty.hidden = true;
    nodes.resultContent.hidden = false;
    nodes.resultContent.classList.add("visible");
    nodes.resultMedia.hidden = false;
    nodes.resultPhoto.hidden = false;
    nodes.speciesLabel.textContent = label;
    nodes.speciesName.textContent = result.message || "本次识别未完成";
    nodes.confidence.textContent = "";
    nodes.confidence.className = "catch-confidence recovery";
    setBadge(nodes.resultBadge, badge, tone);
    nodes.resultPanel.dataset.businessState = "recovery";
    nodes.resultPanel.dataset.resultState = result.result_state;
    state.revealState = "error";
    state.transitionState = "idle";
    nodes.saveHint.textContent = result.guidance || "请重新选择照片后重试。";
    nodes.speciesSelect.innerHTML = '<option value="">此状态不能保存鱼获</option>';
    nodes.speciesSelect.disabled = true;
    nodes.saveButton.disabled = true;
    nodes.memoryButton.disabled = true;
    nodes.timingNote.textContent = `请求 ${result.request_id} · ${result.timing?.total_ms ?? "—"} ms。未播放成功揭晓。`;
  }

  function showRecognitionOutcome(result) {
    const resultState = result.result_state;
    nodes.resultPanel.dataset.resultState = resultState.replace("RESULT_", "");
    if (["ERROR_NO_FISH", "ERROR_IMAGE_QUALITY", "TECHNICAL_FAILURE"].includes(resultState)) {
      showRecovery(result);
      return;
    }
    const top1 = result.classifier?.top1;
    if (!top1 || !isChineseName(top1.species) && resultState === "RESULT_HIGH") {
      showRecovery({
        ...result,
        result_state: "ERROR_IMAGE_QUALITY",
        message: "模型没有返回可用的中文鱼种名称",
        guidance: "本次真实推理已完成，但结果无法按产品页面要求展示；请联系平台维护人员。",
      });
      return;
    }
    const [badge, tone, label] = resultLabel(resultState);
    nodes.resultEmpty.hidden = true;
    nodes.resultContent.hidden = false;
    nodes.resultContent.classList.add("visible");
    nodes.resultPhoto.hidden = false;
    nodes.speciesLabel.textContent = label;
    nodes.resultPanel.classList.add("revealing");
    nodes.resultPanel.dataset.businessState = "ready";
    setBadge(nodes.resultBadge, badge, tone);
    if (resultState === "RESULT_HIGH") {
      nodes.speciesName.textContent = top1.species;
      nodes.confidence.textContent = `${(Number(top1.confidence) * 100).toFixed(1)}%`;
      nodes.confidence.className = "catch-confidence";
    } else if (resultState === "RESULT_MEDIUM") {
      nodes.speciesName.textContent = "请确认鱼种";
      nodes.confidence.textContent = "候选";
      nodes.confidence.className = "catch-confidence medium";
    } else {
      nodes.speciesName.textContent = "暂未确认鱼种";
      nodes.confidence.textContent = "低置信度";
      nodes.confidence.className = "catch-confidence low";
    }
    populateSpecies(result, resultState);
    updateSaveAvailability();
    nodes.timingNote.textContent = `请求 ${result.request_id} · Detector ${result.detector?.model_version || "—"} · Classifier ${result.classifier?.model_version || "—"} · 总耗时 ${result.timing?.total_ms ?? "—"} ms。SAM 轮廓未启用。`;
  }

  function updatePhotoGeometry() {
    if (!state.photo || document.querySelector('[data-component="TransitionHost"]')) return;
    state.photo.processingGeometry = state.photo.geometry(nodes.processingMedia.getBoundingClientRect());
    state.photo.resultGeometry = state.photo.geometry(nodes.resultMedia.getBoundingClientRect());
    const geometry = state.photo.processingGeometry;
    if (!geometry) return;
    const parent = nodes.processingMedia.getBoundingClientRect();
    nodes.photoStage.style.left = `${geometry.rect.left - parent.left}px`;
    nodes.photoStage.style.top = `${geometry.rect.top - parent.top}px`;
    nodes.photoStage.style.width = `${geometry.rect.width}px`;
    nodes.photoStage.style.height = `${geometry.rect.height}px`;
  }

  function mapBboxToElement(image, overlay, bbox) {
    const geometry = state.photo?.geometry(image.getBoundingClientRect());
    const rect = core.PhotoViewport.mapBox(bbox, geometry);
    if (!rect) { overlay.classList.remove("visible"); return; }
    const parent = overlay.offsetParent.getBoundingClientRect();
    for (const key of ["left", "top", "width", "height"]) {
      overlay.style[key] = `${rect[key] - (key === "left" ? parent.left : key === "top" ? parent.top : 0)}px`;
    }
    overlay.classList.add("visible");
  }

  function drawBBoxIfRequested(result) {
    const first = result.detector?.detections?.[0]?.bbox;
    if (!first) return;
    state.bbox = core.PhotoViewport.validBox(first);
    if (!state.bbox) return;
    const details = document.querySelector(".catch-detail");
    const checkbox = document.getElementById("catchShowBBox");
    if (!checkbox) {
      const label = document.createElement("label");
      label.className = "catch-media-caption";
      label.innerHTML = '<span><input id="catchShowBBox" type="checkbox"> 验收时显示真实 BBox（非鱼体轮廓）</span>';
      details.appendChild(label);
      const input = $("catchShowBBox");
      input?.addEventListener("change", () => {
        if (input.checked && state.bbox) mapBboxToElement(nodes.processingPhoto, nodes.bbox, state.bbox);
        else nodes.bbox.classList.remove("visible");
      });
    }
    const input = $("catchShowBBox");
    if (input?.checked) mapBboxToElement(nodes.processingPhoto, nodes.bbox, first);
    // Keep the bbox coordinates available in the details panel; it is never a gold outline.
    if (details) details.dataset.bboxAvailable = "true";
  }

  async function transitionSharedPhoto(eventStart, requestSequence) {
    nodes.grid.classList.add("handoff");
    state.transitionState = "active";
    updateStageNav("result");
    core.renderJourney(app, state, "handoff");
    nodes.resultPanel.dataset.rr09Frame = "R0";
    await core.transitionPhoto({
      photo: state.photo, image: nodes.photoStage, targetImage: nodes.resultPhoto,
      source: nodes.processingMedia, target: nodes.resultMedia, eventStart,
      duration: RR09_FRAMES[2].at, isCurrent: () => state.current(requestSequence),
      onCleanup: (cleanup) => { state.cancelTransition = cleanup; },
      onFrame: ({frame}) => { nodes.resultPanel.dataset.rr09Frame = frame; },
    });
    if (state.current(requestSequence)) {
      nodes.grid.classList.remove("handoff");
      state.transitionState = "done";
      updateStageNav("result");
      core.renderJourney(app, state);
    }
  }

  async function revealHighResult(result, eventStart, requestSequence) {
    nodes.resultContent.hidden = false;
    nodes.resultEmpty.hidden = true;
    nodes.resultContent.classList.add("visible");
    nodes.resultPhoto.hidden = false;
    nodes.resultPhoto.src = state.objectUrl;
    nodes.resultPanel.dataset.businessState = "revealing";
    nodes.resultPanel.classList.add("revealing");
    nodes.resultPanel.dataset.resultState = "HIGH";
    state.revealState = "active";
    core.renderJourney(app, state);
    const top1 = result.classifier?.top1;
    nodes.speciesName.textContent = isChineseName(top1?.species) ? top1.species : "";
    nodes.speciesName.style.visibility = "hidden";
    nodes.speciesLabel.textContent = "识别鱼种";
    nodes.confidence.textContent = `${(Number(top1.confidence) * 100).toFixed(1)}%`;
    nodes.confidence.className = "catch-confidence";
    const revealAt = async (target) => delay(Math.max(0, eventStart + target - performance.now()));
    await transitionSharedPhoto(eventStart, requestSequence); // R0 → R0.5 → R1, measured from live DOM rectangles.
    if (!state.current(requestSequence)) return;
    await revealAt(RR09_FRAMES[3].at); // R2 380ms; no precise transfer without qualified SAM.
    if (!state.current(requestSequence)) return;
    nodes.resultPanel.dataset.rr09Frame = "R2";
    nodes.resultPanel.dataset.revealState = state.photo.mask.status === "TRUSTED_MASK_THREE_LAYERS" ? "qualified-mask-transfer" : "gold-transfer-unavailable";
    await revealAt(RR09_FRAMES[4].at); // R3 550ms; keep the name hidden without a trusted glyph trace.
    if (!state.current(requestSequence)) return;
    nodes.resultPanel.dataset.rr09Frame = "R3";
    await revealAt(RR09_FRAMES[5].at); // R4 680ms: real Chinese name becomes readable.
    if (!state.current(requestSequence)) return;
    nodes.resultPanel.dataset.rr09Frame = "R4";
    nodes.speciesName.style.visibility = "visible";
    await revealAt(RR09_FRAMES[6].at); // R5 850ms: static page, no looping effect.
    if (!state.current(requestSequence)) return;
    nodes.resultPanel.dataset.rr09Frame = "R5";
    nodes.resultPanel.dataset.businessState = "ready";
    nodes.resultPanel.dataset.revealState = "complete";
    state.revealState = "done";
    nodes.resultPanel.classList.remove("revealing");
    nodes.grid.classList.remove("handoff");
    updateStageNav("result");
    core.renderJourney(app, state);
  }

  async function showNonHighResult(result) {
    // Medium/low are real classifier outcomes, but not a confirmed single
    // species. Do not play the success-name reveal or preselect Top-1.
    showRecognitionOutcome(result);
    nodes.resultPhoto.src = state.objectUrl;
    nodes.resultPhoto.hidden = false;
    nodes.resultPanel.dataset.businessState = "ready";
    nodes.resultPanel.dataset.resultState = result.result_state.replace("RESULT_", "");
    state.transitionState = "idle";
    state.revealState = "done";
    swosRenderer.stop();
    core.renderJourney(app, state);
  }

  async function requestRecognition() {
    if (!state.file || state.busy) return;
    const {sequence: requestSequence, controller} = state.begin();
    $("catchCancelButton").hidden = false;
    hidePreviousOutcome();
    clearTasks();
    nodes.processingPanel.dataset.businessState = "processing";
    nodes.resultPanel.dataset.businessState = "processing";
    nodes.processingMedia.classList.add("processing");
    setBadge(nodes.uploadBadge, "照片已就绪", "success");
    setBadge(nodes.processingBadge, "正在识别", "running");
    setBadge(nodes.resultBadge, "等待真实结果", "neutral");
    nodes.businessMessage.textContent = "正在识别你的鱼获，请稍候";
    nodes.start.disabled = true;
    nodes.input.disabled = false;
    updateStageNav("processing", ["upload"]);
    setInProgressTasks();
    setStatus("识别请求已发送。处理结果未就绪前不会显示鱼种名称或成功状态。", "success");
    const requestStartedAt = performance.now();
    swosRenderer.start(requestStartedAt, {
      processingFrames: PROCESSING_FRAMES,
      waterStrength: [0, 10, 45, 75, 60, 35, 10, 0],
      goldStrength: [0, 0, 15, 45, 100, 100, 100, 100],
    });
    const timeout = window.setTimeout(() => controller.abort("request-timeout"), 120_000);
    try {
      const payload = await core.infer(app.dataset.api, state.file, controller.signal, `web_${state.photo.id}_${requestSequence}`);
      if (!state.accept(payload, requestSequence)) return;
      state.inference = payload;
      state.phase = "RESULT";
      swosRenderer.setGeometry(state.bbox);
      const elapsed = performance.now() - requestStartedAt;
      nodes.dimensions.textContent = `${payload.input?.width || "—"} × ${payload.input?.height || "—"} · ${payload.input?.format || "—"}`;
      renderDiagnostics(payload);
      updateResultTasks(payload);
      drawBBoxIfRequested(payload);
      setBadge(nodes.uploadBadge, "原图已校验", "success");
      nodes.businessMessage.textContent = `识别已返回结果 · ${payload.timing?.total_ms ?? "—"} ms`;
      if (payload.result_state.startsWith("ERROR_") || payload.result_state === "TECHNICAL_FAILURE") {
        state.phase = "ERROR";
        state.revealState = "error";
        swosRenderer.stop();
        nodes.processingPanel.dataset.businessState = "recovery";
        nodes.resultPanel.dataset.businessState = "recovery";
        setBadge(nodes.processingBadge, "真实结果需恢复", "review");
        showRecognitionOutcome(payload);
        nodes.resultPhoto.src = state.objectUrl;
        nodes.resultPhoto.hidden = false;
        updateStageNav("result", ["upload", "processing"]);
        setStatus(`${payload.message || "本次识别未完成"} ${payload.guidance || ""}`.trim(), "error");
      } else {
        const chineseTop1 = isChineseName(payload.classifier?.top1?.species);
        if (payload.result_state === "RESULT_HIGH" && !chineseTop1) {
          state.phase = "ERROR";
          state.revealState = "error";
          swosRenderer.stop();
          showRecognitionOutcome({
            ...payload, result_state: "ERROR_IMAGE_QUALITY",
            message: "模型没有返回可用的中文鱼种名称",
            guidance: "真实分类已完成，但页面不会把内部类别键显示成鱼种名。",
          });
          nodes.resultPhoto.src = state.objectUrl;
          nodes.resultPhoto.hidden = false;
          setStatus("真实推理已完成，中文鱼种名映射缺失；未播放名称揭晓。", "error");
          updateStageNav("result", ["upload", "processing"]);
        } else if (payload.result_state === "RESULT_HIGH") {
          // If the model is ready before the nominal 2800ms end, keep P7 until
          // then. This timer controls only the visual handoff, never business
          // status or success.
          const waitRemaining = Math.max(0, PROCESSING_MS - elapsed);
          if (waitRemaining) await delay(waitRemaining);
          if (requestSequence !== state.requestSequence) return;
          // Lock the deterministic 2800ms contract sample even when RAF cadence
          // lands just before P7; the photo and all attached overlays then move
          // together through the shared R0/R0.5/R1 transform.
          swosRenderer.renderAt(PROCESSING_MS);
          swosRenderer.stop(false);
          const revealStart = performance.now();
          setBadge(nodes.processingBadge, "识别完成", "success");
          setBadge(nodes.resultBadge, "正在呈现结果", "running");
          updateStageNav("result", ["upload", "processing"]);
          await revealHighResult(payload, revealStart, requestSequence);
          if (state.current(requestSequence)) { populateSpecies(payload, payload.result_state); updateSaveAvailability(); }
          if (requestSequence !== state.requestSequence) return;
          setBadge(nodes.resultBadge, "高置信度", "success");
          setStatus(`识别完成：${payload.classifier.top1.species} · ${payload.timing?.total_ms ?? "—"} ms。`, "success");
        } else {
          // MEDIUM and LOW have a real prediction, but require user selection;
          // they enter the frozen result form without the HIGH success reveal.
          showNonHighResult(payload);
          setBadge(nodes.processingBadge, "真实分类已返回", "success");
          updateStageNav("result", ["upload", "processing"]);
          setStatus(payload.result_state === "RESULT_MEDIUM" ? "模型返回多个相近候选，请明确确认鱼种后再保存。" : "分类置信度较低；请手动确认鱼种后再保存。", "success");
        }
      }
    } catch (error) {
      if (requestSequence !== state.requestSequence) return;
      swosRenderer.stop();
      state.busy = false;
      if (error?.name === "AbortError" || controller.signal.aborted) {
        state.phase = "CANCELLED";
        core.renderJourney(app, state);
        const timeoutAbort = error?.message === "request-timeout" || controller.signal.reason === "request-timeout";
        nodes.processingPanel.dataset.businessState = "idle";
        nodes.resultPanel.dataset.businessState = "idle";
        setBadge(nodes.processingBadge, timeoutAbort ? "请求超时" : "已取消", "failed");
        nodes.businessMessage.textContent = timeoutAbort ? "没有收到真实结果，可重新尝试" : "已停止等待本次响应";
        setStatus(timeoutAbort ? "识别请求超过 120 秒仍未响应。请检查服务状态后重试。" : "本次请求已取消；旧响应不会覆盖当前照片。", "error");
      } else {
        state.phase = "ERROR";
        state.revealState = "error";
        state.transitionState = "idle";
        core.renderJourney(app, state);
        const requestRef = error?.requestId ? ` · 请求 ${error.requestId}` : "";
        nodes.processingPanel.dataset.businessState = "failed";
        nodes.resultPanel.dataset.businessState = "failed";
        setBadge(nodes.processingBadge, "服务未完成", "failed");
        nodes.businessMessage.textContent = "真实识别没有完成";
        showRecovery({
          request_id: error?.requestId || "",
          result_state: "TECHNICAL_FAILURE",
          message: error.message || "识别服务暂不可用",
          guidance: error?.code === "PRODUCTION_CLASSIFIER_UNAVAILABLE"
            ? "当前没有唯一的生产分类模型，请联系平台维护人员。"
            : "确认网络连接后可以重试。",
          timing: {},
        });
        setStatus(`${error.message || "识别服务暂不可用"}${requestRef}`, "error");
      }
    } finally {
      window.clearTimeout(timeout);
      if (requestSequence === state.requestSequence) {
        updatePhotoGeometry();
        state.busy = false;
        state.controller = null;
        core.renderJourney(app, state);
        $("catchCancelButton").hidden = true;
        nodes.start.disabled = !state.file;
        nodes.input.disabled = false;
        nodes.processingMedia.classList.remove("processing");
        if (nodes.processingPanel.dataset.businessState === "failed" || nodes.processingPanel.dataset.businessState === "recovery") {
          setBadge(nodes.uploadBadge, "可重新选择", "neutral");
        }
      }
    }
  }

  async function loadSpeciesCatalog() {
    try {
      const response = await fetch("/api/species?status=active", {credentials: "same-origin", cache: "no-store"});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const rows = await response.json();
      state.species = Array.isArray(rows) ? rows.filter((row) => row?.species_key && row?.common_name_zh) : [];
    } catch (_error) {
      state.species = [];
      setStatus("活动鱼种目录暂不可用；识别可继续，保存会保持禁用。", "error");
    }
    updateSaveAvailability();
  }

  function openLogin() {
    nodes.loginError.textContent = "";
    nodes.loginBackdrop.classList.add("open");
    $("catchLoginUsername").focus();
  }

  function closeLogin() {
    nodes.loginBackdrop.classList.remove("open");
    nodes.loginForm.reset();
    nodes.loginError.textContent = "";
  }

  async function loginUser(formData) {
    const loginSequence = state.requestSequence;
    const username = String(formData.get("username") || "").trim();
    const password = String(formData.get("password") || "");
    nodes.loginSubmit.disabled = true;
    nodes.loginError.textContent = "正在验证鱼获账户…";
    try {
      const response = await fetch("/api/v1/auth/login", {
        method: "POST", credentials: "same-origin",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({username, password}),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "登录失败，请核对账号和密码。");
      state.appToken = payload.access_token;
      closeLogin();
      updateSaveAvailability();
      if (state.current(loginSequence)) await saveCatch();
    } catch (error) {
      nodes.loginError.textContent = error.message || "账户验证未完成。";
    } finally {
      nodes.loginSubmit.disabled = false;
      $("catchLoginPassword").value = "";
    }
  }

  function parseOptionalNumber(id) {
    const raw = $(id).value.trim();
    return raw ? Number(raw) : null;
  }

  async function saveCatch() {
    if (!state.inference || state.saveInProgress || state.saved) return;
    if (!state.appToken) {
      openLogin();
      return;
    }
    const selected = currentSpeciesChoice();
    const result = state.inference;
    if (!selected || !result.classifier?.model_version) {
      setStatus("请先选择有效的鱼种和模型结果。", "error");
      return;
    }
    if (["RESULT_MEDIUM", "RESULT_LOW"].includes(result.result_state) && !state.selectedByUser) {
      setStatus("请先手动确认鱼种。", "error");
      return;
    }
    state.saveInProgress = true;
    nodes.saveButton.disabled = true;
    nodes.saveButton.textContent = "正在保存…";
    updateSaveAvailability();
    try {
      const imageForm = new FormData();
      imageForm.append("image", state.file, state.file.name || "catch-photo");
      const uploadResponse = await fetch("/api/v1/catches/upload-image", {
        method: "POST", body: imageForm, credentials: "same-origin",
        headers: {Authorization: `Bearer ${state.appToken}`},
      });
      const uploadPayload = await uploadResponse.json().catch(() => ({}));
      if (!uploadResponse.ok) throw new Error(uploadPayload.detail || "照片上传失败；鱼获记录未保存。");
      const top3 = result.classifier.top3 || [];
      const matchingCandidate = top3.find((item) => item.species_key === selected.species_key);
      const top1 = result.classifier.top1;
      const confidence = matchingCandidate
        ? Number(matchingCandidate.confidence)
        : (top1?.species_key === selected.species_key ? Number(top1.confidence) : 0);
      const capturedAt = $("catchCapturedAt").value
        ? new Date($("catchCapturedAt").value).toISOString()
        : new Date().toISOString();
      const catchPayload = {
        image_upload_id: uploadPayload.image_upload_id,
        species_id: selected.species_key,
        species_name: selected.common_name_zh,
        confidence,
        model_version: result.classifier.model_version,
        detector_result: {
          model_version: result.detector.model_version,
          detections: result.detector.detections,
          quality_gate: result.detector.quality_gate,
          coordinate_space: result.detector.coordinate_space,
        },
        classifier_result: {
          top1,
          top3,
          result_state: result.result_state,
          selected_species_key: selected.species_key,
          selected_by_user: state.selectedByUser,
        },
        captured_at: capturedAt,
        length_cm: parseOptionalNumber("catchLength"),
        weight_kg: parseOptionalNumber("catchWeight"),
        location: $("catchLocation").value.trim() || null,
        story: $("catchStory").value.trim() || null,
        client_record_id: result.request_id,
      };
      const saveResponse = await fetch("/api/v1/catches", {
        method: "POST", credentials: "same-origin",
        headers: {"Content-Type": "application/json", Authorization: `Bearer ${state.appToken}`},
        body: JSON.stringify(catchPayload),
      });
      const savePayload = await saveResponse.json().catch(() => ({}));
      if (!saveResponse.ok || !savePayload.saved) throw new Error(savePayload.detail || "鱼获记录未保存，请重试。");
      state.saved = true;
      nodes.saveButton.textContent = "已保存";
      setBadge(nodes.resultBadge, "鱼获已保存", "success");
      nodes.memoryButton.disabled = true;
      nodes.saveHint.textContent = `保存服务已确认记录 ${savePayload.catch_id}。Web 记忆详情页尚未登记。`;
      setStatus("鱼获记录保存成功；照片现已关联到你的账户。", "success");
    } catch (error) {
      setStatus(error.message || "鱼获记录保存失败。", "error");
      nodes.saveHint.textContent = error.message || "鱼获记录未保存。请重试。";
      nodes.saveButton.disabled = false;
    } finally {
      state.saveInProgress = false;
      nodes.saveButton.textContent = state.saved ? "已保存" : "保存本次鱼获";
      updateSaveAvailability();
    }
  }

  $("catchCancelButton").addEventListener("click", () => {
    state.requestSequence += 1;
    state.controller?.abort();
    state.cancelTransition?.();
    state.controller = null;
    state.busy = false;
    hidePreviousOutcome();
    nodes.processingMedia.classList.remove("processing");
    nodes.grid.classList.remove("handoff");
    swosRenderer.stop();
    nodes.start.disabled = !state.file;
    nodes.input.disabled = false;
    $("catchCancelButton").hidden = true;
    state.phase = "CANCELLED";
    updateStageNav("upload");
    core.renderJourney(app, state);
    setBadge(nodes.processingBadge, "已取消", "neutral");
    nodes.processingPanel.dataset.businessState = "cancelled";
    setStatus("本次识别已取消，可重新开始。旧响应不会覆盖当前照片。");
  });
  nodes.input.addEventListener("change", () => chooseFile(nodes.input.files?.[0]));
  nodes.start.addEventListener("click", requestRecognition);
  nodes.clear.addEventListener("click", () => clearSelectedPhoto());
  nodes.drop.addEventListener("dragover", (event) => { event.preventDefault(); nodes.drop.classList.add("drag-over"); });
  nodes.drop.addEventListener("dragleave", () => nodes.drop.classList.remove("drag-over"));
  nodes.drop.addEventListener("drop", (event) => {
    event.preventDefault();
    nodes.drop.classList.remove("drag-over");
    chooseFile(event.dataTransfer?.files?.[0]);
  });
  nodes.speciesSelect.addEventListener("change", () => {
    state.selectedByUser = Boolean(nodes.speciesSelect.value);
    updateSaveAvailability();
  });
  nodes.saveButton.addEventListener("click", saveCatch);
  nodes.loginCancel.addEventListener("click", closeLogin);
  nodes.loginForm.addEventListener("submit", (event) => {
    event.preventDefault();
    loginUser(new FormData(nodes.loginForm));
  });
  nodes.loginBackdrop.addEventListener("click", (event) => {
    if (event.target === nodes.loginBackdrop) closeLogin();
  });
  const photoResize = new ResizeObserver(updatePhotoGeometry);
  photoResize.observe(nodes.processingMedia);
  photoResize.observe(nodes.resultMedia);
  window.addEventListener("resize", () => {
    updatePhotoGeometry();
    const check = $("catchShowBBox");
    const bbox = state.inference?.detector?.detections?.[0]?.bbox;
    if (check?.checked && bbox) mapBboxToElement(nodes.processingPhoto, nodes.bbox, bbox);
    updateStageNav(app.dataset.activeStage || "upload");
  });

  document.querySelectorAll("[data-stage-step]").forEach((node) => node.addEventListener("click", () => {
    if (!node.disabled && !state.busy && !nodes.grid.classList.contains("handoff")) updateStageNav(node.dataset.stageStep);
  }));

  core.renderJourney(app, state);
  updateStageNav(app.dataset.activeStage || "upload");
  setCapturedAtNow();
  clearTasks();
  loadSpeciesCatalog();

  // Expose timeline metadata for deterministic automated contract checks.
  window.YUJIAN_CATCH_RECOGNITION_TIMELINE = Object.freeze({
    nominalProcessingMs: PROCESSING_MS,
    rr09Ms: RR09_MS,
    totalMs: PROCESSING_MS + RR09_MS,
    processingFrames: PROCESSING_FRAMES,
    rr09Frames: RR09_FRAMES,
  });
})();
