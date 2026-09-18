(() => {
  const REFINE_MODE = 'fish_preserve_refine_v2';
  const INPAINT_MODE = 'fish_preserve_inpaint_v2';
  const V1_MODE = 'dual_ip_adapter_v1';
  const state = {
    mode: REFINE_MODE,
    datasetId: '',
    sourceItems: [],
    source: null,
    references: [],
    referenceIndex: 0,
    inpaint: null,
    refine: null,
    runId: null,
    busy: false,
  };

  // Detector + SAM + Visible extraction remain in the existing Direct Lab.
  const POWERPAINT_DIRECT_PREPARE_ENDPOINT = '/api/debug/powerpaint-direct-lab/prepare';
  const COMPLETION_LAB_PREPARE_ENDPOINT = '/api/debug/fish-completion-lab/prepare';
  const REFINE_PROMPT = 'professional wildlife fish portrait, realistic photography, preserve original fish identity, preserve original fish species characteristics, preserve original body proportions, preserve original head shape, preserve original fin structure, preserve original color and texture, complete fish body, natural fish anatomy, clean natural background, soft lighting, premium realistic fishing asset';
  const REFINE_NEGATIVE = 'different fish species, different fish identity, changed body shape, changed head shape, changed body proportions, wrong fish anatomy, extra fins, missing fins, deformed fins, mutated fish, duplicate fish, cartoon, illustration, fake texture, obvious AI artifacts';
  const INPAINT_PROMPT = 'professional wildlife fish portrait, realistic photography, natural fish texture, detailed scales, clean natural background, soft lighting, high resolution';
  const INPAINT_NEGATIVE = 'different fish species, changed body shape, wrong fish anatomy, extra fins, missing fins, deformed fish, cartoon, illustration, fake texture, duplicate fish';
  const ids = [
    'portraitMode', 'portraitModeNote', 'portraitDataset', 'portraitDatasetHint', 'portraitSpecies',
    'portraitSourceState', 'portraitSourceGrid', 'portraitOriginalUpload', 'portraitPrepare',
    'portraitPrepareHint', 'portraitDirectLabLink', 'portraitCompletionLabLink', 'portraitRefineField',
    'portraitPreserveStrength', 'portraitRefineStrength', 'portraitAutoStraighten', 'portraitInpaintMasks',
    'portraitFishMaskPreview', 'portraitCompletionMaskPreview', 'portraitReferenceField',
    'portraitReferenceState', 'portraitReference', 'portraitReferenceImage', 'portraitReferenceTitle',
    'portraitReferenceMeta', 'portraitReferenceSwitch', 'portraitV1Params', 'portraitSourceScale',
    'portraitReferenceScale', 'portraitInpaintStrengthField', 'portraitStrength', 'portraitSteps',
    'portraitSeed', 'portraitWidthField', 'portraitWidth', 'portraitHeightField', 'portraitHeight',
    'portraitParamHint', 'portraitPrompt', 'portraitNegativePrompt', 'portraitSubmit', 'portraitSubmitHint',
    'portraitABTest', 'portraitABTestHint', 'portraitRefresh', 'portraitProgress', 'portraitRunTitle',
    'portraitRunStatus', 'portraitSourceImage', 'portraitSamVisibleCard', 'portraitSamVisibleResultImage',
    'portraitRefinedCard', 'portraitRefinedResultImage', 'portraitFinalCard', 'portraitFinalResultImage',
    'portraitFishMaskCard', 'portraitFishMaskResultImage', 'portraitCompletionMaskCard',
    'portraitCompletionMaskResultImage', 'portraitGeneratedCard', 'portraitGeneratedImage',
    'portraitOutputEmpty', 'portraitMeta', 'portraitSweepResults', 'portraitV1ReferenceCard',
    'portraitReferenceResultImage', 'portraitOriginalLabel', 'portraitFishMaskLabel',
    'portraitCompletionMaskLabel', 'portraitGeneratedLabel', 'portraitWorkerStatus', 'portraitWorkerDetail',
  ];
  const el = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
  const stageLabels = {
    load_source: '读取真实照片',
    load_reference: '读取标准鱼体 B',
    load_masks: '读取 Fish / Completion Mask',
    load_sam_visible: '读取 SAM Visible',
    sdxl_generate: 'SDXL + Dual IP-Adapter',
    sdxl_inpaint: 'SDXL Inpaint V2',
    refine_generate: 'Fish Preserve Refine',
    straighten: '确定性水平归一化',
    persist_result: '保存实验结果',
    complete: '完成',
  };
  const statusLabels = { PENDING: '排队中', QUEUED: '排队中', RUNNING: '运行中', DONE: '已完成', SUCCESS: '成功', FAILED: '失败', SKIPPED: '未执行' };
  const referenceTypeLabels = { transparent_main: 'transparent 主图', transparent_alt: 'transparent 备选', COVER_CARD: 'Cover 列表图', COVER_CARD_TRANSPARENT_LEFT: 'Cover 透明左图', COVER_CARD_TRANSPARENT_RIGHT: 'Cover 透明右图' };

  const modeIsRefine = () => state.mode === REFINE_MODE;
  const modeIsInpaint = () => state.mode === INPAINT_MODE;
  const modeIsV1 = () => state.mode === V1_MODE;
  const numberValue = (id, fallback) => {
    const value = Number(el[id]?.value);
    return Number.isFinite(value) ? value : fallback;
  };
  const optionalInt = (id) => {
    const raw = String(el[id]?.value || '').trim();
    if (!raw) return null;
    const value = Number(raw);
    return Number.isInteger(value) && value >= 0 ? value : null;
  };
  const setStatus = (status, text) => {
    const normalized = String(status || 'UNKNOWN').toUpperCase();
    el.portraitRunStatus.className = `status-tag status-${normalized.toLowerCase()}`;
    el.portraitRunStatus.innerHTML = `<span class="status-dot"></span>${esc(text || statusLabels[normalized] || normalized)}`;
  };
  const normalizeDatasetPayload = (payload) => (Array.isArray(payload) ? payload : (Array.isArray(payload?.datasets) ? payload.datasets : []));

  function renderDatasetOptions(rows) {
    const frozen = rows.filter((row) => String(row.status || '').toUpperCase() === 'FROZEN');
    el.portraitDataset.innerHTML = '<option value="">请选择已冻结 Dataset</option>';
    rows.forEach((row) => {
      const id = String(row.id || row.dataset_id || row.dataset_version || '');
      if (!id) return;
      const status = String(row.status || 'UNKNOWN').toUpperCase();
      const option = document.createElement('option');
      option.value = id;
      option.textContent = `${id} · ${Number(row.total ?? row.image_count ?? 0)} 张 · ${status === 'FROZEN' ? '已冻结' : status}`;
      option.disabled = status !== 'FROZEN';
      el.portraitDataset.appendChild(option);
    });
    if (frozen.length) {
      state.datasetId = String(frozen[0].id || frozen[0].dataset_id || frozen[0].dataset_version);
      el.portraitDataset.value = state.datasetId;
      el.portraitDatasetHint.textContent = `已选择 ${state.datasetId}；数据集为只读实验输入。`;
    } else {
      state.datasetId = '';
      el.portraitDatasetHint.textContent = '暂无已冻结 Dataset。Refine V2 必须先冻结 Dataset。';
    }
  }

  function clearPrepared() {
    state.inpaint = null;
    state.refine = null;
    el.portraitPrepareHint.className = 'portrait-hint';
    [el.portraitFishMaskPreview, el.portraitCompletionMaskPreview].forEach((node) => { if (node) node.hidden = true; });
    el.portraitPrepareHint.textContent = modeIsRefine()
      ? 'Detector、SAM 和 Visible 提取沿用现有 Direct Lab，本页不重写。'
      : '这里复用现有 Fish Completion Lab 的 Detector + SAM，不在本页重写。';
    updateSubmit();
  }

  function renderSourceItems(items) {
    state.sourceItems = Array.isArray(items) ? items : [];
    state.source = null;
    state.references = [];
    state.referenceIndex = 0;
    el.portraitSpecies.value = '';
    clearPrepared();
    el.portraitReference.hidden = true;
    el.portraitReferenceState.textContent = '请选择真实照片 A。';
    if (!state.sourceItems.length) {
      el.portraitSourceGrid.hidden = true;
      el.portraitSourceState.hidden = false;
      el.portraitSourceState.textContent = modeIsRefine() ? '该 Dataset 暂无可选图片。' : '请选择 Dataset 图片或上传文件。';
      updateSubmit();
      return;
    }
    el.portraitSourceState.hidden = true;
    el.portraitSourceGrid.hidden = false;
    el.portraitSourceGrid.innerHTML = state.sourceItems.map((item, index) => (
      `<button type="button" class="portrait-source-card" data-source-index="${index}"><img src="${esc(item.preview_url || item.image_url || '')}" alt="${esc(item.image_id || '真实照片')}"><strong>${esc(item.image_id || `图片 ${index + 1}`)}</strong><small>${esc(item.species_name || item.species_id || '')} · ${esc(item.split || '—')}</small></button>`
    )).join('');
    el.portraitSourceGrid.querySelectorAll('[data-source-index]').forEach((button) => button.addEventListener('click', () => selectSource(Number(button.dataset.sourceIndex))));
    updateSubmit();
  }

  function renderReference() {
    const reference = state.references[state.referenceIndex];
    if (!reference) {
      el.portraitReference.hidden = true;
      el.portraitReferenceState.hidden = false;
      el.portraitReferenceState.textContent = '没有找到该鱼种的标准参考图（transparent / Cover 资产包）。';
      updateSubmit();
      return;
    }
    el.portraitReferenceState.hidden = true;
    el.portraitReference.hidden = false;
    el.portraitReferenceImage.src = reference.url || '';
    const type = reference.type || 'transparent_main';
    el.portraitReferenceTitle.textContent = `${reference.species_name || reference.species_id || '标准鱼体'} · ${referenceTypeLabels[type] || type}`;
    el.portraitReferenceMeta.textContent = `Asset ${reference.asset_id || '—'} · ${reference.version || '—'}${reference.source_kind === 'knowledge_cover' ? ' · 鱼种 Cover 资产包' : ''}`;
    el.portraitReferenceSwitch.hidden = state.references.length < 2;
    updateSubmit();
  }

  async function loadReferences(speciesId) {
    state.references = [];
    state.referenceIndex = 0;
    el.portraitReference.hidden = true;
    el.portraitReferenceState.hidden = false;
    el.portraitReferenceState.textContent = '正在按鱼种匹配标准参考图…';
    if (!speciesId) return;
    try {
      const primary = await platformFetch(`/api/platform/portrait/reference/${encodeURIComponent(speciesId)}`);
      const list = await platformFetch(`/api/platform/assets/fish-reference?species_id=${encodeURIComponent(speciesId)}&asset_type=transparent`);
      const assets = Array.isArray(list.assets) ? list.assets : [];
      if (primary.reference_asset && !assets.some((item) => item.asset_id === primary.reference_asset.asset_id)) assets.unshift(primary.reference_asset);
      state.references = assets;
      renderReference();
    } catch (error) {
      el.portraitReferenceState.textContent = platformError(error);
      el.portraitReferenceState.className = 'error-state';
      updateSubmit();
    }
  }

  async function selectSource(index) {
    const source = state.sourceItems[index];
    if (!source) return;
    state.source = source;
    el.portraitSpecies.value = source.species_name || source.species_id || '';
    el.portraitSourceGrid.querySelectorAll('.portrait-source-card').forEach((button, i) => button.classList.toggle('selected', i === index));
    el.portraitSourceImage.src = source.image_url || source.preview_url || '';
    el.portraitSourceImage.hidden = !source.image_url && !source.preview_url;
    clearPrepared();
    if (modeIsV1()) await loadReferences(source.species_id || source.species_name);
    updateSubmit();
  }

  function updateModeUI() {
    state.mode = el.portraitMode.value;
    const refine = modeIsRefine();
    const inpaint = modeIsInpaint();
    const v1 = modeIsV1();
    if (refine && state.source && !state.source.item_id) {
      state.source = null;
      el.portraitOriginalUpload.value = '';
    }
    el.portraitModeNote.textContent = refine
      ? '复用 PowerPaint Direct Lab 的 SAM Visible；只修复鱼体并做确定性水平归一化，不引入标准鱼模板。'
      : inpaint
        ? '真实鱼照片是主体来源；AI 只处理 Completion Mask 区域。'
        : '历史对照模式：A/B 两张图共同进入 Dual IP-Adapter，不用于替代真实鱼体。';
    el.portraitReferenceField.hidden = !v1;
    el.portraitV1Params.hidden = !v1;
    el.portraitRefineField.hidden = !refine;
    el.portraitInpaintMasks.hidden = !inpaint;
    el.portraitOriginalUpload.hidden = !inpaint;
    el.portraitPrepare.hidden = !(refine || inpaint);
    el.portraitDirectLabLink.hidden = !refine;
    el.portraitCompletionLabLink.hidden = !inpaint;
    el.portraitInpaintStrengthField.hidden = !inpaint;
    el.portraitWidthField.hidden = refine;
    el.portraitHeightField.hidden = refine;
    el.portraitParamHint.textContent = refine
      ? 'Refine V2 固定同一张 SAM Visible、Prompt、Seed、Steps，优先比较保真/修复强度。'
      : inpaint
        ? 'Inpaint V2 第一轮建议比较 Strength：0.15、0.25、0.35。'
        : 'V1 使用 A/B 两个 IP-Adapter conditioning scale。';
    if (refine) {
      el.portraitPrompt.value = REFINE_PROMPT;
      el.portraitNegativePrompt.value = REFINE_NEGATIVE;
      el.portraitSourceState.textContent = state.source ? '已选择真实照片，请复用 Direct Lab 生成 SAM Visible。' : '请先选择已冻结 Dataset 图片。';
    } else if (inpaint) {
      el.portraitPrompt.value = INPAINT_PROMPT;
      el.portraitNegativePrompt.value = INPAINT_NEGATIVE;
      el.portraitSourceState.textContent = state.source ? '已选择真实照片，请生成 Fish Mask + Completion Mask。' : '请先选择 Dataset 或上传图片。';
    } else {
      el.portraitPrompt.value = '';
      el.portraitNegativePrompt.value = '';
      el.portraitSourceState.textContent = '请选择一张真实照片 A。';
      if (state.source) loadReferences(state.source.species_id || state.source.species_name);
    }
    clearPrepared();
    updateSubmit();
  }

  function updateSubmit() {
    const fileSelected = Boolean(el.portraitOriginalUpload?.files?.[0]);
    const sourceReady = Boolean(state.source) || (modeIsInpaint() && fileSelected);
    const ready = modeIsRefine()
      ? Boolean(state.refine?.original_image_uri && state.refine?.sam_visible_uri)
      : modeIsInpaint()
        ? Boolean(state.inpaint?.original_image_uri && state.inpaint?.fish_mask_uri && state.inpaint?.completion_mask_uri)
        : Boolean(state.source && state.references[state.referenceIndex]);
    el.portraitPrepare.disabled = !(modeIsRefine() || modeIsInpaint()) || !sourceReady || state.busy;
    el.portraitSubmit.disabled = !ready || state.busy;
    el.portraitSubmitHint.textContent = state.busy ? '任务运行中，请等待结果。' : ready ? '输入已就绪，可以创建 PipelineRun。' : modeIsRefine() ? '请选择 Dataset 图片并先生成 SAM Visible。' : modeIsInpaint() ? '请选择图片并先生成遮罩。' : '请选择 A 图和标准参考图。';
  }

  async function parsePrepareResponse(response) {
    const text = await response.text();
    let data = {};
    try { data = text ? JSON.parse(text) : {}; } catch { data = { message: text }; }
    if (!response.ok || data.error_code || data.error) throw new Error(data.message || data.detail || data.error?.message || '准备失败');
    return data;
  }

  async function prepareRefine() {
    if (!modeIsRefine() || state.busy) return;
    if (!state.source || !state.datasetId) {
      el.portraitPrepareHint.textContent = 'Refine V2 必须先选择已冻结 Dataset 图片。';
      updateSubmit();
      return;
    }
    el.portraitPrepare.disabled = true;
    el.portraitPrepareHint.textContent = '正在复用 PowerPaint Direct Lab：Detector + SAM + SAM Visible…';
    try {
      const form = new URLSearchParams({ dataset_version: state.datasetId, dataset_item_id: String(state.source.item_id || ''), mask_strategy: 'SAM_PROTECT_VISIBLE' });
      const data = await parsePrepareResponse(await fetch(POWERPAINT_DIRECT_PREPARE_ENDPOINT, { method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: form }));
      const report = data.report || {};
      const assets = report.assets || {};
      const originalUri = assets.original;
      const samVisibleUri = assets.sam_visible || assets.sam_transparent;
      if (!originalUri || !samVisibleUri) throw new Error('Direct Lab 未返回 original / SAM Visible 资源');
      const originalPreview = report.preview_original || state.source.image_url || state.source.preview_url || null;
      const samPreview = report.preview_sam || null;
      const species = el.portraitSpecies.value.trim() || state.source.species_name || state.source.species_id || '';
      state.refine = { source_run_id: data.test_id || null, original_image_uri: originalUri, sam_visible_uri: samVisibleUri, originalPreview, samPreview, species };
      if (originalPreview) { el.portraitSourceImage.src = originalPreview; el.portraitSourceImage.hidden = false; }
      el.portraitPrepareHint.textContent = '已复用 Direct Lab，SAM Visible 准备完成，可以开始 Refine V2。';
      updateSubmit();
    } catch (error) {
      state.refine = null;
      el.portraitPrepareHint.textContent = platformError(error);
      updateSubmit();
    } finally { updateSubmit(); }
  }

  async function prepareInpaint() {
    if (!modeIsInpaint() || state.busy) return;
    const file = el.portraitOriginalUpload.files?.[0];
    if (!file && !state.source) {
      el.portraitPrepareHint.textContent = '请先选择 Dataset 图片或上传文件。';
      updateSubmit();
      return;
    }
    el.portraitPrepare.disabled = true;
    el.portraitPrepareHint.textContent = '正在复用 Fish Completion Lab：Detector + SAM + Completion Mask…';
    try {
      let response;
      if (file) {
        const form = new FormData();
        form.append('file', file, file.name);
        form.append('source_type', 'local_upload');
        response = await fetch(COMPLETION_LAB_PREPARE_ENDPOINT, { method: 'POST', credentials: 'include', body: form });
      } else {
        const form = new URLSearchParams({ source_type: 'dataset_freeze', dataset_version: state.datasetId, dataset_item_id: String(state.source.item_id || '') });
        response = await fetch(COMPLETION_LAB_PREPARE_ENDPOINT, { method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: form });
      }
      const data = await parsePrepareResponse(response);
      const assets = data.assets || {};
      const previewUrls = data.preview_urls || {};
      const originalPreview = previewUrls.original || data.original || state.source?.image_url || null;
      const fishPreview = previewUrls.fish_mask || data.sam_raw_mask || data.refined_visible_mask || data.refined_visible || data.sam_transparent || null;
      const completionPreview = previewUrls.completion_mask || data.auto_completion_mask || null;
      const originalUri = assets.original;
      const fishMaskUri = assets.refined_visible_mask || assets.sam_raw_mask;
      const completionMaskUri = assets.completion_mask || assets.completion_mask_canonical;
      if (!originalUri || !fishMaskUri || !completionMaskUri) throw new Error('Fish Completion Lab 未返回完整原图 / Fish Mask / Completion Mask 资源');
      const species = el.portraitSpecies.value.trim() || state.source?.species_name || state.source?.species_id || '';
      state.inpaint = { completion_lab_test_id: data.test_id || null, original_image_uri: originalUri, fish_mask_uri: fishMaskUri, completion_mask_uri: completionMaskUri, species, originalPreview, fishPreview, completionPreview };
      if (originalPreview) { el.portraitSourceImage.src = originalPreview; el.portraitSourceImage.hidden = false; }
      if (fishPreview) { el.portraitFishMaskPreview.src = fishPreview; el.portraitFishMaskPreview.hidden = false; el.portraitFishMaskResultImage.src = fishPreview; el.portraitFishMaskResultImage.hidden = false; }
      if (completionPreview) { el.portraitCompletionMaskPreview.src = completionPreview; el.portraitCompletionMaskPreview.hidden = false; el.portraitCompletionMaskResultImage.src = completionPreview; el.portraitCompletionMaskResultImage.hidden = false; }
      el.portraitPrepareHint.textContent = '已复用 Fish Completion Lab，遮罩准备完成，可以开始 Fish Preserve Inpaint V2。';
      updateSubmit();
    } catch (error) {
      state.inpaint = null;
      el.portraitPrepareHint.textContent = platformError(error);
      updateSubmit();
    } finally { updateSubmit(); }
  }

  async function prepareInput() { return modeIsRefine() ? prepareRefine() : modeIsInpaint() ? prepareInpaint() : null; }

  function renderSteps(data) {
    const fallback = modeIsRefine() ? ['load_source', 'load_sam_visible', 'refine_generate', 'straighten', 'persist_result'] : modeIsInpaint() ? ['load_source', 'load_masks', 'sdxl_inpaint', 'persist_result'] : ['load_source', 'load_reference', 'sdxl_generate', 'persist_result'];
    const steps = Array.isArray(data.steps) && data.steps.length ? data.steps : fallback.map((name) => ({ name, status: 'PENDING' }));
    el.portraitProgress.innerHTML = steps.map((step) => {
      const status = String(step.status || 'PENDING').toUpperCase();
      const cls = status === 'RUNNING' ? 'running' : (status === 'DONE' || status === 'SUCCESS' ? 'done' : (status === 'FAILED' ? 'failed' : (status === 'SKIPPED' ? 'skipped' : '')));
      const detail = step.error || step.error_message || (step.duration_ms != null ? `${step.duration_ms} ms` : statusLabels[status] || status);
      return `<div class="portrait-step ${cls}"><span class="portrait-step-dot"></span><div><strong>${esc(stageLabels[step.name] || step.name || '实验步骤')}</strong><small>${esc(detail)}</small></div></div>`;
    }).join('');
    if (data.error_message) el.portraitProgress.insertAdjacentHTML('beforeend', `<div class="error-state">${esc(data.error_stage || '失败阶段')}：${esc(data.error_message)}</div>`);
  }

  function renderWorkerHealth(data) {
    const status = String(data?.status || 'UNAVAILABLE').toUpperCase();
    const labels = { CONNECTED: '已连接', READY: '已连接', NOT_CONFIGURED: '未配置', UNAVAILABLE: '不可用' };
    el.portraitWorkerStatus.className = `status-tag status-${status.toLowerCase()}`;
    el.portraitWorkerStatus.innerHTML = `<span class="status-dot"></span>${esc(labels[status] || status)}`;
    const endpoint = data?.worker_url ? ` · ${data.worker_url}` : '';
    el.portraitWorkerDetail.textContent = `${labels[status] || status}${endpoint}${data?.detail ? ` · ${data.detail}` : data?.message ? ` · ${data.message}` : ''}`;
    el.portraitWorkerDetail.className = `portrait-hint ${status === 'CONNECTED' || status === 'READY' ? '' : 'error-state'}`;
  }

  async function loadWorkerHealth() { try { renderWorkerHealth(await platformFetch('/api/platform/portrait/worker-health')); } catch (error) { renderWorkerHealth({ status: 'UNAVAILABLE', message: platformError(error) }); } }

  function renderMeta(metadata) {
    const mode = metadata?.mode || state.mode;
    const params = metadata?.params || {};
    const adapter = metadata?.adapter_config || {};
    const generation = metadata?.generation || {};
    const stat = (value, dash = '—') => value === null || value === undefined || value === '' ? dash : value;
    let items;
    if (mode === REFINE_MODE) items = [['Run ID', metadata.run_id || state.runId], ['模式', 'Fish Preserve Refine V2'], ['保真强度', Number(metadata.preserve_strength ?? params.preserve_strength ?? 0.75).toFixed(2)], ['修复强度', Number(metadata.refine_strength ?? params.refine_strength ?? 0.28).toFixed(2)], ['水平归一化', metadata.auto_straighten === false ? '关闭' : '开启'], ['Steps', stat(metadata.steps ?? params.steps)], ['Seed', stat(metadata.seed ?? params.seed, '随机')]];
    else if (mode === INPAINT_MODE) items = [['Run ID', metadata.run_id || state.runId], ['模式', 'Fish Preserve Inpaint V2'], ['Strength', Number(metadata.strength ?? params.strength ?? 0.25).toFixed(2)], ['Steps', stat(metadata.steps ?? params.steps)], ['Seed', stat(metadata.seed ?? params.seed, '随机')], ['Mask', metadata.mask_type || 'completion_mask']];
    else items = [['Run ID', metadata.run_id || state.runId], ['模式', 'Dual IP-Adapter V1'], ['A 权重', Number(adapter.source_scale ?? params.source_scale ?? 0.8).toFixed(2)], ['B 权重', Number(adapter.reference_scale ?? params.reference_scale ?? 0.35).toFixed(2)], ['Steps', stat(generation.steps ?? params.steps)], ['尺寸', `${generation.width ?? params.width ?? 768} × ${generation.height ?? params.height ?? 768}`]];
    el.portraitMeta.innerHTML = items.map((item) => `<div class="stat"><strong>${esc(item[1])}</strong><span>${esc(item[0])}</span></div>`).join('');
    el.portraitMeta.hidden = false;
  }

  function setCardVisible(card, visible) { if (card) card.hidden = !visible; }

  function renderRun(data) {
    const status = String(data.status || 'UNKNOWN').toUpperCase();
    state.mode = data.mode || state.mode;
    setStatus(status, statusLabels[status] || status);
    el.portraitRunTitle.textContent = `${data.run_id || state.runId || '实验任务'}${data.current_stage ? ` · ${stageLabels[data.current_stage] || data.current_stage}` : ''}`;
    renderSteps(data);
    if (data.source?.image_url && !modeIsV1()) { el.portraitSourceImage.src = data.source.image_url; el.portraitSourceImage.hidden = false; }
    if (data.reference?.url && modeIsV1()) { el.portraitReferenceResultImage.src = data.reference.url; el.portraitReferenceResultImage.hidden = false; }
    if (status === 'FAILED') { state.busy = false; updateSubmit(); }
  }

  async function loadResult(runId) {
    const result = await platformFetch(`/api/platform/portrait/results/${encodeURIComponent(runId)}`);
    const refine = result.metadata?.mode === REFINE_MODE;
    const inpaint = result.metadata?.mode === INPAINT_MODE;
    setCardVisible(el.portraitSamVisibleCard, refine);
    setCardVisible(el.portraitRefinedCard, refine);
    setCardVisible(el.portraitFinalCard, refine);
    setCardVisible(el.portraitFishMaskCard, inpaint);
    setCardVisible(el.portraitCompletionMaskCard, inpaint);
    setCardVisible(el.portraitGeneratedCard, !refine);
    setCardVisible(el.portraitV1ReferenceCard, result.metadata?.mode === V1_MODE);
    if (result.source_image) { el.portraitSourceImage.src = result.source_image; el.portraitSourceImage.hidden = false; }
    if (result.sam_visible_image) { el.portraitSamVisibleResultImage.src = result.sam_visible_image; el.portraitSamVisibleResultImage.hidden = false; }
    if (result.refined_image) { el.portraitRefinedResultImage.src = result.refined_image; el.portraitRefinedResultImage.hidden = false; }
    if (result.final_image) { el.portraitFinalResultImage.src = result.final_image; el.portraitFinalResultImage.hidden = false; }
    if (result.reference_image && result.metadata?.mode === V1_MODE) { el.portraitReferenceResultImage.src = result.reference_image; el.portraitReferenceResultImage.hidden = false; }
    if (result.fish_mask_image) { el.portraitFishMaskResultImage.src = result.fish_mask_image; el.portraitFishMaskResultImage.hidden = false; }
    if (result.completion_mask_image) { el.portraitCompletionMaskResultImage.src = result.completion_mask_image; el.portraitCompletionMaskResultImage.hidden = false; }
    if (result.generated_image && !refine) { el.portraitGeneratedImage.src = result.generated_image; el.portraitGeneratedImage.hidden = false; }
    el.portraitOutputEmpty.hidden = true;
    renderMeta(result.metadata || {});
    return result;
  }

  async function waitForRun(runId) {
    for (;;) {
      const data = await platformFetch(`/api/platform/pipeline/${encodeURIComponent(runId)}`);
      if (state.runId === runId) renderRun(data);
      const status = String(data.status || '').toUpperCase();
      if (status === 'SUCCESS') return loadResult(runId);
      if (status === 'FAILED') throw new Error(data.error_message || 'PipelineRun failed');
      await new Promise((resolve) => setTimeout(resolve, 2500));
    }
  }

  function jobPayload(overrides = {}) {
    const steps = Math.round(numberValue('portraitSteps', 25));
    const seed = optionalInt('portraitSeed');
    if (modeIsRefine()) {
      const refine = { preserve_strength: overrides.preserve_strength ?? numberValue('portraitPreserveStrength', 0.75), refine_strength: overrides.refine_strength ?? numberValue('portraitRefineStrength', 0.28), auto_straighten: el.portraitAutoStraighten.checked, steps, seed };
      return { mode: REFINE_MODE, dataset_id: state.datasetId || null, source_item_id: state.source?.item_id || null, source_run_id: state.refine.source_run_id, original_image_uri: state.refine.original_image_uri, sam_visible_uri: state.refine.sam_visible_uri, species: el.portraitSpecies.value.trim() || state.refine.species || null, prompt: el.portraitPrompt.value.trim() || REFINE_PROMPT, negative_prompt: el.portraitNegativePrompt.value.trim() || REFINE_NEGATIVE, preserve_strength: refine.preserve_strength, refine_strength: refine.refine_strength, auto_straighten: refine.auto_straighten, steps, seed, refine };
    }
    if (modeIsInpaint()) {
      const inpaint = { strength: overrides.strength ?? numberValue('portraitStrength', 0.25), steps, width: Math.round(numberValue('portraitWidth', 768)), height: Math.round(numberValue('portraitHeight', 768)), seed };
      return { mode: INPAINT_MODE, dataset_id: state.datasetId || null, source_item_id: state.source?.item_id || null, original_image_uri: state.inpaint.original_image_uri, fish_mask_uri: state.inpaint.fish_mask_uri, completion_mask_uri: state.inpaint.completion_mask_uri, species: el.portraitSpecies.value.trim() || state.inpaint.species || null, prompt: el.portraitPrompt.value.trim() || INPAINT_PROMPT, negative_prompt: el.portraitNegativePrompt.value.trim() || INPAINT_NEGATIVE, inpaint };
    }
    const reference = state.references[state.referenceIndex];
    return { mode: V1_MODE, dataset_id: state.datasetId, source_item_id: state.source.item_id, reference_asset_id: reference.asset_id, model: 'sdxl_ip_adapter', params: { source_scale: numberValue('portraitSourceScale', 0.8), reference_scale: numberValue('portraitReferenceScale', 0.35), steps, width: Math.round(numberValue('portraitWidth', 768)), height: Math.round(numberValue('portraitHeight', 768)) } };
  }

  async function createJob(overrides = {}) { return platformFetch('/api/platform/portrait/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(jobPayload(overrides)) }); }

  async function submit() {
    if (state.busy) return;
    if (modeIsRefine() && !state.refine) return;
    if (modeIsInpaint() && !state.inpaint) return;
    if (modeIsV1() && (!state.source || !state.references[state.referenceIndex])) return;
    state.busy = true;
    updateSubmit();
    el.portraitOutputEmpty.hidden = false;
    el.portraitMeta.hidden = true;
    setStatus('PENDING', '排队中');
    el.portraitRunTitle.textContent = '正在创建 Fish Portrait PipelineRun…';
    try {
      const data = await createJob();
      state.runId = data.run_id || data.pipeline_run_id;
      if (!state.runId) throw new Error('API 未返回 run_id');
      renderRun(data);
      await waitForRun(state.runId);
    } catch (error) {
      state.busy = false;
      setStatus('FAILED', '创建失败');
      el.portraitRunTitle.textContent = '任务创建失败';
      el.portraitProgress.innerHTML = `<div class="error-state">${esc(platformError(error))}</div>`;
      updateSubmit();
    } finally { if (!state.busy) updateSubmit(); }
  }

  async function runSweep() {
    if (state.busy || (modeIsRefine() && !state.refine) || (modeIsInpaint() && !state.inpaint) || modeIsV1()) {
      el.portraitABTestHint.textContent = modeIsV1() ? 'V1 先运行单次历史对照。' : '请先准备输入。';
      return;
    }
    state.busy = true;
    updateSubmit();
    el.portraitABTest.disabled = true;
    el.portraitSweepResults.hidden = false;
    el.portraitSweepResults.innerHTML = '';
    const cases = modeIsRefine() ? [{ label: 'A', preserve_strength: 0.80, refine_strength: 0.20 }, { label: 'B', preserve_strength: 0.75, refine_strength: 0.28 }, { label: 'C', preserve_strength: 0.70, refine_strength: 0.35 }] : [{ label: 'A', strength: 0.15 }, { label: 'B', strength: 0.25 }, { label: 'C', strength: 0.35 }];
    try {
      for (const experiment of cases) {
        el.portraitABTestHint.textContent = `${experiment.label} 对照生成中…`;
        const data = await createJob(experiment);
        const runId = data.run_id || data.pipeline_run_id;
        if (!runId) throw new Error('A/B 任务未返回 run_id');
        const result = await waitForRun(runId);
        const image = modeIsRefine() ? result.final_image : result.generated_image;
        const label = modeIsRefine() ? `A ${experiment.preserve_strength.toFixed(2)} / B ${experiment.refine_strength.toFixed(2)}` : `Strength ${experiment.strength.toFixed(2)}`;
        el.portraitSweepResults.insertAdjacentHTML('beforeend', `<div class="portrait-sweep-card"><strong>${esc(label)}</strong>${image ? `<img src="${esc(image)}" alt="${esc(label)} 结果">` : '<div class="portrait-empty">无结果</div>'}</div>`);
      }
      el.portraitABTestHint.textContent = modeIsRefine() ? 'A/B 对照完成：.80/.20 · .75/.28 · .70/.35。' : 'Strength 对照完成：0.15 / 0.25 / 0.35。';
    } catch (error) { el.portraitABTestHint.textContent = platformError(error); }
    finally { state.busy = false; el.portraitABTest.disabled = false; updateSubmit(); }
  }

  async function loadItems() {
    if (!state.datasetId) { renderSourceItems([]); return; }
    el.portraitSourceState.hidden = false;
    el.portraitSourceState.className = 'portrait-empty';
    el.portraitSourceState.textContent = '正在加载 Dataset 图片…';
    el.portraitSourceGrid.hidden = true;
    try {
      const payload = await platformFetch(`/api/platform/datasets/${encodeURIComponent(state.datasetId)}/items?page=1&size=60`);
      renderSourceItems(payload.items || []);
    } catch (error) {
      el.portraitSourceState.className = 'error-state';
      el.portraitSourceState.textContent = platformError(error);
      el.portraitSourceGrid.hidden = true;
    }
  }

  async function loadDatasets() {
    try {
      const rows = normalizeDatasetPayload(await platformFetch('/api/platform/datasets'));
      renderDatasetOptions(rows);
      await loadItems();
    } catch (error) {
      el.portraitDataset.innerHTML = '<option value="">数据集加载失败</option>';
      el.portraitDatasetHint.textContent = platformError(error);
      renderSourceItems([]);
    }
  }

  el.portraitMode.addEventListener('change', updateModeUI);
  el.portraitDataset.addEventListener('change', async (event) => { state.datasetId = event.target.value; el.portraitDatasetHint.textContent = state.datasetId ? `正在读取 ${state.datasetId}…` : '请选择已冻结 Dataset。'; await loadItems(); });
  el.portraitSpecies.addEventListener('input', () => { if (state.inpaint) state.inpaint.species = el.portraitSpecies.value.trim(); if (state.refine) state.refine.species = el.portraitSpecies.value.trim(); updateSubmit(); });
  el.portraitOriginalUpload.addEventListener('change', () => { if (el.portraitOriginalUpload.files?.[0]) { state.source = { item_id: null, image_id: el.portraitOriginalUpload.files[0].name, species_name: '' }; el.portraitSpecies.value = ''; clearPrepared(); el.portraitSourceState.textContent = `已选择上传文件：${el.portraitOriginalUpload.files[0].name}`; updateSubmit(); } });
  el.portraitPrepare.addEventListener('click', prepareInput);
  el.portraitReferenceSwitch.addEventListener('click', () => { if (state.references.length < 2) return; state.referenceIndex = (state.referenceIndex + 1) % state.references.length; renderReference(); });
  el.portraitSubmit.addEventListener('click', submit);
  el.portraitABTest.addEventListener('click', runSweep);
  el.portraitRefresh.addEventListener('click', loadDatasets);
  updateModeUI();
  loadWorkerHealth();
  loadDatasets();
})();
