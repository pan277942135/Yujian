(() => {
  const state = {
    datasetId: '',
    sourceItems: [],
    source: null,
    references: [],
    referenceIndex: 0,
    runId: null,
    pollTimer: null,
    busy: false,
  };

  const ids = [
    'portraitDataset', 'portraitDatasetHint', 'portraitSourceState',
    'portraitSourceGrid', 'portraitReferenceState', 'portraitReference',
    'portraitReferenceImage', 'portraitReferenceTitle', 'portraitReferenceMeta',
    'portraitReferenceSwitch', 'portraitSubmit', 'portraitSubmitHint',
    'portraitRefresh', 'portraitProgress', 'portraitRunTitle',
    'portraitRunStatus', 'portraitSourceImage', 'portraitReferenceResultImage',
    'portraitGeneratedImage', 'portraitOutputEmpty', 'portraitMeta',
  ];
  const el = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));

  const stageLabels = {
    load_source: '读取真实照片 A',
    load_reference: '读取标准鱼体 B',
    sdxl_generate: 'SDXL + IP-Adapter 生成',
    persist_result: '保存实验结果',
    complete: '完成',
  };
  const statusLabels = {
    PENDING: '排队中',
    QUEUED: '排队中',
    RUNNING: '运行中',
    DONE: '已完成',
    SUCCESS: '成功',
    FAILED: '失败',
  };
  const referenceTypeLabels = {
    transparent_main: 'transparent 主图',
    transparent_alt: 'transparent 备选',
    COVER_CARD: 'Cover 列表图',
    COVER_CARD_TRANSPARENT_LEFT: 'Cover 透明左图',
    COVER_CARD_TRANSPARENT_RIGHT: 'Cover 透明右图',
  };

  function statusTag(status) {
    const normalized = String(status || 'UNKNOWN').toUpperCase();
    const label = statusLabels[normalized] || normalized;
    return '<span class="status-tag status-' + normalized.toLowerCase() + '"><span class="status-dot"></span>' + esc(label) + '</span>';
  }

  function setStatus(status, text) {
    if (!el.portraitRunStatus) return;
    const normalized = String(status || 'UNKNOWN').toUpperCase();
    el.portraitRunStatus.className = 'status-tag status-' + normalized.toLowerCase();
    el.portraitRunStatus.innerHTML = '<span class="status-dot"></span>' + esc(text || statusLabels[normalized] || normalized);
  }

  function selectedDatasetRow(rows) {
    return rows.find(row => String(row.id || row.dataset_id || '') === state.datasetId);
  }

  function normalizeDatasetPayload(payload) {
    if (Array.isArray(payload)) return payload;
    return Array.isArray(payload && payload.datasets) ? payload.datasets : [];
  }

  function renderDatasetOptions(rows) {
    const ready = rows.filter(row => String(row.status || '').toUpperCase() === 'FROZEN');
    el.portraitDataset.innerHTML = '<option value="">请选择已冻结 Dataset</option>';
    rows.forEach(row => {
      const id = String(row.id || row.dataset_id || '');
      if (!id) return;
      const status = String(row.status || 'UNKNOWN').toUpperCase();
      const option = document.createElement('option');
      option.value = id;
      option.textContent = id + ' · ' + Number(row.total ?? row.image_count ?? 0) + ' 张 · ' + (status === 'FROZEN' ? '已冻结' : status);
      option.disabled = status !== 'FROZEN';
      el.portraitDataset.appendChild(option);
    });
    if (ready.length) {
      state.datasetId = String(ready[0].id || ready[0].dataset_id);
      el.portraitDataset.value = state.datasetId;
      el.portraitDatasetHint.textContent = '已选择 ' + state.datasetId + '；数据集为只读实验输入。';
    } else {
      state.datasetId = '';
      el.portraitDatasetHint.textContent = '暂无已冻结 Dataset，请先完成 Dataset Freeze。';
    }
  }

  function renderSourceItems(items) {
    state.sourceItems = Array.isArray(items) ? items : [];
    state.source = null;
    state.references = [];
    state.referenceIndex = 0;
    el.portraitReference.hidden = true;
    el.portraitReferenceState.hidden = false;
    el.portraitReferenceState.className = 'portrait-empty';
    el.portraitReferenceState.textContent = state.sourceItems.length ? '请选择一张真实照片 A。' : '该 Dataset 暂无可选图片。';
    if (!state.sourceItems.length) {
      el.portraitSourceGrid.hidden = true;
      el.portraitSourceState.hidden = false;
      el.portraitSourceState.textContent = '该 Dataset 暂无可选图片。';
      updateSubmit();
      return;
    }
    el.portraitSourceState.hidden = true;
    el.portraitSourceGrid.hidden = false;
    el.portraitSourceGrid.innerHTML = state.sourceItems.map((item, index) => (
      '<button type="button" class="portrait-source-card" data-source-index="' + index + '">' +
        '<img src="' + esc(item.preview_url || item.image_url || '') + '" alt="' + esc(item.image_id || '真实照片') + '">' +
        '<strong>' + esc(item.image_id || ('图片 ' + (index + 1))) + '</strong>' +
        '<small>' + esc(item.species_name || item.species_id || '') + ' · ' + esc(item.split || '—') + '</small>' +
      '</button>'
    )).join('');
    el.portraitSourceGrid.querySelectorAll('[data-source-index]').forEach(button => {
      button.addEventListener('click', () => selectSource(Number(button.dataset.sourceIndex)));
    });
    updateSubmit();
  }

  function renderReference() {
    const reference = state.references[state.referenceIndex];
    if (!reference) {
      el.portraitReference.hidden = true;
      el.portraitReferenceState.hidden = false;
      el.portraitReferenceState.className = 'portrait-empty';
      el.portraitReferenceState.textContent = '没有找到该鱼种的标准参考图（transparent / Cover 资产包）。';
      updateSubmit();
      return;
    }
    el.portraitReferenceState.hidden = true;
    el.portraitReference.hidden = false;
    el.portraitReferenceImage.src = reference.url || '';
    const referenceType = reference.type || 'transparent_main';
    el.portraitReferenceTitle.textContent = (reference.species_name || reference.species_id || '标准鱼体') + ' · ' + (referenceTypeLabels[referenceType] || referenceType);
    el.portraitReferenceMeta.textContent = 'Asset ' + (reference.asset_id || '—') + ' · ' + (reference.version || '—') + (reference.source_kind === 'knowledge_cover' ? ' · 鱼种 Cover 资产包' : '');
    el.portraitReferenceSwitch.hidden = state.references.length < 2;
    updateSubmit();
  }

  async function loadReferences(speciesId) {
    state.references = [];
    state.referenceIndex = 0;
    el.portraitReference.hidden = true;
    el.portraitReferenceState.hidden = false;
    el.portraitReferenceState.className = 'portrait-empty';
    el.portraitReferenceState.textContent = '正在按鱼种匹配标准参考图…';
    if (!speciesId) return;
    try {
      const primary = await platformFetch('/api/platform/portrait/reference/' + encodeURIComponent(speciesId));
      const list = await platformFetch('/api/platform/assets/fish-reference?species_id=' + encodeURIComponent(speciesId) + '&asset_type=transparent');
      const assets = Array.isArray(list.assets) ? list.assets : [];
      const first = primary.reference_asset;
      if (first && !assets.some(item => item.asset_id === first.asset_id)) assets.unshift(first);
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
    el.portraitSourceGrid.querySelectorAll('.portrait-source-card').forEach((button, buttonIndex) => {
      button.classList.toggle('selected', buttonIndex === index);
    });
    el.portraitSourceImage.src = source.image_url || source.preview_url || '';
    el.portraitSourceImage.hidden = !source.image_url && !source.preview_url;
    el.portraitReferenceResultImage.hidden = true;
    await loadReferences(source.species_id || source.species_name);
  }

  function numberValue(id, fallback) {
    const value = Number(document.getElementById(id).value);
    return Number.isFinite(value) ? value : fallback;
  }

  function updateSubmit() {
    const enabled = Boolean(state.source && state.references[state.referenceIndex] && !state.busy);
    el.portraitSubmit.disabled = !enabled;
    el.portraitSubmitHint.textContent = state.busy
      ? '任务运行中，请等待结果。'
      : enabled
        ? '点击后创建一个可追踪的 Fish Portrait PipelineRun。'
        : '请选择 A 图和标准参考图。';
  }

  function renderSteps(data) {
    const steps = Array.isArray(data.steps) && data.steps.length
      ? data.steps
      : [
          {name: 'load_source', status: 'PENDING'},
          {name: 'load_reference', status: 'PENDING'},
          {name: 'sdxl_generate', status: 'PENDING'},
          {name: 'persist_result', status: 'PENDING'},
        ];
    el.portraitProgress.innerHTML = steps.map(step => {
      const status = String(step.status || 'PENDING').toUpperCase();
      const cls = status === 'RUNNING' ? 'running' : (status === 'DONE' || status === 'SUCCESS' ? 'done' : (status === 'FAILED' ? 'failed' : ''));
      const detail = step.error || step.error_message || (step.duration_ms != null ? step.duration_ms + ' ms' : statusLabels[status] || status);
      return '<div class="portrait-step ' + cls + '"><span class="portrait-step-dot"></span><div><strong>' +
        esc(stageLabels[step.name] || step.name || '实验步骤') + '</strong><small>' + esc(detail) + '</small></div></div>';
    }).join('');
    if (data.error_message) {
      el.portraitProgress.insertAdjacentHTML('beforeend', '<div class="error-state">' + esc(data.error_stage || '失败阶段') + '：' + esc(data.error_message) + '</div>');
    }
  }

  function renderMeta(metadata) {
    const params = metadata && metadata.params ? metadata.params : {};
    el.portraitMeta.innerHTML = [
      '<div class="stat"><strong>' + esc(metadata && metadata.run_id || state.runId || '—') + '</strong><span>Run ID</span></div>',
      '<div class="stat"><strong>' + esc(metadata && metadata.model || 'SDXL + IP-Adapter') + '</strong><span>模型</span></div>',
      '<div class="stat"><strong>' + esc(params.ip_scale ?? '—') + ' · ' + esc(params.steps ?? '—') + ' steps</strong><span>参数</span></div>',
    ].join('');
    el.portraitMeta.hidden = false;
  }

  function renderRun(data) {
    const status = String(data.status || 'UNKNOWN').toUpperCase();
    setStatus(status, statusLabels[status] || status);
    el.portraitRunTitle.textContent = (data.run_id || state.runId || '实验任务') + (data.current_stage ? ' · ' + (stageLabels[data.current_stage] || data.current_stage) : '');
    renderSteps(data);
    if (data.source && data.source.image_url) {
      el.portraitSourceImage.src = data.source.image_url;
      el.portraitSourceImage.hidden = false;
    }
    if (data.reference && data.reference.url) {
      el.portraitReferenceResultImage.src = data.reference.url;
      el.portraitReferenceResultImage.hidden = false;
    }
    if (status === 'FAILED') {
      state.busy = false;
      updateSubmit();
    }
  }

  async function loadResult(runId) {
    const result = await platformFetch('/api/platform/portrait/results/' + encodeURIComponent(runId));
    if (result.source_image) {
      el.portraitSourceImage.src = result.source_image;
      el.portraitSourceImage.hidden = false;
    }
    if (result.reference_image) {
      el.portraitReferenceResultImage.src = result.reference_image;
      el.portraitReferenceResultImage.hidden = false;
    }
    if (result.generated_image) {
      el.portraitGeneratedImage.src = result.generated_image;
      el.portraitGeneratedImage.hidden = false;
      el.portraitOutputEmpty.hidden = true;
    }
    renderMeta(result.metadata || {});
  }

  async function poll(runId) {
    if (state.pollTimer) clearTimeout(state.pollTimer);
    try {
      const data = await platformFetch('/api/platform/pipeline/' + encodeURIComponent(runId));
      if (state.runId !== runId) return;
      renderRun(data);
      const status = String(data.status || '').toUpperCase();
      if (status === 'SUCCESS') {
        await loadResult(runId);
        state.busy = false;
        updateSubmit();
        return;
      }
      if (status === 'FAILED') return;
      state.pollTimer = setTimeout(() => poll(runId), 2500);
    } catch (error) {
      if (state.runId !== runId) return;
      el.portraitProgress.insertAdjacentHTML('beforeend', '<div class="error-state">' + esc(platformError(error)) + '</div>');
      state.pollTimer = setTimeout(() => poll(runId), 4000);
    }
  }

  async function submit() {
    if (!state.source || !state.references[state.referenceIndex] || state.busy) return;
    state.busy = true;
    updateSubmit();
    el.portraitGeneratedImage.hidden = true;
    el.portraitOutputEmpty.hidden = false;
    el.portraitMeta.hidden = true;
    setStatus('PENDING', '排队中');
    el.portraitRunTitle.textContent = '正在创建 Fish Portrait PipelineRun…';
    try {
      const reference = state.references[state.referenceIndex];
      const data = await platformFetch('/api/platform/portrait/jobs', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          dataset_id: state.datasetId,
          source_item_id: state.source.item_id,
          reference_asset_id: reference.asset_id,
          model: 'sdxl_ip_adapter',
          params: {
            ip_scale: numberValue('portraitIpScale', 0.8),
            steps: Math.round(numberValue('portraitSteps', 25)),
            width: Math.round(numberValue('portraitWidth', 768)),
            height: Math.round(numberValue('portraitHeight', 768)),
          },
        }),
      });
      state.runId = data.run_id || data.pipeline_run_id;
      if (!state.runId) throw new Error('API 未返回 run_id');
      renderRun(data);
      await poll(state.runId);
    } catch (error) {
      state.busy = false;
      setStatus('FAILED', '创建失败');
      el.portraitRunTitle.textContent = '任务创建失败';
      el.portraitProgress.innerHTML = '<div class="error-state">' + esc(platformError(error)) + '</div>';
      updateSubmit();
    }
  }

  async function loadItems() {
    const datasetId = state.datasetId;
    if (!datasetId) {
      renderSourceItems([]);
      return;
    }
    el.portraitSourceState.hidden = false;
    el.portraitSourceState.className = 'portrait-empty';
    el.portraitSourceState.textContent = '正在加载 Dataset 图片…';
    el.portraitSourceGrid.hidden = true;
    try {
      const payload = await platformFetch('/api/platform/datasets/' + encodeURIComponent(datasetId) + '/items?page=1&size=60');
      renderSourceItems(payload.items || []);
    } catch (error) {
      el.portraitSourceState.className = 'error-state';
      el.portraitSourceState.textContent = platformError(error);
      el.portraitSourceGrid.hidden = true;
    }
  }

  async function loadDatasets() {
    try {
      const payload = await platformFetch('/api/platform/datasets');
      const rows = normalizeDatasetPayload(payload);
      renderDatasetOptions(rows);
      await loadItems();
    } catch (error) {
      el.portraitDataset.innerHTML = '<option value="">数据集加载失败</option>';
      el.portraitDatasetHint.textContent = platformError(error);
      renderSourceItems([]);
    }
  }

  el.portraitDataset.addEventListener('change', async event => {
    state.datasetId = event.target.value;
    const row = selectedDatasetRow(normalizeDatasetPayload({datasets: []}));
    el.portraitDatasetHint.textContent = state.datasetId ? '正在读取 ' + state.datasetId + '…' : '请选择已冻结 Dataset。';
    await loadItems();
  });
  el.portraitReferenceSwitch.addEventListener('click', () => {
    if (state.references.length < 2) return;
    state.referenceIndex = (state.referenceIndex + 1) % state.references.length;
    renderReference();
  });
  el.portraitSubmit.addEventListener('click', submit);
  el.portraitRefresh.addEventListener('click', loadDatasets);

  loadDatasets();
})();
