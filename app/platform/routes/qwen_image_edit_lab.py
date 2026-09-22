@router.post("/generate")
async def generate_qwen_image_edit_lab(
    image: UploadFile | None = File(default=None),
    prompt: str = Form(default=DEFAULT_PROMPT),
    negative_prompt: str = Form(default=DEFAULT_NEGATIVE_PROMPT),
    seed: str | None = Form(default=None),
    dataset_id: str | None = Form(default=None),
    dataset_item_id: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    dataset_id_value = dataset_id.strip() if isinstance(dataset_id, str) else ""
    dataset_item_id_value = dataset_item_id.strip() if isinstance(dataset_item_id, str) else ""
    has_upload = image is not None
    has_dataset = bool(dataset_id_value or dataset_item_id_value)
    if has_upload and has_dataset:
        raise HTTPException(status_code=422, detail="本地上传和 Dataset 图片不能同时提交")
    if not has_upload and not (dataset_id_value and dataset_item_id_value):
        raise HTTPException(status_code=422, detail="请选择本地图片或 Dataset 图片")

    if has_upload:
        data, media_type, extension = await _read_upload(image)
        source = {"input_source": "LOCAL_UPLOAD"}
    else:
        data, media_type, extension, source = _read_dataset_image(
            db,
            dataset_id_value,
            dataset_item_id_value,
        )

    prompt_value = _normalise_text(prompt, DEFAULT_PROMPT, "Prompt")
    negative_prompt_value = _normalise_text(negative_prompt, DEFAULT_NEGATIVE_PROMPT, "Negative Prompt")
    seed_value = _parse_seed(seed)

    run_id = _new_run_id()
    input_uri = _store_bytes(run_id, "input", data, media_type, extension)
    request_state = {
        "input_image_uri": input_uri,
        "prompt": prompt_value,
        "negative_prompt": negative_prompt_value,
        "seed": seed_value,
        "model": MODEL_ID,
        "worker": WORKER_NAME,
        **source,
    }
    state: dict[str, Any] = {
        "type": PIPELINE_TYPE,
        "request": request_state,
        "result": None,
        "worker": None,
        "stages": [
            {"name": "upload_input", "status": "DONE"},
            {"name": "qwen_generate", "status": "RUNNING"},
            {"name": "persist_result", "status": "PENDING"},
        ],
    }
    run = PipelineRun(
        run_id=run_id,
        pipeline_type=PIPELINE_TYPE,
        status="RUNNING",
        current_stage="qwen_generate",
        started_at=_utcnow(),
        stage_json=json.dumps(state, ensure_ascii=False),
        model_version=MODEL_ID,
    )
    db.add(run)
    adapters.record_operation(
        db,
        "CREATE_QWEN_IMAGE_EDIT_LAB_RUN",
        "PIPELINE_RUN",
        run_id,
        detail={
            "type": PIPELINE_TYPE,
            "input_image": input_uri,
            "input_source": source.get("input_source"),
            "dataset_id": source.get("dataset_id"),
            "dataset_item_id": source.get("dataset_item_id"),
            "image_id": source.get("image_id"),
            "prompt": prompt_value,
            "negative_prompt": negative_prompt_value,
            "model": MODEL_ID,
            "worker": WORKER_NAME,
        },
    )
    db.commit()

    started = time.perf_counter()
    active_stage = "qwen_generate"
    try:
        worker_result = invoke_qwen_refine_worker(
            visible_fish_refined_image_uri=input_uri,
            source_run_id=run_id,
            prompt=prompt_value,
            negative_prompt=negative_prompt_value,
            seed=seed_value,
        )
        state["worker"] = {
            "name": WORKER_NAME,
            "model": worker_result.get("worker_model") or QWEN_MODEL_LABEL,
            "status": worker_result.get("worker_status") or "WORKER_EXECUTED",
            "http_status": worker_result.get("worker_http_status"),
            "protocol": worker_result.get("worker_protocol"),
        }
        _set_stage(state, "qwen_generate", "DONE")
        _set_stage(state, "persist_result", "RUNNING")
        active_stage = "persist_result"
        run.current_stage = "persist_result"
        _set_state(run, state)
        db.commit()

        output_uri, output_bytes = _materialize_output(run_id, worker_result)
        elapsed_ms = worker_result.get("elapsed_ms")
        if elapsed_ms is None:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        actual_seed = worker_result.get("seed")
        state["result"] = {
            "input_image_uri": input_uri,
            "output_image_uri": output_uri,
            "qwen_result_rgb_uri": output_uri,
            "seed": actual_seed if actual_seed is not None else seed_value,
            "elapsed_ms": elapsed_ms,
            "model": MODEL_ID,
            "worker": WORKER_NAME,
            "transparent_asset_status": "PROCESSING",
        }
        _set_stage(state, "persist_result", "DONE")
        _set_stage(state, "transparent_fish_export", "RUNNING")
        run.current_stage = "transparent_fish_export"
        _set_state(run, state)
        db.commit()

        run_status = "SUCCESS"
        try:
            artifacts = process_qwen_output(output_bytes)
            raw_mask_uri = _store_bytes(
                run_id, "qwen_fish_mask_raw", artifacts.fish_mask_raw, "image/png", ".png"
            )
            mask_uri = _store_bytes(
                run_id, "qwen_fish_mask", artifacts.fish_mask, "image/png", ".png"
            )
            transparent_uri = _store_bytes(
                run_id, "qwen_fish_rgba", artifacts.transparent_fish, "image/png", ".png"
            )
            state["result"].update(
                {
                    "fish_mask_raw_uri": raw_mask_uri,
                    "fish_mask_uri": mask_uri,
                    "transparent_fish_uri": transparent_uri,
                    "transparent_asset_status": "SUCCESS",
                    "transparent_asset_metadata": artifacts.metadata,
                }
            )
            _set_stage(state, "transparent_fish_export", "DONE")
        except (QwenOutputError, PortraitWorkerError) as exc:
            error_code = getattr(exc, "error_code", "QWEN_RGBA_EXPORT_FAILED")
            error_message = str(exc)[:3000]
            state["result"].update(
                {
                    "transparent_asset_status": "ERROR",
                    "transparent_asset_error": {
                        "code": error_code,
                        "message": error_message,
                        "details": getattr(exc, "details", {}) or {},
                    },
                }
            )
            _set_stage(state, "transparent_fish_export", "FAILED", f"{error_code}: {error_message}")
            run_status = "PARTIAL_SUCCESS"
        except Exception as exc:
            error_code = "QWEN_RGBA_EXPORT_FAILED"
            error_message = f"{exc.__class__.__name__}: {exc}"[:3000]
            state["result"].update(
                {
                    "transparent_asset_status": "ERROR",
                    "transparent_asset_error": {
                        "code": error_code,
                        "message": error_message,
                        "details": {},
                    },
                }
            )
            _set_stage(state, "transparent_fish_export", "FAILED", f"{error_code}: {error_message}")
            run_status = "PARTIAL_SUCCESS"

        run.status = run_status
        run.current_stage = "complete" if run_status == "SUCCESS" else "transparent_asset_error"
        run.finished_at = _utcnow()
        if run.started_at:
            run.duration_ms = _duration_ms(run.started_at, run.finished_at)
        _set_state(run, state)
        adapters.record_operation(
            db,
            "CREATE_QWEN_IMAGE_EDIT_LAB_RUN",
            "PIPELINE_RUN",
            run_id,
            status=run_status,
            message=(
                "Qwen Image Edit Lab 生成与透明鱼资产导出完成"
                if run_status == "SUCCESS"
                else "Qwen 生成完成，但透明鱼资产导出失败"
            ),
            detail={
                "type": PIPELINE_TYPE,
                "model": MODEL_ID,
                "worker": WORKER_NAME,
                "input_source": source.get("input_source"),
                "dataset_id": source.get("dataset_id"),
                "dataset_item_id": source.get("dataset_item_id"),
                "seed": state["result"]["seed"],
                "elapsed_ms": elapsed_ms,
                "qwen_generation_status": "SUCCESS",
                "transparent_asset_status": state["result"]["transparent_asset_status"],
                "transparent_asset_error": state["result"].get("transparent_asset_error"),
            },
        )
        db.commit()
        return _response(run, state)
    except PortraitWorkerError as exc:
        _mark_failed(
            db,
            run_id,
            stage=active_stage,
            error_code=exc.error_code,
            message=str(exc),
        )
        raise HTTPException(
            status_code=502,
            detail={"error_code": exc.error_code, "message": str(exc), "run_id": run_id},
        ) from exc
    except Exception as exc:
        _mark_failed(
            db,
            run_id,
            stage=active_stage,
            error_code="QWEN_IMAGE_EDIT_LAB_FAILED",
            message=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail={"error_code": "QWEN_IMAGE_EDIT_LAB_FAILED", "message": str(exc), "run_id": run_id},
        ) from exc


@router.get("/runs")
def qwen_image_edit_lab_runs(
    page: int = Query(default=1, ge=1, le=100000),
    size: int = Query(default=10, ge=1, le=100),
    limit: int | None = Query(default=None, ge=1, le=100),
    db: Session = Depends(get_db),
) -> dict[str, Any] | list[dict[str, Any]]:
    # Keep the legacy limit query usable for existing callers while the Lab UI
    # uses the paginated response with a stable ten-row page size.
    page_value = page if isinstance(page, int) else 1
    size_value = size if isinstance(size, int) else 10
    limit_value = limit if isinstance(limit, int) else None
    legacy_limit = limit_value is not None
    if limit_value is not None:
        page_value = 1
        size_value = limit_value

    predicate = PipelineRun.pipeline_type == PIPELINE_TYPE
    total = int(
        db.scalar(
            select(func.count())
            .select_from(PipelineRun)
            .where(predicate)
        )
        or 0
    )
    rows = db.scalars(
        select(PipelineRun)
        .where(predicate)
        .order_by(PipelineRun.created_at.desc(), PipelineRun.run_id.desc())
        .offset((page_value - 1) * size_value)
        .limit(size_value)
    ).all()
    items = [_response(row, _state_for_run(row)) for row in rows]
    if legacy_limit:
        return items
    page_count = max(1, (total + size_value - 1) // size_value)
    return {
        "items": items,
        "total": total,
        "page": page_value,
        "size": size_value,
        "page_count": page_count,
        "has_next": page_value < page_count,
    }


@router.get("/runs/{run_id}")
def qwen_image_edit_lab_run(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    run = db.get(PipelineRun, run_id)
    if run is None or run.pipeline_type != PIPELINE_TYPE:
        raise HTTPException(status_code=404, detail="Qwen Image Edit Lab 记录不存在")
    return _response(run, _state_for_run(run))


@router.get("/runs/{run_id}/media/{kind}")
def qwen_image_edit_lab_media(run_id: str, kind: str, db: Session = Depends(get_db)) -> Response:
    allowed_kinds = {
        "input",
        "output",
        "qwen_result_rgb",
        "fish_mask_raw",
        "fish_mask",
        "transparent_fish",
    }
    if kind not in allowed_kinds:
        raise HTTPException(status_code=404, detail="资源不存在")
    run = db.get(PipelineRun, run_id)
    if run is None or run.pipeline_type != PIPELINE_TYPE:
        raise HTTPException(status_code=404, detail="Qwen Image Edit Lab 记录不存在")
    state = _state_for_run(run)
    request = state.get("request") if isinstance(state.get("request"), dict) else {}
    result = state.get("result") if isinstance(state.get("result"), dict) else {}
    uri_by_kind = {
        "input": request.get("input_image_uri"),
        "output": result.get("qwen_result_rgb_uri") or result.get("output_image_uri"),
        "qwen_result_rgb": result.get("qwen_result_rgb_uri") or result.get("output_image_uri"),
        "fish_mask_raw": result.get("fish_mask_raw_uri"),
        "fish_mask": result.get("fish_mask_uri"),
        "transparent_fish": result.get("transparent_fish_uri"),
    }
    uri = uri_by_kind.get(kind)
    if not uri:
        raise HTTPException(status_code=404, detail="资源不存在")
    try:
        content, media_type = _read_managed_uri(str(uri))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="资源不存在") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="资源暂时不可用") from exc
    return Response(
        content=content,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
    )



__all__ = [
    "DEFAULT_NEGATIVE_PROMPT",
    "DEFAULT_PROMPT",
    "MODEL_ID",
    "PIPELINE_TYPE",
    "WORKER_NAME",
    "generate_qwen_image_edit_lab",
    "extract_qwen_image_edit_lab_transparent",
    "qwen_image_edit_lab_media",
    "qwen_image_edit_lab_run",
    "router",
]
@router.post("/generate")
async def generate_qwen_image_edit_lab(
    image: UploadFile | None = File(default=None),
    prompt: str = Form(default=DEFAULT_PROMPT),
    negative_prompt: str = Form(default=DEFAULT_NEGATIVE_PROMPT),
    seed: str | None = Form(default=None),
    dataset_id: str | None = Form(default=None),
    dataset_item_id: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    dataset_id_value = dataset_id.strip() if isinstance(dataset_id, str) else ""
    dataset_item_id_value = dataset_item_id.strip() if isinstance(dataset_item_id, str) else ""
    has_upload = image is not None
    has_dataset = bool(dataset_id_value or dataset_item_id_value)
    if has_upload and has_dataset:
        raise HTTPException(status_code=422, detail="本地上传和 Dataset 图片不能同时提交")
    if not has_upload and not (dataset_id_value and dataset_item_id_value):
        raise HTTPException(status_code=422, detail="请选择本地图片或 Dataset 图片")

    if has_upload:
        data, media_type, extension = await _read_upload(image)
        source = {"input_source": "LOCAL_UPLOAD"}
    else:
        data, media_type, extension, source = _read_dataset_image(
            db,
            dataset_id_value,
            dataset_item_id_value,
        )

    prompt_value = _normalise_text(prompt, DEFAULT_PROMPT, "Prompt")
    negative_prompt_value = _normalise_text(negative_prompt, DEFAULT_NEGATIVE_PROMPT, "Negative Prompt")
    seed_value = _parse_seed(seed)

    run_id = _new_run_id()
    input_uri = _store_bytes(run_id, "input", data, media_type, extension)
    request_state = {
        "input_image_uri": input_uri,
        "prompt": prompt_value,
        "negative_prompt": negative_prompt_value,
        "seed": seed_value,
        "model": MODEL_ID,
        "worker": WORKER_NAME,
        **source,
    }
    state: dict[str, Any] = {
        "type": PIPELINE_TYPE,
        "request": request_state,
        "result": None,
        "worker": None,
        "stages": [
            {"name": "upload_input", "status": "DONE"},
            {"name": "qwen_generate", "status": "RUNNING"},
            {"name": "persist_result", "status": "PENDING"},
        ],
    }
    run = PipelineRun(
        run_id=run_id,
        pipeline_type=PIPELINE_TYPE,
        status="RUNNING",
        current_stage="qwen_generate",
        started_at=_utcnow(),
        stage_json=json.dumps(state, ensure_ascii=False),
        model_version=MODEL_ID,
    )
    db.add(run)
    adapters.record_operation(
        db,
        "CREATE_QWEN_IMAGE_EDIT_LAB_RUN",
        "PIPELINE_RUN",
        run_id,
        detail={
            "type": PIPELINE_TYPE,
            "input_image": input_uri,
            "input_source": source.get("input_source"),
            "dataset_id": source.get("dataset_id"),
            "dataset_item_id": source.get("dataset_item_id"),
            "image_id": source.get("image_id"),
            "prompt": prompt_value,
            "negative_prompt": negative_prompt_value,
            "model": MODEL_ID,
            "worker": WORKER_NAME,
        },
    )
    db.commit()

    started = time.perf_counter()
    active_stage = "qwen_generate"
    try:
        worker_result = invoke_qwen_refine_worker(
            visible_fish_refined_image_uri=input_uri,
            source_run_id=run_id,
            prompt=prompt_value,
            negative_prompt=negative_prompt_value,
            seed=seed_value,
        )
        state["worker"] = {
            "name": WORKER_NAME,
            "model": worker_result.get("worker_model") or QWEN_MODEL_LABEL,
            "status": worker_result.get("worker_status") or "WORKER_EXECUTED",
            "http_status": worker_result.get("worker_http_status"),
            "protocol": worker_result.get("worker_protocol"),
        }
        _set_stage(state, "qwen_generate", "DONE")
        _set_stage(state, "persist_result", "RUNNING")
        active_stage = "persist_result"
        run.current_stage = "persist_result"
        _set_state(run, state)
        db.commit()

        output_uri, _ = _materialize_output(run_id, worker_result)
        elapsed_ms = worker_result.get("elapsed_ms")
        if elapsed_ms is None:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        actual_seed = worker_result.get("seed")
        state["result"] = {
            "input_image_uri": input_uri,
            "output_image_uri": output_uri,
            "qwen_result_rgb_uri": output_uri,
            "qwen_status": "SUCCESS",
            "seed": actual_seed if actual_seed is not None else seed_value,
            "elapsed_ms": elapsed_ms,
            "model": MODEL_ID,
            "worker": WORKER_NAME,
            "transparent_status": "NOT_STARTED",
            "transparent_asset_status": "NOT_STARTED",
            "fish_mask_raw_uri": None,
            "fish_mask_uri": None,
            "transparent_fish_uri": None,
        }
        _set_stage(state, "persist_result", "DONE")
        run.status = "SUCCESS"
        run.current_stage = "complete"
        run.finished_at = _utcnow()
        if run.started_at:
            run.duration_ms = _duration_ms(run.started_at, run.finished_at)
        _set_state(run, state)
        adapters.record_operation(
            db,
            "CREATE_QWEN_IMAGE_EDIT_LAB_RUN",
            "PIPELINE_RUN",
            run_id,
            status="SUCCESS",
            message="Qwen RGB 生成完成，透明鱼体等待用户触发",
            detail={
                "type": PIPELINE_TYPE,
                "model": MODEL_ID,
                "worker": WORKER_NAME,
                "input_source": source.get("input_source"),
                "dataset_id": source.get("dataset_id"),
                "dataset_item_id": source.get("dataset_item_id"),
                "seed": state["result"]["seed"],
                "elapsed_ms": elapsed_ms,
                "qwen_status": "SUCCESS",
                "transparent_status": "NOT_STARTED",
            },
        )
        db.commit()
        return _response(run, state)
    except PortraitWorkerError as exc:
        _mark_failed(
            db,
            run_id,
            stage=active_stage,
            error_code=exc.error_code,
            message=str(exc),
        )
        raise HTTPException(
            status_code=502,
            detail={"error_code": exc.error_code, "message": str(exc), "run_id": run_id},
        ) from exc
    except Exception as exc:
        _mark_failed(
            db,
            run_id,
            stage=active_stage,
            error_code="QWEN_IMAGE_EDIT_LAB_FAILED",
            message=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail={"error_code": "QWEN_IMAGE_EDIT_LAB_FAILED", "message": str(exc), "run_id": run_id},
        ) from exc


@router.post("/runs/{run_id}/extract-transparent")
def extract_qwen_image_edit_lab_transparent(
    run_id: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    run = db.get(PipelineRun, run_id)
    if run is None or run.pipeline_type != PIPELINE_TYPE:
        raise HTTPException(status_code=404, detail="Qwen Image Edit Lab 记录不存在")

    state = _state_for_run(run)
    result = state.get("result") if isinstance(state.get("result"), dict) else {}
    output_uri = str(
        result.get("qwen_result_rgb_uri")
        or result.get("output_image_uri")
        or ""
    ).strip()
    qwen_status = str(
        result.get("qwen_status")
        or ("SUCCESS" if output_uri and str(run.status or "").upper() in {"SUCCESS", "PARTIAL_SUCCESS"} else run.status)
    ).upper()
    if not output_uri or qwen_status != "SUCCESS":
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "QWEN_RESULT_NOT_READY",
                "message": "Qwen RGB 结果尚未生成，不能提取透明鱼体",
                "run_id": run_id,
            },
        )

    transparent_uri = str(result.get("transparent_fish_uri") or "").strip()
    transparent_status = str(
        result.get("transparent_status")
        or result.get("transparent_asset_status")
        or ("SUCCESS" if transparent_uri else "NOT_STARTED")
    ).upper()
    if transparent_uri and transparent_status == "SUCCESS":
        return _response(run, state)
    if transparent_status == "PROCESSING":
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "TRANSPARENT_FISH_BUSY",
                "message": "透明鱼体正在提取，请稍候",
                "run_id": run_id,
            },
        )

    result.update(
        {
            "transparent_status": "PROCESSING",
            "transparent_asset_status": "PROCESSING",
            "transparent_asset_error": None,
        }
    )
    state["result"] = result
    _set_stage(state, "transparent_fish_export", "RUNNING")
    run.current_stage = "transparent_fish_export"
    _set_state(run, state)
    db.commit()

    try:
        output_bytes, _ = _read_managed_uri(output_uri)
        artifacts = process_qwen_output(output_bytes)
        raw_mask_uri = _store_bytes(
            run_id, "qwen_fish_mask_raw", artifacts.fish_mask_raw, "image/png", ".png"
        )
        mask_uri = _store_bytes(
            run_id, "qwen_fish_mask", artifacts.fish_mask, "image/png", ".png"
        )
        transparent_uri = _store_bytes(
            run_id, "qwen_fish_rgba", artifacts.transparent_fish, "image/png", ".png"
        )
        result.update(
            {
                "fish_mask_raw_uri": raw_mask_uri,
                "fish_mask_uri": mask_uri,
                "transparent_fish_uri": transparent_uri,
                "transparent_status": "SUCCESS",
                "transparent_asset_status": "SUCCESS",
                "transparent_asset_metadata": artifacts.metadata,
                "transparent_asset_error": None,
            }
        )
        _set_stage(state, "transparent_fish_export", "DONE")
        run.current_stage = "complete"
        _set_state(run, state)
        adapters.record_operation(
            db,
            "EXTRACT_QWEN_IMAGE_EDIT_LAB_TRANSPARENT_FISH",
            "PIPELINE_RUN",
            run_id,
            status="SUCCESS",
            message="透明鱼体提取完成",
            detail={
                "type": PIPELINE_TYPE,
                "model": MODEL_ID,
                "transparent_status": "SUCCESS",
                "fish_mask_uri": mask_uri,
                "transparent_fish_uri": transparent_uri,
            },
        )
        db.commit()
        return _response(run, state)
    except (QwenOutputError, PortraitWorkerError) as exc:
        error_code = getattr(exc, "error_code", "QWEN_RGBA_EXPORT_FAILED")
        error_message = str(exc)[:3000]
    except Exception as exc:
        error_code = "QWEN_RGBA_EXPORT_FAILED"
        error_message = f"{exc.__class__.__name__}: {exc}"[:3000]

    result.update(
        {
            "transparent_status": "ERROR",
            "transparent_asset_status": "ERROR",
            "transparent_asset_error": {
                "code": error_code,
                "message": error_message,
                "details": getattr(locals().get("exc"), "details", {}) or {},
            },
        }
    )
    _set_stage(state, "transparent_fish_export", "FAILED", f"{error_code}: {error_message}")
    run.current_stage = "complete"
    _set_state(run, state)
    adapters.record_operation(
        db,
        "EXTRACT_QWEN_IMAGE_EDIT_LAB_TRANSPARENT_FISH",
        "PIPELINE_RUN",
        run_id,
        status="FAILED",
        message=error_message,
        detail={
            "type": PIPELINE_TYPE,
            "model": MODEL_ID,
            "transparent_status": "ERROR",
            "error_code": error_code,
        },
    )
    db.commit()
    return _response(run, state)


@router.get("/runs")
def qwen_image_edit_lab_runs(
    page: int = Query(default=1, ge=1, le=100000),
    size: int = Query(default=10, ge=1, le=100),
    limit: int | None = Query(default=None, ge=1, le=100),
    db: Session = Depends(get_db),
) -> dict[str, Any] | list[dict[str, Any]]:
    # Keep the legacy limit query usable for existing callers while the Lab UI
    # uses the paginated response with a stable ten-row page size.
    page_value = page if isinstance(page, int) else 1
    size_value = size if isinstance(size, int) else 10
    limit_value = limit if isinstance(limit, int) else None
    legacy_limit = limit_value is not None
    if limit_value is not None:
        page_value = 1
        size_value = limit_value

    predicate = PipelineRun.pipeline_type == PIPELINE_TYPE
    total = int(
        db.scalar(
            select(func.count())
            .select_from(PipelineRun)
            .where(predicate)
        )
        or 0
    )
    rows = db.scalars(
        select(PipelineRun)
        .where(predicate)
        .order_by(PipelineRun.created_at.desc(), PipelineRun.run_id.desc())
        .offset((page_value - 1) * size_value)
        .limit(size_value)
    ).all()
    items = [_response(row, _state_for_run(row)) for row in rows]
    if legacy_limit:
        return items
    page_count = max(1, (total + size_value - 1) // size_value)
    return {
        "items": items,
        "total": total,
        "page": page_value,
        "size": size_value,
        "page_count": page_count,
        "has_next": page_value < page_count,
    }


@router.get("/runs/{run_id}")
def qwen_image_edit_lab_run(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    run = db.get(PipelineRun, run_id)
    if run is None or run.pipeline_type != PIPELINE_TYPE:
        raise HTTPException(status_code=404, detail="Qwen Image Edit Lab 记录不存在")
    return _response(run, _state_for_run(run))


@router.get("/runs/{run_id}/media/{kind}")
def qwen_image_edit_lab_media(run_id: str, kind: str, db: Session = Depends(get_db)) -> Response:
    allowed_kinds = {
        "input",
        "output",
        "qwen_result_rgb",
        "fish_mask_raw",
        "fish_mask",
        "transparent_fish",
    }
    if kind not in allowed_kinds:
        raise HTTPException(status_code=404, detail="资源不存在")
    run = db.get(PipelineRun, run_id)
    if run is None or run.pipeline_type != PIPELINE_TYPE:
        raise HTTPException(status_code=404, detail="Qwen Image Edit Lab 记录不存在")
    state = _state_for_run(run)
    request = state.get("request") if isinstance(state.get("request"), dict) else {}
    result = state.get("result") if isinstance(state.get("result"), dict) else {}
    uri_by_kind = {
        "input": request.get("input_image_uri"),
        "output": result.get("qwen_result_rgb_uri") or result.get("output_image_uri"),
        "qwen_result_rgb": result.get("qwen_result_rgb_uri") or result.get("output_image_uri"),
        "fish_mask_raw": result.get("fish_mask_raw_uri"),
        "fish_mask": result.get("fish_mask_uri"),
        "transparent_fish": result.get("transparent_fish_uri"),
    }
    uri = uri_by_kind.get(kind)
    if not uri:
        raise HTTPException(status_code=404, detail="资源不存在")
    try:
        content, media_type = _read_managed_uri(str(uri))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="资源不存在") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="资源暂时不可用") from exc
    return Response(
        content=content,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
    )



__all__ = [
    "DEFAULT_NEGATIVE_PROMPT",
    "DEFAULT_PROMPT",
    "MODEL_ID",
    "PIPELINE_TYPE",
    "WORKER_NAME",
    "generate_qwen_image_edit_lab",
    "qwen_image_edit_lab_media",
    "qwen_image_edit_lab_run",
    "router",
]
