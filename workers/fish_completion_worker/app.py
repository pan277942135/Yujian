"""GPU-only PowerPaint completion worker; no mock or input pass-through."""
from __future__ import annotations
import importlib.util, io, os, time
from pathlib import Path
from typing import Any
import numpy as np, torch
from fastapi import FastAPI, Header, HTTPException
from google.cloud import storage
from PIL import Image
from pydantic import BaseModel, Field
app=FastAPI(title="Yujian Fish Completion Worker",version="0.1")
class CompletionRequest(BaseModel):
    image_uri:str=Field(min_length=1); mask_uri:str=Field(min_length=1); prompt:str=Field(min_length=1)
_controller:Any=None; _model_error:str|None=None
def _read_uri(uri:str)->bytes:
    if uri.startswith("gs://"):
        bucket_name,object_name=uri[len("gs://"):].split("/",1)
        return storage.Client().bucket(bucket_name).blob(object_name).download_as_bytes()
    return Path(uri.removeprefix("file://")).read_bytes()
def _write_result(image_uri:str,data:bytes)->str:
    bucket_name=os.getenv("COMPLETION_OUTPUT_BUCKET","").strip()
    if not bucket_name: raise RuntimeError("COMPLETION_OUTPUT_BUCKET is not configured")
    source_name=image_uri.split("/",3)[-1] if image_uri.startswith("gs://") else "local/input.png"
    object_name=f"{source_name.rsplit('/',1)[0]}/09_generated_roi.png"
    blob=storage.Client().bucket(bucket_name).blob(object_name); blob.upload_from_string(data,content_type="image/png")
    return f"gs://{bucket_name}/{object_name}"
def _load_controller()->None:
    global _controller,_model_error
    if _controller is not None or _model_error is not None: return
    if not torch.cuda.is_available(): _model_error="CUDA is unavailable; PowerPaint must run on the GPU worker"; return
    checkpoint_dir=os.getenv("POWERPAINT_CHECKPOINT_DIR","/models/ppt-v1").strip()
    if not Path(checkpoint_dir).is_dir(): _model_error=f"PowerPaint checkpoint directory does not exist: {checkpoint_dir}"; return
    try:
        spec = importlib.util.spec_from_file_location("powerpaint_official_app", "/opt/PowerPaint/app.py")
        if spec is None or spec.loader is None:
            raise ImportError("cannot load official PowerPaint app")
        official_app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official_app)
        OfficialPowerPaintController = official_app.PowerPaintController
        dtype=torch.float16 if os.getenv("POWERPAINT_DTYPE","float16")=="float16" else torch.float32
        _controller=OfficialPowerPaintController(dtype,checkpoint_dir,os.getenv("POWERPAINT_LOCAL_FILES_ONLY","true").lower()=="true","ppt-v1")
    except Exception as exc: _model_error=f"{exc.__class__.__name__}: {exc}"
@app.on_event("startup")
def initialize_model()->None: _load_controller()
@app.get("/health")
def health()->dict[str,Any]:
    _load_controller()
    if _controller is None: raise HTTPException(503,{"status":"unavailable","gpu":"NVIDIA L4","error":_model_error})
    return {"status":"ok","gpu":torch.cuda.get_device_name(0),"model":"PowerPaint","checkpoint":"loaded"}
@app.post("/completion")
def completion(payload:CompletionRequest,authorization:str|None=Header(default=None))->dict[str,Any]:
    expected=os.getenv("WORKER_AUTH_TOKEN","").strip()
    if expected and authorization!=f"Bearer {expected}": raise HTTPException(401,"invalid worker authorization")
    _load_controller()
    if _controller is None: raise HTTPException(503,{"error_code":"POWERPAINT_UNAVAILABLE","message":_model_error})
    try:
        image=Image.open(io.BytesIO(_read_uri(payload.image_uri))).convert("RGB"); mask=Image.open(io.BytesIO(_read_uri(payload.mask_uri))).convert("RGB")
        if image.size!=mask.size: raise ValueError(f"image/mask size mismatch: {image.size} != {mask.size}")
        if not np.any(np.asarray(mask.convert("L"))>127): raise ValueError("completion mask is empty")
        started=time.perf_counter()
        outputs,_=_controller.predict({"image":image,"mask":mask},payload.prompt,0.75,int(os.getenv("POWERPAINT_STEPS","30")),7.5,int(os.getenv("POWERPAINT_SEED","20260907")),"blurry, low quality, distorted fish, duplicate fish","shape-guided",None,None)
        result=outputs[0] if isinstance(outputs,(list,tuple)) else outputs
        if not isinstance(result,Image.Image): raise TypeError("PowerPaint returned an invalid image")
        output=io.BytesIO(); result.convert("RGB").save(output,format="PNG")
        return {"result_uri":_write_result(payload.image_uri,output.getvalue()),"model_version":"PowerPaint","inference_time_ms":round((time.perf_counter()-started)*1000,2),"gpu_info":torch.cuda.get_device_name(0)}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(500,{"error_code":"POWERPAINT_INFERENCE_FAILED","message":f"{exc.__class__.__name__}: {exc}"}) from exc
