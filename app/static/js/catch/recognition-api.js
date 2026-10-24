(() => {
  "use strict";
  const core=window.YujianCatch=window.YujianCatch||{};
  core.infer = async (url,file,signal,requestId) => {
    const form=new FormData(); form.append("file",file,file.name||"catch-photo");
    const response=await fetch(url,{method:"POST",body:form,credentials:"same-origin",signal,headers:{"X-Request-ID":requestId}});
    const payload=await response.json().catch(()=>({}));
    if (!response.ok) {
      const detail=payload.detail||{};
      const error=new Error((typeof detail==="object"?detail.message:detail)||`识别请求失败（HTTP ${response.status}）`);
      error.code=detail.code||`HTTP_${response.status}`;error.requestId=detail.request_id;throw error;
    }
    const states=["RESULT_HIGH","RESULT_MEDIUM","RESULT_LOW","ERROR_NO_FISH","ERROR_IMAGE_QUALITY","TECHNICAL_FAILURE"];
    if (!states.includes(payload.result_state)) throw new Error("识别服务返回了无法识别的结果状态，请重试。");
    return payload;
  };
})();
