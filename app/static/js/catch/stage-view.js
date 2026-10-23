(() => {
  "use strict";
  const core=window.YujianCatch=window.YujianCatch||{};
  core.renderJourney = (root,state,visual="") => {
    root.dataset.stage=state.phase;
    const steps=[...root.querySelectorAll("[data-journey-step]")];
    const hasPhoto=Boolean(state.photo), hasResult=Boolean(state.inference);
    const high=state.inference?.result_state==="RESULT_HIGH";
    const rows=[
      !hasPhoto?{state:"pending",label:"等待照片"}:{state:state.busy?"active":"done",label:state.busy?"正在读取照片":"照片已就绪"},
      state.busy?{state:"active",label:"光影随识别时间推进"}:hasResult?{state:"done",label:"真实推理已返回"}:{state:"pending",label:"等待真实识别"},
      state.photo?.mask?.trust_state==="TRUSTED_MASK"?{state:state.busy?"active":"done",label:"可信鱼体几何已验证"}:state.photo?.mask?.trust_state==="REVIEW_ONLY"?{state:"unavailable",label:"SAM 候选待评审 · 不绘制轮廓"}:state.bbox?{state:"unavailable",label:"仅有检测框 · 不绘制轮廓"}:{state:"unavailable",label:"等待真实鱼体几何"},
      state.transitionState==="active"||visual==="handoff"?{state:"active",label:"同源照片正在接续"}:state.transitionState==="done"?{state:"done",label:"照片已进入结果 Hero"}:{state:"pending",label:"等待照片接续"},
      state.revealState==="active"?{state:"active",label:"真实中文名称揭晓中"}:state.revealState==="done"||(!high&&hasResult)?{state:"done",label:high?"真实结果已揭晓":"真实识别结果已返回"}:state.phase==="ERROR"?{state:"error",label:"识别未完成"}:{state:"pending",label:"等待真实分类结果"},
    ];
    steps.forEach((node,index)=>{
      const row=rows[index];node.dataset.state=row.state;
      node.querySelector("[data-journey-status]").textContent=row.label;
      if(row.state==="active")node.setAttribute("aria-current","step");else node.removeAttribute("aria-current");
      const image=node.querySelector("[data-journey-photo]");
      const placeholder=node.querySelector(".journey-placeholder");
      if(image&&state.photo?.url){image.src=state.photo.url;image.hidden=false;if(placeholder)placeholder.hidden=true;}
      else if(image){image.removeAttribute("src");image.hidden=true;if(placeholder)placeholder.hidden=false;}
    });
  };
})();
