(() => {
  "use strict";
  const core=window.YujianCatch=window.YujianCatch||{};
  core.renderJourney = (root,state,visual="") => {
    root.dataset.stage=state.phase;
    const steps=[...root.querySelectorAll("[data-journey-step]")];
    let active=-1;
    if (state.photo) active=0;
    if (state.phase==="PROCESSING") active=1;
    if (visual==="handoff") active=3;
    if (state.phase==="RESULT") active=4;
    steps.forEach((node,index)=>{
      // No SWOS or trusted fish mask is implemented in B1. Never claim these effects ran.
      const unavailable=index===1||index===2;
      const failed=state.phase==="ERROR" && index===4;
      const status=failed?"error":unavailable?"unavailable":(index===active?"active":(index<active?"done":"pending"));
      node.dataset.state=status;
      node.querySelector("[data-journey-status]").textContent=failed?"识别未完成":unavailable?"视觉效果待接入":status==="active"?"当前阶段":status==="done"?"已完成":"等待开始";
      if (index===active) node.setAttribute("aria-current","step"); else node.removeAttribute("aria-current");
    });
  };
})();
