import {api, byId, jsonHeaders, message, startBase, speciesId} from "./common.js";

export async function start(initialId) {
  if (!await startBase(initialId) || !speciesId) return;
  try {
    const data = await api(`/api/v1/admin/fish/species/${encodeURIComponent(speciesId)}`);
    const row = data.species || {};
    byId("nameCn").value = row.name_cn || "";
    byId("category").value = row.category || "";
    byId("scientificName").value = row.scientific_name || "";
    byId("family").value = row.family || "";
    byId("genus").value = row.genus || "";
    byId("aliases").value = (row.alias || []).join("\n");
    byId("summary").value = row.summary || "";
    byId("basicStatus").textContent = `当前鱼种状态：${row.status || "UNKNOWN"}`;
    const notice = byId("basicExposureNotice");
    notice.textContent = row.status === "ACTIVE" ? "ACTIVE：保存基本信息后将即时公开到鱼种 API。" : "非 ACTIVE：基本信息会立即保存到 CMS，但当前不会由公共鱼种 API 返回。";
    notice.classList.toggle("active", row.status === "ACTIVE");
    window.fishBasicStatus = row.status || "UNKNOWN";
  } catch (error) { message(error.message, true); }

  byId("saveBasic").addEventListener("click", async () => {
    const aliases = byId("aliases").value.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
    if (window.fishBasicStatus === "ACTIVE" && !window.confirm("此鱼种当前为 ACTIVE。保存后，中文名、类别、学名、别名和简介会立即更新公共 API。继续保存吗？")) return;
    try {
      const saved = await api(`/api/v1/admin/fish/species/${encodeURIComponent(speciesId)}`, {
        method: "PATCH", headers: jsonHeaders,
        body: JSON.stringify({
          name_cn: byId("nameCn").value,
          category: byId("category").value,
          scientific_name: byId("scientificName").value || null,
          family: byId("family").value || null,
          genus: byId("genus").value || null,
          alias: aliases,
          summary: byId("summary").value,
        }),
      });
      byId("basicStatus").textContent = `已保存 · 当前鱼种状态：${saved.species?.status || saved.status || window.fishBasicStatus}`;
      message(window.fishBasicStatus === "ACTIVE" ? "鱼种基本信息已保存并即时公开；资产发布状态未更改。" : "鱼种基本信息已保存；资产发布状态未更改。请通过公共 API 核验后再认定是否公开。");
    } catch (error) { message(error.message, true); }
  });
}
