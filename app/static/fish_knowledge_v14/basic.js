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
    byId("aliases").value = JSON.stringify(row.alias || [], null, 2);
    byId("summary").value = row.summary || "";
    byId("basicStatus").textContent = `当前鱼种状态：${row.status || "UNKNOWN"}`;
  } catch (error) { message(error.message, true); }

  byId("saveBasic").addEventListener("click", async () => {
    let aliases;
    try {
      aliases = JSON.parse(byId("aliases").value || "[]");
      if (!Array.isArray(aliases)) throw new Error("别名必须是 JSON 数组");
    } catch (error) { return message(error.message || "别名格式错误", true); }
    try {
      await api(`/api/v1/admin/fish/species/${encodeURIComponent(speciesId)}`, {
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
      message("鱼种基本信息已保存；资产发布状态未更改");
    } catch (error) { message(error.message, true); }
  });
}
