import {api, byId, jsonHeaders, message, startBase, speciesId} from "./common.js";
const list = value => JSON.stringify(value || [], null, 2);
function parse(id) { const value = JSON.parse(byId(id).value || "[]"); if (!Array.isArray(value)) throw new Error(`${id} 必须是 JSON 数组`); return value; }
export async function start(initialId) {
  if (!await startBase(initialId) || !speciesId) return;
  try {
    const data = await api(`/api/v1/admin/fish/species/${encodeURIComponent(speciesId)}`);
    for (const [id,value] of [["bodyShape",data.profile?.body_shape || ""],["features",list(data.profile?.features)],["habitat",list(data.profile?.habitat)],["food",data.profile?.food || ""],["season",list(data.profile?.season)],["waterLayer",data.fishing?.water_layer || ""],["fishingSeason",list(data.fishing?.season)],["bait",list(data.fishing?.bait)],["method",list(data.fishing?.method)],["fishingSummary",data.fishing?.summary || ""]]) byId(id).value = value;
  } catch (error) { message(error.message, true); }
  byId("saveProfile").addEventListener("click", async () => {
    try {
      const payload = {body_shape:byId("bodyShape").value || null,features:parse("features"),habitat:parse("habitat"),food:byId("food").value || null,season:parse("season")};
      await api(`/api/v1/admin/fish/species/${encodeURIComponent(speciesId)}/profile`,{method:"PUT",headers:jsonHeaders,body:JSON.stringify(payload)}); message("基础资料已保存");
    } catch (error) { message(error.message, true); }
  });
  byId("saveFishing").addEventListener("click", async () => {
    try {
      const payload = {water_layer:byId("waterLayer").value || null,season:parse("fishingSeason"),bait:parse("bait"),method:parse("method"),summary:byId("fishingSummary").value};
      await api(`/api/v1/admin/fish/species/${encodeURIComponent(speciesId)}/fishing`,{method:"PUT",headers:jsonHeaders,body:JSON.stringify(payload)}); message("钓鱼知识已保存");
    } catch (error) { message(error.message, true); }
  });
}
