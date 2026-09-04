from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.fish_knowledge.cards import FishCard, normalize_card_type
from app.fish_knowledge.cover import FishSpeciesCover
from app.fish_knowledge.fishing import FishFishing
from app.fish_knowledge.profile import FishProfile
from app.fish_knowledge.similarity import FishSimilarity
from app.fish_knowledge.species import FishSpecies
from app.species_policy import ensure_target_species


FISH_KNOWLEDGE_SEED_VERSION = "MODEL_M1_v0.5_20_V1"
BAITIAO_SPECIES_ID = "sharpbelly"

INITIAL_BAITIAO_COVER = {
    "species_id": BAITIAO_SPECIES_ID,
    "image_url": "",
    "style": "ANIME_CARD",
    "title": "白条图鉴卡",
    "status": "DRAFT",
}

INITIAL_BAITIAO_CARDS = tuple(
    {
        "card_type": card_type,
        "title": title,
        "image_url": "",
        "description": "",
        "sort_order": order,
        "status": "DRAFT",
    }
    for order, (card_type, title) in enumerate(
        (
            ("HERO", "白条英雄卡"),
            ("IDENTIFICATION", "白条识别卡"),
            ("ECO", "白条生态卡"),
            ("GEAR", "白条装备卡"),
            ("SKILL", "白条作钓技术卡"),
        )
    )
)

# Media is intentionally not seeded. Gallery and video URLs must point to real,
# reviewed assets uploaded through Fish Knowledge Admin.
INITIAL_FISH_KNOWLEDGE = (
    {
        "species": {
            "id": "grass_carp",
            "name_cn": "草鱼",
            "alias": ["鲩鱼", "草鲩"],
            "scientific_name": "Ctenopharyngodon idella",
            "category": "淡水鱼",
            "family": "鲤科",
            "genus": "Ctenopharyngodon",
            "summary": "体型修长、鳞片整齐，是常见的大型淡水目标鱼。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "纺锤形，体长而略侧扁",
            "features": ["背部青灰、腹部较浅", "鳞片较大且边缘整齐", "头部宽平、口端位且无须", "尾鳍深叉"],
            "habitat": ["江河中下游", "湖泊", "水库及池塘"],
            "food": "成鱼以水草和岸边植物为主，也会摄食嫩叶等植物性食物。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "中下层，摄食时也会进入中上层",
            "season": ["晚春", "夏", "初秋"],
            "bait": ["嫩玉米", "发酵玉米", "青草或芦苇嫩叶", "商品草鱼饵"],
            "method": ["浮钓", "离底钓", "底钓", "打草窝或玉米窝"],
            "summary": "水温升高后摄食更积极，可先观察草线、浮叶和鱼星再选择水层。",
        },
    },
    {
        "species": {
            "id": "bighead_carp",
            "name_cn": "鳙鱼",
            "alias": ["花鲢", "胖头鱼", "包头鱼"],
            "scientific_name": "Hypophthalmichthys nobilis",
            "category": "淡水鱼",
            "family": "鲤科",
            "genus": "Hypophthalmichthys",
            "summary": "头部宽大、体侧带深色斑驳，是典型的滤食性大水面鱼。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "体高而侧扁，头部比例很大",
            "features": ["头部宽大", "眼位低于头部中线", "体侧灰黑并有不规则斑点", "腹棱主要位于腹鳍之后"],
            "habitat": ["大型湖泊", "水库开阔水域", "缓流江河"],
            "food": "以浮游动物为主，也摄食部分浮游植物和有机碎屑。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "中上层",
            "season": ["夏", "初秋"],
            "bait": ["酸香型雾化饵", "发酵谷物饵", "商品鲢鳙饵"],
            "method": ["浮钓", "定层搜索", "连续雾化诱鱼"],
            "summary": "适合在开阔水面按鱼情逐步搜索水层，饵料需保持稳定雾化。",
        },
    },
    {
        "species": {
            "id": "silver_carp",
            "name_cn": "白鲢",
            "alias": ["鲢鱼", "白鲢鱼"],
            "scientific_name": "Hypophthalmichthys molitrix",
            "category": "淡水鱼",
            "family": "鲤科",
            "genus": "Hypophthalmichthys",
            "summary": "体色银白、腹棱明显，常在大水面中上层成群活动。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "体高侧扁，整体较鳙鱼轻薄",
            "features": ["体侧银白且斑纹少", "眼位较低", "头部相对鳙鱼更小", "腹棱从胸腹部延伸至肛门"],
            "habitat": ["湖泊", "水库", "江河缓流开阔水域"],
            "food": "以浮游植物为主，并滤食细小浮游生物和有机颗粒。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "中上层",
            "season": ["夏", "初秋"],
            "bait": ["酸甜型雾化饵", "发酵谷物饵", "商品鲢鳙饵"],
            "method": ["浮钓", "定层搜索", "高频雾化诱鱼"],
            "summary": "白鲢对水层和雾化带变化敏感，可从浅到深逐层寻找鱼群。",
        },
    },
    {
        "species": {
            "id": "common_carp",
            "name_cn": "鲤鱼",
            "alias": ["鲤拐子", "黄鲤"],
            "scientific_name": "Cyprinus carpio",
            "category": "淡水鱼",
            "family": "鲤科",
            "genus": "Cyprinus",
            "summary": "口角有须、背鳍较长，是分布广泛的底层杂食性鱼。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "体厚、略呈纺锤形，腹部圆",
            "features": ["口角有两对须", "背鳍基部较长", "体色多为黄褐或青褐", "鳞片较大"],
            "habitat": ["江河缓流区", "湖泊", "水库", "池塘"],
            "food": "杂食性，摄食底栖动物、谷物、植物碎屑和水生昆虫。",
            "season": ["春", "夏", "秋", "冬"],
        },
        "fishing": {
            "water_layer": "底层和近底层",
            "season": ["春", "夏", "秋"],
            "bait": ["玉米", "小麦", "蚯蚓", "螺肉", "商品鲤鱼饵"],
            "method": ["底钓", "离底钓", "窝料守钓"],
            "summary": "常沿坎位、缓流和软硬底交界觅食，作钓时宜减少频繁惊扰。",
        },
    },
    {
        "species": {
            "id": "crucian_carp",
            "name_cn": "鲫鱼",
            "alias": ["野鲫", "土鲫"],
            "scientific_name": "Carassius auratus",
            "category": "淡水鱼",
            "family": "鲤科",
            "genus": "Carassius",
            "summary": "体型较高、口边无须，是中国淡水垂钓中最常见的目标鱼之一。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "体高而侧扁，背部隆起较明显",
            "features": ["口边无须", "背鳍基部较长", "体色多为银灰或黄褐", "尾柄相对较短"],
            "habitat": ["池塘", "湖泊", "河汊", "水库湾区"],
            "food": "杂食性，摄食藻类、底栖小动物、昆虫幼体和谷物碎屑。",
            "season": ["春", "夏", "秋", "冬"],
        },
        "fishing": {
            "water_layer": "底层和近底层",
            "season": ["全年", "春秋更活跃"],
            "bait": ["蚯蚓", "红虫", "谷物饵", "商品鲫鱼饵"],
            "method": ["台钓", "传统底钓", "轻量逗钓"],
            "summary": "可根据水温和鱼口调整饵团大小、味型与钓层，低温时宜用小钩细线。",
        },
    },
    {
        "species": {
            "id": "largemouth_bass",
            "name_cn": "加州鲈",
            "alias": ["大口黑鲈", "加州鲈鱼"],
            "scientific_name": "Micropterus salmoides",
            "category": "淡水鱼",
            "family": "太阳鱼科",
            "genus": "Micropterus",
            "summary": "口裂大、体侧有深色纵带，是常见的伏击型淡水掠食鱼。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "体侧扁，头大、尾柄有力",
            "features": ["上颌后端通常超过眼后缘", "体侧有深色纵向斑带", "背鳍前部具硬棘", "尾鳍浅凹"],
            "habitat": ["水库岸线", "池塘", "湖泊水草区", "倒木和乱石结构区"],
            "food": "肉食性，主要捕食小鱼、虾、蛙类和大型水生昆虫。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "中上层至近底层，随结构和饵鱼移动",
            "season": ["春", "夏", "秋"],
            "bait": ["软虫", "米诺", "铅头钩软鱼", "复合亮片"],
            "method": ["路亚搜索", "贴结构抛投", "慢拖和跳底"],
            "summary": "优先搜索水草边、倒木、石坎和明暗交界，并按活性调整回收速度。",
        },
    },
    {
        "species": {
            "id": "snakehead",
            "name_cn": "黑鱼",
            "alias": ["乌鳢", "财鱼", "生鱼"],
            "scientific_name": "Channa argus",
            "category": "淡水鱼",
            "family": "鳢科",
            "genus": "Channa",
            "summary": "身体细长、头似蛇形，常伏击水草区的小鱼和蛙类。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "圆筒状细长，头部宽扁",
            "features": ["头部似蛇形", "体侧有深色不规则斑块", "背鳍和臀鳍都很长", "口大且具细齿"],
            "habitat": ["水草密集的湖湾", "池塘", "沼泽和缓流河汊"],
            "food": "肉食性，捕食鱼、虾、蛙和其他小型水生动物。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "浅水和水草表层，也会潜伏近底",
            "season": ["晚春", "夏", "初秋"],
            "bait": ["雷蛙", "软蛙", "软鱼", "米诺"],
            "method": ["水面路亚", "草洞定点", "贴障碍搜索"],
            "summary": "重点寻找草洞、浮萍边和浅滩障碍，命中后需尽快控鱼离开重障碍。",
        },
    },
    {
        "species": {
            "id": "yellow_catfish",
            "name_cn": "黄骨鱼",
            "alias": ["黄颡鱼", "黄辣丁", "昂刺鱼"],
            "scientific_name": "Tachysurus fulvidraco",
            "category": "淡水鱼",
            "family": "鲿科",
            "genus": "Tachysurus",
            "summary": "体表无鳞、带多对触须，背鳍和胸鳍硬刺明显。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "前部较粗、后部侧扁，头宽而平",
            "features": ["体表无鳞", "口边有四对须", "体色黄褐并带深色斑纹", "背鳍和胸鳍有硬刺"],
            "habitat": ["江河缓流底部", "湖泊", "水库湾区", "泥沙或碎石底"],
            "food": "底栖杂食偏肉食，摄食水生昆虫、软体动物、虾和小鱼。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "底层，夜间可能上移觅食",
            "season": ["晚春", "夏", "秋"],
            "bait": ["蚯蚓", "虾肉", "鱼肉条", "动物性腥饵"],
            "method": ["底钓", "夜钓", "贴石缝和缓流边作钓"],
            "summary": "傍晚和夜间通常更活跃，摘钩时需避开背鳍、胸鳍硬刺。",
        },
    },
    {
        "species": {
            "id": "black_carp",
            "name_cn": "青鱼",
            "alias": ["螺蛳青", "青鲩"],
            "scientific_name": "Mylopharyngodon piceus",
            "category": "淡水鱼",
            "family": "鲤科",
            "genus": "Mylopharyngodon",
            "summary": "体色深青、头部较尖，是偏好螺蚌的大型底层鱼。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "长筒形，体厚而略侧扁",
            "features": ["背部和体侧颜色较深", "头部较草鱼尖长", "鳞片大且色深", "各鳍多呈灰黑色"],
            "habitat": ["江河深水区", "大型湖泊", "水库深坎和硬底区"],
            "food": "以螺、蚌等软体动物为主，也摄食甲壳动物。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "底层",
            "season": ["夏", "秋"],
            "bait": ["螺蛳", "蚌肉", "玉米", "颗粒饵"],
            "method": ["重装备底钓", "螺蛳窝守钓", "深坎和硬底定点"],
            "summary": "多在深水硬底和螺蚌丰富区域活动，线组与抄鱼装备需匹配大体型。",
        },
    },
    {
        "species": {
            "id": "tilapia",
            "name_cn": "罗非鱼",
            "alias": ["非洲鲫", "福寿鱼"],
            "scientific_name": "Oreochromis spp.",
            "category": "淡水鱼",
            "family": "慈鲷科",
            "genus": "Oreochromis",
            "summary": "这是多种常见罗非鱼及养殖杂交品系的通用类别，喜温且适应性强。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "体高侧扁，背部隆起",
            "features": ["背鳍基部长且前部具硬棘", "尾鳍常见纵向条纹", "体侧可有深浅相间横纹", "口端位"],
            "habitat": ["南方河湖", "水库浅湾", "养殖池塘", "温暖缓流水域"],
            "food": "杂食性，摄食藻类、有机碎屑、水生昆虫和人工饲料。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "中下层和近岸浅水",
            "season": ["晚春", "夏", "初秋"],
            "bait": ["腥味商品饵", "蚯蚓", "虾肉", "颗粒饲料"],
            "method": ["浮钓近底", "底钓", "近岸结构区搜索"],
            "summary": "温暖水域鱼口更积极，可根据密度和抢食程度调整饵料雾化与钓层。",
        },
    },
    {
        "species": {
            "id": "mandarin_fish",
            "name_cn": "鳜鱼",
            "alias": ["桂鱼", "桂花鱼"],
            "scientific_name": "Siniperca chuatsi",
            "category": "淡水鱼",
            "family": "鳜科",
            "genus": "Siniperca",
            "summary": "口大、斑纹伪装明显，常在石缝和结构边伏击小鱼。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "体高侧扁，头大、背部隆起",
            "features": ["口裂大", "体侧有褐色不规则斑块", "背鳍前部硬棘明显", "鳃盖后缘有棘"],
            "habitat": ["清洁江河", "水库乱石区", "湖泊结构区", "桥墩和陡坎"],
            "food": "肉食性，主要捕食活鱼和虾。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "中下层，紧贴石缝和障碍物",
            "season": ["春", "夏", "秋"],
            "bait": ["软鱼", "米诺", "铅头钩", "合规使用的活虾或小鱼"],
            "method": ["贴底跳跃", "贴结构慢搜", "桥墩和石坎定点"],
            "summary": "重点搜索乱石、陡坎和桥墩阴影，需防止中鱼后钻入障碍。",
        },
    },
    {
        "species": {
            "id": "topmouth_culter",
            "name_cn": "翘嘴鲌",
            "alias": ["翘嘴", "翘嘴红鲌"],
            "scientific_name": "Culter alburnus",
            "category": "淡水鱼",
            "family": "鲤科",
            "genus": "Culter",
            "summary": "口向上翘、身体修长，是大水面常见的中上层掠食鱼。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "修长侧扁，背部较平、腹缘锐利",
            "features": ["下颌突出、口明显向上", "体侧银白", "胸鳍较长", "尾鳍深叉"],
            "habitat": ["大型江河", "湖泊开阔区", "水库主湖面和湾口"],
            "food": "肉食性，主要追食小鱼和虾。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "中上层，追饵时接近水面",
            "season": ["晚春", "夏", "秋"],
            "bait": ["米诺", "亮片", "铅笔", "软鱼", "虾"],
            "method": ["远投路亚", "搜索炸水和鸟群", "浮钓活饵"],
            "summary": "先寻找饵鱼群、湾口和风浪交界，再根据鱼群深度选择拟饵和泳层。",
        },
    },
    {
        "species": {
            "id": "chinese_catfish",
            "name_cn": "鲶鱼",
            "alias": ["土鲶", "本地鲶"],
            "scientific_name": "Silurus asotus",
            "category": "淡水鱼",
            "family": "鲶科",
            "genus": "Silurus",
            "summary": "头宽、体表无鳞、臀鳍很长，是夜间活跃的底栖掠食鱼。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "前部粗圆、后部侧扁，头部宽平",
            "features": ["体表无鳞", "口宽且有长须", "臀鳍基部很长", "尾部逐渐侧扁"],
            "habitat": ["江河深潭", "湖泊底部", "水库库湾", "洞穴和障碍附近"],
            "food": "肉食偏杂食，摄食鱼、虾、蛙、水生昆虫和动物性食物。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "底层",
            "season": ["晚春", "夏", "秋"],
            "bait": ["蚯蚓", "虾", "鱼肉条", "动物性腥饵"],
            "method": ["底钓", "夜钓", "深潭和障碍边守钓"],
            "summary": "夜间和浑水时往往更积极，可重点搜索深潭、回水和障碍边缘。",
        },
    },
    {
        "species": {
            "id": "sharpbelly",
            "name_cn": "白条",
            "alias": ["餐条", "白鲦"],
            "scientific_name": "Hemiculter leucisculus",
            "category": "淡水鱼",
            "family": "鲤科",
            "genus": "Hemiculter",
            "summary": "体侧银亮、腹缘较锐，常在近岸中上层快速成群抢食。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "小型修长侧扁，腹缘呈明显棱线",
            "features": ["体侧银白发亮", "口略向上", "腹部有锐利腹棱", "尾鳍深叉"],
            "habitat": ["江河近岸", "湖泊", "水库湾区", "池塘表层"],
            "food": "杂食性，摄食浮游生物、水生昆虫、藻类和细小有机颗粒。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "中上层",
            "season": ["春", "夏", "秋"],
            "bait": ["小粒腥香饵", "麸类饵", "小昆虫"],
            "method": ["短竿浮钓", "小钩快频率", "拉饵钓浮"],
            "summary": "鱼群进窝后适合小钩轻漂和稳定节奏，避免过大饵团拖慢入口。",
        },
    },
    {
        "species": {
            "id": "yellowcheek",
            "name_cn": "鳡鱼",
            "alias": ["黄颊鱼", "水老虎"],
            "scientific_name": "Elopichthys bambusa",
            "category": "淡水鱼",
            "family": "鲤科",
            "genus": "Elopichthys",
            "summary": "身体修长、口裂宽大，是大型江河和水库中的高速掠食鱼。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "长梭形，头尖、尾柄强壮",
            "features": ["口裂宽大", "头部和鳃盖可见黄铜色", "体侧银灰", "尾鳍深叉"],
            "habitat": ["大型江河主流", "湖泊开阔水域", "大型水库"],
            "food": "强肉食性，主要追捕中小型鱼类。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "中上层和开阔水域",
            "season": ["夏", "秋"],
            "bait": ["大型米诺", "亮片", "VIB", "软鱼"],
            "method": ["远投搜索", "追踪饵鱼群", "急流和湾口作钓"],
            "summary": "需在大水面寻找饵鱼和追逐迹象，并使用匹配大体型高速鱼的装备。",
        },
    },
    {
        "species": {
            "id": "blunt_snout_bream",
            "name_cn": "鳊鱼",
            "alias": ["武昌鱼", "团头鲂"],
            "scientific_name": "Megalobrama amblycephala",
            "category": "淡水鱼",
            "family": "鲤科",
            "genus": "Megalobrama",
            "summary": "体高而扁、头部较小，常在水草和缓流区摄食植物性饵料。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "体高侧扁，轮廓近菱形",
            "features": ["头部较小、吻部钝", "体侧银灰", "背部隆起明显", "臀鳍基部较长"],
            "habitat": ["湖泊水草区", "水库湾区", "江河缓流", "池塘"],
            "food": "以水生植物和藻类为主，也摄食谷物和小型底栖动物。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "中下层和近底层",
            "season": ["晚春", "夏", "秋"],
            "bait": ["嫩玉米", "小麦", "谷物饵", "藻香或植物性商品饵"],
            "method": ["浮钓离底", "底钓", "水草边定点"],
            "summary": "可在水草外沿、缓坡和风浪送食区寻找鱼群，并灵活调整离底高度。",
        },
    },
    {
        "species": {
            "id": "mud_carp",
            "name_cn": "鲮鱼",
            "alias": ["土鲮", "鲮公"],
            "scientific_name": "Cirrhinus molitorella",
            "category": "淡水鱼",
            "family": "鲤科",
            "genus": "Cirrhinus",
            "summary": "口位偏下、身体结实，是华南暖水河流中常见的底层鱼。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "长纺锤形，腹部圆、尾柄有力",
            "features": ["吻部圆钝、口下位", "背部青灰、体侧银灰", "鳞片整齐", "各鳍常带灰黄色"],
            "habitat": ["华南江河", "水库", "湖泊", "有流水的砂石底水域"],
            "food": "刮食藻类和附着生物，也摄食有机碎屑及谷物饵料。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "底层和近底层",
            "season": ["晚春", "夏", "秋"],
            "bait": ["花生麸", "米糠谷物饵", "薯类饵", "藻香商品饵"],
            "method": ["底钓", "小钩细线", "流水缓冲区定点"],
            "summary": "常在暖水、缓流和有附着藻类的硬底活动，中鱼后冲刺力较强。",
        },
    },
    {
        "species": {
            "id": "chinese_hooksnout_carp",
            "name_cn": "马口鱼",
            "alias": ["马口"],
            "scientific_name": "Opsariichthys bidens",
            "category": "淡水鱼",
            "family": "鲤科",
            "genus": "Opsariichthys",
            "summary": "口裂宽、游速快，常见于清澈溪流和有流速的浅滩。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "小型修长侧扁，尾柄有力",
            "features": ["口裂宽大", "体侧有蓝绿光泽", "雄鱼繁殖期色彩更鲜明", "尾鳍深叉"],
            "habitat": ["清澈溪流", "河流浅滩", "石底急缓流交界"],
            "food": "偏肉食，摄食水生和落水昆虫、小虾及小鱼。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "中上层和浅滩流层",
            "season": ["春", "夏", "秋"],
            "bait": ["小亮片", "小米诺", "微型软虫", "昆虫饵"],
            "method": ["溪流路亚", "轻量浮钓", "急缓流交界搜索"],
            "summary": "适合轻量装备顺流或斜向搜索，重点观察石后缓流和浅滩落差。",
        },
    },
    {
        "species": {
            "id": "yellowfin_culter",
            "name_cn": "黄尾鲴",
            "alias": ["黄尾", "黄片"],
            "scientific_name": "Xenocypris davidi",
            "category": "淡水鱼",
            "family": "鲤科",
            "genus": "Xenocypris",
            "summary": "口位偏下、尾鳍带黄色，常在湖库近底层刮食藻类。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "体长侧扁，腹部较圆",
            "features": ["口小且偏下位", "体侧银灰", "尾鳍和下部鳍常带黄色", "腹部无明显长棱"],
            "habitat": ["湖泊", "水库", "江河缓流和硬底区"],
            "food": "以附着藻类、有机碎屑和小型底栖生物为主。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "底层和近底层",
            "season": ["春", "夏", "秋"],
            "bait": ["藻香饵", "麦麸谷物饵", "发酵粮食饵"],
            "method": ["小钩细线底钓", "离底轻钓", "硬底和缓坡定点"],
            "summary": "口型较小，宜使用轻细线组和状态柔软的小饵，重点找有附着藻类的硬底。",
        },
    },
    {
        "species": {
            "id": "redfin_culter",
            "name_cn": "红眼鳟",
            "alias": ["赤眼鳟", "红眼鱼"],
            "scientific_name": "Squaliobarbus curriculus",
            "category": "淡水鱼",
            "family": "鲤科",
            "genus": "Squaliobarbus",
            "summary": "眼上缘常带红色、身体银灰，是活跃于江河湖库的杂食性鱼。",
            "status": "ACTIVE",
        },
        "profile": {
            "body_shape": "长筒形，后部稍侧扁",
            "features": ["眼上缘有明显红色", "体侧银灰、背部较深", "鳞片较大", "尾鳍深叉"],
            "habitat": ["江河中下游", "湖泊", "水库开阔区和湾口"],
            "food": "杂食性，摄食藻类、水生植物、昆虫、虾和小鱼。",
            "season": ["春", "夏", "秋"],
        },
        "fishing": {
            "water_layer": "中层至近底层",
            "season": ["春", "夏", "秋"],
            "bait": ["玉米", "小麦", "蚯蚓", "昆虫饵", "谷物商品饵"],
            "method": ["浮钓", "离底钓", "缓流和湾口搜索"],
            "summary": "活动水层会随流速和食物变化，可在湾口、缓流边和开阔浅滩寻找鱼群。",
        },
    },
)


INITIAL_SIMILARITY = (
    ("grass_carp", "black_carp", "草鱼体色偏青黄、头部较宽圆；青鱼体色更深，头部更尖长，常见于近底层。"),
    ("black_carp", "grass_carp", "青鱼整体色深、吻部较尖，常以螺蚌为食；草鱼体色较浅、头宽，偏植物食性。"),
    ("bighead_carp", "silver_carp", "鳙鱼头更大且体侧有深色斑驳；白鲢体色银白，腹棱更长、更明显。"),
    ("silver_carp", "bighead_carp", "白鲢体色更均匀银白、腹棱较长；鳙鱼头部更大，体侧常有灰黑斑点。"),
    ("common_carp", "crucian_carp", "鲤鱼口角有须、身体更厚且背鳍较长；鲫鱼无须，体型通常更高更扁。"),
    ("crucian_carp", "common_carp", "鲫鱼口边无须、体高侧扁；鲤鱼口角有两对须，身体更厚长。"),
    ("snakehead", "largemouth_bass", "黑鱼身体细长、背鳍和臀鳍都很长；加州鲈体高侧扁，体侧常有深色纵带。"),
    ("largemouth_bass", "snakehead", "加州鲈体高侧扁、上颌后端常超过眼后缘；黑鱼呈长筒形并有蛇形头部。"),
    ("yellow_catfish", "chinese_catfish", "黄骨鱼多为黄褐斑纹并有明显背鳍、胸鳍硬刺；鲶鱼体色较暗，臀鳍基部很长。"),
    ("chinese_catfish", "yellow_catfish", "鲶鱼头宽、尾部侧扁且臀鳍很长；黄骨鱼体型较短，黄褐色并有明显硬刺。"),
    ("topmouth_culter", "sharpbelly", "翘嘴鲌口明显上翘、体型可较大且以鱼虾为食；白条体型小，腹棱更锐利。"),
    ("sharpbelly", "topmouth_culter", "白条体型小、腹棱明显；翘嘴鲌下颌突出更强，身体更长且常见大型个体。"),
    ("yellowfin_culter", "mud_carp", "黄尾鲴尾鳍常带黄色、体型较侧扁；鲮鱼身体更结实，吻部圆钝且多见华南流水。"),
    ("redfin_culter", "grass_carp", "红眼鳟眼上缘有红色、体侧银灰；草鱼眼部无红色标记，体色偏青黄。"),
)


def seed_initial_fish_knowledge(db: Session) -> dict[str, int | str]:
    """Idempotently seed missing V1 knowledge without overwriting Admin edits."""

    ensure_target_species(db)
    existing_species_ids = set(db.scalars(select(FishSpecies.id)).all())
    new_species_ids: set[str] = set()
    species_created = 0
    profile_created = 0
    fishing_created = 0
    cover_created = 0
    cards_created = 0

    for item in INITIAL_FISH_KNOWLEDGE:
        species_values = item["species"]
        species_id = str(species_values["id"])
        if db.get(FishSpecies, species_id) is None:
            db.add(FishSpecies(**species_values))
            new_species_ids.add(species_id)
            species_created += 1
    if species_created:
        db.flush()

    for item in INITIAL_FISH_KNOWLEDGE:
        species_id = str(item["species"]["id"])
        if db.get(FishProfile, species_id) is None:
            db.add(FishProfile(species_id=species_id, **item["profile"]))
            profile_created += 1
        if db.get(FishFishing, species_id) is None:
            db.add(FishFishing(species_id=species_id, **item["fishing"]))
            fishing_created += 1

    similarity_created = 0
    # Seed similarity only when a source species is first introduced. This keeps
    # subsequent Admin deletions authoritative instead of recreating them on startup.
    for species_id, similar_species_id, difference in INITIAL_SIMILARITY:
        if species_id not in new_species_ids:
            continue
        exists = db.scalar(
            select(FishSimilarity.id).where(
                FishSimilarity.species_id == species_id,
                FishSimilarity.similar_species_id == similar_species_id,
            )
        )
        if exists is None:
            db.add(
                FishSimilarity(
                    species_id=species_id,
                    similar_species_id=similar_species_id,
                    difference=difference,
                )
            )
            similarity_created += 1

    # 白条 uses the existing stable model/catalog key ``sharpbelly``.  The
    # public API additionally accepts ``baitiao`` as a product-facing alias.
    # These are intentionally empty DRAFT slots: media must be supplied by an
    # operator after the real artwork has been reviewed.
    baitiao = db.get(FishSpecies, BAITIAO_SPECIES_ID)
    if baitiao is not None:
        cover_exists = db.scalar(
            select(FishSpeciesCover.id).where(FishSpeciesCover.species_id == BAITIAO_SPECIES_ID)
        )
        if cover_exists is None:
            db.add(FishSpeciesCover(**INITIAL_BAITIAO_COVER))
            cover_created += 1

        existing_card_types = {
            normalize_card_type(card.card_type)
            for card in db.scalars(
                select(FishCard).where(FishCard.species_id == BAITIAO_SPECIES_ID)
            ).all()
        }
        for card_values in INITIAL_BAITIAO_CARDS:
            if card_values["card_type"] in existing_card_types:
                continue
            db.add(FishCard(species_id=BAITIAO_SPECIES_ID, **card_values))
            existing_card_types.add(card_values["card_type"])
            cards_created += 1

    if species_created or profile_created or fishing_created or similarity_created or cover_created or cards_created:
        db.commit()
    return {
        "version": FISH_KNOWLEDGE_SEED_VERSION,
        "species_created": species_created,
        "profile_created": profile_created,
        "fishing_created": fishing_created,
        "similarity_created": similarity_created,
        "cover_created": cover_created,
        "cards_created": cards_created,
        "existing_species": len(existing_species_ids),
    }
