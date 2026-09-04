import express, { Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import cors from 'cors';

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Navigation bar items from original Python app (app/unified_nav.py)
const NAV_ITEMS = [
  { href: '/', label: '总览', mode: 'root' },
  { href: '/batches', label: '数据批次', mode: 'exact' },
  { href: '/batches/upload', label: '数据导入', mode: 'prefix' },
  { href: '/review/bulk', label: '快速审核', mode: 'prefix' },
  { href: '/review', label: '单张审核', mode: 'exact' },
  { href: '/species', label: '鱼种管理', mode: 'prefix' },
  { href: '/fish-knowledge', label: '鱼鉴内容', mode: 'prefix' },
  { href: '/feedback', label: '用户反馈', mode: 'prefix' },
  { href: '/datasets', label: '数据集', mode: 'prefix' },
  { href: '/crop-datasets', label: 'Crop Dataset', mode: 'prefix' },
  { href: '/training', label: '模型训练', mode: 'prefix' },
  { href: '/inference', label: '模型实测', mode: 'prefix' },
  { href: '/intelligence', label: '模型智能分析', mode: 'prefix' },
  { href: '/crop-qa', label: 'Crop QA', mode: 'prefix' },
];

const NAV_HREFS = new Set(NAV_ITEMS.map((item) => item.href));

function isNavActive(currentPath: string, href: string, mode: string): boolean {
  if (mode === 'root') return currentPath === '/' || currentPath === '';
  if (mode === 'exact') return currentPath === href;
  return currentPath.startsWith(href);
}

function buildUnifiedNavHeader(currentPath: string, extraControls: string = ''): string {
  const linksHtml = NAV_ITEMS.map((item) => {
    const active = isNavActive(currentPath, item.href, item.mode);
    return `<a href="${item.href}" class="${active ? 'active' : ''}" ${active ? 'aria-current="page"' : ''}>${item.label}</a>`;
  }).join('');

  return `
<style id="yujian-unified-nav-style">
  header.app-nav{background:#fff;border-bottom:1px solid #e5e7eb;padding:10px 20px;display:flex;gap:6px;align-items:center;position:sticky;top:0;z-index:50;overflow-x:auto;white-space:nowrap;scrollbar-width:thin}
  header.app-nav a{display:inline-flex;align-items:center;min-height:34px;padding:0 10px;border-radius:8px;text-decoration:none;color:#4b5563;font-weight:650;font-size:14px;flex:0 0 auto}
  header.app-nav a:hover{background:#f3f4f6;color:#111827}
  header.app-nav a.active{background:#111827;color:#fff}
  header.app-nav .filters{flex:0 0 auto;margin-left:auto}
  @media(max-width:760px){header.app-nav{padding:8px 12px;gap:4px}header.app-nav a{padding:0 9px;font-size:13px}}
</style>
<header class="app-nav" aria-label="主导航">
  ${linksHtml}
  ${extraControls}
</header>`;
}

// Mobile UX snippet
const mobileUxPath = path.join(process.cwd(), 'views', '_mobile_ux.html');
let mobileUxHtml = '';
try {
  mobileUxHtml = fs.readFileSync(mobileUxPath, 'utf-8');
} catch (e) {
  console.warn('Could not read _mobile_ux.html:', e);
}

function renderTemplate(viewName: string, req: Request, res: Response) {
  const filePath = path.join(process.cwd(), 'views', viewName);
  if (!fs.existsSync(filePath)) {
    return res.status(404).send(`Template ${viewName} not found`);
  }

  let html = fs.readFileSync(filePath, 'utf-8');

  // Replace Jinja2 mobile UX include
  html = html.replace(/\{%\s*include\s*["']_mobile_ux\.html["']\s*%\}/g, mobileUxHtml);

  // Replace header with unified navigation
  const currentPath = req.path;
  const headerMatch = html.match(/<header\b[^>]*>(.*?)<\/header>/is);
  if (headerMatch) {
    const originalBody = headerMatch[1];
    // Keep extra controls (like filters) that are not primary nav links
    const extraControls = originalBody
      .replace(/<a\b[^>]*\bhref=["']([^"']+)["'][^>]*>.*?<\/a>/gis, (match, href) => {
        return NAV_HREFS.has(href) ? '' : match;
      })
      .trim();

    const newHeader = buildUnifiedNavHeader(currentPath, extraControls);
    html = html.replace(/<header\b[^>]*>.*?<\/header>/is, newHeader);
  }

  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(html);
}

// ==========================================
// IN-MEMORY DATA STORE
// ==========================================

interface SpeciesItem {
  species_key: string;
  catalog_order: number;
  common_name_zh: string;
  common_name_en?: string;
  scientific_name?: string;
  status: 'active' | 'candidate' | 'retired';
  is_other: boolean;
  notes?: string;
  created_at: string;
  updated_at: string;
}

const INITIAL_SPECIES: SpeciesItem[] = [
  { species_key: 'grass_carp', catalog_order: 0, common_name_zh: '草鱼', common_name_en: 'Grass carp', scientific_name: 'Ctenopharyngodon idella', status: 'active', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'bighead_carp', catalog_order: 1, common_name_zh: '鳙鱼', common_name_en: 'Bighead carp', scientific_name: 'Hypophthalmichthys nobilis', status: 'active', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'silver_carp', catalog_order: 2, common_name_zh: '白鲢', common_name_en: 'Silver carp', scientific_name: 'Hypophthalmichthys molitrix', status: 'active', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'common_carp', catalog_order: 3, common_name_zh: '鲤鱼', common_name_en: 'Common carp', scientific_name: 'Cyprinus carpio', status: 'active', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'crucian_carp', catalog_order: 4, common_name_zh: '鲫鱼', common_name_en: 'Crucian carp', scientific_name: 'Carassius carassius', status: 'active', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'largemouth_bass', catalog_order: 5, common_name_zh: '加州鲈', common_name_en: 'Largemouth bass', scientific_name: 'Micropterus salmoides', status: 'active', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'snakehead', catalog_order: 6, common_name_zh: '黑鱼', common_name_en: 'Snakehead', scientific_name: 'Channa argus', status: 'active', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'yellow_catfish', catalog_order: 7, common_name_zh: '黄骨鱼', common_name_en: 'Yellow catfish', scientific_name: 'Pelteobagrus fulvidraco', status: 'active', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'black_carp', catalog_order: 8, common_name_zh: '青鱼', common_name_en: 'Black carp', scientific_name: 'Mylopharyngodon piceus', status: 'active', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'tilapia', catalog_order: 9, common_name_zh: '罗非鱼', common_name_en: 'Tilapia', scientific_name: 'Oreochromis niloticus', status: 'active', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'mandarin_fish', catalog_order: 10, common_name_zh: '鳜鱼', common_name_en: 'Mandarin fish', scientific_name: 'Siniperca chuatsi', status: 'active', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'topmouth_culter', catalog_order: 11, common_name_zh: '翘嘴鲌', common_name_en: 'Topmouth culter', scientific_name: 'Culter alburnus', status: 'active', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'blunt_snout_bream', catalog_order: 12, common_name_zh: '鳊鱼 / 武昌鱼', common_name_en: 'Blunt snout bream', scientific_name: 'Megalobrama amblycephala', status: 'active', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'chinese_catfish', catalog_order: 13, common_name_zh: '鲶鱼', common_name_en: 'Chinese catfish', scientific_name: 'Silurus asotus', status: 'active', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'mud_carp', catalog_order: 14, common_name_zh: '鲮鱼', common_name_en: 'Mud carp', scientific_name: 'Cirrhinus molitorella', status: 'active', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'sharpbelly', catalog_order: 15, common_name_zh: '白条', common_name_en: 'Sharpbelly', scientific_name: 'Hemiculter leucisculus', status: 'active', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'chinese_hooksnout_carp', catalog_order: 16, common_name_zh: '马口鱼', common_name_en: 'Chinese hooksnout carp', scientific_name: 'Opsariichthys bidens', status: 'candidate', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'yellowcheek', catalog_order: 17, common_name_zh: '鳡鱼', common_name_en: 'Yellowcheek', scientific_name: 'Elopichthys bambusa', status: 'candidate', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'yellowfin_culter', catalog_order: 18, common_name_zh: '黄尾鲴', common_name_en: 'Yellowfin culter', scientific_name: 'Xenocypris davidi', status: 'candidate', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'redfin_culter', catalog_order: 19, common_name_zh: '红眼鳟', common_name_en: 'Redfin culter', scientific_name: 'Squaliobarbus curriculus', status: 'candidate', is_other: false, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
  { species_key: 'other_freshwater_fish', catalog_order: 20, common_name_zh: '其他淡水鱼', common_name_en: 'Other freshwater fish', status: 'active', is_other: true, created_at: '2026-08-01T00:00:00Z', updated_at: '2026-08-01T00:00:00Z' },
];

const speciesStore = new Map<string, SpeciesItem>(INITIAL_SPECIES.map((s) => [s.species_key, { ...s }]));

// Batches
interface BatchItem {
  batch_id: string;
  source: string;
  created_at: string;
  image_count: number;
  raw_image_count: number;
  status: string;
  manifest_uri: string;
  raw_uri: string;
  review: {
    approved: number;
    pending: number;
    rejected: number;
    needs_review: number;
    hard_case: number;
  };
}

const batchesStore = new Map<string, BatchItem>([
  [
    'batch-202608-pilot',
    {
      batch_id: 'batch-202608-pilot',
      source: 'pilot',
      created_at: '2026-08-20T08:00:00Z',
      image_count: 1420,
      raw_image_count: 1420,
      status: 'REGISTERED',
      manifest_uri: 'gs://yujian-ai-factory/batches/batch-202608-pilot/manifest.jsonl',
      raw_uri: 'gs://yujian-ai-factory/batches/batch-202608-pilot/raw/',
      review: { approved: 1180, pending: 180, rejected: 60, needs_review: 0, hard_case: 0 },
    },
  ],
  [
    'batch-202608-doubao',
    {
      batch_id: 'batch-202608-doubao',
      source: 'doubao',
      created_at: '2026-08-25T10:30:00Z',
      image_count: 850,
      raw_image_count: 850,
      status: 'REGISTERED',
      manifest_uri: 'gs://yujian-ai-factory/batches/batch-202608-doubao/manifest.jsonl',
      raw_uri: 'gs://yujian-ai-factory/batches/batch-202608-doubao/raw/',
      review: { approved: 690, pending: 110, rejected: 50, needs_review: 0, hard_case: 0 },
    },
  ],
  [
    'batch-202609-feedback',
    {
      batch_id: 'batch-202609-feedback',
      source: 'app_feedback',
      created_at: '2026-09-02T14:15:00Z',
      image_count: 320,
      raw_image_count: 320,
      status: 'REGISTERED',
      manifest_uri: 'gs://yujian-ai-factory/batches/batch-202609-feedback/manifest.jsonl',
      raw_uri: 'gs://yujian-ai-factory/batches/batch-202609-feedback/raw/',
      review: { approved: 210, pending: 80, rejected: 30, needs_review: 0, hard_case: 0 },
    },
  ],
]);

// Incoming batches
const incomingStore = [
  {
    folder: 'incoming-20260904-field-collect',
    incoming_prefix: 'incoming/20260904-field-collect/',
    canonical_batch_id: 'batch-20260904-field',
    source: 'workbuddy',
    image_count: 240,
    manifest_count: 240,
    promoted: false,
    audit: {
      batch_id: 'batch-20260904-field',
      status_counts: { CANDIDATE: 228, NEEDS_REVIEW: 9, AUTO_REJECT: 3 },
    },
  },
];

// Presence & Dedupe stats per batch
const presenceStore = new Map<string, any>([
  [
    'batch-202608-pilot',
    {
      batch_id: 'batch-202608-pilot',
      scanned: true,
      single_fish: 1350,
      multi_fish: 45,
      no_fish: 15,
      uncertain: 10,
      remaining: 0,
      filterable_no_fish: 15,
    },
  ],
  [
    'batch-202608-doubao',
    {
      batch_id: 'batch-202608-doubao',
      scanned: true,
      single_fish: 810,
      multi_fish: 28,
      no_fish: 8,
      uncertain: 4,
      remaining: 0,
      filterable_no_fish: 8,
    },
  ],
  [
    'batch-202609-feedback',
    {
      batch_id: 'batch-202609-feedback',
      scanned: true,
      single_fish: 302,
      multi_fish: 12,
      no_fish: 4,
      uncertain: 2,
      remaining: 0,
      filterable_no_fish: 4,
    },
  ],
]);

const dedupeStore = new Map<string, any>([
  [
    'batch-202608-pilot',
    {
      batch_id: 'batch-202608-pilot',
      scanned: true,
      groups: 24,
      duplicate_images: 48,
      exact_duplicates: 18,
      near_duplicates: 30,
      remaining: 0,
      filterable_duplicate_images: 48,
    },
  ],
  [
    'batch-202608-doubao',
    {
      batch_id: 'batch-202608-doubao',
      scanned: true,
      groups: 12,
      duplicate_images: 22,
      exact_duplicates: 8,
      near_duplicates: 14,
      remaining: 0,
      filterable_duplicate_images: 22,
    },
  ],
  [
    'batch-202609-feedback',
    {
      batch_id: 'batch-202609-feedback',
      scanned: true,
      groups: 5,
      duplicate_images: 9,
      exact_duplicates: 3,
      near_duplicates: 6,
      remaining: 0,
      filterable_duplicate_images: 9,
    },
  ],
]);

// Datasets
const datasetsStore = [
  {
    dataset_version: 'v0.2.0-cum',
    parent_version: 'v0.1.0-cum',
    created_at: '2026-08-29T16:00:00Z',
    manifest_uri: 'gs://yujian-ai-models/datasets/v0.2.0-cum/manifest.jsonl',
    class_map_uri: 'gs://yujian-ai-models/datasets/v0.2.0-cum/class_map.json',
    train_count: 1456,
    val_count: 312,
    test_count: 312,
    species_count: 16,
    selection_mode: 'cumulative_approved',
    source_cutoff_at: '2026-08-29T12:00:00Z',
    git_commit: 'e4f5g6h',
    status: 'FROZEN',
    pipeline_type: 'WHOLE_IMAGE_V1',
    metadata: {
      split: { train: 0.7, val: 0.15, test: 0.15 },
      seed: 20260826,
    },
  },
  {
    dataset_version: 'v0.1.0-cum',
    parent_version: null,
    created_at: '2026-08-22T10:00:00Z',
    manifest_uri: 'gs://yujian-ai-models/datasets/v0.1.0-cum/manifest.jsonl',
    class_map_uri: 'gs://yujian-ai-models/datasets/v0.1.0-cum/class_map.json',
    train_count: 980,
    val_count: 210,
    test_count: 210,
    species_count: 12,
    selection_mode: 'cumulative_approved',
    source_cutoff_at: '2026-08-22T08:00:00Z',
    git_commit: 'a1b2c3d',
    status: 'FROZEN',
    pipeline_type: 'WHOLE_IMAGE_V1',
    metadata: {
      split: { train: 0.7, val: 0.15, test: 0.15 },
      seed: 20260826,
    },
  },
];

// Feedback events
interface FeedbackItem {
  id: string;
  source_event_id: string;
  feedback_type: string;
  source: string;
  image_gcs_uri?: string;
  model_version?: string;
  predicted_species?: string;
  confidence?: number;
  corrected_species?: string;
  user_note?: string;
  status: string;
  created_at: string;
}

const feedbackStore: FeedbackItem[] = [
  {
    id: 'fb-001',
    source_event_id: 'evt-20260901-01',
    feedback_type: 'confirmed',
    source: 'app',
    image_gcs_uri: 'gs://yujian-app/catches/user123/20260901_grasscarp.jpg',
    model_version: 'yujian-classifier-v0.2',
    predicted_species: '草鱼',
    confidence: 0.96,
    corrected_species: '草鱼',
    user_note: '太准了，河边钓的大草鱼！',
    status: 'PENDING',
    created_at: '2026-09-01T14:32:00Z',
  },
  {
    id: 'fb-002',
    source_event_id: 'evt-20260901-02',
    feedback_type: 'corrected',
    source: 'app',
    image_gcs_uri: 'gs://yujian-app/catches/user456/20260901_fish.jpg',
    model_version: 'yujian-classifier-v0.2',
    predicted_species: '鲫鱼',
    confidence: 0.72,
    corrected_species: '鲤鱼',
    user_note: '嘴角有须子，是小鲤鱼苗不是鲫鱼',
    status: 'PENDING',
    created_at: '2026-09-01T15:10:00Z',
  },
  {
    id: 'fb-003',
    source_event_id: 'evt-20260902-01',
    feedback_type: 'new_species_candidate',
    source: 'app',
    image_gcs_uri: 'gs://yujian-app/catches/user789/20260902_special.jpg',
    model_version: 'yujian-classifier-v0.2',
    predicted_species: '其他淡水鱼',
    confidence: 0.65,
    corrected_species: '马口鱼',
    user_note: '山溪里钓的马口，花纹特别漂亮',
    status: 'PENDING',
    created_at: '2026-09-02T11:45:00Z',
  },
];

// Sample Image assets for review queue
interface ImageAssetItem {
  id: number;
  batch_id: string;
  image_id: string;
  file_name: string;
  source_url?: string;
  source_platform: string;
  claimed_species: string;
  truth_species?: string | null;
  truth_status: string;
  review_status: string;
  scene?: string;
  lighting?: string;
  quality?: string;
  group_id?: string;
  notes?: string;
  ai_suggestion?: string;
  ai_confidence?: number;
  truth_prefill?: string;
  truth_prefill_source?: string;
  label_conflict?: boolean;
  label_conflict_message?: string | null;
  reviewed_by?: string;
  reviewed_at?: string;
}

const sampleSpeciesList = ['草鱼', '鲫鱼', '鲤鱼', '黑鱼', '加州鲈', '黄骨鱼', '白条', '鳜鱼', '鳙鱼', '青鱼'];
const reviewStore: ImageAssetItem[] = [];

let imageIdCounter = 1;
for (const batch of ['batch-202608-pilot', 'batch-202608-doubao', 'batch-202609-feedback']) {
  for (let i = 1; i <= 25; i++) {
    const sp = sampleSpeciesList[(i + (batch === 'batch-202608-doubao' ? 3 : batch === 'batch-202609-feedback' ? 6 : 0)) % sampleSpeciesList.length];
    const isApproved = i <= 14;
    const isRejected = i === 15 || i === 16;
    const status = isApproved ? 'approved' : isRejected ? 'rejected' : 'pending';
    const numStr = String(i).padStart(4, '0');
    const idStr = `${batch}_img_${numStr}`;

    reviewStore.push({
      id: imageIdCounter++,
      batch_id: batch,
      image_id: idStr,
      file_name: `${sp}_${numStr}.jpg`,
      source_url: `https://example.com/fish/${idStr}.jpg`,
      source_platform: batch.includes('doubao') ? 'doubao' : batch.includes('feedback') ? 'app_feedback' : 'pilot',
      claimed_species: sp,
      truth_species: isApproved ? sp : null,
      truth_status: isApproved ? 'LIKELY_CORRECT' : 'UNCERTAIN',
      review_status: status,
      scene: i % 2 === 0 ? 'river' : 'hand_hold',
      lighting: 'daylight',
      quality: 'high',
      notes: status === 'pending' && i % 4 === 0 ? '请专家复核鳍条特征' : '',
      ai_suggestion: sp,
      ai_confidence: 0.92,
      truth_prefill: sp,
      truth_prefill_source: 'ai_high_confidence',
      label_conflict: false,
      reviewed_by: isApproved ? 'reviewer-01' : undefined,
      reviewed_at: isApproved ? '2026-08-25T12:00:00Z' : undefined,
    });
  }
}

// Load Fish Knowledge Seed data
let fishKnowledgeData: any = { species: [] };
try {
  const seedPath = path.join(process.cwd(), 'data', 'fish_knowledge', 'fish_seed.json');
  if (fs.existsSync(seedPath)) {
    fishKnowledgeData = JSON.parse(fs.readFileSync(seedPath, 'utf-8'));
  }
} catch (e) {
  console.warn('Could not read fish_seed.json:', e);
}

// ==========================================
// HTML PAGE ROUTES
// ==========================================

app.get('/', (req, res) => renderTemplate('overview.html', req, res));
app.get('/batches', (req, res) => renderTemplate('batches.html', req, res));
app.get('/batches/upload', (req, res) => renderTemplate('batch_upload.html', req, res));
app.get('/batch-upload', (req, res) => renderTemplate('batch_upload.html', req, res));
app.get('/review', (req, res) => renderTemplate('review.html', req, res));
app.get('/review/bulk', (req, res) => renderTemplate('bulk_review.html', req, res));
app.get('/species', (req, res) => renderTemplate('species.html', req, res));
app.get('/datasets', (req, res) => renderTemplate('datasets.html', req, res));
app.get('/crop-datasets', (req, res) => renderTemplate('crop_datasets.html', req, res));
app.get('/crop-qa', (req, res) => renderTemplate('crop_qa.html', req, res));
app.get('/crop-review', (req, res) => renderTemplate('crop_review.html', req, res));
app.get('/feedback', (req, res) => renderTemplate('feedback.html', req, res));
app.get('/fish-knowledge', (req, res) => renderTemplate('fish_knowledge.html', req, res));
app.get('/training', (req, res) => renderTemplate('training.html', req, res));
app.get('/inference', (req, res) => renderTemplate('inference.html', req, res));
app.get('/inspect', (req, res) => renderTemplate('inspect.html', req, res));
app.get('/intelligence', (req, res) => renderTemplate('intelligence.html', req, res));

// ==========================================
// HEALTH CHECKS
// ==========================================

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '0.1.0' });
});

app.get('/health/deploy', (req, res) => {
  res.json({
    status: 'ok',
    version: '0.1.0',
    git_commit: process.env.APP_GIT_COMMIT || 'node-migrated-v1',
    revision: 'cloud-run-v1',
    service: 'yujian-model-factory',
    feedback_ingest_path: '/api/feedback/ingest',
    inference_upload_path: '/api/v1/inference/upload',
    user_auth_path: '/api/v1/auth/login',
    user_catches_path: '/api/v1/catches',
    feedback_ingest_key_configured: true,
  });
});

app.get('/health/detector', (req, res) => {
  res.json({
    status: 'ok',
    model_version: 'yolox_fish_nano_v1.0',
    onnx_sha256: 'mock-onnx-sha256-42981a',
    onnx_bytes: 3892400,
    input_size: [416, 416],
  });
});

// ==========================================
// API ROUTES
// ==========================================

// Helper for flywheel summary
function getFlywheelSummary() {
  let approvedCount = 0;
  for (const b of batchesStore.values()) {
    approvedCount += b.review.approved;
  }
  const activeSpeciesCount = Array.from(speciesStore.values()).filter((s) => s.status === 'active').length;
  const candidateSpeciesCount = Array.from(speciesStore.values()).filter((s) => s.status === 'candidate').length;

  const approvedSpecies = [
    { species: '草鱼', count: 320 },
    { species: '鲫鱼', count: 290 },
    { species: '鲤鱼', count: 260 },
    { species: '黑鱼', count: 210 },
    { species: '加州鲈', count: 195 },
    { species: '黄骨鱼', count: 180 },
    { species: '白条', count: 165 },
    { species: '鳜鱼', count: 150 },
    { species: '鳙鱼', count: 130 },
    { species: '白鲢', count: 110 },
    { species: '青鱼', count: 60 },
  ];

  return {
    approved_master_pool: approvedCount || 2080,
    active_species: activeSpeciesCount,
    candidate_species: candidateSpeciesCount,
    new_approved_since_latest_dataset: 184,
    new_feedback: feedbackStore.length,
    latest_dataset: datasetsStore[0]?.dataset_version || 'v0.2.0-cum',
    approved_species: approvedSpecies,
  };
}

// 1. Overview API
app.get('/api/overview', (req, res) => {
  const totalImages = Array.from(batchesStore.values()).reduce((acc, b) => acc + b.image_count, 0);
  const totalApproved = Array.from(batchesStore.values()).reduce((acc, b) => acc + b.review.approved, 0);
  const totalPending = Array.from(batchesStore.values()).reduce((acc, b) => acc + b.review.pending, 0);
  const totalRejected = Array.from(batchesStore.values()).reduce((acc, b) => acc + b.review.rejected, 0);

  res.json({
    total_images: totalImages,
    batch_count: batchesStore.size,
    dataset_count: datasetsStore.length,
    review: {
      approved: totalApproved,
      pending: totalPending,
      rejected: totalRejected,
      needs_review: 0,
      hard_case: 0,
    },
    species: [
      { species: '草鱼', count: 320 },
      { species: '鲫鱼', count: 290 },
      { species: '鲤鱼', count: 260 },
      { species: '黑鱼', count: 210 },
      { species: '加州鲈', count: 195 },
      { species: '黄骨鱼', count: 180 },
      { species: '白条', count: 165 },
      { species: '鳜鱼', count: 150 },
      { species: '鳙鱼', count: 130 },
      { species: '白鲢', count: 110 },
      { species: '青鱼', count: 60 },
    ],
    unconfirmed_truth_count: 24,
    flywheel: getFlywheelSummary(),
  });
});

app.get('/api/flywheel/summary', (req, res) => {
  res.json(getFlywheelSummary());
});

// 2. Batches API
app.get('/api/batches', (req, res) => {
  res.json(Array.from(batchesStore.values()));
});

app.get('/api/incoming', (req, res) => {
  res.json(incomingStore);
});

app.post('/api/batches/audit', (req, res) => {
  const { batch_id, incoming_prefix, source } = req.body;
  res.json({
    batch_id: batch_id || 'batch-audited',
    incoming_prefix,
    source,
    image_count: 240,
    manifest_count: 240,
    status_counts: { CANDIDATE: 228, NEEDS_REVIEW: 9, AUTO_REJECT: 3 },
    checked_at: new Date().toISOString(),
  });
});

app.post('/api/batches/promote', (req, res) => {
  const { batch_id, incoming_prefix, source } = req.body;
  const existing = batchesStore.get(batch_id);
  if (existing) {
    return res.json({ already_exists: true, batch_id, status: existing.status });
  }

  const newBatch: BatchItem = {
    batch_id,
    source: source || 'other',
    created_at: new Date().toISOString(),
    image_count: 240,
    raw_image_count: 240,
    status: 'INGESTED',
    manifest_uri: `gs://yujian-ai-factory/batches/${batch_id}/manifest.jsonl`,
    raw_uri: `gs://yujian-ai-factory/batches/${batch_id}/raw/`,
    review: { approved: 0, pending: 240, rejected: 0, needs_review: 0, hard_case: 0 },
  };
  batchesStore.set(batch_id, newBatch);

  res.json({
    already_exists: false,
    batch_id,
    source,
    image_count: 240,
    status: 'INGESTED',
  });
});

app.post('/api/batches/sync', (req, res) => {
  const { batch_id } = req.body;
  const batch = batchesStore.get(batch_id);
  if (batch) {
    batch.status = 'REGISTERED';
  }
  res.json({ batch_id, synced: true, status: 'REGISTERED' });
});

// Presence & Dedupe scan API
app.get('/api/presence/batches', (req, res) => {
  res.json(Array.from(presenceStore.values()));
});

app.post('/api/presence/scan', (req, res) => {
  const { batch_id } = req.body;
  const stat = presenceStore.get(batch_id) || {
    batch_id,
    scanned: true,
    single_fish: 220,
    multi_fish: 12,
    no_fish: 5,
    uncertain: 3,
    remaining: 0,
    filterable_no_fish: 5,
  };
  presenceStore.set(batch_id, stat);
  res.json({ ...stat, processed: 40, remaining: 0 });
});

app.post('/api/presence/reject-no-fish', (req, res) => {
  const { batch_id } = req.body;
  const batch = batchesStore.get(batch_id);
  if (batch) {
    batch.review.rejected += 5;
    batch.review.pending = Math.max(0, batch.review.pending - 5);
  }
  res.json({ batch_id, rejected: 5 });
});

app.get('/api/dedupe/batches', (req, res) => {
  res.json(Array.from(dedupeStore.values()));
});

app.post('/api/dedupe/scan', (req, res) => {
  const { batch_id } = req.body;
  const stat = dedupeStore.get(batch_id) || {
    batch_id,
    scanned: true,
    groups: 8,
    duplicate_images: 14,
    exact_duplicates: 4,
    near_duplicates: 10,
    remaining: 0,
    filterable_duplicate_images: 14,
  };
  dedupeStore.set(batch_id, stat);
  res.json({ ...stat, processed: 100, remaining: 0 });
});

app.post('/api/dedupe/reject-duplicates', (req, res) => {
  const { batch_id } = req.body;
  const batch = batchesStore.get(batch_id);
  if (batch) {
    batch.review.rejected += 14;
    batch.review.pending = Math.max(0, batch.review.pending - 14);
  }
  res.json({ batch_id, rejected: 14 });
});

// 3. Species API
app.get('/api/species', (req, res) => {
  const status = req.query.status as string | undefined;
  let list = Array.from(speciesStore.values()).sort((a, b) => a.catalog_order - b.catalog_order);
  if (status) {
    list = list.filter((s) => s.status === status);
  }
  res.json(list);
});

app.post('/api/species', (req, res) => {
  const { common_name_zh, species_key, common_name_en, scientific_name, notes } = req.body;
  if (!common_name_zh) {
    return res.status(400).json({ detail: 'common_name_zh is required' });
  }

  const key = species_key || `species_${Date.now().toString(36)}`;
  const order = speciesStore.size;

  const newSpecies: SpeciesItem = {
    species_key: key,
    catalog_order: order,
    common_name_zh,
    common_name_en: common_name_en || null,
    scientific_name: scientific_name || null,
    status: 'candidate',
    is_other: false,
    notes: notes || null,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };

  speciesStore.set(key, newSpecies);
  res.json(newSpecies);
});

app.patch('/api/species/:species_key/status', (req, res) => {
  const { species_key } = req.params;
  const { status } = req.body;

  const item = speciesStore.get(species_key);
  if (!item) {
    return res.status(404).json({ detail: 'Species not found' });
  }

  item.status = status;
  item.updated_at = new Date().toISOString();
  speciesStore.set(species_key, item);

  res.json(item);
});

// 4. Review API
app.get('/api/review', (req, res) => {
  const { status, batch_id, species, q } = req.query;
  const limit = parseInt(req.query.limit as string) || 50;
  const offset = parseInt(req.query.offset as string) || 0;

  let filtered = reviewStore;

  if (status && status !== 'all') {
    filtered = filtered.filter((img) => img.review_status === status);
  }
  if (batch_id) {
    filtered = filtered.filter((img) => img.batch_id === batch_id);
  }
  if (species) {
    filtered = filtered.filter((img) => img.truth_species === species || img.claimed_species === species);
  }
  if (q) {
    const qLower = String(q).toLowerCase();
    filtered = filtered.filter(
      (img) =>
        img.image_id.toLowerCase().includes(qLower) ||
        img.file_name.toLowerCase().includes(qLower) ||
        (img.claimed_species && img.claimed_species.toLowerCase().includes(qLower))
    );
  }

  const paged = filtered.slice(offset, offset + limit).map((img) => ({
    ...img,
    collected_label: img.claimed_species,
    media_url: `/media/${img.batch_id}/${img.image_id}`,
  }));

  res.json(paged);
});

app.get('/api/review/stats', (req, res) => {
  const { batch_id, species, q } = req.query;
  let filtered = reviewStore;

  if (batch_id) filtered = filtered.filter((img) => img.batch_id === batch_id);
  if (species) filtered = filtered.filter((img) => img.truth_species === species || img.claimed_species === species);
  if (q) {
    const qLower = String(q).toLowerCase();
    filtered = filtered.filter((img) => img.image_id.toLowerCase().includes(qLower));
  }

  const counts: Record<string, number> = {
    approved: filtered.filter((i) => i.review_status === 'approved').length,
    pending: filtered.filter((i) => i.review_status === 'pending').length,
    rejected: filtered.filter((i) => i.review_status === 'rejected').length,
    needs_review: filtered.filter((i) => i.review_status === 'needs_review').length,
    hard_case: filtered.filter((i) => i.review_status === 'hard_case').length,
  };

  res.json({
    filtered: filtered.length,
    status: counts,
  });
});

app.patch('/api/review/:batch_id/:image_id', (req, res) => {
  const { batch_id, image_id } = req.params;
  const { review_status, truth_species, truth_status, notes, reviewer } = req.body;

  const img = reviewStore.find((i) => i.batch_id === batch_id && i.image_id === image_id);
  if (!img) {
    return res.status(404).json({ detail: 'image not found' });
  }

  if (review_status !== undefined) img.review_status = review_status;
  if (truth_species !== undefined) img.truth_species = truth_species;
  if (truth_status !== undefined) img.truth_status = truth_status;
  if (notes !== undefined) img.notes = notes;
  img.reviewed_by = reviewer || 'web-review';
  img.reviewed_at = new Date().toISOString();

  // Update batch review statistics
  const batch = batchesStore.get(batch_id);
  if (batch) {
    const images = reviewStore.filter((i) => i.batch_id === batch_id);
    batch.review.approved = images.filter((i) => i.review_status === 'approved').length;
    batch.review.pending = images.filter((i) => i.review_status === 'pending').length;
    batch.review.rejected = images.filter((i) => i.review_status === 'rejected').length;
  }

  res.json({
    ...img,
    collected_label: img.claimed_species,
    media_url: `/media/${img.batch_id}/${img.image_id}`,
  });
});

// Media endpoint (produces crisp SVG placeholder for local demo)
app.get('/media/:batch_id/:image_id', (req, res) => {
  const { batch_id, image_id } = req.params;
  const img = reviewStore.find((i) => i.batch_id === batch_id && i.image_id === image_id);
  const fishName = img ? img.claimed_species : '鱼类样本';

  const svg = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 480" width="100%" height="100%">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#1e293b" />
    </linearGradient>
    <linearGradient id="fishGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38bdf8" />
      <stop offset="50%" stop-color="#818cf8" />
      <stop offset="100%" stop-color="#c084fc" />
    </linearGradient>
  </defs>
  <rect width="640" height="480" fill="url(#bg)" rx="16" />
  <g transform="translate(320, 220)">
    <!-- Stylized fish icon -->
    <path d="M-120,0 C-70,-80 70,-80 120,0 C70,80 -70,80 -120,0 Z" fill="url(#fishGrad)" opacity="0.85" />
    <path d="M120,0 L180,-50 L160,0 L180,50 Z" fill="url(#fishGrad)" opacity="0.85" />
    <circle cx="-60" cy="-15" r="10" fill="#ffffff" />
    <circle cx="-58" cy="-15" r="5" fill="#0f172a" />
  </g>
  <rect x="20" y="20" width="600" height="440" fill="none" stroke="#334155" stroke-width="2" rx="12" stroke-dasharray="6,6" />
  <text x="320" y="340" text-anchor="middle" fill="#f8fafc" font-size="24" font-weight="bold" font-family="sans-serif">${fishName}</text>
  <text x="320" y="375" text-anchor="middle" fill="#94a3b8" font-size="14" font-family="sans-serif">${image_id}</text>
  <text x="320" y="405" text-anchor="middle" fill="#64748b" font-size="12" font-family="sans-serif">批次: ${batch_id}</text>
</svg>`;

  res.setHeader('Content-Type', 'image/svg+xml');
  res.setHeader('Cache-Control', 'public, max-age=3600');
  res.send(svg);
});

// 5. Datasets API
app.get('/api/datasets', (req, res) => {
  res.json(datasetsStore);
});

app.get('/api/datasets/summary', (req, res) => {
  res.json(getFlywheelSummary());
});

app.get('/api/datasets/:dataset_version/crop-readiness', (req, res) => {
  res.json({
    dataset_version: req.params.dataset_version,
    total_images: 1456,
    crop_ready_images: 1420,
    readiness_percentage: 97.5,
  });
});

app.post('/api/datasets/freeze', (req, res) => {
  const { dataset_version, parent_version, seed, train, val } = req.body;
  const newVer = {
    dataset_version: dataset_version || `v0.${datasetsStore.length + 1}.0-cum`,
    parent_version: parent_version || datasetsStore[0]?.dataset_version || null,
    created_at: new Date().toISOString(),
    manifest_uri: `gs://yujian-ai-models/datasets/${dataset_version}/manifest.jsonl`,
    class_map_uri: `gs://yujian-ai-models/datasets/${dataset_version}/class_map.json`,
    train_count: 1680,
    val_count: 360,
    test_count: 360,
    species_count: 18,
    selection_mode: 'cumulative_approved',
    source_cutoff_at: new Date().toISOString(),
    git_commit: 'node-migrated',
    status: 'FROZEN',
    pipeline_type: 'WHOLE_IMAGE_V1',
    metadata: {
      split: { train: train || 0.7, val: val || 0.15, test: 0.15 },
      seed: seed || 20260826,
    },
  };
  datasetsStore.unshift(newVer);
  res.json(newVer);
});

// 6. Feedback API
app.get('/api/feedback', (req, res) => {
  res.json(feedbackStore);
});

app.post('/api/feedback', (req, res) => {
  const newItem: FeedbackItem = {
    id: `fb-${Date.now().toString(36)}`,
    status: 'PENDING',
    created_at: new Date().toISOString(),
    ...req.body,
  };
  feedbackStore.unshift(newItem);
  res.json(newItem);
});

app.post('/api/feedback/materialize', (req, res) => {
  const { batch_id, limit } = req.body;
  const count = Math.min(feedbackStore.length, limit || 500);

  const batch: BatchItem = {
    batch_id: batch_id || `batch-fb-${Date.now().toString(36)}`,
    source: 'app_feedback',
    created_at: new Date().toISOString(),
    image_count: count,
    raw_image_count: count,
    status: 'REGISTERED',
    manifest_uri: `gs://yujian-ai-factory/batches/${batch_id}/manifest.jsonl`,
    raw_uri: `gs://yujian-ai-factory/batches/${batch_id}/raw/`,
    review: { approved: 0, pending: count, rejected: 0, needs_review: 0, hard_case: 0 },
  };
  batchesStore.set(batch.batch_id, batch);

  res.json({
    batch_id: batch.batch_id,
    materialized_count: count,
    status: 'REGISTERED',
  });
});

// 7. Fish Knowledge CMS API (v1 and v1.1 / v1.2)
app.get('/api/v1/fish-knowledge/species', (req, res) => {
  res.json(fishKnowledgeData.species || []);
});

app.get('/api/v1/fish-knowledge/species/:id', (req, res) => {
  const sp = (fishKnowledgeData.species || []).find((s: any) => s.id === req.params.id);
  if (!sp) return res.status(404).json({ detail: 'Species not found' });
  res.json(sp);
});

app.get('/api/v1/fish-knowledge/admin/species', (req, res) => {
  res.json(fishKnowledgeData.species || []);
});

app.get('/api/v1/fish-knowledge/admin/species/:id', (req, res) => {
  const sp = (fishKnowledgeData.species || []).find((s: any) => s.id === req.params.id);
  if (!sp) return res.status(404).json({ detail: 'Species not found' });
  res.json(sp);
});

app.post('/api/v1/fish-knowledge/admin/species', (req, res) => {
  const newSp = req.body;
  if (!newSp.id) newSp.id = `sp_${Date.now().toString(36)}`;
  fishKnowledgeData.species.push(newSp);
  res.json(newSp);
});

app.put('/api/v1/fish-knowledge/admin/species/:id', (req, res) => {
  const idx = (fishKnowledgeData.species || []).findIndex((s: any) => s.id === req.params.id);
  if (idx === -1) return res.status(404).json({ detail: 'Species not found' });
  fishKnowledgeData.species[idx] = { ...fishKnowledgeData.species[idx], ...req.body };
  res.json(fishKnowledgeData.species[idx]);
});

app.delete('/api/v1/fish-knowledge/admin/species/:id', (req, res) => {
  fishKnowledgeData.species = (fishKnowledgeData.species || []).filter((s: any) => s.id !== req.params.id);
  res.json({ success: true });
});

// Fallback for sub-resources of fish knowledge
app.post('/api/v1/fish-knowledge/admin/species/:id/cover', (req, res) => {
  const sp = (fishKnowledgeData.species || []).find((s: any) => s.id === req.params.id);
  if (sp) sp.cover = req.body;
  res.json(sp ? sp.cover : { success: true });
});

app.post('/api/v1/fish-knowledge/admin/species/:id/publish', (req, res) => {
  const sp = (fishKnowledgeData.species || []).find((s: any) => s.id === req.params.id);
  if (sp) sp.status = 'ACTIVE';
  res.json({ success: true, status: 'ACTIVE' });
});

// 8. Auth API & User Catches MVP API
app.post('/api/v1/auth/login', (req, res) => {
  res.json({
    token: 'mock-jwt-token-yujian-factory',
    user: {
      id: 'usr-admin-01',
      username: req.body.username || 'admin',
      nickname: 'Admin User',
    },
  });
});

app.get('/api/v1/catches', (req, res) => {
  res.json([]);
});

// 9. Intelligence & Automation & Training stubs
app.get('/api/intelligence/summary', (req, res) => {
  res.json({
    model_version: 'yujian-v0.2.0',
    accuracy: 0.942,
    top3_accuracy: 0.985,
    confusion_pairs: [
      { predicted: '鳙鱼', actual: '白鲢', error_count: 14 },
      { predicted: '青鱼', actual: '草鱼', error_count: 8 },
    ],
    hard_cases: 12,
  });
});

app.get('/api/crop-datasets', (req, res) => {
  res.json([]);
});

app.get('/api/training/runs', (req, res) => {
  res.json([
    {
      run_id: 'run-20260830-classifier-yolox',
      model_type: 'classifier',
      base_model: 'mobilenet_v3_large',
      dataset_version: 'v0.2.0-cum',
      status: 'COMPLETED',
      accuracy: 0.942,
      created_at: '2026-08-30T09:00:00Z',
    },
  ]);
});

// Inspect Images API
app.get('/api/inspect/images', (req, res) => {
  const { batch_id, review_status, presence, species, q, new_since_latest } = req.query;
  const limit = parseInt(req.query.limit as string) || 30;
  const offset = parseInt(req.query.offset as string) || 0;

  let filtered = reviewStore;

  if (batch_id) filtered = filtered.filter((i) => i.batch_id === batch_id);
  if (review_status && review_status !== 'all') filtered = filtered.filter((i) => i.review_status === review_status);
  if (species) filtered = filtered.filter((i) => i.truth_species === species || i.claimed_species === species);
  if (q) {
    const qL = String(q).toLowerCase();
    filtered = filtered.filter((i) => i.image_id.toLowerCase().includes(qL) || i.file_name.toLowerCase().includes(qL));
  }

  const items = filtered.slice(offset, offset + limit).map((img) => ({
    ...img,
    media_url: `/media/${img.batch_id}/${img.image_id}`,
    presence_label: 'SINGLE_FISH',
    presence_source: 'detector',
    presence_model: 'yolox_nano',
    presence_confidence: 0.94,
    presence_override: null,
    is_duplicate: false,
    effective_status: img.review_status.toUpperCase(),
  }));

  res.json({
    total: filtered.length,
    offset,
    limit,
    items,
  });
});

app.patch('/api/inspect/presence/:batch_id/:image_id', (req, res) => {
  res.json({ success: true, updated: req.params.image_id });
});

// Crop Review & Dataset Crop Review APIs
app.get('/api/crop-review/:batch/items', (req, res) => {
  const { batch } = req.params;
  const status = req.query.status as string;
  const items = reviewStore
    .filter((i) => i.batch_id === batch)
    .slice(0, 30)
    .map((img) => ({
      image_id: img.image_id,
      batch_id: img.batch_id,
      media_url: `/media/${img.batch_id}/${img.image_id}`,
      preview_url: `/media/${img.batch_id}/${img.image_id}`,
      species_name: img.truth_species || img.claimed_species,
      suggested_species: img.ai_suggestion || img.claimed_species,
      status: 'REVIEW_REQUIRED',
      crop_status: 'GENERATED',
      detector_version: 'yolox_nano_v1.0',
      candidate_bbox: [0.15, 0.2, 0.7, 0.55],
      accepted_bbox: [0.15, 0.2, 0.7, 0.55],
    }));

  res.json({ total: items.length, items });
});

app.get('/api/crop-review/:batch/summary', (req, res) => {
  res.json({
    batch_id: req.params.batch,
    total_images: 25,
    candidate_bbox_count: 25,
    accepted_bbox_count: 18,
    review_required: 7,
    bbox_required: 7,
  });
});

app.patch('/api/crop-review/:batch/:id', (req, res) => {
  res.json({ success: true, image_id: req.params.id, ...req.body });
});

app.get('/api/dataset-crop-review/:dataset/items', (req, res) => {
  const items = reviewStore.slice(0, 25).map((img) => ({
    image_id: img.image_id,
    source_type: 'FROZEN_DATASET',
    source_dataset_version: req.params.dataset,
    split: 'train',
    class_index: 0,
    media_url: `/media/${img.batch_id}/${img.image_id}`,
    preview_url: `/media/${img.batch_id}/${img.image_id}`,
    species_name: img.truth_species || img.claimed_species,
    status: 'REVIEW_REQUIRED',
    crop_status: 'GENERATED',
    detector_version: 'yolox_nano_v1.0',
    candidate_bbox: [0.12, 0.18, 0.75, 0.6],
    accepted_bbox: [0.12, 0.18, 0.75, 0.6],
  }));
  res.json({ total: items.length, items });
});

app.get('/api/dataset-crop-review/:dataset/summary', (req, res) => {
  res.json({
    dataset_version: req.params.dataset,
    total_images: 25,
    candidate_bbox_count: 25,
    accepted_bbox_count: 20,
    bbox_required: 5,
  });
});

app.patch('/api/dataset-crop-review/:dataset/:id', (req, res) => {
  res.json({ success: true, image_id: req.params.id, ...req.body });
});

// Crop QA
app.get('/api/crop-qa', (req, res) => {
  const items = reviewStore.slice(0, 20).map((img) => ({
    image_id: img.image_id,
    batch_id: img.batch_id,
    media_url: `/media/${img.batch_id}/${img.image_id}`,
    crop_url: `/media/${img.batch_id}/${img.image_id}`,
    species: img.truth_species || img.claimed_species,
    status: 'ACCEPTED',
    bbox: [0.15, 0.2, 0.7, 0.55],
    reviewed_at: '2026-08-30T10:00:00Z',
  }));
  res.json({ total: items.length, items });
});

// Crop Datasets API
app.get('/api/crop-datasets/sources', (req, res) => {
  res.json({
    items: Array.from(batchesStore.values()).map((b) => ({
      batch_id: b.batch_id,
      accepted_count: 20,
      total_count: b.image_count,
      buildable: true,
    })),
    frozen_datasets: datasetsStore.map((d) => ({
      dataset_version: d.dataset_version,
      image_count: d.train_count + d.val_count + d.test_count,
    })),
  });
});

app.get('/api/crop-datasets/:version/validation', (req, res) => {
  res.json({
    state: 'READY_FOR_TRAINING',
    manifest_path: `gs://yujian-ai-models/crop-datasets/${req.params.version}/manifest.jsonl`,
    validation: {
      checks: {
        manifest_exists: true,
        bbox_coordinates_valid: true,
        class_balance: true,
      },
    },
  });
});

app.post('/api/crop-datasets/build', (req, res) => {
  res.json({
    state: 'BUILT',
    rows: 450,
    valid_rows: 442,
    classes: ['草鱼', '鲫鱼', '鲤鱼', '加州鲈', '黑鱼'],
    manifest_path: `gs://yujian-ai-models/crop-datasets/${req.body.dataset_version}/manifest.jsonl`,
    validation: {
      checks: {
        manifest_exists: true,
        bbox_coordinates_valid: true,
        aspect_ratio_normal: true,
      },
    },
  });
});

app.post('/api/crop-datasets/:version/freeze', (req, res) => {
  res.json({
    dataset_version: req.params.version,
    image_count: 442,
    state: 'READY_FOR_TRAINING',
  });
});

// Inference API
app.get('/api/inference/models', (req, res) => {
  res.json([
    {
      id: 'yujian-classifier-v0.2',
      name: '渔见淡水鱼分类器 v0.2',
      model_version: 'v0.2.0-cum',
      classes: 16,
      pipeline: 'WHOLE_IMAGE_V1',
    },
    {
      id: 'yujian-crop-v0.1',
      name: '渔见精准切片分类器 v0.1',
      model_version: 'crop-v0.1',
      classes: 12,
      pipeline: 'CROP_CLASSIFIER_V1',
    },
  ]);
});

app.post('/api/inference/predict', (req, res) => {
  res.json({
    model_version: 'yujian-classifier-v0.2',
    predicted_species: '草鱼',
    confidence: 0.958,
    top_k: [
      { species: '草鱼', confidence: 0.958 },
      { species: '青鱼', confidence: 0.024 },
      { species: '鲤鱼', confidence: 0.011 },
    ],
    bbox: [0.12, 0.15, 0.76, 0.58],
  });
});

// Training Runs API
app.post('/api/training/runs', (req, res) => {
  const { dataset_version, model_version, model_family, epochs } = req.body;
  const newRun = {
    run_id: `run-${Date.now().toString(36)}`,
    dataset_version: dataset_version || 'v0.2.0-cum',
    model_version: model_version || 'yujian-v0.3.0',
    model_family: model_family || 'mobilenet_v3_large',
    status: 'COMPLETED',
    started_at: new Date(Date.now() - 3600000).toISOString(),
    finished_at: new Date().toISOString(),
    artifact_uri: `gs://yujian-ai-models/runs/model.onnx`,
    metrics_uri: `gs://yujian-ai-models/runs/metrics.json`,
    params: {
      model_version: model_version || 'yujian-v0.3.0',
      epochs_completed: epochs || 30,
      epochs_requested: epochs || 30,
    },
  };
  res.json(newRun);
});

app.get('/api/training/runs/:run_id/metrics', (req, res) => {
  res.json({
    run_id: req.params.run_id,
    dataset_version: 'v0.2.0-cum',
    model_version: 'yujian-v0.2.0',
    params: {
      epochs_completed: 30,
      epochs_requested: 30,
    },
    test: {
      accuracy: 0.942,
      top3_accuracy: 0.985,
      macro_precision: 0.931,
      macro_recall: 0.928,
      macro_f1: 0.929,
      count: 312,
    },
    history: [
      { epoch: 10, val_macro_f1: 0.82 },
      { epoch: 20, val_macro_f1: 0.89 },
      { epoch: 30, val_macro_f1: 0.93 },
    ],
    per_class: [
      { species: '草鱼', precision: 0.95, recall: 0.94, f1: 0.945, support: 45 },
      { species: '鲫鱼', precision: 0.96, recall: 0.93, f1: 0.944, support: 42 },
      { species: '鲤鱼', precision: 0.93, recall: 0.91, f1: 0.92, support: 38 },
      { species: '黑鱼', precision: 0.94, recall: 0.95, f1: 0.945, support: 32 },
      { species: '加州鲈', precision: 0.92, recall: 0.93, f1: 0.925, support: 28 },
    ],
    confusion_matrix: [
      [42, 1, 1, 0, 1],
      [1, 39, 2, 0, 0],
      [1, 2, 35, 0, 0],
      [0, 0, 0, 30, 2],
      [1, 0, 0, 1, 26],
    ],
    classes: ['草鱼', '鲫鱼', '鲤鱼', '黑鱼', '加州鲈'],
    error_pairs: [
      { predicted: '鳙鱼', actual: '白鲢', count: 12 },
      { predicted: '青鱼', actual: '草鱼', count: 6 },
    ],
    warnings: [],
  });
});

// Intelligence API
app.get('/api/intelligence', (req, res) => {
  res.json({
    model: {
      model_version: 'yujian-v0.2.0',
      evaluation_status: 'READY',
      evaluation_source: 'Test Set Evaluation',
    },
    metrics: {
      accuracy: 0.942,
      macro_f1: 0.929,
      test_samples: 312,
    },
    evaluation_artifacts: {
      contract_version: 'v1.0-standard',
      test_samples: 312,
    },
    confusion_report: {
      top_confusions: [
        { predicted: '鳙鱼', actual: '白鲢', error_count: 14, rate: 0.12 },
        { predicted: '青鱼', actual: '草鱼', error_count: 8, rate: 0.08 },
        { predicted: '鲤鱼', actual: '鲫鱼', error_count: 5, rate: 0.05 },
      ],
    },
    data_gaps: {
      species_gaps: [
        { species: '马口鱼', target: 200, current: 45, gap: 155, status: 'URGENT' },
        { species: '鳡鱼', target: 200, current: 60, gap: 140, status: 'URGENT' },
        { species: '黄尾鲴', target: 200, current: 85, gap: 115, status: 'MODERATE' },
        { species: '红眼鳟', target: 200, current: 110, gap: 90, status: 'MODERATE' },
      ],
    },
    production_tasks: [
      {
        task_id: 'task-gap-chinese-hooksnout',
        title: '补充【马口鱼】自然野钓溪流样本',
        target_species: '马口鱼',
        needed_count: 155,
        priority: 'HIGH',
      },
      {
        task_id: 'task-gap-yellowcheek',
        title: '补充【鳡鱼】大体型路亚样本',
        target_species: '鳡鱼',
        needed_count: 140,
        priority: 'HIGH',
      },
    ],
    model_comparison: {
      baseline: { version: 'v0.1.0', accuracy: 0.884, macro_f1: 0.865 },
      current: { version: 'v0.2.0', accuracy: 0.942, macro_f1: 0.929 },
    },
  });
});

app.post('/api/intelligence/tasks/:taskId/batch', (req, res) => {
  res.json({
    message: '已生成建议批次参数',
    batch: {
      batch_id: `batch-${req.params.taskId}`,
      source: 'field_sampling',
    },
  });
});

// Catch-all for other /api/* routes to avoid breaking any button
app.all('/api/*', (req, res) => {
  res.status(200).json({ status: 'ok', message: 'Handled by in-memory stub' });
});

// Listen on 0.0.0.0:3000
app.listen(PORT, '0.0.0.0', () => {
  console.log(`渔见 AI Model Factory server running on http://0.0.0.0:${PORT}`);
});
