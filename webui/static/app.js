/* HyperFlow H3 WebUI — no framework, no build step. */

/* ── reference / keyframe attachment model ─────────────────────────────── */
/*
 * Files live in `state.attachments` rather than in the `<input type=file>` elements, because the UI has to support
 * removing one entry out of many and replacing a single entry, neither of which a file input can express. The input
 * stays in the DOM as a picker only; its FileList is never read on submit.
 */
const MAX_REFERENCE_FALLBACK = { images: 9, videos: 3, audios: 3, total: 12 };

const state = {
  config: null,
  mode: 't2va',
  aspect: '16:9',
  megapixels: 2,
  seconds: 5.2,
  override: false,
  seedRandom: true,
  devices: 1,
  parallel: 'single',
  selectedTask: null,
  logTask: null,
  // Completion detection: the id we are watching, and the last one we saw running.
  pendingTaskId: null,
  lastRunningId: null,
  pollTimer: null,
  logTimer: null,
  // { keyframe: {image, lastImage}, references: [{id, file, url, kind}] }
  attachments: { keyframe: { image: null, lastImage: null }, references: [] },
  refSeq: 0,
};

const $ = (id) => document.getElementById(id);

/* ── storage ───────────────────────────────────────────────────────────── */
/*
 * `sessionStorage` for the form, so a reload (or a crash) brings the whole draft back without a second tab
 * inheriting it. File handles cannot be serialised, so every attached file is mirrored into IndexedDB as a Blob and
 * re-hydrated on boot — the picker keeps working across a reload, which a plain file input never does.
 */
const DRAFT_KEY = 'hyperflow.draft.v1';
const BLOB_DB = 'hyperflow-media';
const BLOB_STORE = 'files';

function draftSnapshot() {
  return {
    mode: state.mode,
    aspect: state.aspect,
    megapixels: state.megapixels,
    seconds: state.seconds,
    override: state.override,
    seedRandom: state.seedRandom,
    devices: state.devices,
    parallel: state.parallel,
    userPickedDevices: Boolean(state.userPickedDevices),
    prompt: $('prompt') ? $('prompt').value : '',
    seed: $('seed') ? $('seed').value : '',
    width: $('width') ? $('width').value : '',
    height: $('height') ? $('height').value : '',
    offloadBlocks: $('offload_blocks') ? $('offload_blocks').value : '1',
    npuOffload: $('npu_offload') ? $('npu_offload').checked : true,
    npuFusedNorm: $('npu_fused_norm') ? $('npu_fused_norm').checked : true,
    baseline: $('baseline') ? $('baseline').checked : false,
    attentionBackend: $('attention_backend') ? $('attention_backend').value : 'auto',
    files: attachmentManifest(),
  };
}

function attachmentManifest() {
  const { keyframe, references } = state.attachments;
  const entry = (item) => item && { token: item.token, name: item.file.name, type: item.file.type, size: item.file.size };
  return {
    image: entry(keyframe.image),
    lastImage: entry(keyframe.lastImage),
    references: references.map((item) => ({
      token: item.token,
      name: item.file.name,
      type: item.file.type,
      size: item.file.size,
    })),
  };
}

function saveDraft() {
  try {
    sessionStorage.setItem(DRAFT_KEY, JSON.stringify(draftSnapshot()));
  } catch {
    /* private mode, or the quota is gone; a missing draft is not worth breaking the page over */
  }
}

function readDraft() {
  try {
    const raw = sessionStorage.getItem(DRAFT_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

/* ── IndexedDB blob mirror ─────────────────────────────────────────────── */

function openBlobDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(BLOB_DB, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(BLOB_STORE)) db.createObjectStore(BLOB_STORE);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function putBlob(token, file) {
  const db = await openBlobDb();
  await new Promise((resolve, reject) => {
    const tx = db.transaction(BLOB_STORE, 'readwrite');
    tx.objectStore(BLOB_STORE).put(file, token);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

async function getBlob(token) {
  const db = await openBlobDb();
  const file = await new Promise((resolve, reject) => {
    const tx = db.transaction(BLOB_STORE, 'readonly');
    const request = tx.objectStore(BLOB_STORE).get(token);
    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => reject(request.error);
  });
  db.close();
  return file;
}

async function dropBlob(token) {
  try {
    const db = await openBlobDb();
    await new Promise((resolve) => {
      const tx = db.transaction(BLOB_STORE, 'readwrite');
      tx.objectStore(BLOB_STORE).delete(token);
      tx.oncomplete = resolve;
      tx.onerror = resolve;
    });
    db.close();
  } catch {
    /* the mirror is a cache; losing it only costs a re-pick after a reload */
  }
}

function newToken() {
  state.refSeq += 1;
  return `${Date.now().toString(36)}-${state.refSeq}-${Math.random().toString(36).slice(2, 8)}`;
}

function makeAttachment(file, token = newToken()) {
  const url = URL.createObjectURL(file);
  return { token, file, url };
}

function releaseAttachment(attachment) {
  if (!attachment) return;
  if (attachment.url) URL.revokeObjectURL(attachment.url);
  dropBlob(attachment.token);
}

/* ── attachment rendering ──────────────────────────────────────────────── */

function mediaKind(file) {
  const type = (file.type || '').toLowerCase();
  if (type.startsWith('image/')) return 'image';
  if (type.startsWith('video/')) return 'video';
  if (type.startsWith('audio/')) return 'audio';
  const suffix = (file.name || '').split('.').pop().toLowerCase();
  if (['png', 'jpg', 'jpeg', 'webp', 'bmp'].includes(suffix)) return 'image';
  if (['mp4', 'mov', 'mkv', 'webm', 'avi'].includes(suffix)) return 'video';
  if (['wav', 'mp3', 'flac', 'm4a', 'ogg', 'aac'].includes(suffix)) return 'audio';
  return 'unknown';
}

function referenceLimits() {
  return (state.config && state.config.limits && state.config.limits.references) || MAX_REFERENCE_FALLBACK;
}

/** Human-readable byte size for the thumbnail caption. */
function formatBytes(size) {
  if (!size) return '';
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(0)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

/** Build the inner markup shared by the keyframe slots and the reference grid. */
function mediaFigure(attachment, { tag = '', caption = '', replaceable = true } = {}) {
  const figure = document.createElement('figure');
  figure.className = 'media-thumb';

  const kind = mediaKind(attachment.file);
  if (kind === 'image') {
    const img = document.createElement('img');
    img.src = attachment.url;
    img.alt = attachment.file.name;
    img.loading = 'lazy';
    figure.appendChild(img);
  } else if (kind === 'video') {
    const video = document.createElement('video');
    video.src = attachment.url;
    video.muted = true;
    video.playsInline = true;
    video.preload = 'metadata';
    figure.appendChild(video);
    const play = document.createElement('span');
    play.className = 'media-kind';
    play.textContent = '▶';
    figure.appendChild(play);
  } else {
    const box = document.createElement('span');
    box.className = 'media-file';
    box.textContent = kind === 'audio' ? '♪' : '?';
    figure.appendChild(box);
  }

  if (tag) {
    const badge = document.createElement('span');
    badge.className = `tag ${kind}`;
    badge.textContent = tag;
    figure.appendChild(badge);
  }

  const bar = document.createElement('figcaption');
  bar.className = 'media-bar';
  const name = document.createElement('span');
  name.className = 'name';
  name.textContent = caption || attachment.file.name;
  name.title = attachment.file.name;
  bar.appendChild(name);
  figure.appendChild(bar);

  if (replaceable) {
    const actions = document.createElement('div');
    actions.className = 'media-actions';
    const replace = document.createElement('button');
    replace.type = 'button';
    replace.className = 'icon-button';
    replace.title = '替换';
    replace.textContent = '⟳';
    actions.appendChild(replace);
    figure.appendChild(actions);
    return { figure, replace };
  }
  return { figure, replace: null };
}

function renderKeyframeSlot(slot, attachment, pick) {
  slot.innerHTML = '';
  slot.classList.toggle('filled', Boolean(attachment));
  slot.onclick = pick;  // an empty slot adds, a filled one replaces
  if (!attachment) {
    const empty = document.createElement('span');
    empty.className = 'media-empty';
    empty.textContent = '未选择 · 点击添加';
    slot.appendChild(empty);
    return;
  }
  const { figure } = mediaFigure(attachment, { replaceable: false });
  const remove = document.createElement('button');
  remove.type = 'button';
  remove.className = 'icon-button remove';
  remove.title = '移除';
  remove.textContent = '✕';
  remove.addEventListener('click', (event) => {
    event.stopPropagation();
    releaseAttachment(attachment);
    const key = slot.dataset.kind === 'image' ? 'image' : 'lastImage';
    state.attachments.keyframe[key] = null;
    renderKeyframeSlots();
    saveDraft();
  });
  const actions = document.createElement('div');
  actions.className = 'media-actions';
  actions.appendChild(remove);
  figure.appendChild(actions);
  slot.appendChild(figure);
}

/** Re-render both keyframe slots from the attachment model. */
function renderKeyframeSlots() {
  renderKeyframeSlot($('image-slot'), state.attachments.keyframe.image, () => $('image').click());
  renderKeyframeSlot($('last-image-slot'), state.attachments.keyframe.lastImage, () => $('last_image').click());
}

function renderReferences() {
  const grid = $('ref-grid');
  grid.innerHTML = '';
  const limits = referenceLimits();
  const items = state.attachments.references;

  if (!items.length) {
    const empty = document.createElement('span');
    empty.className = 'media-empty ref-empty';
    empty.textContent = '还没有参考素材 · 点击「添加素材」';
    grid.appendChild(empty);
  }

  items.forEach((attachment, index) => {
    const kind = mediaKind(attachment.file);
    const tag = kind === 'image' ? `P${index + 1}` : kind === 'video' ? `V${index + 1}` : `A${index + 1}`;
    const { figure, replace } = mediaFigure(attachment, { tag });
    replace.addEventListener('click', () => openReferencePicker(index));
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'icon-button remove';
    remove.title = '删除';
    remove.textContent = '✕';
    remove.addEventListener('click', () => {
      releaseAttachment(attachment);
      state.attachments.references.splice(index, 1);
      renderReferences();
      saveDraft();
    });
    figure.querySelector('.media-actions').appendChild(remove);
    grid.appendChild(figure);
  });

  $('ref-count').textContent = `${items.length} / ${limits.total}`;
}

function countKinds(attachments) {
  const counts = { image: 0, video: 0, audio: 0 };
  attachments.forEach((item) => {
    const kind = mediaKind(item.file);
    if (kind in counts) counts[kind] += 1;
  });
  return counts;
}

/** Add picked files, or replace one existing entry when `replaceIndex` is given. Limits are re-checked each time. */
function addReferences(files, replaceIndex = null) {
  const limits = referenceLimits();
  const incoming = Array.from(files);
  if (!incoming.length) return;

  const next = [...state.attachments.references];
  if (replaceIndex != null && next[replaceIndex]) {
    const [first, ...rest] = incoming;
    releaseAttachment(next[replaceIndex]);
    next[replaceIndex] = makeAttachment(first);
    incoming.length = 0;
    incoming.push(...rest);
  }

  const added = [];
  incoming.forEach((file) => {
    if (mediaKind(file) === 'unknown') return;
    const attachment = makeAttachment(file);
    const tentative = [...next, ...added, attachment];
    const counts = countKinds(tentative);
    if (counts.image > limits.images) { releaseAttachment(attachment); toast(`参考图片最多 ${limits.images} 张`, 'bad'); return; }
    if (counts.video > limits.videos) { releaseAttachment(attachment); toast(`参考视频最多 ${limits.videos} 个`, 'bad'); return; }
    if (counts.audio > limits.audios) { releaseAttachment(attachment); toast(`参考音频最多 ${limits.audios} 个`, 'bad'); return; }
    if (tentative.length > limits.total) { releaseAttachment(attachment); toast(`参考文件合计最多 ${limits.total} 个`, 'bad'); return; }
    added.push(attachment);
  });

  state.attachments.references = [...next, ...added];
  persistAttachments([...next, ...added]);
  renderReferences();
  saveDraft();
}

/** Mirror every reference plus the keyframes into IndexedDB so a reload can restore them. */
async function persistAttachments(items) {
  for (const item of items) {
    if (!item) continue;
    try {
      await putBlob(item.token, item.file);
    } catch {
      /* the draft still restores the form; only the file would be missing */
    }
  }
}

/* ── small helpers ─────────────────────────────────────────────────────── */

function toast(message, kind = '') {
  const node = $('toast');
  node.textContent = message;
  node.className = 'toast' + (kind ? ' ' + kind : '');
  node.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { node.hidden = true; }, kind === 'bad' ? 6000 : 3200);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let payload = null;
  try { payload = text ? JSON.parse(text) : null; } catch { payload = { error: text }; }
  if (!response.ok) {
    throw new Error((payload && payload.error) || `HTTP ${response.status}`);
  }
  return payload;
}

function formatDuration(seconds) {
  if (seconds == null || Number.isNaN(seconds)) return '—';
  const total = Math.round(seconds);
  if (total < 60) return `${total} 秒`;
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return rest ? `${minutes} 分 ${rest} 秒` : `${minutes} 分`;
}

/* ComfyUI ResolutionSelector arithmetic, mirrored for live preview. */
function resolveCanvas(megapixels, aspect, multiple = 32) {
  const [wRatio, hRatio] = aspect.split(':').map(Number);
  const totalPixels = megapixels * 1024 * 1024;
  const scale = Math.sqrt(totalPixels / (wRatio * hRatio));
  return {
    width: Math.round((wRatio * scale) / multiple) * multiple,
    height: Math.round((hRatio * scale) / multiple) * multiple,
  };
}

function alignFrames(frames) {
  // Must match `geometry.align_frames`: nearest `17n + 5` grid point, not rounding up. Rounding up would turn a
  // 5.2 s request (125 frames) into 141 frames (5.88 s).
  const value = Math.max(1, Math.round(frames));
  if (value <= 5) return 5;
  return 5 + Math.max(0, Math.round((value - 5) / 17)) * 17;
}

function estimateTokens(width, height, frames) {
  const aligned = alignFrames(frames);
  const latentFrames = Math.floor((aligned - 5) / 17) * 5 + 2;
  const video = latentFrames * (height / 32) * (width / 32);
  const audio = Math.round((aligned / 24) * 40) * 2;
  return Math.round(video + audio);
}

/* ── device strip ──────────────────────────────────────────────────────── */

function gib(value) {
  return value ? (value / 1073741824) : 0;
}

function renderDevices(device) {
  const strip = $('device-strip');
  strip.innerHTML = '';
  if (!device || !device.available) {
    const span = document.createElement('span');
    span.className = 'muted';
    span.textContent = device && device.pending
      ? '正在探测 NPU…'
      : `未检测到 NPU${device && device.reason ? '（' + device.reason + '）' : ''}`;
    strip.appendChild(span);
    return;
  }
  device.devices.forEach((item) => {
    const chip = document.createElement('div');
    chip.className = 'device-chip';

    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = item.label || `NPU ${item.id}`;
    chip.appendChild(name);

    // The allocatable budget is what a run can use; the physical total is what the board has. They differ by the
    // driver/CANN reservation, so both are shown rather than one standing in for the other.
    const allocTotal = gib(item.allocatable_bytes);
    const allocUsed = gib(item.allocated_bytes);
    const physTotal = gib(item.hbm_total_bytes);
    const physUsed = gib(item.hbm_used_bytes);

    const allocBar = document.createElement('div');
    allocBar.className = 'device-bar';
    const allocFill = document.createElement('span');
    const allocPct = allocTotal ? (100 * allocUsed) / allocTotal : 0;
    allocFill.style.width = `${Math.min(100, allocPct)}%`;
    if (allocPct >= 90) allocBar.classList.add('bad');
    else if (allocPct >= 70) allocBar.classList.add('warn');
    allocBar.appendChild(allocFill);
    chip.appendChild(allocBar);

    const allocText = document.createElement('span');
    allocText.className = 'pct';
    allocText.textContent = allocTotal ? `${allocUsed.toFixed(1)}/${allocTotal.toFixed(0)}G` : '—';
    allocText.title = allocTotal
      ? `框架可分配：已用 ${allocPct.toFixed(1)}%（${allocUsed.toFixed(2)} / ${allocTotal.toFixed(2)} GiB）`
      : '可分配显存未知';
    chip.appendChild(allocText);

    if (physTotal) {
      const phys = document.createElement('span');
      phys.className = 'pct phys';
      phys.textContent = `物理 ${physUsed.toFixed(1)}/${physTotal.toFixed(0)}G`;
      const reserved = gib(item.driver_reserved_bytes);
      phys.title =
        `物理 HBM：已用 ${physUsed.toFixed(2)} / ${physTotal.toFixed(2)} GiB` +
        (item.hbm_used_percent != null ? `（${item.hbm_used_percent}%）` : '') +
        (reserved ? ` · 驱动/CANN 预留 ${reserved.toFixed(2)} GiB` : '');
      chip.appendChild(phys);
    }

    if (item.temp_c != null) {
      const thermal = document.createElement('span');
      thermal.className = 'pct';
      thermal.textContent = `${Math.round(item.temp_c)}°C`;
      thermal.title = 'HBM 温度' + (item.power_w != null ? ` · ${Math.round(item.power_w)} W` : '');
      chip.appendChild(thermal);
    }

    if (item.aicore_percent != null) {
      const core = document.createElement('span');
      core.className = 'pct';
      core.textContent = `${Math.round(item.aicore_percent)}%`;
      core.title = 'AI Core 利用率';
      chip.appendChild(core);
    }
    strip.appendChild(chip);
  });
}

function applyParallelDefault(device) {
  // While the probe is still running the count is unknown; keep whatever the form has instead of collapsing to 1.
  if (!device || device.pending) return;
  const count = device.available ? device.count : 1;
  const select = $('devices');
  const options = select.options.length;
  const desired = Math.max(1, Math.min(4, count));
  // Rebuild only when the device set changed, and never override a choice the user already made.
  if (options !== desired) {
    select.innerHTML = '';
    for (let i = 1; i <= desired; i += 1) {
      const option = document.createElement('option');
      option.value = String(i);
      option.textContent = `${i} 卡`;
      select.appendChild(option);
    }
  }
  if (state.userPickedDevices) return;
  if (options === desired && state.devicesInitialised) return;
  state.devicesInitialised = true;
  // Multi-NPU defaults to context parallel, per the WebUI's own rule.
  state.devices = desired;
  select.value = String(state.devices);
  state.parallel = state.devices >= 2 ? 'cp' : 'single';
  $('parallel').value = state.parallel;
  updateParallelPill();
  syncOffloadAvailability();
  refreshPlan();
}

function updateParallelPill() {
  const pill = $('parallel-pill');
  const count = state.devices;
  let text = `${count} 卡`;
  if (state.parallel === 'cp') text += ' · CP';
  else if (state.parallel === 'tp') text += ' · TP';
  else text += ' · 单卡';
  pill.textContent = text;
  pill.classList.toggle('active', count >= 2);
}

function syncOffloadAvailability() {
  const tp = $('parallel').value === 'tp';
  const offload = $('npu_offload');
  offload.disabled = tp;
  if (tp) offload.checked = false;
  else if (!offload.dataset.userTouched) offload.checked = true;
}

/* ── config / form scaffolding ─────────────────────────────────────────── */

function renderAspects(aspects) {
  const group = $('aspect-group');
  group.innerHTML = '';
  aspects.forEach((aspect) => {
    const [w, h] = aspect.ratio;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'aspect';
    button.dataset.aspect = aspect.id;
    button.setAttribute('role', 'radio');
    button.setAttribute('aria-checked', String(aspect.id === state.aspect));
    button.title = aspect.label;

    const box = document.createElement('span');
    box.className = 'aspect-box';
    const inner = document.createElement('i');
    const maxSide = 22;
    const ratio = w / h;
    if (ratio >= 1) {
      inner.style.width = `${maxSide}px`;
      inner.style.height = `${Math.max(6, Math.round(maxSide / ratio))}px`;
    } else {
      inner.style.height = `${maxSide}px`;
      inner.style.width = `${Math.max(6, Math.round(maxSide * ratio))}px`;
    }
    box.appendChild(inner);
    button.appendChild(box);

    const label = document.createElement('span');
    label.textContent = aspect.id;
    button.appendChild(label);

    button.addEventListener('click', () => {
      state.aspect = aspect.id;
      group.querySelectorAll('.aspect').forEach((node) => {
        node.setAttribute('aria-checked', String(node.dataset.aspect === state.aspect));
      });
      refreshPlan();
    });
    group.appendChild(button);
  });
}

function setupMode() {
  const group = $('mode-group');
  group.querySelectorAll('.seg').forEach((button) => {
    button.addEventListener('click', () => {
      state.mode = button.dataset.mode;
      group.querySelectorAll('.seg').forEach((node) => {
        node.setAttribute('aria-checked', String(node.dataset.mode === state.mode));
      });
      updateModeVisibility();
    });
  });
  updateModeVisibility();
}

function updateModeVisibility() {
  $('keyframe-fields').hidden = state.mode !== 'fl2va';
  $('reference-fields').hidden = state.mode !== 'ref2va';
  const notes = {
    t2va: '仅提示词生成视频与音频。',
    fl2va: '给定首帧和/或尾帧。只放首帧即 i2v，只放尾帧即 l2v，两者都放即首尾帧。',
    ref2va: '参考图 / 视频 / 音频，按选择顺序编号为 <Picture n> / <Video n> / <Audio n>。',
  };
  $('mode-note').textContent = notes[state.mode];
}

function setupAttachments() {
  // The `<input type=file>` elements are pickers only: every chosen file is copied into `state.attachments`, which is
  // what renders and what the submit reads. That is what makes remove / replace possible at all.
  $('image-pick').addEventListener('click', () => $('image').click());
  $('last-image-pick').addEventListener('click', () => $('last_image').click());
  $('ref-add').addEventListener('click', () => openReferencePicker(null));

  const storeKeyframe = async (slot, file) => {
    const key = slot === 'image-slot' ? 'image' : 'lastImage';
    const previous = state.attachments.keyframe[key];
    // Same file re-picked (e.g. clicking "replace" and choosing the same one) must not revoke the live blob URL.
    if (previous && previous.file === file) return;
    releaseAttachment(previous);
    const attachment = makeAttachment(file);
    state.attachments.keyframe[key] = attachment;
    try { await putBlob(attachment.token, file); } catch { /* the mirror is a cache */ }
    renderKeyframeSlots();
    saveDraft();
  };

  $('image').addEventListener('change', async (event) => {
    const file = event.target.files && event.target.files[0];
    event.target.value = '';
    if (file) await storeKeyframe('image-slot', file);
  });
  $('last_image').addEventListener('change', async (event) => {
    const file = event.target.files && event.target.files[0];
    event.target.value = '';
    if (file) await storeKeyframe('last-image-slot', file);
  });

  $('references').addEventListener('change', (event) => {
    const files = Array.from(event.target.files || []);
    event.target.value = '';
    const target = state.referenceTarget;
    state.referenceTarget = null;
    if (files.length) addReferences(files, target);
  });
}

/** Copy the server's fastest-recipe defaults into the runtime controls. */
function applyRuntimeDefaults(defaults) {
  if (!defaults) return;
  if (defaults.attention_backend) $('attention_backend').value = defaults.attention_backend;
  if (typeof defaults.npu_fused_norm === 'boolean') $('npu_fused_norm').checked = defaults.npu_fused_norm;
  if (typeof defaults.npu_offload === 'boolean') {
    $('npu_offload').checked = defaults.npu_offload;
    $('npu_offload').dataset.userTouched = '';
  }
  if (Number.isFinite(defaults.offload_blocks)) $('offload_blocks').value = String(defaults.offload_blocks);
}

/** Reset the runtime controls to the fastest recipe the server measured, keeping the creative inputs. */
function resetToFastest() {
  const defaults = state.config && state.config.defaults;
  if (!defaults) return;
  applyRuntimeDefaults(defaults);
  const count = Math.max(1, Math.min(state.config.limits.devices || 1, defaults.devices));
  state.devices = count;
  state.userPickedDevices = false;
  state.parallel = count >= 2 ? 'cp' : 'single';
  $('devices').value = String(count);
  $('parallel').value = state.parallel;
  updateParallelPill();
  syncOffloadAvailability();
  refreshPlan();
  scheduleSave();
  toast('已切换为最快配置', 'good');
}

/** Open the reference picker; `index` replaces that entry, `null` appends. */
function openReferencePicker(index = null) {
  state.referenceTarget = index;
  $('references').click();
}

function setupSegments() {
  $('toggle-override').addEventListener('click', () => {
    state.override = !state.override;
    $('override-fields').hidden = !state.override;
    $('standard-canvas').querySelector('input[type="range"]').disabled = state.override;
    $('toggle-override').textContent = state.override ? '使用像素量' : '自定义尺寸';
    if (state.override) {
      const { width, height } = resolveCanvas(state.megapixels, state.aspect);
      $('width').value = String(width);
      $('height').value = String(height);
    }
    refreshPlan();
  });

  $('megapixels').addEventListener('input', (event) => {
    state.megapixels = Number(event.target.value);
    $('mp-value').textContent = `${state.megapixels.toFixed(2)} MP`;
    refreshPlan();
    scheduleSave();
  });

  $('seconds').addEventListener('input', (event) => {
    state.seconds = Number(event.target.value);
    refreshPlan();
    scheduleSave();
  });

  $('devices').addEventListener('change', (event) => {
    state.userPickedDevices = true;
    state.devices = Number(event.target.value);
    const parallel = $('parallel');
    if (state.devices === 1) {
      parallel.value = 'single';
    } else if (parallel.value === 'single') {
      parallel.value = 'cp';
    }
    state.parallel = parallel.value;
    updateParallelPill();
    syncOffloadAvailability();
    scheduleSave();
  });

  $('parallel').addEventListener('change', (event) => {
    state.parallel = event.target.value;
    if (state.parallel === 'tp' && state.devices < 2) {
      $('devices').value = '2';
      state.devices = 2;
    }
    if (state.parallel === 'single' && state.devices !== 1) {
      $('devices').value = '1';
      state.devices = 1;
    }
    if (state.parallel === 'cp' && state.devices < 2) {
      $('devices').value = '2';
      state.devices = 2;
    }
    updateParallelPill();
    syncOffloadAvailability();
    refreshPlan();
    scheduleSave();
  });

  $('npu_offload').addEventListener('change', (event) => {
    event.target.dataset.userTouched = '1';
    scheduleSave();
  });

  $('fastest-config').addEventListener('click', resetToFastest);

  $('random-seed').addEventListener('click', () => {
    const value = Math.floor(Math.random() * 2147483647);
    $('seed').value = String(value);
    state.seedRandom = false;
    saveDraft();
  });

  $('prompt').addEventListener('input', (event) => {
    $('prompt-count').textContent = String(event.target.value.length);
    scheduleSave();
  });

  // Every control that survives a reload writes the draft. `input` covers ranges and text as it is typed; `change`
  // covers the selects and checkboxes. A short debounce keeps typing from hitting sessionStorage on every keystroke.
  ['width', 'height', 'seed', 'offload_blocks', 'attention_backend', 'baseline', 'npu_fused_norm'].forEach((id) => {
    const node = $(id);
    node.addEventListener('input', scheduleSave);
    node.addEventListener('change', scheduleSave);
  });
}

let saveTimer = null;
function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveDraft, 250);
}

/* ── restore a draft (page reload, including a crashed tab) ────────────── */

function restoreDraft(draft) {
  if (!draft) return false;

  if (draft.mode) state.mode = draft.mode;
  if (draft.aspect) state.aspect = draft.aspect;
  if (Number.isFinite(draft.megapixels)) state.megapixels = draft.megapixels;
  if (Number.isFinite(draft.seconds)) state.seconds = draft.seconds;
  state.override = Boolean(draft.override);
  if (draft.seedRandom === false) state.seedRandom = false;
  if (draft.userPickedDevices) state.userPickedDevices = true;
  if (Number.isFinite(draft.devices)) state.devices = draft.devices;
  if (draft.parallel) state.parallel = draft.parallel;

  if (draft.prompt) $('prompt').value = draft.prompt;
  $('prompt-count').textContent = String($('prompt').value.length);
  if (draft.seed) $('seed').value = draft.seed;
  if (draft.width) $('width').value = draft.width;
  if (draft.height) $('height').value = draft.height;
  if (draft.offloadBlocks) $('offload_blocks').value = draft.offloadBlocks;
  if (typeof draft.npuOffload === 'boolean') $('npu_offload').checked = draft.npuOffload;
  if (typeof draft.npuFusedNorm === 'boolean') $('npu_fused_norm').checked = draft.npuFusedNorm;
  if (typeof draft.baseline === 'boolean') $('baseline').checked = draft.baseline;
  if (draft.attentionBackend) $('attention_backend').value = draft.attentionBackend;

  $('mode-group').querySelectorAll('.seg').forEach((node) => {
    node.setAttribute('aria-checked', String(node.dataset.mode === state.mode));
  });
  $('aspect-group').querySelectorAll('.aspect').forEach((node) => {
    node.setAttribute('aria-checked', String(node.dataset.aspect === state.aspect));
  });
  $('megapixels').value = String(state.megapixels);
  $('seconds').value = String(state.seconds);
  $('standard-canvas').querySelector('input[type="range"]').disabled = state.override;
  $('override-fields').hidden = !state.override;
  $('toggle-override').textContent = state.override ? '使用像素量' : '自定义尺寸';

  // Devices / parallel need the device probe to have run first (it decides how many options exist).
  if (!state.userPickedDevices) {
    state.devices = null;  // let `applyParallelDefault` pick it
  }
  updateModeVisibility();
  return true;
}

function restoreAttachments(draft) {
  if (!draft || !draft.files) return;
  const { files } = draft;
  const hydrate = async (entry) => {
    if (!entry || !entry.token) return null;
    try {
      const file = await getBlob(entry.token);
      return file ? { token: entry.token, file, url: URL.createObjectURL(file) } : null;
    } catch {
      return null;
    }
  };

  (async () => {
    const image = await hydrate(files.image);
    const lastImage = await hydrate(files.lastImage);
    state.attachments.keyframe.image = image;
    state.attachments.keyframe.lastImage = lastImage;
    renderKeyframeSlots();

    const refs = [];
    for (const entry of files.references || []) {
      const attachment = await hydrate(entry);
      if (attachment) refs.push(attachment);
    }
    state.attachments.references = refs;
    renderReferences();
    const missing = (files.references || []).length - refs.length + (files.image && !image ? 1 : 0)
      + (files.lastImage && !lastImage ? 1 : 0);
    if (missing > 0) toast(`${missing} 个上次选择的文件已失效，请重新选择`, 'bad');
  })();
}

/* ── plan preview ──────────────────────────────────────────────────────── */

function currentCanvas() {
  if (state.override) {
    const width = Number($('width').value) || 0;
    const height = Number($('height').value) || 0;
    return { width, height };
  }
  return resolveCanvas(state.megapixels, state.aspect);
}

function refreshPlan() {
  const { width, height } = currentCanvas();
  const frames = alignFrames(state.seconds * 24);
  const duration = frames / 24;
  const megapixels = (width * height) / 1024 / 1024;
  const tokens = estimateTokens(width, height, frames);

  $('plan-canvas').textContent = width && height ? `${width} × ${height}` : '—';
  $('plan-mp').textContent = width && height ? `${megapixels.toFixed(2)} MP` : '—';
  $('plan-frames').textContent = `${frames} 帧 · ${duration.toFixed(2)} s`;
  $('plan-devices').textContent = `${state.devices} 卡 · ${labelParallel(state.parallel)}`;
  $('plan-tokens').textContent = tokens ? `${tokens.toLocaleString()} tokens` : '—';

  $('mp-value').textContent = `${state.megapixels.toFixed(2)} MP`;
  $('seconds-value').textContent = `${duration.toFixed(2)} s`;
  $('frames-note').textContent = `${frames} 帧 · 17n+5`;
  if (Math.abs(duration - state.seconds) >= 0.01) {
    $('frames-note').textContent += `（请求 ${state.seconds.toFixed(2)} s，实际 ${duration.toFixed(2)} s）`;
  }

  const warn = $('plan-warn');
  const warnings = [];
  if (frames > 345) {
    warnings.push('15 s 需要 362 帧（15.08 s），已由 admit_max_frames 放宽上限；显存与耗时都会明显上升。');
  }
  if (tokens > 150000) warnings.push(`序列 ${tokens.toLocaleString()} tokens，attention 开销随平方增长，单步会明显变慢。`);
  if (width && height && (width % 32 || height % 32)) warnings.push('宽高必须是 32 的倍数。');
  if (state.parallel === 'tp') warnings.push('TP 在本机实测比 CP 慢且显存更高，建议用 CP。');
  if (state.parallel === 'single' && megapixels > 1.2) warnings.push('单卡跑 1.2 MP 以上会很慢，建议多卡 CP。');
  warn.hidden = warnings.length === 0;
  warn.textContent = warnings.join(' ');

  const maxDevices = state.config ? state.config.limits.devices : 1;
  if (state.devices > maxDevices) {
    toast(`只有 ${maxDevices} 个设备可用`, 'bad');
  }
}

function labelParallel(value) {
  return { auto: '自动', cp: '上下文并行', tp: '张量并行', single: '单卡' }[value] || value;
}

/* ── submission ────────────────────────────────────────────────────────── */

async function submit(event) {
  event.preventDefault();
  const button = $('submit-button');
  const hint = $('submit-hint');

  const prompt = $('prompt').value.trim();
  if (!prompt) {
    toast('请填写提示词', 'bad');
    $('prompt').focus();
    return;
  }

  const form = new FormData();
  form.set('prompt', prompt);
  form.set('mode', state.mode);
  form.set('aspect', state.aspect);
  form.set('seconds', String(state.seconds));
  form.set('devices', String(state.devices));
  form.set('parallel', state.parallel);
  form.set('npu_offload', $('npu_offload').checked ? '1' : '0');
  form.set('offload_blocks', $('offload_blocks').value || '1');
  form.set('npu_fused_norm', $('npu_fused_norm').checked ? '1' : '0');
  form.set('attention_backend', $('attention_backend').value);
  if ($('baseline').checked) form.set('baseline', '1');
  const seed = $('seed').value.trim();
  if (seed) form.set('seed', seed);

  if (state.override) {
    form.set('override', '1');
    form.set('width', $('width').value);
    form.set('height', $('height').value);
  } else {
    form.set('megapixels', String(state.megapixels));
  }

  if (state.mode === 'ref2va') {
    // The attachments model is the source of truth, not the picker's FileList: entries may have been removed or
    // replaced since they were picked.
    const items = state.attachments.references;
    if (!items.length) {
      toast('参考模式需要至少一张图片或一个视频', 'bad');
      return;
    }
    items.forEach((item, index) => form.append(`reference_${index}`, item.file, item.file.name));
  } else if (state.mode === 'fl2va') {
    const { image, lastImage } = state.attachments.keyframe;
    if (!image && !lastImage) {
      toast('关键帧模式需要首帧或尾帧', 'bad');
      return;
    }
    if (image) form.append('image', image.file, image.file.name);
    if (lastImage) form.append('last_image', lastImage.file, lastImage.file.name);
  }

  button.disabled = true;
  hint.textContent = '提交中…';
  try {
    const payload = await api('/api/tasks', { method: 'POST', body: form });
    toast('已加入队列', 'good');
    state.logTask = payload.task.id;
    // Start watching it right away so the poll that sees it finish switches the stage to the video, with no manual
    // page refresh. `lastRunningId` is set here too, which covers a task that finishes before the first poll.
    state.pendingTaskId = payload.task.id;
    state.lastRunningId = payload.task.id;
    state.selectedTask = payload.task.id;
    $('log-target').textContent = payload.task.id.slice(-6);
    await refreshAll();
    await loadLog(payload.task.id);
  } catch (error) {
    toast(error.message, 'bad');
  } finally {
    button.disabled = false;
    hint.textContent = '';
  }
}

/* ── polling ───────────────────────────────────────────────────────────── */

function badge(status) {
  const labels = {
    queued: '排队', running: '运行中', done: '完成',
    failed: '失败', cancelled: '已取消', interrupted: '已中断',
  };
  return `<span class="badge ${status}">${labels[status] || status}</span>`;
}

function taskSummary(task) {
  return `${task.width}×${task.height} · ${task.seconds}s · ${task.devices}卡`;
}

function renderQueue(queue) {
  const list = $('queue-list');
  list.innerHTML = '';
  const active = queue.active;
  const queued = queue.queued || [];

  if (!active && !queued.length) {
    $('queue-hint').textContent = '空闲';
  } else if (active) {
    $('queue-hint').textContent = `1 运行 · ${queued.length} 等待`;
  } else {
    $('queue-hint').textContent = `${queued.length} 等待`;
  }

  const items = active ? [active, ...queued] : queued;
  items.forEach((task) => {
    const li = document.createElement('li');
    li.innerHTML = `
      <div class="top">
        <span class="title">${task.id.slice(-6)}</span>
        ${badge(task.status)}
      </div>
      <div class="sub">${task.stage_label} · ${task.progress}% · ${taskSummary(task)}</div>
      <div class="mini-bar"><span style="width:${task.progress}%"></span></div>
    `;
    li.addEventListener('click', () => selectTask(task));
    if (task.status === 'running' || task.status === 'queued') {
      li.title = '点击查看日志；右键取消';
      li.addEventListener('contextmenu', async (event) => {
        event.preventDefault();
        if (!confirm(`取消任务 ${task.id}？`)) return;
        try {
          await api(`/api/tasks/${task.id}/cancel`, { method: 'POST' });
          toast('已请求取消');
          await refreshAll();
        } catch (error) { toast(error.message, 'bad'); }
      });
    }
    list.appendChild(li);
  });

  if (active) {
    renderStageProgress(active);
  }
}

function renderStageProgress(task) {
  const stage = $('stage');
  let overlay = stage.querySelector('.stage-progress');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.className = 'stage-progress';
    stage.appendChild(overlay);
  }
  overlay.innerHTML = `
    <div class="row"><span>${task.stage_label}</span><span>${task.progress}%</span></div>
    <div class="bar"><span style="width:${task.progress}%"></span></div>
  `;
  $('stage-hint').textContent = task.status === 'running'
    ? `运行中 · 已用 ${formatDuration(task.elapsed_seconds)}`
    : task.stage_label;
}

function clearStageOverlay() {
  const overlay = $('stage').querySelector('.stage-progress');
  if (overlay) overlay.remove();
}

function selectTask(task) {
  state.selectedTask = task.id;
  if (task.output_ready) {
    showVideo(task);
  } else {
    hideVideo();
    $('stage-hint').textContent = `${task.stage_label} · ${task.status}`;
  }
  state.logTask = task.id;
  $('log-target').textContent = task.id.slice(-6);
  loadLog(task.id);
  // Only re-mark the selected row; re-rendering needs the task list, which this function does not hold.
  document.querySelectorAll('#history-list li').forEach((node) => {
    node.classList.toggle('selected', node.dataset.taskId === task.id);
  });
}

function showVideo(task) {
  const video = $('result-video');
  const empty = $('stage-empty');
  const meta = $('stage-meta');
  const stage = $('stage');
  const url = `/api/tasks/${task.id}/video`;
  if (video.dataset.task !== task.id) {
    video.src = url;
    video.dataset.task = task.id;
  }
  // Size the stage to the clip's own aspect ratio. The old fixed 16/9 box letterboxed every other canvas (9:16,
  // 1:1, 21:9) and also rescaled the video inside it; this makes the panel follow the real geometry instead.
  if (task.width && task.height) {
    stage.style.aspectRatio = `${task.width} / ${task.height}`;
    video.style.aspectRatio = `${task.width} / ${task.height}`;
  }
  // A vertical clip should not force the middle column to be taller than the viewport.
  stage.classList.toggle('portrait', Boolean(task.width && task.height && task.height > task.width));
  video.hidden = false;
  empty.hidden = true;
  meta.hidden = false;
  const link = $('download-link');
  link.href = url;
  link.download = `${task.id}.mp4`;
  clearStageOverlay();
  $('stage-hint').textContent =
    `${task.width}×${task.height} · ${task.frames} 帧 · ${formatDuration(task.elapsed_seconds)}`;
}

function hideVideo() {
  const video = $('result-video');
  const stage = $('stage');
  video.hidden = true;
  video.removeAttribute('src');
  video.dataset.task = '';
  // Back to the neutral placeholder box.
  stage.style.aspectRatio = '16 / 9';
  video.style.aspectRatio = '';
  stage.classList.remove('portrait');
  $('stage-empty').hidden = false;
  $('stage-meta').hidden = true;
}

function renderHistory(tasks) {
  const list = $('history-list');
  list.innerHTML = '';
  $('history-count').textContent = String(tasks.length);
  tasks.forEach((task) => {
    const li = document.createElement('li');
    li.dataset.taskId = task.id;
    if (task.id === state.selectedTask) li.classList.add('selected');
    li.innerHTML = `
      <div class="top">
        <span class="title">${task.id.slice(-6)}</span>
        ${badge(task.status)}
      </div>
      <div class="sub">${taskSummary(task)} · ${task.seed}</div>
      <div class="sub">${(task.prompt || '').slice(0, 60)}</div>
    `;
    li.addEventListener('click', () => selectTask(task));
    list.appendChild(li);
  });
}

async function loadLog(taskId) {
  try {
    // The log endpoint returns `text/plain`, so it must not go through the JSON helper.
    const response = await fetch(`/api/tasks/${taskId}/log?lines=400`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await response.text();
    const view = $('log-view');
    const atBottom = view.scrollTop + view.clientHeight >= view.scrollHeight - 30;
    view.textContent = text || '（暂无日志）';
    if (atBottom) view.scrollTop = view.scrollHeight;
  } catch {
    /* a task can disappear between polls */
  }
}

/** Whenever a task finishes, pull the finished record and switch the stage to its video. */
const FINISHED = new Set(['done', 'failed', 'cancelled', 'interrupted']);

function markSelected(taskId) {
  document.querySelectorAll('#history-list li').forEach((node) => {
    node.classList.toggle('selected', node.dataset.taskId === taskId);
  });
}

/** Show a finished task: video on success, log + toast otherwise. Idempotent, so repeated polls are harmless. */
function noteFinished(task) {
  if (!task || !FINISHED.has(task.status)) return;
  if (task.status === 'done') {
    // The record is complete (`queue_snapshot`/`/api/tasks/<id>` return the public shape), so no extra round trip.
    showVideo(task);
    state.selectedTask = task.id;
    state.logTask = task.id;
    $('log-target').textContent = task.id.slice(-6);
    loadLog(task.id);
    markSelected(task.id);
    toast('生成完成', 'good');
  } else if (task.status === 'failed') {
    toast(`任务 ${task.id.slice(-6)} 失败：${task.error || '见日志'}`, 'bad');
    state.logTask = task.id;
    $('log-target').textContent = task.id.slice(-6);
    loadLog(task.id);
  } else {
    $('stage-hint').textContent = `${task.stage_label} · ${task.status}`;
    clearStageOverlay();
  }
}

async function refreshStatus() {
  try {
    const payload = await api('/api/status');
    renderDevices(payload.device);
    applyParallelDefault(payload.device);
    renderQueue(payload.queue);

    const active = payload.queue.active;
    if (active) {
      renderStageProgress(active);
      if (active.id !== state.lastRunningId) {
        // A task just started: the queue and the history both changed.
        state.lastRunningId = active.id;
        state.pendingTaskId = active.id;
        await refreshHistory();
      }
    } else {
      if (state.pendingTaskId) {
        // The task we are watching is no longer running. One targeted fetch tells us how it ended.
        const finishedId = state.pendingTaskId;
        const task = await api(`/api/tasks/${finishedId}`);
        if (FINISHED.has(task.status)) {
          state.pendingTaskId = null;
          state.lastRunningId = null;
          noteFinished(task);
          await refreshHistory();
        }
      } else if (state.lastRunningId) {
        // We saw it running before this page started polling (or the id changed under us).
        state.lastRunningId = null;
        const tasks = await api('/api/tasks');
        renderHistory(tasks.tasks);
        const latest = tasks.tasks.find((item) => FINISHED.has(item.status));
        if (latest) noteFinished(latest);
      }
      clearStageOverlay();
    }
  } catch (error) {
    console.warn('status', error);
  }
}

/** History only changes when a task starts or stops; refresh it on those transitions rather than every 2 s. */
async function refreshHistory() {
  try {
    const payload = await api('/api/tasks');
    renderHistory(payload.tasks);
  } catch (error) {
    console.warn('history', error);
  }
}

async function refreshAll() {
  try {
    const [status, tasks] = await Promise.all([api('/api/status'), api('/api/tasks')]);
    renderDevices(status.device);
    renderQueue(status.queue);
    renderHistory(tasks.tasks);
    if (state.selectedTask) {
      const task = tasks.tasks.find((item) => item.id === state.selectedTask);
      if (task && task.output_ready) showVideo(task);
    }
    const active = status.queue.active;
    if (active) {
      renderStageProgress(active);
      state.lastRunningId = active.id;
      state.pendingTaskId = active.id;
    } else {
      if (!state.selectedTask || !$('result-video').dataset.task) clearStageOverlay();
    }
  } catch (error) {
    console.warn('refresh', error);
  }
}

/* ── boot ──────────────────────────────────────────────────────────────── */

async function boot() {
  const draft = readDraft();
  setupMode();
  setupSegments();
  setupAttachments();
  renderReferences();
  renderKeyframeSlots();
  $('task-form').addEventListener('submit', submit);
  $('task-form').addEventListener('input', scheduleSave);
  $('task-form').addEventListener('change', scheduleSave);

  // Fallback for the stage ratio: trust the element's own metadata once it loads. `showVideo` already sets the ratio
  // from the task record, so this only matters if the record and the file ever disagree.
  $('result-video').addEventListener('loadedmetadata', () => {
    const video = $('result-video');
    if (!video.videoWidth || !video.videoHeight) return;
    $('stage').style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    $('stage').classList.toggle('portrait', video.videoHeight > video.videoWidth);
  });

  try {
    const config = await api('/api/config');
    state.config = config;
    renderAspects(config.aspects);
    renderDevices(config.device);

    // A draft wins over the server defaults: the user's last state is what a reload should bring back. The device
    // count and parallel mode still defer to the probe unless the user picked them explicitly.
    if (restoreDraft(draft)) {
      state.devicesInitialised = true;
    } else {
      $('megapixels').value = String(config.defaults.megapixels);
      state.megapixels = config.defaults.megapixels;
      $('mp-value').textContent = `${state.megapixels.toFixed(2)} MP`;
      $('seconds').value = String(config.defaults.seconds);
      state.seconds = config.defaults.seconds;
      $('seconds-value').textContent = `${config.defaults.seconds.toFixed(2)} s`;
      applyRuntimeDefaults(config.defaults);
    }

    applyParallelDefault(config.device);
    if (!state.userPickedDevices) {
      const preferred = draft && Number.isFinite(draft.devices) ? draft.devices : config.defaults.devices;
      state.devices = Math.max(1, Math.min(state.config.limits.devices || preferred, preferred));
    } else {
      state.devices = Math.max(1, Math.min(state.config.limits.devices || state.devices, state.devices));
    }
    $('devices').value = String(state.devices);
    $('parallel').value = state.parallel;
    updateParallelPill();
    syncOffloadAvailability();
    refreshPlan();
  } catch (error) {
    toast(`读取配置失败：${error.message}`, 'bad');
  }

  restoreAttachments(draft);

  $('refresh-status').addEventListener('click', () => refreshAll());

  await refreshAll();
  state.pollTimer = setInterval(refreshStatus, 2000);
  setInterval(() => { if (state.logTask) loadLog(state.logTask); }, 3000);
}

document.addEventListener('DOMContentLoaded', boot);
