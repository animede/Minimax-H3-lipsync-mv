const state = { image: null, sourceFile: null, sourceMode: null, job: null, source: null, startedAt: 0 };
const $ = (id) => document.getElementById(id);

function readableBytes(value) {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function setImage(file) {
  // MIMEタイプが空で渡るファイルマネージャがあるため拡張子でも判定する
  const okType = file && (["image/png", "image/jpeg", "image/webp"].includes(file.type) ||
    (!file.type && /\.(png|jpe?g|webp)$/i.test(file.name)));
  if (!okType) {
    $("formError").textContent = "PNG、JPEG、WebP画像を選択してください。";
    return;
  }
  state.image = file;
  $("imagePreview").src = URL.createObjectURL(file);
  $("imageEmpty").classList.add("hidden");
  $("imagePreviewWrap").classList.remove("hidden");
  $("formError").textContent = "";
  updateGenerateState();
}

function setAudio(file) {
  const lowerName = file?.name.toLowerCase() || "";
  const isText = lowerName.endsWith(".txt");
  const valid = isText || [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"].some(ext => lowerName.endsWith(ext));
  if (!file || !valid) {
    $("formError").textContent = "対応している音声ファイルまたはTXTを選択してください。";
    return;
  }
  state.sourceFile = file;
  state.sourceMode = isText ? "text" : "song";
  $("scriptText").value = "";
  $("audioName").textContent = file.name;
  $("audioMeta").textContent = `${readableBytes(file.size)} · ${isText ? "AivisSpeechで文単位に読み上げ" : "楽曲"}`;
  $("audioPreview").classList.toggle("hidden", isText);
  $("textPreview").classList.toggle("hidden", !isText);
  if (isText) {
    file.text().then((value) => { $("textPreview").textContent = value.slice(0, 1200); });
    $("audioPreview").removeAttribute("src");
  } else {
    $("audioPreview").src = URL.createObjectURL(file);
    $("textPreview").textContent = "";
  }
  $("audioEmpty").classList.add("hidden");
  $("audioPreviewWrap").classList.remove("hidden");
  $("formError").textContent = "";
  updateGenerateState();
}

function clearSource() {
  state.sourceFile = null;
  state.sourceMode = null;
  $("audioInput").value = "";
  $("audioPreview").pause();
  $("audioPreview").removeAttribute("src");
  $("textPreview").textContent = "";
  $("audioPreviewWrap").classList.add("hidden");
  $("audioEmpty").classList.remove("hidden");
  updateGenerateState();
}

function setPastedText(value) {
  const text = value.trim();
  if (!text) {
    if (state.sourceFile?.name === "pasted-script.txt") clearSource();
    return;
  }
  const file = new File([text], "pasted-script.txt", { type: "text/plain;charset=utf-8" });
  state.sourceFile = file;
  state.sourceMode = "text";
  $("audioName").textContent = "貼り付けた読み上げ原稿";
  $("audioMeta").textContent = `${readableBytes(file.size)} · AivisSpeechで文単位に読み上げ`;
  $("audioPreview").classList.add("hidden");
  $("audioPreview").removeAttribute("src");
  $("textPreview").textContent = text.slice(0, 1200);
  $("textPreview").classList.remove("hidden");
  $("audioEmpty").classList.add("hidden");
  $("audioPreviewWrap").classList.remove("hidden");
  $("formError").textContent = "";
  updateGenerateState();
}

function updateGenerateState() {
  $("generateButton").disabled = !(state.image && state.sourceFile) || !!state.job && ["queued", "running"].includes(state.job.status);
}

// 枠外ドロップでブラウザがファイルを開いてページ遷移するのを防ぐ
for (const event of ["dragover", "drop"]) window.addEventListener(event, (e) => e.preventDefault());

function setupDrop(zoneId, inputId, setter, { allowUrl = false } = {}) {
  const zone = $(zoneId);
  const input = $(inputId);
  input.addEventListener("change", () => setter(input.files[0]));
  for (const event of ["dragenter", "dragover"]) zone.addEventListener(event, (e) => {
    e.preventDefault(); zone.classList.add("dragging");
  });
  for (const event of ["dragleave", "drop"]) zone.addEventListener(event, (e) => {
    e.preventDefault(); zone.classList.remove("dragging");
  });
  zone.addEventListener("drop", async (e) => {
    const file = e.dataTransfer.files[0];
    if (file) return setter(file);
    // 他タブの画像やファイルマネージャは File が無く URL だけ来ることがある
    const url = (e.dataTransfer.getData("text/uri-list") ||
                 e.dataTransfer.getData("text/plain") || "").split("\n")[0].trim();
    if (!url || !allowUrl) {
      $("formError").textContent = "ドロップからファイルを取り出せませんでした。クリックで選択してください。";
      return;
    }
    try {
      let resp = null;
      if (!url.startsWith("file:")) {
        try {
          resp = await fetch(url, { mode: "cors" });
          if (!resp.ok) throw new Error(resp.status);
        } catch { resp = null; }
      }
      if (!resp) {
        resp = await fetch(`/api/fetch-image?url=${encodeURIComponent(url)}`);
        if (!resp.ok) throw new Error((await resp.json().catch(() => null))?.detail || resp.status);
      }
      const blob = await resp.blob();
      const name = decodeURIComponent(url.split("/").pop().split("?")[0]) || "dropped.png";
      setter(new File([blob], name, { type: blob.type }));
    } catch (err) {
      $("formError").textContent = `取り込めませんでした(${err.message})。クリックで選択してください。`;
    }
  });
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
}

function formatClock(seconds) {
  const elapsed = Math.max(0, Math.floor(seconds || 0));
  return `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;
}

function totalElapsed(job) {
  if (job.total_elapsed_s) return job.total_elapsed_s;
  if (job.run_started_at) return Math.max(0, Date.now() / 1000 - job.run_started_at);
  return state.startedAt ? Math.max(0, (Date.now() - state.startedAt) / 1000) : 0;
}

function formatDuration(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  if (value < 60) return `${value.toFixed(1)}秒`;
  const minutes = Math.floor(value / 60);
  return `${minutes}分 ${(value - minutes * 60).toFixed(1)}秒`;
}

function renderTimings(job) {
  const timings = job.timings || [];
  const sceneTimings = timings.filter((item) => item.kind === "scene");
  const lines = [];
  timings.filter((item) => item.kind !== "scene").forEach((item) => {
    lines.push(`${item.label}  ${formatDuration(item.duration_s)}`);
    if (item.stage === "generating") {
      sceneTimings.forEach((scene) => {
        lines.push(`  └ ${scene.label}  ${formatDuration(scene.duration_s)}`);
      });
    }
  });
  if (!timings.some((item) => item.stage === "generating" && item.kind !== "scene")) {
    sceneTimings.forEach((scene) => {
      lines.push(`  └ ${scene.label}  ${formatDuration(scene.duration_s)}`);
    });
  }
  if (["queued", "running"].includes(job.status) && job.timing_label && job.timing_started_at) {
    const activeSeconds = Date.now() / 1000 - job.timing_started_at;
    lines.push(`▶ ${job.timing_label}  ${formatDuration(activeSeconds)}（処理中）`);
  }
  if (job.total_elapsed_s) lines.push(`\n合計  ${formatDuration(job.total_elapsed_s)}`);
  $("generationLog").textContent = lines.length ? lines.join("\n") : "所要時間を計測しています…";
  $("generationLog").scrollTop = $("generationLog").scrollHeight;
}

function addScenarioBlock(parent, label, value) {
  if (!value) return;
  const block = document.createElement("section");
  block.className = "scenario-block";
  const heading = document.createElement("h4");
  heading.textContent = label;
  const text = document.createElement("p");
  text.textContent = value;
  block.append(heading, text);
  parent.append(block);
}

function renderScenario(job) {
  const scenario = job.scenario || {};
  const overview = scenario.overview || {};
  const scenes = Array.isArray(scenario.scenes) ? scenario.scenes : [];
  const available = Object.keys(overview).length > 0 || scenes.length > 0;
  $("analysisSections").classList.toggle("hidden", !available);
  if (!available) return;

  const target = $("scenarioContent");
  target.replaceChildren();
  if (overview.title) {
    const title = document.createElement("h3");
    title.className = "scenario-title";
    title.textContent = overview.title;
    target.append(title);
  }
  addScenarioBlock(target, "ビジュアルテーマ", overview.visual_theme);
  addScenarioBlock(target, "カラー設計", overview.color_script);
  addScenarioBlock(target, "ストーリー", overview.story_arc);

  if (Array.isArray(overview.continuity_rules) && overview.continuity_rules.length) {
    const block = document.createElement("section");
    block.className = "scenario-block";
    const heading = document.createElement("h4");
    heading.textContent = "連続性ルール";
    const list = document.createElement("ul");
    overview.continuity_rules.forEach((rule) => {
      const item = document.createElement("li");
      item.textContent = rule;
      list.append(item);
    });
    block.append(heading, list);
    target.append(block);
  }

  if (scenes.length) {
    const list = document.createElement("ol");
    list.className = "scenario-scenes";
    scenes.forEach((scene) => {
      const timing = (job.scenes || []).find((item) => Number(item.index) === Number(scene.index));
      const item = document.createElement("li");
      const heading = document.createElement("strong");
      const timingText = timing
        ? ` · 長さ ${Number(timing.duration).toFixed(1)}秒（${Number(timing.start).toFixed(1)}–${Number(timing.end).toFixed(1)}秒）`
        : "";
      heading.textContent = `Scene ${scene.index}${timingText}`;
      const details = [scene.emotion, scene.shot, scene.camera].filter(Boolean).join(" · ");
      if (details) {
        const text = document.createElement("span");
        text.textContent = details;
        item.append(heading, text);
      } else {
        item.append(heading);
      }
      list.append(item);
    });
    target.append(list);
  }
}

function renderJob(job) {
  state.job = job;
  $("emptyOutput").classList.add("hidden");
  $("jobOutput").classList.remove("hidden");
  const progress = Math.round((job.progress || 0) * 100);
  $("stageLabel").textContent = job.stage.replaceAll("_", " ");
  $("statusMessage").textContent = job.message || "処理中";
  $("progressText").textContent = `${progress}%`;
  $("progressBar").style.width = `${progress}%`;
  $("sceneProgress").textContent = job.scene_count ? `シーン ${job.current_scene || 0} / ${job.scene_count}` : "シーンを解析中";
  $("elapsedTime").textContent = `合計 ${formatClock(totalElapsed(job))}`;
  const width = Number(job.width) || 1024;
  const height = Number(job.height) || 768;
  $("actualSizeLabel").textContent = `${width} × ${height} 実寸プレビュー`;
  $("actualVideo").style.width = `${width}px`;
  $("actualVideo").style.height = `${height}px`;
  $("resultVideo").style.aspectRatio = `${width} / ${height}`;
  renderTimings(job);
  renderScenario(job);
  $("jobError").textContent = job.error || "";
  const active = ["queued", "running"].includes(job.status);
  $("cancelButton").classList.toggle("hidden", !active);
  $("retryButton").classList.toggle("hidden", !["failed", "cancelled"].includes(job.status));
  if (job.status === "completed") {
    const videoUrl = `/api/jobs/${job.id}/output`;
    $("resultVideo").src = videoUrl;
    $("actualVideo").src = videoUrl;
    $("downloadButton").href = `/api/jobs/${job.id}/download`;
    $("videoArea").classList.remove("hidden");
  }
  updateGenerateState();
}

function watchJob(jobId) {
  if (state.source) state.source.close();
  state.source = new EventSource(`/api/jobs/${jobId}/events`);
  state.source.onmessage = (event) => {
    const job = JSON.parse(event.data);
    renderJob(job);
    if (["completed", "failed", "cancelled"].includes(job.status)) state.source.close();
  };
  state.source.onerror = () => state.source?.close();
}

async function generate() {
  $("formError").textContent = "";
  const body = new FormData();
  body.append("character", state.image);
  body.append(state.sourceMode === "text" ? "text" : "song", state.sourceFile);
  body.append("concept", $("concept").value.trim());
  const [width, height] = $("videoSize").value.split("x");
  body.append("width", width);
  body.append("height", height);
  $("generateButton").disabled = true;
  state.startedAt = Date.now();
  try {
    const response = await fetch("/api/jobs", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "生成を開始できませんでした");
    renderJob(data);
    watchJob(data.id);
  } catch (error) {
    $("formError").textContent = error.message;
    updateGenerateState();
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    const ready = data.llms?.scenario?.ok;
    const ttsReady = data.tts?.ok;
    $("healthBadge").textContent = ready ? `LLM ONLINE · TTS ${ttsReady ? "ONLINE" : "OFFLINE"}` : "接続を確認してください";
    $("healthBadge").classList.toggle("ok", ready);
  } catch { $("healthBadge").textContent = "OFFLINE"; }
}

setupDrop("imageDrop", "imageInput", setImage, { allowUrl: true });
setupDrop("audioDrop", "audioInput", setAudio);
$("scriptText").addEventListener("input", (event) => setPastedText(event.target.value));
// 投稿先(縦横比)× 画質 -> 生成サイズ。すべて 32 の倍数(H3 の制約)。画素数は最大でも
// 1344×768 程度に抑えてある(48GB 級 GPU で 10 秒シーンが収まる範囲)。
const VIDEO_SIZE_TABLE = {
  youtube:   { hq: "1344x768", std: "1024x576", fast: "768x448" },
  vertical:  { hq: "768x1344", std: "576x1024", fast: "448x768" },
  igfeed:    { hq: "768x960",  std: "640x800",  fast: "512x640" },
  square:    { hq: "960x960",  std: "768x768",  fast: "576x576" },
  classic:   { hq: "1024x768", std: "768x576",  fast: "512x384" },
  classic_v: { hq: "768x1024", std: "576x768",  fast: "384x512" },
};
// この GPU で 10 秒シーンが収まる画素数の目安(/api/h3/capacity、null = 制限なし)。
let gpuCapacity = { max_pixels: null, gpu_gb: null };
function updateVideoSize() {
  const size = VIDEO_SIZE_TABLE[$("videoPlatform").value][$("videoQuality").value];
  $("videoSize").value = size;
  $("sizeSpec").textContent = size.replace("x", " × ");
  const [w, h] = size.split("x").map(Number);
  const limit = gpuCapacity.max_pixels;
  $("sizeWarning").textContent = (limit && w * h > limit)
    ? `⚠ このGPU(${gpuCapacity.gpu_gb ?? "?"}GB)ではメモリ不足で失敗する可能性があります。画質を下げてください。`
    : "";
}
fetch("/api/h3/capacity")
  .then((r) => r.json())
  .then((cap) => { gpuCapacity = cap; updateVideoSize(); })
  .catch(() => {});
$("videoPlatform").addEventListener("change", updateVideoSize);
$("videoQuality").addEventListener("change", updateVideoSize);
updateVideoSize();
$("removeImage").addEventListener("click", (e) => {
  e.preventDefault(); state.image = null; $("imagePreviewWrap").classList.add("hidden"); $("imageEmpty").classList.remove("hidden"); updateGenerateState();
});
$("removeAudio").addEventListener("click", (e) => {
  e.preventDefault(); $("scriptText").value = ""; clearSource();
});
$("generateButton").addEventListener("click", generate);
$("cancelButton").addEventListener("click", async () => {
  if (!state.job) return;
  await fetch(`/api/jobs/${state.job.id}/cancel`, { method: "POST" });
});
$("openActual").addEventListener("click", () => $("actualDialog").showModal());
$("retryButton").addEventListener("click", async () => {
  if (!state.job) return;
  state.startedAt = Date.now();
  const response = await fetch(`/api/jobs/${state.job.id}/retry`, { method: "POST" });
  const data = await response.json();
  if (!response.ok) { $("jobError").textContent = data.detail || "再開できませんでした"; return; }
  renderJob(data); watchJob(data.id);
});
$("closeDialog").addEventListener("click", () => $("actualDialog").close());
$("fitToggle").addEventListener("click", () => {
  const viewport = $("actualViewport");
  viewport.classList.toggle("fit");
  $("fitToggle").textContent = viewport.classList.contains("fit") ? "100%表示" : "画面に合わせる";
});
setInterval(() => {
  if (state.job && ["queued", "running"].includes(state.job.status)) {
    $("elapsedTime").textContent = `合計 ${formatClock(totalElapsed(state.job))}`;
    renderTimings(state.job);
  }
}, 1000);
checkHealth();

// ---------------- 生成履歴(リロード/再起動後も過去ジョブへ辿れる) ----------------
function historyLabel(job) {
  const when = job.created_at ? new Date(job.created_at * 1000).toLocaleString("ja-JP") : job.id;
  const src = job.input_mode === "narration"
    ? (job.text_file || "テキスト") : (job.song_file || "楽曲");
  return `${when} · ${src}`;
}

const HISTORY_STATUS = { completed: "完了", failed: "失敗", cancelled: "中止", running: "生成中", queued: "待機" };

async function loadHistory() {
  const list = $("historyList");
  if (!list) return;
  try {
    const r = await fetch("/api/jobs");
    const d = await r.json();
    const jobs = d.jobs || [];
    if (!jobs.length) {
      list.innerHTML = '<li class="history-empty">まだありません</li>';
      return;
    }
    list.innerHTML = "";
    for (const job of jobs) {
      const li = document.createElement("li");
      li.className = "history-item";
      const open = document.createElement("a");
      open.href = "#";
      open.textContent = historyLabel(job);
      open.addEventListener("click", (e) => {
        e.preventDefault();
        state.job = job;
        renderJob(job);
        if (["queued", "running"].includes(job.status)) watchJob(job.id);
        $("jobOutput").scrollIntoView({ behavior: "smooth" });
      });
      const status = document.createElement("span");
      status.className = `history-status history-${job.status}`;
      status.textContent = HISTORY_STATUS[job.status] || job.status;
      li.appendChild(open);
      li.appendChild(status);
      if (job.status === "completed") {
        const dl = document.createElement("a");
        dl.href = `/api/jobs/${job.id}/download`;
        dl.setAttribute("download", "");
        dl.className = "history-download";
        dl.textContent = "MP4";
        li.appendChild(dl);
      }
      list.appendChild(li);
    }
  } catch (e) {
    list.innerHTML = '<li class="history-empty">履歴の取得に失敗しました</li>';
  }
}
loadHistory();
