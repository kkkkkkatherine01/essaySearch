const form = document.getElementById("query-form");
const input = document.getElementById("query-input");
const submitBtn = document.getElementById("submit-btn");
const progressSection = document.getElementById("progress");
const logBox = document.getElementById("log-box");
const selectionBox = document.getElementById("selection-box");
const selectionList = document.getElementById("selection-list");
const defaultCheckedCountEl = document.getElementById("default-checked-count");
const confirmSelectionBtn = document.getElementById("confirm-selection-btn");
const candidatesBox = document.getElementById("candidates-box");
const candidatesList = document.getElementById("candidates-list");
const resultSection = document.getElementById("result");
const statsBox = document.getElementById("stats-box");
const answerBox = document.getElementById("answer-box");
const citationCheckBox = document.getElementById("citation-check-box");
const citationCheckTitle = document.getElementById("citation-check-title");
const citationCheckList = document.getElementById("citation-check-list");
const referencesBox = document.getElementById("references-box");
const evidenceBox = document.getElementById("evidence-box");
const evidenceList = document.getElementById("evidence-list");
const toggleEvidenceBtn = document.getElementById("toggle-evidence-btn");
const errorBox = document.getElementById("error-box");
const historyList = document.getElementById("history-list");

// Matches backend config.MAX_PAPERS's default — just used to decide how
// many candidates come pre-checked in the selection UI.
const DEFAULT_CHECKED_COUNT = 6;

const STEP_ORDER = ["searching", "awaiting_selection", "downloading", "generating", "done"];
let pollTimer = null;
let renderedLogCount = 0;
let selectionRenderedForJob = null;
let currentJobId = null;

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  resetUI();
  submitBtn.disabled = true;
  submitBtn.textContent = "处理中...";

  try {
    const res = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `请求失败 (${res.status})`);
    }
    const { job_id } = await res.json();
    currentJobId = job_id;
    progressSection.classList.remove("hidden");
    pollJob(job_id);
    loadHistory();
  } catch (err) {
    showError(err.message);
    submitBtn.disabled = false;
    submitBtn.textContent = "开始检索";
  }
});

confirmSelectionBtn.addEventListener("click", async () => {
  const checked = [...selectionList.querySelectorAll("input[type=checkbox]:checked")].map(
    (el) => el.value
  );
  if (checked.length === 0) {
    showError("至少要选一篇论文");
    return;
  }
  confirmSelectionBtn.disabled = true;
  confirmSelectionBtn.textContent = "提交中...";
  try {
    const res = await fetch(`/api/jobs/${currentJobId}/select`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paper_ids: checked }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `请求失败 (${res.status})`);
    }
    selectionBox.classList.add("hidden");
  } catch (err) {
    showError(err.message);
    confirmSelectionBtn.disabled = false;
    confirmSelectionBtn.textContent = "确认，下载并生成综述";
  }
});

toggleEvidenceBtn.addEventListener("click", () => {
  const isHidden = evidenceList.classList.contains("hidden");
  evidenceList.classList.toggle("hidden");
  toggleEvidenceBtn.textContent = isHidden ? "隐藏引用证据片段 ▴" : "显示引用证据片段 ▾";
});

function resetUI() {
  progressSection.classList.remove("hidden");
  resultSection.classList.add("hidden");
  statsBox.textContent = "";
  errorBox.classList.add("hidden");
  candidatesBox.classList.add("hidden");
  candidatesList.innerHTML = "";
  selectionBox.classList.add("hidden");
  selectionList.innerHTML = "";
  citationCheckBox.classList.add("hidden");
  citationCheckList.innerHTML = "";
  evidenceBox.classList.add("hidden");
  evidenceList.innerHTML = "";
  evidenceList.classList.add("hidden");
  toggleEvidenceBtn.textContent = "显示引用证据片段 ▾";
  logBox.innerHTML = "";
  renderedLogCount = 0;
  selectionRenderedForJob = null;
  confirmSelectionBtn.disabled = false;
  confirmSelectionBtn.textContent = "确认，下载并生成综述";
  document.querySelectorAll(".step").forEach((el) => {
    el.classList.remove("active", "complete");
  });
  if (pollTimer) clearTimeout(pollTimer);
}

function pollJob(jobId) {
  const tick = async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (!res.ok) throw new Error("任务查询失败");
      const job = await res.json();
      renderJob(job);

      if (job.status === "done" || job.status === "failed") {
        submitBtn.disabled = false;
        submitBtn.textContent = "开始检索";
        loadHistory();
        return;
      }
      pollTimer = setTimeout(tick, 1500);
    } catch (err) {
      showError(err.message);
      submitBtn.disabled = false;
      submitBtn.textContent = "开始检索";
    }
  };
  tick();
}

function renderJob(job) {
  updateSteps(job.status);

  const newLogs = job.logs.slice(renderedLogCount);
  for (const entry of newLogs) {
    const line = document.createElement("div");
    line.className = "log-line";
    line.textContent = entry.message;
    logBox.appendChild(line);
  }
  renderedLogCount = job.logs.length;
  logBox.scrollTop = logBox.scrollHeight;

  if (job.status === "awaiting_selection") {
    renderSelection(job);
  }

  if (job.downloaded && job.downloaded.length > 0) {
    candidatesBox.classList.remove("hidden");
    candidatesList.innerHTML = "";
    for (const c of job.downloaded) {
      const li = document.createElement("li");
      const link = c.landing_url || c.pdf_url || "#";
      li.innerHTML = `<a href="${escapeHtml(link)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a> <span style="color:var(--muted)">(${escapeHtml(c.source)})</span>`;
      candidatesList.appendChild(li);
    }
  }

  if (job.status === "done") {
    resultSection.classList.remove("hidden");
    renderStats(job);
    answerBox.textContent = job.answer || "(无内容)";
    renderCitationCheck(job.citation_flags, job.citation_check_failed);
    referencesBox.textContent = job.references
      ? "参考文献:\n" + job.references
      : "";
    renderEvidence(job.evidence || []);
  } else if (job.status === "failed") {
    showError(job.error || "任务失败，原因未知");
  }
}

function renderSelection(job) {
  if (selectionRenderedForJob === job.id) return;
  selectionRenderedForJob = job.id;

  selectionBox.classList.remove("hidden");
  defaultCheckedCountEl.textContent = String(Math.min(DEFAULT_CHECKED_COUNT, job.candidates.length));
  selectionList.innerHTML = "";

  job.candidates.forEach((c, idx) => {
    const li = document.createElement("li");
    const checked = idx < DEFAULT_CHECKED_COUNT ? "checked" : "";
    const link = c.landing_url || c.pdf_url || "#";
    const abstract = (c.abstract || "").slice(0, 220);
    const hasScore = c.relevance_score != null;
    const scoreClass = hasScore ? (c.relevance_score >= 7 ? "high" : c.relevance_score >= 4 ? "mid" : "low") : "";
    const scoreBadge = hasScore
      ? `<span class="relevance-badge relevance-${scoreClass}" title="${escapeHtml(c.relevance_reason || "")}">相关性 ${escapeHtml(String(c.relevance_score))}/10</span>`
      : "";
    li.innerHTML = `
      <label>
        <input type="checkbox" value="${escapeHtml(c.id)}" ${checked} />
        <div class="selection-item-body">
          ${scoreBadge}
          <a href="${escapeHtml(link)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>
          <span class="meta">${escapeHtml(c.source)}${c.year ? " · " + c.year : ""}</span>
          ${c.relevance_reason ? `<p class="relevance-reason">${escapeHtml(c.relevance_reason)}</p>` : ""}
          <p class="abstract">${escapeHtml(abstract)}${abstract.length >= 220 ? "…" : ""}</p>
        </div>
      </label>
    `;
    selectionList.appendChild(li);
  });
}

function renderStats(job) {
  const parts = [];
  if (job.duration != null) parts.push(`耗时 ${job.duration.toFixed(1)}s`);
  if (job.cost != null) parts.push(`花费 $${job.cost.toFixed(4)}`);
  if (job.total_tokens != null) parts.push(`${job.total_tokens.toLocaleString()} tokens`);
  statsBox.textContent = parts.join("  ·  ");
  statsBox.classList.toggle("hidden", parts.length === 0);
}

const PROBLEM_LABELS = {
  no_matching_source: "来源缺失",
  not_supported: "证据不支撑",
};

function renderCitationCheck(flags, checkFailed) {
  // Three distinct states, not two: never ran (older job — stay hidden,
  // don't claim anything), ran but every attempt failed (show that
  // explicitly instead of looking identical to "never ran"), and ran
  // successfully (flags is a list, possibly empty).
  if (checkFailed) {
    citationCheckBox.classList.remove("hidden", "ok", "warn");
    citationCheckBox.classList.add("failed");
    citationCheckTitle.textContent = "⚠ 引用自查未能完成（不影响已生成的综述，具体原因见上方日志）";
    citationCheckList.innerHTML = "";
    return;
  }

  if (flags == null) {
    citationCheckBox.classList.add("hidden");
    return;
  }

  citationCheckBox.classList.remove("hidden", "failed");
  citationCheckList.innerHTML = "";

  if (flags.length === 0) {
    citationCheckBox.classList.remove("warn");
    citationCheckBox.classList.add("ok");
    citationCheckTitle.textContent = "✓ 引用自查：未发现可疑引用";
    return;
  }

  citationCheckBox.classList.remove("ok");
  citationCheckBox.classList.add("warn");
  citationCheckTitle.textContent = `⚠ 引用自查：发现 ${flags.length} 处可疑引用`;
  for (const f of flags) {
    const li = document.createElement("li");
    const sourceLabel =
      f.detected_by === "deterministic" ? "确定性检查 · 字符串比对" : "LLM 判断";
    li.innerHTML = `
      <div class="citation-check-meta">
        <span class="citation-check-problem">${escapeHtml(PROBLEM_LABELS[f.problem] || f.problem)}</span>
        <span class="citation-check-source" title="检测方式">${escapeHtml(sourceLabel)}</span>
        · 引用为 ${escapeHtml(f.cited_as || "(未标注)")}
      </div>
      <div class="citation-check-claim">"${escapeHtml(f.claim)}"</div>
      <div class="citation-check-explanation">${escapeHtml(f.explanation)}</div>
    `;
    citationCheckList.appendChild(li);
  }
}

function renderEvidence(evidence) {
  if (!evidence.length) {
    evidenceBox.classList.add("hidden");
    return;
  }
  evidenceBox.classList.remove("hidden");
  evidenceList.innerHTML = "";
  for (const e of evidence) {
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="evidence-meta">${escapeHtml(e.source)} · score ${escapeHtml(String(e.score))}</div>
      <div class="evidence-context">${escapeHtml(e.context)}</div>
      <div class="evidence-citation">${escapeHtml(e.citation)}</div>
    `;
    evidenceList.appendChild(li);
  }
}

function updateSteps(status) {
  const currentIndex = STEP_ORDER.indexOf(status);
  document.querySelectorAll(".step").forEach((el) => {
    const step = el.dataset.step;
    const stepIndex = STEP_ORDER.indexOf(step);
    el.classList.remove("active", "complete");
    if (status === "failed") return;
    if (stepIndex < currentIndex || status === "done") {
      el.classList.add("complete");
    } else if (stepIndex === currentIndex) {
      el.classList.add("active");
    }
  });
}

const STATUS_LABELS = {
  queued: "排队中",
  searching: "搜索中",
  awaiting_selection: "等待选择论文",
  downloading: "下载中",
  generating: "生成中",
  done: "完成",
  failed: "失败",
};

async function loadHistory() {
  try {
    const res = await fetch("/api/jobs");
    if (!res.ok) return;
    const jobs = await res.json();
    renderHistory(jobs);
  } catch {
    // history is a nice-to-have; a failed refresh shouldn't disrupt the main flow
  }
}

function renderHistory(jobs) {
  historyList.innerHTML = "";
  if (jobs.length === 0) {
    const li = document.createElement("li");
    li.className = "history-empty";
    li.textContent = "还没有历史任务";
    historyList.appendChild(li);
    return;
  }
  for (const job of jobs) {
    const li = document.createElement("li");
    li.className = "history-item";
    if (job.id === currentJobId) li.classList.add("active");
    const time = new Date(job.created_at * 1000).toLocaleString();
    const isTerminal = job.status === "done" || job.status === "failed";
    li.innerHTML = `
      <span class="history-status status-${escapeHtml(job.status)}">${escapeHtml(STATUS_LABELS[job.status] || job.status)}</span>
      <span class="history-query">${escapeHtml(job.query)}</span>
      <span class="history-time">${escapeHtml(time)}</span>
      ${isTerminal ? '<button type="button" class="history-delete-btn" title="删除">×</button>' : ""}
    `;
    li.addEventListener("click", () => loadJob(job.id));
    if (isTerminal) {
      li.querySelector(".history-delete-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        deleteJob(job.id);
      });
    }
    historyList.appendChild(li);
  }
}

async function deleteJob(jobId) {
  if (!confirm("确定要删除这条历史任务记录吗？（不会删除已下载的论文缓存）")) return;
  try {
    const res = await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `删除失败 (${res.status})`);
    }
    if (jobId === currentJobId) {
      currentJobId = null;
      progressSection.classList.add("hidden");
      resultSection.classList.add("hidden");
    }
    loadHistory();
  } catch (err) {
    showError(err.message);
  }
}

function loadJob(jobId) {
  currentJobId = jobId;
  resetUI();
  submitBtn.disabled = true;
  submitBtn.textContent = "处理中...";
  pollJob(jobId);
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

loadHistory();
