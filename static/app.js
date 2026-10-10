(() => {
  const STORAGE_KEY = "soundstage_access_token";
  const KNOWN_JOBS_KEY = "soundstage_known_jobs";
  const LEGACY_STORAGE_KEY = "foley_omni_access_token";
  const LEGACY_KNOWN_JOBS_KEY = "foley_omni_known_jobs";

  const els = {
    gate: document.getElementById("gate"),
    gateSubmit: document.getElementById("gate-submit"),
    tokenInput: document.getElementById("token-input"),
    gateError: document.getElementById("gate-error"),
    appShell: document.getElementById("app-shell"),
    appMain: document.getElementById("app-main"),
    sessionBar: document.getElementById("session-bar"),
    roleBadge: document.getElementById("role-badge"),
    logoutBtn: document.getElementById("logout-btn"),
    jobForm: document.getElementById("job-form"),
    formError: document.getElementById("form-error"),
    submitBtn: document.getElementById("submit-btn"),
    jobsList: document.getElementById("jobs-list"),
    healthLine: document.getElementById("health-line"),
    adminPanel: document.getElementById("admin-panel"),
    clearQueueBtn: document.getElementById("clear-queue-btn"),
    adminMsg: document.getElementById("admin-msg"),
    videoInput: document.getElementById("video-input"),
    dropzone: document.getElementById("dropzone"),
    dropzoneFile: document.getElementById("dropzone-file"),
    dropzoneTitle: document.getElementById("dropzone-title"),
  };

  let token = localStorage.getItem(STORAGE_KEY) || localStorage.getItem(LEGACY_STORAGE_KEY) || "";
  if (token && !localStorage.getItem(STORAGE_KEY)) {
    localStorage.setItem(STORAGE_KEY, token);
    localStorage.removeItem(LEGACY_STORAGE_KEY);
  }
  if (!localStorage.getItem(KNOWN_JOBS_KEY) && localStorage.getItem(LEGACY_KNOWN_JOBS_KEY)) {
    localStorage.setItem(KNOWN_JOBS_KEY, localStorage.getItem(LEGACY_KNOWN_JOBS_KEY));
    localStorage.removeItem(LEGACY_KNOWN_JOBS_KEY);
  }
  let role = null;
  let pollTimer = null;
  const jobCache = new Map();
  const blobUrlCache = new Map();

  const EMPTY_REEL = `
    <div class="reel-empty">
      <p class="reel-empty__title">Nothing generating yet</p>
      <p>Your soundtrack clips will appear here.</p>
    </div>
  `;

  function knownJobIds() {
    try {
      return JSON.parse(localStorage.getItem(KNOWN_JOBS_KEY) || "[]");
    } catch {
      return [];
    }
  }

  function rememberJob(jobId) {
    const ids = knownJobIds();
    if (!ids.includes(jobId)) {
      ids.unshift(jobId);
      localStorage.setItem(KNOWN_JOBS_KEY, JSON.stringify(ids.slice(0, 30)));
    }
  }

  function headers() {
    return { "X-API-Key": token };
  }

  async function api(path, options = {}) {
    const res = await fetch(path, {
      ...options,
      headers: {
        ...headers(),
        ...(options.headers || {}),
      },
    });
    let body = null;
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      body = await res.json();
    }
    if (!res.ok) {
      const detail = body && body.detail ? body.detail : res.statusText;
      const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      err.status = res.status;
      throw err;
    }
    return body;
  }

  function show(el) {
    if (el) el.classList.remove("hidden");
  }

  function hide(el) {
    if (el) el.classList.add("hidden");
  }

  function setError(el, msg) {
    if (!msg) {
      hide(el);
      el.textContent = "";
      return;
    }
    el.textContent = msg;
    show(el);
  }

  function setAuthenticated(on) {
    document.body.classList.toggle("is-authenticated", on);
    if (on) {
      hide(els.gate);
      show(els.appShell);
    } else {
      show(els.gate);
      hide(els.appShell);
    }
  }

  function syncDropzone() {
    const file = els.videoInput.files && els.videoInput.files[0];
    if (file) {
      els.dropzone.classList.add("is-filled");
      els.dropzoneTitle.textContent = "Video selected";
      els.dropzoneFile.textContent = file.name;
      show(els.dropzoneFile);
    } else {
      els.dropzone.classList.remove("is-filled");
      els.dropzoneTitle.textContent = "Drop a video";
      els.dropzoneFile.textContent = "";
      hide(els.dropzoneFile);
    }
  }

  async function refreshHealth() {
    try {
      const h = await fetch("/api/health").then((r) => r.json());
      const busy = h.gpu_busy ? "GPU busy" : "GPU idle";
      const model = h.model_loaded ? "model ready" : "model loading…";
      els.healthLine.textContent = `${model} · ${busy} · queue ${h.queue_depth}/${h.max_queue_size}`;
    } catch {
      els.healthLine.textContent = "Health unavailable";
    }
  }

  async function enterApp() {
    const me = await api("/api/me");
    role = me.role;
    setAuthenticated(true);
    els.roleBadge.textContent = role === "guest" ? "Guest" : role === "admin" ? "Admin" : role;
    els.roleBadge.className = `role ${role}`;
    if (role === "admin") show(els.adminPanel);
    else hide(els.adminPanel);
    await refreshHealth();
    await refreshJobs();
    startPolling();
  }

  async function tryRestoreSession() {
    if (!token) return;
    try {
      await enterApp();
    } catch {
      token = "";
      localStorage.removeItem(STORAGE_KEY);
    }
  }

  els.gateSubmit.addEventListener("click", async () => {
    setError(els.gateError, "");
    token = (els.tokenInput.value || "").trim();
    if (!token) {
      setError(els.gateError, "Token is required");
      return;
    }
    els.gateSubmit.disabled = true;
    try {
      localStorage.setItem(STORAGE_KEY, token);
      await enterApp();
    } catch (err) {
      localStorage.removeItem(STORAGE_KEY);
      token = "";
      setError(els.gateError, err.message || "Invalid token");
    } finally {
      els.gateSubmit.disabled = false;
    }
  });

  els.tokenInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") els.gateSubmit.click();
  });

  els.logoutBtn.addEventListener("click", () => {
    token = "";
    role = null;
    localStorage.removeItem(STORAGE_KEY);
    stopPolling();
    setAuthenticated(false);
    els.tokenInput.value = "";
    hide(els.adminPanel);
  });

  els.videoInput.addEventListener("change", syncDropzone);

  ["dragenter", "dragover"].forEach((evt) => {
    els.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      els.dropzone.classList.add("is-dragover");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    els.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      els.dropzone.classList.remove("is-dragover");
    });
  });
  els.dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer && e.dataTransfer.files;
    if (!files || !files.length) return;
    const dt = new DataTransfer();
    dt.items.add(files[0]);
    els.videoInput.files = dt.files;
    syncDropzone();
  });

  els.jobForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    setError(els.formError, "");
    const video = els.videoInput.files[0];
    if (!video) {
      setError(els.formError, "Choose a video file");
      return;
    }
    const words = document.getElementById("words").value.trim();
    const audioCaption = document.getElementById("audio-caption").value.trim();
    const music = document.getElementById("music").value.trim();
    if (!words && !audioCaption && !music) {
      setError(els.formError, "Provide at least one of Words, Audio caption, or Music");
      return;
    }

    const fd = new FormData();
    fd.append("video", video);
    fd.append("words", words);
    fd.append("audio_caption", audioCaption);
    fd.append("music", music);
    fd.append("seed", document.getElementById("seed").value || "103");
    fd.append("sample_steps", document.getElementById("sample-steps").value || "50");

    els.submitBtn.disabled = true;
    try {
      const res = await fetch("/api/jobs", {
        method: "POST",
        headers: { "X-API-Key": token },
        body: fd,
      });
      const body = await res.json();
      if (!res.ok) {
        throw new Error(body.detail || res.statusText);
      }
      rememberJob(body.job_id);
      jobCache.set(body.job_id, body);
      await refreshJobs();
      await refreshHealth();
      els.jobForm.reset();
      document.getElementById("seed").value = "103";
      document.getElementById("sample-steps").value = "50";
      syncDropzone();
    } catch (err) {
      setError(els.formError, err.message || "Submit failed");
    } finally {
      els.submitBtn.disabled = false;
    }
  });

  els.clearQueueBtn.addEventListener("click", async () => {
    if (!confirm("Cancel all queued jobs? The running job will keep going.")) return;
    els.clearQueueBtn.disabled = true;
    els.adminMsg.textContent = "";
    try {
      const res = await api("/api/admin/queue/clear", { method: "POST" });
      els.adminMsg.textContent = `Cancelled ${res.cancelled_count} queued job(s).`;
      await refreshJobs();
      await refreshHealth();
    } catch (err) {
      els.adminMsg.textContent = err.message || "Clear failed";
    } finally {
      els.clearQueueBtn.disabled = false;
    }
  });

  async function fetchJob(jobId) {
    const data = await api(`/api/jobs/${jobId}`);
    jobCache.set(jobId, data);
    return data;
  }

  async function refreshJobs() {
    const ids = knownJobIds();
    if (!ids.length) {
      els.jobsList.innerHTML = EMPTY_REEL;
      return;
    }
    const jobs = [];
    for (const id of ids) {
      try {
        jobs.push(await fetchJob(id));
      } catch (err) {
        if (err.status === 403 || err.status === 404) {
          const remaining = knownJobIds().filter((x) => x !== id);
          localStorage.setItem(KNOWN_JOBS_KEY, JSON.stringify(remaining));
        }
      }
    }
    renderJobs(jobs);
  }

  function renderJobs(jobs) {
    if (!jobs.length) {
      els.jobsList.innerHTML = EMPTY_REEL;
      return;
    }

    const existing = new Map();
    els.jobsList.querySelectorAll(".job[data-job-id]").forEach((node) => {
      existing.set(node.getAttribute("data-job-id"), node);
    });

    const frag = document.createDocumentFragment();
    for (const job of jobs) {
      let el = existing.get(job.job_id);
      const signature = `${job.status}|${job.progress || ""}|${job.position || ""}|${job.error || ""}|${job.result_ready}`;
      if (el && el.dataset.signature === signature) {
        frag.appendChild(el);
        existing.delete(job.job_id);
        continue;
      }

      el = document.createElement("article");
      el.className = "job";
      el.dataset.jobId = job.job_id;
      el.dataset.signature = signature;

      const pos =
        job.status === "queued" && job.position != null
          ? ` · #${job.position} in queue`
          : "";
      const progress = job.progress
        ? `<p class="meta">${escapeHtml(job.progress)}${escapeHtml(pos)}</p>`
        : "";
      const err = job.error ? `<p class="error">${escapeHtml(job.error)}</p>` : "";

      let actions = "";
      if (job.status === "queued") {
        actions += `<button type="button" class="btn btn--ghost btn--small" data-cancel="${job.job_id}">Remove from queue</button>`;
      }
      if (job.status === "done" && job.result_ready) {
        actions += `<button type="button" class="btn btn--primary btn--small" data-download="${job.job_id}">Download MP4</button>`;
      }

      el.innerHTML = `
        <div class="job-top">
          <div>
            <div class="job-id">${escapeHtml(job.job_id)}</div>
            <div class="status ${job.status}">${escapeHtml(job.status)}</div>
          </div>
        </div>
        ${progress}
        ${err}
        <div class="job-actions">${actions}</div>
        <div class="player" data-player="${job.job_id}"></div>
      `;
      frag.appendChild(el);

      if (job.status === "done" && job.result_ready) {
        const player = el.querySelector(`[data-player="${job.job_id}"]`);
        const video = document.createElement("video");
        video.controls = true;
        video.playsInline = true;
        if (blobUrlCache.has(job.job_id)) {
          video.src = blobUrlCache.get(job.job_id);
        } else {
          loadVideoBlob(job.job_id, video);
        }
        player.appendChild(video);
      }
    }

    els.jobsList.innerHTML = "";
    els.jobsList.appendChild(frag);

    els.jobsList.querySelectorAll("[data-cancel]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.getAttribute("data-cancel");
        btn.disabled = true;
        try {
          await api(`/api/jobs/${id}`, { method: "DELETE" });
          await refreshJobs();
          await refreshHealth();
        } catch (err) {
          alert(err.message || "Cancel failed");
          btn.disabled = false;
        }
      });
    });

    els.jobsList.querySelectorAll("[data-download]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.getAttribute("data-download");
        try {
          const blob = blobUrlCache.has(id)
            ? await fetch(blobUrlCache.get(id)).then((r) => r.blob())
            : await fetchResultBlob(id);
          const url = URL.createObjectURL(blob);
          const tmp = document.createElement("a");
          tmp.href = url;
          tmp.download = `soundstage-${id}.mp4`;
          tmp.click();
          URL.revokeObjectURL(url);
        } catch (err) {
          alert(err.message || "Download failed");
        }
      });
    });
  }

  async function fetchResultBlob(jobId) {
    const res = await fetch(`/api/jobs/${jobId}/result`, { headers: headers() });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || detail;
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }
    return res.blob();
  }

  async function loadVideoBlob(jobId, videoEl) {
    try {
      const blob = await fetchResultBlob(jobId);
      const url = URL.createObjectURL(blob);
      blobUrlCache.set(jobId, url);
      videoEl.src = url;
    } catch {
      videoEl.replaceWith(
        Object.assign(document.createElement("p"), {
          className: "error",
          textContent: "Could not load result preview",
        })
      );
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function startPolling() {
    stopPolling();
    pollTimer = setInterval(async () => {
      await refreshJobs();
      await refreshHealth();
    }, 2500);
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  tryRestoreSession();
})();
