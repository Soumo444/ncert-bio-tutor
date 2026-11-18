(() => {
  "use strict";

  const READ_ONLY = window.APP_READ_ONLY === true;

  const state = {
    history: [],
    busy: false,
    docs: [],
    health: null,
    quality: "fast",
    mode: (document.querySelector("#mode .active") || { dataset: {} }).dataset.mode || "auto",
    emptyHTML: "",
  };

  const el = {
    messages: document.getElementById("messages"),
    emptyState: document.getElementById("emptyState"),
    emptyTitle: document.getElementById("emptyTitle"),
    emptyText: document.getElementById("emptyText"),
    composer: document.getElementById("composer"),
    question: document.getElementById("question"),
    sendBtn: document.getElementById("sendBtn"),
    topK: document.getElementById("topK"),
    topKVal: document.getElementById("topKVal"),
    mode: document.getElementById("mode"),
    modeHint: document.getElementById("modeHint"),
    quality: document.getElementById("quality"),
    qualityHint: document.getElementById("qualityHint"),
    verifyToggle: document.getElementById("verifyToggle"),
    verifyHint: document.getElementById("verifyHint"),
    docList: document.getElementById("docList"),
    docCount: document.getElementById("docCount"),
    dropzone: document.getElementById("dropzone"),
    fileInput: document.getElementById("fileInput"),
    uploadProgress: document.getElementById("uploadProgress"),
    uploadText: document.getElementById("uploadText"),
    reindexBtn: document.getElementById("reindexBtn"),
    statusDot: document.getElementById("statusDot"),
    statusText: document.getElementById("statusText"),
    modelInfo: document.getElementById("modelInfo"),
    providerPill: document.getElementById("providerPill"),
    chatSub: document.getElementById("chatSub"),
    newChatBtn: document.getElementById("newChatBtn"),
    sidebar: document.getElementById("sidebar"),
    sidebarToggle: document.getElementById("sidebarToggle"),
    overlay: document.getElementById("overlay"),
    composerNote: document.getElementById("composerNote"),
  };

  // ------------------------------------------------------------------ utils
  const escapeHtml = (value) =>
    value.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function inlineMarkdown(text) {
    return text
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/[\[【](\d{1,2})[\]】]/g, '<span class="cite" title="Source $1">$1</span>');
  }

  function renderMarkdown(raw) {
    const lines = escapeHtml(raw || "").split("\n");
    let html = "";
    let list = null;
    let paragraph = [];
    let table = null;

    const flushParagraph = () => {
      if (paragraph.length) {
        html += `<p>${inlineMarkdown(paragraph.join(" "))}</p>`;
        paragraph = [];
      }
    };
    const closeList = () => {
      if (list) {
        html += `</${list}>`;
        list = null;
      }
    };
    const flushTable = () => {
      if (!table) return;
      const head = `<thead><tr>${table.header
        .map((cell) => `<th>${inlineMarkdown(cell)}</th>`)
        .join("")}</tr></thead>`;
      const body = table.rows.length
        ? `<tbody>${table.rows
            .map((row) => `<tr>${row
              .map((cell) => `<td>${inlineMarkdown(cell)}</td>`)
              .join("")}</tr>`)
            .join("")}</tbody>`
        : "";
      html += `<div class="table-wrap"><table>${head}${body}</table></div>`;
      table = null;
    };

    const isTableRow = (line) => line.startsWith("|") && line.endsWith("|");
    const isTableDivider = (line) => /^\|?[\s:|-]+\|?$/.test(line) && line.includes("-");
    const splitRow = (line) => line.slice(1, -1).split("|").map((cell) => cell.trim());

    for (const line of lines) {
      const trimmed = line.trim();

      if (isTableRow(trimmed)) {
        if (table && isTableDivider(trimmed)) continue;
        if (!table) {
          flushParagraph();
          closeList();
          table = { header: splitRow(trimmed), rows: [] };
        } else {
          table.rows.push(splitRow(trimmed));
        }
        continue;
      }
      flushTable();

      if (!trimmed) {
        flushParagraph();
        closeList();
        continue;
      }
      let match;
      if ((match = trimmed.match(/^[-*\u2022]\s+(.*)$/))) {
        flushParagraph();
        if (list !== "ul") {
          closeList();
          html += "<ul>";
          list = "ul";
        }
        html += `<li>${inlineMarkdown(match[1])}</li>`;
      } else if ((match = trimmed.match(/^\d+[.)]\s+(.*)$/))) {
        flushParagraph();
        if (list !== "ol") {
          closeList();
          html += "<ol>";
          list = "ol";
        }
        html += `<li>${inlineMarkdown(match[1])}</li>`;
      } else if ((match = trimmed.match(/^#{1,6}\s+(.*)$/))) {
        flushParagraph();
        closeList();
        html += `<h4>${inlineMarkdown(match[1])}</h4>`;
      } else {
        closeList();
        paragraph.push(trimmed);
      }
    }
    flushTable();
    flushParagraph();
    closeList();
    return html || "<p></p>";
  }

  const iconFor = (name) => {
    const extension = (name.split(".").pop() || "").toLowerCase();
    const label = { pdf: "PDF", docx: "DOC", txt: "TXT", md: "MD", csv: "CSV", xlsx: "XLS", xlsm: "XLS" }[extension] || "DOC";
    return { label, cls: extension };
  };

  const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`);

  async function api(path, options = {}) {
    const response = await fetch(path, options);
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const data = await response.json();
        detail = data.detail || detail;
      } catch (_) { /* ignore */ }
      throw new Error(detail);
    }
    return response.json();
  }

  // ------------------------------------------------------------- mode switch
  function updateModeUi() {
    const auto = state.mode === "auto";
    if (el.modeHint) {
      el.modeHint.textContent = auto
        ? "Documents first · general knowledge when needed"
        : "Strict · refuses when the files don’t cover it";
    }
    if (el.verifyHint) {
      el.verifyHint.textContent = auto ? "runs in Docs-only mode" : "second model call";
    }
    if (el.verifyToggle) {
      el.verifyToggle.disabled = auto;
    }
  }

  // ------------------------------------------------------------- health/docs
  async function refreshHealth() {
    try {
      const health = await api("/api/health");
      state.health = health;
      const ready = health.any_provider_ready;
      const hasDocs = health.chunk_count > 0;

      el.statusDot.className = "dot " + (!ready ? "bad" : hasDocs ? "ok" : "warn");
      const docCount = health.document_count || 0;
      el.statusText.textContent = !ready
        ? "No LLM provider configured"
        : hasDocs
          ? `${health.chunk_count} chunks · ${docCount} docs`
          : "Ready · no documents";

      el.modelInfo.textContent = `${health.llm_provider} · ${health.llm_model}`;
      el.providerPill.hidden = false;
      el.providerPill.textContent = health.llm_provider;
      if (el.qualityHint) {
        const models = health.answer_models || {};
        el.qualityHint.textContent = `${state.quality} · ${models[state.quality] || health.llm_model}`;
      }
      el.chatSub.textContent = hasDocs
        ? `${health.chunk_count} chunks indexed from ${state.docs.length || "your"} document(s)`
        : "Upload documents to start asking questions.";

      if (!ready) {
        el.composerNote.textContent = "Add a free API key in .env";
      } else if (state.mode === "auto") {
        el.composerNote.textContent = "Documents first · general knowledge labelled";
      } else {
        el.composerNote.textContent = "Strict mode · answers only from your documents";
      }
      updateModeUi();
    } catch (error) {
      el.statusDot.className = "dot bad";
      el.statusText.textContent = "Server unreachable";
    }
  }

  async function refreshDocs() {
    try {
      const data = await api("/api/documents");
      state.docs = data.documents || [];
      renderDocs();
      updateBanners();
      updateEmpty();
      if (state.health) refreshHealth();
    } catch (error) {
      console.error(error);
    }
  }

  function renderDocs() {
    el.docCount.textContent = state.docs.length;
    el.docList.innerHTML = "";
    for (const doc of state.docs) {
      const { label, cls } = iconFor(doc.source);
      const item = document.createElement("li");
      item.className = "doc-item";
      item.innerHTML = `
        <span class="doc-icon ${cls}">${label}</span>
        <span class="doc-meta">
          <span class="doc-name" title="${escapeHtml(doc.source)}">${escapeHtml(doc.source)}</span>
          <span class="doc-sub">${doc.chunks} chunks${doc.pages ? ` · ${doc.pages} pages` : ""}</span>
        </span>
        ${READ_ONLY ? "" : `<button class="doc-delete" type="button" title="Remove document" data-delete="${escapeHtml(doc.source)}">×</button>`}`;
      el.docList.appendChild(item);
    }
  }

  function updateEmpty() {
    if (!el.emptyState) return;
    const count = state.docs.length;
    const health = state.health;
    if (count === 0) {
      el.emptyTitle.textContent = "No documents indexed yet";
      el.emptyText.textContent =
        "Drop a PDF, Word file, text file, Markdown or spreadsheet into the upload box. It is chunked, embedded and stored locally — then ask questions about it.";
    } else if (health && health.chunk_count) {
      el.emptyTitle.textContent = "Ask your first question";
      el.emptyText.textContent = `${health.chunk_count} chunks from ${count} document(s) are ready. When they cover your question the answer cites file and page; otherwise you get a normal answer, labelled as general knowledge.`;
    }
  }

  function updateBanners() {
    const existing = el.messages.querySelector(".banner");
    if (existing) existing.remove();

    const health = state.health;
    if (!health) return;

    let banner = null;
    if (!health.any_provider_ready) {
      banner = document.createElement("div");
      banner.className = "banner bad";
      banner.innerHTML =
        "No LLM key found. Open <code>.env</code> and paste a free <code>GEMINI_API_KEY</code> " +
        'from <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer">aistudio.google.com/apikey</a>, then restart the server.';
    } else if (state.docs.length === 0) {
      banner = document.createElement("div");
      banner.className = "banner";
      banner.innerHTML = "Your index is empty. Upload a document (PDF, DOCX, TXT, MD, CSV, XLSX) to start chatting.";
    }
    if (banner) el.messages.insertBefore(banner, el.messages.firstChild);
  }

  // ------------------------------------------------------------------- chat
  function showSidebar(open) {
    el.sidebar.classList.toggle("open", open);
    el.overlay.classList.toggle("show", open);
  }

  function clearEmpty() {
    if (el.emptyState && el.emptyState.parentElement) {
      el.emptyState.remove();
    }
  }

  function addUserMessage(text) {
    clearEmpty();
    const article = document.createElement("article");
    article.className = "msg user";
    article.innerHTML = `
      <div class="avatar">YOU</div>
      <div class="msg-body"><div class="bubble">${escapeHtml(text)}</div></div>`;
    el.messages.appendChild(article);
    scrollToBottom();
  }

  function addAssistantShell() {
    const article = document.createElement("article");
    article.className = "msg assistant";
    article.innerHTML = `
      <div class="avatar">AI</div>
      <div class="msg-body">
        <div class="bubble">
          <div class="answer"><span class="typing"><i></i><i></i><i></i></span></div>
        </div>
        <div class="msg-meta" hidden></div>
        <details class="sources" hidden>
          <summary><span class="chev">▶</span><span class="sources-label">Sources</span></summary>
          <div class="source-grid"></div>
        </details>
      </div>`;
    el.messages.appendChild(article);
    return {
      article,
      answer: article.querySelector(".answer"),
      meta: article.querySelector(".msg-meta"),
      sources: article.querySelector(".sources"),
      sourcesLabel: article.querySelector(".sources-label"),
      sourceGrid: article.querySelector(".source-grid"),
    };
  }

  function renderSources(view, sources) {
    if (!sources || !sources.length) return;
    view.sources.hidden = false;
    view.sourcesLabel.textContent = `${sources.length} source${sources.length === 1 ? "" : "s"} retrieved`;
    view.sourceGrid.innerHTML = sources
      .map((source) => {
        const where = [
          source.page ? `p.${source.page}` : null,
          source.section ? source.section : null,
          source.row_start ? `rows ${source.row_start}-${source.row_end}` : null,
        ].filter(Boolean).join(" · ");
        const score = Math.round((source.score || 0) * 100);
        return `
          <div class="source-card">
            <div class="source-top">
              <span class="source-num">${source.n}</span>
              <span class="source-file" title="${escapeHtml(source.source)}">${escapeHtml(source.source)}</span>
              ${where ? `<span class="source-where">${escapeHtml(where)}</span>` : ""}
              <span class="source-score"><span class="score-bar"><i style="width:${score}%"></i></span>${score}%</span>
            </div>
            <p class="source-preview">${escapeHtml(source.preview)}</p>
          </div>`;
      })
      .join("");
  }

  function renderMeta(view, payload, rawText) {
    view.meta.hidden = false;
    const badges = [];

    if (payload.mode === "auto") {
      if (payload.citations && payload.citations.length) {
        const files = [...new Set(payload.citations.map((c) => c.source))];
        badges.push(
          `<span class="badge ok">from your documents · ${files.length} file${files.length === 1 ? "" : "s"}</span>`
        );
      }
      if (payload.general_knowledge) {
        badges.push('<span class="badge ghost">general knowledge · may be out of date</span>');
      }
    }
    if (payload.quiz) {
      badges.push('<span class="badge ghost">practice question</span>');
    }
    if (payload.grading) {
      badges.push('<span class="badge ghost">answer graded</span>');
    }

    if (payload.refused) {
      view.article.classList.add("refused");
      badges.push('<span class="badge warn">no answer in documents</span>');
    } else if (payload.groundedness) {
      const score = Math.round((payload.groundedness.faithfulness || 0) * 100);
      const support = payload.groundedness.context_support;
      const cls = score >= 85 ? "ok" : score >= 60 ? "warn" : "bad";
      const fill = Math.round((payload.groundedness.faithfulness || 0) * 100);
      badges.push(
        `<span class="badge ${cls}" title="${escapeHtml(payload.groundedness.notes || "")}">grounded ${score}%<span class="score-fill"><i style="width:${fill}%"></i></span></span>`
      );
      if (support && support !== "supported") {
        badges.push(`<span class="badge warn">${escapeHtml(support.replace(/_/g, " "))}</span>`);
      }
    }

    if (payload.mode !== "auto" && payload.citations && payload.citations.length) {
      const files = [...new Set(payload.citations.map((c) => c.source))];
      badges.push(`<span class="badge ghost">${payload.citations.length} citation(s) · ${files.length} file(s)</span>`);
    }
    if (payload.provider) {
      const who = payload.model ? `${payload.provider} · ${payload.model}` : payload.provider;
      badges.push(`<span class="badge ghost">${escapeHtml(who)}</span>`);
    }
    if (payload.total_ms) {
      badges.push(`<span class="badge ghost">${fmtMs(payload.total_ms)}</span>`);
    }

    const copy = document.createElement("button");
    copy.className = "copy-btn";
    copy.type = "button";
    copy.textContent = "copy";
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(rawText);
        copy.textContent = "copied";
        setTimeout(() => (copy.textContent = "copy"), 1200);
      } catch (_) { /* ignore */ }
    });

    view.meta.innerHTML = badges.join("");
    view.meta.appendChild(copy);
  }

  function scrollToBottom() {
    el.messages.scrollTop = el.messages.scrollHeight;
  }

  async function sendQuestion(question) {
    if (state.busy) return;
    question = question.trim();
    if (!question) return;

    state.busy = true;
    el.sendBtn.disabled = true;
    el.question.disabled = true;
    el.question.value = "";
    el.question.style.height = "auto";
    addUserMessage(question);

    const view = addAssistantShell();
    let raw = "";
    let firstToken = true;
    let payload = {};

    const paint = () => {
      view.answer.innerHTML = "";
      const span = document.createElement("span");
      span.style.whiteSpace = "pre-wrap";
      span.textContent = raw;
      view.answer.appendChild(span);
      const cursor = document.createElement("span");
      cursor.className = "cursor";
      view.answer.appendChild(cursor);
      scrollToBottom();
    };

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          history: state.history.slice(-10),
          top_k: Number(el.topK.value),
          verify: el.verifyToggle.checked,
          quality: state.quality,
          mode: state.mode,
        }),
      });

      if (!response.ok || !response.body) {
        let detail = `HTTP ${response.status}`;
        try {
          const data = await response.json();
          detail = data.detail || detail;
        } catch (_) { /* ignore */ }
        throw new Error(detail);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let boundary;
        while ((boundary = buffer.indexOf("\n\n")) >= 0) {
          const block = buffer.slice(0, boundary).trim();
          buffer = buffer.slice(boundary + 2);
          if (!block.startsWith("data:")) continue;
          const data = block.slice(5).trim();
          if (!data || data === "[DONE]") continue;

          let event;
          try {
            event = JSON.parse(data);
          } catch (_) {
            continue;
          }

          if (event.type === "sources") {
            view.sourcesData = event.sources;
            renderSources(view, event.sources);
          } else if (event.type === "provider") {
            el.providerPill.textContent = event.label;
          } else if (event.type === "delta") {
            if (firstToken) {
              firstToken = false;
              view.answer.innerHTML = "";
            }
            raw += event.text;
            paint();
          } else if (event.type === "verifying") {
            const badge = document.createElement("span");
            badge.className = "badge ghost";
            badge.textContent = "verifying grounding…";
            view.meta.hidden = false;
            view.meta.innerHTML = "";
            view.meta.appendChild(badge);
          } else if (event.type === "done") {
            payload = event;
          } else if (event.type === "error") {
            throw new Error(event.message);
          }
        }
      }

      if (payload.mode === "auto") {
        const cited = new Set((payload.citations || []).map((c) => c.n));
        const used = (view.sourcesData || []).filter((source) => cited.has(source.n));
        if (used.length) {
          renderSources(view, used);
          view.sourcesLabel.textContent = `${used.length} source${used.length === 1 ? "" : "s"} cited`;
        } else if (view.sources) {
          view.sources.hidden = true;
        }
      }
      if (payload.refused && view.sources) {
        view.sources.hidden = true;
      }

      view.answer.innerHTML = renderMarkdown(raw);
      renderMeta(view, payload, raw);

      state.history.push({ role: "user", content: question });
      state.history.push({ role: "assistant", content: raw });
      if (state.history.length > 12) state.history = state.history.slice(-12);
    } catch (error) {
      view.answer.innerHTML = renderMarkdown(
        `**Something went wrong.** ${error.message}\n\nCheck the terminal running the server for details.`
      );
      view.answer.style.color = "var(--danger)";
    } finally {
      state.busy = false;
      el.sendBtn.disabled = false;
      el.question.disabled = false;
      el.question.focus();
      scrollToBottom();
    }
  }

  // ---------------------------------------------------------------- uploads
  async function uploadFiles(fileList) {
    if (READ_ONLY) return;
    if (!fileList || !fileList.length) return;
    const form = new FormData();
    for (const file of fileList) form.append("files", file);

    el.uploadProgress.hidden = false;
    el.uploadText.textContent = `Indexing ${fileList.length} file(s)…`;
    el.dropzone.classList.add("drag");

    try {
      const response = await fetch("/api/upload", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);

      const ingested = data.ingested || [];
      const rejected = data.rejected || [];
      el.uploadText.textContent = ingested.length
        ? `Indexed: ${ingested.join(", ")}`
        : "Nothing was indexed";
      if (rejected.length) {
        el.uploadText.textContent += ` · skipped: ${rejected.map((r) => r.file).join(", ")}`;
      }
      await refreshDocs();
      await refreshHealth();
      setTimeout(() => (el.uploadProgress.hidden = true), 3500);
    } catch (error) {
      el.uploadText.textContent = `Upload failed: ${error.message}`;
      setTimeout(() => (el.uploadProgress.hidden = true), 6000);
    } finally {
      el.dropzone.classList.remove("drag");
      el.fileInput.value = "";
    }
  }

  async function reindex() {
    if (READ_ONLY) return;
    el.reindexBtn.disabled = true;
    el.reindexBtn.textContent = "Rebuilding…";
    try {
      await api("/api/reindex", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reset: false, force: true }),
      });
      await refreshDocs();
      await refreshHealth();
    } catch (error) {
      alert(`Reindex failed: ${error.message}`);
    } finally {
      el.reindexBtn.disabled = false;
      el.reindexBtn.textContent = "↻ Rebuild index";
    }
  }

  async function deleteDoc(name) {
    if (READ_ONLY) return;
    if (!confirm(`Remove "${name}" from the index and delete the file?`)) return;
    try {
      await api(`/api/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
      await refreshDocs();
      await refreshHealth();
    } catch (error) {
      alert(`Delete failed: ${error.message}`);
    }
  }

  // ------------------------------------------------------------------ events
  function autoResize() {
    el.question.style.height = "auto";
    el.question.style.height = Math.min(el.question.scrollHeight, 180) + "px";
  }

  el.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    sendQuestion(el.question.value);
  });

  el.question.addEventListener("input", autoResize);
  el.question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendQuestion(el.question.value);
    }
  });

  el.topK.addEventListener("input", () => {
    el.topKVal.textContent = el.topK.value;
  });

  el.mode.addEventListener("click", (event) => {
    const button = event.target.closest("[data-mode]");
    if (!button || state.busy) return;
    state.mode = button.dataset.mode;
    [...el.mode.children].forEach((child) =>
      child.classList.toggle("active", child === button)
    );
    updateModeUi();
    if (state.health) refreshHealth();
  });

  el.quality.addEventListener("click", (event) => {
    const button = event.target.closest("[data-quality]");
    if (!button || state.busy) return;
    state.quality = button.dataset.quality;
    [...el.quality.children].forEach((child) =>
      child.classList.toggle("active", child === button)
    );
    if (state.health) refreshHealth();
  });

  el.docList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-delete]");
    if (button) deleteDoc(button.dataset.delete);
  });

  el.dropzone.addEventListener("click", () => el.fileInput.click());
  el.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      el.fileInput.click();
    }
  });
  el.fileInput.addEventListener("change", () => uploadFiles(el.fileInput.files));

  ["dragenter", "dragover"].forEach((name) =>
    el.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      el.dropzone.classList.add("drag");
    })
  );
  ["dragleave", "drop"].forEach((name) =>
    el.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      if (name === "drop") uploadFiles(event.dataTransfer.files);
      el.dropzone.classList.remove("drag");
    })
  );

  el.reindexBtn.addEventListener("click", reindex);

  el.newChatBtn.addEventListener("click", () => {
    if (state.busy) return;
    state.history = [];
    el.messages.innerHTML = state.emptyHTML;
    el.emptyState = document.getElementById("emptyState");
    el.emptyTitle = document.getElementById("emptyTitle");
    el.emptyText = document.getElementById("emptyText");
    updateEmpty();
    updateBanners();
    el.question.focus();
  });

  el.sidebarToggle.addEventListener("click", () => showSidebar(!el.sidebar.classList.contains("open")));
  el.overlay.addEventListener("click", () => showSidebar(false));

  el.messages.addEventListener("click", (event) => {
    const suggestion = event.target.closest("[data-q]");
    if (suggestion) {
      el.question.value = suggestion.dataset.q;
      autoResize();
      sendQuestion(suggestion.dataset.q);
    }
  });

  // -------------------------------------------------------------------- init
  state.emptyHTML = el.messages.innerHTML;
  updateModeUi();
  refreshHealth().then(refreshDocs);
  el.question.focus();
})();
