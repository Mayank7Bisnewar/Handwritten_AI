/**
 * Handwritten Notes AI — main.js
 * Upload, OCR, preview, copy, download, voice, history
 */

'use strict';

/* ─────────────────────────────────────────────
   Toast helper
───────────────────────────────────────────── */
function showToast(msg, type = 'info', duration = 3500) {
  const icons = { success: '✅', error: '❌', info: '💡' };
  const container = document.getElementById('toast-container');
  if (!container) return;
  const el = document.createElement('div');
  el.className = `toast-msg ${type}`;
  el.innerHTML = `<span>${icons[type]}</span><span>${msg}</span>`;
  container.appendChild(el);
  setTimeout(() => el.remove(), duration);
}

/* ─────────────────────────────────────────────
   Typing animation
───────────────────────────────────────────── */
function typeText(el, text, speed = 18) {
  el.value = '';
  el.classList.add('typing-cursor');
  let i = 0;
  const interval = setInterval(() => {
    el.value += text[i++];
    el.scrollTop = el.scrollHeight;
    if (i >= text.length) {
      clearInterval(interval);
      el.classList.remove('typing-cursor');
    }
  }, speed);
}

/* ─────────────────────────────────────────────
   Progress bar animation
───────────────────────────────────────────── */
function animateProgress(fillEl, labelEl, targetPct, duration = 2000) {
  const steps = 60;
  const stepMs = duration / steps;
  const increment = targetPct / steps;
  let current = 0;
  const timer = setInterval(() => {
    current = Math.min(current + increment, targetPct);
    fillEl.style.width = current + '%';
    if (labelEl) labelEl.textContent = Math.round(current) + '%';
    if (current >= targetPct) clearInterval(timer);
  }, stepMs);
}

/* ─────────────────────────────────────────────
   Upload page logic
───────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {

  /* ── File input / drag-drop ── */
  const dropZone    = document.getElementById('drop-zone');
  const fileInput   = document.getElementById('file-input');
  const previewWrap = document.getElementById('preview-wrap');
  const previewImg  = document.getElementById('preview-img');
  const previewName = document.getElementById('preview-name');
  const removeBtn   = document.getElementById('remove-btn');
  const submitBtn   = document.getElementById('submit-btn');
  const langSelect  = document.getElementById('lang-select');

  // Progress / AI animation
  const progressWrap = document.getElementById('progress-wrap');
  const progressFill = document.getElementById('progress-fill');
  const progressPct  = document.getElementById('progress-pct');
  const aiProcessing = document.getElementById('ai-processing');
  const aiStatus     = document.getElementById('ai-status');

  // Output panel (index page inline result)
  const outputSection  = document.getElementById('output-section');
  const outputTextarea = document.getElementById('output-textarea');
  const confFill       = document.getElementById('conf-fill');
  const confPct        = document.getElementById('conf-pct');
  const copyBtn        = document.getElementById('copy-btn');
  const dlTxtBtn       = document.getElementById('dl-txt-btn');
  const dlPdfBtn       = document.getElementById('dl-pdf-btn');
  const voiceBtn       = document.getElementById('voice-btn');

  let currentFile  = null;
  let currentNoteId = null;
  let speaking     = false;

  /* ── Drag & Drop ── */
  if (dropZone) {
    dropZone.addEventListener('dragover', e => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    });
    dropZone.addEventListener('click', () => fileInput && fileInput.click());
  }

  if (fileInput) {
    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) handleFile(fileInput.files[0]);
    });
  }

  function handleFile(file) {
    const allowed = ['image/png', 'image/jpeg', 'image/bmp', 'image/tiff', 'image/webp'];
    if (!allowed.includes(file.type)) {
      showToast('Please upload a valid image (PNG, JPG, BMP, TIFF, WebP).', 'error');
      return;
    }
    if (file.size > 16 * 1024 * 1024) {
      showToast('File too large. Maximum size is 16 MB.', 'error');
      return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = e => {
      if (previewImg) previewImg.src = e.target.result;
      if (previewName) previewName.textContent = file.name;
      if (previewWrap) previewWrap.classList.add('visible');
    };
    reader.readAsDataURL(file);
  }

  if (removeBtn) {
    removeBtn.addEventListener('click', e => {
      e.stopPropagation();
      currentFile = null;
      if (fileInput) fileInput.value = '';
      if (previewWrap) previewWrap.classList.remove('visible');
    });
  }

  /* ── Submit / OCR ── */
  if (submitBtn) {
    submitBtn.addEventListener('click', () => {
      if (!currentFile) { showToast('Please upload an image first.', 'error'); return; }
      runOCR();
    });
  }

  function runOCR() {
    const statusMessages = [
      'INITIALISING AI…',
      'PREPROCESSING IMAGE…',
      'RUNNING OCR ENGINE…',
      'DECODING TEXT…',
      'FINALISING OUTPUT…',
    ];
    let msgIdx = 0;

    // Show animations
    if (progressWrap) progressWrap.classList.add('visible');
    if (aiProcessing) aiProcessing.classList.add('visible');
    if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = 'Processing…'; }

    // Cycle status messages
    const msgTimer = setInterval(() => {
      if (aiStatus) aiStatus.textContent = statusMessages[msgIdx % statusMessages.length];
      msgIdx++;
    }, 900);

    // Animate progress (fake progress until response)
    if (progressFill) animateProgress(progressFill, progressPct, 85, 3500);

    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('language', langSelect ? langSelect.value : 'en');

    fetch('/api/upload', { method: 'POST', body: formData })
      .then(r => r.json())
      .then(data => {
        clearInterval(msgTimer);
        if (progressFill) animateProgress(progressFill, progressPct, 100, 300);

        if (data.error) {
          showToast(data.error, 'error');
          resetUI();
          return;
        }

        currentNoteId = data.note_id;

        // Show output panel
        if (outputSection) outputSection.classList.remove('d-none');
        if (outputTextarea) typeText(outputTextarea, data.text || '(No text detected)');

        // Confidence meter
        const conf = data.confidence || 0;
        if (confFill) setTimeout(() => { confFill.style.width = conf + '%'; }, 300);
        if (confPct)  confPct.textContent = conf.toFixed(1) + '%';

        // Wire download buttons
        if (dlTxtBtn) dlTxtBtn.href = `/api/download/txt/${data.note_id}`;
        if (dlPdfBtn) dlPdfBtn.href = `/api/download/pdf/${data.note_id}`;

        showToast('Text extracted successfully!', 'success');
        resetUI();
        if (outputSection) outputSection.scrollIntoView({ behavior: 'smooth' });
      })
      .catch(() => {
        clearInterval(msgTimer);
        showToast('Network error. Is the server running?', 'error');
        resetUI();
      });
  }

  function resetUI() {
    if (progressWrap) progressWrap.classList.remove('visible');
    if (aiProcessing) aiProcessing.classList.remove('visible');
    if (submitBtn) { submitBtn.disabled = false; submitBtn.innerHTML = '<span>⚡</span> Extract Text'; }
  }

  /* ── Copy button ── */
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const text = outputTextarea ? outputTextarea.value : '';
      if (!text) { showToast('Nothing to copy.', 'error'); return; }
      navigator.clipboard.writeText(text)
        .then(() => showToast('Copied to clipboard!', 'success'))
        .catch(() => showToast('Copy failed — try manually.', 'error'));
    });
  }

  /* ── Voice reading ── */
  if (voiceBtn) {
    voiceBtn.addEventListener('click', () => {
      if (!('speechSynthesis' in window)) {
        showToast('Voice not supported in your browser.', 'error');
        return;
      }
      if (speaking) {
        window.speechSynthesis.cancel();
        speaking = false;
        voiceBtn.classList.remove('speaking');
        voiceBtn.innerHTML = '🔊 Read Aloud';
        return;
      }
      const text = outputTextarea ? outputTextarea.value : '';
      if (!text) { showToast('No text to read.', 'error'); return; }
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate  = 0.95;
      utter.onend = () => {
        speaking = false;
        voiceBtn.classList.remove('speaking');
        voiceBtn.innerHTML = '🔊 Read Aloud';
      };
      speaking = true;
      voiceBtn.classList.add('speaking');
      voiceBtn.innerHTML = '⏹ Stop';
      window.speechSynthesis.speak(utter);
    });
  }

  /* ── History page delete ── */
  document.querySelectorAll('.delete-note-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.noteId;
      if (!confirm('Delete this note from history?')) return;
      fetch(`/api/delete/${id}`, { method: 'DELETE' })
        .then(r => r.json())
        .then(data => {
          showToast(data.message || 'Deleted.', 'success');
          const card = document.getElementById(`history-item-${id}`);
          if (card) card.remove();
        })
        .catch(() => showToast('Delete failed.', 'error'));
    });
  });

  /* ── Navbar active link ── */
  const path = window.location.pathname;
  document.querySelectorAll('.nav-link').forEach(link => {
    if (link.getAttribute('href') === path) link.classList.add('active');
  });

  /* ── Animated counter (hero stats) ── */
  document.querySelectorAll('[data-count-to]').forEach(el => {
    const target = parseInt(el.dataset.countTo, 10);
    let current  = 0;
    const step   = Math.ceil(target / 60);
    const timer  = setInterval(() => {
      current = Math.min(current + step, target);
      el.textContent = current.toLocaleString() + (el.dataset.suffix || '');
      if (current >= target) clearInterval(timer);
    }, 30);
  });
});
