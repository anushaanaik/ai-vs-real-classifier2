/**
 * script.js — Frontend logic for AI vs Real Image Classifier
 */

const API_BASE = window.location.origin; // ✅ no hardcoded port

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropZone      = document.getElementById('dropZone');
const fileInput     = document.getElementById('fileInput');
const uploadCard    = document.getElementById('uploadCard');
const resultCard    = document.getElementById('resultCard');
const previewImg    = document.getElementById('previewImg');
const resetBtn      = document.getElementById('resetBtn');

const stateLoading  = document.getElementById('stateLoading');
const stateResult   = document.getElementById('stateResult');
const stateError    = document.getElementById('stateError');

const verdictBadge  = document.getElementById('verdictBadge');
const confidencePct = document.getElementById('confidencePct');
const confidenceBar = document.getElementById('confidenceBar');
const aiProb        = document.getElementById('aiProb');
const realProb      = document.getElementById('realProb');
const ttaNote       = document.getElementById('ttaNote');
const errorMsg      = document.getElementById('errorMsg');


// ── State helpers ─────────────────────────────────────────────────────────────

function showState(state) {
  [stateLoading, stateResult, stateError].forEach(el => el.classList.add('hidden'));
  state.classList.remove('hidden');
}

function showResultCard() {
  uploadCard.classList.add('hidden');
  resultCard.classList.remove('hidden');
}

function resetUI() {
  resultCard.classList.add('hidden');
  uploadCard.classList.remove('hidden');
  previewImg.src = '';
  fileInput.value = '';
  confidenceBar.style.width = '0';
  confidenceBar.style.background = 'var(--accent)';
}


// ── Image selection ───────────────────────────────────────────────────────────

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    alert('Please select a valid image file (JPEG, PNG, or WebP).');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    alert('File is too large. Maximum size is 10 MB.');
    return;
  }

  // Show preview immediately, then kick off API call
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    showResultCard();
    showState(stateLoading);
    uploadImage(file);
  };
  reader.readAsDataURL(file);
}

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

document.addEventListener('paste', (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      handleFile(item.getAsFile());
      break;
    }
  }
});

resetBtn.addEventListener('click', resetUI);


// ── API call ──────────────────────────────────────────────────────────────────

async function uploadImage(file) {
  // ✅ Convert to Base64 (strip data:image/...;base64, prefix)
  const base64 = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload  = () => resolve(reader.result.split(',')[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_b64: base64,
        filename:  file.name,
        tta_steps: 8,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    renderResult(data);

  } catch (err) {
    showState(stateError);
    errorMsg.textContent = err.message || 'Network error — is the backend running?';
  }
}


// ── Render result ─────────────────────────────────────────────────────────────

const LABEL_META = {
  AI:        { emoji: '🤖', text: 'AI Generated', cls: 'is-ai',        barColor: '#ff5e7e' },
  REAL:      { emoji: '📷', text: 'Real Photo',   cls: 'is-real',      barColor: '#43e08a' },
  UNCERTAIN: { emoji: '❓', text: 'Uncertain',    cls: 'is-uncertain', barColor: '#f5a623' },
};

function renderResult(data) {
  const meta = LABEL_META[data.label] || LABEL_META.UNCERTAIN;
  const pct  = Math.round(data.confidence * 100);

  verdictBadge.textContent = `${meta.emoji}  ${meta.text}`;
  verdictBadge.className   = `verdict ${meta.cls}`;

  confidencePct.textContent = `${pct}%`;
  confidenceBar.style.background = meta.barColor;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      confidenceBar.style.width = `${pct}%`;
    });
  });

  aiProb.textContent   = `${Math.round(data.ai_prob * 100)}%`;
  realProb.textContent = `${Math.round(data.real_prob * 100)}%`;
  ttaNote.textContent  = `Averaged over ${data.tta_steps} TTA pass${data.tta_steps === 1 ? '' : 'es'}`;

  showState(stateResult);
}