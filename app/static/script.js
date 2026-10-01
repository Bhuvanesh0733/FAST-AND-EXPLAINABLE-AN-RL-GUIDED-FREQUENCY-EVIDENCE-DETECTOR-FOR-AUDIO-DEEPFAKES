const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const detectButton = document.getElementById("detectButton");
const status = document.getElementById("status");
const results = document.getElementById("results");
const waveformCanvas = document.getElementById("waveformCanvas");
const spectrogramCanvas = document.getElementById("spectrogramCanvas");
const scopeAxis = document.getElementById("scopeAxis");
const verdictBadge = document.getElementById("verdictBadge");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceValue = document.getElementById("confidenceValue");
const bandList = document.getElementById("bandList");
const aiExplanation = document.getElementById("aiExplanation");
const sourceTag = document.getElementById("sourceTag");

let selectedFile = null;

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") fileInput.click();
});
dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) stageFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
    if (fileInput.files.length) stageFile(fileInput.files[0]);
});
detectButton.addEventListener("click", () => {
    if (selectedFile) runDetection(selectedFile);
});

function stageFile(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    detectButton.disabled = false;
    status.textContent = "";
    results.hidden = true;
}

async function runDetection(file) {
    detectButton.disabled = true;
    status.textContent = "Analyzing\u2026";
    results.hidden = true;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/api/predict", { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) {
            status.textContent = data.error || "Something went wrong.";
            detectButton.disabled = false;
            return;
        }
        status.textContent = "";
        detectButton.disabled = false;
        renderResults(data);
    } catch (err) {
        status.textContent = "Could not reach the server.";
        detectButton.disabled = false;
    }
}

function renderResults(data) {
    results.hidden = false;

    verdictBadge.textContent = data.prediction === "bonafide" ? "Bona fide" : "Synthetic";
    verdictBadge.className = "verdict-badge " + data.prediction;

    const pct = Math.round(data.confidence * 100);
    confidenceFill.style.width = pct + "%";
    confidenceFill.style.background =
        data.prediction === "bonafide" ? "var(--mark-teal)" : "var(--mark-amber)";
    confidenceValue.textContent = pct + "% confidence";

    aiExplanation.textContent = data.explanation || "";
    sourceTag.textContent = data.explanation_source === "ai" ? "Groq" : "rule-based";

    drawWaveform(data.waveform);
    drawSpectrogram(data.spectrogram, data.evidence_bands);
    renderAxis();
    renderBandList(data.band_labels, data.band_keep_probs, data.evidence_bands);
}

function drawWaveform(samples) {
    const ctx = waveformCanvas.getContext("2d");
    const w = (waveformCanvas.width = waveformCanvas.clientWidth);
    const h = waveformCanvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "#c99a3d"; // amber trace, like a scope readout
    ctx.lineWidth = 1;
    ctx.beginPath();
    samples.forEach((v, i) => {
        const x = (i / (samples.length - 1)) * w;
        const y = h / 2 - v * (h / 2) * 0.9;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
}

function lerpColor(a, b, t) {
    return a.map((c, i) => Math.round(c + (b[i] - c) * t));
}

function drawSpectrogram(mel, evidenceBands) {
    const ctx = spectrogramCanvas.getContext("2d");
    const nMels = mel.length;
    const nFrames = mel[0].length;
    const w = (spectrogramCanvas.width = spectrogramCanvas.clientWidth);
    const h = spectrogramCanvas.height;

    const cellW = w / nFrames;
    const cellH = h / nMels;

    let min = Infinity;
    let max = -Infinity;
    for (const row of mel) {
        for (const v of row) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
    }

    // dark "instrument screen" gradient: near-black -> slate -> amber peaks
    const low = [20, 22, 27];
    const mid = [58, 68, 80];
    const high = [201, 154, 61];

    for (let m = 0; m < nMels; m++) {
        for (let t = 0; t < nFrames; t++) {
            const norm = (mel[m][t] - min) / (max - min + 1e-9);
            const color = norm < 0.5 ? lerpColor(low, mid, norm / 0.5) : lerpColor(mid, high, (norm - 0.5) / 0.5);
            ctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
            const y = h - (m + 1) * cellH; // low mel index = low frequency = bottom
            ctx.fillRect(t * cellW, y, cellW + 1, cellH + 1);
        }
    }

    const nBands = evidenceBands.length;
    const bandEdges = [];
    for (let i = 0; i <= nBands; i++) bandEdges.push(Math.round((i / nBands) * nMels));

    ctx.fillStyle = "rgba(20, 106, 92, 0.35)"; // teal evidence highlight, visible on dark screen
    for (let i = 0; i < nBands; i++) {
        if (!evidenceBands[i]) continue;
        const yTop = h - bandEdges[i + 1] * cellH;
        const bandHeight = (bandEdges[i + 1] - bandEdges[i]) * cellH;
        ctx.fillRect(0, yTop, w, bandHeight);
    }
}

function renderAxis() {
    // static labels: clips are resampled to 16kHz, so Nyquist is 8kHz
    scopeAxis.innerHTML = "";
    ["0 Hz", "2k", "4k", "6k", "8k"].forEach((label) => {
        const span = document.createElement("span");
        span.textContent = label;
        scopeAxis.appendChild(span);
    });
}

function renderBandList(labels, probs, kept) {
    bandList.innerHTML = "";
    labels.forEach((label, i) => {
        const li = document.createElement("li");
        li.className = "band-row" + (kept[i] ? " kept" : "");
        const pct = Math.round(probs[i] * 100);
        li.innerHTML = `
      <span>${label}</span>
      <span class="band-track"><span class="band-fill" style="width:${pct}%"></span></span>
      <span class="band-pct">${pct}%</span>
    `;
        bandList.appendChild(li);
    });
}