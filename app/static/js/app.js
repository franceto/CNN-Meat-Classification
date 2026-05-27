const imageInput = document.getElementById("imageInput");
const uploadBox = document.getElementById("uploadBox");
const uploadForm = document.getElementById("uploadForm");
const predictBtn = document.getElementById("predictBtn");
const changeImageBtn = document.getElementById("changeImageBtn");
const imagePreview = document.getElementById("imagePreview");
const imagePreviewWrap = document.getElementById("imagePreviewWrap");
const imageInfoWrap = document.getElementById("imageInfoWrap");
const emptyResult = document.getElementById("emptyResult");
const resultWrap = document.getElementById("resultWrap");
const predClass = document.getElementById("predClass");
const predConfidence = document.getElementById("predConfidence");
const resultOriginalImage = document.getElementById("resultOriginalImage");
const gradcamImage = document.getElementById("gradcamImage");
const loadingOverlay = document.getElementById("loadingOverlay");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatMessages = document.getElementById("chatMessages");
const chatToggle = document.getElementById("chatToggle");
const chatPanel = document.getElementById("chatPanel");
const chatClose = document.getElementById("chatClose");
const singleModeBtn = document.getElementById("singleModeBtn");
const batchModeBtn = document.getElementById("batchModeBtn");
const singleModePanel = document.getElementById("singleModePanel");
const batchModePanel = document.getElementById("batchModePanel");

let probChart = null;
let lastContext = "";
let currentObjectUrl = "";
let currentFileInfo = null;
function setMode(mode) {
    const isSingle = mode === "single";

    singleModeBtn.classList.toggle("active", isSingle);
    batchModeBtn.classList.toggle("active", !isSingle);

    singleModePanel.classList.toggle("d-none", !isSingle);
    batchModePanel.classList.toggle("d-none", isSingle);

    if (!isSingle) {
        resetResult();
        emptyResult.classList.remove("d-none");
        resultWrap.classList.add("d-none");
        batchResultWrap.classList.add("d-none");
    } else {
        batchResultWrap.classList.add("d-none");
        emptyResult.classList.remove("d-none");
    }
}
function showLoading(show) {
    loadingOverlay.classList.toggle("d-none", !show);
}

function addMessage(role, text) {
    const msg = document.createElement("div");
    const bubble = document.createElement("div");

    msg.className = `msg ${role}`;
    bubble.className = "bubble";
    bubble.textContent = text;

    msg.appendChild(bubble);
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function resetResult() {
    emptyResult.classList.remove("d-none");
    resultWrap.classList.add("d-none");

    predClass.textContent = "-";
    predConfidence.textContent = "0%";

    if (gradcamImage) gradcamImage.removeAttribute("src");

    if (probChart) {
        probChart.destroy();
        probChart = null;
    }

    lastContext = "";
}

function setLocalInfo(file, img) {
    currentFileInfo = {
        name: file.name,
        format: (file.type.split("/")[1] || "-").toUpperCase(),
        size_kb: Number((file.size / 1024).toFixed(2)),
        width: img.naturalWidth,
        height: img.naturalHeight
    };

    document.getElementById("infoName").textContent = currentFileInfo.name;
    document.getElementById("infoFormat").textContent = currentFileInfo.format;
    document.getElementById("infoSize").textContent = `${currentFileInfo.size_kb} KB`;
    document.getElementById("infoResolution").textContent = `${currentFileInfo.width} × ${currentFileInfo.height} px`;

    imageInfoWrap.classList.remove("d-none");
}

function previewFile(file) {
    if (!file) return;

    if (currentObjectUrl) {
        URL.revokeObjectURL(currentObjectUrl);
    }

    currentObjectUrl = URL.createObjectURL(file);

    const img = new Image();

    img.onload = () => {
        imagePreview.src = currentObjectUrl;
        resultOriginalImage.src = currentObjectUrl;

        imagePreviewWrap.classList.remove("d-none");
        uploadBox.classList.add("compact-hidden");
        predictBtn.classList.remove("d-none");

        setLocalInfo(file, img);
        resetResult();
    };

    img.src = currentObjectUrl;
}

function renderChart(probabilities) {
    const labels = Object.keys(probabilities);
    const values = Object.values(probabilities).map(v => Number((v * 100).toFixed(2)));
    const ctx = document.getElementById("probChart");

    if (probChart) {
        probChart.destroy();
        probChart = null;
    }

    probChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: ["rgba(139, 30, 30, 0.78)", "rgba(255, 179, 92, 0.85)"],
                borderColor: ["rgba(139, 30, 30, 1)", "rgba(214, 112, 28, 1)"],
                borderWidth: 1,
                borderRadius: 12,
                barThickness: 26
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: item => `${item.raw.toFixed(2)}%`
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: value => `${value}%`
                    }
                },
                y: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

function renderResult(data) {
    const result = data.result;

    emptyResult.classList.add("d-none");
    resultWrap.classList.remove("d-none");

    predClass.textContent = result.class;
    predConfidence.textContent = `${(result.confidence * 100).toFixed(2)}%`;
    gradcamImage.src = `${result.overlay_url}?t=${Date.now()}`;

    renderChart(result.probabilities);

    lastContext = JSON.stringify({
        class: result.class,
        confidence: result.confidence,
        probabilities: result.probabilities,
        image_info: currentFileInfo || data.info
    }, null, 2);
}

chatToggle.addEventListener("click", () => {
    chatPanel.classList.toggle("d-none");
});

chatClose.addEventListener("click", () => {
    chatPanel.classList.add("d-none");
});
singleModeBtn.addEventListener("click", () => {
    setMode("single");
});

batchModeBtn.addEventListener("click", () => {
    setMode("batch");
});
imageInput.addEventListener("click", () => {
    imageInput.value = "";
});

imageInput.addEventListener("change", () => {
    previewFile(imageInput.files[0]);
});

changeImageBtn.addEventListener("click", () => {
    imageInput.click();
});

imagePreview.addEventListener("click", () => {
    imageInput.click();
});

["dragenter", "dragover"].forEach(eventName => {
    uploadBox.addEventListener(eventName, e => {
        e.preventDefault();
        uploadBox.classList.add("dragover");
    });
});

["dragleave", "drop"].forEach(eventName => {
    uploadBox.addEventListener(eventName, e => {
        e.preventDefault();
        uploadBox.classList.remove("dragover");
    });
});

uploadBox.addEventListener("drop", e => {
    const file = e.dataTransfer.files[0];

    if (!file) return;

    imageInput.files = e.dataTransfer.files;
    previewFile(file);
});

uploadForm.addEventListener("submit", async e => {
    e.preventDefault();

    const file = imageInput.files[0];

    if (!file) {
        addMessage("bot", "Bạn cần upload ảnh trước khi dự đoán.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    showLoading(true);
    predictBtn.disabled = true;

    try {
        const res = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        if (!data.ok) {
            throw new Error(data.error || "Lỗi dự đoán.");
        }

        renderResult(data);
        addMessage("bot", `Đã dự đoán xong. Kết quả: ${data.result.class} với xác suất ${(data.result.confidence * 100).toFixed(2)}%.`);
    } catch (err) {
        addMessage("bot", `Lỗi: ${err.message}`);
    } finally {
        predictBtn.disabled = false;
        showLoading(false);
    }
});

chatForm.addEventListener("submit", async e => {
    e.preventDefault();

    const question = chatInput.value.trim();

    if (!question) return;

    addMessage("user", question);
    chatInput.value = "";

    const formData = new FormData();
    formData.append("question", question);
    formData.append("context", lastContext || "Chưa có kết quả dự đoán.");

    addMessage("bot", "AI đang suy nghĩ...");

    try {
        const res = await fetch("/chat", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        chatMessages.lastElementChild.remove();

        if (!data.ok) {
            throw new Error(data.answer || "Lỗi gọi AI.");
        }

        addMessage("bot", data.answer);
    } catch (err) {
        chatMessages.lastElementChild.remove();
        addMessage("bot", `Lỗi AI: ${err.message}`);
    }
});
const batchForm = document.getElementById("batchForm");
const batchArchiveInput = document.getElementById("batchArchiveInput");
const batchFolderInput = document.getElementById("batchFolderInput");
const batchPredictBtn = document.getElementById("batchPredictBtn");
const batchResultWrap = document.getElementById("batchResultWrap");
const batchCount = document.getElementById("batchCount");
const batchReportLink = document.getElementById("batchReportLink");
const batchUploadChoices = document.getElementById("batchUploadChoices");
const batchSelectedInfo = document.getElementById("batchSelectedInfo");
const batchSelectedIcon = document.getElementById("batchSelectedIcon");
const batchSelectedName = document.getElementById("batchSelectedName");
const batchSelectedCount = document.getElementById("batchSelectedCount");
const batchSelectedFormat = document.getElementById("batchSelectedFormat");
const batchSelectedSize = document.getElementById("batchSelectedSize");
const changeBatchFileBtn = document.getElementById("changeBatchFileBtn");
function formatBytes(bytes) {
    if (!bytes) return "0 KB";

    const units = ["B", "KB", "MB", "GB"];
    let size = bytes;
    let unit = 0;

    while (size >= 1024 && unit < units.length - 1) {
        size /= 1024;
        unit++;
    }

    return `${size.toFixed(unit === 0 ? 0 : 2)} ${units[unit]}`;
}

function countImageFiles(files) {
    return Array.from(files).filter(file => file.type.startsWith("image/")).length;
}

function getFolderName(files) {
    const first = files[0];

    if (!first) return "Folder ảnh";

    if (first.webkitRelativePath) {
        return first.webkitRelativePath.split("/")[0];
    }

    return "Folder ảnh";
}

function showBatchSelectedInfo({ name, count, format, size, icon }) {
    batchSelectedName.textContent = name;
    batchSelectedCount.textContent = count;
    batchSelectedFormat.textContent = format;
    batchSelectedSize.textContent = size;

    batchSelectedIcon.className = icon;

    batchUploadChoices.classList.add("d-none");
    batchSelectedInfo.classList.remove("d-none");
    batchPredictBtn.classList.remove("d-none");
}

function resetBatchSelectedInfo() {
    batchArchiveInput.value = "";
    batchFolderInput.value = "";

    batchUploadChoices.classList.remove("d-none");
    batchSelectedInfo.classList.add("d-none");
    batchPredictBtn.classList.add("d-none");

    batchSelectedName.textContent = "-";
    batchSelectedCount.textContent = "-";
    batchSelectedFormat.textContent = "-";
    batchSelectedSize.textContent = "-";
}
function updateBatchButton() {
    const hasArchive = batchArchiveInput.files.length > 0;
    const hasFolder = batchFolderInput.files.length > 0;
    batchPredictBtn.classList.toggle("d-none", !(hasArchive || hasFolder));
}

batchArchiveInput.addEventListener("change", () => {
    const file = batchArchiveInput.files[0];

    if (!file) return;

    batchFolderInput.value = "";

    const ext = file.name.split(".").pop().toUpperCase();

    showBatchSelectedInfo({
        name: file.name,
        count: "Chờ xử lý",
        format: ext,
        size: formatBytes(file.size),
        icon: "bi bi-file-earmark-zip"
    });
});

batchFolderInput.addEventListener("change", () => {
    const files = batchFolderInput.files;

    if (!files.length) return;

    batchArchiveInput.value = "";

    const imageCount = countImageFiles(files);
    const totalSize = Array.from(files).reduce((sum, file) => sum + file.size, 0);

    showBatchSelectedInfo({
        name: getFolderName(files),
        count: `${imageCount} ảnh`,
        format: "Folder",
        size: formatBytes(totalSize),
        icon: "bi bi-folder"
    });
});

changeBatchFileBtn.addEventListener("click", () => {
    resetBatchSelectedInfo();
});

batchForm.addEventListener("submit", async e => {
    e.preventDefault();

    const formData = new FormData();

    if (batchArchiveInput.files.length > 0) {
        formData.append("files", batchArchiveInput.files[0]);
    } else {
        for (const file of batchFolderInput.files) {
            formData.append("files", file);
        }
    }

    showLoading(true);
    batchPredictBtn.disabled = true;

    try {
        const res = await fetch("/batch_predict", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        if (!data.ok) {
            throw new Error(data.error || "Lỗi dự đoán hàng loạt.");
        }

        emptyResult.classList.add("d-none");
        resultWrap.classList.add("d-none");
        batchResultWrap.classList.remove("d-none");

        batchCount.textContent = `${data.result.count} ảnh`;
        batchSelectedCount.textContent = `${data.result.count} ảnh`;
        batchReportLink.href = data.result.report_url;
        batchReportLink.setAttribute("download", "bao_cao_du_doan_hang_loat.pdf");

        addMessage("bot", `Đã dự đoán hàng loạt ${data.result.count} ảnh và tạo báo cáo PDF.`);
    } catch (err) {
        addMessage("bot", `Lỗi batch: ${err.message}`);
    } finally {
        batchPredictBtn.disabled = false;
        showLoading(false);
    }
});