document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const videoInput = document.getElementById('videoInput');
    const dropzoneContent = document.getElementById('dropzoneContent');
    const filePreview = document.getElementById('filePreview');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const removeFileBtn = document.getElementById('removeFileBtn');
    const lineRatio = document.getElementById('lineRatio');
    const lineRatioVal = document.getElementById('lineRatioVal');
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');

    const emptyState = document.getElementById('emptyState');
    const processingCard = document.getElementById('processingCard');
    const resultsCard = document.getElementById('resultsCard');

    const progressBar = document.getElementById('progressBar');
    const progressPct = document.getElementById('progressPct');
    const frameProgress = document.getElementById('frameProgress');
    const liveCount = document.getElementById('liveCount');
    const liveFps = document.getElementById('liveFps');

    const finalCount = document.getElementById('finalCount');
    const originalVideo = document.getElementById('originalVideo');
    const outputVideo = document.getElementById('outputVideo');
    const syncPlayBtn = document.getElementById('syncPlayBtn');
    const downloadBtn = document.getElementById('downloadBtn');

    let selectedFile = null;
    let pollInterval = null;

    // Slider label update
    lineRatio.addEventListener('input', (e) => {
        lineRatioVal.textContent = `${e.target.value}%`;
    });

    // Dropzone Events
    dropzone.addEventListener('click', () => videoInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        videoInput.value = '';
        dropzoneContent.classList.remove('hidden');
        filePreview.classList.add('hidden');
        submitBtn.disabled = true;
    });

    function handleFileSelect(file) {
        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;

        dropzoneContent.classList.add('hidden');
        filePreview.classList.remove('hidden');
        submitBtn.disabled = false;
    }

    // Form Submit
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append('video', selectedFile);
        formData.append('line_ratio', (lineRatio.value / 100.0).toString());
        formData.append('direction', document.getElementById('direction').value);

        // Show Processing Card
        emptyState.classList.add('hidden');
        resultsCard.classList.add('hidden');
        processingCard.classList.remove('hidden');

        submitBtn.disabled = true;

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                alert(`上傳失敗: ${err.error}`);
                resetUI();
                return;
            }

            const data = await response.json();
            startStatusPolling(data.task_id, data.input_filename);
        } catch (err) {
            alert('上傳遭遇網路錯誤，請重新再試');
            resetUI();
        }
    });

    function startStatusPolling(taskId, inputFilename) {
        if (pollInterval) clearInterval(pollInterval);

        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/status/${taskId}`);
                if (!res.ok) return;

                const data = await res.json();

                progressBar.style.width = `${data.progress}%`;
                progressPct.textContent = `${data.progress}%`;
                frameProgress.textContent = `${data.current_frame} / ${data.total_frames}`;
                liveCount.textContent = data.count;
                liveFps.textContent = `${data.fps} FPS`;

                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    showResults(inputFilename, data.output_filename, data.count);
                } else if (data.status === 'error') {
                    clearInterval(pollInterval);
                    alert(`處理出錯: ${data.error}`);
                    resetUI();
                }
            } catch (e) {
                console.error('Polling error', e);
            }
        }, 500);
    }

    function showResults(inputFilename, outputFilename, countVal) {
        processingCard.classList.add('hidden');
        resultsCard.classList.remove('hidden');

        finalCount.textContent = countVal;

        originalVideo.src = `/uploads/${inputFilename}`;
        outputVideo.src = `/outputs/${outputFilename}`;

        downloadBtn.href = `/outputs/${outputFilename}`;
        downloadBtn.download = outputFilename;

        submitBtn.disabled = false;
    }

    function resetUI() {
        processingCard.classList.add('hidden');
        resultsCard.classList.add('hidden');
        emptyState.classList.remove('hidden');
        submitBtn.disabled = false;
    }

    // Synchronized Video Playback
    syncPlayBtn.addEventListener('click', () => {
        if (originalVideo.paused) {
            originalVideo.currentTime = 0;
            outputVideo.currentTime = 0;
            originalVideo.play();
            outputVideo.play();
            syncPlayBtn.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
                暫停同步播放
            `;
        } else {
            originalVideo.pause();
            outputVideo.pause();
            syncPlayBtn.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                同步播放對比
            `;
        }
    });
});
