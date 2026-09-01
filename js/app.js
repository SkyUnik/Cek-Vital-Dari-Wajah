/**
 * app.js
 * Main application orchestrator for Cek-Vital-Dari-Wajah rPPG monitor.
 */

import { FaceDetector } from './face-detector.js';
import { RPPG_DSP } from './rppg-dsp.js';
import { RPPG_Renderer } from './renderer.js';

class HeartbeatApp {
  constructor() {
    this.video = document.getElementById('webcam-video');
    this.canvas = document.getElementById('overlay-canvas');
    this.startBtn = document.getElementById('toggle-btn');
    this.btnIcon = document.getElementById('btn-icon');
    this.btnText = document.getElementById('btn-text');
    this.statusBadge = document.getElementById('status-badge');
    this.bpmDisplay = document.getElementById('bpm-metric');
    this.fpsDisplay = document.getElementById('fps-metric');
    this.deviceSelect = document.getElementById('camera-select');
    this.mirrorBtn = document.getElementById('mirror-btn');
    this.placeholder = document.getElementById('video-placeholder');
    this.terminalLogs = document.getElementById('terminal-logs');

    this.detector = new FaceDetector({ modelPath: './models/version-RFB-320.onnx', confThreshold: 0.65 });
    this.dsp = new RPPG_DSP({ minSignalSeconds: 5.0, maxSignalSeconds: 5.0, samplingFrequency: 1.0 });
    this.renderer = new RPPG_Renderer(this.canvas);

    this.stream = null;
    this.isRunning = false;
    this.isMirrored = true;
    this.animationFrameId = null;
    this.isDetecting = false;
    this.lastDetectionTime = 0;
    this.detectionInterval = 100; // Face detector run interval in ms (tracking smoothly)
    this.currentDetection = null;

    // Offscreen canvas for sampling ROI pixel data
    this.roiCanvas = document.createElement('canvas');
    this.roiCtx = this.roiCanvas.getContext('2d', { willReadFrequently: true });

    this.init();
  }

  async init() {
    this.bindEvents();
    this.log('Initializing ONNX Runtime Web...', 'info');
    this.setStatus('Loading ONNX Model...', 'warning');

    try {
      await this.detector.init();
      this.log('Model loaded. Ready to start.', 'success');
      this.setStatus('Ready', 'ready');
      this.startBtn.disabled = false;
      await this.loadCameraDevices();
    } catch (err) {
      this.log(`Model init error: ${err.message}`, 'error');
      this.setStatus('Model Error', 'error');
    }
  }

  bindEvents() {
    this.startBtn.addEventListener('click', () => this.toggleStream());
    if (this.mirrorBtn) {
      this.mirrorBtn.addEventListener('click', () => {
        this.isMirrored = !this.isMirrored;
        this.video.classList.toggle('-scale-x-100', this.isMirrored);
        this.mirrorBtn.classList.toggle('text-emerald-400', this.isMirrored);
        this.mirrorBtn.classList.toggle('border-emerald-500/50', this.isMirrored);
      });
    }
    if (this.deviceSelect) {
      this.deviceSelect.addEventListener('change', () => {
        if (this.isRunning) {
          this.stopStream().then(() => this.startStream());
        }
      });
    }

    window.addEventListener('resize', () => this.syncCanvasSize());
  }

  async loadCameraDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');

      if (this.deviceSelect && videoDevices.length > 0) {
        this.deviceSelect.innerHTML = '';
        videoDevices.forEach((device, idx) => {
          const opt = document.createElement('option');
          opt.value = device.deviceId;
          opt.text = device.label || `Camera ${idx + 1}`;
          this.deviceSelect.appendChild(opt);
        });
        this.deviceSelect.classList.remove('hidden');
      }
    } catch (e) {
      console.warn('Camera enumeration error:', e);
    }
  }

  async toggleStream() {
    if (this.isRunning) {
      await this.stopStream();
    } else {
      await this.startStream();
    }
  }

  async startStream() {
    this.setStatus('Requesting Camera...', 'warning');
    this.log('Requesting webcam access...', 'info');

    const deviceId = this.deviceSelect ? this.deviceSelect.value : undefined;
    const constraints = {
      audio: false,
      video: deviceId
        ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } }
        : { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } }
    };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.video.srcObject = this.stream;
      await this.video.play();

      this.isRunning = true;
      this.dsp.reset();
      this.detector.resetTracking();

      if (this.placeholder) {
        this.placeholder.classList.add('hidden');
      }

      this.updateBtnState(true);
      this.setStatus('Tracking Face', 'active');
      this.log('Camera started. Real-time rPPG loop running.', 'success');

      this.syncCanvasSize();
      this.startLoop();
    } catch (err) {
      this.log(`Camera access denied: ${err.message}`, 'error');
      this.setStatus('Camera Denied', 'error');
    }
  }

  async stopStream() {
    this.isRunning = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    this.video.srcObject = null;
    this.renderer.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    if (this.placeholder) {
      this.placeholder.classList.remove('hidden');
    }

    this.updateBtnState(false);
    this.setStatus('Stopped', 'idle');
    this.log('Session stopped.', 'info');
  }

  updateBtnState(running) {
    if (running) {
      this.startBtn.classList.replace('bg-emerald-600', 'bg-rose-600');
      this.startBtn.classList.replace('hover:bg-emerald-500', 'hover:bg-rose-500');
      this.btnText.textContent = 'Stop Measurement';
      this.btnIcon.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />`;
    } else {
      this.startBtn.classList.replace('bg-rose-600', 'bg-emerald-600');
      this.startBtn.classList.replace('hover:bg-rose-500', 'hover:bg-emerald-500');
      this.btnText.textContent = 'Start Measurement';
      this.btnIcon.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />`;
    }
  }

  syncCanvasSize() {
    if (this.video && this.video.videoWidth > 0) {
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
    }
  }

  startLoop() {
    const processFrame = async (now) => {
      if (!this.isRunning) return;

      this.syncCanvasSize();

      // Trigger face detection periodically
      if (!this.isDetecting && (now - this.lastDetectionTime >= this.detectionInterval)) {
        this.isDetecting = true;
        this.lastDetectionTime = now;
        this.detector.detect(this.video).then(detection => {
          this.currentDetection = detection;
          this.isDetecting = false;
        }).catch(err => {
          this.isDetecting = false;
        });
      }

      let dspResult = {
        isReady: false,
        fps: 0,
        bpm: 0,
        meanBpm: 0,
        filteredSignal: [],
        powerSpectrum: [],
        bufferProgress: 0
      };

      if (this.currentDetection && this.currentDetection.roi) {
        const roi = this.currentDetection.roi;

        // Sample mean RGB in forehead ROI
        const meanRGB = this.sampleROIMean(this.video, roi);
        if (meanRGB) {
          dspResult = this.dsp.processFrame(meanRGB, now, false);

          if (dspResult.isReady && dspResult.meanBpm > 0) {
            this.bpmDisplay.textContent = dspResult.meanBpm.toFixed(1);
          } else {
            this.bpmDisplay.textContent = '--';
          }

          this.fpsDisplay.textContent = dspResult.fps.toFixed(1);
        }
      }

      // Render canvas overlay with mirror support
      this.renderer.render(this.video, this.currentDetection, dspResult, this.isMirrored);

      if ('requestVideoFrameCallback' in this.video) {
        this.video.requestVideoFrameCallback(processFrame);
      } else {
        this.animationFrameId = requestAnimationFrame(processFrame);
      }
    };

    if ('requestVideoFrameCallback' in this.video) {
      this.video.requestVideoFrameCallback(processFrame);
    } else {
      this.animationFrameId = requestAnimationFrame(processFrame);
    }
  }

  /**
   * Extract average R, G, B color values in the forehead ROI
   */
  sampleROIMean(video, roi) {
    const rw = Math.max(2, Math.round(roi.width));
    const rh = Math.max(2, Math.round(roi.height));
    const rx = Math.max(0, Math.min(video.videoWidth - rw, Math.round(roi.x)));
    const ry = Math.max(0, Math.min(video.videoHeight - rh, Math.round(roi.y)));

    this.roiCanvas.width = rw;
    this.roiCanvas.height = rh;

    this.roiCtx.drawImage(video, rx, ry, rw, rh, 0, 0, rw, rh);
    const imgData = this.roiCtx.getImageData(0, 0, rw, rh);
    const data = imgData.data;
    const len = data.length;

    let rSum = 0;
    let gSum = 0;
    let bSum = 0;
    const pixelCount = len / 4;

    for (let i = 0; i < len; i += 4) {
      rSum += data[i];
      gSum += data[i + 1];
      bSum += data[i + 2];
    }

    return {
      r: rSum / pixelCount,
      g: gSum / pixelCount,
      b: bSum / pixelCount
    };
  }

  setStatus(text, type = 'idle') {
    if (!this.statusBadge) return;
    this.statusBadge.textContent = text;
    const colorMap = {
      idle: 'bg-slate-800 text-slate-400 border-slate-700',
      warning: 'bg-amber-950/70 text-amber-400 border-amber-800/80',
      ready: 'bg-emerald-950/70 text-emerald-400 border-emerald-800/80',
      active: 'bg-indigo-950/70 text-indigo-400 border-indigo-800/80 animate-pulse',
      error: 'bg-rose-950/70 text-rose-400 border-rose-800/80'
    };
    this.statusBadge.className = `px-3 py-1 text-xs font-mono font-medium rounded-full border transition-all ${colorMap[type] || colorMap.idle}`;
  }

  log(msg, type = 'info') {
    if (!this.terminalLogs) return;
    const line = document.createElement('div');
    const time = new Date().toLocaleTimeString();
    const colors = {
      info: 'text-slate-400',
      success: 'text-emerald-400',
      error: 'text-rose-400',
      warning: 'text-amber-400'
    };
    line.className = `font-mono text-xs ${colors[type] || colors.info}`;
    line.textContent = `[${time}] ${msg}`;
    this.terminalLogs.appendChild(line);
    this.terminalLogs.scrollTop = this.terminalLogs.scrollHeight;
  }
}

// Auto-boot on DOM ready
window.addEventListener('DOMContentLoaded', () => {
  window.app = new HeartbeatApp();
});
