/**
 * face-detector.js
 * In-browser face detection using ONNX Runtime Web.
 */

export class FaceDetector {
  constructor(options = {}) {
    this.modelPath = options.modelPath || './models/version-RFB-320.onnx';
    this.confThreshold = options.confThreshold || 0.7;
    this.inputWidth = 320;
    this.inputHeight = 240;

    this.session = null;
    this.isLoaded = false;
    this.trackedBox = null;
    this.smoothingAlpha = 0.35; // Smoothing factor for bounding box jitter reduction

    // Offscreen canvas for frame preprocessing
    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCanvas.width = this.inputWidth;
    this.offscreenCanvas.height = this.inputHeight;
    this.offscreenCtx = this.offscreenCanvas.getContext('2d', { willReadFrequently: true });
  }

  async init() {
    if (typeof ort === 'undefined') {
      throw new Error('ONNX Runtime Web (ort) is not loaded.');
    }

    try {
      if (ort.env && ort.env.wasm) {
        const vendorBase = typeof window !== 'undefined'
          ? new URL('./js/vendor/', window.location.href).href
          : './js/vendor/';
        ort.env.wasm.wasmPaths = vendorBase;
        ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
      }

      this.session = await ort.InferenceSession.create(this.modelPath, {
        executionProviders: ['wasm', 'webgl']
      });
      this.isLoaded = true;
      console.log('Face detector ONNX model loaded successfully.');
      return true;
    } catch (err) {
      console.warn('Local WASM path failed, trying fallback CDN initialization...', err);
      try {
        if (ort.env && ort.env.wasm) {
          ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
        }
        this.session = await ort.InferenceSession.create(this.modelPath, {
          executionProviders: ['wasm', 'webgl']
        });
        this.isLoaded = true;
        return true;
      } catch (e2) {
        console.error('Failed to initialize ONNX model:', e2);
        throw e2;
      }
    }
  }

  /**
   * Detect face in HTMLVideoElement or HTMLCanvasElement
   * @param {HTMLVideoElement|HTMLCanvasElement} sourceElement
   * @returns {Object|null} Detected face box, forehead ROI, and landmark points
   */
  async detect(sourceElement) {
    if (!this.isLoaded || !this.session) return null;

    const srcWidth = sourceElement.videoWidth || sourceElement.width;
    const srcHeight = sourceElement.videoHeight || sourceElement.height;
    if (!srcWidth || !srcHeight) return null;

    // Draw frame to 320x240 offscreen canvas
    this.offscreenCtx.drawImage(sourceElement, 0, 0, this.inputWidth, this.inputHeight);
    const imgData = this.offscreenCtx.getImageData(0, 0, this.inputWidth, this.inputHeight);
    const pixels = imgData.data;

    // Prepare NCHW Float32 tensor (1, 3, 240, 320)
    const tensorSize = 3 * this.inputHeight * this.inputWidth;
    const floatData = new Float32Array(tensorSize);
    const channelSize = this.inputHeight * this.inputWidth;

    for (let i = 0; i < channelSize; i++) {
      const pIdx = i * 4;
      const r = (pixels[pIdx] - 127.0) / 128.0;
      const g = (pixels[pIdx + 1] - 127.0) / 128.0;
      const b = (pixels[pIdx + 2] - 127.0) / 128.0;

      floatData[i] = r;                  // Red channel (channel 0)
      floatData[channelSize + i] = g;    // Green channel (channel 1)
      floatData[2 * channelSize + i] = b;// Blue channel (channel 2)
    }

    const inputTensor = new ort.Tensor('float32', floatData, [1, 3, this.inputHeight, this.inputWidth]);

    // Run inference
    const outputMap = await this.session.run({ input: inputTensor });
    const scoresTensor = outputMap['scores'] || outputMap[Object.keys(outputMap)[0]];
    const boxesTensor = outputMap['boxes'] || outputMap[Object.keys(outputMap)[1]];

    const scores = scoresTensor.data;
    const boxes = boxesTensor.data;
    const numBoxes = boxes.length / 4;

    let bestScore = -1;
    let bestBox = null;

    // Find detection with highest face confidence
    for (let i = 0; i < numBoxes; i++) {
      const faceScore = scores[i * 2 + 1];
      if (faceScore > this.confThreshold && faceScore > bestScore) {
        bestScore = faceScore;
        const x1 = Math.max(0, Math.min(1.0, boxes[i * 4]));
        const y1 = Math.max(0, Math.min(1.0, boxes[i * 4 + 1]));
        const x2 = Math.max(0, Math.min(1.0, boxes[i * 4 + 2]));
        const y2 = Math.max(0, Math.min(1.0, boxes[i * 4 + 3]));

        bestBox = {
          x: x1 * srcWidth,
          y: y1 * srcHeight,
          width: (x2 - x1) * srcWidth,
          height: (y2 - y1) * srcHeight,
          confidence: faceScore
        };
      }
    }

    if (!bestBox) {
      return null;
    }

    // Apply smoothing to reduce frame-to-frame bounding box jitter
    if (!this.trackedBox) {
      this.trackedBox = { ...bestBox };
    } else {
      const a = this.smoothingAlpha;
      this.trackedBox.x = (1 - a) * this.trackedBox.x + a * bestBox.x;
      this.trackedBox.y = (1 - a) * this.trackedBox.y + a * bestBox.y;
      this.trackedBox.width = (1 - a) * this.trackedBox.width + a * bestBox.width;
      this.trackedBox.height = (1 - a) * this.trackedBox.height + a * bestBox.height;
      this.trackedBox.confidence = bestBox.confidence;
    }

    const box = this.trackedBox;

    // Forehead Region of Interest (ROI) matching heartbeat-master:
    // roi = Rect(box.tl().x + 0.3 * box.width, box.tl().y + 0.1 * box.height, 0.4 * box.width, 0.15 * box.height)
    const roi = {
      x: box.x + 0.3 * box.width,
      y: box.y + 0.1 * box.height,
      width: 0.4 * box.width,
      height: 0.15 * box.height
    };

    // Tracking landmark feature points (matching heartbeat-master corners)
    const corners = [
      { x: box.x + 0.30 * box.width, y: box.y + 0.22 * box.height }, // Forehead left
      { x: box.x + 0.70 * box.width, y: box.y + 0.22 * box.height }, // Forehead right
      { x: box.x + 0.36 * box.width, y: box.y + 0.45 * box.height }, // Left eye
      { x: box.x + 0.64 * box.width, y: box.y + 0.45 * box.height }, // Right eye
      { x: box.x + 0.50 * box.width, y: box.y + 0.60 * box.height }  // Nose bridge
    ];

    return {
      box,
      roi,
      corners,
      confidence: box.confidence
    };
  }

  resetTracking() {
    this.trackedBox = null;
  }
}
