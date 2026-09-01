# Cek Vital Dari Wajah (rPPG Heart Rate Monitor)

Real-time, client-side Remote Photoplethysmography (rPPG) heart rate estimation running entirely in the browser using ONNX Runtime Web and WebAssembly. Zero-build static architecture deployable directly to GitHub Pages.

## Features
- **In-Browser ML Face Detection**: Ultra-fast face tracking via ONNX Runtime Web (`version-RFB-320.onnx` / YuNet).
- **Exact heartbeat-master DSP Pipeline**:
  - Forehead Region of Interest (ROI) extraction.
  - Green-channel mean sampling with jump denoising.
  - Smoothness-priors detrending ($\lambda = \text{fps}$) & moving average filter.
  - Real DFT power spectrum analysis with $42\text{--}240\text{ BPM}$ bandpass masking.
- **Reference GUI Overlay**:
  - Red face bounding box.
  - Green forehead ROI rectangle & tracking crosshairs.
  - Live BPM & FPS overlay text.
  - Dual red waveform traces: Time-domain pulse signal & FFT power spectrum.

## Quick Start
To run locally:
```bash
# Start a local static file server
python3 -m http.server 8080
```
Open `http://localhost:8080` in Chrome/Safari/Firefox and click **Start Measurement**.

## GitHub Pages Deployment
1. Push repository to GitHub.
2. Go to **Settings > Pages**.
3. Under **Branch**, select `main` / `root` and click **Save**.