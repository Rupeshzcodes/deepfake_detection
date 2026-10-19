# 🛡️ Sentinel Defense: Enterprise Multi-Modal Deepfake Detection Suite

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Headless-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-00bcd4.svg)](https://developers.google.com/mediapipe)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An explainable, defense-in-depth deepfake verification system combining fine-tuned deep CNN-LSTM sequential architectures with deterministic multi-modal computer vision forensic heuristics.

---

## 📌 Executive Overview

Deepfake generation techniques (GANs, Diffusion Models, and Autoencoders) frequently cause standalone deep learning classifiers to suffer from generalization collapse and adversarial vulnerability when evaluated against unseen synthesis pipelines.

**Sentinel Defense** addresses these vulnerabilities via a **Dual-Engine Fusion Framework**:
1. **Deep Learning Core**: A hybrid **ResNet50 + Spatial-LSTM** architecture that evaluates both high-level semantic feature degradation and inter-patch spatial dependencies.
2. **Deterministic Forensic Heuristics**: An explainable computer vision pipeline inspecting biological plausibility (corneal reflections, eyelid aspect ratio), frequency spectra (2D FFT), compression history (Error Level Analysis), sensor noise variance (Laplacian operator), and surface texture gradients (Sobel operator).

By combining data-driven feature extraction with physical and mathematical signal validation, the system outputs calibrated manipulation probabilities accompanied by an interpretable forensic audit report.

```mermaid
flowchart TD
    A[Input Media File] --> B{Face Mesh Detection}
    
    %% Neural Stream
    A --> C[ResNet50 Spatial Backbone]
    C -->|7x7x2048 to 49x2048| D[Spatial Sequence LSTM]
    D --> E[Deep Learning Probability Score]
    
    %% Forensic Heuristics
    B -->|Face Detected: Human Mode| F[Module A: Biometric Eye & Glint Analysis]
    B -->|Multi-Face Topology| G[3D Head Pose SolvePnP]
    A --> H[Module B: 2D FFT Frequency Spectrum]
    A --> I[Module C: Error Level Analysis ELA]
    A --> J[Module F: Laplacian Sensor Noise Variance]
    A --> K[Module G: Sobel Texture Gradient Analysis]
    
    %% Fusion
    E & F & H & I & J & K --> L{Weighted Probability Fusion Engine}
    F -.->|Dead Eye Glint Discrepancy > 85%| M[⚠️ Biometric Veto Override: 99% Fake]
    L --> N[Streamlit Interactive Dashboard]
    M --> N
    N --> O[Visual Decomposition: Mesh / ELA / FFT / Noise / Texture]