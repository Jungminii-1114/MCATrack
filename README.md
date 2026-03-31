# 🚁 MCATrack (Memory-based Cross-Attention Tracker)

MCATrack is a PyTorch-based implementation of a robust object tracking model, specifically optimized for UAV (Unmanned Aerial Vehicle) tracking in complex environments. It leverages both **Appearance (Grayscale)** and **Temporal Motion (Magno-Motion)** features through a custom 2-channel ResNet backbone and a Dynamic Target Cross Guidance (DTCG) module.

## ✨ Key Features & Architecture

This repository implements the core tracking pipeline, breaking down the tracking problem into three main components:

### 1. Magno-Motion Module 🧠
Inspired by the magnocellular pathway in the human visual system, this module captures temporal motion while suppressing camera ego-motion.
* **Camera Ego-Motion Compensation:** Uses **ORB (Oriented FAST and Rotated BRIEF)** feature matching to calculate the Affine Transformation matrix between frames. If feature points are insufficient (e.g., clear sky backgrounds), it softly falls back to Phase Correlation.
* **Motion Map Generation:** Calculates the absolute difference between the aligned past frame and the current frame, integrating it with the historical memory map using an exponential moving average ($\alpha$).

### 2. 2-Channel Backbone 🧬
A modified `ResNet50` architecture. The first convolutional layer (`conv1`) is customized to accept a **2-channel tensor** `[1, 2, H, W]`:
* **Channel 1:** Raw Appearance (Grayscale Image)
* **Channel 2:** Motion Map (Output from Magno-Motion Module)

### 3. DTCG (Dynamic Target Cross Guidance) 🎯
A Cross-Attention mechanism that solves the template drifting problem. It fuses three distinct feature maps:
* **Initial Template ($Z_0$):** The pure, unchanging target from the first frame.
* **Dynamic Template ($Z_t$):** The recently updated appearance and motion of the target.
* **Search Region ($X_t$):** The wide-area crop of the current frame where the target is expected to be.
* DTCG uses Cross-Attention to enrich $Z_0$ with relevant recent features from $Z_t$, and performs Depth-wise Cross Correlation against $X_t$ to output the final prediction response map.

---

## 🔍 Special Feature: ORB Motion Compensation Visualization

UAV tracking datasets often suffer from severe camera shake. To ensure the generated "Motion Map" only reflects the UAV's movement (and not the camera's panning), the `Magno_Motion` module dynamically aligns the past frame to the current frame.

To better understand this process, I've included a standalone visualization script:
* **`orb_visualization.py`**: This script isolates the ORB feature matching and RANSAC process. Running this will show you exactly how keypoints are detected on the UAV/background and how the affine matrix ($M$) aligns the frames before the motion difference is calculated.

---

## 📁 Repository Structure



## References
* Paper : *"Tracking Tiny Drones against Clutter: Large-Scale Infrared Benchmark with Motion-Centric Adaptive Algorithm"* (Paper Link)[https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Tracking_Tiny_Drones_against_Clutter_Large-Scale_Infrared_Benchmark_with_Motion-Centric_ICCV_2025_paper.pdf]

## Appendix
