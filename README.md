# Solar-IQ: A Robust Solar Panel Detector

**Proof-of-Concept:** A robust solar panel detector built with YOLOv8, designed to perform reliably in real-world outdoor conditions with glare, shadows, and clutter. This project was built and tested in one day.

---

## 🚀 Live Demo

This 30-second demo shows the final ONNX model running inference on a test image.

*(Embed your 30-second Loom video or a GIF of it here)*

---

## The Problem

Standard computer vision models are brittle and often fail when deployed outdoors. Environmental factors like sun glare, varied lighting, and complex backgrounds cause significant performance degradation, making them unreliable for autonomous systems like the RoboForce TITAN.

This project serves as a proof-of-concept for a robust perception pipeline that directly addresses these challenges.

## ✨ Key Features & Performance

* **Model:** Fine-tuned `YOLOv8n` on a real-world dataset of 621 outdoor images.
* **Performance:** Achieved **~76 mAP@0.5** on the validation set after 30 epochs.
* **Deployment Ready:** Exports to a portable **ONNX** format, ready for optimization with TensorRT for deployment on edge GPUs.
* **Code Quality:** Written in strict-typed Python (`Pylance --strict`), formatted with Black, and linted with Ruff.
* **Reproducible Workflow:** Managed with a `Makefile` for consistent setup, training, and execution.

---

## 🛠️ Setup & Usage

### Prerequisites
* Python 3.10+
* An NVIDIA GPU with CUDA installed
* Configured Kaggle API credentials (`~/.kaggle/kaggle.json`)

### 1. Installation

Clone the repository and install the required dependencies using the Makefile.

```bash
git clone [Your-Repo-URL]
cd Solar-IQ
make install
2. Data Acquisition
Download and unzip the dataset from Kaggle using the Makefile.

Bash

make download
This will create the data/solar_bb/ directory with the required images/ and labels/ structure.

3. Training
Train the model for 30 epochs. Results will be saved in the runs/ directory.

Bash

make train
4. Inference Demo
Run the demo script on a test image using the exported ONNX model. The output will be saved as output.jpg.

Bash

make demo
🗺️ Project Roadmap
This POC establishes a strong baseline. The next steps to create a production-grade system include:

Advanced Augmentations: Integrate a custom albumentations pipeline to further harden the model against specific environmental conditions like extreme glare and dust.

Hard-Negative Mining: Introduce a set of "negative" images (rooftops without panels, skylights, etc.) to reduce the false-positive rate.

Full TensorRT Optimization: Complete the TensorRT installation to compile a .engine file for maximum FPS on the target hardware.

Continuous Integration: Set up GitHub Actions to automatically lint, test, and validate the codebase on every commit.