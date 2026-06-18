# Background Removal Pipeline — Model V5

**Ensemble Segmentation: BiRefNet + rembg (IS-Net)**  
**Target:** Egyptian Monuments & Landmarks (Foreground Only — People Excluded)

A robust, production-ready background removal system specifically engineered for Egyptian archaeological artifacts and large architectural structures.

---

## 🎯 Objective

This pipeline serves as the **first stage** in the full Egyptian Artifact Recognition system. It removes complex backgrounds from user-captured images while preserving fine details of monuments, statues, temples, and artifacts.

The cleaned foreground images are then passed to the classification model (EfficientNet-B4 ensemble) and subsequently to the NLP description generator.

> **Upcoming Integration:** This module will be integrated into the mobile application. The user will upload a photo → it will automatically go through background removal → classification → contextual description.

---

## 🏗️ Pipeline Architecture

| Component                  | Model                     | Purpose                              |
|---------------------------|---------------------------|--------------------------------------|
| Primary Segmentation      | **BiRefNet** (ZhengPeng7) | Fine edges and complex boundaries    |
| Secondary Segmentation    | **rembg / IS-Net**        | Reliable general-purpose baseline    |
| Post-Processing           | Guided Filter + Morphology| Edge-preserving refinement           |
| Subject Selection         | LargeStructureSelector    | Keep largest monument structure only |

---

## 🔧 Key Features & Optimizations

- **VRAM Optimized (v5.1)**: SAM2 completely removed. BiRefNet forced to CPU to prevent CUDA OOM.
- **Large Structure Selector**: Removes tourists and small objects while keeping major monuments.
- **Ensemble Weighted Merge**: BiRefNet (60%) + rembg (40%).
- **Empty Mask Guard**: Skips images with insufficient foreground.
- **Checkpoint System**: Supports resuming after kernel crashes.
- **Comprehensive Logging & Reporting**: JSON report with full statistics.

---

## ⚙️ Configuration Highlights

- **BiRefNet Device**: CPU (forced for stability)
- **rembg Device**: CUDA when available
- **Input Resolution**: Original + internal 1024×1024 processing
- **Minimum Subject Ratio**: 5% of image area
- **Post-Processing**: Morphological operations + Guided Filter + Bilateral Filter

---

## 📁 Output Structure
Model 1/
├── results/              # Final transparent PNGs (_nobg.png)
├── masks/                # Grayscale alpha masks
├── comparisons/          # 3-panel visual comparison (Original | Mask | Result)
├── processing_report.json
└── checkpoint.json       # Resume support
text---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, Hugging Face Transformers, timm
- **Segmentation**: BiRefNet, rembg (IS-Net)
- **Image Processing**: OpenCV, Pillow, scikit-image, SciPy
- **Environment**: Anaconda + JupyterLab

---

## 📊 Performance Characteristics

- Designed specifically for **Egyptian museum and outdoor monument** photography
- Handles glass reflections, varying lighting, and complex backgrounds
- Preserves intricate details (carvings, hieroglyphs, statue features)
- Optimized for batch processing of large datasets

---

## 🔄 Integration Flow (Planned)
graph LR
    A[User Photo] --> B[Background Removal Pipeline]
    B --> C[EfficientNet-B4 + CLIP Ensemble]
    C --> D[NLP Description Generator]
    D --> E[Mobile App Response]

This background removal model is a critical component of the full-stack AI system for Egyptian Cultural Heritage Recognition.
