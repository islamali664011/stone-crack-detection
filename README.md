# Stone Crack Detection in the Complex of Sultan Faraj ibn Barquq

## Project Overview

This project presents a computer vision-based approach for detecting structural deterioration in historic Islamic architecture, specifically the complex of Sultan Faraj ibn Barquq in Cairo.

The system analyzes a collection of pre-uploaded images (via Google Colab) to automatically identify:

* Stone cracks
* Surface protrusions (irregularities)
* Gaps and joints between stones

The goal is to support digital heritage conservation by providing fast, automated, and scalable damage assessment.

---

## Objectives

* Detect micro and macro cracks in stone surfaces
* Identify morphological irregularities such as protrusions
* Analyze joints and separations between stone blocks
* Compute quantitative metrics for structural condition assessment

---

## Methodology

The system combines multiple computer vision techniques:

### 1. Image Preprocessing

* Grayscale conversion
* Gaussian blur for noise reduction
* CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhancement

### 2. Crack Detection Techniques

* **Canny Edge Detection** → for sharp crack boundaries
* **Morphological Black-Hat Transformation** → for thin dark cracks
* **Gradient-Based Detection (Sobel)** → for intensity variations

### 3. Post-processing

* Morphological filtering
* Noise removal using connected component analysis
* Crack segmentation refinement

---

## Output Metrics

For each image, the system computes:

* Crack density (%)
* Number of crack segments
* Total crack pixels

---

##  Dataset

The dataset consists of a curated set of images from:

**The Complex of Sultan Faraj ibn Barquq (Cairo, Egypt)**

Images are loaded from Google Drive via Google Colab.

> Note: Due to size and ownership, the dataset is not fully included in this repository.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Run on a folder of images:

```bash
python src/batch_process.py
```

### Run on a single image:

```bash
python src/single_image.py
```

---

## Environment

* Python 3.x
* OpenCV
* NumPy
* Matplotlib

---

## Applications

* Digital heritage conservation
* Structural condition assessment
* Archaeological documentation
* AI-assisted restoration planning

---

## Future Work

* Integration with machine learning models
* Crack severity classification
* Structural failure prediction
* 3D reconstruction of damaged areas

---

## Author
Islam Ali Muhammed

Developed as part of an interdisciplinary approach combining:

* Islamic architectural studies
* Computer vision
* Python-based analysis

---

##  License

This project is for academic and research purposes.
