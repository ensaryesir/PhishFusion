# PhishFusion: Hybrid Phishing Detection

<div align="center">

![Hybrid Detection](https://img.shields.io/badge/Phishing_Detection-Hybrid_(Visual_+_URL)-blue?style=flat-square)
![Swin Transformer](https://img.shields.io/badge/Model-Swin_Transformer-orange?style=flat-square)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green?style=flat-square)

<p align="center">
  <a href="https://www.usenix.org/conference/usenixsecurity22/presentation/liu-ruofan">Original Paper</a> •
  <a href="https://sites.google.com/view/phishintention">Dataset</a> •
</p>

</div>

**PhishFusion** is an advanced evolution of PhishIntention that employs a **Hybrid Vision-URL approach** for more accurate phishing detection. 

We have enhanced the original reference-based system by:
1.  **Replacing ResNet50 with Swin Transformer**: Utilizing a powerful hierarchical vision transformer (`modules/swin_siamese.py`) for the Siamese logo matching network, significantly improving feature extraction capabilities.
2.  **Adding URL Analysis**: Integrating a comprehensive heuristic and lexical URL scanner (`modules/url_analyzer.py`) to detect brand impersonation, typosquatting, and suspicious TLDs.
3.  **Adaptive Hybrid Fusion**: Combining visual confidence scores with URL risk assessments in `phishintention.py` to reduce false positives (e.g., legitimate login widgets) and improve detection on sophisticated attacks.

---

## 🚀 Installation & Setup

### Prerequisites
*   **Pixi** (Recommended package manager): [Install Pixi](https://pixi.sh/latest/)
*   **Chrome Browser** (for dynamic analysis)

### Step 1: Clone and Install
```bash
# Clone the repository
git clone https://github.com/ensaryesir/PhishFusion.git
cd PhishFusion

# Install dependencies using Pixi (creates a reproducible environment)
pixi install
```

### Step 2: Setup Models & Drivers
1.  **Model Weights**: 
    The core model `models/ocr_swin_siamese.pth` should be included in this repository. 
    > **Note**: This file is approx 110MB. If you have issues pulling it, ensure you have Git LFS installed (`git lfs pull`).

2.  **ChromeDriver**:
    *   Check your Chrome version: `chrome://version/`
    *   Download the matching [ChromeDriver](https://googlechromelabs.github.io/chrome-for-testing/)
    *   Place `chromedriver.exe` in the `chromedriver-win64/` folder (or `chromedriver-linux64/` on Linux).

---

## 💻 Usage

Run the main detection script using Pixi:

```bash
pixi run python phishintention.py --folder <target_folder> --output_txt <results_file.txt>
```

### Example
```bash
pixi run python phishintention.py --folder datasets/test_sites --output_txt results.txt
```

### Input Data Structure
The input folder should follow this structure for each website to be tested:
```text
datasets/test_sites/
├── test_site_1/
│   ├── info.txt    (Contains the URL, e.g., https://example.com)
│   ├── shot.png    (Screenshot of the webpage)
│   └── html.txt    (HTML source code, optional but recommended)
├── test_site_2/
│   └── ...
```

---

## 🛠 Project Components

We have modularized and improved the codebase:

| Component | File | Description |
|-----------|------|-------------|
| **Core Logic** | `phishintention.py` | Main script implementing the **Hybrid Fusion** logic (URL Risk + Visual Score). |
| **Vision Model** | `modules/swin_siamese.py` | **[NEW]** Swin Transformer-based Siamese network for OCR-aided logo matching. Replaces the older ResNetV2-50. |
| **URL Scanner** | `modules/url_analyzer.py` | **[NEW]** heuristic engine detecting brand impersonation, homographs, IP abuse, and more. |
| **Weights** | `models/ocr_swin_siamese.pth` | **[NEW]** Trained weights for the Swin Transformer model. |
| **Layout** | `modules/awl_detector.py` | Existing Object Detection model for finding regions of interest (Logos, Inputs). |

### Hybrid Fusion Logic
The system makes a final decision based on an **Adaptive Fusion Score**:

1.  **Visual Confidence**: How strongly does the screenshot logo match a known target (e.g., Microsoft)?
2.  **URL Risk**: Is the URL suspicious (e.g., `microsoft-verify.xyz`)?
3.  **Semantic Alignment**: Does the visible brand match the URL domain?

```python
# Pseudo-code of Fusion Logic
Fusion_Score = (Weight_Visual * Optical_Confidence) + (Weight_URL * URL_Risk_Score)
```

If the Visual Confidence is extremely high (>95%), we trust it more. If the URL is explicitly malicious (Risk > 0.8), it weighs heavier. This hybridization prevents false positives where visual logos exist but the URL is legitimate (e.g., "Login with Google" on a 3rd party site).

---
**Disclaimer**: This tool is for research and educational purposes only.
