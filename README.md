# Improved Document Image Deduplicator

A powerful tool to detect duplicate or near-duplicate images inside **PDF** and **Word (.docx)** documents. It combines multiple image comparison techniques – perceptual hashing, ORB feature matching, template matching, structural similarity (SSIM), and optional deep learning features – to reliably find even visually similar or partially copied images, and generates a comprehensive PDF report with visual evidence.

## Features

- **Multi‑format support**: Extracts all images from PDF (via `PyMuPDF`) and Word documents (via `python-docx`).
- **Multiple detection strategies**:
  - **Perceptual hashing (pHash)** – fast and robust against compression/resizing.
  - **Average hashing (aHash)** – quick pre‑filtering.
  - **ORB feature matching** – finds geometric correspondences and area of duplication.
  - **Template matching** – detects large duplicated regions.
  - **SSIM** – evaluates structural similarity.
  - **Optional deep learning** – uses ResNet50 (via PyTorch) to extract semantic features for additional verification.
- **Advanced duplicate validation**:
  - Computes the *ratio* of matched area to image size to avoid false positives from small texture matches.
  - Uses Lowe’s ratio test and RANSAC homography to filter outliers.
- **Visual report**: Generates a polished PDF with a page per duplicate pair, showing composite images where matching keypoints are connected by green lines.
- **Configurable thresholds**: All detection parameters can be fine‑tuned via command‑line arguments.

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/lolohhhh123/Image-Duplicate-Detection.git
   cd Image-Duplicate-Detection
   
Install dependencies
It is recommended to use a virtual environment.

bash
pip install -r requirements.txt
Note: The deep learning model (ResNet50) requires torch and torchvision. If you have a CUDA‑capable GPU, install the appropriate version from PyTorch official site for better performance. The tool will fall back to hashing+ORB if PyTorch is not available.

---
##Usage
Basic command:

bash
python IDD.py <input_file> [options]
<input_file> – path to a .pdf or .docx document.
--output, -o – output PDF report path (default: improved_report.pdf).
--phash_threshold – pHash Hamming distance threshold (default: 5, lower = stricter).
--ahash_threshold – aHash distance threshold (default: 10).
--min_matches – minimum number of ORB inlier matches (default: 20).
--min_area – minimum matched area ratio (0‑1) (default: 0.2).
--min_ssim – minimum SSIM score (0‑1) (default: 0.8).
---

##Example
bash
python IDD.py my_report.docx --output duplicate_report.pdf --phash_threshold 3 --min_area 0.15
This will scan my_report.docx, extract all images, detect duplicates using a stricter pHash threshold (3) and a lower area requirement (0.15), and save the report as duplicate_report.pdf.
---

##How It Works
Image extraction – All embedded images are extracted along with their page/position metadata.
Hash computation – pHash and aHash are calculated for fast initial screening.
Pairwise comparison (in order of increasing computational cost):
If pHash distance ≤ threshold → considered duplicate (quick exit).
Else if aHash distance ≤ threshold → run ORB+template matching for verification.
If deep learning is enabled and semantic similarity is high → run ORB verification.
For remaining pairs with similar aspect ratios → direct ORB+template matching + SSIM.
Duplicate criteria – A pair is reported if it satisfies at least one of the above pathways and passes the feature‑based verification (≥ minimum matches and area ratio).
Report generation – For each duplicate pair, a page is created containing metadata, detection method, and a composite image where matching inliers are drawn as lines.
---

##Output Report
The generated PDF report contains:
A summary page with total duplicate pairs.
One page per duplicate pair with:
Image indices and source pages.
Detection method and key metrics (hash distances, number of matches, area ratio, SSIM, deep similarity if used).
A composite image (if feature matches exist) showing the two images side‑by‑side with green lines connecting corresponding keypoints.
---

##Limitations & Notes
Large documents with many images may take time; adjust thresholds to balance speed and accuracy.
The deep learning feature extraction uses a pretrained ResNet50 and may require a GPU for speed.
For PDFs, only raster images embedded in the file are extracted; vector graphics are not handled.
The tool does not modify the original document; it only produces a report.
