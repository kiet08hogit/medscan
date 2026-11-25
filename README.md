# MedScan

CS 415 Computer Vision final project: identify medications from packaging / prescription-bottle images using classical image processing + OCR, then retrieve public label information from the openFDA Drug Label API.

**This is a university research / demonstration project, not a medical device.** Always verify identification against the original packaging. MedScan does not give personalized dosing advice.

## Research question

> How do different image preprocessing techniques affect medication label recognition accuracy under varying real-world conditions?

The experimental script (`evaluation.py`) compares Methods A–F on the same labeled images and reports name accuracy, strength accuracy, CER, WER, incorrect identification rate, and average runtime.

## System pipeline

1. **Input** — upload or dataset folder (JPG / JPEG / PNG)
2. **Preprocess** — resize, grayscale, blur, CLAHE, thresholding, edges, contours, perspective warp, morphology
3. **OCR** — Tesseract and/or EasyOCR extract text + boxes + confidences
4. **Match** — RapidFuzz scores candidates against a medication vocabulary; low-confidence / ambiguous results are rejected
5. **Lookup** — confirmed names are queried against openFDA (`https://api.fda.gov/drug/label.json`)
6. **Evaluate** — offline experiment runner writes CSV metrics and Matplotlib graphs

## Computer vision techniques

| Technique | Role |
|-----------|------|
| Resize | Speeds OCR on large phone photos without changing aspect ratio |
| Grayscale | Removes color packaging noise; OCR usually needs intensity only |
| Gaussian blur | Light denoising before thresholding |
| CLAHE | Local contrast boost for uneven lighting |
| Adaptive threshold | Local binarization when illumination varies across the label |
| Otsu threshold | Automatic global binarization for roughly bimodal histograms |
| Canny edges | Finds package boundaries |
| Contours | Approximates a quadrilateral packaging region |
| Perspective transform | Warps angled packages toward a frontal view (falls back if no reliable rectangle) |
| Morphology | Closing/opening to fill gaps or remove speckles in binary text |

Each technique is a separate function in `preprocessing.py` so methods can be ablated independently.

## Project structure

```text
MedScan/
├── app.py
├── preprocessing.py
├── recognition.py
├── medication_matching.py
├── medication_lookup.py
├── evaluation.py
├── utils.py
├── dataset/
│   ├── images/
│   └── ground_truth.csv
├── results/
│   ├── graphs/
│   └── metrics/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation (macOS, including Apple Silicon)

```bash
git clone https://github.com/kiet08hogit/medscan.git
cd medscan
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Install Tesseract (required for the Tesseract OCR path)

```bash
brew install tesseract
tesseract --version
```

EasyOCR runs on **CPU by default** (`gpu=False`) so no CUDA setup is required. The first EasyOCR run downloads model weights and may take several minutes.

## Run the demo app

```bash
streamlit run app.py
```

In the sidebar you can:

1. Upload or load a dataset image
2. Choose a preprocessing method
3. Choose Tesseract or EasyOCR
4. Inspect original vs processed images and intermediate CV steps
5. View OCR text, boxes, and confidences
6. Review medication candidates / scores
7. Confirm a medication and retrieve openFDA fields

## Collecting and labeling the dataset

1. Photograph real OTC / empty labeled packaging under varied conditions:
   - `normal_lighting`
   - `low_lighting`
   - `angled`
   - `blurry`
   - `curved_bottle`
   - `cluttered_background`
2. Save images as JPG/PNG in `dataset/images/` (do not commit personally identifying prescription photos).
3. Edit `dataset/ground_truth.csv` with one row per image:

| column | meaning |
|--------|---------|
| `filename` | exact file name in `dataset/images/` |
| `medication_name` | expected drug / brand name used for scoring |
| `strength` | e.g. `200 mg` |
| `expected_text` | approximate full label text for CER/WER |
| `condition` | imaging condition tag from the list above |

Example rows:

```csv
filename,medication_name,strength,expected_text,condition
ibuprofen_normal_01.jpg,ibuprofen,200 mg,Ibuprofen 200 mg Pain Reliever,normal_lighting
tylenol_lowlight_01.jpg,acetaminophen,500 mg,TYLENOL Acetaminophen 500 mg,low_lighting
advil_angled_01.jpg,ibuprofen,200 mg,Advil Ibuprofen 200 mg,angled
```

Expand `DEFAULT_MEDICATION_VOCABULARY` in `medication_matching.py` (or rely on ground-truth names, which are merged into the vocab during evaluation).

## Running experiments

After images + ground truth exist:

```bash
python evaluation.py
```

Optional flags:

```bash
python evaluation.py \
  --images dataset/images \
  --ground-truth dataset/ground_truth.csv \
  --metrics-dir results/metrics \
  --graphs-dir results/graphs
```

Outputs:

- `results/metrics/per_image_results.csv`
- `results/metrics/summary_by_method.csv`
- `results/metrics/summary_by_condition.csv`
- `results/graphs/*.png`

**Do not fabricate results.** If the dataset is empty, the script exits with an error instead of inventing metrics.

### Methods compared

| ID | Pipeline |
|----|----------|
| A | Original + Tesseract |
| B | Grayscale + Tesseract |
| C | Gaussian blur + adaptive threshold + Tesseract |
| D | CLAHE + Otsu + Tesseract |
| E | Perspective correction + preprocessing + Tesseract |
| F | Best preprocessing (perspective + CLAHE) + EasyOCR |

### Metrics

- **Medication name identification accuracy** — fraction of images whose top fuzzy match agrees with ground truth (soft RapidFuzz match)
- **Medication strength extraction accuracy** — whether regex-extracted strength matches ground truth
- **CER / WER** — character / word error rates vs `expected_text`
- **Incorrect identification rate** — accepted matches that are wrong (false accepts)
- **Average processing time** — preprocess + OCR wall time per image

## Tests

```bash
pytest -q
```

Covered: preprocessing transforms, text normalization, strength extraction, medication matching, CER/WER helpers.

## Current limitations

- Classical CV + OCR struggles on reflective plastic, severe blur, and heavily curved bottles
- Perspective correction only helps when a clear rectangular boundary is detectable
- Medication vocabulary is a small starter list; coverage must grow with your dataset
- openFDA fields vary widely by product; many OTC labels omit interactions / adverse-reaction sections
- No personalization, dosing calculator, or prescription-history storage (by design)
- Not validated for clinical use

## Privacy

Avoid photographing labels that show patient names, Rx numbers, or addresses. Dataset images are gitignored by default.

## References

- OpenCV documentation: https://docs.opencv.org/
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- RapidFuzz: https://github.com/rapidfuzz/RapidFuzz
- openFDA Drug Label API: https://open.fda.gov/apis/drug/label/
- Streamlit: https://docs.streamlit.io/
