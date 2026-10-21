# Dermora AI

Dermoscopic lesion analysis, classification and triage support, built on a
YOLOv8n model fine-tuned on ISIC 2018 Task 3 (HAM10000).

Seven classes: **MEL**, **NV**, **BCC**, **AKIEC**, **BKL**, **DF**, **VASC**.

---

## Run it

```bat
run.bat
```

That reuses the existing `D:\yolov8\.venv` (which already has torch and
ultralytics), installs the four small packages the app adds, runs the self test
and opens <http://127.0.0.1:8000>.

Manual equivalent:

```bat
pip install -r requirements.txt
python selftest.py          :: verifies every stage of the pipeline
python app.py               :: http://127.0.0.1:8000
```

Options: `python app.py --port 9000 --no-tta --imgsz 320`

---

## What it does

| Stage | What happens |
|---|---|
| **Quality gate** | Focus, exposure, contrast, glare and skin-tone fraction. An unusable image is flagged before it can produce a confident-looking answer. |
| **Classification** | Raw detection-head scores are pooled over the top-k anchors per class, mapped to logits and softmaxed into a true 7-way posterior, averaged over 4-view dihedral test-time augmentation. |
| **Uncertainty** | Top-two margin, normalised entropy, TTA view agreement and peak detector score, all surfaced rather than hidden. |
| **Explainability** | Eigen-CAM over the P3/P4/P5 neck features, plus a concentration statistic that says whether attention was focal or diffuse. |
| **Morphometry** | DullRazor hair removal, Otsu lesion segmentation, then the ABCD rule of dermoscopy (Stolz) scored into a Total Dermoscopy Score — computed independently of the network, so it can disagree with it. |
| **Triage** | Malignant + pre-malignant probability mass mapped onto URGENT / PRIORITY / REVIEW / ROUTINE. |
| **Reporting** | One-click PDF case report; CSV export of the session; batch mode for worklist triage. |

---

## Why this is not the previous version

The checkpoint is a *detection* model trained with whole-image boxes
(`class 0.5 0.5 1.0 1.0`), i.e. a classifier wearing a detector's clothes. The
earlier app consumed it like this:

```python
boxes = results[0].boxes
if len(boxes) == 0:
    return error("No lesion detected")     # hard failure on a valid image
cls  = int(boxes.cls[0])                   # NMS already discarded 6 classes
probs = [0]*7; probs[cls] = conf           # a fabricated "distribution"
```

Three consequences: the differential diagnosis was destroyed by NMS, the
"confidence distribution" chart was a one-hot vector dressed up as a histogram,
and any image scoring below the 0.25 NMS threshold produced an error instead of
an answer.

Dermora reads the head **before** NMS, so all seven classes keep their mass, no
input can hard-fail, and the numbers on screen are real. `evaluate.py` measures
the difference on a labelled split.

---

## Measure it

```bat
python evaluate.py --images D:/yolov8/dataset/images/val --labels D:/yolov8/dataset/labels/val
```

Prints top-1 accuracy, balanced accuracy, macro F1, malignant sensitivity,
per-class recall and a calibration table — and reruns the old post-NMS approach
on the same images for a head-to-head comparison. Writes `evaluation.json` and
`confusion_matrix.png`.

---

## Layout

```
dermora-ai/
├── app.py              FastAPI server and API
├── selftest.py         pre-flight check — run this before demoing
├── evaluate.py         accuracy on a labelled split + baseline comparison
├── dermora/
│   ├── engine.py       raw-head inference, top-k pooling, TTA, posterior
│   ├── explain.py      Eigen-CAM attention mapping
│   ├── morphology.py   hair removal, segmentation, ABCD rule, quality gate
│   ├── knowledge.py    clinical reference data and triage thresholds
│   ├── cardinfo.py     torch-free checkpoint introspection
│   ├── pipeline.py     orchestration and the JSON contract
│   └── report.py       PDF case report
├── static/             interface (no framework, no CDN, works offline)
├── model/best.pt       the trained checkpoint
└── samples/            example dermoscopic images
```

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | service and model status |
| GET | `/api/model-card` | checkpoint provenance, metrics, per-epoch history, taxonomy, limitations |
| POST | `/api/analyze` | multipart image → full case |
| POST | `/api/analyze-sample?name=` | analyse a bundled sample |
| POST | `/api/batch` | up to 40 images → triage-ordered worklist |
| GET | `/api/cases` | session history |
| GET | `/api/case/{id}` | one case |
| GET | `/api/case/{id}/report.pdf` | PDF report |
| DELETE | `/api/cases` | clear the session |

Interactive docs at `/docs`.

Keyboard: `1`–`4` switch image layers, `U` upload, `R` download report, `Esc` reset.

---

## Limitations

- Trained on HAM10000, which skews heavily towards Fitzpatrick I–III skin.
  Performance on darker skin is uncharacterised.
- Severe class imbalance: 66.9% nevi against 1.1% dermatofibroma.
- Validation metrics in the checkpoint are detection metrics against whole-image
  boxes, which is why mAP@50 is 0.81 while precision is 0.59. Use `evaluate.py`
  for classification metrics that actually mean something.
- Dermoscopic images only. Ordinary phone photographs are out of distribution.

**Dermora AI is an academic prototype. It is not a medical device, has no
regulatory clearance, and must not be used to make or defer treatment
decisions.**
