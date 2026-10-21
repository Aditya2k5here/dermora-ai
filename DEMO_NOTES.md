# Dermora AI — presentation notes

## Before you walk in

1. `run.bat` — wait for the self test to print **ALL CHECKS PASSED**.
2. Analyse one sample so the model is warm; the first inference is always the slow one.
3. Optional but worth it: run the evaluation once and have the numbers on hand.

   ```bat
   python evaluate.py --images D:/yolov8/dataset/images/val --labels D:/yolov8/dataset/labels/val
   ```

   Quote the accuracy figure it prints, not the mAP in the checkpoint — see "hard questions" below.
4. Keep `confusion_matrix.png` and a generated PDF report open in tabs as backup if the live demo misbehaves.

---

## Five-minute run

**0:00 — The problem.** Melanoma is ~1% of skin cancers but the majority of skin-cancer deaths, and outcome is almost entirely a function of how early it is caught. Dermoscopy is the screening tool; reading it well takes years of training.

**0:40 — The model.** YOLOv8n fine-tuned on ISIC 2018 Task 3 — 10,015 dermoscopic images, seven diagnostic classes. Open the **Model** tab: the metrics, hyper-parameters and 50-epoch training curve on that screen are read live out of `best.pt`, not typed into a slide.

**1:30 — A case.** Drop a sample. Talk over the four progress stages — quality gate, inference, attention, morphometry. Land on the result and read the top line: class, confidence ring, triage band.

**2:15 — The honest part.** Point at the four metric tiles. "Confidence alone is not enough to act on, so the system also reports how far ahead the winning class is, whether all four augmented views agreed, and whether the image was even good enough to analyse." Switch image layers with `1`–`4`: attention map, original, segmentation contour, detector ROI.

**3:15 — The second opinion.** Scroll to Morphometry. "This is the 1994 Stolz ABCD rule computed with classical image processing, completely independent of the network. When the two disagree, the case is flagged for human review rather than quietly resolved."

**4:00 — Scale.** Triage tab, drop a folder. Worklist ordered urgent first. "This is the actual clinical use case — not diagnosing one lesion, but ordering a queue."

**4:40 — Close.** Download the PDF report. Finish on the disclaimer: decision support, not a diagnosis.

---

## Hard questions, and honest answers

**"Your checkpoint says mAP@50 of 0.81 — is that your accuracy?"**
No, and it is important not to claim it is. Every training label is a whole-image box, so IoU is trivially satisfied and mAP mostly measures whether the right class scored highest on a box that was never in doubt. That is why `evaluate.py` exists: it reports top-1 classification accuracy, balanced accuracy and per-class recall, which are the numbers that mean something here.

**"Why is precision only 0.59 when recall is 0.87?"**
Because the model is confident on multiple classes per image and detection precision punishes every extra box. It is also the right trade for cancer triage: the cost of a missed melanoma vastly exceeds the cost of an unnecessary dermatology referral, and the triage thresholds are deliberately tuned in that direction.

**"A detection model for classification — isn't that the wrong architecture?"**
Yes, strictly. It was inherited from the earlier iteration. Rather than pretend otherwise, the app exploits the one advantage it gives — a per-anchor, per-class score grid — by pooling the top-k anchors per class instead of taking whatever survived NMS. That recovers a real posterior from an architecture that was not designed to produce one. A classification head fine-tuned on the same data would be the honest next step, and it is listed as future work.

**"How do I know it looked at the lesion and not the background?"**
The attention layer, plus the concentration statistic underneath it: the fraction of saliency mass inside the top decile of the map. Diffuse attention gets called out in the Flags tab rather than hidden.

**"What about darker skin?"**
HAM10000 is overwhelmingly Fitzpatrick I–III. Performance on darker skin is uncharacterised, and that is stated on the model card in the app rather than buried. It is the single biggest limitation of this dataset.

**"Is this usable clinically?"**
No. It is an academic prototype with no regulatory clearance, and the interface says so on every screen and in every exported report.

---

## If something breaks live

| Symptom | Fix |
|---|---|
| Status dot stuck amber | Model still loading — the checkpoint takes a few seconds. Wait, don't reload. |
| "Model unavailable" | `best.pt` missing from `model\`. Copy it back, restart. |
| Toast says "degraded mode" | The raw-head path failed on this ultralytics build; results still render from the fallback. Say so plainly and move on. |
| Upload rejected | Not an image, over 25 MB, or under 48 px. |
| Port already in use | `python app.py --port 8010` |
| Everything is on fire | Fall back to the pre-generated PDF report and `confusion_matrix.png`. |

---

## What to say if asked "what would you do next"

1. Fine-tune a proper classification head (EfficientNet-B0 or a ViT-small) on the same split and ensemble it with the detector — the architecture mismatch is the biggest single source of avoidable error.
2. Temperature-scale the posterior on a held-out set so that a stated 80% confidence is empirically 80% correct.
3. Add serial-image comparison to cover the E of ABCDE, which no single-image system can assess.
4. Evaluate on Fitzpatrick IV–VI data before making any claim about general use.
