# XMF: BanglaASD-XMF

**Early Autism Detection using Voice Recognition, Face Verification, and Cry in Bangladesh**
(UGC research proposal, Dept. of CSE, Comilla University), carried out as in the *Q1 Master Plan*.
The goal is an explainable, missing-modality-robust fusion of facial-expression video, voice, cry and
questionnaire/text, used for autism **screening** (decision support, never a diagnosis), with a clinician-facing app.

## 1. Architecture

```mermaid
flowchart TB
  subgraph DATA["Stage 1: four unified datasets (builder notebooks D1-D4)"]
    D1["D1 XMF-Photo<br/>FER2013 + RAF-DB + CK+<br/>7 expressions, pHash de-dup"]
    D2["D2 XMF-Audio<br/>ReCANVo + Donate-a-Cry + infant-cry<br/>MD5 de-dup, log-mel/MFCC/eGeMAPS/wav2vec2/AST/CNN14"]
    D3["D3 XMF-Video<br/>MMASD skeleton sequences<br/>+ eye-tracking gaze (NB13)"]
    D4["D4 XMF-Text<br/>TASD statements<br/>+ Q-CHAT-10/AQ-10 (leakage audit)"]
    D0["D0 BanglaASD cohort<br/>consented Bangladeshi children<br/>(after ethics approval)"]
  end
  subgraph ENC["Stage 2: branch encoders, proposed vs baselines (P1, A1, V1, T1)"]
    P["Face: XMF-Face<br/>EfficientNet-B0 + 256-d embedding<br/>vs ResNet, EfficientNet, MobileNet, ConvNeXt,<br/>ViT, DeiT, Swin, YOLOv8-cls, YOLOv8-ATT"]
    A["Voice/Cry: XMF-Audio<br/>gated wav2vec2 + eGeMAPS + MFCC<br/>vs MFCC-SVM, eGeMAPS-LR, wav2vec2/AST/CNN14-LR,<br/>log-mel CNN, CNN-LSTM"]
    V["Video: XMF-Temporal<br/>2-layer transformer + attention pooling<br/>vs MLP, 1D-CNN, BiLSTM, CNN-LSTM"]
    T["Text: XMF-Text<br/>XLM-R + 256-d projection + attention pooling<br/>vs TF-IDF LR/SVM, BiLSTM, DistilBERT, BERT, RoBERTa"]
  end
  subgraph FUS["Stage 3: fusion and clinical evaluation (NB00-NB12)"]
    MT["learned missing-modality tokens"]
    CA["bidirectional cross-modal attention<br/>face <-> voice/cry"]
    PO["clip -> child attention pooling"]
    G["gated fusion"]
    EV["evidential head<br/>P(ASD) + uncertainty -> low / high / refer"]
    MT --> CA --> PO --> G --> EV
    BL["fusion baselines: early concat, late average, stacking, TFN, LMF, MulT"]
  end
  D1 --> P
  D2 --> A
  D3 --> V
  D4 --> T
  P --> MT
  A --> MT
  V --> MT
  T --> MT
  D0 --> FUS
  EV --> X["XAI: modality Shapley, IG over time, SHAP/TCAV behaviours,<br/>Grad-CAM family with faithfulness + sanity checks"]
  EV --> S["Statistics: child-wise CV, locked test, 95% CIs, DeLong, McNemar,<br/>calibration, decision curves vs M-CHAT, fairness"]
  EV --> APP["App export: ONNX FP16/INT8"]
```

## 2. Notebook order (runs automatically, one at a time)

| stage | notebook | what it does |
|---|---|---|
| 1 | D1 Photo / D2 Audio / D3 Video / D4 Text | build the four unified datasets (manifest + datasheet each) |
| 2 | P1 / A1 / V1 / T1 | **proposed model vs baselines** in each sector, same splits and budget for all |
| 2 | NB13 | gaze: ASD vs TD on real eye-tracking data, child-wise CV, shortcut baselines |
| 3 | NB00 to NB12 | cohort pipeline: audit and splits, face and audio pipelines, frame and video baselines, **XMF vs 6 fusion baselines**, ablations and shortcut tests, XAI, statistics, app export |

**Until the consented cohort exists, Stage 3 runs on a clearly labelled synthetic DEV cohort.** That only proves
the pipeline works, and none of its numbers are results.

## 3. Protocol for every comparison
* The same splits, image size, epoch budget and seeds for all models. Subject-, speaker- or child-wise splits wherever IDs exist.
* Early stopping on validation; the test data is used once. Label smoothing, weight decay, augmentation and mixup.
* Over/underfitting is reported for every model as train vs validation accuracy, the gap, and learning curves.
* 95% bootstrap CIs, McNemar or DeLong against the proposed model, with Holm correction.
* The scraped "Autistic Children Facial Data Set" and its mirrors are **not used** (38+ retractions).

## 4. How it runs
`.github/workflows/queue.yml` wakes every 20 minutes. It starts the next Kaggle notebook when the previous one has
finished, and commits each finished notebook's tables, figures and logs to `outputs/<notebook>/`.
It needs one repository secret, `KAGGLE_API_TOKEN` (Settings → Secrets and variables → Actions).

## 5. Layout
`asd_lib/` shared code · `notebooks/src/` notebook sources · `kaggle_build/` built notebooks ·
`tools/` queue and sync tools · `outputs/` results, filled in automatically.
