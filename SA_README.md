# SkinAnalytica

ISIC-grade dermoscopy AI platform built on NeuroScope AI infrastructure.

## Architecture

Three-model ensemble: EfficientNetV2-L + ViT-L/16 + ConvNeXt-Large  
Training: HAM10000 + ISIC 2019 + ISIC 2020 (~68,000 images)  
Loss: Enhanced Focal Loss + WeightedRandomSampler  
Augmentation: Mixup + CutMix + RandAugment + hair removal  
Calibration: Temperature scaling  
Explainability: Grad-CAM++ + SHAP + TIxAI (2025)

## Notebooks

| Notebook | Purpose |
|---|---|
| SA01_Model_Training.ipynb | Train each backbone (run 3 nights) |
| SA02_Ensemble.ipynb | Combine models, calibrate, export |
| SA03_Explainability.ipynb | Grad-CAM++, SHAP, TIxAI |
| SA04_Research_Module.ipynb | ISIC bulk analysis, bias reports |
| SA05_api.py | FastAPI endpoints |
| SA06_ONNX_Export.ipynb | ONNX + INT8 quantisation |

## Datasets

Place datasets in:
```
datasets/
  ham10000/images_part1/     <- HAM10000 part 1
  ham10000/images_part2/     <- HAM10000 part 2
  ham10000/metadata/         <- HAM10000_metadata.csv
  isic_2019/images/          <- ISIC 2019 training images
  isic_2019/metadata/        <- ISIC_2019_Training_GroundTruth.csv
  isic_2020/images/          <- ISIC 2020 JPEG images
  isic_2020/metadata/        <- train.csv
  ph2/images/                <- PH2 (validation only)
```

Downloads: https://challenge.isic-archive.com/data/

## Run order

Night 1: SA01 with MODEL_NAME = "tf_efficientnetv2_l"  
Night 2: SA01 with MODEL_NAME = "vit_large_patch16_224"  
Night 3: SA01 with MODEL_NAME = "convnext_large"  
Day 4:   SA02 → SA03 → SA04 → SA06  

## API

```bash
uvicorn main:app --port 8001
```

POST /analyze        - single image  
GET  /report/{id}    - fetch result  
GET  /report/{id}/fhir - FHIR export  
POST /webhooks/tula  - Tula integration  

## Versions

V1: 2018-2020 archive (current)  
V2: ISIC 2024 SLICE-3D (post-Paul meeting)  
V3: ISIC 2025 MILK10K longitudinal (Q3 2026)
