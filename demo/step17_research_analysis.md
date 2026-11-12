# Step 17: Descriptive Research Analysis

Source CSV: `C:/Users/Jerry/Desktop/step16_final_results.csv`

Comparison: **Step 11 (10-epoch)** versus **Step 15 (improved)**

Dataset: `C:\Users\Jerry\Documents\GitHub\underwater-vision\data.yaml`  
Dataset split: `test`  
Images per condition: 63  
Image size: 640; batch: 4; confidence: 0.001; NMS IoU: 0.7

## Model comparison (percentage-point differences)

| Enhancement | Old mAP@50-95 | New mAP@50-95 | Difference (pp) |
|---|---:|---:|---:|
| Original (No Enhancement) | 0.0787 | 0.4212 | +34.25 |
| CLAHE | 0.0799 | 0.4121 | +33.23 |
| White Balance | 0.0722 | 0.4025 | +33.04 |
| Gamma Correction | 0.0788 | 0.4235 | +34.47 |

## Enhancement compared with original images

| Model | Enhancement | mAP@50-95 change (pp) |
|---|---|---:|
| Step 11 (10-epoch) | CLAHE | +0.12 |
| Step 11 (10-epoch) | White Balance | -0.65 |
| Step 11 (10-epoch) | Gamma Correction | +0.01 |
| Step 15 (improved) | CLAHE | -0.91 |
| Step 15 (improved) | White Balance | -1.87 |
| Step 15 (improved) | Gamma Correction | +0.23 |

## Interpretation

On the original images, the newer model had higher mAP@50-95 by 34.25 percentage points compared with the earlier model. This is an observed difference on this split, not proof of improvement on future data.

## Limitations and responsible reporting

- These eight rows contain *aggregate* metrics only. They cannot support p-values, confidence intervals, or claims of statistical significance.
- The training runs may differ in more than epoch count, so this comparison does not isolate the causal effect of training duration.
- If the test images informed earlier decisions, this is not a fully untouched final test set. Use a truly unseen holdout for final claims.
- Aquarium imagery is not representative of all open-water environments.
- Report failures, class-wise performance and qualitative examples alongside these aggregate metrics.
- For inferential statistics, retain image-level paired predictions or evaluation data and use a suitable image-level resampling design.

## Reproducibility

- Earlier checkpoint: `C:\Users\Jerry\Documents\GitHub\underwater-vision\training_runs\aquarium_training\weights\best.pt`
- Improved checkpoint: `C:\Users\Jerry\Documents\GitHub\underwater-vision\training_runs\step15_research2\weights\best.pt`
- Enhance the same images with the same parameters for each model.
- Preserve original results CSV, dataset version and software versions.
