<div align="center">

# 🌊 Ocean Studio
### Underwater Vision AI — Image Enhancement, Object Detection & Research Evaluation

**A Python desktop research workstation for exploring underwater image enhancement and YOLO-based object detection.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Qt](https://img.shields.io/badge/GUI-PySide6-41CD52?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython-6/)
[![OpenCV](https://img.shields.io/badge/Image_Processing-OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Ultralytics](https://img.shields.io/badge/Object_Detection-Ultralytics_YOLO-111F68)](https://docs.ultralytics.com/)

[Getting Started](#getting-started) · [Features](#features) · [Dataset](#included-dataset) · [Training](#fine-tuning-yolo) · [Evaluation](#evaluation-and-research-workflows) · [Troubleshooting](#troubleshooting)

</div>

---

> [!NOTE]
> **Repository-specific documentation.** This README is based on the public files in [`jrbMods/underwater-vision`](https://github.com/jrbMods/underwater-vision), reviewed on **12 November 2026**. The repo currently contains the app source, dataset folders/configurations, training outputs, and `yolo26n.pt`. Some optional branding assets and Windows helper scripts described below are **not currently committed** to the repository.

## Overview

Underwater computer vision is difficult because underwater photographs can suffer from low contrast, color casts, reduced visibility, and uneven lighting. **Ocean Studio** brings together traditional image enhancement and modern object detection in one desktop interface. It supports both visual exploration and quantitative research, including YOLO fine-tuning and labeled-dataset evaluation.

The application is implemented mainly in [`main.py`](main.py), with a **PySide6** graphical interface, **OpenCV** image transformations, **NumPy** for image data, and **Ultralytics YOLO** for model training, inference, and validation.

The main research question motivating the workflow is:

> **How do different underwater image-enhancement methods affect the performance of a YOLO object detector?**

The software provides tools to investigate that question. **A higher-contrast image does not automatically produce more accurate detections**, so experiments should compare predictions with labeled ground truth.

### Intended uses

- Explore the appearance of underwater images under different enhancement algorithms.
- Detect objects using pretrained or custom-trained YOLO checkpoints.
- Audit and prepare labeled underwater datasets.
- Fine-tune YOLO for a dataset's object classes.
- Compare detection behavior and accuracy across enhancement methods.
- Export figures and tabular results for academic reports.

## Screenshots

The current repository does **not** yet include a screenshots folder. To display real screenshots here, add the following images to `docs/images/` and remove the HTML comments below.

<!-- Once the files are uploaded, uncomment these image references:
![Ocean Studio main workspace](docs/images/main-workspace.png)
![Underwater image enhancement comparison](docs/images/enhancement-comparison.png)
![YOLO underwater detection](docs/images/detection.png)
![Accuracy comparison dashboard](docs/images/accuracy-report.png)
-->

Suggested captures: **main workspace**, **four-method enhancement comparison**, **detection using a custom checkpoint**, and **accuracy/report view**. Use genuine application screenshots and remove any private paths or contact details you do not want publicly visible.

## Features

### Image enhancement

Ocean Studio supports four analysis conditions:

| Method | Description | Adjustable setting |
| --- | --- | --- |
| **Original** | Unmodified image, used as the reference condition. | None |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization on the LAB lightness channel. | Clip limit; default **2.0** |
| **White Balance** | Gray-world channel-balancing correction blended with the original. | Correction strength; default **100%** |
| **Gamma Correction** | Lookup-table intensity transformation. | Gamma; default **0.80** |

The operations are applied to the **original source image** rather than repeatedly applying each effect to the previous result. This makes comparisons more meaningful and avoids unintentionally accumulating transformations.

The application can present the four image versions together and export a comparison montage as PNG.

### YOLO object detection

- Use the supplied `yolo26n.pt` checkpoint for pretrained detection.
- Select a custom `.pt` checkpoint, for example a fine-tuned `best.pt`.
- Adjust the confidence threshold.
- Display detection bounding boxes and class names.
- Export annotated images.
- Try a low-confidence diagnostic detection when a custom model returns no boxes.
- Compare detections from the **same model** on Original, CLAHE, White Balance, and Gamma images.

> [!IMPORTANT]
> A standard pretrained YOLO checkpoint and an Aquarium-specific fine-tuned checkpoint can recognize **different class sets**. Their output counts should not be treated as a direct measure of relative accuracy. Use compatible labeled data and formal evaluation to compare performance.

### Batch processing and research reports

The application can process folders of images with the supported enhancement methods and detector, record detection counts, average detection confidence, and prediction timing, and export results in CSV format. Other reporting views visualize or summarize these measurements.

These measurements describe model behavior. **Detection count and average confidence are not substitutes for precision, recall, or mAP.**

### Labeled-dataset auditing

The dataset-checking workflow reads a YOLO dataset YAML file and inspects available splits and annotations. It can summarize:

- Image and label-file counts by split.
- Bounding-box counts and class distribution.
- Missing folders or inconsistent annotations.
- Warnings and potential formatting problems.

It can export the audit to CSV. An audit helps detect data problems; it is **not** a measure of detection accuracy.

### Training and controlled experiments

The app includes interfaces for YOLO training and a more deliberate training experiment with settings such as maximum epochs, batch size, image size, patience for early stopping, random seed, and compute device. Training outputs can include `best.pt`, `last.pt`, `results.csv`, and an experiment JSON record.

The research workflow also supports:

1. Evaluating a model against labeled validation or test images.
2. Comparing four enhancement conditions using one trained model.
3. Comparing **two** trained models across **four** enhancement conditions (eight evaluations).
4. Exporting metric summaries, accuracy graphs, and Markdown analysis reports.

### Interface and application logging

The current `main.py` implements an Ocean Studio-themed interface with model controls, research tabs, image previews, an optional branded workspace logo, and a **fixed blurred background image** loaded from the assets directory. A **terminal/logs** action provides access to application output; on Windows it can open a PowerShell window following the live log.

The application writes logs to `logs/ocean_studio.log` by default, with a home-directory fallback if the project directory is not writable.

## Technology stack

| Component | Technology | Role |
| --- | --- | --- |
| Language | Python | Application logic and experiments |
| UI | PySide6 / Qt | Desktop windows, controls, dialogs, reports |
| Image processing | OpenCV (`cv2`) | Color conversion, enhancement, image input/output |
| Numerical operations | NumPy | Pixel arrays and numerical calculations |
| Deep learning / detection | Ultralytics YOLO | Prediction, training, validation |
| ML runtime | PyTorch (Ultralytics dependency) | CPU/GPU execution |
| Data exchange | YAML, CSV, JSON, PNG | Dataset configuration, reports, experiments, images |

GPU use depends on your device, driver, and PyTorch installation; a compatible CUDA setup is required for NVIDIA acceleration.

## Repository structure

The following reflects the **currently visible top-level files and directories** on GitHub:

```text
underwater-vision/
├── datasets/
│   └── coco128/
├── test/
├── train/
├── training_runs/
├── valid/
├── .gitattributes
├── coco128.yaml
├── data.yaml
├── main.py
└── yolo26n.pt
```

**Important distinctions:**

- `data.yaml` describes the Aquarium dataset's seven object classes.
- `train/`, `valid/`, and `test/` are the Aquarium split folders in the repository.
- `datasets/coco128/` and `coco128.yaml` relate to the general COCO128 dataset, not the seven-class Aquarium label scheme.
- `training_runs/` stores generated training artifacts; a specific `best.pt` should be selected from the relevant experiment's `weights/` directory.
- `assets/`, `profile.json`, `requirements.txt`, and the Windows `.bat` helpers are **optional local additions described below**; they are **not currently in the public root listing**.

## Getting started

### Prerequisites

- A computer with Python installed. **Python 3.11 or 3.12** is a practical starting point.
- Internet access for first-time package installation and any checkpoint or dataset downloads.
- Sufficient storage for PyTorch, datasets, and model runs.
- A working display environment for the Qt desktop interface.

The main application is cross-platform Python/Qt code, while the provided instructions prioritize Windows. The PowerShell live-log window is Windows-specific; the code includes an in-app fallback when PowerShell is unavailable.

### 1. Clone the repository

```bash
git clone https://github.com/jrbMods/underwater-vision.git
cd underwater-vision
```

Alternatively, use **Code → Download ZIP** on the GitHub repository page and extract it.

### 2. Create a virtual environment

**Windows — PowerShell:**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If PowerShell restricts activating scripts, you can call `.venv\Scripts\python.exe` directly as shown; no activation is necessary.

### 3. Install dependencies

The repository does not currently contain a published `requirements.txt`, so install the direct dependencies manually:

**Windows:**

```powershell
.\.venv\Scripts\python.exe -m pip install PySide6 opencv-python numpy ultralytics
```

**macOS / Linux:**

```bash
python -m pip install PySide6 opencv-python numpy ultralytics
```

`ultralytics` installs its supported dependency stack, including PyTorch. For GPU-specific installations, follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

**For reproducible experiments:** record the versions actually used, for example:

```bash
python -m pip freeze > environment-lock.txt
```

Do not assume that unpinned package versions will reproduce identical results on another machine.

### 4. Start Ocean Studio

**Windows:**

```powershell
.\.venv\Scripts\python.exe main.py
```

**macOS / Linux:**

```bash
python main.py
```

You should see the Ocean Studio workspace. The application can run without optional logo and profile files, although their visual placeholders or default values may be displayed.

### Optional: Windows one-click `.bat` files

The project has been used locally with a launcher and a requirements installer. **These files are not currently present in the public repository root.** If you add them, place them beside `main.py` and document their actual names and behavior here. Typical names are:

```text
Start_Ocean_Studio.bat
Install_Ocean_Studio_Requirements.bat
requirements.txt
```

The installer can create a virtual environment and optionally launch the application after successfully installing dependencies. Until these scripts are committed, the manual installation instructions above are the supported path for a fresh clone.

## Using the application

### Open and enhance an image

1. Launch Ocean Studio.
2. Use **Open Image** to select a JPG or PNG photograph.
3. Select an enhancement method and adjust its slider.
4. Apply the enhancement.
5. Compare the original and processed previews.
6. Optionally open the four-method comparison view and export a montage.

### Detect underwater objects

1. Open an image.
2. Use the pretrained checkpoint or choose a compatible fine-tuned `best.pt` model.
3. Adjust the confidence threshold as needed.
4. Click **Detect Objects**.
5. Inspect boxes and labels, then save the annotated result.

If no detections appear, this does not necessarily mean the app failed. Check the selected model's classes, the confidence threshold, and the domain of the input image.

### Batch image analysis

1. Prepare a folder containing supported images.
2. Open the batch YOLO analysis tool.
3. Select that folder and run the analysis.
4. Export the resulting CSV.
5. Open the batch report/chart view for summary visualizations.

Keep a record of the checkpoint, confidence threshold, enhancement parameters, dataset, and machine used for each run.

## Included dataset

### Aquarium Combined (Roboflow)

The repository's [`data.yaml`](data.yaml) identifies the **Aquarium Combined, version 2** dataset and records the following object classes:

| Class ID | Class name |
| ---: | --- |
| 0 | fish |
| 1 | jellyfish |
| 2 | penguin |
| 3 | puffin |
| 4 | shark |
| 5 | starfish |
| 6 | stingray |

Dataset metadata in the configuration:

- **Workspace:** `brad-dwyer`
- **Project:** `aquarium-combined`
- **Version:** `2`
- **Recorded dataset license:** **CC BY 4.0**
- **Original dataset:** [Roboflow Universe — Aquarium Combined v2](https://universe.roboflow.com/brad-dwyer/aquarium-combined/dataset/2)

**Attribution:** Dataset creators / maintainers should receive proper attribution when redistributing the images or using them in research. Consult the source page and [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/) for the required notice and credit.

> [!CAUTION]
> The Aquarium dataset is useful for prototyping, but performance on aquarium photographs does **not** establish performance in murky open-water, deep-sea, or other real-world environments.

### Dataset folders and paths

The repository includes `train/`, `valid/`, and `test/` at the root. The current `data.yaml` contains the relative entries:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
```

**Check these paths before training.** Ultralytics resolves them relative to its dataset root configuration, which can differ between working directories and extracted dataset layouts. If you receive a missing-image-folder error, make a **local** copy of `data.yaml` configured for your clone's actual absolute paths, e.g.:

```yaml
path: C:/path/to/underwater-vision
train: train/images
val: valid/images
test: test/images
names:
  0: fish
  1: jellyfish
  2: penguin
  3: puffin
  4: shark
  5: starfish
  6: stingray
```

Replace the example `C:/path/to/underwater-vision` with the actual directory on your machine. Check that the folder names and image/label layout match before using this example. You may also use the in-app dataset audit to catch missing paths and annotation problems.

### COCO128 is separate

[`coco128.yaml`](coco128.yaml) defines COCO's standard classes for a small general-purpose detection example. It is **not interchangeable** with the Aquarium configuration. Class IDs must correspond to the classes that the chosen model was trained to recognize.

## Fine-tuning YOLO

### From the desktop UI

The application contains a training dialog and a research training view. In general:

1. Choose the Aquarium dataset's `data.yaml`.
2. Begin from `yolo26n.pt` (or choose an appropriate pretrained starting model if your code supports it).
3. Set the epoch count, image resolution, and batch size.
4. For a research run, choose a random seed and early-stopping patience.
5. Start training and monitor the application/terminal output.
6. Locate the saved `weights/best.pt` checkpoint.
7. Evaluate the checkpoint on labeled data with matching class definitions.

The research training interface contains a default example of up to **50 epochs**, **12** epochs patience, **4** images per batch, **640** image resolution, and random seed **42**. These are workflow defaults, not guaranteed optimal hyperparameters.

### Optional equivalent CLI example

If you prefer a terminal, Ultralytics supports training calls such as:

```bash
yolo detect train model=yolo26n.pt data=local_aquarium.yaml epochs=50 imgsz=640 batch=4 patience=12 seed=42
```

Here, `local_aquarium.yaml` is a dataset configuration whose paths you have checked for your machine. The output directory is determined by your Ultralytics settings and run arguments; inspect the training output for the actual location.

### Selecting checkpoints

- **`best.pt`**: checkpoint selected during training based on validation criteria.
- **`last.pt`**: most recent saved training checkpoint, useful for resuming some runs.
- **`yolo26n.pt`**: starting general-purpose pretrained checkpoint in the repo.

Record which checkpoint you used in each experiment. Models trained for different class schemes cannot be compared simply by counting detections.

## Evaluation and research workflows

### Metrics

The application supports YOLO's standard bounding-box validation metrics:

| Metric | What it describes |
| --- | --- |
| **Precision** | Correct detections as a proportion of reported detections under the evaluation procedure. |
| **Recall** | Labeled objects recovered by the detector under the evaluation procedure. |
| **mAP@50** | Mean average precision with an intersection-over-union threshold of 0.50. |
| **mAP@50–95** | Mean average precision averaged across IoU thresholds from 0.50 to 0.95. |

Model confidence is not the same thing as accuracy, and more reported bounding boxes do not necessarily mean improved performance.

### Four-condition enhancement study

Run **one fixed trained model** on the same labeled images under:

1. Original (no enhancement).
2. CLAHE.
3. White Balance.
4. Gamma Correction.

Use identical data splits, model weights, image sizes, and validation settings for fair comparisons. The source images and annotation labels should remain unchanged; enhanced copies can be generated for evaluation.

### Eight-evaluation model comparison

The controlled-experiment view compares two trained YOLO checkpoints across the four conditions above. It exports per-condition values for precision, recall, mAP@50, and mAP@50–95, plus experiment parameters and model paths.

For example, you might compare a short practice-trained model and a more carefully trained checkpoint. **Do not assume the longer-trained model must win**: use the observed results.

### Results and statistical interpretation

The reporting tools can produce CSV summaries, charts, and Markdown analysis. Differences should be described as **observed differences**, not necessarily statistically significant effects. Aggregate accuracy values from a few conditions are not sufficient to calculate trustworthy p-values or to establish generalization beyond the selected images.

For stronger thesis evidence:

- Maintain independent train, validation, and test partitions.
- Avoid repeatedly tuning on the final test set.
- Repeat training with multiple seeds where practical.
- Analyze performance by class and by visibility conditions.
- Evaluate on a separate unseen underwater dataset when available.
- Archive environment versions, dataset identifiers, checkpoint hashes, and experiment parameters.

### Results table to populate with real values

| Model | Enhancement | Precision | Recall | mAP@50 | mAP@50–95 |
| --- | --- | ---: | ---: | ---: | ---: |
| Your checkpoint | Original | — | — | — | — |
| Your checkpoint | CLAHE | — | — | — | — |
| Your checkpoint | White Balance | — | — | — | — |
| Your checkpoint | Gamma Correction | — | — | — | — |

**No accuracy numbers are included here because this README does not have verified exported experimental results.** Fill in measured values and identify the dataset, split, model, and run settings in the accompanying text.

## Branding, background, and application icon (optional)

The application supports local branding assets. These files are **not currently part of the public repository tree**, so create an `assets/` folder beside `main.py` if you want to use them.

```text
underwater-vision/
├── assets/
│   ├── workspace_logo.png
│   ├── logo.png
│   ├── university_logo.png
│   ├── background.png
│   └── app_icon.ico
└── profile.json
```

| File | Purpose |
| --- | --- |
| `workspace_logo.png` | Main workspace logo |
| `logo.png` | Personal/project logo in About |
| `university_logo.png` | University logo in About |
| `background.png` (or `.jpg`, `.jpeg`, `.webp`) | Fixed, softly blurred/darkened background |
| `app_icon.ico` or `app_icon.png` | Application/window icon |

Legacy `background_custom.*` filenames are also recognized by the current code. The fixed background is changed by **replacing the image file**, not from an in-app image picker.

### Optional `profile.json`

The source code currently reads `first_name`, `last_name`, `email`, `fnum`, `university`, `program`, and `project_title`:

```json
{
  "first_name": "Your first name",
  "last_name": "Your surname",
  "email": "",
  "fnum": "",
  "university": "Your university",
  "program": "Master's thesis",
  "project_title": "Underwater Vision AI"
}
```

The field `fnum` is the key expected by this version of the code. If you do not want to make your contact details public, keep `profile.json` local and publish a `profile.example.json` with placeholders instead.

## Logging and diagnostics

- The app records available Python output, errors, and Qt messages to `logs/ocean_studio.log`.
- On Windows, **Open terminal / logs** opens a PowerShell window that follows new log lines.
- When the PowerShell viewer is unavailable, the application has an in-app log viewer fallback.
- Inspect log messages for failed model loads, missing datasets, or processing errors.

The log file may contain local paths and user-specific information. Review it before sharing publicly.

## Troubleshooting

<details>
<summary><strong><code>ModuleNotFoundError</code> or missing Python package</strong></summary>

Check that you installed the libraries in the same Python environment you use to run `main.py`.

```powershell
.\.venv\Scripts\python.exe -m pip install PySide6 opencv-python numpy ultralytics
.\.venv\Scripts\python.exe main.py
```

</details>

<details>
<summary><strong>The application opens but detects no objects</strong></summary>

Verify the active model, its class labels, and the confidence slider. Try an image whose objects belong to the selected model's classes. Use the low-confidence diagnostic feature to determine whether the model produces boxes at a lower threshold. Zero detections can be a genuine model result.

</details>

<details>
<summary><strong>Dataset path not found / missing validation images</strong></summary>

Inspect the paths in `data.yaml` and compare them to your actual clone. The repository's YAML uses `../train/images`, `../valid/images`, and `../test/images`; relative paths may require adjustment. Use a local YAML with an explicit dataset root and check the dataset audit before training.

</details>

<details>
<summary><strong>Training is very slow</strong></summary>

CPU training may be slow. Use a smaller batch/image size for testing, or configure a compatible GPU environment. Check the current workload and available memory before increasing batch size.

</details>

<details>
<summary><strong>Qt warning: <code>QPaintDevice: Cannot destroy paint device that is being painted</code></strong></summary>

This warning indicates an unfinished painting operation, often during chart rendering/export. Ensure you are using the latest source in your project folder, keep the application and Qt dependencies consistent, and capture the traceback/log output if it persists.

</details>

<details>
<summary><strong>My logo, background, or app icon doesn't appear</strong></summary>

Make sure you created `assets/` beside `main.py` and used the supported filenames. Restart the app after changing image files. A transparent PNG often works best for logos; `.ico` is preferable for the Windows application icon.

</details>

<details>
<summary><strong>Windows PowerShell does not show live logs</strong></summary>

Check that `logs/ocean_studio.log` exists or whether the app used its home-directory fallback. The PowerShell button is Windows-specific. Use the internal log viewer or read the file directly if the external terminal cannot open.

</details>

## Development and reproducibility recommendations

To make this repository easier for other researchers to run and review:

1. Commit a **pinned `requirements.txt`** or a verified dependency lock file.
2. Add real screenshots under `docs/images/`.
3. Add an example profile rather than personal contact details.
4. Use `.gitignore` for `.venv/`, logs, caches, temporary comparison files, and local private data.
5. Keep model checkpoints in labeled release artifacts or document their size and provenance.
6. Add automated tests for image enhancement, dataset parsing, and CSV analysis.
7. Record dataset versions/licenses and give required attribution.
8. Add a software `LICENSE` file after deciding your desired terms.

### Suggested `.gitignore` additions

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
logs/
profile.json
.env
*.log
.DS_Store
```

**Do not blindly ignore `train/`, `valid/`, `test/`, `training_runs/`, or `*.pt` if you intentionally distribute these files in this repository.** Decide which files belong in the public release, verify dataset and model redistribution rights, and document the decision explicitly.

## Privacy, ethics, and limitations

- Underwater visibility, color, turbidity, species, habitats, and camera conditions can differ greatly from the Aquarium training data.
- Classification errors and missed detections are possible; detections must not be treated as guaranteed biological observations.
- Training and evaluating on overlapping images can inflate reported performance.
- Dataset class mappings must match the checkpoint used for evaluation.
- Personal contact information, filesystem paths, and training logs should be reviewed before publication.
- Users remain responsible for respecting dataset licenses and relevant privacy/field-research requirements.

## Citation and research use

If you use or extend Ocean Studio in academic work, cite this repository and include the specific version or commit hash:

```bibtex
@misc{jrbmods_underwater_vision,
  author       = {{jrbMods}},
  title        = {Ocean Studio: Underwater Vision AI},
  howpublished = {GitHub repository},
  url          = {https://github.com/jrbMods/underwater-vision},
  note         = {Accessed 2026-11-12; cite the specific commit used}
}
```

This is a repository citation example, **not** a claim about an officially published paper or DOI. Also cite the original datasets and underlying methods/packages separately as appropriate.

### Useful upstream documentation

- [Ultralytics YOLO documentation](https://docs.ultralytics.com/)
- [PySide6 / Qt for Python](https://doc.qt.io/qtforpython-6/)
- [OpenCV](https://docs.opencv.org/)
- [PyTorch](https://pytorch.org/)
- [Roboflow Aquarium Combined v2](https://universe.roboflow.com/brad-dwyer/aquarium-combined/dataset/2)

## License

**Software license not specified in the current repository.** The presence of source code on GitHub does not by itself grant permission to reuse or redistribute it under an open-source license. Add a `LICENSE` file when you've chosen the terms you want for your code.

The **Aquarium Combined dataset** declares **CC BY 4.0** in `data.yaml`. That dataset license is separate from the application source license, model-weights terms, and licenses of third-party libraries.

## Author and repository

**Repository:** [github.com/jrbMods/underwater-vision](https://github.com/jrbMods/underwater-vision)  
**GitHub account:** [@jrbMods](https://github.com/jrbMods)

Personal author, university, and thesis-submission details can be added once you decide what information you want to publish publicly. They are not inferred here.

---

<div align="center">

**Ocean Studio — Underwater Vision AI**  
*Explore underwater images. Train detectors. Measure results.*

[↑ Back to top](#-ocean-studio)

</div>
