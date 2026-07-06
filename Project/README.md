# Captcha Digit Classifier

Small Python project for training and running a CNN that recognizes single digits in captcha images.

## Files

- `captcha_predict.py` predicts digits from a captcha image using `FinalModel_1.keras`.
- `captcha_train.py` trains a new model from the `kaptcha/` dataset.
- `captcha_augment.py` optionally creates a larger augmented dataset from `kaptcha/`.
- `kaptcha/` contains the source digit images organized by class (`1` to `9`).
- `testCaptcha*.png` are sample images for testing prediction.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Prediction

```bash
python captcha_predict.py testCaptcha_2.png --output prediction_result.png
```

## Train Model

```bash
python captcha_train.py --data kaptcha --model FinalModel_1.keras
```

Optional augmentation:

```bash
python captcha_augment.py --input kaptcha --output augmented_captcha --target-per-class 1000
python captcha_train.py --data augmented_captcha --model FinalModel_1.keras
```
