# AI Pharma QA — sample-image build

This folder contains a runnable **software-pipeline demo** for four visual labels:
Broken, Cracked, Discolored, and Good.

## Included sample data
- `dataset/train/`: 24 generated sample images per class (96 total)
- `dataset/validation/`: 6 generated sample images per class (24 total)

**Important:** These are computer-generated illustrative tablet-like images, not photographs of real medicines and not a scientifically representative dataset. They are included only to verify that folder loading, training, saving, and Streamlit inference run end-to-end. Do not use the resulting model for real pharmaceutical quality decisions.

## Setup (Windows CMD)
```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train
```cmd
python train_model.py
```
This creates `tablet_model.keras` and updates `class_names.json`.

## Run the demo
```cmd
streamlit run app.py
```

## Replace the sample data before meaningful experiments
Replace the generated images with properly labeled, real tablet images. Keep the same folder names and ensure that images from the same physical tablet do not appear in both train and validation sets. Use a separate held-out test set for evaluation.
