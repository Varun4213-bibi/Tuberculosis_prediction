
# Tuberculosis Prediction

This project is a tuberculosis prediction system using chest X-ray images. It includes code for training a tuberculosis detector model and a web app to make predictions.

## Dataset

The project uses chest X-ray images for training and testing. You can download a commonly used tuberculosis chest X-ray dataset here:

[Tuberculosis (TB) Chest X-ray Dataset on Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

Please download and extract the dataset. Then, run the Python script `TB_Chest_Radiography_Database/file.py` to split the dataset into training, validation, and test sets as required by the training script.

## Model Checkpoints and Performance

Model checkpoint files (`.pth`) are **not included** in this repository due to file size limitations. You will need to train the model yourself using the provided training script or obtain the model checkpoint from an external source.

The model used is a ResNet-50 architecture pretrained on ImageNet and fine-tuned for tuberculosis detection on chest X-ray images.

After training for 10 epochs, the model achieved a best validation accuracy of approximately 99% 

The final trained model is saved as `tb_detector_resnet50.pth` during training.

You can use the training script `train_tb_detector.py` to train the model and evaluate its performance on your dataset.

## Setup and Usage

1. Create a Python virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. To train the model, run:

```bash
python train_tb_detector.py
```

4. To run the web app for prediction:

```bash
python app.py
```

Then open your browser and go to `http://localhost:5000`.


also uploded screen shots of my apps interface and a sample prediction for one of the tuberculosis chest xray



