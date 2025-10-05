import os
from flask import Flask, request, render_template, redirect, url_for
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# Load model and set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['Normal', 'Tuberculosis']

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('tb_detector_resnet50.pth', map_location=device))
model = model.to(device)
model.eval()

# Image preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def transform_image(image_bytes):
    image = Image.open(image_bytes).convert('RGB')
    return data_transforms(image).unsqueeze(0)

def get_prediction(image_tensor):
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    confidence, preds = torch.max(probs, 1)
    print(f"Prediction: {class_names[preds.item()]} with confidence {confidence.item():.4f}")
    return class_names[preds.item()], confidence.item()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        img_tensor = transform_image(file)
        prediction = get_prediction(img_tensor)
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
