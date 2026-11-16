
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Recreate the model architecture
class PneumoniaNet(nn.Module):
    def __init__(self):
        super(PneumoniaNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Load the saved weights
device = torch.device('cpu')
model = PneumoniaNet()
model.load_state_dict(torch.load('pneumonia_model.pth', map_location=device))
model.eval()

def predict_xray(input_image):
    img = Image.fromarray(input_image).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]

    return {
        "Normal": float(probabilities[0]),
        "Pneumonia": float(probabilities[1])
    }

interface = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(label="Upload Chest X-Ray"),
    outputs=gr.Label(num_top_classes=2, label="AI Diagnosis Confidence"),
    title="🩺 Pediatric Chest X-Ray Analyzer",
    description="Upload a chest X-Ray image to test the model dynamically.",
)

interface.launch()
