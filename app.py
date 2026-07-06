import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image

# 1. Define the CNN Architecture (must match your trained model)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(8 * 8 * 64, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# 2. Load the trained model weights
device = torch.device("cpu")  # Deploy on CPU by default
model = CNN()
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

# 3. Define the prediction function
def predict_image(img):
    if img is None:
        return {}
    
    # Convert image to RGB format to support PNGs and grayscale
    img = img.convert("RGB")
    
    # Preprocess image to match training parameters
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    
    ])
    
    img_tensor = eval_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs[0], dim=0)
        
    classes = ['Cat', 'Dog']
    return {classes[i]: float(probabilities[i]) for i in range(len(classes))}

# 4. Set up and launch the Gradio Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Cat/Dog Image"),
    outputs=gr.Label(num_top_classes=2, label="Predictions"),
    title="🐱 Cat vs Dog Classifier 🐶",
    description="Upload an image to see if the model predicts a Cat or a Dog with confidence levels."
)

if __name__ == "__main__":
    interface.launch()
