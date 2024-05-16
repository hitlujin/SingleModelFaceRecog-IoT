import torch
import torchvision.transforms as transforms
from PIL import Image
from celebAload import CelebALoader
from torchvision import models
# Load the pre-trained model (ResNet in this example)
model = models.resnet50(pretrained=True)
# Modify the classifier to fit the number of attributes in CelebA
model.fc = torch.nn.Linear(model.fc.in_features, 40) # Assuming 40 attributes
# Load the dataset
data_loader = CelebALoader('/path/to/celebA/dataset', download=True)
# Define transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Prediction function
def predict_attributes(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        predicted_attributes = torch.sigmoid(outputs).gt(0.5).type(torch.uint8)
    
    return predicted_attributes
# Example usage is omitted as per client's request