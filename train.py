from PIL import Image
import glob
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from occupancynet import *

def dice_loss(pred, target):
    smooth = 1.
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

def loss_function(pred, target):
    # return F.binary_cross_entropy(pred, target)
    return F.binary_cross_entropy(pred, target) + dice_loss(pred, target)  # Combining BCE with Dice for better segmentation

# Define a custom transform to handle alpha channel
def remove_alpha_channel(img):
    return img.convert('RGB') if img.mode == 'RGBA' else img

class StereoDataset(Dataset):
    def __init__(self, image_pairs, output_tensors):
        self.image_pairs = image_pairs
        self.output_tensors = output_tensors

    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        left_image, right_image = self.image_pairs[idx]
        output_tensor = self.output_tensors[idx]
        return (left_image, right_image), output_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Left and right images of stereo cameras, and 3D occupancy maps
left_image_root = "blender_stereo_images/left_images/"
right_image_root = "blender_stereo_images/right_images/"
paths = glob.glob("blender_stereo_images/occupancy_grids/*.npy")

# Image transformation to tensor, resize to match your network's expected input size
transform = transforms.Compose([
    transforms.Lambda(remove_alpha_channel),
    transforms.Resize((256, 256)),  # Resize to match your network's expected input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

image_pairs = []
output_tensors = []

for i, path in enumerate(paths):

    filename = "image_" + str(i)

    # Load left image
    left_image_path = f"{left_image_root}{filename}.png"  # Assuming images are PNG
    with Image.open(left_image_path) as img:
        left_image = transform(img)
    
    # Load right image
    right_image_path = f"{right_image_root}{filename}.png"
    with Image.open(right_image_path) as img:
        right_image = transform(img)
    
    # Load occupancy grid (output tensor)
    output_tensor = torch.from_numpy(np.load(path)).unsqueeze(0)  # Add channel dimension
    output_tensor = output_tensor.float()

    # Ensure output_tensor has the correct shape if needed
    if output_tensor.shape != (1, 64, 64, 64):

        raise ValueError(f"Expected output tensor shape (1, 64, 64, 64), got {output_tensor.shape}")

    image_pairs.append((left_image, right_image))
    output_tensors.append(output_tensor)
    
train_dataset = StereoDataset(image_pairs, output_tensors)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

model = OccupancyNet('regnet', embed_dim=64, grid_size=16)
model.to(device)

optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 100

# Initialize best loss as infinity
best_loss = float('inf')
best_model = None

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for (left_images, right_images), targets in train_loader:
        left_images, right_images, targets = left_images.to(device), right_images.to(device), targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(left_images, right_images)
        
        # Compute loss
        loss = loss_function(predictions, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Calculate average loss for the epoch
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Check if this model is better than the best one seen so far
    if avg_loss < best_loss:
        best_loss = avg_loss
        # Save the model's state dict
        best_model = model.state_dict().copy()  # Copy to avoid reference issues

# After the training loop, save the best model weights
if best_model is not None:
    torch.save(best_model, 'best.pt')
    print(f"Best model saved with loss: {best_loss:.4f}")
else:
    print("No model was saved as no improvement was observed.")