import torch
from torchvision import models, transforms
from torchvision.ops.boxes import nms as nms_torch

from PIL import Image
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('yaep/')
from yaep.efficientdet.model import BiFPN

def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transform to preprocess the image
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to be called at each layer's forward pass
features = {}
def hook_function(level):
    def hook(model, input, output):
        features[level] = output
    return hook

# For efficientnet_b0
efficientnet_feature_levels = {"features.3" : "p3", # p3
                                       "features.5" : "p4", # p4
                                       "features.7" : "p5"} # p5

def main():

    backbone = models.efficientnet_b0(weights='DEFAULT')
    backbone.eval()
    backbone.to(device)

    # Register hooks for each layer
    for level, module in backbone.named_modules():
        if level in efficientnet_feature_levels.keys():
            module.register_forward_hook(hook_function(efficientnet_feature_levels[level]))

    image_path = "images/tabby.jpg"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        image = image.resize((512, 512), Image.LANCZOS)
        input_tensor = transform(image).unsqueeze(0).to(device)
        backbone(input_tensor)

        for key, value in features.items():
            print(f"Level: {key} | Feature Shape: {value.shape}")

        # debug
        bifpn = BiFPN(num_channels=64, conv_channels=[40, 112, 320], first_time=True)
        bifpn.to(device)
        outputs = bifpn(list(features.values()))

        keys = ["p3_out", "p4_out", "p5_out", "p6_out", "p7_out"]
        for key, value in zip(keys, outputs):
            print(f"Level: {key} | Feature Shape: {value.shape}")

        # Assuming feature_maps is a list or dict where feature_maps[i] corresponds to Pi where i ranges from 3 to 7
        for i, feat_map in enumerate(outputs):
            plt.figure(figsize=(10, 10))
            plt.imshow(feat_map.cpu()[0, 0, :, :].detach().numpy(), cmap='viridis')  # Adjust the channel index as needed
            plt.title(f'Feature Map P{i+3}')
            plt.colorbar()
            plt.savefig("images/" + "p" + str(3+i) + ".png")
            
    else:
        print("Image file does not exist.")

if __name__ == '__main__':
    main()