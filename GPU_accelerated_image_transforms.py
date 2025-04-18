import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Ensure the device is CUDA (GPU) enabled
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a set of GPU-accelerated augmentations using TorchVision transforms
augmentation_pipeline = transforms.Compose([
    transforms.RandomResizedCrop(224),    # Randomly crop and resize images
    transforms.RandomHorizontalFlip(),    # Randomly flip the image horizontally
    transforms.RandomRotation(30),        # Randomly rotate the image by up to 30 degrees
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # Color jitter
    transforms.ToTensor(),                # Convert the image to a tensor
])

# Function to load and apply augmentation to an image
def augment_image(image_path):
    image = Image.open(image_path)
    
    # Apply augmentation pipeline
    augmented_image = augmentation_pipeline(image)
    
    return augmented_image

# Example image path (replace with your own image)
image_path = 'path_to_image.jpg'

# Augment the image and move it to GPU
augmented_image = augment_image(image_path).to(device)

# Visualize the augmented image
augmented_image = augmented_image.cpu().permute(1, 2, 0).numpy()  # Move back to CPU and change format for matplotlib
plt.imshow(augmented_image)
plt.show()
