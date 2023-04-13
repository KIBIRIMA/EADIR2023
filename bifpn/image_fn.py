import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def resize_image(image, size):
    """Resize an image to the specified size."""
    return image.resize(size)

def convert_to_grayscale(image):
    """Convert an RGB image to grayscale."""
    return image.convert('L')

def normalize_image(image):
    """Normalize an image by dividing all pixel values by 255."""
    return image / 255.0

def visualize_image(image):
    """Display an image using matplotlib."""
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def visualize_depth_map(sample):
    """Display a depth map using matplotlib."""
    image, mask = sample
    mask = np.array(mask)
    mask = mask / np.max(mask)  # Normalize the depth map values to [0, 1]
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5, cmap='jet', vmin=0, vmax=1)
    plt.axis('off')
    plt.show()
