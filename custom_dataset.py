import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Charger l'image à partir du chemin d'accès et appliquer les prétraitements
        image = self.load_image(image_path)
        mask = self.load_image(mask_path)

        # Appliquer des transformations si nécessaire
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

    def load_image(self, image_path):
        # Charger l'image à partir du chemin d'accès
        image = Image.open(image_path)

        # Appliquer les prétraitements nécessaires
        # Par exemple, redimensionner l'image, convertir en tenseur PyTorch, normaliser, etc.

        return image

    def load_mask(self, mask_path):
        # Charger le masque à partir du chemin d'accès
        mask = Image.open(mask_path)

        # Appliquer les prétraitements nécessaires
        # Par exemple, redimensionner le masque, convertir en tenseur PyTorch, etc.

        return mask
     def display_images(self, num_samples=5):
        for i in range(num_samples):
            image_path = self.image_paths[i]
            mask_path = self.mask_paths[i]

            image = self.load_image(image_path)
            mask = self.load_mask(mask_path)

            plt.subplot(num_samples, 2, 2*i+1)
            plt.imshow(image)
            plt.title("Input Image")
            plt.axis('off')

            plt.subplot(num_samples, 2, 2*i+2)
            plt.imshow(mask, cmap='gray')
            plt.title("Target Mask")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

