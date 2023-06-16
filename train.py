import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from customdataset import CustomDataset
from model import EfficientNetB0_BiFPN

# Chemins d'accès aux images et aux masques
image_dir = "C:\Users\Admin\Downloads\Project Dataset\train"
mask_dir = "C:\Users\Admin\Downloads\Project Dataset\masks"

# Hyperparamètres
batch_size = 16
lr = 0.001
num_epochs = 10

# Instanciation du Dataset
dataset = CustomDataset(image_dir, mask_dir)

# Création du DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instanciation du modèle
model = EfficientNetB0_BiFPN()

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Boucle d'entraînement
for epoch in range(num_epochs):
    for images, masks in dataloader:
        # Remise à zéro des gradients
        optimizer.zero_grad()
        
        # Passage avant du modèle
        outputs = model(images)
        
        # Calcul de la perte
        loss = criterion(outputs, masks)
        
        # Rétropropagation et mise à jour des poids
        loss.backward()
        optimizer.step()
        
        # Affichage des statistiques d'entraînement
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
