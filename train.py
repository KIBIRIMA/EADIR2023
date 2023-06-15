import torch
import torch.nn as nn
import torch.optim as optim
from custom_dataset import CustomDataset
from model import Model

# Créer une instance de CustomDataset avec les chemins d'accès aux données appropriés
dataset = CustomDataset(image_paths, mask_paths, target_size)

# Créer un DataLoader pour itérer sur les données en mini-lots pendant l'entraînement
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Créer une instance du modèle
model = Model()

# Définir la fonction de perte
criterion = nn.MSELoss()

# Définir l'optimiseur avec le taux d'apprentissage souhaité
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Mettre en place la boucle d'entraînement
num_epochs = 10

for epoch in range(num_epochs):
    # Entraînement du modèle
    model.train()

    for image, target in dataloader:
        # Remettre les gradients à zéro
        optimizer.zero_grad()

        # Passage avant à travers le modèle
        output = model(image)

        # Calcul de la perte
        loss = criterion(output, target)

        # Rétropropagation
        loss.backward()

        # Mise à jour des poids
        optimizer.step()

    # Affichage de la perte de l'époque actuelle
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # Éventuellement, effectuer une évaluation/validation du modèle après chaque époque
