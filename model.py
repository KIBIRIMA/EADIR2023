import torch
import torch.nn as nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()

        # Charger le modèle pré-entraîné d'EfficientNet-B0
        backbone = models.efficientnet_b0(pretrained=True)

        # Supprimer la dernière couche (classification) du modèle EfficientNet-B0
        backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Définir les blocs de décodeur BiFPN
        decoder_blocks = nn.ModuleList([
            BiFPNBlock(112, 56),
            BiFPNBlock(56, 28),
            BiFPNBlock(28, 14)
        ])

        # Créer le modèle complet en combinant le backbone (EfficientNet-B0) et le neck (BiFPN)
        self.backbone = backbone
        self.decoder_blocks = decoder_blocks
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(14, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Passer l'image à travers le backbone (EfficientNet-B0)
        features = self.backbone(x)

        # Passer les caractéristiques à travers les blocs de décodeur BiFPN
        for block in self.decoder_blocks:
            features = block(features)

        # Upsample des caractéristiques de résolution 14x14 à 56x56
        upsampled_features = self.upsample(features)

        # Passer les caractéristiques upsamplées à travers une couche de convolution pour obtenir une carte de profondeur 1
        output = self.conv(upsampled_features)

        # Appliquer une fonction d'activation (par exemple, sigmoid) à la sortie pour obtenir une probabilité
        output = self.sigmoid(output)

        return output


class BiFPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BiFPNBlock, self).__init__()

        self.up_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.down_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pooled_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Fusionner les caractéristiques de résolution inférieure avec les caractéristiques de résolution supérieure
        upsampled = self.up_conv(x[0]) + self.lateral_conv(x[1])
        downsampled = self.down_conv(upsampled) + self.pooled_conv(x[0])

        # Appliquer des activations non linéaires (par exemple, ReLU et Sigmoid)
        upsampled = self.relu(upsampled)
        downsampled = self.sigmoid(downsampled)

        # Retourner les caractéristiques fusionnées
        return [upsampled, downsampled]

# Instanciation du modèle
num_classes = 1  # Nombre de classes de sortie (dans cet exemple, une seule classe pour la segmentation binaire)
model = Model(num_classes)

# Exemple d'utilisation du modèle
input_tensor = torch.randn(1, 3, 224, 224)  # Tensor d'exemple en entrée (batch_size=1, 3 canaux RGB, résolution 224x224)
output = model(input_tensor)
print("Output shape:", output.shape)
