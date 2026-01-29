import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2

# -----------------------------
# Définition du modèle (même que lors de l'entraînement)
# -----------------------------

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))  # Tu peux laisser ce pool ici, c’est les couches de convolutions
    return nn.Sequential(*layers)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = (out.argmax(dim=1) == labels).float().mean()
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")

class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # <-- changement ici
            nn.Flatten(),
            nn.Linear(512, num_diseases)  # <-- 512 au lieu de 512*X*X, car sortie est 512x1x1
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# -----------------------------
# Chargement du modèle et prédiction
# -----------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 38  # Ajuste si tu as un autre nombre de classes
    model = CNN_NeuralNet(3, num_classes)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()

    classes = {
        0: 'Apple___Apple_scab',
        1: 'Apple___Black_rot',
        2: 'Apple___Cedar_apple_rust',
        3: 'Apple___healthy',
        4: 'Blueberry___healthy',
        5: 'Cherry_(including_sour)___Powdery_mildew',
        6: 'Cherry_(including_sour)___healthy',
        7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        8: 'Corn_(maize)___Common_rust_',
        9: 'Corn_(maize)___Northern_Leaf_Blight',
        10: 'Corn_(maize)___healthy',
        11: 'Grape___Black_rot',
        12: 'Grape___Esca_(Black_Measles)',
        13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        14: 'Grape___healthy',
        15: 'Orange___Haunglongbing_(Citrus_greening)',
        16: 'Peach___Bacterial_spot',
        17: 'Peach___healthy',
        18: 'Pepper,_bell___Bacterial_spot',
        19: 'Pepper,_bell___healthy',
        20: 'Potato___Early_blight',
        21: 'Potato___Late_blight',
        22: 'Potato___healthy',
        23: 'Raspberry___healthy',
        24: 'Soybean___healthy',
        25: 'Squash___Powdery_mildew',
        26: 'Strawberry___Leaf_scorch',
        27: 'Strawberry___healthy',
        28: 'Tomato___Bacterial_spot',
        29: 'Tomato___Early_blight',
        30: 'Tomato___Late_blight',
        31: 'Tomato___Leaf_Mold',
        32: 'Tomato___Septoria_leaf_spot',
        33: 'Tomato___Spider_mites Two-spotted_spider_mite',
        34: 'Tomato___Target_Spot',
        35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        36: 'Tomato___Tomato_mosaic_virus',
        37: 'Tomato___healthy'
    }

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    img_path = "feuille_test.JPG"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image non trouvée : {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    print(f"Classe prédite : {predicted_class}")
    print(f"Nom de la classe : {classes.get(predicted_class, 'Classe inconnue')}")

if __name__ == "__main__":
    main()
