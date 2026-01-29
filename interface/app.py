import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db
from streamlit_autorefresh import st_autorefresh

# ---------- Initialisation Firebase ----------
if not firebase_admin._apps:
    cred = credentials.Certificate("../simulation_capteurs/service_account.json")  # adapte le chemin
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://smart-agriculture-1-default-rtdb.firebaseio.com/'
    })

# ---------- Définition du modèle ----------
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch): pass
    def validation_step(self, batch): pass
    def validation_epoch_end(self, outputs): pass
    def epoch_end(self, epoch, result): pass

class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
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

# ---------- Chargement modèle ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_NeuralNet(3, 38)
model.load_state_dict(torch.load("../prediction_local/model.pth", map_location=device))
model.to(device)
model.eval()

# ---------- Classes ----------
classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ---------- Transformations image ----------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ---------- Prédiction ----------
def predict_pytorch(image_pil):
    image = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        return classes[predicted_class], probs[0][predicted_class].item()



import matplotlib.pyplot as plt
import datetime
import random

# ----------------------- Configuration UI -----------------------
st.set_page_config(page_title="🌿 Détection de Maladies - SmartAgri", layout="wide", initial_sidebar_state="expanded")

# 🌙 Mode sombre Streamlit (à activer depuis les paramètres de ton profil Streamlit)

# ----------------------- Sidebar Navigation -----------------------
st.sidebar.title("🌱 Menu")
page = st.sidebar.radio("Naviguer vers", ["🏠 Accueil", "📸 Prédiction", "🌡️ Données Climat", "📘 Tutoriel"])

# ----------------------- Page : Accueil -----------------------
if page == "🏠 Accueil":
    st.title("🌿 Détection intelligente des maladies des plantes")
    st.markdown("""
    Bienvenue sur notre application de **Smart Agriculture** !  
    Elle combine **Vision par Ordinateur** et **Internet des Objets (IoT)** pour :
    
    ✅ Diagnostiquer automatiquement les maladies des feuilles  
    ✅ Afficher les conditions climatiques en temps réel  
    ✅ Fournir une interface conviviale aux agriculteurs
    
    ---
    💡 *Entraîné sur plus de 50 000 images - modèle basé CNN (PyTorch)*
    """)
    st.image("hero_agriculture.jpg", use_container_width=True)

# ----------------------- Page : Tutoriel -----------------------
elif page == "📘 Tutoriel":
    st.title("📘 Comment utiliser l'application ?")
    st.markdown("""
    1. **Accédez à la section 📸 Prédiction**  
    2. Téléversez une image nette de **feuille malade**  
    3. Attendez quelques secondes pour voir le résultat  
    4. Consultez les alertes climatiques en 🧊 Données Climat  

    ℹ️ Les probabilités indiquent la **confiance du modèle**.

    ---
    🔗 Dataset : [New Plant Diseases Dataset (Kaggle)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
    """)

# ----------------------- Page : Climat -----------------------
elif page == "🌡️ Données Climat":
    st.title("🌡️ Données Climatiques en Temps Réel")
    st_autorefresh(interval=5000, limit=None, key="climate_refresh")

    TEMP_MAX = 35
    TEMP_MIN = 10
    HUM_MAX = 85
    HUM_MIN = 30

    try:
        data = db.reference('climat').get()
        if data:
            temperature = data.get('temperature')
            humidite = data.get('humidite')

            col1, col2 = st.columns(2)
            col1.metric("🌡️ Température (°C)", f"{temperature}")
            col2.metric("💧 Humidité (%)", f"{humidite}")

            if temperature > TEMP_MAX:
                st.error(f"🚨 Température trop élevée ! ({temperature}°C)")
            elif temperature < TEMP_MIN:
                st.warning(f"⚠️ Température trop basse ! ({temperature}°C)")

            if humidite > HUM_MAX:
                st.error(f"🚨 Humidité trop élevée ! ({humidite}%)")
            elif humidite < HUM_MIN:
                st.warning(f"⚠️ Humidité trop basse ! ({humidite}%)")

            # -------- Graphiques simulés (ou historiques si tu les stockes)
            st.markdown("### 📈 Historique (24h - simulé)")
            hours = [f"{(datetime.datetime.now() - datetime.timedelta(hours=i)).strftime('%Hh')}" for i in range(23, -1, -1)]
            temp_values = [round(random.uniform(18, 40), 1) for _ in hours]
            hum_values = [round(random.uniform(40, 90), 1) for _ in hours]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(hours, temp_values, label='🌡️ Température', color='orangered', marker='o')
            ax.plot(hours, hum_values, label='💧 Humidité', color='royalblue', marker='o')
            ax.set_xticks(hours[::2])
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        else:
            st.warning("Aucune donnée disponible dans le nœud 'climat'.")

    except Exception as e:
        st.error("❌ Erreur lors de la récupération des données Firebase.")
        st.text(str(e))

# ----------------------- Page : Prédiction -----------------------
elif page == "📸 Prédiction":
    st.title("📸 Détection Automatique de Maladie des Feuilles")

    uploaded_file = st.file_uploader("🌿 Téléversez une image de feuille malade", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Image chargée", use_container_width=True)

        # --- Prédiction
        try:
            class_predite, proba = predict_pytorch(image)  # <== ta fonction prédictive
            st.success(f"✅ **Classe prédite** : `{class_predite}`")
            st.info(f"📊 **Probabilité** : `{round(proba * 100, 2)}%`")
        except:
            st.error("Erreur lors de la prédiction. Assure-toi que le modèle est bien chargé.")

# ----------------------- Footer -----------------------
st.sidebar.markdown("---")
st.sidebar.markdown("👩‍💻 Réalisé par : *Kawtar ISMGANE & Manal ZERROUKI*  \n📅 2025 - 2026")
