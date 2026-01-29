import random
import time
import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("service_account.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://smart-agriculture-1-default-rtdb.firebaseio.com/'
    })

ref = db.reference('climat')

while True:
    temperature = round(random.uniform(18, 40), 1)
    humidite = round(random.uniform(40, 90), 1)
    ref.set({
        'temperature': temperature,
        'humidite': humidite
    })
    print(f"Données envoyées : {temperature}°C / {humidite}%")
    time.sleep(5)
