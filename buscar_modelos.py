import os
import google.generativeai as genai

# 1. Pon tu clave aquí
API_KEY = "AQ.Ab8RN6Ia2hMJsr1vIaSI4aeIRs655rE-XRcwit_rPPWcXGTOGQ"
genai.configure(api_key=API_KEY)

print("Buscando modelos compatibles con tu cuenta...")
print("-" * 40)

# 2. Le pedimos a Google la lista exacta
for modelo in genai.list_models():
    # Filtramos solo los modelos que sirven para generar texto (chat)
    if "generateContent" in modelo.supported_generation_methods:
        print(f"Nombre exacto a usar: {modelo.name.replace('models/', '')}")
