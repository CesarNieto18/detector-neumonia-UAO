import numpy as np
from tensorflow.keras.models import load_model

def model_fun():
    try:
        model = load_model("conv_MLP_84.h5")
        print("Modelo cargado correctamente.")
        return model
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return None


