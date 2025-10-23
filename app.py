import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit
st.set_page_config(
    page_title="🤖 Object recognition machine.",
    page_icon="📸", # Cambié el ícono a una cámara
    layout="wide"
)

# Función para cargar el modelo YOLOv5 de manera compatible con versiones anteriores de PyTorch
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        # Importar yolov5
        import yolov5

        # Intento de carga con manejo de compatibilidad para PyTorch
        try:
            # Primer método: cargar con weights_only=False si la versión lo soporta
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            # Segundo método: si el primer método falla, intentar un enfoque más básico
            try:
                model = yolov5.load(model_path)
                return model
            except Exception as e:
                # Si todo falla, intentar cargar el modelo con torch directamente (hub)
                st.warning(f"⚠️ Intentando método alternativo de carga (torch.hub)...")

                # Modificar sys.path temporalmente para poder importar torch correctamente (aunque hub.load ya maneja mucho)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)

                # Cargar el modelo con torch directamente
                # Usamos map_location para asegurar que se cargue en la CPU si no hay GPU, por seguridad en entornos variados
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, map_location=device)
                return model

    except Exception as e:
        st.error(f"❌ ¡CRASH! Error al cargar el modelo: {str(e)}")
        st.info("""
        **¡Ups! Parece que tuvimos problemas con YOLOv5.** (・へ・)
        
        **Recomendaciones para el despliegue local:**
        1. **Instalar una versión compatible de PyTorch y YOLOv5** (por ejemplo, para compatibilidad):
           ```
           pip install torch==1.12.0 torchvision==0.13.0
           pip install yolov5==7.0.9
           ```
        2. **Verifica la ruta** de tu archivo de modelo (si usas uno local) o asegúrate de tener **conexión a internet** para descargar el pre-entrenado.
        """)
        return None

# Título y descripción de la aplicación con el nuevo estilo
st.title("🤖 Object recognition machine.")
st.markdown("""
¡Con el poder de **YOLOv5** podemos reconocer los objetos que hay en una imagen!
