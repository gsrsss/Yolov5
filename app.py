import streamlit as st
from PIL import Image
import torch
import numpy as np
import io

# --- CONFIGURACIÓN DE PÁGINA Y TÍTULO PERSONALIZADO ---
st.set_page_config(
    page_title="Máquina de Reconocimiento de Objetos (YOLOv5)",
    page_icon="🤖",
    layout="wide"
)

# Título y descripción con un toque personal
st.title("Object Recognition Machine! (ง ͠° ͟ل͜ ͡°)ง")
st.markdown(
    """
    ¡Con el poder de **YOLOv5** podemos reconocer los objetos que hay en una imagen!
    
    ¿No te lo crees? ¡Sube una foto o tómate una con un objeto en la mano y mira cómo funciona!
    """
)
st.markdown("---")

# --- CARGA DEL MODELO YOLOv5 ---
@st.cache_resource
def load_model():
    """Carga el modelo YOLOv5 una sola vez para mejorar el rendimiento."""
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # Ajusta el umbral de confianza para filtrar detecciones débiles (ej. 0.25)
        model.conf = 0.25 
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo YOLOv5. Asegúrate de tener conexión a Internet y PyTorch instalado. Error: {e}")
        return None

# Cargar el modelo
model = load_model()

if model:
    # --- INTERFAZ DE USUARIO PARA SUBIR IMAGEN ---
    
    # 1. Subir un archivo
    uploaded_file = st.file_uploader(
        "Sube una imagen (JPEG, PNG) para detectar objetos:", 
        type=['png', 'jpg', 'jpeg']
    )
    
    # 2. Capturar desde la cámara
    camera_image = st.camera_input("...o utiliza tu cámara para una detección en tiempo real 📸")
    
    # Determinar la fuente de la imagen
    if uploaded_file is not None:
        image_source = uploaded_file
        st.sidebar.info("Archivo cargado correctamente. (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧")
    elif camera_image is not None:
        image_source = camera_image
        st.sidebar.info("Imagen capturada desde la cámara. ✨")
    else:
        # Mensaje de bienvenida/espera si no hay imagen
        st.info("Esperando tu imagen. ¡Sube o captura para empezar la magia! 🪄")
        image_source = None

    st.markdown("---")

    # --- PROCESAMIENTO Y DETECCIÓN DE OBJETOS ---
    if image_source is not None:
        try:
            # 1. Preparar la imagen
            image = Image.open(image_source)
            
            st.markdown("## 🔎 Resultado de la Detección")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Imagen Original:")
                st.image(image, caption='Imagen de entrada', use_column_width=True)
            
            # 2. Realizar la inferencia con YOLOv5
            results = model(image) 
            
            # Obtener las detecciones en formato DataFrame de pandas
            detections_df = results.pandas().xyxy[0]
            
            # 3. Generar la imagen con las cajas delimitadoras
            img_with_boxes = Image.fromarray(results.render()[0])

            with col2:
                st.subheader("Detección de YOLOv5:")
                st.image(img_with_boxes, caption='Objetos Reconocidos', use_column_width=True) 
                
            st.markdown("---")

            # 4. Mostrar el resumen de las detecciones
            st.markdown("## 📊 Informe de Objetos Encontrados:")
            
            if not detections_df.empty:
                # Preparar el DataFrame para una visualización amigable
                detections_df = detections_df[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']]
                detections_df.columns = ['Objeto Detectado',
