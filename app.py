import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit
st.set_page_config(
    page_title="Detector YOLOv5 (⌐■_■)",
    page_icon="🤖",
    layout="wide"
)

# Función para cargar el modelo YOLOv5 (Sin cambios en la lógica interna)
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        # Importar yolov5
        import yolov5

        # Para versiones de PyTorch anteriores a 2.0, cargar directamente con weights_only=False
        # o usar el parámetro map_location para asegurar compatibilidad
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
                # Si todo falla, intentar cargar el modelo con torch directamente
                st.warning(f"Intentando método alternativo de carga...")

                # Modificar sys.path temporalmente para poder importar torch correctamente
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)

                # Cargar el modelo con torch directamente
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model

    except Exception as e:
        st.error(f"❌ ¡Oh no! Error al cargar el modelo: {str(e)}")
        st.info("""
        Recomendaciones:
        1. Instalar una versión compatible de PyTorch y YOLOv5:
           ```
           pip install torch==1.12.0 torchvision==0.13.0
           pip install yolov5==7.0.9
           ```
        2. Asegúrate de tener el archivo del modelo en la ubicación correcta
        3. Si el problema persiste, intenta descargar el modelo directamente de torch hub
        """)
        return None

# Título y descripción de la aplicación
st.title("🤖 Máquina de Reconocimiento de Objetos")
st.markdown("""
Con el poder de **YOLOv5**, ¡podemos reconocer los objetos que hay en una imagen! (☉_☉)
¿No te lo crees? Tómate una foto con un objeto en la mano y ¡mira cómo funciona!
""")

# Cargar el modelo
with st.spinner("Iniciando el motor YOLOv5... (ɔ■_■)ɔ"):
    model = load_yolov5_model()

# Si el modelo se cargó correctamente, configuramos los parámetros
if model:
    # Sidebar para los parámetros de configuración
    st.sidebar.title("Panel de Control 🔧")

    # Ajustar parámetros del modelo
    with st.sidebar:
        st.subheader('Ajustes de Detección')
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")

        # Opciones adicionales
        st.subheader('Opciones Avanzadas (¡con cuidado!)')
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False)
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False)
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10)
        except:
            st.warning("Algunas opciones avanzadas no están disponibles")

    # Contenedor principal para la cámara y resultados
    main_container = st.container()

    with main_container:
        # Capturar foto con la cámara
        picture = st.camera_input("¡Toma la foto aquí! 📸", key="camera")

        if picture:
            # Procesar la imagen capturada
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            # Realizar la detección
            with st.spinner("Analizando pixeles... (l■_■)l"):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()

            # Asegúrate de que 'try' y 'except' estén al mismo nivel de indentación
            try:
                # Parsear resultados
                predictions = results.pred[0] # Esto se mantiene por si se usa luego
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]

                # Mostrar resultados
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Tu Foto (con magia ✨)")
                    # Renderizar las detecciones en la imagen
                    results.render()
                    # ----- CORRECCIÓN AQUÍ -----
                    # Cambiamos use_column_width por use_container_width
                    st.image(cv2_img, channels='BGR', use_container_width=True)
                    # ---------------------------

                with col2:
                    st.subheader("¿Qué encontramos? 🧐")
                    # (El código original estaba incompleto aquí)
                    # Mostramos los resultados en un DataFrame de pandas
                    df_results = results.pandas().xyxy[0]
                    # Simplificamos la tabla para el portafolio
                    df_results = df_results[['name', 'confidence']]
                    df_results['confidence'] = df_results['confidence'].apply(lambda x: f"{x*100:.1f}%")
                    df_results.rename(columns={'name': 'Objeto', 'confidence': 'Confianza'}, inplace=True)
                    
                    if df_results.empty:
                        st.success("¡No se detectó nada! (O soy muy tímido) (・_・;)")
                    else:
                        st.dataframe(df_results, use_container_width=True, hide_index=True)
            
            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")
else:
    st.error("El modelo no pudo ser cargado. La aplicación no puede continuar. (T_T)")
