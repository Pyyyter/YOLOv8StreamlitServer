#region imports
# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper
#endregion

# Setting page layout
st.set_page_config(
    page_title="Olho de Deus",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
# st.title("Detecção")

# Sidebar
st.sidebar.header("Ajuste de parâmetros")

# Model Options
model_type = st.sidebar.radio(
    "Tarefa", ['Detecção', 'Segmentação'])

confidence = float(st.sidebar.slider(
    "Confiança", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detecção':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentação':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Não foi possível carregar o modelo. O caminho definido foi: {model_path}")
    st.error(ex)

st.sidebar.header("Configuração da fonte de detecção")
source_radio = st.sidebar.radio(
    "Selecionar fonte", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Escolha a imagem...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Imagem padrão",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Imagem enviada",
                        use_column_width=True)
        except Exception as ex:
            st.error("Ocorreu um erro ao abrir a imagem!")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Resultado',
                     use_column_width=True)
        else:
            if st.sidebar.button('Processar imagem'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Resultado',
                         use_column_width=True)
                try:
                    with st.expander("Resultado"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("A imagem ainda não foi enviada!")
                    st.write(ex)

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Selecione uma fonte válida!")
