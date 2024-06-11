#region imports
# Python In-built packages
from pathlib import Path
import PIL
import cv2
import numpy as np
from lrp.yolo import YOLOv8LRP
# External packages
import streamlit as st

# Local Modules
import settings
import helper
import torchvision
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
    lrp = YOLOv8LRP(model, power=2, eps=1e-05, device='cpu')
except Exception as ex:
    st.error(f"Erro carregando models")
    st.error(ex)

st.sidebar.header("Selecione a fonte de dados")
source_radio = st.sidebar.radio(
    "Selecionar fonte", settings.SOURCES_LIST)
use_heatmap = st.sidebar.checkbox("Usar mapa de calor LRP", value=False)
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
            if use_heatmap:
                if st.sidebar.button('Processar imagem'):
                    
                        desired_size = (512, 640)
                        transform = torchvision.transforms.Compose([
                            torchvision.transforms.Resize(desired_size),
                            torchvision.transforms.ToTensor(),
                        ])

                        image = transform(uploaded_image).to('cpu').float()
                        explanation_lrp = lrp.explain(image, contrastive=False).cpu()
                        out_img2=explanation_lrp.detach().numpy()
                        out_img2_normalized = (out_img2 - out_img2.min()) / (out_img2.max() - out_img2.min())
                        scaled = (out_img2_normalized * 255).astype(np.uint8)
                        st.image(scaled, caption='Resultado',
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
