from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Imagem'
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP, YOUTUBE]

# Images config
IMAGES_DIR = ROOT / 'assets/images'
DEFAULT_IMAGE = IMAGES_DIR / 'hotspot.png'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'assets/videos'
VIDEO_1_PATH = VIDEO_DIR / 'hotspot.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'panel_ir.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'panel_ir.mp4'
VIDEOS_DICT = {
    'Hotspot': VIDEO_1_PATH,
    'Painel IR': VIDEO_2_PATH,
    'Painel OPT': VIDEO_3_PATH,
}

# ML Model config
MODEL_DIR = ROOT / 'assets/weights'
DETECTION_MODEL = MODEL_DIR / 'hotspot.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'

# Webcam
WEBCAM_PATH = 0
