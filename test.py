from ultralytics import YOLO
from lrp.yolo import YOLOv8LRP
from PIL import Image
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision
image = Image.open("iksi.jpg")

desired_size = (512, 640)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(desired_size),
    torchvision.transforms.ToTensor(),
])

image = transform(image).to('cpu').float()

yolo = YOLO('assets/weights/photovoltaicModule.pt')
detection = yolo(image)
# Instantiate a w2-rule relevance propagation object

lrp = YOLOv8LRP(yolo, power=2, eps=1e-05, device='cpu')
explanation_lrp = lrp.explain(image, contrastive=False).cpu()

# Plotting the explanation for class 'person crp'
out_img2=explanation_lrp.detach().numpy()
# arr_ = np.squeeze(out_img2)
# plt.imshow(arr_)
# plt.show()
out_img2_normalized = (out_img2 - out_img2.min()) / (out_img2.max() - out_img2.min())

# Scale values to range [0, 255] and convert to uint8
out_img2_uint8 = (out_img2_normalized * 255).astype(np.uint8)

# Save the image using OpenCV
cv2.imwrite('output_image.jpg', out_img2_uint8)