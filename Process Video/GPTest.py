import cv2
import numpy as np

# Carregar arquivos YOLO
net = cv2.dnn.readNet("C:\Users\pyyyt\iCloudDrive\Codes\yolov8-streamlit-detection-tracking\weights\yolov8n.pt", "caminho_para_arquivo_config")
classes = []
with open("caminho_para_arquivo_classes", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Carregar vídeo
video = cv2.VideoCapture("caminho_para_o_video.mp4")

# Definir codec e criar objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_video_com_deteccoes.avi', fourcc, 20.0, (int(video.get(3)), int(video.get(4))))

while True:
    ret, frame = video.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detectar objetos no frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Processar detecções
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Obter coordenadas do objeto detectado
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordenadas do retângulo
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y + 30), font, 2, color, 2)

    output_video.write(frame)
    cv2.imshow('Detecção de Objetos', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
output_video.release()
cv2.destroyAllWindows()
