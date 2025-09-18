import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import cv2

# Configurações
weights = "yolov5s.pt"     # modelo treinado
device = select_device('cpu')  # '0' para GPU, 'cpu' para CPU
imgsz = (640, 640)

# Carrega o modelo
model = DetectMultiBackend(weights, device=device, dnn=False, data="data/coco128.yaml", fp16=False)
names = model.names  # nomes das classes

# Inicia a câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pré-processa a imagem (como no detect.py)
    img = letterbox(frame, imgsz, stride=model.stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1].copy()  # BGR → RGB, HWC → CHW \\ altera o formato da imagem
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inferência
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Processa as detecções
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                w, h = x2 - x1, y2 - y1
                label = names[int(cls)]
                print(f"Objeto: {label}, Confiança: {conf:.2f}, Caixa: x={x1}, y={y1}, w={w}, h={h}, FPS: {fps:.2f}")

                # Aqui você pode usar a posição do objeto (centro da caixa)
                cx = x1 + w // 2
                cy = y1 + h // 2

                # Exemplo: desenhar na tela
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Aqui é onde entra a lógica para o carrinho
                if cx < frame.shape[1]//2:
                    print("Vira esquerda")
                else:
                    print("Vira direita")

    cv2.imshow("Detecção", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
