import torch
from ultralytics import YOLO
print(torch.cuda.is_available())
print(torch.cuda.device_count())


def main():
    model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)

    model.train(data='config.yaml', epochs=20,workers=2, batch=128,imgsz=640)

if __name__ == '__main__':
    main()
