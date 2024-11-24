from ultralytics import YOLO


class YOLOProcessor:
    def __init__(self, model_path, conf_thresh, img_size, device):
        self.device = device
        self.model = YOLO(model_path).to(self.device)
        self.model.conf = conf_thresh
        self.model.imgsz = img_size

    def infer(self, frame):
        return self.model(frame, device=self.device, verbose=False)
