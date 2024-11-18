import os
import cv2
import torch
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from torchvision.ops import nms

def is_daytime(image, brightness_threshold=82.3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness > brightness_threshold

def apply_nms(boxes, scores, classes, iou_threshold=0.5, score_threshold=0.2, max_boxes=100):
    if len(boxes) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    mask = scores > score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]
    
    if len(boxes) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    final_boxes = []
    final_scores = []
    final_classes = []
    
    unique_classes = torch.unique(classes)
    
    for cls in unique_classes:
        cls_mask = classes == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        keep_indices = nms(cls_boxes, cls_scores, iou_threshold=iou_threshold)
        
        final_boxes.append(cls_boxes[keep_indices])
        final_scores.append(cls_scores[keep_indices])
        final_classes.append(classes[cls_mask][keep_indices])
    
    if final_boxes:
        final_boxes = torch.cat(final_boxes, dim=0)
        final_scores = torch.cat(final_scores, dim=0)
        final_classes = torch.cat(final_classes, dim=0)
        
        sorted_indices = torch.argsort(final_scores, descending=True)
        if len(final_boxes) > max_boxes:
            sorted_indices = sorted_indices[:max_boxes]
        
        final_boxes = final_boxes[sorted_indices]
        final_scores = final_scores[sorted_indices]
        final_classes = final_classes[sorted_indices]
    else:
        final_boxes = torch.tensor([], device=boxes.device)
        final_scores = torch.tensor([], device=scores.device)
        final_classes = torch.tensor([], device=classes.device)

    return final_boxes, final_scores, final_classes

def save_predictions_to_txt(filename, outputs, metadata, output_file, img_width, img_height, score_threshold=0.2, iou_threshold=0.5):
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    scores = instances.scores if instances.has("scores") else None
    classes = instances.pred_classes if instances.has("pred_classes") else None

    if boxes is not None and scores is not None and classes is not None:
        boxes, scores, classes = apply_nms(boxes.tensor, scores, classes, iou_threshold=iou_threshold)

        with open(output_file, "a") as f:
            for i in range(len(boxes)):
                confidence_score = scores[i].item()
                if confidence_score >= score_threshold:
                    box = boxes[i].numpy().flatten()
                    x0, y0, x1, y1 = box
                    w, h = x1 - x0, y1 - y0
                    
                    x_center = x0 + w / 2
                    y_center = y0 + h / 2
                    
                    x_center /= img_width
                    y_center /= img_height
                    w /= img_width
                    h /= img_height
                    
                    label = metadata.thing_classes[classes[i]]
                    
                    f.write(f"{filename} {label} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {confidence_score:.4f}\n")

def main():
    # Thiết lập predictors
    cfg_day = get_cfg()
    cfg_day.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_day.MODEL.WEIGHTS = "/app/models/best_day.pth"
    cfg_day.MODEL.ROI_HEADS.NUM_CLASSES = 4
    predictor_day = DefaultPredictor(cfg_day)

    cfg_night = get_cfg()
    cfg_night.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_night.MODEL.WEIGHTS = "/app/models/best_night.pth"
    cfg_night.MODEL.ROI_HEADS.NUM_CLASSES = 4
    predictor_night = DefaultPredictor(cfg_night)

    # Thiết lập metadata
    MetadataCatalog.get("daytime_metadata").thing_classes = ["0", "1", "2", "3"]
    MetadataCatalog.get("nighttime_metadata").thing_classes = ["0", "1", "2", "3"]

    # Đường dẫn input/output từ environment variables
    input_dir = os.environ.get('INPUT_DIR', '/app/input')
    output_file = os.environ.get('OUTPUT_FILE', '/app/output/predictions.txt')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Biến đếm
    dem = 0
    sang = 0

    # Xử lý từng ảnh
    for idx, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            img_height, img_width = image.shape[:2]

            if is_daytime(image):
                sang += 1
                outputs = predictor_day(image)
                metadata = MetadataCatalog.get("daytime_metadata")
            else:
                dem += 1
                outputs = predictor_night(image)
                metadata = MetadataCatalog.get("nighttime_metadata")

            save_predictions_to_txt(filename, outputs, metadata, output_file, 
                                 img_width, img_height, score_threshold=0.2, iou_threshold=0.5)
            
            print(f"Processed file {idx + 1}: {filename}")

    print(f"Kết quả dự đoán đã được lưu vào file {output_file}")
    print(f"Tổng số ảnh ban ngày: {sang}, Tổng số ảnh ban đêm: {dem}")

if __name__ == "__main__":
    main()