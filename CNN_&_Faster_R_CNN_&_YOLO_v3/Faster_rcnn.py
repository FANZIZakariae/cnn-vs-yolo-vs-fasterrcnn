import tensorflow as tf
import numpy as np
import cv2

try:
    import tensorflow_hub as hub
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The 'tensorflow_hub' module is required. Install it with:\n\n    pip install tensorflow-hub"
    )


class Detector:
    """
    Object Detection using TensorFlow Hub Faster R-CNN ResNet50.
    """

    def __init__(self, model_url='https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1',
                 coco_labels_path='./coco.names', score_threshold=0.5):
        self.load_model(model_url)
        self.load_coco_labels(coco_labels_path)
        self.set_detection_params(score_threshold)

    def load_model(self, model_url):
        print(f'* Loading model from TensorFlow Hub: {model_url}')
        self.detector = hub.load(model_url)
        print('* Model loaded successfully')

    def load_coco_labels(self, coco_labels_path):
        try:
            with open(coco_labels_path, 'rt') as f:
                self.labels = f.read().strip().split('\n')
            print(f'* Loaded COCO labels from {coco_labels_path}')
        except FileNotFoundError:
            # Fallback default 90 COCO labels
            self.labels = [f'class_{i}' for i in range(90)]
            print('* Warning: coco.names file not found. Using dummy labels.')

    def set_detection_params(self, score_threshold):
        self.score_threshold = score_threshold

    def run_detection_on_img(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.uint8)[tf.newaxis, ...]
        detections = self.detector(input_tensor)

        boxes = detections['detection_boxes'][0].numpy()
        class_ids = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()

        h, w, _ = img.shape
        results = []

        for i in range(len(scores)):
            if scores[i] < self.score_threshold:
                continue

            y1, x1, y2, x2 = boxes[i]
            x, y, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            box_w, box_h = x2 - x, y2 - y

            label = self.labels[class_ids[i] - 1] if 0 < class_ids[i] <= len(self.labels) else "N/A"
            confidence = f"{int(scores[i] * 100)}%"

            results.append({
                'label': label,
                'confidence': confidence,
                'bbox_xywh': [x, y, box_w, box_h]
            })

        return results

    def draw_detections(self, img, detections):
        for det in detections:
            x, y, w, h = det['bbox_xywh']
            label = det['label']
            conf = det['confidence']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f'{label} {conf}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img
