import cv2
from detector import StoneCrackDetector

image_path = "data/test.jpg"

detector = StoneCrackDetector()

image = cv2.imread(image_path)

cracks, pre = detector.detect_cracks(image)
metrics = detector.calculate_crack_metrics(cracks)

print(metrics)

detector.visualize_results(image, pre, cracks)
