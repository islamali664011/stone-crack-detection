import os
import cv2
from detector import StoneCrackDetector

detector = StoneCrackDetector()

image_folder = "data/images"

images = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg",".png",".jpeg"))]

results = []

print("عدد الصور:", len(images))

for img_name in images:

    path = os.path.join(image_folder, img_name)
    image = cv2.imread(path)

    if image is None:
        print("مشكلة في الصورة:", img_name)
        continue

    cracks, pre = detector.detect_cracks(image)
    metrics = detector.calculate_crack_metrics(cracks)

    results.append({
        "image": img_name,
        "density": metrics['crack_density_percent'],
        "segments": metrics['num_crack_segments'],
        "pixels": metrics['total_crack_pixels']
    })

    print("=================================")
    print("الصورة:", img_name)
    print("Density:", metrics['crack_density_percent'])
    print("Segments:", metrics['num_crack_segments'])

    detector.visualize_results(image, pre, cracks)
