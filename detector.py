import cv2
import numpy as np
import matplotlib.pyplot as plt

class StoneCrackDetector:
    def __init__(self):
        self.kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.kernel_7x7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(blurred)

    def detect_edges_canny(self, image, low_threshold=50, high_threshold=150):
        return cv2.Canny(image, low_threshold, high_threshold)

    def detect_morphological_cracks(self, image):
        black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, self.kernel_7x7)

        if black_hat.max() > 0:
            black_hat = 255 * (black_hat - black_hat.min()) / (black_hat.max() - black_hat.min())

        black_hat = np.uint8(black_hat)
        binary = np.where(black_hat > 10, 255, 0).astype(np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel_3x3)

    def detect_gradient_based_cracks(self, image):
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)

        mag = np.sqrt(grad_x**2 + grad_y**2)
        mag = np.uint8(mag / mag.max() * 255) if mag.max() > 0 else np.uint8(mag)

        return np.where(mag > 30, 255, 0).astype(np.uint8)

    def combine_detection_methods(self, c, m, g):
        combined = cv2.bitwise_or(c, m)
        combined = cv2.bitwise_or(combined, g)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self.kernel_5x5)

        nb, labels, stats, _ = cv2.connectedComponentsWithStats(combined, 8)
        output = np.zeros_like(combined)

        for i in range(1, nb):
            if stats[i, cv2.CC_STAT_AREA] >= 50:
                output[labels == i] = 255

        return output

    def detect_cracks(self, image):
        pre = self.preprocess_image(image)
        c = self.detect_edges_canny(pre)
        m = self.detect_morphological_cracks(pre)
        g = self.detect_gradient_based_cracks(pre)

        return self.combine_detection_methods(c, m, g), pre

    def calculate_crack_metrics(self, cracks):
        total = cracks.size
        crack_pixels = np.sum(cracks > 0)

        return {
            "crack_density_percent": (crack_pixels / total) * 100,
            "num_crack_segments": cv2.connectedComponents(cracks)[0] - 1,
            "total_crack_pixels": crack_pixels
        }

    def visualize_results(self, original, pre, cracks):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(12,4))
        ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        ax[1].imshow(pre, cmap='gray')
        ax[2].imshow(cracks, cmap='gray')

        for a in ax:
            a.axis('off')

        plt.show()
