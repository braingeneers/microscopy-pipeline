import cv2
import numpy as np
import sys

def find_max_brightness(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to load.")

    # Apply Gaussian blur to reduce random noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Find maximum pixel value in the image
    max_pixel = np.max(img)

    # Absolute maximum brightness for 8-bit image
    abs_max = 255

    # Calculate ratio
    ratio = max_pixel / abs_max

    print(f"Maximum brightness in image: {max_pixel}")
    print(f"Absolute maximum possible: {abs_max}")
    print(f"Ratio: {ratio:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python findmaxbrightness.py <image_path>")
    else:
        find_max_brightness(sys.argv[1])