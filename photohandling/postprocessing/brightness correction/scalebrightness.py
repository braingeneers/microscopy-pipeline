import cv2
import numpy as np
import sys
import os

def scale_brightness(image_path, output_path, scale):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    img_scaled = cv2.convertScaleAbs(img, alpha=scale, beta=0)
    cv2.imwrite(output_path, img_scaled)
    print(f"Saved scaled image to {output_path}")

def process_folder(input_folder, output_folder, scale):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            scale_brightness(input_path, output_path, scale)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scalebrightness.py <input_folder> <output_folder> <scale>")
        print("Example: python scalebrightness.py input_folder output_folder 1.5")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    scale = float(sys.argv[3])
    process_folder(input_folder, output_folder, scale)