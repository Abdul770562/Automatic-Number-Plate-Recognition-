import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 📌 Load YOLO and TrOCR models once
yolo_model = YOLO("C:/Users/Abdul/Downloads/anpr/license_plate_detector.pt")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")

def capture_image_from_webcam(save_path="captured_plate.jpg"):
    cap = cv2.VideoCapture(0)  # 0 is default camera

    if not cap.isOpened():
        raise Exception("❌ Could not open webcam.")

    print("📸 Press 'c' to capture, or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from webcam.")
            break

        cv2.imshow("Webcam - Press 'c' to capture", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            cv2.imwrite(save_path, frame)
            print(f"✅ Image captured and saved to {save_path}")
            break
        elif key & 0xFF == ord('q'):
            print("❌ Capture cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return save_path if os.path.exists(save_path) else None

def is_single_line_plate(crop):
    """
    Determine if a license plate is single-line or two-line using multiple criteria
    """
    h, w = crop.shape[:2]
    
    # 1. Basic aspect ratio check (single line plates are typically wider)
    aspect_ratio = w / h
    if aspect_ratio > 3.0:  # Very wide plates are definitely single line
        return True
    if aspect_ratio < 2.1:  # Very tall plates are definitely two line
        return False
    
    # 2. Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Analyze horizontal projection (sum of white pixels in each row)
    horizontal_projection = np.sum(binary, axis=1)
    
    # 4. Find significant peaks in the projection
    # Normalize the projection
    if np.max(horizontal_projection) > 0:
        normalized_projection = horizontal_projection / np.max(horizontal_projection)
    else:
        return True  # Default to single line if no clear features
    
    # 5. Find rows with significant text content (above threshold)
    text_threshold = 0.3  # Adjust this value based on your needs
    text_rows = normalized_projection > text_threshold
    
    # 6. Find continuous regions of text
    text_regions = []
    in_region = False
    start_row = 0
    
    for i, has_text in enumerate(text_rows):
        if has_text and not in_region:
            start_row = i
            in_region = True
        elif not has_text and in_region:
            if i - start_row > h * 0.1:  # Region must be at least 10% of height
                text_regions.append((start_row, i))
            in_region = False
    
    # Handle case where text continues to the end
    if in_region and len(text_rows) - start_row > h * 0.1:
        text_regions.append((start_row, len(text_rows)))
    
    # 7. Decision logic
    if len(text_regions) <= 1:
        return True  # Single line
    elif len(text_regions) == 2:
        # Check if the two regions are well separated
        gap = text_regions[1][0] - text_regions[0][1]
        if gap > h * 0.1:  # Gap is more than 10% of height
            return False  # Two line
        else:
            return True  # Single line with noise
    else:
        # More than 2 regions, likely single line with artifacts
        return True

def enhance_plate_image(crop):
    """
    Enhance the license plate image for better OCR
    """
    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Convert back to BGR for consistency
    enhanced_bgr = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    
    return enhanced_bgr

def detect_and_read_plate(img_path, save_debug=False):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image '{img_path}' not found.")

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (640, 640))

    results = yolo_model(img_resized)

    if not results or not results[0].boxes:
        return "No license plate detected."

    best_result = ""
    best_confidence = 0

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        
        crop = img_resized[y1:y2, x1:x2]

        if save_debug:
            cv2.imwrite(f"plate_crop_{i}.jpg", crop)

        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            continue

        # Enhance the image
        enhanced_crop = enhance_plate_image(crop)
        
        if save_debug:
            cv2.imwrite(f"enhanced_crop_{i}.jpg", enhanced_crop)

        # Determine if it's single or two-line plate
        is_single_line = is_single_line_plate(enhanced_crop)
        
        print(f"Plate {i+1}: {'Single' if is_single_line else 'Two'} line detected")

        if is_single_line:
            # Process as single line
            aligned = enhanced_crop
        else:
            # Process as two-line plate
            top = enhanced_crop[:h // 2, :]
            bottom = enhanced_crop[h // 2:, :]
            
            # Resize both parts to same height
            height = 64
            if top.shape[0] > 0 and top.shape[1] > 0:
                top_r = cv2.resize(top, (int(top.shape[1] * height / top.shape[0]), height))
            else:
                top_r = np.zeros((height, 100, 3), dtype=np.uint8)
                
            if bottom.shape[0] > 0 and bottom.shape[1] > 0:
                bottom_r = cv2.resize(bottom, (int(bottom.shape[1] * height / bottom.shape[0]), height))
            else:
                bottom_r = np.zeros((height, 100, 3), dtype=np.uint8)
            
            aligned = cv2.hconcat([top_r, bottom_r])

        if save_debug:
            cv2.imwrite(f"aligned_plate_{i}.jpg", aligned)

        # OCR
        image_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values

        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        clean_text = text.strip()
        
        # Choose the result with highest confidence
        if confidence > best_confidence:
            best_confidence = confidence
            best_result = clean_text
            
        print(f"Plate {i+1} text: {clean_text} (confidence: {confidence:.2f})")

    return best_result if best_result else "No valid plates detected."

def preprocess_for_ocr(image):
    """
    Additional preprocessing specifically for OCR
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply morphological operations to clean up
    kernel = np.ones((1,1), np.uint8)
    processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # Apply Gaussian blur to smooth
    processed = cv2.GaussianBlur(processed, (1, 1), 0)
    
    # Convert back to BGR
    return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

if __name__ == "__main__":
    # Option to test with existing image or capture new one
    test_mode = input("Test with existing image? (y/n): ").lower().strip()
    
    if test_mode == 'y':
        img_path = input("Enter image path: ").strip()
        if not os.path.exists(img_path):
            print(f"❌ Image '{img_path}' not found.")
            exit()
    else:
        # 1️⃣ Capture from webcam
        img_path = capture_image_from_webcam()
        
        if not img_path:
            print("❌ No image captured.")
            exit()

    # 2️⃣ Run detection and OCR
    output_file = "ocr_result.txt"
    result = detect_and_read_plate(img_path, save_debug=True)

    print("🧾 Final Detected Plate Text:", result)

    # Save to file
    with open(output_file, "w") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Timestamp: {now}\nPlate Text: {result}\n")

    print(f"✅ OCR result saved to '{output_file}'")

    # Show results if debug files exist
    debug_files = [f for f in os.listdir('.') if f.startswith('aligned_plate_')]
    if debug_files:
        for debug_file in debug_files:
            if os.path.exists(debug_file):
                aligned_img = cv2.imread(debug_file)
                plt.figure(figsize=(10, 3))
                plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.title(f"Processed Plate: {debug_file}")
                plt.show()