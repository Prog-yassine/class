import os
import time
import numpy as np
import cv2
from picamera2 import Picamera2  # Native modern Raspberry Pi camera driver

# Configurations
MODEL_PATH = "yolov8n.onnx"
CONF_THRESHOLD = 0.35

# ⚠️ Must match the exact order your classes were arranged during training
CLASS_NAMES = ["apple", "banana", "orange"]

COLORS = {
    "apple": (0, 0, 255),     # Red
    "banana": (0, 255, 255),  # Yellow
    "orange": (0, 165, 255)   # Orange
}
DEFAULT_COLOR = (0, 255, 0)

def main():
    # 1. Load ONNX Model via OpenCV DNN
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found in this folder.")
        return

    print("🔄 Loading ONNX model into OpenCV DNN...")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)

    # 2. Initialize native Picamera2 (Exactly like the YouTuber)
    print("🔄 Initializing Picamera2 hardware link...")
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1280, 1280)
    picam2.preview_configuration.main.format = "BGR888"  
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    print("🚀 Camera active via native drivers! Press 'q' to exit.")

    while True:
        start_time = time.time()

        # 3. Capture frame-by-frame from Picamera2
        frame = picam2.capture_array()
        orig_h, orig_w, _ = frame.shape

        # 4. Preprocess Image for YOLOv8 (640x640)
        blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        
        # 5. Run Model Inference (Entirely Offline)
        outputs = net.forward()
        
        # Safe matrix handling for 32-bit architecture
        predictions = outputs[0] if len(outputs.shape) == 3 else outputs
        if predictions.shape[0] < predictions.shape[1]:
            predictions = np.transpose(predictions, (1, 0))

        # 6. Parse Predictions and Draw Squares
        for pred in predictions:
            bounding_box = pred[0:4]
            class_scores = pred[4:]
            
            if len(class_scores) == 0:
                continue
                
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence >= CONF_THRESHOLD:
                if class_id < len(CLASS_NAMES):
                    label_name = CLASS_NAMES[class_id]
                    
                    # Convert coordinates to real camera frame pixels
                    cx, cy, bw, bh = bounding_box
                    x1 = int((cx - bw / 2) * (orig_w / 640.0))
                    y1 = int((cy - bh / 2) * (orig_h / 640.0))
                    x2 = int((cx + bw / 2) * (orig_w / 640.0))
                    y2 = int((cy + bh / 2) * (orig_h / 640.0))

                    box_color = COLORS.get(label_name, DEFAULT_COLOR)

                    # Draw item square
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

                    # Draw text label background header
                    text_str = f"{label_name.upper()} ({confidence*100:.0f}%)"
                    (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), box_color, -1)
                    cv2.putText(frame, text_str, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 7. Calculate and display live FPS stats
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0.0
        fps_text = f"FPS: {fps:.1f}"
        
        cv2.putText(frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 8. Render window layout
        cv2.imshow("Camera View - Kiosk AI", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean cleanup
    picam2.stop()
    cv2.destroyAllWindows()
    print("👋 Kiosk system closed successfully.")

if __name__ == "__main__":
    main()
