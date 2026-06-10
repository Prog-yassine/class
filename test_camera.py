import os
import time
import numpy as np
import cv2

# Configuration
CAMERA_INDEX = 0  # Default camera index for Raspberry Pi / Linux
MODEL_PATH = "yolov8n.onnx"
CONF_THRESHOLD = 0.40

# ⚠️ CRITICAL: Must match the EXACT class order of your trained dataset
CLASS_NAMES = ["apple", "banana", "orange"]

# Define a distinct color for each object box (BGR format)
COLORS = {
    "apple": (0, 0, 255),    # Red
    "banana": (0, 255, 255), # Yellow
    "orange": (0, 165, 255)  # Orange
}
DEFAULT_COLOR = (0, 255, 0)   # Green fallback

def main():
    # 1. Verify Model File
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: '{MODEL_PATH}' not found in this folder.")
        print("Please place your simplified ONNX model here before running.")
        return

    print("🔄 Loading ONNX model into OpenCV DNN...")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    
    # 2. Initialize Camera Feed
    print("🔄 Opening camera view...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera index {CAMERA_INDEX}.")
        return

    # Set Resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("🚀 Camera active! Press 'q' on your keyboard to exit.")

    while True:
        ret, frame = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.read()
        ret, frame = frame if isinstance(frame, tuple) else (ret, frame)
        if not ret:
            print("⚠️ Warning: Failed to grab camera frame.")
            time.sleep(0.01)
            continue

        h, w, _ = frame.shape

        # 3. Preprocess Image for YOLOv8 (640x640)
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        
        # 4. Run Model Inference (Entirely Offline)
        outputs = net.forward()
        
        # Shape handling for 32-bit systems
        predictions = np.squeeze(outputs)
        if len(predictions.shape) == 3:
            predictions = predictions[0]
            
        if predictions.shape[0] < predictions.shape[1]:
            predictions = np.transpose(predictions, (1, 0))

        # 5. Process Detections and Draw Boxes
        for pred in predictions:
            # YOLOv8 format: pred[0:4] are bounding box coordinates (cx, cy, w, h)
            # pred[4:] are class probabilities
            scores = pred[4:]
            if len(scores) == 0:
                continue
                
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence >= CONF_THRESHOLD:
                if class_id < len(CLASS_NAMES):
                    label_name = CLASS_NAMES[class_id]
                    
                    # Convert YOLO relative bounding boxes back to pixels
                    cx, cy, box_w, box_h = pred[0:4]
                    
                    # Map from 640x640 coordinate space back to real camera frame space
                    x1 = int((cx - box_w / 2) * (w / 640.0))
                    y1 = int((cy - box_h / 2) * (h / 640.0))
                    x2 = int((cx + box_w / 2) * (w / 640.0))
                    y2 = int((cy + box_h / 2) * (h / 640.0))

                    # Select bounding box color
                    box_color = COLORS.get(label_name, DEFAULT_COLOR)

                    # Draw rectangle box above object
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

                    # Draw label background block and text string
                    text_str = f"{label_name.upper()} {confidence*100:.0f}%"
                    (text_w, text_h), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_w, y1), box_color, -1)
                    cv2.putText(frame, text_str, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 6. Render live display window on screen
        cv2.imshow("Raspberry Pi AI Camera View", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Camera system closed.")

if __name__ == "__main__":
    main()
