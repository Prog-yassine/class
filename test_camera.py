import os
import time
import numpy as np
import cv2

# System Configurations
CAMERA_INDEX = 0  # Change to 1 if you have multiple cameras connected
MODEL_PATH = "yolov8n.onnx"
CONF_THRESHOLD = 0.35  # Minimum detection confidence level (35%)

# ⚠️ MUST match the exact order your classes were arranged during training
CLASS_NAMES = ["apple", "banana", "orange"]

# Box colors in BGR format
COLORS = {
    "apple": (0, 0, 255),     # Solid Red
    "banana": (0, 255, 255),  # Solid Yellow
    "orange": (0, 165, 255)   # Vivid Orange
}
DEFAULT_COLOR = (0, 255, 0)   # Green fallback

def main():
    # 1. Check Model Existence
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' was not found in this directory.")
        print("Please move your exported ONNX model into this folder to proceed.")
        return

    print("🔄 Initializing OpenCV DNN Engine (Offline mode)...")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    
    # 2. Start Camera Feed Hardware
    print("🔄 Opening live camera stream window...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌ Error: Cannot access camera index {CAMERA_INDEX}.")
        print("Please check physical connections or switch CAMERA_INDEX to 1.")
        return

    # Set standard hardware capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Establish named desktop render frame
    window_name = "Raspberry Pi 4 - Kiosk AI Camera View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("🚀 Video pipeline active! Look at your desktop.")
    print("👉 Click on the video window and press 'q' to shut down.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        orig_h, orig_w, _ = frame.shape

        # 3. Scale and normalize image frame to YOLO standard square sizes
        blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        
        # 4. Generate forward calculation array matrices
        outputs = net.forward()
        
        # Dynamic matrix reshape to safe shape conversion regardless of export tool variants
        if len(outputs.shape) == 3:
            predictions = outputs[0]
        else:
            predictions = outputs

        # Check if row and column arrays are reversed (Common variation across platforms)
        if predictions.shape[0] < predictions.shape[1]:
            predictions = np.transpose(predictions, (1, 0))

        # 5. Process tracking positions and draw on overlay layer
        for pred in predictions:
            # Safely split class fields away from coordinates mapping bounds
            bounding_box = pred[0:4]
            class_scores = pred[4:]
            
            if len(class_scores) == 0:
                continue
                
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            # Bounding box filter block execution
            if confidence >= CONF_THRESHOLD:
                if class_id < len(CLASS_NAMES):
                    label_name = CLASS_NAMES[class_id]
                    
                    # Read relative midpoint vectors
                    cx, cy, bw, bh = bounding_box
                    
                    # Calculate real standard absolute pixel scale targets
                    x1 = int((cx - bw / 2) * (orig_w / 640.0))
                    y1 = int((cy - bh / 2) * (orig_h / 640.0))
                    x2 = int((cx + bw / 2) * (orig_w / 640.0))
                    y2 = int((cy + bh / 2) * (orig_h / 640.0))

                    # Apply color configurations
                    box_color = COLORS.get(label_name, DEFAULT_COLOR)

                    # Draw square above item bounding frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

                    # Compose label background header tag text string bounds
                    text_str = f"{label_name.upper()} ({confidence*100:.0f}%)"
                    (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Draw solid block behind text label for contrast
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), box_color, -1)
                    cv2.putText(frame, text_str, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 6. Render the composite frame image to our open window UI
        cv2.imshow(window_name, frame)

        # Break active tracking loop cleanly on 'q' terminal keyboard interruption
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Shutdown drivers and release handles
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Camera tracking system closed safely.")

if __name__ == "__main__":
    main()
