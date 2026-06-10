import os
import time
import threading
import numpy as np
import cv2
from picamera2 import Picamera2

# Configurations
MODEL_PATH = "yolov8n.onnx"
CONF_THRESHOLD = 0.35
CLASS_NAMES = ["apple", "banana", "orange"]

class SmoothDetector:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Error: '{MODEL_PATH}' was not found.")

        print("🔄 Loading ONNX model into OpenCV DNN...")
        self.net = cv2.dnn.readNetFromONNX(MODEL_PATH)

        print("🔄 Initializing Camera Module 3 widescreen feed...")
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = (1280, 720)
        self.picam2.preview_configuration.main.format = "BGR888"  
        self.picam2.preview_configuration.align()
        self.picam2.configure("preview")
        self.picam2.start()
        
        try:
            self.picam2.set_controls({"AfMode": 2})  # Continuous Autofocus
        except Exception:
            pass

        # Threading shared variables
        self.frame = None
        self.predictions = []
        self.running = True
        self.fps = 0.0

        # Start the background AI worker thread
        self.ai_thread = threading.Thread(target=self.ai_inference_loop, daemon=True)

    def start(self):
        self.ai_thread.start()
        print("🚀 Smooth detection window active. Press 'q' to quit.")
        
        cv2.namedWindow("Object Detection View", cv2.WINDOW_NORMAL)

        while self.running:
            start_time = time.time()
            
            # Grab the latest frame available from Camera Module 3
            img = self.picam2.capture_array()
            self.frame = img.copy()  # Update the shared frame for the AI thread
            
            orig_h, orig_w, _ = img.shape

            # Draw the latest predictions available without waiting for the AI calculation
            # This keeps the camera rendering at full native hardware speed!
            local_preds = self.predictions.copy()
            for pred in local_preds:
                bounding_box = pred[0:4]
                class_scores = pred[4:]
                
                if len(class_scores) == 0:
                    continue
                    
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]

                if confidence >= CONF_THRESHOLD and class_id < len(CLASS_NAMES):
                    label_name = CLASS_NAMES[class_id]
                    
                    cx, cy, bw, bh = bounding_box
                    x1 = int((cx - bw / 2) * (orig_w / 640.0))
                    y1 = int((cy - bh / 2) * (orig_h / 640.0))
                    x2 = int((cx + bw / 2) * (orig_w / 640.0))
                    y2 = int((cy + bh / 2) * (orig_h / 640.0))

                    # Clean green bounding box and text overlay
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text_str = f"{label_name.upper()} {confidence*100:.0f}%"
                    cv2.putText(img, text_str, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display Smooth FPS Counter
            cv2.putText(img, f"Camera FPS: {self.fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Object Detection View", img)

            if cv2.waitKey(1) == ord("q"):
                self.running = False
                break

            # Calculate actual rendering display speed
            elapsed = time.time() - start_time
            self.fps = 1 / elapsed if elapsed > 0 else 30.0

        # Cleanup
        self.picam2.stop()
        cv2.destroyAllWindows()

    def ai_inference_loop(self):
        """ Runs completely on a separate CPU core to handle the heavy AI math """
        while self.running:
            if self.frame is None:
                time.sleep(0.01)
                continue

            # Work on a static snapshot copy of the current frame
            ai_frame = self.frame.copy()

            # Run YOLO parsing calculations asynchronously
            blob = cv2.dnn.blobFromImage(ai_frame, 1.0/255.0, (640, 640), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward()
            
            preds = outputs[0] if len(outputs.shape) == 3 else outputs
            if preds.shape[0] < preds.shape[1]:
                preds = np.transpose(preds, (1, 0))

            # Push new boxes over to the main display thread
            self.predictions = preds
            
            # Tiny pause to let the CPU breathe
            time.sleep(0.01)

if __name__ == "__main__":
    detector = SmoothDetector()
    detector.start()
