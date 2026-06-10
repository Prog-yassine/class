import os
import time
import threading
import numpy as np
import cv2
from picamera2 import Picamera2

# Configurations
MODEL_PATH = "yolov8n.onnx"
CONF_THRESHOLD = 0.30  # Slightly lower threshold for faster processing passes
CLASS_NAMES = ["apple", "banana", "orange"]

class SmoothDetector:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Error: '{MODEL_PATH}' was not found.")

        print("🔄 Loading ONNX model into OpenCV DNN...")
        self.net = cv2.dnn.readNetFromONNX(MODEL_PATH)

        print("🔄 Initializing Camera Module 3 at optimized fast resolution...")
        self.picam2 = Picamera2()
        
        # 🔥 OPTIMIZATION: Dropping resolution to 640x480 massively speeds up CPU math
        self.picam2.preview_configuration.main.size = (640, 480)
        self.picam2.preview_configuration.main.format = "BGR888"  
        self.picam2.preview_configuration.align()
        self.picam2.configure("preview")
        self.picam2.start()
        
        try:
            self.picam2.set_controls({"AfMode": 2})
        except Exception:
            pass

        self.frame = None
        self.predictions = []
        self.running = True
        self.camera_fps = 0.0
        self.ai_fps = 0.0

        self.ai_thread = threading.Thread(target=self.ai_inference_loop, daemon=True)

    def start(self):
        self.ai_thread.start()
        cv2.namedWindow("Object Detection View", cv2.WINDOW_NORMAL)

        while self.running:
            start_time = time.time()
            
            img = self.picam2.capture_array()
            self.frame = img.copy()  
            
            orig_h, orig_w, _ = img.shape
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
                    # Mapping directly across matching 640 scale grids avoids heavy scaling math
                    x1 = int((cx - bw / 2) * (orig_w / 640.0))
                    y1 = int((cy - bh / 2) * (orig_h / 640.0))
                    x2 = int((cx + bw / 2) * (orig_w / 640.0))
                    y2 = int((cy + bh / 2) * (orig_h / 640.0))

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{label_name.upper()} {confidence*100:.0f}%", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display stats on screen
            cv2.putText(img, f"Video View Speed: {self.camera_fps:.1f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img, f"AI Compute Speed: {self.ai_fps:.1f} FPS", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.imshow("Object Detection View", img)

            if cv2.waitKey(1) == ord("q"):
                self.running = False
                break

            elapsed = time.time() - start_time
            self.camera_fps = 1 / elapsed if elapsed > 0 else 30.0

        self.picam2.stop()
        cv2.destroyAllWindows()

    def ai_inference_loop(self):
        while self.running:
            if self.frame is None:
                time.sleep(0.01)
                continue

            ai_start = time.time()
            ai_frame = self.frame.copy()

            # Pass smaller native layout sizes cleanly to the forward engine
            blob = cv2.dnn.blobFromImage(ai_frame, 1.0/255.0, (640, 640), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward()
            
            preds = outputs[0] if len(outputs.shape) == 3 else outputs
            if preds.shape[0] < preds.shape[1]:
                preds = np.transpose(preds, (1, 0))

            self.predictions = preds
            
            ai_elapsed = time.time() - ai_start
            self.ai_fps = 1 / ai_elapsed if ai_elapsed > 0 else 0.0
            
            time.sleep(0.001)

if __name__ == "__main__":
    detector = SmoothDetector()
    detector.start()
