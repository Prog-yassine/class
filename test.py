import os
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import cv2
import json
import qrcode
import threading
import time
import numpy as np

WINDOW_TITLE = "Fruit Kiosk - AI Detection"
CAMERA_INDEX = 0  # Default camera index for Raspberry Pi / Linux
DETECTION_INTERVAL = 0.3

# The 3 items for your kiosk
ITEM_INFO = {
    "apple": {
        "label": "Apple",
        "price": 1.20,
        "calories": 52,
        "origin": "France",
        "description": "La pomme est un fruit croquant et juteux, riche en fibres alimentaires qui aident à la digestion. Elle est également une excellente source de vitamine C.",
        "color": "#d32f2f"
    },
    "banana": {
        "label": "Banana",
        "price": 0.80,
        "calories": 89,
        "origin": "Équateur",
        "description": "La banane est un fruit tropical doux et énergétique, parfaite pour un en-cas rapide. Elle fournit une bonne quantité de potassium.",
        "color": "#fbc02d"
    },
    "orange": {
        "label": "Orange",
        "price": 1.00,
        "calories": 47,
        "origin": "Espagne",
        "description": "L'orange est un agrume juteux, exceptionnellement riche en vitamine C pour faire le plein d'énergie au quotidien.",
        "color": "#f57c00"
    },
}

# ⚠️ CRITICAL: Adjust the sequence below to match the EXACT class order of your trained dataset
CLASS_NAMES = ["apple", "banana", "orange"]

class FruitKioskApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="#ffffff")

        self.running = True
        self.selected_item = None
        self.detail_view = False
        self.detected_items = {}
        self.item_images = {}

        # Load ONNX model via OpenCV DNN (32-bit Pi compatible)
        model_path = "yolov8n.onnx"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Please place your trained '{model_path}' file in the project folder.")
            
        self.net = cv2.dnn.readNetFromONNX(model_path)

        # Linux Camera Init
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Ensure your camera is connected.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Layout Arrangement
        self.main_frame = tk.Frame(self.root, bg="#ffffff")
        self.main_frame.pack(fill="both", expand=True)

        # LEFT Panel (QR and Details)
        self.left_frame = tk.Frame(self.main_frame, bg="#f8fafc")
        self.left_frame.pack(side="left", fill="both", expand=True)

        # RIGHT Panel (List of items)
        self.right_frame = tk.Frame(self.main_frame, bg="#e2e8f0", width=340)
        self.right_frame.pack(side="right", fill="y")
        self.right_frame.pack_propagate(False)

        # Center panel card contents
        self.info_frame = tk.Frame(self.left_frame, bg="#ffffff")
        self.info_frame.pack(fill="both", expand=True, padx=24, pady=24)

        self.top_bar = tk.Frame(self.info_frame, bg="#f1f5f9")
        self.top_bar.pack(fill="x", pady=(10, 0))

        self.back_button = tk.Button(
            self.top_bar,
            text="← Retour",
            command=self.clear_selection,
            bg="#e2e8f0",
            fg="#1f2937",
            relief="flat",
            bd=0,
            activebackground="#cbd5e1",
            activeforeground="#1f2937",
            font=("Helvetica", 12, "bold"),
            padx=14,
            pady=8
        )
        self.back_button.pack(side="left", padx=(0, 10))

        self.header_title = tk.Label(
            self.top_bar,
            text="Scan to Pay",
            font=("Helvetica", 26, "bold"),
            fg="#1f2937",
            bg="#f1f5f9"
        )
        self.header_title.pack(side="left")

        self.total_label = tk.Label(
            self.info_frame,
            text="Total: $0.00",
            font=("Helvetica", 20, "bold"),
            fg="#0369a1",
            bg="#ffffff"
        )
        self.total_label.pack(pady=(10, 10))

        self.center_card = tk.Frame(
            self.info_frame,
            bg="#f8fafc",
            bd=0,
            highlightthickness=1,
            highlightbackground="#cbd5e1"
        )
        self.center_card.pack(fill="both", expand=True, pady=(0, 10))

        self.center_title = tk.Label(
            self.center_card,
            text="Scan to Pay",
            font=("Helvetica", 32, "bold"),
            fg="#1f2937",
            bg="#f8fafc"
        )
        self.center_title.pack(pady=(30, 10))

        self.center_subtitle = tk.Label(
            self.center_card,
            text="No item selected",
            font=("Helvetica", 17),
            fg="#64748b",
            bg="#f8fafc"
        )
        self.center_subtitle.pack()

        self.detail_panel = tk.Frame(self.center_card, bg="#f8fafc")
        self.detail_panel.pack(fill="x", padx=40, pady=20)

        self.item_image_label = tk.Label(self.detail_panel, bg="#f8fafc")
        self.item_image_label.pack(side="left", padx=(0, 30), pady=10)

        self.item_info_frame = tk.Frame(self.detail_panel, bg="#f8fafc")
        self.item_info_frame.pack(side="left", fill="both", expand=True, pady=10)

        self.price_label = tk.Label(
            self.item_info_frame,
            text="",
            font=("Helvetica", 20, "bold"),
            fg="#0369a1",
            bg="#f8fafc"
        )
        self.price_label.pack(anchor="nw")

        self.calories_label = tk.Label(
            self.item_info_frame,
            text="",
            font=("Helvetica", 16),
            fg="#64748b",
            bg="#f8fafc"
        )
        self.calories_label.pack(anchor="nw", pady=(10, 0))

        self.origin_label = tk.Label(
            self.item_info_frame,
            text="",
            font=("Helvetica", 16),
            fg="#64748b",
            bg="#f8fafc"
        )
        self.origin_label.pack(anchor="nw", pady=(10, 0))

        self.description_title = tk.Label(
            self.item_info_frame,
            text="Description",
            font=("Helvetica", 16, "bold"),
            fg="#1f2937",
            bg="#f8fafc"
        )
        self.description_title.pack(anchor="nw", pady=(20, 6))

        self.detail_label = tk.Label(
            self.center_card,
            text="",
            font=("Helvetica", 15),
            fg="#374151",
            bg="#f8fafc",
            justify="left",
            wraplength=960
        )
        self.detail_label.pack(padx=40, pady=(0, 30))

        self.qr_label = tk.Label(self.center_card, bg="#f8fafc")
        self.qr_label.pack(pady=24)

        # RIGHT PANEL ELEMENTS
        tk.Label(
            self.right_frame,
            text="Articles détectés",
            font=("Helvetica", 22, "bold"),
            fg="#1f2937",
            bg="#e2e8f0"
        ).pack(pady=18)

        self.items_canvas = tk.Canvas(
            self.right_frame,
            bg="#e2e8f0",
            highlightthickness=0
        )
        self.items_canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)

        self.scrollbar = tk.Scrollbar(
            self.right_frame,
            orient="vertical",
            command=self.items_canvas.yview,
            bg="#e2e8f0",
            troughcolor="#f8fafc"
        )
        self.scrollbar.pack(side="right", fill="y", pady=10)

        self.items_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.items_container = tk.Frame(self.items_canvas, bg="#e2e8f0")
        self.canvas_window = self.items_canvas.create_window((0, 0), window=self.items_container, anchor="nw", width=320)
        
        self.items_container.bind(
            "<Configure>",
            lambda e: self.items_canvas.configure(scrollregion=self.items_canvas.bbox("all"))
        )
        self.items_canvas.bind(
            "<Configure>",
            lambda e: self.items_canvas.itemconfig(self.canvas_window, width=e.width)
        )

        self.footer_frame = tk.Frame(self.right_frame, bg="#e2e8f0")
        self.footer_frame.pack(fill="x", padx=10, pady=10)

        tk.Button(
            self.footer_frame,
            text="Clear Selection",
            command=self.clear_selection,
            bg="#f1f5f9",
            fg="#1f2937",
            activebackground="#cbd5e1",
            relief="flat",
            bd=0,
            padx=12,
            pady=10
        ).pack(fill="x", pady=6)

        tk.Button(
            self.footer_frame,
            text="Exit",
            command=self.close,
            bg="#ef4444",
            fg="white",
            activebackground="#dc2626",
            relief="flat",
            bd=0,
            padx=12,
            pady=10
        ).pack(fill="x", pady=6)

        self.root.bind("<Escape>", lambda e: self.close())
        self.load_item_images()

        # Offline AI Inference thread initialization
        self.detect_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detect_thread.start()

        self.update_right_panel()
        self.update_center_panel()

    def close(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

    def load_item_images(self):
        for key, info in ITEM_INFO.items():
            image = self.build_placeholder_image(info["label"], info["color"])
            self.item_images[key] = ImageTk.PhotoImage(image)

    def build_placeholder_image(self, label, color):
        image = Image.new("RGB", (360, 280), color="#141414")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 340, 260), fill=color)
        draw.text((30, 30), label, fill="white")
        return image

    def clear_selection(self):
        self.selected_item = None
        self.detail_view = False
        self.update_center_panel()

    def select_item(self, item_key):
        self.selected_item = item_key
        self.detail_view = True
        self.update_center_panel()

    def build_qr_payload(self):
        items = []
        total = 0.0
        for key, data in self.detected_items.items():
            items.append({
                "id": key,
                "name": data["display_name"],
                "qty": data["qty"],
                "price": data["price"]
            })
            total += data["qty"] * data["price"]

        return json.dumps({"items": items, "total": round(total, 2)})

    def generate_qr_image(self, payload_text):
        qr = qrcode.QRCode(version=1, box_size=8, border=2)
        qr.add_data(payload_text)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        img = img.resize((300, 300))
        return ImageTk.PhotoImage(img)

    def update_center_panel(self):
        total = sum(d["qty"] * d["price"] for d in self.detected_items.values())
        self.total_label.config(text=f"Total: ${total:.2f}")

        show_back = self.detail_view and self.selected_item in self.detected_items
        if show_back:
            self.back_button.pack(side="left", padx=(0, 10))
        else:
            self.back_button.pack_forget()

        if self.detail_view and self.selected_item in self.detected_items:
            item = self.detected_items[self.selected_item]
            self.header_title.config(text=item["display_name"])
            self.center_title.config(text=item["display_name"])
            self.center_subtitle.config(text=f"Prix: ${item['price']:.2f}  |  Calories: {item['calories']} kcal")
            self.item_image_label.config(image=self.item_images[self.selected_item])
            self.item_image_label.image = self.item_images[self.selected_item]
            self.price_label.config(text=f"Prix : ${item['price']:.2f}")
            self.calories_label.config(text=f"Calories : {item['calories']} kcal")
            self.origin_label.config(text=f"Origine : {item['origin']}")
            self.detail_label.config(text=item['description'])
            self.detail_panel.pack(fill="x", padx=40, pady=20)
            self.qr_label.pack_forget()
        else:
            self.header_title.config(text="Scan to Pay")
            self.center_title.config(text="Scan to Pay")
            self.center_subtitle.config(text="Place an item in view to detect it and show its QR code.")
            self.item_image_label.config(image="")
            self.price_label.config(text="")
            self.calories_label.config(text="")
            self.origin_label.config(text="")
            self.detail_label.config(text="")
            self.detail_panel.pack_forget()
            self.qr_label.pack(pady=24)
            qr_img = self.generate_qr_image(self.build_qr_payload())
            self.qr_label.config(image=qr_img)
            self.qr_label.image = qr_img

    def update_right_panel(self):
        for w in self.items_container.winfo_children():
            w.destroy()

        if not self.detected_items:
            tk.Label(self.items_container, text="No fruits detected", fg="#64748b", bg="#e2e8f0").pack(anchor="nw")
        else:
            for key, data in self.detected_items.items():
                selected = self.selected_item == key and self.detail_view
                bg = "#0369a1" if selected else "#f1f5f9"
                fg = "white" if selected else "#1f2937"

                tk.Button(
                    self.items_container,
                    text=f"{data['display_name']}\nQty: {data['qty']}  |  ${data['price']:.2f}",
                    command=lambda k=key: self.select_item(k),
                    bg=bg,
                    fg=fg,
                    anchor="w",
                    justify="left",
                    relief="flat",
                    bd=0,
                    activebackground="#e2e8f0",
                    activeforeground="#1f2937",
                    padx=16,
                    pady=16,
                    font=("Helvetica", 12, "bold")
                ).pack(fill="x", pady=6)

        self.root.after(500, self.update_right_panel)

    def detection_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            self.detected_items = self.detect_fruits(frame)

            if self.selected_item not in self.detected_items:
                self.selected_item = None
                self.detail_view = False

            self.root.after(0, self.update_center_panel)
            time.sleep(DETECTION_INTERVAL)

    def detect_fruits(self, frame):
        detected = {}
        
        # OpenCV DNN Preprocessing (YOLOv8 size format, scaling normalization, BGR to RGB swap)
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Forward pass output
        outputs = self.net.forward()
        
        # Post-process dimensions array mapping
        predictions = np.squeeze(outputs)
        predictions = np.transpose(predictions, (1, 0))

        counts = {name: 0 for name in CLASS_NAMES}
        conf_threshold = 0.40

        for pred in predictions:
            scores = pred[4:] 
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence >= conf_threshold:
                if class_id < len(CLASS_NAMES):
                    label = CLASS_NAMES[class_id]
                    counts[label] += 1

        for label, count in counts.items():
            if count > 0 and label in ITEM_INFO:
                info = ITEM_INFO[label]
                detected[label] = {
                    "display_name": info["label"],
                    "price": info["price"],
                    "qty": count,
                    "calories": info["calories"],
                    "origin": info["origin"],
                    "description": info["description"]
                }

        return detected


if __name__ == "__main__":
    root = tk.Tk()
    app = FruitKioskApp(root)
    root.mainloop()
