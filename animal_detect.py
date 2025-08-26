
# animal_detector_gui.py

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# List of carnivorous animals
carnivores = ['lion', 'tiger', 'leopard', 'cheetah', 'wolf', 'hyena', 'crocodile', 'sharks', 'alligators', 'bear']

class AnimalDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Animal Detection System")
        self.model = YOLO("yolov8n.pt")
        self.image_panel = None
        self.video_path = None
        self.frame = None

        self.coco_to_carnivore = {
            "cat": "tiger/lion/leopard/cheetah",
            "hyena": "hyena",
            "dog": "wolf",
            "bear": "bear",
            "crocodiles": "crocodiles",
            "alligator": "alligator",
            "eagle": "eagle",
            "vultures": "vultures",
            "sharks": "sharks",
        }

        self.label = tk.Label(root, text="Upload an Image or Video")
        self.label.pack(pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Upload Image", command=self.upload_image).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Upload Video", command=self.upload_video).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Exit", command=root.quit).pack(side=tk.LEFT, padx=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            image = cv2.imread(file_path)
            annotated_image, carnivore_count = self.detect_animals(image)
            self.show_image(annotated_image)
            messagebox.showinfo("Detection Result", f"Carnivorous animals detected: {carnivore_count}")

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if self.video_path:
            self.process_video()

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open video file.")
            return

        carnivore_count_total = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, carnivore_count = self.detect_animals(frame)
            carnivore_count_total += carnivore_count

            cv2.imshow("Video - Press Q to exit", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Video Result", f"Total carnivorous animals detected: {carnivore_count_total}")

    # def detect_animals(self, image):
    #     results = model(image)[0]
    #     carnivore_count = 0

    #     class_map = {
    #         'cat':'non-carnivore',
    #         'dog':'non-carnivore',
    #         'cow':'non-carnivore',
    #         'horse':'non-carnivore',
    #         'sheep':'non-carnivore',
    #         'elephant':'non-carnivore',
    #         'buffalo':'non-carnivore',
    #         'zebra':'non-carnivore',
    #         'deer':'non-carnivore',
    #         'lion':'carnivore',
    #         'tiger':'carnivore',
    #         'cheetah':'carnivore',
    #         'leopard':'carnivore',
    #         'hyena':'carnivore',
    #         'crocodile':'carnivore',
    #         'alligator':'carnivore',
    #         'wolf':'carnivore',
    #         'sharks':'carnivore',
    #         'eagle':'carnivore',
    #         'bear':'carnivore',
    #         'vultures':'carnivore',
    #         'piranhas':'carnivore',
    #     }

    #     for box in results.boxes:
    #         cls_id = int(box.cls[0])
    #         label = model.names[cls_id]
    #         conf = box.conf[0]
    #         x1, y1, x2, y2 = map(int, box.xyxy[0])

    #         is_carnivore = class_map.get(label.lower(), 'unknown') == 'carnivore'
    #         color = (0, 0, 255) if is_carnivore else (0, 255, 0)
    #         if is_carnivore:
    #             carnivore_count += 1

    #         cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    #         cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #     return image, carnivore_count

    # Mapping from YOLO label to our known carnivores (based on closest match)
    # coco_to_carnivore = {
    #     "cat": "tiger",     # YOLO may detect tiger/lion as 'cat'
    #     "dog": "wolf",      # Wild dogs, wolves
    #     "bear": "bear",     # Present in COCO
    # }

    def detect_animals(self, image):
        results = model(image)[0]
        carnivore_count = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Map label to possible carnivore
            mapped_label = self.coco_to_carnivore.get(label.lower())
            is_carnivore = mapped_label is not None

            # Highlight color
            color = (0, 0, 255) if is_carnivore else (0, 255, 0)
            if is_carnivore:
                carnivore_count += 1
                display_label = f"{mapped_label} ({label})"
            else:
                display_label = label

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f'{display_label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image, carnivore_count


    def show_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)

        if self.image_panel is None:
            self.image_panel = tk.Label(image=image_tk)
            self.image_panel.image = image_tk
            self.image_panel.pack(pady=10)
        else:
            self.image_panel.configure(image=image_tk)
            self.image_panel.image = image_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = AnimalDetectorApp(root)
    root.mainloop()
