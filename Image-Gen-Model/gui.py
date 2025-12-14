# gui.py
import tkinter as tk
from tkinter import filedialog
from PIL       import Image, ImageTk
from generator import ImageGenerator  # Your class from above
import threading

class ImageGeneratorApp:
    def __init__(self, master):
        self.master = master
        master.title("Stable Diffusion GUI")

        self.generator = ImageGenerator()

        self.prompt_entry = tk.Entry(master, width=50)
        self.prompt_entry.pack()

        self.generate_button = tk.Button(master, text="Generate", command=self.generate_image_thread)
        self.generate_button.pack()

        self.generate_button = tk.Button(master, text="Save Result", command=self.save_image)
        self.generate_button.pack()

        self.image_label = tk.Label(master)
        self.image_label.pack()
        self.photo = None

    def save_image(self):
        if self.photo is None:
            print("No image to save!")
            return
        # Ask user where to save
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if save_path:
            # Copy current image to chosen location
            img = Image.open(self.photo)
            img.save(save_path)
            print(f"Image saved to {save_path}")
                    
    def generate_image_thread(self):
        # Use a thread to avoid freezing GUI
        prompt = self.prompt_entry.get()
        threading.Thread(target=self.generate_image, args=(prompt,)).start()

    def generate_image(self, prompt):
        output_path = self.generator.generate_image(prompt)
        if output_path:
            img = Image.open(output_path)
            img = img.resize((512, 512))  # optional resize for GUI
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo  # keep reference
