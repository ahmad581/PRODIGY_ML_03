# import os
import cv2
import joblib
import numpy as np
import dearpygui.dearpygui as dpg
import tkinter as tk
from tkinter import filedialog

# img_path = os.getenv(IMAGE_PATH)
model = joblib.load('dogs vs cats.pkl')
image = None
classification = {0: "Cat", 1: "Dog"}

dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()


def classify_img(sender, app_data):
    model_prediction = model.predict(image[:1])
    if model_prediction is not None:
        dpg.set_value("Result", f"This Is A: {classification[int(model_prediction)]}")
    else:
        dpg.set_value("Status", "Status: There Was An Error Predicting The Class Of The Image!!")


def load_img(sender, app_data):
    img_path = dpg.get_value("Image_Path")
    if img_path:
        try:
            global image
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))
            img = img.flatten()
            image = np.array([img])
            dpg.set_value("Status", f"Status: Loaded The Image successfully.")
        except Exception as e:
            dpg.set_value("Status", f"Status: Error: {str(e)}")
    else:
        dpg.set_value("Status", "Status: Please provide a valid file path.")


def browse_file():
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename()
    if image_path:
        dpg.set_value("Image_Path", image_path)
    else:
        dpg.set_value("Status", "Status: Please provide a valid file path.")


with dpg.window(label="Price Prediction", width=400, height=200):
    dpg.add_text("Enter The Path Of The Image:")
    dpg.add_input_text(label="Image Path", tag="Image_Path", width=400)
    dpg.add_button(label="Browse", callback=browse_file)
    dpg.add_button(label="Load Image", callback=load_img)
    dpg.add_text(label="", tag="Status")
    dpg.add_button(label="Classify The Image", callback=classify_img)
    dpg.add_text(label="", tag="Result")

dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
