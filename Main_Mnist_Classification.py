import os
import cv2
import PIL 
import numpy as np
import tkinter as tk
import tensorflow as tf
from tkinter import *
from PIL import ImageTk, Image, ImageDraw
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

window = tk.Tk()
window.title("Mnist Classification")
window.geometry('500x500')
window.config(background = "#152238")

Exit_img = PhotoImage(file = "./DL_Projects/Mnist Classification/Images/ExitButton.png")
Classify_img = PhotoImage(file = "./DL_Projects/Mnist Classification/Images/ClassifyButton.png")
Erase_img = PhotoImage(file = "./DL_Projects/Mnist Classification/Images/EraseButton.png")

model = tf.keras.models.load_model("./DL_Projects/Mnist Classification/mnist_classification_200epochs_augdata.model")
class_names = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]

image1 = PIL.Image.new("RGB", (500, 500), (0,0,0))
draw = ImageDraw.Draw(image1)


def Close():
    window.destroy()
    
def move(event):
    brush_size = 40
    x = event.x
    y = event.y
    canvas.create_oval((x - brush_size / 2,y - brush_size / 2, x + brush_size / 2,y + brush_size / 2), fill = 'black')
    draw.ellipse((x - brush_size / 2,y - brush_size / 2, x + brush_size / 2,y + brush_size / 2), fill = 'white')

def Clear_canvas():
    global image1, draw
    canvas.delete("all")
    image1 = PIL.Image.new("RGB", (500, 500), (0,0,0))
    draw = ImageDraw.Draw(image1)
    Prediction_label.destroy()

def classify():
    global Prediction_label
    img = np.array(image1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (28,28))
    img = img / 255.0
    img = img.reshape(28,28,1)
    prediction = model.predict(np.array([img]), verbose = 0)
    index = np.argmax(prediction)
    Prediction_label = Label(window, text = f'{class_names[index]}', font = ('Helvetica', 20, 'bold'), bg = "#aca79e")
    Prediction_label.pack()
    Prediction_label.place(x = 222, y = 326)


canvas = tk.Canvas(window,width = 350, height = 300, bg = "#aca79e", highlightbackground = "#aca79e", borderwidth = 0, cursor = "spraycan")
canvas.pack()
canvas.place(x = 70, y = 20)
canvas.bind('<B1-Motion>', move)

classify_button = Button(window, image = Classify_img, command=classify, borderwidth = 0, bg = "#152238", activebackground="#152238")
classify_button.pack()
classify_button.place(x = 198, y = 400)

clear_canvas_button = Button(window, image = Erase_img, command=Clear_canvas, borderwidth = 0, bg = "#152238", activebackground="#152238")
clear_canvas_button.pack()
clear_canvas_button.place(x = 442, y = 140)

exit_button = Button(master = window, image = Exit_img, command = Close , borderwidth = 0, bg = "#152238", activebackground="#152238")
exit_button.pack()
exit_button.place(x = 215, y = 460)


window.mainloop()