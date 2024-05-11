# gui to upload a photo and display it

from Image import hash                                                                      
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from time import sleep
class GUI:
    def __init__(self , adaline):
        self.root = tk.Tk()
        self.adaline = adaline
        self.label_image=None
        self.label_output=None

    def prepere_image(self , image):
        image = image.resize((100, 100))
        image = np.array(image).flatten()
        image = hash(image.tolist())
        return image

    
    def open_file(self):
        # if a photo and output are already displayed, remove them
        for widget in self.root.winfo_children():
            if self.label_image :
                self.label_image.destroy()
            if self.label_output:
                self.label_output.destroy()
        file_path = filedialog.askopenfilename()
        image = Image.open(file_path)
        image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        self.label_image = tk.Label(self.root, image=photo)
        self.label_image.image = photo
        self.label_image.place(x=350,y=200)
        result = self.adaline.predict(self.prepere_image(image))
       
        if result == 1:
            result = 'Apple'
        else:
            result = 'Banana'
        self.print_results(result)


    def print_results(self , results): # print the results in the GUI
        if results == 'Apple':
            self.label_output = tk.Label(self.root, text=results ,fg='red',font=('Helvetica',16))
        else:
            self.label_output = tk.Label(self.root, text=results ,fg='orange',font=('Helvetica', 16))
        self.label_output.place(x=470,y=510)

    def print_classification_rate(self , calssification_rate):
        label_classification_rate=tk.Label(self.root,text='Classification rate : ' + str(calssification_rate)+'%',font=('Helvetica', 14))
        label_classification_rate.pack()
    
    def main(self,plotData):
    # resize the window
        self.root.geometry('700x700')
        self.root.title("Image Classification")  # Set the title of the window
        button = tk.Button(self.root, text='Open File', command=self.open_file,bd=5,background='#0063B1',fg='white',font=('Helvetica', 10)) # button to open a file
        button.place(x=100,y=350)
        self.print_classification_rate(plotData)
        label_choose_from_computer=tk.Label(self.root,text='Choose image from your computer')
        label_choose_from_computer.place(x=50,y=390)
        # add a plot
        self.root.mainloop()


