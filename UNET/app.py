from tkinter import *
import tkinter as tk
from tkinter import ttk
import pyautogui
import pyscreenshot as ImageGrab
from PIL import ImageGrab
import datetime
import time
from predict_GUI_class import Predictor

class Application():
    def __init__(self, master):
        self.master = master
        self.rect = None
        self.x = self.y = 0
        self.start_x = None
        self.start_y = None
        self.curX = None
        self.curY = None

        master.geometry('300x200')
        master.resizable(False, False)
        master.title('Classify Me')
        # root.configure(background = 'red')
        # # root.attributes("-transparentcolor","red")
        
        T = Text(root, height = 5, width = 52)
        
        label = Label(root, text = "Welcome to Classify Me")
        label.config(font =("Courier", 14))
        
        text = """Click the 'Classify' button to take a screenshot of an aerial image you wish to segment."""
        label.pack()
        T.pack()
        T.insert(tk.END, text)

        capture_button = ttk.Button(
            master,
            text='Classify',
            command=self.createScreenCanvas
        )

        capture_button.pack(
            ipadx=10,
            ipady=10,
            expand=True
        )

        exit_button = ttk.Button(
            master,
            text='Exit',
            command=lambda: root.quit()
        )

        exit_button.pack(
            ipadx=5,
            ipady=5,
            expand=True
        )

        self.master_screen = Toplevel(root)
        self.master_screen.withdraw()
        # self.master_screen.attributes("-transparent", "blue")
        self.picture_frame = Frame(self.master_screen, background = "blue")
        self.picture_frame.pack(fill=BOTH, expand=YES)
        self.predictor = Predictor()

    def takeBoundedScreenShot(self, x1, y1, x2, y2):
        print(x1,y1,x2,y2)
        im = pyautogui.screenshot(region=(x1, y1, x2, y2))
        x = datetime.datetime.now()
        fileName = "screenshots/" + x.strftime("%f") + ".png"
        im.save(fileName)
        return fileName

    def createScreenCanvas(self):
        self.master_screen.deiconify()
        root.withdraw()

        self.screenCanvas = Canvas(self.picture_frame, cursor="cross", bg="grey11")
        self.screenCanvas.pack(fill=BOTH, expand=YES)

        self.screenCanvas.bind("<ButtonPress-1>", self.on_button_press)
        self.screenCanvas.bind("<B1-Motion>", self.on_move_press)
        self.screenCanvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.master_screen.attributes('-fullscreen', True)
        self.master_screen.attributes('-alpha', .1)
        self.master_screen.lift()
        self.master_screen.attributes("-topmost", True)

    def on_button_release(self, event):
        self.recPosition()
        filename = None
        if self.start_x <= self.curX and self.start_y <= self.curY:
            print("right down")
            filename = self.takeBoundedScreenShot(self.start_x+450, self.start_y+300, self.curX, self.curY+300)

        elif self.start_x >= self.curX and self.start_y <= self.curY:
            print("left down")
                        # self.takeBoundedScreenShot(self.curX, self.start_y, self.start_x - self.curX, self.curY - self.start_y)

            filename = self.takeBoundedScreenShot(self.curX, self.start_y, self.start_x - self.curX, self.curY - self.start_y)

        elif self.start_x <= self.curX and self.start_y >= self.curY:
            print("right up")
            filename = self.takeBoundedScreenShot(self.start_x, self.curY, self.curX - self.start_x, self.start_y - self.curY)

        elif self.start_x >= self.curX and self.start_y >= self.curY:
            print("left up")
            filename = self.takeBoundedScreenShot(self.curX, self.curY, self.start_x - self.curX, self.start_y - self.curY)

        self.exitScreenshotMode()
        out = self.predict(filename)
        print(out)
        return event

    def exitScreenshotMode(self):
        print("Screenshot mode exited")
        self.screenCanvas.destroy()
        self.master_screen.withdraw()
        root.deiconify()
        print("here")

    def exit_application(self):
        print("Application exit")
        root.quit()

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = self.screenCanvas.canvasx(event.x)
        self.start_y = self.screenCanvas.canvasy(event.y)

        self.rect = self.screenCanvas.create_rectangle(self.x, self.y, 1, 1, outline='red', width=3)

    def on_move_press(self, event):
        self.curX, self.curY = (event.x, event.y)
        # expand rectangle as you drag the mouse
        self.screenCanvas.coords(self.rect, self.start_x, self.start_y, self.curX, self.curY)

    def recPosition(self):
        print(self.start_x)
        print(self.start_y)
        print(self.curX)
        print(self.curY)

    def predict(self, filename):
        print(filename)
        self.predictor.predict(filename)
        return 1

if __name__ == '__main__':
    root = Tk()
    app = Application(root)
    root.mainloop()