import cv2
import numpy as np
import PIL
from PIL import Image

class Notification:
    def print_result(self, img, age, gender):
        basewidth = 200
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        printScreen = np.array(img)
        cv2.imshow("ScreenShot", cv2.cvtColor(printScreen, cv2.COLOR_BGR2RGB))
        print(age, gender)
