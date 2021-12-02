import PIL
import cv2
import numpy
from PIL import Image


class Executer:
    def __init__(self, tcpServer):
        self.andRaspTCP = tcpServer

    def print_result(self, age, gender, time):
        #basewidth = 200
        #wpercent = (basewidth/float(img.size[0]))
        #hsize = int((float(img.size[1]) * float(wpercent)))
        #img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        #strimg = numpy.array(img)
        #strimg.tostring()

        noti = time+" "+age+" "+gender + "\n"

        self.startCommand(noti)

    def startCommand(self, noti):
        self.andRaspTCP.sendAll(noti)
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        #result, imgencode = cv2.imencode('.png',encode_param)
        #img = numpy.array(imgencode)
        #self.andRaspTCP.send(str(len(img)).encode().ljust(16))
        #self.andRaspTCP.send(img)