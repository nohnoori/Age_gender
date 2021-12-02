import sys
import time
import cv2
from PIL import ImageGrab
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
import numpy as np
from PyQt5.QtCore import QSize, QTimer

from wide_resnet import WideResNet
from PyQt5.QtGui import QIcon, QImage, QPixmap
from executer import Executer
from collections import deque
import queue

import tcpServer
import wide_resnet
import tcpServerThread
from tcpServerThread import *
from tcpServer import *


resAges = np.zeros(21)
resGenders = np.zeros(21)
p_x = np.zeros(21)
p_y = np.zeros(21)
p_w = np.zeros(21)
p_h = np.zeros(21)
fin = np.zeros(21)
cnt = np.zeros(21)

# [1. 2. 3. 4. 5. 6. 7. 8. 9. 10.]
dq = deque()
for i in range(1, 21, 1):
    dq.append(i)


#####################################################################
class Main_App(QtWidgets.QWidget):

    def __init__(self, depth=16, width=8, face_size=32, fps=30, parent=None):
        super().__init__(parent)
        self.case_path = ".//pretrained_models//haarcascade_frontalface_default.xml"

        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        self.classifier = cv2.CascadeClassifier(self.case_path)
        self.image = QtGui.QImage()
        self.mode = ""

        try:
            self.model.load_weights('.//pretrained_models//weights.30-4.16.hdf5')
        except AssertionError:
            self.model = None

        self.fps = fps
        self.video_size = QSize(1680, 1030)

        self.gray_image = None
        self.detect_faces = []

        # Setup the UI
        self.setWindowTitle('CCTV')
        main_layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight, parent=self)
        form_lbx = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom, parent=self)
        form_lbx2 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom, parent=self)
        self.setLayout(form_lbx)

        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setFixedSize(self.video_size)

        gb_w = QtWidgets.QGroupBox(self)
        gb_w.setGeometry(6, 6, 1780, 968)
        gb_w.setTitle('video')
        form_lbx.addWidget(gb_w)
        lbx = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight, parent=self)
        gb_w.setLayout(lbx)
        lbx.addWidget(self.image_label)
        # self.run_button = QtWidgets.QPushButton('Stop')
        # form_lbx.addWidget(self.run_button)
        # self.run_button.clicked.connect(self.record_video.start_recording)

        self.mode_label = QtWidgets.QLabel("", self)
        self.mode_label.setAlignment(QtCore.Qt.AlignCenter)
        font = self.mode_label.font()
        font.setPointSize(20)
        font.setFamily('Times New Roman')
        self.mode_label.setFont(font)

        # gb_m = QtWidgets.QGroupBox(self)
        # gb_m.setFixedHeight(300)
        # gb_m.setFixedWidth(300)
        # gb_m.setTitle('mode')
        # form_lbx2.addWidget(gb_m)
        # lbx1 = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom, parent=self)
        # gb_m.setLayout(lbx1)
        # lbx1.addWidget(self.mode_label)

        self.tb = QtWidgets.QTextBrowser()
        self.tb.setFixedSize(300, 500)
        form_lbx2.addWidget(self.tb)

        lbtn = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton('Start')
        self.stop_button = QtWidgets.QPushButton('Stop')

        self.run_button.setCheckable(True)
        self.run_button.toggled.connect(self.run_camera)
        self.stop_button.setCheckable(True)
        self.stop_button.toggled.connect(self.stop_camera)

        lbtn.addWidget(self.run_button)
        lbtn.addWidget(self.stop_button)
        form_lbx2.addLayout(lbtn)
        form_lbx2.addStretch(1)

        main_layout.addLayout(form_lbx)
        main_layout.addLayout(form_lbx2)
        self.setLayout(main_layout)

        # # Setup the camera
        # self.capture = cv2.VideoCapture(cv2.CAP_DSHOW)
        # self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        # self.timer.start(int(900 / self.fps))

    # up:stop / down:run
    def run_camera(self):
        # Setup the camera
        self.capture = cv2.VideoCapture(cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.timer.start(int(900 / self.fps))

    def stop_camera(self):
        self.timer.stop()

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=20, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w, h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w - 1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h - 1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def same_person(self, cropped, width, height):
        (x, y, w, h) = cropped

        r1 = (x + w) * 0.2
        r2 = (y + h) * 0.2
        r_x, r_y = x - r1, y - r2
        r_w, r_h = w + 2 * r1, h + 2 * r2

        for j in range(1, 11, 1):
            if len(dq) != 10:
                break
            else:
                loc = dq.popleft()
                p_x[loc], p_y[loc], p_w[loc], p_h[loc] = r_x, r_y, r_w, r_h
                return loc

        if x + w == width or x == 0 or y == 0 or y + h == height:
            for j in range(1, 11, 1):
                if (p_x[j] <= x) and (x + w <= p_x[j] + p_w[j]):
                    if (p_y[j] <= y) and (y + h <= p_y[j] + p_h[j]):
                        if 0 < cnt[j] < 50:
                            return j * 100 + cnt[j]
                        p_x[j], p_y[j], p_w[j], p_h[j] = 0, 0, 0, 0
                        fin[j], cnt[j] = 0, 0
                        dq.append(j)
                        break
                else:
                    if j == 10:
                        loc = dq.popleft()
                        p_x[loc], p_y[loc], p_w[loc], p_h[loc] = r_x, r_y, r_w, r_h
                        return loc

        for j in range(1, 11, 1):
            if (p_x[j] <= x) and (x + w <= p_x[j] + p_w[j]):
                if (p_y[j] <= y) and (y + h <= p_y[j] + p_h[j]):
                    p_x[j], p_y[j], p_w[j], p_h[j] = r_x, r_y, r_w, r_h
                    cnt[j] = cnt[j] + 1
                    if cnt[j] == 50:
                        fin[j] = j
                        return j * 100 + 50
                    if j in fin:
                        return -1
                    else:
                        return j
            else:
                if j == 10:
                    loc = dq.popleft()
                    p_x[loc], p_y[loc], p_w[loc], p_h[loc] = r_x, r_y, r_w, r_h
                    return loc

        # #self.mode = data
        # if self.mode == 'F':
        #     self.mode_label.setText('now\nFemale\nwatching')
        # elif self.mode == 'M':
        #     self.mode_label.setText('now\nmale\nwatching')
        # elif self.mode == 'A':
        #     self.mode_label.setText('now\nall\nwatching')
        # elif self.mode == 'X':
        #     self.mode_label.setText('no watching')

    # def final_result(self, final_age, final_gender, width, height, currTime):
    #     # img = ImageGrab.grab(bbox=(0, 0, width, height))
    #
    #     self.tb.append('[ %s ] send %s %d ' % (currTime, final_gender, final_age))
    #     print(final_gender, final_age)

        # if final_gender == 'F' and self.mode == 'F':
        # Executer(andRaspTCP).print_result(final_age, final_gender, currTime)
        # elif final_gender == 'M' and self.mode == 'M':
        #     Executer(andRaspTCP).print_result(img, final_age, final_gender, currTime)
        # elif self.mode == 'A':
        #     Executer(andRaspTCP).print_result(img, final_age, final_gender, currTime)


    def display_video_stream(self):
        _, frame = self.capture.read(0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        height = frame.shape[0]
        width = frame.shape[1]
        currTime = "%s" % time.strftime("%Y/%m/%d %H:%M:%S")

        face_cascade = cv2.CascadeClassifier(self.case_path)
        self.gray_image = gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.detect_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=7,
            minSize=(self.face_size, self.face_size)
        )

        face_imgs = np.empty((len(self.detect_faces), self.face_size, self.face_size, 3))
        pid = np.empty(len(face_imgs))

        # placeholder for cropped faces
        for i, face in enumerate(self.detect_faces):
            face_img, cropped = self.crop_face(frame, face, margin=20, size=self.face_size)
            (x, y, w, h) = cropped

            p = self.same_person(cropped, width, height)
            face_imgs[i, :, :, :] = face_img
            pid[i] = p

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)

        if len(face_imgs) > 0:
            # predict ages and genders of the detected faces
            results = self.model.predict(face_imgs)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

        for i, face in enumerate(self.detect_faces):
            idx = int(pid[i])

            # draw results
            label = "{}, {}, {}".format(idx, int(predicted_ages[i]),
                                        'F' if predicted_genders[i][0] > 0.5 else 'M')
            self.draw_label(frame, (face[0], face[1]), label)

            if idx != -1:
                if idx > 100 and idx % 100 != 0:
                    pi = int(idx / 100)
                    n = int(idx % 100)
                    final_age = int(resAges[pi] / n)
                    final_gender = 'F' if resGenders[pi] > n / 2 else 'M'

                    resAges[pi], resGenders[pi] = 0, 0
                    # self.final_result(final_age, final_gender, width, height, currTime)
                    self.tb.append('[ %s ] send %s %d ' % (currTime, final_gender, final_age))
                    print(final_gender, final_age)
                    noti = noti = time+" "+final_age+" "+final_gender + "\n"
                    #Executer(andRaspTCP).print_result(final_age, final_gender, currTime)
                else:
                    resAges[idx] = resAges[idx] + int(predicted_ages[i])
                    resGenders[idx] = resGenders[idx] + 1 if predicted_genders[i][0] > 0.5 else 0

        cv2.putText(frame, currTime, (int(width * 2 / 3), height), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

        # Display the image in the image area
        image = QImage(frame, frame.shape[1], frame.shape[0],
                       frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))


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
        print(noti)

        self.startCommand(noti)

    def startCommand(self, noti):
        self.andRaspTCP.sendAll(noti)
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        #result, imgencode = cv2.imencode('.png',encode_param)
        #img = numpy.array(imgencode)
        #self.andRaspTCP.send(str(len(img)).encode().ljust(16))
        #self.andRaspTCP.send(img)


if __name__ == '__main__':
    commandQueue = queue.Queue()

    # init module
    andRaspTCP = tcpServer.TCPServer(commandQueue, "", 8556)
    andRaspTCP.start()

    # set module to executer
    # commandExecuter = executer.Executer(andRaspTCP)

    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QMainWindow()
    main_widget = Main_App()
    main_window.setCentralWidget(main_widget)
    main_window.setWindowTitle('CCTV')
    main_window.setWindowIcon(QIcon('icon.png'))
    main_window.showMaximized()
    sys.exit(app.exec_())