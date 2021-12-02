"""
Face detection

encoding=utf-8
"""
from time import sleep

import queue
import cv2
import os
import time
import numpy as np
import argparse
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from receive_android import Mode
from send_android import Notification
from collections import deque
from PIL import ImageGrab
import tcpServer
import tcpServerThread
import executer
from executer import Executer

HIGH_VALUE = 10000
WIDTH = HIGH_VALUE
HEIGHT = HIGH_VALUE

# 한 번에 인식할 사람 수 만큼 배열 생성
resAges = np.zeros(31)
resGenders = np.zeros(31)
p_x = np.zeros(31)  # 0 - 10, 1부터 입력
p_y = np.zeros(31)
p_w = np.zeros(31)
p_h = np.zeros(31)
fin = np.zeros(31)  # 0 - 10, 1부터 입력
cnt = np.zeros(31)  # 0 - 10, 1부터 입력

# 빈공간 관리할 큐생성(10명)
# [1. 2. 3. 4. 5. 6. 7. 8. 9. 10.]
dq = deque()
for i in range(1,30, 1):
    dq.append(i)


class FaceCV(object):
    """
    Singleton class for face recongnition task
    """
    CASE_FPATH = ".\\pretrained_models\\haarcascade_frontalface_default.xml"
    #WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"

    def __new__(cls, weight_file=None, depth=16, width=8, face_size=32):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=32):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        #fpath = get_file('history_16_8.h5',
                       #  self.WRN_WEIGHTS_PATH,
                        # cache_subdir=model_dir)
        self.mode = Mode()
        self.model.load_weights('.//pretrained_models//weights.30-4.16.hdf5')

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
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
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
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

        # 동일인 인정 범위 계산
        r1 = (x + w) * 0.2
        r2 = (y + h) * 0.2
        r_x, r_y = x - r1, y - r2
        r_w, r_h = w + 2 * r1, h + 2 * r2

        # 인식된 사람이 없을 때
        for j in range(1, 31, 1):
            if len(dq) != 30:
                break
            else:
                loc = dq.popleft()
                p_x[loc], p_y[loc], p_w[loc], p_h[loc] = r_x, r_y, r_w, r_h
                return loc

        # 인식 마지노선 설정
        if x + w == width or x == 0 or y == 0 or y + h == height:
            for j in range(1, 31, 1):
                # 기존에 있던 사람이면 배열 제거 큐 추가
                if (p_x[j] <= x) and (x + w <= p_x[j] + p_w[j]):
                    if (p_y[j] <= y) and (y + h <= p_y[j] + p_h[j]):
                        if 0 < cnt[j] < 50:
                            return j * 100 + cnt[j]
                        p_x[j], p_y[j], p_w[j], p_h[j] = 0, 0, 0, 0
                        fin[j], cnt[j] = 0, 0
                        dq.append(j)
                        break
                # 새로운 사람이면 배열 추가
                else:
                    if j == 30:
                        loc = dq.popleft()
                        p_x[loc], p_y[loc], p_w[loc], p_h[loc] = r_x, r_y, r_w, r_h
                        return loc

        # 인식 된 사람이 있을 때
        # 인식 된 or 새로운 인물
        for j in range(1, 31, 1):
            # 반경 내에 있을 경우
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
            # 반경 외
            else:
                if j == 30:
                    loc = dq.popleft()
                    p_x[loc], p_y[loc], p_w[loc], p_h[loc] = r_x, r_y, r_w, r_h
                    return loc

    def final_result(self, final_age, final_gender, currTime):
        #img = ImageGrab.grab(bbox=(0, 0, width, height))

        # if final_gender == '여' and self.mode.to_real_demo() == 'F':
        Executer(andRaspTCP).print_result(str(final_age), final_gender, currTime)
        # elif final_gender == '남' and self.mode.to_real_demo() == 'M':
        #     Notification().print_result(img, final_age, final_gender)
        # elif self.mode.to_real_demo() == 'A':
        #     Notification().print_result(img, final_age, final_gender)

    def detect_face(self):
        face_cascade = cv2.CascadeClassifier(self.CASE_FPATH)

        # 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture(cv2.CAP_DSHOW)

        video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        video_capture.set(cv2.CAP_PROP_FPS, 30)

        # video_capture = cv2.VideoCapture(0)
        # #
        # # 프레임 설정
        # # 840 * 680
        # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 840)
        # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)
        # video_capture.set(cv2.CAP_PROP_FPS, 30)

        # infinite loop, break by key ESC
        while True:
            if not video_capture.isOpened():
                sleep(5)

            # Capture frame-by-frame
            ret, frame = video_capture.read()
            frame = cv2.flip(frame, 1)
            height = frame.shape[0]
            width = frame.shape[1]

            currTime = "%s" % time.strftime("%Y-%m-%d %H:%M:%S")

            # # fps 계산 및 확인 (fps = 31)
            # curTime = time.time()
            # sec = curTime - prevTime
            # prevTime = curTime
            #
            # fps = 1/sec
            # sumFps = sumFps + fps
            # avgFps = sumFps/cnt
            #
            # print("sec {}, Estimated fps {}, avgFps {}".format(sec, fps,avgFps))
            # cnt = cnt + 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(self.face_size, self.face_size)
            )

            face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
            pid = np.empty(len(face_imgs))

            # placeholder for cropped faces
            for i, face in enumerate(faces):
                face_img, cropped = self.crop_face(frame, face, margin=20, size=self.face_size)
                (x, y, w, h) = cropped

                p = self.same_person(cropped, width, height)
                face_imgs[i, :, :, :] = face_img
                pid[i] = p

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                # face_imgs[i, :, :, :] = face_img

            if len(face_imgs) > 0:
                # predict ages and genders of the detected faces
                results = self.model.predict(face_imgs)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()


            #########
            for i, face in enumerate(faces):
                idx = int(pid[i])

                # draw results
                label = "{}, {}, {}".format(idx, int(predicted_ages[i] - 5),
                                            'F' if predicted_genders[i][0] > 0.5 else 'M')
                self.draw_label(frame, (face[0], face[1]), label)

                if idx != -1:
                    if idx > 100 and idx % 100 != 0:
                        pi = int(idx/100)
                        n = int(idx % 100)
                        final_age = int(resAges[pi] / n) - 3
                        final_gender = 'F' if resGenders[pi] > n/2 else 'M'
                        if resAges[pi] == 0 and (pi in fin):
                            resAges[pi], resGenders[pi] = 0, 0
                            pass
                        else:
                            self.final_result(final_age, final_gender, currTime)
                    else:
                        resAges[idx] = resAges[idx] + int(predicted_ages[i])
                        resGenders[idx] = resGenders[idx] + 1 if predicted_genders[i][0] > 0.5 else 0

            cv2.putText(frame, currTime, (int(width*2/3), height), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

            cv2.resizeWindow('CCTV', 1920, 1080)
            cv2.moveWindow('CCTV', 0, 0)
            cv2.namedWindow('CCTV', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('CCTV', cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
            cv2.imshow('CCTV', frame)

            if cv2.waitKey(5) == 27:  # ESC key press
                break
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    depth = args.depth
    width = args.width

    face = FaceCV(depth=depth, width=width)

    face.detect_face()


if __name__ == "__main__":
    commandQueue = queue.Queue()

    # init module
    andRaspTCP = tcpServer.TCPServer(commandQueue, "", 8556)
    andRaspTCP.start()

    # set module to executer
    commandExecuter = executer.Executer(andRaspTCP)

    main()