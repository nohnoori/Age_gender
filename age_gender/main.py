import tcpServer
import executer
import queue
import cv2
import time


# make public queue
commandQueue = queue.Queue()

# init module
andRaspTCP = tcpServer.TCPServer(commandQueue, "", 55005)
andRaspTCP.start()

# set module to executer
commandExecuter = executer.Executer(andRaspTCP)

while True:
    try:
        command = commandQueue.get()
        commandExecuter.startCommand(command)
    except:
        pass

# 다시 이미지로 디코딩해서 화면에 출력. 그리고 종료

decimg = cv2.imdecode(img, 1)
cv2.imshow('CLIENT', decimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# while True:
# time.sleep(3)
# andRaspTCP.sendAll("321\n")