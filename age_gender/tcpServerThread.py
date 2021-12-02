import socket, threading
import socket
import cv2
import numpy
from pyqt import Main_App


class TCPServerThread(threading.Thread):
    def __init__(self, commandQueue, tcpServerThreads, connections, connection, clientAddress):
        threading.Thread.__init__(self)

        self.commandQueue = commandQueue
        self.tcpServerThreads = tcpServerThreads
        self.connections = connections
        self.connection = connection
        self.clientAddress = clientAddress

    def run(self):
        try:
            while True:
                # Main_App().recv_mode(self.connection.recv(1024).decode())
                data = self.connection.recv(1024).decode()
                self.commandQueue.put(data)

        except:
            self.connections.remove(self.connection)
            self.tcpServerThreads.remove(self)
            print("1")
            exit(0)
        self.connections.remove(self.connection)
        self.tcpServerThreads.remove(self)
        print("2")

    def send(self, message):
        print('tcp server :: ', message)
        try:
            for i in range(len(self.connections)):
                self.connections[i].sendall(message.encode())
        except:
            pass

