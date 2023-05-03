from concurrent.futures import thread
import sys
from telnetlib import theNULL
import threading
from controller.cnn.CNNController import CNNController
from math import pi
import airsim
import cv2
import time
import math
import argparse
from controller.controller import Controller
from controller.rl.RLController import dqn
from threading import Thread
import tensorflow as tf
import numpy as np
i=0
client = airsim.MultirotorClient() #setting.json
client.confirmConnection()# architecture 
client.enableApiControl(True)#
print("arming the drone...")
client.armDisarm(True)#

landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    print("taking off...")
    client.takeoffAsync().join()

landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    print("takeoff failed - check Unreal message log for details")

cnn_controller = CNNController(client, 'controller/cnn/models/CNNModel.json', 'controller/cnn/models/CNNWeight.hdf5')
controller = Controller(client)


# airsim.wait_key('Press any key to move vehicle to destination at 2 m/s')
# client.moveToPositionAsync(15, 0, -3, 2).join()#NED 000 -> 00-3 -> 100-3
print ("nhap toa do x : ")
X= input ()
print ("\n nhap toa do y : ")
Y= input ()
print ("\n nhap toa do z : ")
Z=input ()
while True:
        client.moveByVelocityAsync(1, 0, 0, 1).join()
        img = controller.getRGBImg()
        dqn_predict = dqn.model.predict(img.reshape(1,144,256,3))
        cnn_predict = cnn_controller.predict(img)
        print(np.argmax(cnn_predict))
        controller.take_action(np.argmax(cnn_predict))
        i=i+1
        if i>10:
         
         client.moveToPositionAsync(X, Y, Z, 2).join()# di den vi tri chi dinh
         client.landAsync().join()
        # time.sleep(0.1)
        
    
# client.moveToPositionAsync(25, 0, -3, 2).join()
# client.hoverAsync().join()
# client.landAsync().join()
# client.armDisarm(False)
# client.enableApiControl(False)
