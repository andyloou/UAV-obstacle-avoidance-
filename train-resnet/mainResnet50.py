from dataclasses import field
import json
import airsim
import cv2
import time
import sys
import math
from gym_airsim.myAirSimClient import myAirSimClient
import numpy as np
from math import acos, pi, sqrt
import matplotlib.pyplot as plt
sys.path.append('../cnn')
# from gym_airsim.CNNController import CNNController
from controller.controller import Controller
from controller.cnn.CNNController import CNNController

loc = (50,-42,-1.9)
start_point = (70, 370) # điểm bắt đầu ở trên bản đồ
end_point = (445, 65)   # điểm kết thúc trên bản đồ
scale = (end_point[0] - start_point[0])/loc[0]
color = (0,255,255)

img_re = cv2.imread(".\\bg5.png")
save = ".\\result.png"



def angle(vector):
    x, y = vector[0],vector[1]
    if y>=0:
        return acos(x/sqrt(x**2+y**2))
    else:
        return 2*pi - acos(x/sqrt(x**2+y**2))
def distance3D(vector):
    (x,y,z) = vector
    return math.sqrt(x**2+y**2+z**2)

pre_x, pre_y = start_point
p_x,p_y,p_z = (0,0,0)
angl = angle((loc[0],loc[1]))*180/pi
stand_cnt = 0
stand_loc = (0,0,0)

client = myAirSimClient()
cnn_controller = CNNController(client, 'C:\\Users\\dangh\\Desktop\\Project 1\\train-resnet\\ResNet50finailcheckpoint.json', 'C:\\Users\\dangh\\Desktop\\Project 1\\train-resnet\\ResNet_50finalcheckpoint.h5')
#cnn_controller = CNNController(client, 'C:\\Users\\dangh\\Downloads\\ResNet50finailcheckpoint.json', 'C:\\Users\\dangh\\Downloads\\ResNet_50finalcheckpoint.h5')
controller = Controller(client)

print("arming the drone...")
client.armDisarm(True)

landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    print("taking off...")
    client.takeoffAsync().join()

landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    print("takeoff failed - check Unreal message log for details")
first = 0
d_cnt = 0
col=0
step=0
colx=[]
stepx=[]
i=0
st = time.time()
while True:
    try:
        a=client.simGetVehiclePose().position
        x,y,z=a
        d_cnt += distance3D((x-p_x, y-p_y, z-p_z))

        if abs(x-loc[0])<1 and abs(y-loc[1])<1:   # land
            client.hoverAsync().join()  
            client.landAsync().join()  
            print("Landed") 
            cv2.imwrite(save,img_re)
            break
        else:
            if stand_cnt > 8:   # go back
                client.moveToPositionAsync(0,0,loc[2],
                                            velocity=1.5,
                                            timeout_sec=1, 
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                            yaw_mode=airsim.YawMode(False, angl)).join()
                stand_loc = x,y,z
                stand_cnt = 0
                time.sleep(0.5)
                print("Go back")
            else:   # Adjust camera direction towards waypoint
                angl = angle((loc[0]-x,loc[1]-y))*180/pi
                client.moveToPositionAsync(loc[0],loc[1],loc[2], 
                                            velocity=0, 
                                            timeout_sec=0.4, 
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                            yaw_mode=airsim.YawMode(False, angl)).join()
                print("Moving")
                if distance3D((stand_loc[0]-x, stand_loc[1]-y, stand_loc[2]-z)) < 1:
                    stand_cnt+=1
                else:
                    stand_loc = x,y,z
                    stand_cnt = 0
        i+=1
        stepx.append(i)    
            ##### DQN and CNN #####
        img = controller.getRGBImg()
       # dqn_predict = dqn.model.predict(img.reshape(1,144,256,3))
        if client.simGetCollisionInfo().has_collided == True:
            print("Va chạm")
            col+=1
            client.moveToPositionAsync(0,0,loc[2],
                                            velocity=1.5,
                                            timeout_sec=1, 
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                            yaw_mode=airsim.YawMode(False, angl)).join()
            print("Go back")
            time.sleep(0.5)
        colx.append(col)
        cnn_predict = cnn_controller.predict(img)
        print("CNN: %d - %.3f" %(np.argmax(cnn_predict), 100*np.max(cnn_predict))+r"%")
        controller.take_action(np.argmax(cnn_predict))    
            ## Draw line
        x_plot, y_plot = int(x*scale)+start_point[0], int(y*scale)+start_point[1]
        img_re = cv2.line(img_re, (pre_x,pre_y),(x_plot,y_plot), color, 2)
        # cv2.imshow("camera",img)
        cv2.imshow("Map",img_re)
        cv2.waitKey(1)
        pre_x,pre_y = x_plot,y_plot
        p_x,p_y,p_z = x,y,z
    except KeyboardInterrupt:
        break
#Ve bieu do
plt.plot(stepx,colx )
plt.xlabel (" Step ")
plt.ylabel (" Collision  ")

print("Tong so va cham: ",col)
print("Time:", time.time()-st)
print("Distance:", d_cnt)
plt.show()
