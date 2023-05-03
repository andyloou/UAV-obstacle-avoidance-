# import setup_path
import airsim
import numpy
import sys
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
# client.takeoffAsync().join()

# airsim.wait_key('Press any key to takeoff')
# print("Taking off...")
# client.armDisarm(True)
# client.takeoffAsync().join()
# airsim.wait_key('Press any key to move vehicle to destination at 5 m/s')
# client.moveToPositionAsync(25.5, 0, -3, 5).join()
# airsim.wait_key('press to land')
# client.landAsync().join()
# print ("done")
a=client.simGetVehiclePose().position

print(a)
x,y,z=a

# print ("\n"+ x,y)