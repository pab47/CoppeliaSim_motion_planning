import sim
import sys

print("Program Started")
sim.simxFinish(-1) # just in case, close all opened connections
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)# Connect to CoppeliaSim

if(clientID != -1):
    print('Connected Successfully.')
else:
    sys.exit('Failed To connect.')