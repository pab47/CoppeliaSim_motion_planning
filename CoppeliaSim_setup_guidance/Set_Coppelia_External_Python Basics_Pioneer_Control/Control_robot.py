import sim
import time
import sys

print("Program Started")
sim.simxFinish(-1) #CLose the previous connection
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5) # Establish the connection

if(clientID != -1):
    print('Connected Successfully.')
else:
    sys.exit('Failed To connect.')

time.sleep(1)


#Get the object handle for the motors
error_code, left_motor_handle = sim.simxGetObjectHandle(clientID, '/PioneerP3DX/leftMotor', sim.simx_opmode_oneshot_wait)
error_code, right_motor_handle = sim.simxGetObjectHandle(clientID, '/PioneerP3DX/rightMotor', sim.simx_opmode_oneshot_wait)


#Give the commands to the robot about velocity
error_code = sim.simxSetJointTargetVelocity(clientID, left_motor_handle, 0.4, sim.simx_opmode_oneshot_wait)
error_code = sim.simxSetJointTargetVelocity(clientID, right_motor_handle, 1, sim.simx_opmode_oneshot_wait)