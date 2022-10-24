# %% Setup
import kinematics as kin
import transforms as tf
from visualization import VizScene
import numpy as np
import time
import scipy.io as loadmat
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
# Define robot link lengths and offsets
l_1 = 270.35/1000 # m 
l_2 = 69.0/1000 # m
l_3 = 364.35/1000 # m
l_4 = 69.0/1000 # m
l_5 = 374.29/1000 # m
l_6 = 10.0/1000 # m
l_7 = 368.30/1000 # m

# Define DH parameters
dh = [[0.0, l_1, l_2, -np.pi/2.0],
      [np.pi/2.0, 0.0, 0.0, np.pi/2.0],
      [0.0, l_3, l_4, -np.pi/2.0],
      [0.0, 0.0, 0.0, np.pi/2.0],
      [0.0, l_5, l_6, -np.pi/2.0],
      [0.0, 0.0, 0.0, np.pi/2.0],
      [0.0, l_7, 0.0, 0.0]]

# Define base frame
base = kin.se3(kin.rotz(-np.pi/4), np.array([.06353, -.2597, .119]))
print(base)

# Define joint types
jt_types = ['r', 'r', 'r', 'r', 'r', 'r', 'r']

# %% Importing data from the 10 .mat files

mat_00 = loadmat.loadmat('data/part3_trial00.mat')
t_00 = mat_00['t']
q_00 = mat_00['q']
q_dot_00 = mat_00['q_dot']

mat_01 = loadmat.loadmat('data/part3_trial01.mat')
t_01 = mat_01['t']
q_01 = mat_01['q']
q_dot_01 = mat_01['q_dot']

mat_02 = loadmat.loadmat('data/part3_trial02.mat')
t_02 = mat_02['t']
q_02 = mat_02['q']
q_dot_02 = mat_02['q_dot']

mat_03 = loadmat.loadmat('data/part3_trial03.mat')
t_03 = mat_03['t']
q_03 = mat_03['q']
q_dot_03 = mat_03['q_dot']

mat_04 = loadmat.loadmat('data/part3_trial04.mat')
t_04 = mat_04['t']
q_04 = mat_04['q']
q_dot_04 = mat_04['q_dot']

mat_05 = loadmat.loadmat('data/part3_trial05.mat')
t_05 = mat_05['t']
q_05 = mat_05['q']
q_dot_05 = mat_05['q_dot']

mat_06 = loadmat.loadmat('data/part3_trial06.mat')
t_06 = mat_06['t']
q_06 = mat_06['q']
q_dot_06 = mat_06['q_dot']

mat_07 = loadmat.loadmat('data/part3_trial07.mat')
t_07 = mat_07['t']
q_07 = mat_07['q']
q_dot_07 = mat_07['q_dot']

mat_08 = loadmat.loadmat('data/part3_trial08.mat')
t_08 = mat_08['t']
q_08 = mat_08['q']
q_dot_08 = mat_08['q_dot'] 

mat_09 = loadmat.loadmat('data/part3_trial09.mat')
t_09 = mat_09['t']
q_09 = mat_09['q']
q_dot_09 = mat_09['q_dot']

print(len(t_00.reshape(-1)))
print(q_00[0])


# %% Part 1

# Recorded data from first position
q_pos_1 = [ 5.21553468e-01, -7.55869033e-01, -3.76975779e-01,  1.10830112e-01,
  1.53398079e-03, -1.56926235e+00,  3.03383050e+00]
e_baxter_position_1 = np.array([ 0.54356003, -0.28592334,  1.10135507])
e_baxter_orientation_1 =  np.array([[-0.70168595, -0.56479674, -0.43432876],
                           [ 0.33473625, -0.79945728,  0.4988183 ],
                           [-0.62895824,  0.20462821,  0.75002589]])

# Recorded data from second position
q_pos_2 = [ 1.48335942,  0.39730102, -0.15608255,  0.99286906, -2.56059743, -0.26844664,
  0.30679616]
e_baxter_position_2 =  np.array([0.4372663,  -0.03076617, -0.56523])
e_baxter_orientation_2 =  np.array([[ 0.9918558,  -0.12038398, -0.04159054],
                                    [-0.12185556, -0.99193534, -0.0348641 ],
                                    [-0.03705804,  0.0396482,  -0.99852628]])

# Recorded data from third position
q_pos_3 = [ 0.56872338, -0.16413594, -1.92629637,  1.67280605, -3.05108779,  0.45252433,
 -1.31883998]
e_baxter_position_3 =  np.array([ 0.37499241, -1.0944868,   0.75941688])
e_baxter_orientation_3 =  np.array([[ 0.39476682,  0.91486435,  0.0847489 ],
                                    [-0.35378952,  0.23649027, -0.90493388],
                                    [-0.84793404,  0.3272546,   0.41702793]] )

# Recorded data from fourth position
q_pos_4 = [ 0.70601466,  0.29950975, -2.77190328,  0.47515055, -3.05070429,  0.18292721,
 -3.05377225]
e_baxter_position_4 =  np.array([ 1.24015552, -0.4721267, 0.37760778])
e_baxter_orientation_4 =  np.array([[-0.06559776, -0.1580762,   0.98524558],
                                    [-0.49874935, -0.84999455, -0.16958285],
                                    [ 0.86426039, -0.50251485, -0.02308266]] )

# Make 4 arm objects, one for each trial
arm1 = kin.SerialArm(dh, jt=jt_types, base = base)
arm2 = kin.SerialArm(dh, jt=jt_types, base = base)
arm3 = kin.SerialArm(dh, jt=jt_types, base = base)
arm4 = kin.SerialArm(dh, jt=jt_types, base = base)

# Get end effector positions for each position
ee1 = arm1.fk(q_pos_1, base= True)
ee2 = arm2.fk(q_pos_2, base= True)
ee3 = arm3.fk(q_pos_3, base= True)
ee4 = arm4.fk(q_pos_4, base= True)

# homogeneous transformation matrices for each measured position
measured_1 = tf.se3(e_baxter_orientation_1, e_baxter_position_1)
measured_2 = tf.se3(e_baxter_orientation_2, e_baxter_position_2)
measured_3 = tf.se3(e_baxter_orientation_3, e_baxter_position_3)
measured_4 = tf.se3(e_baxter_orientation_4, e_baxter_position_4)

# Difference between measured and calculated end effector positions
print(ee1-measured_1)
print(ee2-measured_2)
print(ee3-measured_3)
print(ee4-measured_4)



# Visualization of the end effector positions. Measured positions are marked in green
viz = VizScene()
viz.add_arm(arm1, draw_frames=True)
viz.add_arm(arm2, draw_frames = True)
viz.add_arm(arm3, draw_frames = True)
viz.add_arm(arm4, draw_frames = True)

viz.add_marker(measured_1[0:3,3], size = 20)
viz.add_marker(measured_2[0:3,3], size = 20)
viz.add_marker(measured_3[0:3,3], size = 20)
viz.add_marker(measured_4[0:3,3], size = 20)

viz.update(qs = [q_pos_1, q_pos_2, q_pos_3, q_pos_4])

time_to_run = 10
refresh_rate = 60

for i in range(time_to_run*refresh_rate):
    viz.update()
    time.sleep(1.0/refresh_rate)

viz.close_viz()

# %% Part 2

# TODO: Input remaining data and compute end effector positions to compare to measured positions

# First iteration

# This isn't the commanded angle but just the first set of angles from the terminal output
q_commanded_1 = [ 0.24083498, -0.58521367, -3.03574798, -0.0479369,  -3.05108779,  1.12172345,
 -3.04456837]

ee_pos_measured_1 = [ 0.93752018, -0.7118265, 0.62142902]
ee_rot_measured_1 = [[ 0.39406314, -0.46179166,  0.79464627],
                      [-0.43988057, -0.85391272, -0.27809739],
                      [ 0.80698162, -0.23996152, -0.5396287 ]] 

# End effector Homogeneous Transformation from measured data
ee_hom_1 = tf.se3(ee_rot_measured_1, ee_pos_measured_1)

# %% Animating the motion of part 3, iteration 1 to check that the DH parameters are correct

# show the robot and a goal (just for demo's sake)
viz = VizScene()
viz.add_arm(arm, draw_frames=True)
# viz.add_marker(T[0:3,3], size = 20)
viz.update(qs = [q])

time_to_run = 10
refresh_rate = 60

for i in range(t_00.shape[1]):
    viz.update(qs = q_00[i].tolist())
    time.sleep(1.0/refresh_rate)


viz.close_viz()

# %%

arm = kin.SerialArm(dh, jt=jt_types, base = base)

tip_velocities_1 = np.empty((0,6))
tip_velocities_2 = np.empty((0,6))
tip_velocities_3 = np.empty((0,6))
tip_velocities_4 = np.empty((0,6))
tip_velocities_5 = np.empty((0,6))
tip_velocities_6 = np.empty((0,6))
tip_velocities_7 = np.empty((0,6))
tip_velocities_8 = np.empty((0,6))
tip_velocities_9 = np.empty((0,6))
tip_velocities_10 = np.empty((0,6))

for i in range(0, t_00.shape[1]):
    
    tip_velocities_1 = np.append(tip_velocities_1, (arm.jacob(q_00[i]) @ q_dot_00[i]).reshape(1,6), axis = 0)
    tip_velocities_2 = np.append(tip_velocities_2, (arm.jacob(q_01[i]) @ q_dot_01[i]).reshape(1,6), axis = 0)
    tip_velocities_3 = np.append(tip_velocities_3, (arm.jacob(q_02[i]) @ q_dot_02[i]).reshape(1,6), axis = 0)
    tip_velocities_4 = np.append(tip_velocities_4, (arm.jacob(q_03[i]) @ q_dot_03[i]).reshape(1,6), axis = 0)
    tip_velocities_5 = np.append(tip_velocities_5, (arm.jacob(q_04[i]) @ q_dot_04[i]).reshape(1,6), axis = 0)
    tip_velocities_6 = np.append(tip_velocities_6, (arm.jacob(q_05[i]) @ q_dot_05[i]).reshape(1,6), axis = 0)
    tip_velocities_7 = np.append(tip_velocities_7, (arm.jacob(q_06[i]) @ q_dot_06[i]).reshape(1,6), axis = 0)
    tip_velocities_8 = np.append(tip_velocities_8, (arm.jacob(q_07[i]) @ q_dot_07[i]).reshape(1,6), axis = 0)
    tip_velocities_9 = np.append(tip_velocities_9, (arm.jacob(q_08[i]) @ q_dot_08[i]).reshape(1,6), axis = 0)
    tip_velocities_10 = np.append(tip_velocities_10, (arm.jacob(q_09[i]) @ q_dot_09[i]).reshape(1,6), axis = 0)

# %% 

# TODO: Plot the linear and angular velocities of the end effector for each iteration. Not sure if we want to plot them here or just export the velocites to a csv file and plot with excel.
#  
fig_linear_plots, linears = plt.subplots(3)
fig_angular_plots, angulars = plt.subplots(3)

velocities = [tip_velocities_1, tip_velocities_2, tip_velocities_3, tip_velocities_4, tip_velocities_5, tip_velocities_6, tip_velocities_7, tip_velocities_8, tip_velocities_9, tip_velocities_10]
times = [t_00, t_01, t_02, t_03, t_04, t_05, t_06, t_07, t_08, t_09]

# for i in range(len(velocities)):
#     linears[0].plot(times[i-1], velocities[i-1][:,0].reshape(1,4998), label = 'x')
#     linears[1].plot(times[i-1], velocities[i-1][:,1].reshape(1,4998), label = 'y')
#     linears[2].plot(times[i-1], velocities[i-1][:,2].reshape(1,4998), label = 'z')
#     angulars[0].plot(times[i-1], velocities[i-1][:,3].reshape(1,4998), label = 'x')
#     angulars[1].plot(times[i-1], velocities[i-1][:,4].reshape(1,4998), label = 'y')
#     angulars[2].plot(times[i-1], velocities[i-1][:,5].reshape(1,4998), label = 'z')

# plt.show()

# %%
