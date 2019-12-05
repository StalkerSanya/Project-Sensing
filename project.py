
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from gmplot import gmplot


# read messages from gps_topic 
def load_gps_data(topic_name, file_path):
	# open bag
    bag  = rosbag.Bag(file_path, 'r')
	# set arrays to store data
    num_msg = bag.get_message_count(topic_filters=[topic_name])
    # 1 column - time; 2 column - latitude; 3 column longitude; 4 column altitude
    gps = np.zeros((num_msg, 4), dtype = "float32")
	# set a flag
    Iter = 0
	# loop over the topic to read evey message
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        nsec = t.to_nsec()
        gps[Iter, 0] = (nsec-1317027872349658966)*1e-9
        gps[Iter, 1] = msg.latitude
        gps[Iter, 2] = msg.longitude
        gps[Iter, 3] = msg.altitude
        Iter += 1
    bag.close()
    return gps

#Loading imu dataset: accelerometer, gyroscop
def load_imu_data(topic_name, file_path):
	# open bag
    bag  = rosbag.Bag(file_path, 'r')
	# set arrays to store data
    num_msg = bag.get_message_count(topic_filters=[topic_name])
    # 1 column - time; 2 column - latitude; 3 column longitude; 4 column altitude
    acc = np.zeros((num_msg, 4), dtype = "float32")
    gyr = np.zeros((num_msg, 4), dtype = "float32")
	# set a flag
    Iter = 0
	# loop over the topic to read evey message
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        nsec = t.to_nsec()
        acc[Iter, 0] = (nsec-1317027872349658966)*1e-9
        acc[Iter, 1] = msg.linear_acceleration.x
        acc[Iter, 2] = msg.linear_acceleration.y
        acc[Iter, 3] = msg.linear_acceleration.z
        gyr[Iter, 0] = (nsec-1317027872349658966)*1e-9
        gyr[Iter, 1] = msg.angular_velocity.x
        gyr[Iter, 2] = msg.angular_velocity.y
        gyr[Iter, 3] = msg.angular_velocity.z
        Iter += 1
    bag.close()
    return acc, gyr

	
def load_tf_data(topic_name, file_path):
	# open bag
    bag  = rosbag.Bag(file_path, 'r')
	# set arrays to store data
    num_msg = bag.get_message_count(topic_filters=[topic_name])
    # 1 column - time; 2 column - latitude; 3 column longitude; 4 column altitude
    tf = np.zeros((num_msg, 4), dtype = "float32")
	# set a flag
    Iter = 0
	# loop over the topic to read evey message
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        nsec = t.to_nsec()
        tf[Iter, 0] = (nsec-1317027872349658966)*1e-9
        tf[Iter, 1] = msg.transforms[0].transform.translation.x
        tf[Iter, 2] = msg.transforms[0].transform.translation.y
        tf[Iter, 3] = msg.transforms[0].transform.translation.z
        Iter += 1
    bag.close()
    return tf

# google map plotting
def gmap_plot(gps_lat, gps_lon, filename):
    gmap = gmplot.GoogleMapPlotter(gps_lat[0], gps_lon[0], 20)
    gmap.plot(gps_lat, gps_lon, 'cornflowerblue', edge_width=10)
    # Draw
    gmap.draw(filename +".html")


def tran_gps_to_meter(gps):
    R = 6.371*10**6
    gps_m = np.zeros(gps.shape, dtype ="float32")
    gps_m[:,0] = gps[:,0]
    gps_m[:,3] =gps[:,3]
    for i in range(gps.shape[0]):
        phi = gps[i,1]*np.pi/180
        psi = gps[i,2]*np.pi/180
        h = gps[i,3]
        gps_m[i,1] = (R+h)*np.cos(phi)*np.cos(psi)
        gps_m[i,2] = (R+h)*np.cos(phi)*np.sin(psi)
    return gps_m

       
def tran_gps_to_degree(gps):
    R = 6.371*10**6
    gps_ang = np.zeros(gps.shape, dtype ="float32")
    gps_ang[:,0] = gps[:,0]
    gps_ang[:,3] =gps[:,3]
    for i in range(gps.shape[0]):
        X = gps[i,1]
        Y = gps[i,2]
        h = gps[i,3]
        psi = np.arctan2(Y,X)*180/np.pi
        gps_ang[i,2] = psi
        gps_ang[i,1] = np.arccos(Y/(R+h)/np.sin(psi*np.pi/180))*180/np.pi
    return gps_ang


def tran_X_to_degree(gps, gph):
    R = 6.371*10**6
    gps_ang = np.zeros(gps.shape, dtype ="float32")
    for i in range(gps.shape[0]):
        X = gps[i,0]
        Y = gps[i,1]
        h = gph[i,3]
        psi = np.arctan2(Y,X)*180/np.pi
        gps_ang[i,1] = psi
        gps_ang[i,0] = np.arccos(Y/(R+h)/np.sin(psi*np.pi/180))*180/np.pi
    return gps_ang

def tilt_from_gyr(gyr):
    gr = np.zeros(gyr.shape, dtype= "float32")
    t0 = gyr[0,0] - 0.0
    gr[0,1] = gyr[0,1]*t0
    gr[0,2] = gyr[0,2]*t0
    gr[0,3] = gyr[0,3]*t0
    gr[:,0] = gyr[:,0]
    for i in range(1, gyr.shape[0]):
        dt = gyr[i,0] - gyr[i-1,0]
        gr[i,1] = gr[i-1,1] +gyr[i,1]*dt
        gr[i,2] = gr[i-1,2] +gyr[i,2]*dt
        gr[i,3] = gr[i-1,3] +gyr[i,3]*dt
    return gr

def remove_gravity(acc, gr):
    ## Removing from acceleration data gravity acceleration
    acr = np.zeros(acc.shape, dtype= "float32")
    acr[:,0] = acc[:,0]
    for i in range(0,acc.shape[0]):
        acr[i,1] = acc[i,1] - np.sin(gr[i,1])*9.80665
        acr[i,2] = acc[i,2] - np.sin(gr[i,2])*9.80665
        acr[i,3] = acc[i,3] - np.cos(gr[i,3])*9.80665
    return acr


def acc_to_vel(acc):
    ##  Obtaining velocity data from acceleration data by integration
    vel = np.zeros((acc.shape[0], 3), dtype= "float32")
    t0 = acc[0,0] - 0.0
    vel[0,1] = acc[0,1]*t0
    vel[0,2] = acc[0,2]*t0
    vel[:,0] = acc[:,0]
    for i in range(1,acc.shape[0]):
        dt = acc[i,0] - acc[i-1,0]
        vel[i,1] = vel[i-1,1] +acc[i,1]*dt
        vel[i,2] = vel[i-1,2] +acc[i,2]*dt
    return vel


def Kalman_Filter(gps, vel, acc):
    ## Multidimensional Kalman Filter
    X = np.zeros((acc.shape[0], 4), dtype = "float32")
    X[0,0] = gps[0,1]
    X[0,1] = gps[0,2]
    X[0,2] = vel[0,1]
    X[0,3] = vel[0,2]
    
    dt = 1/10
    
    A = np.array([[1, 0, dt,  0  ],
                  [0, 1, 0 ,  dt ],
                  [0, 0, 1 ,  0  ],
                  [0, 0, 0 ,  1  ]])
    
    B = np.array([[1/2*(dt)**2,        0       ],
                  [      0    ,    1/2*(dt)**2 ],
                  [     dt    ,        0       ],
                  [      0    ,        dt      ]])
    
    Pk = np.array([[0.001,   0  , 0  , 0 ],
                   [0    , 0.001, 0  , 0 ],
                   [0    ,   0  , 0.5, 0 ],
                   [0    ,   0  , 0  ,0.5]])
     
    H = np.array([[1,0,0,0],
                  [0,1,0,0]])
    
    R = np.array([[0.01,   0 ],
                  [ 0  , 0.01]])
    
    C = np.array([[1,0],
                  [0,1]])
    
    I = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]])
    
    for i in range(1, X.shape[0]):
        Xp = np.dot(A,X[i-1, :]) + np.dot(B, acc[i,1:3])
        Pp = A.dot(Pk.dot(A.transpose()))
        K = np.dot(Pp.dot(H.transpose()), np.linalg.inv(H.dot(Pp.dot(H.transpose())) + R))
        Y = np.dot(C, np.array([gps[i,1], gps[i,2]]))
        X[i,:] = Xp + K.dot(Y - H.dot(Xp))
        Pk = np.dot((I - K.dot(H)), Pp)
    return X[:,:2]
#def tran_tf_to_degree(tf, gph):
#    R = 6.371*10**6
#    tf_ang = np.zeros(tf.shape, dtype ="float32")
#    tf_ang[:,0] = tf[:,0]
#    tf_ang[:,3] = tf[:,3]
#    for i in range(tf.shape[0]):
#        X = tf[i,1]
#        Y = tf[i,2]
#        h = gph[i,3]
#        psi = np.arctan2(Y,X)*180/np.pi
#        tf_ang[i,2] = psi
#        tf_ang[i,1] = np.arccos(Y/np.sin(psi/180*np.pi))*180/np.pi
#    return tf_ang

    
if __name__ == '__main__':
    file_path = "./dataset/kitti_2011_09_26_drive_0005_synced.bag"
    topic_gps = "/kitti/oxts/gps/fix"
    topic_imu = "/kitti/oxts/imu"
    topic_tf = "/tf"
    # freqency 10 Hz
    gps_raw = load_gps_data(topic_gps, file_path)
    gmap_plot(gps_raw[:,1], gps_raw[:,2], "gps_trac")
    acc_raw, gyr_raw = load_imu_data(topic_imu, file_path)
    gps_m = tran_gps_to_meter(gps_raw)
    g_tilt = tilt_from_gyr(gyr_raw)
    accr = remove_gravity(acc_raw, g_tilt)
    vel = acc_to_vel(accr)
    
    
    X = Kalman_Filter(gps_m, vel, accr)
    Xd = tran_X_to_degree(X, gps_raw)
    gmap_plot(Xd[:,0], Xd[:,1], "trac1")
    
    
    plt.figure(1)
    plt.scatter(gps_m[:,1],gps_m[:,2], s = 5)
    plt.title("TRegectory before Kalman Filter")
    plt.xlabel("X, m")
    plt.ylabel("Y, m")
    plt.show()
    
    plt.figure(2)
    plt.scatter(X[:,0],X[:,1], s = 5)
    plt.title("TRegectory after Kalman Filter")
    plt.xlabel("X, m")
    plt.ylabel("Y, m")
    plt.show()
    
    
    tf  = load_tf_data(topic_tf, file_path)
    print(tf)
    
    plt.figure(3)
    plt.scatter(tf[:,1], tf[:,2], s = 5)
    plt.title("TRegectory from tf_static")
    plt.xlabel("X, m")
    plt.ylabel("Y, m")
    plt.show()
    

    
    