import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import copy
from multiprocess import Pool
from om import OM
import time

class Env_online():
    def __init__(self, vehicle_num, target_num, map_size, task_appear_rate=0.5):
        self.vehicle_num = vehicle_num
        self.targets = np.zeros(shape=(target_num,4),dtype=np.int32)
        self.targets_value = np.zeros(target_num,dtype=np.int32)
        self.targets_num = target_num
        self.target_appear_rate = task_appear_rate # [0.5,0.75,1]
        self.map_size = map_size
        self.time_lim = 1000
        # self.time_lim = self.map_size / 15 # speed range [10,15,20]
        self.time_left = self.time_lim
        self.total_reward = 0
        self.time = 0
        self.end = False
        self.assignment = [[] for i in range(vehicle_num)]
        self.algorithm = 'Online Matching'
        self.play = 0
        self.rond = 0
        self.task_delay = np.zeros(target_num)
        self.unsucess = []
        
    def task_generator(self, i):
        self.targets[i,0] = random.randint(1,self.map_size) - 0.5*self.map_size # x position
        self.targets[i,1] = random.randint(1,self.map_size) - 0.5*self.map_size # y position
        self.targets[i,2] = random.randint(1,10) # reward
        self.targets[i,3] = random.randint(5,30) # time consumption to finish the mission  
        self.targets_value[i] = self.targets[i,2]
        
    def run(self):
        print("Online matching start")
        start_time = time.time()
        om = OM(self.vehicle_num, self.time_lim)
        time_window = np.floor(self.time_lim/self.targets_num)
        # print("time left behind,", self.time_lim - self.targets_num*time_window)
        # print("time window", time_window)
        for i in range(self.targets_num):
            self.task_generator(i)
            reward, assignment, time_delay, unsucees = om.assignment(self.target_appear_rate, self.targets[i], time_window)
            self.time_left -= time_window
            self.task_delay[i] = time_window *i + time_delay
            if assignment > -1:
                self.total_reward += reward
                self.assignment[assignment].append(i)
            if unsucees:
                self.unsucess.append(i)
        self.vehicle_speed = om.vehicles_speed
        end_time = time.time()
        # print("time_delay", self.task_delay)
        # print("time_left", self.time_left)
        print("assignment", self.assignment)
        # print('targets_value',self.targets_value)
        print("total rewards", self.total_reward)
        # print("vehicals left time: ", om.vehicles_lefttime)
        print("Online matching time", end_time-start_time)

            
    def reset(self):
        self.targets_value = np.zeros(self.target_num,dtype=np.int32)
        self.targets[:,2] = np.zeros(shape=(self.target_num,4),dtype=np.int32)
        self.time_left = self.time_lim
        self.total_reward = 0
        self.end = False
    
    def visualize(self, size, play, rond):
        if self.assignment == None:
            plt.scatter(x=0,y=0,s=200,c='k')
            plt.scatter(x=self.targets[1:,0],y=self.targets[1:,1],s=self.targets[1:,2]*10,c='r')
            plt.title('Target distribution')
            plt.savefig('task_pic/'+size+'/'+self.algorithm+ "-%d-%d.png" % (play,rond))
            plt.cla()
        else:
            plt.title('Task assignment by '+self.algorithm +', total reward : '+str(self.total_reward))     
            plt.scatter(x=0,y=0,s=200,c='k')
            plt.scatter(x=self.targets[1:,0],y=self.targets[1:,1],s=self.targets[1:,2]*10,c='r')
            for i in range(len(self.assignment)):
                trajectory = np.array([[0,0,20]])
                for j in range(len(self.assignment[i])):
                    position = self.targets[self.assignment[i][j],:3]
                    trajectory = np.insert(trajectory,j+1,values=position,axis=0)  
                plt.scatter(x=trajectory[1:,0],y=trajectory[1:,1],s=trajectory[1:,2]*10,c='b')
                plt.plot(trajectory[:,0], trajectory[:,1]) 
            plt.savefig('task_pic/'+size+'/'+self.algorithm+ "-%d-%d.png" % (play,rond))
            plt.cla()

        