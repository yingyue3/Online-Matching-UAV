import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import copy
from multiprocess import Pool
from ga import GA
from aco import ACO
from pso import PSO

class Env_offline():
    def __init__(self, vehicle_num, target_num, map_size, visualized=True, time_cost=None, repeat_cost=None):
        self.vehicles_position = np.zeros(vehicle_num,dtype=np.int32)
        self.vehicles_speed = np.zeros(vehicle_num,dtype=np.int32)
        self.targets = np.zeros(shape=(target_num+1,4),dtype=np.int32)
        if vehicle_num==5:
            self.size='small'
        if vehicle_num==10:
            self.size='medium'
        if vehicle_num==15:
            self.size='large'
        self.map_size = map_size
        self.speed_range = [10, 15, 30]
        #self.time_lim = 1e6
        self.time_lim = self.map_size / self.speed_range[1]
        self.vehicles_lefttime = np.ones(vehicle_num,dtype=np.float32) * self.time_lim
        self.distant_mat = np.zeros((target_num+1,target_num+1),dtype=np.float32)
        self.total_reward = 0
        self.reward = 0
        self.visualized = visualized
        self.time = 0
        self.time_cost = time_cost
        self.repeat_cost = repeat_cost
        self.end = False
        self.assignment = [[] for i in range(vehicle_num)]
        self.task_generator()
        
    def task_generator(self):
        for i in range(self.vehicles_speed.shape[0]):
            choose = random.randint(0,2)
            self.vehicles_speed[i] = self.speed_range[choose]
        for i in range(self.targets.shape[0]-1):
            self.targets[i+1,0] = random.randint(1,self.map_size) - 0.5*self.map_size # x position
            self.targets[i+1,1] = random.randint(1,self.map_size) - 0.5*self.map_size # y position
            self.targets[i+1,2] = random.randint(1,10) # reward
            self.targets[i+1,3] = random.randint(5,30) # time consumption to finish the mission  
        for i in range(self.targets.shape[0]):
            for j in range(self.targets.shape[0]):
                self.distant_mat[i,j] = np.linalg.norm(self.targets[i,:2]-self.targets[j,:2])
        self.targets_value = copy.deepcopy((self.targets[:,2]))
    
    def task_inheritation(self, targets, time_delay, vehicles_speed):
        self.vehicles_speed = vehicles_speed
        for i in range(self.targets.shape[0]-1):
            self.targets[i+1] = targets[i]
            self.targets[i+1,3] += time_delay[i]

        for i in range(self.targets.shape[0]):
            for j in range(self.targets.shape[0]):
                self.distant_mat[i,j] = np.linalg.norm(self.targets[i,:2]-self.targets[j,:2])
        self.targets_value = copy.deepcopy((self.targets[:,2]))
        
        
    def step(self, action):
        count = 0
        for j in range(len(action)):
            k = action[j]
            delta_time = self.distant_mat[self.vehicles_position[j],k] / self.vehicles_speed[j] + self.targets[k,3]
            self.vehicles_lefttime[j] = self.vehicles_lefttime[j] - delta_time
            if self.vehicles_lefttime[j] < 0:
                count = count + 1
                continue
            else:
                if k == 0:
                    self.reward = - self.repeat_cost
                else:
                    self.reward = self.targets[k,2] - delta_time * self.time_cost + self.targets[k,2]
                    if self.targets[k,2] == 0:
                        self.reward = self.reward - self.repeat_cost
                    self.vehicles_position[j] = k
                    self.targets[k,2] = 0
                self.total_reward = self.total_reward + self.reward
            self.assignment[j].append(action)
        if count == len(action):
            self.end = True
        
    def run(self, assignment, algorithm, play, rond):
        self.assignment = assignment
        self.algorithm = algorithm
        self.play = play
        self.rond = rond
        self.get_total_reward()
        if self.visualized:
            self.visualize()        
            
    def reset(self):
        self.vehicles_position = np.zeros(self.vehicles_position.shape[0],dtype=np.int32)
        self.vehicles_lefttime = np.ones(self.vehicles_position.shape[0],dtype=np.float32) * self.time_lim
        self.targets[:,2] = self.targets_value
        self.total_reward = 0
        self.reward = 0
        self.end = False
        
    def get_total_reward(self):
        for i in range(len(self.assignment)):
            speed = self.vehicles_speed[i]
            for j in range(len(self.assignment[i])):
                position = self.targets[self.assignment[i][j],:4]
                self.total_reward = self.total_reward + position[2]
                if j == 0:
                    self.vehicles_lefttime[i] = self.vehicles_lefttime[i] - np.linalg.norm(position[:2]) / speed - position[3]
                else:
                    self.vehicles_lefttime[i] = self.vehicles_lefttime[i] - np.linalg.norm(position[:2]-position_last[:2]) / speed - position[3]
                position_last = position
                if self.vehicles_lefttime[i] > self.time_lim:
                    self.end = True
                    break
            if self.end:
                self.total_reward = 0
                break
            
    def visualize(self):
        if self.assignment == None:
            plt.scatter(x=0,y=0,s=200,c='k')
            plt.scatter(x=self.targets[1:,0],y=self.targets[1:,1],s=self.targets[1:,2]*10,c='r')
            plt.title('Target distribution')
            plt.savefig('task_pic/'+self.size+'/'+self.algorithm+ "-%d-%d.png" % (self.play,self.rond))
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
            plt.savefig('task_pic/'+self.size+'/'+self.algorithm+ "-%d-%d.png" % (self.play,self.rond))
            plt.cla()
            
def evaluate(vehicle_num, target_num, map_size):
    if vehicle_num==5:
        size='small'
    if vehicle_num==10:
        size='medium'
    if vehicle_num==15:
        size='large'
    env = Env_offline(vehicle_num,target_num,map_size,visualized=False)
    ga = GA(vehicle_num,env.vehicles_speed,target_num,env.targets,env.time_lim)
    ga_result = ga.run()
    ga_task_assignmet = ga_result[0]
    print("target:==============")
    print(env.targets)
    print("target value:===============")
    print(env.targets_value)
    env.run(ga_task_assignmet,'GA',0,0)
    print(env.total_reward)
    print(ga_task_assignmet)

    
    
    
if __name__=='__main__':
    # small scale
    evaluate(10,60,5e3)
    # # medium scale
    # evaluate(10,60,1e4)
    # # large scale
    # evaluate(15,90,1.5e4)
