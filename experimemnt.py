import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import copy
from multiprocess import Pool
from ga import GA
from aco import ACO
from pso import PSO
from om import OM
from env_online import Env_online
from env_offline import Env_offline
from tqdm import tqdm
import time

def evaluate(vehicle_num, target_num, map_size):
    if vehicle_num==5:
        size='small'
    if vehicle_num==10:
        size='medium'
    if vehicle_num==15:
        size='large'
    env = Env_online(vehicle_num,target_num,map_size,task_appear_rate=0.5)
    env.run()
    # env.visualize()

    
    
    
if __name__=='__main__':
    # small scale
    # evaluate(5,30,5e3)
    # # medium scale
    # evaluate(10,60,1e4)
    # # large scale
    # evaluate(15,90,1.5e4)
    count_offline = 0
    count_online = 0
    time_offline = 0
    time_online = 0
    for i in tqdm (range (50), desc="Progress..."):
        env_online = Env_online(15,90,1.5e4,task_appear_rate=0.5)
        om_result = env_online.run()
        env_offline = Env_offline(15,90,1.5e4)
        env_offline.time_lim = env_online.time_lim
        env_offline.task_inheritation(env_online.targets,env_online.vehicle_speed)
        # GA = GA(5,env_offline.vehicles_speed,env_offline.target_num,env_offline.targets,env_offline.time_lim)
        pso = PSO(15,env_offline.target_num ,env_offline.targets,env_offline.vehicles_speed,env_offline.time_lim)
        ga_result = pso.run()
        ga_task_assignmet = ga_result[0]
        env_offline.run(ga_task_assignmet,'GA',0,0)
        time_online += om_result[1]
        time_offline += ga_result[1]
        count_online += om_result[0]
        count_offline += env_offline.total_reward
    print("total time Online: ", time_online)
    print("total time Offline: ", time_offline)
    print("total reward Online: ", count_online)
    print("total reward Offline: ", count_offline)
    print("ratio: ", count_online/count_offline)


