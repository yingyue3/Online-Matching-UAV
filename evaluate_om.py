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
    evaluate(10,60,1e4)
    # # medium scale
    # evaluate(10,60,1e4)
    # # large scale
    # evaluate(15,90,1.5e4)
