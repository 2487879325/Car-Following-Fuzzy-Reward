import numpy as np
import copy


import skfuzzy as fuzz

import skfuzzy.control as ctrl

from scipy import stats

from sklearn.preprocessing import MinMaxScaler
class Env(object):
    def __init__(self, TTC_threshold):
        self.action_Bound = 2
        self.n_actions = 1
        self.timeWindow = 1 # if you want to consider informaiton 
        #from previous seconds, you can use timewindow > 1
        self.penalty = 100 # penalty for collisions
        self.n_features = 3*self.timeWindow
        self.TTC_threshold = TTC_threshold


    def reset(self, data):
        self.timeStep = self.timeWindow # starting form 1 to n
        self.LVSpdData = data[:, 3]

        self.SimSpaceData = np.zeros(data[:, 0].shape)
        self.SimSpeedData = np.zeros(data[:,1].shape)
        self.SimSpaceData[0] = data[0,0] # initialize with initial spacing
        self.SimSpeedData[0] = data[0,1] # initialize with initial speed

        temp = data[:self.timeWindow, :3].reshape(1, self.timeWindow * 3)
        temp = temp[0, :]
        self.s = temp
        self.currentState = self.s[-3:]
        self.isCollision = 0
        self.isStall = 0
        self.TimeLen = data.shape[0]
        self.lastAction = 0

        relSpd = self.currentState[2]
        space = self.currentState[0]
        self.TTC = - space / relSpd
        return self.s

    
    def step(self, action,sim,speed,dis,acc,rel):
        # update state
        self.timeStep += 1
        LVSpd = self.LVSpdData[self.timeStep-1]
        svSpd = self.currentState[1] + action*0.1

        if svSpd <= 0:
            svSpd = 0.00001
            self.isStall = 1
        else:
            self.isStall = 0

        relSpd = LVSpd - svSpd
        space = self.currentState[0]+relSpd*0.1
        self.currentState=[space, svSpd, relSpd]
        self.s = np.hstack((self.s[3:],self.currentState))

        #judge collision and back
        if space < 0:
            self.isCollision = 1

        #store the space history for error calculating
        self.SimSpaceData[self.timeStep-1] = space
        self.SimSpeedData[self.timeStep-1] = svSpd

        # caculate the reward
        jerk = (action - self.lastAction) / 0.1
        hdw = space / svSpd
        self.TTC = -space / relSpd  # negative sign because of relative speed sign

        fJerk = -(jerk ** 2)/3600   # the maximum range is change from -3 to 3 in 0.1 s, then the jerk = 60

        fAcc = - action**2/60

        self.lastAction = action

        if self.TTC >= 0 and self.TTC <= self.TTC_threshold:
            fTTC = np.log(self.TTC/self.TTC_threshold) 
        else:
            fTTC = 0

        mu = 0.422618  
        sigma = 0.43659
        if hdw <= 0:
            fHdw = -1
        else:
            fHdw = (np.exp(-(np.log(hdw) - mu) ** 2 / (2 * sigma ** 2)) / (hdw * sigma * np.sqrt(2 * np.pi)))



        
        # print(int(np.squeeze( speed.transform(svSpd.reshape(-1, 1)),axis=1)))
        sim.input['speed_L'] =float(np.squeeze( rel.transform(np.reshape(-relSpd,(-1, 1))),axis=1))

        # sim.input['speed'] =int(np.squeeze( speed.transform(np.reshape(svSpd,(-1, 1))),axis=1))

        sim.input['acceleration'] = float(np.squeeze( acc.transform (np.reshape(action,(-1,1))),axis=1))

        if space>=30:
          Gap=30
        else:
          Gap=space
        sim.input['distance'] = float(np.squeeze( dis.transform(np.reshape(Gap,(-1, 1))),axis=1))

        sim.compute()   # 运行系统

        reward = sim.output['reward'] - self.penalty * self.isCollision

        # calculate the reward
        # reward = fJerk + fTTC + fHdw - self.penalty * self.isCollision

        # record reward info
        rewardInfo_test = [self.TTC, hdw, jerk, fTTC, fHdw, fJerk, fAcc]

        rewardInfo = [self.TTC, hdw, jerk,reward]
        # judge the end
        if self.timeStep == self.TimeLen or self.isCollision == 1:
            done = True
        else:
            done = False
        s_=self.s

        return s_, reward, done, rewardInfo, rewardInfo_test