import numpy as np
import pygame
import pickle
import sys
from models import CAE
import torch
import gym
from stable_baselines import SAC

class Joystick(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.DEADBAND = 0.1
        self.LEFT_SCALE = 3
        self.RIGHT_SCALE = 1.0

    def input(self):

        pygame.event.get()
        z1 = self.gamepad.get_axis(0)
        z = 0.0 
        if abs(z1) > self.DEADBAND:
            if z1 > 0:
                z = - self.RIGHT_SCALE * z1
            else:
                z = - self.LEFT_SCALE* z1
        
        e_stop = self.gamepad.get_button(7)
        return [z], e_stop


class Model(object):

    def __init__(self):
        self.model = CAE()
        model_dict = torch.load('models/CAE_best_model', map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval

    def decoder(self, z, s):
        if abs(z[0][0]) < 0.01:
             return [0.0] * 6
        # z = np.asarray([z])
        z = np.asarray(z)
        # print(s.shape)
        z_tensor = torch.FloatTensor(np.concatenate((z,s),axis=1))
        # print(z_tensor.shape)
        a_tensor = self.model.decoder(z_tensor)
        return a_tensor.tolist()
    def encoder(self, a, s):
        
        x = np.concatenate((a,s),axis=1)
        x_tensor = torch.FloatTensor(x)
        z = self.model.encoder(x_tensor)
        return z.tolist()


def main():

    # create environment instance
    env = gym.make('HalfCheetah-v2')

    # reset the environment
    env.reset()
    # env.viewer.set_camera(camera_id=0)

    # create the human input device
    joystick = Joystick()
    model = Model()
    
    action_scale = 1

    # initialize state
    obs, reward, done, info = env.step([0.0]*6)
    s = np.array([obs])
    record_z=[]


    for i in range(10000):

        z, e_stop = joystick.input()
        if e_stop:
            return True
        

        # action_exp = np.array([demo.act(obs)])
        # x = np.expand_dims(np.concatenate((action_exp, s), axis=1),axis=0)
        # x= torch.FloatTensor(x)
        # z = model.encoder(action_exp, s)
        # print(z)

        z = [z]
        action_arm = model.decoder(z, s[:,:])

        if abs(z[0][0]) > 0.01:
            print(z[0][0])
        
        action = np.asarray(action_arm)        

        obs, reward, done, info = env.step(action)
        s = np.array([obs])

        env.render()
    env.close()



if __name__ == "__main__":
    main()
