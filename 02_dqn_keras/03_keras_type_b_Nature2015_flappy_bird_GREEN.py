#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

import tensorflow as tf
import os
import os.path
import pickle
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import time, datetime

# import json
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

FRAME_PER_ACTION = 1
game_name = 'bird_dqn_Keras'  # the name of the game being played for log files
CONFIG = 'nothreshold'
action_size = 2               # number of valid actions

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

model_path = "save_model/" + game_name
graph_path = "save_graph/" + game_name

# Make folder for save data
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(graph_path):
    os.makedirs(graph_path)
    
class DQN_agent:
    def __init__(self):
        
        # get size of state and action
        self.progress = " "
        self.action_size = action_size
        
        # train time define
        self.training_time = 5*60
        
        # These are hyper parameters for the DQN
        self.learning_rate = 0.0001
        self.discount_factor = 0.99
        
        self.epsilon_max = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.epsilon = self.epsilon_max
        
        self.ep_trial_step = 2000
        
        # Parameter for Experience Replay
        self.size_replay_memory = 5000
        self.batch_size = 64
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)

        # Parameter for Target Network
        self.target_update_cycle = 200

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.Copy_Weights()
        
    def build_model(self):
        print("Now we build the model")
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        
        model.compile(loss='mse',optimizer = Adam(lr = self.learning_rate))
        print("We finish building the model")
        return model

    # after some time interval update the target model to be same with model
    def Copy_Weights(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        
        # choose an action epsilon greedily
        Q_value = self.model.predict(state)       #input a stack of 4 images, get the prediction
        action = np.zeros([self.action_size])
        action_index = 0

        # if time_step % FRAME_PER_ACTION == 0:
        if random.random() <= self.epsilon:
            print("----------Random Action----------")
            action_index = random.randrange(self.action_size)
            action[action_index] = 1
        else:
            action_index = np.argmax(Q_value)
            action[action_index] = 1
            
        return action, action_index

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))
        
        while len(self.memory) > self.size_replay_memory:
            self.memory.popleft()
    
def main():
    agent = DQN_agent()
    
    # store the previous observations in replay memory
    # memory = deque()

    # saving and loading networks
    if os.path.isfile(model_path+"/model.h5"):
        agent.model.load_weights(model_path+"/model.h5")
        
        if os.path.isfile(model_path + '/append_memory.pickle'):                        
            with open(model_path + '/append_memory.pickle', 'rb') as f:
                agent.memory = pickle.load(f)

            with open(model_path + '/epsilon_episode.pickle', 'rb') as ggg:
                agent.epsilon, episode = pickle.load(ggg)
                agent.epsilon = 0.001

            print("\n\n Successfully loaded \n\n")
    else:
        agent.epsilon = epsilon_max
        print("\n\n Could not find old network weights")

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(action_size)
    do_nothing[0] = 1
    
    state, reward, done = game_state.frame_step(do_nothing)
    
    state = skimage.color.rgb2gray(state)
    state = skimage.transform.resize(state,(80,80))
    state = skimage.exposure.rescale_intensity(state,out_range=(0,255))
    state = state / 255.0
    
    stacked_state = np.stack((state, state, state, state), axis=2)

    # start training

    # In Keras, need to reshape
    stacked_state = stacked_state.reshape(1, stacked_state.shape[0], stacked_state.shape[1], stacked_state.shape[2])  #1*80*80*4
        
    time_step = 0
    episode = 0
    start_time = time.time()
    
    agent.Copy_Weights()
    
    while time.time() - start_time < 5*60:        
        if len(agent.memory) < agent.size_replay_memory:
            agent.progress = "Exploration"            
        else:
            agent.progress = "Training" 
            
        loss = 0
        q_value_next = 0
        reward = 0
        
        ep_step = 0
        done = False
        while not done and ep_step < agent.ep_trial_step:
            ep_step += 1
            time_step += 1
            
            action, action_index = agent.get_action(stacked_state)

            # run the selected action and observe next state and reward
            next_state, reward, done = game_state.frame_step(action)
            
            next_state = skimage.color.rgb2gray(next_state)
            next_state = skimage.transform.resize(next_state,(80,80))
            next_state = skimage.exposure.rescale_intensity(next_state, out_range=(0, 255))
            next_state = next_state / 255.0
            next_state = next_state.reshape(1, next_state.shape[0], next_state.shape[1], 1) #1x80x80x1

            stacked_next_state = np.append(next_state, stacked_state[:, :, :, :3], axis=3)

            # store the transition in memory
            # memory.append((stacked_state, action_index, reward, stacked_next_state, done))
            agent.append_sample(stacked_state, action_index, reward, stacked_next_state, done)
            
            # update the old values
            stacked_state = stacked_next_state
            
            # only train if done observing
            if agent.progress == "Training":
                # agent.train_model()
                # if done or ep_step % agent.target_update_cycle == 0:
                
                # sample a minibatch to train on
                minibatch = random.sample(agent.memory, batch_size)

                #Now we do the experience replay
                states, actions, rewards, next_states, dones = zip(*minibatch)
                states      = np.concatenate(states)
                next_states = np.concatenate(next_states)
                
                q_value      = agent.model.predict(states)
                q_value_next = agent.model.predict(next_states)
                tgt_q_value_next = agent.target_model.predict(next_states)
                
                q_value[range(batch_size), actions] = rewards + discount_factor*np.max(tgt_q_value_next, axis=1)*np.invert(dones)

                loss += agent.model.train_on_batch(states, q_value)
                
                # scale down epsilon
                if agent.epsilon > epsilon_min:
                    agent.epsilon -= epsilon_decay
                
                if done or ep_step % agent.target_update_cycle == 0:
                    # return# copy q_net --> target_net
                    agent.Copy_Weights()
                    
            if done or ep_step == agent.ep_trial_step:
                if agent.progress == "Training":
                    episode += 1
                print("Episode :{:>5}".format(episode), "/ Episode step :{:>4}".format(ep_step), "/ Progress :", agent.progress, \
                      "/ Epsilon :{:>2.6f}".format(agent.epsilon), "/ Memory size :{:>5}".format(len(agent.memory)))
                break
            
    agent.model.save_weights(model_path+"/model.h5")
    # with open(model_path+"/model.json", "w") as outfile:
    #     json.dump(agent.model.to_json(), outfile)
    
    with open(model_path + '/append_memory.pickle', 'wb') as f:
        pickle.dump(agent.memory, f)
        
    save_object = (agent.epsilon, episode) 
    with open(model_path + '/epsilon_episode.pickle', 'wb') as ggg:
        pickle.dump(save_object, ggg)
    print("\n\n Now we save model \n\n")
    sys.exit()

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()
