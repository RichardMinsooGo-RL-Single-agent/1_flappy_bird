#!/usr/bin/env python
from __future__ import print_function
# Import modules
import cv2
import tensorflow as tf
import os.path
import random
import numpy as np
import time, datetime
from collections import deque
import pylab
import sys
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import json
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

FRAME_PER_ACTION = 1
# Import game
sys.path.append("game/")
import wrapped_flappy_bird as game

game_name = '01_bird_Q_net_Keras_a'  # the name of the game being played for log files
action_size = 2               # number of valid actions

model_path = "save_model/" + game_name
graph_path = "save_graph/" + game_name

# Make folder for save data
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(graph_path):
    os.makedirs(graph_path)
    
class DQN_agent:
    def __init__(self):

        # Get parameters
        # get size of state and action
        self.progress = " "
        
        self.action_size = action_size
        
        # train time define
        self.training_time = 20*60
        
        # These are hyper parameters for the DQN
        self.learning_rate = 0.0001
        self.discount_factor = 0.99
        
        self.epsilon_max = 1.0
        # final value of epsilon
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.0001
        self.epsilon = self.epsilon_max
        
        self.step = 0
        self.score = 0
        self.episode = 0
        
        self.ep_trial_step = 2000
        
        # parameters for skipping and stacking
        # Parameter for Experience Replay
        self.size_replay_memory = 5000
        self.batch_size = 64
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # Parameters for network
        self.img_rows , self.img_cols = 80, 80
        self.img_channels = 4 #We stack 4 frames

        # create main model and target model
        self.model = self.build_model('network')
            
    def reset_env(self, game_state):
        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(action_size)
        do_nothing[0] = 1

        state, reward, done = game_state.frame_step(do_nothing)
        
        state = cv2.resize(state, (self.img_rows, self.img_cols))
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        
        ret, state = cv2.threshold(state,1,255,cv2.THRESH_BINARY)
        stacked_state = np.stack((state, state, state, state), axis = 2)
        return stacked_state        
    
    # Resize and make input as grayscale
    def preprocess(self, state):
        state = cv2.resize(state, (self.img_rows, self.img_cols))
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        
        ret, state_out = cv2.threshold(state,1,255,cv2.THRESH_BINARY)
        state_out = np.reshape(state_out, (self.img_rows, self.img_cols, 1))
        return state_out

    def build_model(self, network_name):
        print("Now we build the model")
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(self.img_rows,self.img_cols,self.img_channels)))  #80*80*4
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # For Q-net or sarsa there is only one batch
    def train_model(self, state, action, reward, next_state, next_action, done):
        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        
        # like Q Learning, get maximum Q value at s'
        # But from target model
        if done:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor * self.model.predict(next_state)[0][next_action])

        target = np.reshape(target, [1, self.action_size])
        
        # Decrease epsilon while training
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else :
            self.epsilon = self.epsilon_min
            
        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(state, target, epochs=1, verbose=0)
        
    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        # choose an action epsilon greedily
        action_arr = np.zeros([self.action_size])
        action = 0
        
        if random.random() < self.epsilon:
            # print("----------Random Action----------")
            action = random.randrange(self.action_size)
            action_arr[action] = 1
        else:
            # Predict the reward value based on the given state
            Q_value = self.model.predict(state)       #input a stack of 4 images, get the prediction
            action = np.argmax(Q_value)
            action_arr[action] = 1
            
        return action_arr, action

    def save_model(self):
        # Save the variables to disk.
        self.model.save_weights(model_path+"/model.h5")
        save_object = (self.epsilon, self.episode, self.step)
        with open(model_path + '/epsilon_episode.pickle', 'wb') as ggg:
            pickle.dump(save_object, ggg)

        print("\n Model saved in file: %s" % model_path)

def main():
    
    agent = DQN_agent()
    
    # Initialize variables
    # Load the file if the saved file exists
    if os.path.isfile(model_path+"/model.h5"):
        agent.model.load_weights(model_path+"/model.h5")
        if os.path.isfile(model_path + '/epsilon_episode.pickle'):
            
            with open(model_path + '/epsilon_episode.pickle', 'rb') as ggg:
                agent.epsilon, agent.episode, agent.step = pickle.load(ggg)
            
        print('\n\n Variables are restored!')

    else:
        print('\n\n Variables are initialized!')
        agent.epsilon = agent.epsilon_max
    
    # open up a game state to communicate with emulator
    game_state = game.GameState()
    # stacked_state = agent.reset_env(game_state)
    # In Keras, need to reshape
    # stacked_state = stacked_state.reshape(1, stacked_state.shape[0], stacked_state.shape[1], stacked_state.shape[2])  #1*80*80*4        
    
    avg_score = 0
    episodes, scores = [], []
    
    # start training    
    # Step 3.2: run the game
    display_time = datetime.datetime.now()
    print("\n\n",game_name, "-game start at :",display_time,"\n")
    start_time = time.time()
    
    while time.time() - start_time < agent.training_time:

        stacked_state = agent.reset_env(game_state)
        # In Keras, need to reshape
        stacked_state = stacked_state.reshape(1, stacked_state.shape[0], stacked_state.shape[1], stacked_state.shape[2])  #1*80*80*4        
        
        done = False
        agent.score = 0
        # loss = 0
        # q_value_next = 0
        # reward = 0
        
        ep_step = 0
        
        while not done and ep_step < agent.ep_trial_step:
            ep_step += 1
            agent.step += 1

            # Select action
            action_arr, action = agent.get_action(stacked_state)

            # run the selected action and observe next state and reward
            next_state, reward, done = game_state.frame_step(action_arr)
            
            next_state = agent.preprocess(next_state)
            next_state = next_state.reshape(1, agent.img_rows , agent.img_cols, 1)
            
            stacked_next_state = np.append(next_state, stacked_state[:, :, :, :3], axis=3)

            next_action_arr, next_action = agent.get_action(stacked_next_state)
            
            agent.train_model(stacked_state, action, reward, stacked_next_state, next_action, done)
            
            # update the old values
            stacked_state = stacked_next_state

            agent.score += reward
            
            if done or ep_step == agent.ep_trial_step:
                agent.episode += 1
                scores.append(agent.score)
                episodes.append(agent.episode)
                avg_score = np.mean(scores[-min(30, len(scores)):])
                print('episode :{:>6,d}'.format(agent.episode),'/ ep step :{:>5,d}'.format(ep_step), \
                      '/ time step :{:>7,d}'.format(agent.step), \
                      '/ epsilon :{:>1.4f}'.format(agent.epsilon),'/ last 30 avg :{:> 4.1f}'.format(avg_score) )
                break
    # Save model
    agent.save_model()
    
    pylab.plot(episodes, scores, 'b')
    pylab.savefig("./save_graph/flappybird_Q_net.png")

    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()
