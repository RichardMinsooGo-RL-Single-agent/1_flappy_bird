#!/usr/bin/env python
from __future__ import print_function

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

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
# from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Lambda, Input, Add, Subtract
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

FRAME_PER_ACTION = 1
# Import game
sys.path.append("game/")
import wrapped_flappy_bird as game

game_name = '05_bird_dueling_Keras_b'  # the name of the game being played for log files
action_size = 2               # number of valid actions

model_path = "save_model/" + game_name
graph_path = "save_graph/" + game_name

# Make folder for save data
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(graph_path):
    os.makedirs(graph_path)
    
class DuelingDQN:
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
        
        # Parameter for Target Network
        self.target_update_cycle = 200
        
        # Parameters for network
        self.img_rows , self.img_cols = 80, 80
        self.img_channels = 4 #We stack 4 frames

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        
    def build_model(self):
        print("Now we build the model")
        state = Input(shape=(self.img_rows,self.img_cols,self.img_channels))
        net1 = Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(self.img_rows,self.img_cols,self.img_channels))(state)  #80*80*4
        net2 = Activation('relu')(net1)
        net3 = Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same')(net2)
        net4 = Activation('relu')(net3)
        net5 = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same')(net4)
        net6 = Activation('relu')(net5)
        net7 = Flatten()(net6)
        net8 = Dense(512)(net7)
        net9 = Activation('relu')(net8)
        
        state_layer_1 = Dense(512)(net9)
        action_layer_1 = Dense(512)(net9)

        v = Dense(1, activation='linear', kernel_initializer='he_uniform')(state_layer_1)
        v = Lambda(lambda v: tf.tile(v, [1, self.action_size]))(v)
        a = Dense(self.action_size, activation='linear', kernel_initializer='he_uniform')(action_layer_1)
        a = Lambda(lambda a: a - tf.reduce_mean(a, axis=-1, keep_dims=True))(a)
        tgt_output = Add()([v, a])
        # model = Model(inputs = state, outputs = q)
        
        model = Model(inputs=state, outputs=tgt_output)
        model.compile(loss='mse',optimizer = Adam(lr = self.learning_rate))
        
        model.summary()
        
        return model

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

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))
        
        while len(self.memory) > self.size_replay_memory:
            self.memory.popleft()
            
    # after some time interval update the target model to be same with model
    def Copy_Weights(self):
        self.target_model.set_weights(self.model.get_weights())
            
        # print(" Weights are copied!!")

    def save_model(self):
        # Save the variables to disk.
        self.model.save_weights(model_path+"/model.h5")
        save_object = (self.epsilon, self.episode, self.step)
        with open(model_path + '/epsilon_episode.pickle', 'wb') as ggg:
            pickle.dump(save_object, ggg)

        print("\n Model saved in file: %s" % model_path)

def main():
    
    agent = DuelingDQN()
    
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
    
    """
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
    # In Keras, need to reshape
    stacked_state = stacked_state.reshape(1, stacked_state.shape[0], stacked_state.shape[1], stacked_state.shape[2])  #1*80*80*4        
    """
    
    avg_score = 0
    episodes, scores = [], []
    
    # start training    
    # Step 3.2: run the game
    display_time = datetime.datetime.now()
    print("\n\n",game_name, "-game start at :",display_time,"\n")
    start_time = time.time()
    
    # Initialize target network.
    agent.Copy_Weights()
    
    while time.time() - start_time < agent.training_time:

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
        # In Keras, need to reshape
        stacked_state = stacked_state.reshape(1, stacked_state.shape[0], stacked_state.shape[1], stacked_state.shape[2])  #1*80*80*4        
        
        done = False
        agent.score = 0
        
        loss = 0
        q_value_next = 0
        reward = 0
        
        ep_step = 0
        
        while not done and ep_step < agent.ep_trial_step:
            if len(agent.memory) < agent.size_replay_memory:
                agent.progress = "Exploration"            
            else:
                agent.progress = "Training"

            ep_step += 1
            agent.step += 1

            # Select action
            action_arr, action = agent.get_action(stacked_state)

            # run the selected action and observe next state and reward
            next_state, reward, done = game_state.frame_step(action_arr)
            
            next_state = skimage.color.rgb2gray(next_state)
            next_state = skimage.transform.resize(next_state,(80,80))
            next_state = skimage.exposure.rescale_intensity(next_state, out_range=(0, 255))
            next_state = next_state / 255.0
            next_state = next_state.reshape(1, agent.img_rows , agent.img_cols, 1)
            
            stacked_next_state = np.append(next_state, stacked_state[:, :, :, :3], axis=3)

            # store the transition in memory
            agent.append_sample(stacked_state, action, reward, stacked_next_state, done)
            
            # update the old values
            stacked_state = stacked_next_state
            # only train if done observing
            if agent.progress == "Training":
                # agent.train_model()
                # sample a minibatch to train on
                minibatch = random.sample(agent.memory, agent.batch_size)

                #Now we do the experience replay
                states, actions, rewards, next_states, dones = zip(*minibatch)
                states      = np.concatenate(states)
                next_states = np.concatenate(next_states)
                y_array     = agent.model.predict(states)
                q_value_next = agent.model.predict(next_states)            
                tgt_q_value_next = agent.target_model.predict(next_states)
                
                y_array[range(agent.batch_size), actions] = rewards + agent.discount_factor*np.max(tgt_q_value_next, axis=1)*np.invert(dones)

                loss += agent.model.train_on_batch(states, y_array)
                
                # scale down epsilon
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon -= agent.epsilon_decay
                else:
                    agent.epsilon = agent.epsilon_min
                
                if done or ep_step % agent.target_update_cycle == 0:
                    # return# copy q_net --> target_net
                    agent.Copy_Weights()
                    
            agent.score += reward
            
            if done or ep_step == agent.ep_trial_step:
                if agent.progress == "Training":
                    agent.episode += 1
                    scores.append(agent.score)
                    episodes.append(agent.episode)
                    avg_score = np.mean(scores[-min(30, len(scores)):])
                print('episode :{:>6,d}'.format(agent.episode),'/ ep step :{:>5,d}'.format(ep_step), \
                      '/ time step :{:>7,d}'.format(agent.step),'/ status :', agent.progress, \
                      '/ epsilon :{:>1.4f}'.format(agent.epsilon),'/ last 30 avg :{:> 4.1f}'.format(avg_score) )
                break
    # Save model
    agent.save_model()
    
    pylab.plot(episodes, scores, 'b')
    pylab.savefig("./save_graph/flappybird_duelingdqn.png")

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
