# Import modules
import cv2
import tensorflow as tf
import os.path
import random
import numpy as np
import time, datetime
from collections import deque
import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Import game
import sys
sys.path.append("game/")

import Deep_Parameters
import wrapped_flappy_bird as game

# Hyper Parameters:
FRAME_PER_ACTION = 1
game_name = 'bird_TF_doubledqn_c'    # the name of the game being played for log files
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
        self.progress = " "
        
        # get size of state and action
        self.action_size = action_size
        
        # train time define
        self.training_time = 10*60
        
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

        self.first_conv   = Deep_Parameters.first_conv
        self.second_conv  = Deep_Parameters.second_conv
        self.third_conv   = Deep_Parameters.third_conv
        self.first_dense  = Deep_Parameters.first_dense
        self.second_dense = Deep_Parameters.second_dense

        # Initialize Network
        self.input, self.output = self.build_model('network')
        self.tgt_input, self.tgt_output = self.build_model('target')
        self.train_step, self.action_tgt, self.y_tgt, self.Loss = self.loss_and_train()
            
    def reset_env(self, game_state):
        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(action_size)
        do_nothing[0] = 1

        state, reward, done = game_state.frame_step(do_nothing)
        
        # state = self.preprocess(state)
        state = cv2.cvtColor(cv2.resize(state, (self.img_rows, self.img_cols)), cv2.COLOR_BGR2GRAY)
        
        ret, state = cv2.threshold(state,1,255,cv2.THRESH_BINARY)
        # agent.setInitState(state)
        stacked_state = np.stack((state, state, state, state), axis = 2)
        return stacked_state        
    
    # Resize and make input as grayscale
    def preprocess(self, state):
        state_out = cv2.resize(state, (self.img_rows, self.img_cols))
        state_out = cv2.cvtColor(state_out, cv2.COLOR_BGR2GRAY)
        ret, state_out = cv2.threshold(state_out,1,255,cv2.THRESH_BINARY)
        state_out = np.reshape(state_out, (self.img_rows, self.img_cols, 1))
        return state_out

    # Convolution and pooling
    def conv2d(self, x, w, stride):
        return tf.nn.conv2d(x,w,strides=[1, stride, stride, 1], padding='SAME')

    # Get Variables
    def conv_weight_variable(self, name, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def weight_variable(self, name, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, name, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    def build_model(self, network_name):
        # input layer
        x_image = tf.placeholder(tf.float32, shape = [None,
                                                      self.img_rows,
                                                      self.img_cols,
                                                      self.img_channels])
        # network weights
        with tf.variable_scope(network_name):
            # Convolution variables
            w_conv1 = self.conv_weight_variable('_w_conv1', self.first_conv)
            b_conv1 = self.bias_variable('_b_conv1',[self.first_conv[3]])

            w_conv2 = self.conv_weight_variable('_w_conv2',self.second_conv)
            b_conv2 = self.bias_variable('_b_conv2',[self.second_conv[3]])

            w_conv3 = self.conv_weight_variable('_w_conv3',self.third_conv)
            b_conv3 = self.bias_variable('_b_conv3',[self.third_conv[3]])

            # Densely connect layer variables
            w_fc1 = self.weight_variable('_w_fc1',self.first_dense)
            b_fc1 = self.bias_variable('_b_fc1',[self.first_dense[1]])

            w_fc2 = self.weight_variable('_w_fc2',self.second_dense)
            b_fc2 = self.bias_variable('_b_fc2',[self.second_dense[1]])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(x_image,w_conv1,4) + b_conv1)
        h_conv1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, w_conv2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, w_conv3, 1) + b_conv3)

        h_pool3_flat = tf.reshape(h_conv3, [-1, self.first_dense[0]])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1)+b_fc1)

        # Q Value layer
        output = tf.matmul(h_fc1, w_fc2) + b_fc2
        return x_image, output

    def loss_and_train(self):
        # Loss function and Train
        action_tgt = tf.placeholder(tf.float32, shape = [None, self.action_size])
        y_tgt = tf.placeholder(tf.float32, shape = [None])

        y_prediction = tf.reduce_sum(tf.multiply(self.output, action_tgt), reduction_indices = 1)
        Loss = tf.reduce_mean(tf.square(y_prediction - y_tgt))
        train_step = tf.train.AdamOptimizer(learning_rate = self.learning_rate, epsilon = 1e-02).minimize(Loss)

        return train_step, action_tgt, y_tgt, Loss

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        # sample a minibatch to train on
        minibatch = random.sample(self.memory, self.batch_size)

        # Save the each batch data
        states      = [batch[0] for batch in minibatch]
        actions     = [batch[1] for batch in minibatch]
        rewards     = [batch[2] for batch in minibatch]
        next_states = [batch[3] for batch in minibatch]
        dones       = [batch[4] for batch in minibatch]

        # Get target values
        y_array = []
        # Selecting actions
        q_value_next = self.output.eval(feed_dict = {self.input: next_states})
        tgt_q_value_next = self.tgt_output.eval(feed_dict = {self.tgt_input: next_states})

        for i in range(0,self.batch_size):
            done = minibatch[i][4]
            if done:
                y_array.append(rewards[i])
            else:
                # y_array.append(rewards[i] + self.discount_factor * np.max(tgt_q_value_next[i]))
                a = np.argmax(tgt_q_value_next[i])
                y_array.append(rewards[i] + self.discount_factor * q_value_next[i][a] )

        # Training!! 
        feed_dict = {self.action_tgt: actions, self.y_tgt: y_array, self.input: states}
        _, self.loss = self.sess.run([self.train_step, self.Loss], feed_dict = feed_dict)
        
        # Decrease epsilon while training
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else :
            self.epsilon = self.epsilon_min

    # get action from model using epsilon-greedy policy
    def get_action(self, stacked_state):
        # choose an action epsilon greedily
        action = np.zeros(self.action_size)
        action_index = 0
        
        if random.random() < self.epsilon:
            # print("----------Random Action----------")
            action_index = random.randrange(self.action_size)
            action[action_index] = 1
        else:
            Q_value = self.output.eval(feed_dict= {self.input:[stacked_state]})[0]
            action_index = np.argmax(Q_value)
            action[action_index] = 1
            
        return action, action_index

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))
        
        while len(self.memory) > self.size_replay_memory:
            self.memory.popleft()
            
    # after some time interval update the target model to be same with model
    def Copy_Weights(self):
        # Get trainable variables
        trainable_variables = tf.trainable_variables()
        # network variables
        src_vars = [var for var in trainable_variables if var.name.startswith('network')]

        # target variables
        dest_vars = [var for var in trainable_variables if var.name.startswith('target')]

        for i in range(len(src_vars)):
            self.sess.run(tf.assign(dest_vars[i], src_vars[i]))
            
        # print(" Weights are copied!!")

    def save_model(self):
        # Save the variables to disk.
        save_path = self.saver.save(self.sess, model_path + "/model.ckpt")
        save_object = (self.epsilon, self.episode, self.step)
        with open(model_path + '/epsilon_episode.pickle', 'wb') as ggg:
            pickle.dump(save_object, ggg)

        print("\n Model saved in file: %s" % model_path)

def main():
    
    agent = DQN_agent()
    
    # Initialize variables
    # Load the file if the saved file exists
    agent.sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    agent.saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        agent.saver.restore(agent.sess, ckpt.model_checkpoint_path)
        if os.path.isfile(model_path + '/epsilon_episode.pickle'):
            
            with open(model_path + '/epsilon_episode.pickle', 'rb') as ggg:
                agent.epsilon, agent.episode, agent.step = pickle.load(ggg)
            
        print('\n\n Variables are restored!')

    else:
        agent.sess.run(init)
        print('\n\n Variables are initialized!')
        agent.epsilon = agent.epsilon_max
    
    # open up a game state to communicate with emulator
    game_state = game.GameState()
    
    stacked_state = agent.reset_env(game_state)

    # start training    
    # Step 3.2: run the game
    display_time = datetime.datetime.now()
    print("\n\n",game_name, "-game start at :",display_time,"\n")
    start_time = time.time()
    
    # Initialize target network.
    agent.Copy_Weights()
    
    while time.time() - start_time < agent.training_time:

        done = False
        agent.score = 0
        ep_step = 0
        
        while not done and ep_step < agent.ep_trial_step:
            
            if len(agent.memory) < agent.size_replay_memory:
                agent.progress = "Exploration"            
            else:
                agent.progress = "Training"

            ep_step += 1
            agent.step += 1

            # Select action
            action, action_index = agent.get_action(stacked_state)

            # run the selected action and observe next state and reward
            next_state, reward, done = game_state.frame_step(action)
            next_state = agent.preprocess(next_state)
            stacked_next_state = np.append(stacked_state[:,:,1:],next_state,axis = 2)
            
            # store the transition in memory
            agent.append_sample(stacked_state, action, reward, stacked_next_state, done)
            
            # only train if done observing
            if agent.progress == "Training":
                # Training!
                agent.train_model()
                if done or agent.step % agent.target_update_cycle == 0:
                    # return# copy q_net --> target_net
                    agent.Copy_Weights()
                    
            # update the old values
            stacked_state = stacked_next_state
            agent.score += reward
            
            # If game is over (done)
            if done or ep_step == agent.ep_trial_step:
                if agent.progress == "Training":
                    agent.episode += 1
                print('episode :{:>7,d}'.format(agent.episode),'/ ep step :{:>6,d}'.format(ep_step), \
                      '/ time step :{:>10,d}'.format(agent.step),'/ progress :',agent.progress, \
                      '/ epsilon :{:>1.5f}'.format(agent.epsilon),'/ score :{:> 5f}'.format(agent.score) )
                break
    # Save model
    agent.save_model()

    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()

if __name__ == "__main__":
    main()