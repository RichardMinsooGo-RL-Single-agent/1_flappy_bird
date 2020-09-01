# This is parameter setting for all deep learning algorithms
# import sys
# Import games
# sys.path.append("DQN_GAMES/")

training_time = 5*60 # int(10*60*60/36)   # seconds, 1800 is 30 min * 60 sec (540*60)/(4*5)

discount_factor= 0.99
Learning_rate = 0.001
Epsilon = 1
Final_epsilon = 0.00001

Num_action = 2

size_replay_memory = 10000
batch_size = 32

target_update_cycle = 100

first_conv   = [8,8,4,32]
second_conv  = [4,4,32,64]
third_conv   = [3,3,64,64]
first_dense  = [1600, 512]
second_dense = [512, Num_action]

state_dense  = [1600, 512]
state_output = [512, Num_action]

action_dense  = [1600, 512]
action_output = [512, Num_action]
