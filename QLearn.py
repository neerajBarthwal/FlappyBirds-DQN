import sys
import numpy as np
sys.path.append("game/")
from game import wrapped_flappy_bird as game
from collections import deque 
from skimage import transform, color, exposure
import random

# Hyperparameters
GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.5 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-6


class Qlearn:
    
    """
        This class trains the 'input model' to play Flappy Bird.
    """
    
    def __init__(self,active_network, target_network):
        self.active_network = active_network
        self.target_network = target_network
        self.replay_memory = deque()
        self.epsilon = INITIAL_EPSILON
        self.timestep = 0
        self.actions = ACTIONS
        self.update_target_network()
    
    def update_target_network(self):
        self.target_network.set_weights(self.active_network.get_weights())
        
    def scale_down_image(self,img):
        
        img = color.rgb2gray(img)
        img = transform.resize(img,(84,84,1))
        img = exposure.rescale_intensity(img,out_range=(0,255))
        img = img / 255.0
        return img
    
    def re_shape(self,s_t):
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
        return s_t
    
    def pre_process_state(self, x_t):
        
        down_scaled_xt = self.scale_down_image(x_t)
        s_t = np.stack((down_scaled_xt, down_scaled_xt, down_scaled_xt, down_scaled_xt), axis=2)
        #In Keras, need to reshape
        s_t = self.re_shape(s_t)  #1*80*80*4 ????
        return s_t
    
    def experience_env(self,next_state, action_index, reward, terminal):

        new_state = np.append(next_state, self.current_state[:,:,:,:3], axis=3)
        self.replay_memory.append((self.current_state, action_index, reward, new_state,terminal))
        
        if len(self.replay_memory) > OBSERVATION:
            self.replay_memory.popleft()
        
        if self.timestep > OBSERVATION:
            self.train_network()
            
        self.current_state = new_state
        self.timestep+=1
        
        #print info
        state = ""
        
        if self.timestep <= OBSERVATION:
            state = "Observe"
        elif self.timestep > OBSERVATION and self.timestep < OBSERVATION + EXPLORE:
            state = "Explore"
        else:
            state = "Train"
        
        print("TIMESTEP: ", self.timestep, "STATE: ", state, "EPSILON: ", self.epsilon, "REWARD: ",reward)
    
    def train_network(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_memory,BATCH_SIZE)
        
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]
        #terminal_batch = [data[4] for data in minibatch]
        
        state_batch = np.concatenate(state_batch)
        #targets = self.model.predict(state_batch)
        
        nextState_batch = np.concatenate(nextState_batch)
        
        QValue_batch = self.target_network.predict(nextState_batch)
        
        loss = 0
        targets = np.zeros((state_batch.shape[0], ACTIONS))
        #print(targets.shape)
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                targets[i,action_batch[i]] = reward_batch[i]
            else:
                targets[i,action_batch[i]] = reward_batch[i]+ (GAMMA*np.max(QValue_batch[i]))
        #print(targets)
        #print(targets.shape)
        #targets[range(BATCH_SIZE), action_batch] = reward_batch + GAMMA*np.max(QValue_batch, axis=1)*np.invert(terminal_batch)
        
        loss+=self.active_network.train_on_batch(state_batch,targets)
        print("LOSS: ",loss)
        
        
        if self.timestep % 10000 == 0:
            self.update_target_network()
            #save model and its weights after every 10,000 steps
            model_json = self.active_network.to_json()
            with open("model.json","w") as json_file:
                json_file.write(model_json)
            print("Active weights: ",self.active_network.get_weights())
            print("Target Weights: ", self.target_network.get_weights())
            self.active_network.save_weights("active_model.h5",overwrite=True)
            self.target_network.save_weights("target_model.h5",overwrite=True)
            print("Model saved")
        
                
        
        # Step 2: calculate y 
#         y_batch = []
#         print(len(nextState_batch))
#         nextState_batch = np.concatenate(nextState_batch)
#         print(nextState_batch.shape)
#         QValue_batch = self.model.predict(nextState_batch)
#         loss = 0
#         for i in range(0,BATCH_SIZE):
#             terminal = minibatch[i][4]
#             
#             if terminal:
#                 y_batch.append(reward_batch[i])
#             else:
#                 y_batch.append(reward_batch[i]+GAMMA*np.max(QValue_batch[i]))
#         print(len(y_batch))
#         y_batch = np.concatenate((y_batch))
#         print(y_batch.shape)
        #loss += self.model.train_on_batch(nextState_batch,y_batch)
        
        
        
    def get_action(self):
        #print(self.current_state.shape)
        q_value = self.active_network.predict(self.current_state)
        action = np.zeros(self.actions)
        action_index = 0
        
        if self.timestep %FRAME_PER_ACTION==0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index]= 1
            else:
                action_index = np.argmax(q_value)
                action[action_index] = 1
        else:
            print("***********************************************************************************88")
            action[0] = 1
            
        if self.epsilon > FINAL_EPSILON and self.timestep > OBSERVATION:
            self.epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EXPLORE
        return action, action_index
                
        
    def play_game(self):
        """
            This method trains the model to flappy birds.
            
            TODO: Insert more docs
        """
        
        #1. open up a game state to communicate with emulator
        flappy_bird = game.GameState()
        
        # get the first state by doing nothing.
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        
        x_t, r_0, terminal = flappy_bird.frame_step(do_nothing)
        #run the selected action and observed next state and reward
        self.current_state = self.pre_process_state(x_t)
        
        while True:
            action, action_index = self.get_action()
            next_state, reward, terminal = flappy_bird.frame_step(action)
            next_state = self.scale_down_image(next_state)
            next_state = next_state.reshape(1, next_state.shape[0], next_state.shape[1],1) #1x84x84x1
            #print(type(next_state))
            self.experience_env(next_state, action_index, reward, terminal)
            #print('hi')
    