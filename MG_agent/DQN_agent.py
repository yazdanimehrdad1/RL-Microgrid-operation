def file_read():
    print("DQN_agent file was read successfully")


import numpy as np
import pandas as pd
import random
from collections import namedtuple, deque

# from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

BUFFER_SIZE = 10 #int(1e5)  # replay buffer size
BATCH_SIZE = 4 #64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[22]:


# test_data = pd.read_csv('sample_data.csv')
# test_data.head(100)


# In[ ]:





# ## Defining the NN using Pytorch
# The following structure defines the feed forward 
# The backpropogation is done in the learn function
# 
# The number of hidden layers, activation functions, and output layer need to be studied further. 

# In[23]:


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = torch.sigmoid(self.fc1(state))
        x = torch.sigmoid(self.fc2(x))

        return self.fc3(x)




class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer (number of samples added to the memory)
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    
    def sample(self):
        experiences = random.sample(self.memory, k= self.batch_size)
        #print(len(experiences[0][0]))
        #print(len(experiences[1][0]))
        #print(len(experiences[2][0]))
        #print(len(experiences[3][0]))
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        #print("This is sample function in ReplayBuffer")
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)





class Agent():
    def __init__(self, state_size, action_size, seed):
    
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
    
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr= LR)


        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        
        # Save the experience in the replay memory
        # print("this is the action I'm adding to memory", action)
        self.memory.add(state, action, reward, next_state, done)
        
        
        # every UPDATE_EVERY learn from the experiences stored in the memory
        self.t_step = ( self.t_step +1) % UPDATE_EVERY
        if self.t_step ==0:
            #If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                print('\r Number of samples in memory: {}\t\t\t BATCH_SIZE: {}'.format(len(self.memory), BATCH_SIZE))
                experiences = self.memory.sample()
                # print(experiences)
                # print("Use these samples to learn using the QNetwork")
                self.learn(experiences,GAMMA)
                
    def act(self, state, eps):
        
        # Return action based on the current policy(epsilon greedy policy)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
#         print("State is ", state)
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)


        self.qnetwork_local.train()
        #print(action_values)
        
        # Epsilon-greedy action selection
        randomVal = random.random()
        print("randomVal", randomVal, " <=> ", "eps", eps)
        print("Suggested action values from NN", action_values)
        if randomVal > eps:
            #return np.argmax(action_values.cpu().data.numpy())
            return action_values
        else:
            #return random.choice(np.arange(self.action_size))
            return torch.randn(1,4)
        
    
    
    def learn(self, experiences, gamma=0.1):
        
        # experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        # gamma (float): discount factor

        #print("The experiences being used for agent learning are: ")
        #print(experiences)
        
        states, actions, rewards, next_states, dones = experiences
        # print("The V_stack of the state/next_state is: ",next_states.shape)
        # print("The V_stack of actions: ", actions.shape)


        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        print("Q_targets_next ",Q_targets_next)
        Q_targets = rewards+ (gamma * Q_targets_next*(1-dones))
        print("Q_target is: ", Q_targets)
        # print('actions',actions)
        # print('states',states)
        Q_expected = self.qnetwork_local(states)[1] #.gather(1,actions) need to take care of the indexting
        print("Q_expected is: ", Q_expected)

        # compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next*(1-dones))
        #
        #
        loss = F.mse_loss(Q_expected, Q_targets)
        print('loss : ', loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()