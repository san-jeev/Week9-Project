# Week 9 Homework 
# Develop and train a Keras-RL Reinforcement learning agent called CartPole. 
# The CartPole environment consists of a pole, balanced on a cart. 
# The deep Q agent has to learn how to balance the pole vertically, while the cart underneath it moves. 
# The agent is given the position of the cart, the velocity of the cart, the angle of the pole, and the rotational rate of the pole as inputs. 
# The agent can apply a force on either side of the cart. If the pole falls more than 15 degrees from vertical, it’s game over for our agent.

# Import dependencies
import gym # OpenAI gym is a toolkit for developing and comparing reinforcement learning algorithms.
from gym import wrappers # gym wrapper call to save .mp4 files to ./videos/1234/cartpole.mp4
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils import model_to_dot
from keras.callbacks import Callback as KerasCallback
from PIL import Image
import pydot
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # ignore warnings
from time import time # to have timestamps in the CarPole video file

# !pip install keras-rl

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

print('\033[1m \033[4mWeek 9 HW:\033[0m' ' We will create a DQN agent using Keras to master the CartPole-v0 environment and take several hundred episodes to eventually balance the pole' '\n')

# Set Parameters, Environment Variables
print('\033[1m \033[4mConfigure appropriate CartPole-v0 Environment Variables to benchmark the ability of reinforcement learning agent\033[0m')
ENV_NAME = 'CartPole-v0'

# Get the environment and extract the number of actions available in the Cartpole problem
# Environments are intended to have various levels of difficulty. We are going to use CartPole-v0 environment to benchmark the ability of reinforcement learning agent to solve the CartPole problem
env = gym.make(ENV_NAME)
env_wrapper = wrappers.Monitor(env, './videos/' + str(time()) + '/')  # Insert a gym wrapper call after you make the env to save .mp4 files to ./videos/1234/something.mp4
print('\033[1m' 'env= ' '\033[0m' + str(env))
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
print('\033[1m' 'nb_actions= ' '\033[0m' + str(nb_actions) + '\n')

# First build a simple 3 hidden layers neural network model with 16 neurons each that will later learn through reinforcement learning and solve the Cart-pole problem
# The input is a 1 x state space vector and there will be an output neuron for each possible action that will predict the Q value of that action for each step. 
# By taking the argmax of the outputs, we can choose the action with the highest Q value 
print('\033[1m \033[4mFirst build simple 3 hidden layers neural network model with 16 neurons each\033[0m' '\n')
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='linear'))
print('\033[1m \033[4mModel Summary\033[0m' '\n')
print(model.summary())
print("")

print('\033[1m \033[4mModel Layer Output Shape\033[0m' '\n')
print(model.output)

for layer in model.layers:
    print(layer.output_shape)

print('\n')
print('Model Layer Output Shape layer.get_output_at(0).get_shape().as_list()')
for layer in model.layers:
    print(layer.get_output_at(0).get_shape().as_list())
    
print('\n')
print('Model Layer Output Shape  l.output_shape')
for l in model.layers:
    print (l.output_shape)
print('\n')
      
# Plot the Model and its layers
print('\033[1m \033[4mPlot the Model and its Layers\033[0m' '\n')
path = F"/content/gdrive/My Drive" 
plot_model(model, to_file='DQN-model.png')
model_img = Image.open("DQN-model.png")
plt.figure(1, figsize = (16 , 16))
plt.title('Model')
plt.imshow(model_img)
plt.show()
print("")

# Create a deep Q network Agent now that we have a model and memory and policy is defined 
print('\033[1m \033[4mCreate a deep Q network Agent\033[0m' '\n')

# Set policy as Epsilon Greedy and memory as Sequential Memory to store results of actions performed and rewards for each action
policy = EpsGreedyQPolicy() # greedy Q Policy is used to balance exploration and exploitation. 

# SequentialMemory is fast and efficient data structure to store agent’s experiences in; As new experiences are added to this memory and it becomes full, old experiences are forgotten.
memory = SequentialMemory(limit=50000, window_length=1) 

# nb_steps_warmup: Determines how long we wait before we start doing experience replay, which if you recall, is when we actually start training the network. This lets us build up enough experience to build a proper minibatch. If you choose a value for this parameter that’s smaller than your batch size, Keras RL will sample with a replacement.
# target_model_update: The Q function is recursive and when the agent updates it’s network for Q(s,a) that update also impacts the prediction it will make for Q(s’, a). This can make for a very unstable network. The way most deep Q network implementations address this limitation is by using a target network, which is a copy of the deep Q network that isn’t trained, but rather replaced with a fresh copy every so often. The target_model_update parameter controls how often this happens.
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)

# Compile the deep Q network Agent using Adam (adaptive learning rate optimization algorithm) that's been designed specifically for training deep neural networks. 
# Adam leverages the power of adaptive learning rates methods to find individual learning rates for each parameter.
print('\033[1m \033[4mCompile the deep Q network Agent\033[0m' '\n')
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Keras-RL provides several Keras-like callbacks that allow for convenient model checkpointing and logging. 
# We will use both of those callbacks below.
class Callback(KerasCallback):
    def _set_env(self, env):
        self.env = env
        
class ModelIntervalCheckpoint(Callback):
    def __init__(self, filepath, interval, verbose=0):
        super(ModelIntervalCheckpoint, self).__init__()
        self.filepath = filepath
        self.interval = interval
        self.verbose = verbose
        self.total_steps = 0
        
class FileLogger(Callback):
    def __init__(self, filepath, interval=None):
        self.filepath = filepath
        self.interval = interval

        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dict that maps from episode to metrics array.
        self.metrics = {}
        self.starts = {}
        self.data = {}
        
def build_callbacks(env_name):
    checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    return callbacks

print('\033[1m \033[4mUse Keras-RL callbacks for convenient model checkpointing and logging\033[0m' '\n')
callbacks = build_callbacks(ENV_NAME)

# Train the Reinforcement Learning model usinf .fit(); Create training data through the trials we run and feed this information into it directly after running the trial. 
# Rather than training on the trials as they come in, we add them to memory and train on a random sample of that memory.
# By taking a random sample, we don’t bias our training set, and instead ideally learn about scaling all environments we would encounter equally well.
print('\033[1m \033[4mTrain the deep Q network Agent for 5000 steps\033[0m' '\n')
dqn.fit(env, nb_steps=5000, visualize=True, verbose=2, callbacks=callbacks) # set visualize=True to watch the agent interact with the environment

print("")
print('\033[1m \033[4mNote:\033[0m' ' After the first 250 episodes, we see that the total rewards for the episode approach 200 and the episode steps also approach 200. This means that the agent has learned to balance the pole on the cart until the environment ends at a maximum of 200 steps.')
print("")

# Test the Reinforcement Learning model using .test() method to evaluate for some number of episodes 
print('\033[1m \033[4mTest the deep Q network Agent for 5 Episodes\033[0m' '\n')
dqn.test(env, nb_episodes=5, visualize=True) # set visualize=True to watch agent balance the pole
