from actor import ActorNetwork
from critic import CriticNetwork
import numpy as np
from numpy.random import choice
import random
from replay_buffer import ExperienceReplay
from collections import namedtuple, deque


class Agent:
    # Reinforcement learning agent that learns using Actor Critic Network
    def __init__(self, state_size, batch_size, is_eval=False):
        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # sit, buy, sell
        self.memory_size = 1000000  # Replay memory size
        self.batch_size = batch_size
        self.replay_memory = ExperienceReplay(self.memory_size, self.batch_size)
        self.inventory = []
        self.is_eval = is_eval  # Whether or not Training is ongoing
        self.gamma = 0.99  # Discount factor in Bellman equation

        # Actor Policy model mapping states to actions
        self.actor = ActorNetwork(self.state_size, self.action_size)  # Instantiates the Actor networks

        # Critic(Value) Model that maps state action pairs to Q values.

        self.critic = CriticNetwork(self.state_size, self.action_size)  # Instantiate the critic model

    # returns an action given a state using the Actor(policy network)
    def select_action(self, state):

        action_logits = self.actor.model.predict(
            state)  # output of the softmax layer of the actor network returning probabilities for each action
        self.last_state = state
        if not self.is_eval:
            return choice(range(3), p=action_logits[
                0])  # Returns a stochastic policy based on action probabilities in training mode and a deterministic action corresponding to the maximum probability during testing

        return np.argmax(action_logits[0])

    # Set of actions to be carried out by the agent st every step in the episode
    def step(self, action, reward, next_state, done):
        self.replay_memory.add(self.last_state, action, reward, next_state, done)  # Adds new experience to memory
        if len(
                self.replay_memory) > self.batch_size:  # Asserts that enough experiences are present in mempry to train
            experiences = self.replay_memory.sample(
                self.batch_size)  # Samples a random batch from memory to train
            self.learn(experiences)  # Learns from the sampled experiences

            self.last_state = next_state  # Updates the state to the next state

    # Learning from the sampled experiences through the Actor and the critic
    def learn(self, experiences):
        states = np.stack([e.state for e in experiences if e is not None], 0).astype(np.float32).reshape(-1,
                                                                                                         self.state_size)
        actions = np.stack([e.action for e in experiences if e is not None], 0).astype(np.float32).reshape(-1,
                                                                                                           self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        next_states = np.stack([e.next_state for e in experiences if e is not None], 0).astype(np.float32).reshape(-1,
                                                                                                                   self.state_size)  # Return seperate arrays for each experience replay component

        next_actions = self.actor.model.predict_on_batch(next_states)  # Predict Action based on the next state
        Q_next = self.critic.model.predict_on_batch(
            [next_states, next_actions])  # Predict the Q value of the actor output for the the next state

        Q_targets = rewards + self.gamma * Q_next * (
                    1 - dones)  # Target Q value to serve as label for the Critic network based on Temporal difference
        self.critic.model.train_on_batch(x=[states, actions],
                                         y=Q_targets)  # Fit the critic model to the Temporal difference target

        # Train actor model(local)
        action_gradients = np.reshape(self.critic.get_action_gradients([states, actions]), (-1,
                                                                                            self.action_size))  # gradient of the Critic network output w.r.t the action probabilities fed from the actor network

        self.actor.custom_train([states, action_gradients])  # Custom training function
        # print("Learning")







