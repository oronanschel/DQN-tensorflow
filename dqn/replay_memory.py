"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import os
import random
import logging
import numpy as np

from utils import save_npy, load_npy

class ReplayMemory:
  def __init__(self, config, model_dir):
    self.model_dir = model_dir
    self.HEADSNUM = config.heads_num

    self.cnn_format = config.cnn_format
    self.memory_size = config.memory_size
    self.actions = np.empty(self.memory_size, dtype = np.uint8)
    self.rewards = np.empty(self.memory_size, dtype = np.integer)

    self.masks = np.empty((self.memory_size,self.HEADSNUM), dtype=np.float16)

    self.screens = np.empty((self.memory_size, config.screen_height,config.screen_width), dtype = np.float16)
    self.terminals = np.empty(self.memory_size, dtype = np.bool)
    self.history_length = config.history_length
    self.dims = (config.screen_height,config.screen_width)
    self.batch_size = config.batch_size
    self.count = 0
    self.top = 0
    self.bottom = 0

    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
    self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)


  def add(self, screen, reward, action, terminal,mask=0):
    assert screen.shape == self.dims
    # NB! screen is post-state, after action and reward
    self.actions[self.top] = action
    self.rewards[self.top] = reward
    self.screens[self.top, ...] = screen
    self.terminals[self.top] = terminal
    self.masks[self.top] = mask

    if self.count == self.memory_size:
      self.bottom = (self.bottom + 1) % self.memory_size
    else:
      self.count += 1

    self.top = (self.top + 1) % self.memory_size

  def getState(self, index):
    assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
    # # normalize index to expected range, allows negative indexes
    # index = index % self.count
    # if is not in the beginning of matrix
    # if index >= self.history_length - 1:
    #   # use faster slicing
    #   return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
    # else:
    #   # otherwise normalize indexes and use slower list based access
    #   indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
    #   return self.screens[indexes, ...]
    screens = self.screens[(index - (self.history_length - 1)):(index + 1), ...]
    # black = False
    # for i in range(index - 1, (index - self.history_length), -1):
    #   if self.terminals[i] == True:
    #     black = True
    #   if black == True:
    #     screens[i-(index-1)] = np.zeros(shape=screens[i-(index-1)].shape)

    return screens


  def sample(self,batch_size = 32):
    # memory must include poststate, prestate and history
    assert self.count > self.history_length
    # sample random indexes
    indexes = []

    states = np.empty((batch_size, self.history_length) + self.dims, dtype=np.float16)
    next_state = np.empty((batch_size, self.history_length) + self.dims, dtype=np.float16)
    actions = np.empty(batch_size,dtype=np.uint8)
    rewards = np.empty(batch_size,dtype=np.integer)
    terminals = np.empty(batch_size,dtype=np.bool)
    masks = np.empty((batch_size,self.HEADSNUM),dtype=np.float16)

    count = 0
    while count < batch_size:
      # find random index 
      # sample one index (ignore states wraping over
      index = random.randint(self.bottom,
                   self.bottom + self.count - self.history_length)

      init_ind = np.arange(index, index + self.history_length)
      trans_ind = init_ind + 1
      end_ind   = index +self.history_length - 1
      if np.any(self.terminals.take(init_ind[0:-1],mode='wrap')):
        continue

      states     [count, ...] = self.screens.take(init_ind, axis=0, mode='wrap')
      next_state [count, ...] = self.screens.take(trans_ind, axis=0, mode='wrap')
      actions    [count, ...] = self.actions.take(end_ind, mode='wrap')
      rewards    [count, ...] = self.rewards.take(end_ind, mode='wrap')
      terminals  [count, ...] = self.terminals.take(end_ind, mode='wrap')
      masks      [count, ...] = self.masks.take(end_ind, axis=0, mode='wrap')

      count+=1

      # NB! having index first is fastest in C-order matrices
      # self.prestates[len(indexes), ...] = self.getState(index - 1)
      # self.poststates[len(indexes), ...] = self.getState(index)
    #   indexes.append(index)
    #
    # actions = self.actions[indexes]
    # rewards = self.rewards[indexes]
    # terminals = self.terminals[indexes]
    # masks  = self.masks[indexes]

    if self.cnn_format == 'NHWC':
      return np.transpose(states, (0, 2, 3, 1)), actions, \
        rewards, np.transpose(next_state, (0, 2, 3, 1)), terminals, masks
    else:
      return states, actions, rewards, next_state, terminals, masks

  def save(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates','masks'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates,self.masks])):
      save_npy(array, os.path.join(self.model_dir, name))

  def load(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates','masks'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates,self.masks])):
      array = load_npy(os.path.join(self.model_dir, name))
