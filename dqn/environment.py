import cv2
import gym
import random
import numpy as np

class myENV:
  def __init__(self):
    self.STATENUM = 15
    self.pos = 0
    self.stp = 0
    self.screen = np.zeros(shape=[self.STATENUM,1])
    self.screen[self.pos] = 1
  def reset(self):
    self.pos = 0
    self.stp = 0
    self.screen = np.zeros(shape=[self.STATENUM,1])
    self.screen[self.pos] = 1
    return self.screen
  def step(self,action):

    self.stp +=1
    if action == 0:
      self.pos += 1
      self.pos = min(self.pos, self.STATENUM - 1)
    elif action == 1:
      self.pos -=1
      self.pos = max(self.pos, 0)

    reward = 0
    if self.pos == self.STATENUM - 1:
      reward = 1

    terminal = 0
    if self.stp == self.STATENUM  + 9:
      terminal = 1

    self.screen = np.zeros(shape=[self.STATENUM,1])
    self.screen[self.pos] = 1

    return self.screen , reward , terminal






class Environment(object):
  def __init__(self, config):
    #self.env = gym.make(config.env_name)
    self.env = myENV()

    screen_width, screen_height, self.action_repeat, self.random_start = \
        config.screen_width, config.screen_height, config.action_repeat, config.random_start

    self.display = config.display
    #self.dims = (screen_width, screen_height)

    self._screen = None
    self.reward = 0
    self.terminal = True

  def new_game(self, from_random_game=False):
    self._screen = self.env.reset()
    return self._screen, 0, 0,0

  def new_random_game(self):
    self._screen = self.env.reset()
    return self._screen, 0, 0,0

  def _step(self, action):
    self._screen, self.reward, self.terminal = self.env.step(action)

  def _random_step(self):
    action = self.env.action_space.sample()
    self._step(action)

  @ property
  def screen(self):
    return self._screen
    #return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_RGB2GRAY)/255., self.dims)
    #return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_BGR2YCR_CB)/255., self.dims)[:,:,0]

  @property
  def action_size(self):
    return 2

  @property
  def lives(self):
    return 0

  @property
  def state(self):
    return self.screen, self.reward, self.terminal

  def render(self):
    if self.display:
      self.env.render()

  def after_act(self, action):
    self.render()

class GymEnvironment(Environment):
  def __init__(self, config):
    super(GymEnvironment, self).__init__(config)

  def act(self, action, is_training=True):
    return self.env.step(action)



