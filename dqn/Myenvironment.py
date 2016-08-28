import cv2
import gym
import random
import numpy as np

class myENV:
  def __init__(self,_STATENUM):
    self.STATENUM = _STATENUM
    self.pos = 0
    self.stp = 0
    self.screen = np.zeros(shape=[self.STATENUM,1])
    self.terminal = 0
    self._reset = 0

  def reset(self):
    self.terminal = 0
    self.pos = 0
    self.stp = 0
    self._reset = 0

  def getScreen(self):
    self.screen = np.zeros(shape=[self.STATENUM,1])
    # self.screen[0:self.pos] = 1
    self.screen[self.pos] = 1
    return self.screen

  def game_over(self):
    self.terminal = False
    # if self.stp >= self.STATENUM + 9:
    #   self.terminal = 1
    if self.pos >= self.STATENUM -1:
      self.terminal = True
    return self.terminal

  def step(self,action):

    self.stp +=1
    if action == 0:
      self.pos += 1
      self.pos = min(self.pos, self.STATENUM - 1)
    elif action == 1:
      self.pos -=1
      self.pos = max(self.pos, 0)

    reward = 0
    if self.pos == self.STATENUM - 1 and self._reset == 0:
      reward = 1
      self._reset = 1
    return reward






class Environment(object):
  def __init__(self, config):
    #self.env = gym.make(config.env_name)
    self.env = myENV(config.screen_width)

    screen_width, screen_height, self.action_repeat, self.random_start = \
        config.screen_width, config.screen_height, config.action_repeat, config.random_start

    self.display = config.display
    self.screen_dim = [screen_width,  screen_height ]

    self._screen = None
    self.reward = 0
    self.terminal = True

  def reset_game(self):
    self.env.reset()

  def getScreenGrayscale(self):
    return self.env.getScreen()

  def game_over(self):
    return self.env.game_over()

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

  @property
  def screen(self):
    return self._screen
    #return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_RGB2GRAY)/255., self.dims)
    #return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_BGR2YCR_CB)/255., self.dims)[:,:,0]

  @property
  def action_size(self):
    return 2


  def lives(self):
    return 0

  def getMinimalActionSet(self):
    return [0,1]

  def getScreenDims(self):
    return self.screen_dim

  @property
  def state(self):
    return self.screen, self.reward, self.terminal

  def render(self):
    if self.display:
      self.env.render()

  def after_act(self, action):
    self.render()

class MyGymEnvironment(Environment):
  def __init__(self, config):
    super(MyGymEnvironment, self).__init__(config)

  def act(self, action, is_training=True):
    return self.env.step(action)



