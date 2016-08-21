import os
import time
import datetime
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .plot_learning import plot
from .base import BaseModel
from .history import History
from .ops import linear, conv2d
from .replay_memory import ReplayMemory
from utils import get_time, save_pkl, load_pkl
from matplotlib import pyplot as plt
import scipy ,cv2

class Agent(BaseModel):
  def __init__(self, config, environment, sess):
    super(Agent, self).__init__(config)
    self.sess = sess
    self.weight_dir = 'weights'
    self.valid_size = config.valid_size

    # env
    self.lives = environment.lives()
    self.env = environment
    self.death_ends_episode = config.death_ends_episode
    self.noop_action = 0
    self.frame_skip = config.frame_skip
    self.action_size = len(self.env.getMinimalActionSet())
    self.screen_dims = self.env.getScreenDims()
    self.random_start = config.random_start
    self.legal_actions = self.env.getMinimalActionSet()

    # bootstrap
    self.p = self.config.p
    self.HEADSNUM = config.heads_num



    self.batch_size = config.batch_size
    self.eval_steps = config.eval_steps
    self.save_freq = config.save_freq


    self.history = History(self.config)
    self.memory = ReplayMemory(self.config, self.model_dir)

    self.avg_v = 0
    self.avg_loss = 0

    with tf.variable_scope('step'):
      self.step_op = tf.Variable(0, trainable=False, name='step')
      self.step_input = tf.placeholder('int32', None, name='step_input')
      self.step_assign_op = self.step_op.assign(self.step_input)

    self.create_results_file()
    self.build_dqn()


  def create_results_file(self,_folder_name = None):
    tempdir = os.path.join(os.getcwd(), "models")
    self.create_dir(tempdir)
    if _folder_name:
      folder_name = _folder_name
    else:
      folder_name = self.config.folder_name
      # folder_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    self.mydir = os.path.join(tempdir, folder_name)
    self.prog_file = os.path.join(self.mydir, 'training_progress.csv')
    folder_existed = self.create_dir(self.mydir)
    if folder_existed == False:
      data_file = open(self.prog_file, 'wb')
      data_file.write('avg_loss,avg_ep_reward,max_ep_reward,min_ep_reward,num_game,epsilon,'
                      'l1_grad_l1_norm,l2_grad_l1_norm,l3_grad_l1_norm,l4_grad_l1_norm,l1_l1_norm,l2_l1_norm,l3_l1_norm,l4_l1_norm'
                      ',lr,step,')
      for i in range(0,self.HEADSNUM):
        data_file.write('avg_v['+str(i)+'],')
      data_file.write('heads_num,')
      data_file.write('\n')
      data_file.close()

  def update_results(self, avg_loss, avg_v, avg_ep_reward, max_ep_reward, min_ep_reward, num_game,ep,
                     l1_norm,lr,step):
      fd = open(self.prog_file,'a')
      fd.write('%f,%f,%f,%f,%f,%f,' % ( avg_loss, avg_ep_reward, max_ep_reward, min_ep_reward, num_game, ep))
      fd.write('%f,%f,%f,%f,%f,%f,%f,%f,' % ( l1_norm[0],l1_norm[1],l1_norm[2],l1_norm[3],l1_norm[4],l1_norm[5],l1_norm[6],l1_norm[7] ))
      fd.write('%f,%d,' % (lr,step))
      for i in range(0, self.HEADSNUM):
        fd.write('%f,' % (avg_v[i]))
      fd.write('%d,' % (self.HEADSNUM))
      fd.write('\n')
      fd.close()
  def create_dir(self,p):
    try:
      os.makedirs(p)
      return False
    except OSError, e:
      if e.errno != 17:
        raise  # This was not a "directory exist" error..
      else:
        return True

  def get_observation(self):
    screen = self.env.getScreenGrayscale().reshape(self.screen_dims[1], self.screen_dims[0])
    resized = cv2.resize(screen, (self.screen_width, self.screen_height), interpolation=cv2.INTER_LINEAR)
    return resized

  def new_random_game(self):
    num_actions = np.random.randint(4, self.random_start)
    for i in range(num_actions):
      self.env.act(self.noop_action)
      self.history.add(self.get_observation())

  def act(self,action):
    reward = 0
    for i in range(self.frame_skip):
      reward += self.env.act(self.legal_actions[action])

    screen = self.get_observation()
    terminal = self.env.game_over()
    return screen, reward, terminal

  def train(self):
    start_step = self.step_op.eval()

    self.lives = self.env.lives()
    self.env.reset_game()
    self.new_random_game()

    self.current_head = 0
    for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
      # 1. predict
      action = self.predict(self.history.get(),self.current_head,is_training=True)
      # 2. act
      screen, reward, terminal = self.act(action)
      # 3. observe + learn
      mask = np.random.binomial(1, self.p, size=[self.HEADSNUM])
      self.observe(screen, reward, action, terminal,mask)

      if terminal:
        self.env.reset_game()
        self.new_random_game()
        self.current_head = np.random.randint(self.HEADSNUM)

      if self.step >= self.learn_start and self.step % self.eval_freq == 0:
        # save the model
        self.step_assign_op.eval({self.step_input: self.step + 1})
        self.save_model(self.step+1)

        num_game, ep_reward = 0, 0.
        ep_rewards, actions = [], []
        num_game = 0

        self.env.reset_game()
        self.new_random_game()
        for estep in range(0,self.eval_steps):
          # 1. predict
          action = self.predict(self.history.get(), self.current_head,test_ep=0.01,is_training=False)
          # 2. act
          screen, reward, terminal = self.act(action)
          self.history.add(screen)

          if terminal==True:
            self.env.reset_game()
            self.new_random_game()

            num_game += 1
            ep_rewards.append(ep_reward)
            ep_reward = 0.
            # replace playing head
          else:
            ep_reward += reward


        try:
          max_ep_reward = np.max(ep_rewards)
          min_ep_reward = np.min(ep_rewards)
          avg_ep_reward = np.mean(ep_rewards)
        except:
          max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

        # gradients, weights sample
        g_w_l1_norm = self.gradient_weights_l1_norm()

        lr = self.learning_rate_op.eval({self.learning_rate_step: self.step})

        ep_rec = (self.ep_end +max(0., (self.ep_start - self.ep_end)* (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))
        self.update_results(self.avg_loss, self.v_avg, avg_ep_reward, max_ep_reward, min_ep_reward, num_game,
                            ep_rec,g_w_l1_norm,lr,self.step)

        print '\navg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
              % (self.avg_loss, np.mean(self.v_avg), avg_ep_reward, max_ep_reward, min_ep_reward, num_game)

        plot(self.mydir,heads_num = self.HEADSNUM)

  def gradient_weights_l1_norm(self):
    if self.memory.count < self.history_length:
      return
    else:
      s_t, action, reward, s_t_plus_1, terminal ,mask = self.memory.sample(self.valid_size)

    q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})

    q_t_plus_1_slice = q_t_plus_1[:, [i for i in range(0, self.action_size)]]
    max_q_t_plus_1 = np.max(q_t_plus_1_slice, axis=1)
    target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward
    target_q_t = target_q_t.reshape(self.valid_size, 1)
    max_q_t_plus_1 = max_q_t_plus_1.reshape(self.valid_size, 1)
    for k in range(1, self.HEADSNUM):
      q_t_plus_1_slice = q_t_plus_1[:, [i for i in range(self.action_size * k, self.action_size * (k + 1))]]
      max_q_t_plus_1_slice = np.max(q_t_plus_1_slice, axis=1)
      target_q_t_slice = (1. - terminal) * self.discount * max_q_t_plus_1_slice + reward
      target_q_t_slice = target_q_t_slice.reshape(self.valid_size, 1)
      max_q_t_plus_1_slice = max_q_t_plus_1_slice.reshape(self.valid_size, 1)
      target_q_t = np.concatenate((target_q_t, target_q_t_slice), axis=1)
      max_q_t_plus_1 = np.concatenate((max_q_t_plus_1, max_q_t_plus_1_slice), axis=1)

    grads_weights_samp =  self.sess.run([x for x in self.grad_and_val_l1_norm], {
      self.target_q_t: target_q_t,
      self.action: action,
      self.s_t: s_t,
      self.learning_rate_step: self.step,
      self.mask: mask
    })

    avg_loss = self.sess.run([self.loss_valid], {
      self.target_q_t: target_q_t,
      self.action: action,
      self.s_t: s_t,
      self.learning_rate_step: self.step,
      self.mask: mask,
      self.max_q_t_plus_1 : max_q_t_plus_1
    })

    avg_v = self.sess.run([x for x in self.avg_v], {
      self.target_q_t: target_q_t,
      self.action: action,
      self.s_t: s_t,
      self.learning_rate_step: self.step,
      self.mask: mask,
      self.max_q_t_plus_1 : max_q_t_plus_1
    })



    self.avg_loss = avg_loss[0]

    #TODO: compute v avrage
    self.v_avg = avg_v



    return grads_weights_samp

  def predict(self, s_t, current_head, test_ep=None,is_training=True):

    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))
    if random.random() < ep:
      action = random.randrange(self.action_size)
    else:
      action = self.q_action.eval({self.s_t: [s_t], self.head: current_head,self.is_training:is_training})[0]

      #action = self.q_action.eval({self.s_t: [s_t]})[0]

    return action

  def observe(self, screen, reward, action, terminal,mask):
    reward = max(self.min_reward, min(self.max_reward, reward))

    new_lives = self.env.lives()
    if(self.death_ends_episode and new_lives < self.lives):
      terminal = True

    self.lives = new_lives


    self.history.add(screen)
    self.memory.add(screen, reward, action, terminal,mask)

    if self.step > self.learn_start:
      if self.step % self.train_frequency == 0:
        self.q_learning_mini_batch()

      if self.step % self.target_q_update_step == self.target_q_update_step - 1:
        self.update_target_q_network()

  def q_learning_mini_batch(self):
    if self.memory.count <= self.history_length:
      return
    else:
      s_t, action, reward, s_t_plus_1, terminal, mask = self.memory.sample()

    q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})

    # t - shape [32]
    terminal = np.array(terminal) + 0.

    q_t_plus_1_slice = q_t_plus_1[:, [i for i in range(0,self.action_size)]]
    max_q_t_plus_1 = np.max(q_t_plus_1_slice, axis=1)
    target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward
    target_q_t = target_q_t.reshape(self.batch_size,1)
    for k in range(1,self.HEADSNUM):
      q_t_plus_1_slice = q_t_plus_1[:, [i for i in range(self.action_size*k, self.action_size*(k+1))]]
      max_q_t_plus_1 =  np.max( q_t_plus_1_slice, axis=1)
      target_q_t_slice =  (1. - terminal) * self.discount * max_q_t_plus_1 + reward
      target_q_t_slice = target_q_t_slice.reshape(self.batch_size,1)
      target_q_t = np.concatenate(  (target_q_t,target_q_t_slice), axis=1)
      # TODO: Think about mixing the targets

    _ = self.sess.run([self.optim], {
    self.target_q_t: target_q_t,
    self.action: action,
    self.s_t: s_t,
    self.learning_rate_step: self.step,
    self.mask : mask,
    })

  def build_dqn(self):
    self.w = {}
    self.t_w = {}

    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    # training network
    with tf.variable_scope('prediction'):
      if self.cnn_format == 'NHWC':
        self.s_t = tf.placeholder('float32',
            [None, self.screen_width, self.screen_height, self.history_length], name='s_t')
      else:
        self.s_t = tf.placeholder('float32',
            [None, self.history_length, self.screen_width, self.screen_height], name='s_t')



      if self.config.ToyProblem:
        shape = self.s_t.get_shape().as_list()
        self.s_t_flat = tf.reshape(self.s_t, [-1, reduce(lambda x, y: x * y, shape[1:])])
        self.l3, self.w['l3_w'], self.w['l3_b'] = linear(self.s_t_flat, 128, activation_fn=activation_fn, name='l3')

      else:
        self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
            32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1')
        self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
            64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2')
        self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
            64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3')


      shape = self.l3.get_shape().as_list()
      self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
            linear(self.l3_flat, 512, activation_fn=activation_fn, name='value_hid')

        self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
            linear(self.l3_flat, 512, activation_fn=activation_fn, name='adv_hid')

        self.value, self.w['val_w_out'], self.w['val_w_b'] = \
          linear(self.value_hid, 1, name='value_out')

        self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
          linear(self.adv_hid, self.action_size, name='adv_out')

        # Average Dueling
        self.q = self.value + (self.advantage - 
          tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
      else:
        self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
        self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.action_size*self.HEADSNUM, name='q')

      # action selection policy
      self.head = tf.placeholder('int32', name='head')
      self.is_training = tf.placeholder(tf.bool, name = 'is_training')
      if self.config.test_policy == 'Ensemble':
        # test action
        q_heads_l_policy = [tf.slice(self.q, [0, k * self.action_size], [1, self.action_size]) for k in range(0,self.HEADSNUM)]
        self.tmp_q_sum = tf.accumulate_n(q_heads_l_policy)

        self.q_action_test = tf.argmax(self.tmp_q_sum, dimension=1)
        # train action
        tmp_q_slice = tf.slice(self.q, [0, self.head * self.action_size], [-1, self.action_size])
        self.q_action_train = tf.argmax(tmp_q_slice, dimension=1)
        # action node
        self.q_action = tf.select([self.is_training], self.q_action_train, self.q_action_test)

      elif self.config.test_policy == 'MajorityVote':
        # test action
        q_heads_l_policy = [tf.slice(self.q, [0, k * self.action_size], [1, self.action_size]) for k in
                            range(0, self.HEADSNUM)]

        q_head_votes = [tf.argmax(q_h_tmp, dimension=1) for q_h_tmp in q_heads_l_policy]

        self.y, _, self.count = tf.unique_with_counts(tf.concat(0,q_head_votes))

        self.max_ind = tf.argmax(self.count,dimension=0)
        #TODO: Do one hot vector to extract q_action_test

        self.q_action_test = [0]
        # train action
        tmp_q_slice = tf.slice(self.q, [0, self.head * self.action_size], [-1, self.action_size])
        self.q_action_train = tf.argmax(tmp_q_slice, dimension=1)
        # action node
        self.q_action = tf.select([self.is_training], self.q_action_train, self.q_action_test)

      elif self.config.test_policy == 'MaxQHead':
        # test action
        self.q_action_test = tf.argmax(self.q, dimension=1) % self.action_size
        # train action
        tmp_q_slice = tf.slice(self.q, [0, self.head * self.action_size], [-1, self.action_size])
        self.q_action_train = tf.argmax(tmp_q_slice, dimension=1)
        # action node
        self.q_action = tf.select([self.is_training], self.q_action_train, self.q_action_test)

      else:
        tmp_q_slice = tf.slice(self.q, [0, self.head * self.action_size], [-1, self.action_size])
        self.q_action = tf.argmax(tmp_q_slice, dimension=1)


    # target network
    with tf.variable_scope('target'):
      if self.cnn_format == 'NHWC':
        self.target_s_t = tf.placeholder('float32', 
            [None, self.screen_width, self.screen_height, self.history_length], name='target_s_t')
      else:
        self.target_s_t = tf.placeholder('float32', 
            [None, self.history_length, self.screen_width, self.screen_height], name='target_s_t')

      if self.config.ToyProblem:
        shape = self.target_s_t.get_shape().as_list()
        self.target_s_t_flat = tf.reshape(self.target_s_t, [-1, reduce(lambda x, y: x * y, shape[1:])])
        self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = linear(self.target_s_t_flat, 128, activation_fn=activation_fn, name='target_l3')

      else:
        self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t,
            32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='target_l1')
        self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1,
            64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='target_l2')
        self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2,
            64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l3')

      shape = self.target_l3.get_shape().as_list()
      self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        self.t_value_hid, self.t_w['l4_val_w'], self.t_w['l4_val_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_value_hid')

        self.t_adv_hid, self.t_w['l4_adv_w'], self.t_w['l4_adv_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_adv_hid')

        self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = \
          linear(self.t_value_hid, 1, name='target_value_out')

        self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = \
          linear(self.t_adv_hid, self.action_size, name='target_adv_out')

        # Average Dueling
        self.target_q = self.t_value + (self.t_advantage - 
          tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
      else:
        self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_l4')
        self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
            linear(self.target_l4, self.action_size*self.HEADSNUM, name='target_q')

      self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
      self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

    with tf.variable_scope('pred_to_target'):
      self.t_w_input = {}
      self.t_w_assign_op = {}

      for name in self.w.keys():
        self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
        self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

    # optimizer
    with tf.variable_scope('optimizer'):
      self.target_q_t = tf.placeholder('float32', [None, self.HEADSNUM], name='target_q_t')
      self.action = tf.placeholder('int64', [None], name='action')

      q_acted_l = []
      for k in range(0,self.HEADSNUM):
        action_one_hot_slice = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name='action_one_hot')
        q_slice = tf.slice(self.q,[0,k*self.action_size],[-1,self.action_size])
        q_acted_slice = tf.reduce_sum(q_slice * action_one_hot_slice, reduction_indices=1, name='q_acted')
        q_acted_slice = tf.reshape(q_acted_slice,[self.batch_size,1])
        q_acted_l.append(q_acted_slice)

      q_acted = tf.concat(1,q_acted_l)

      self.delta = self.target_q_t - q_acted

      #self.delta = tf.scalar_mul(1/self.HEADSNUM,self._delta)

      self.mask = tf.placeholder('float32',shape=[None,self.HEADSNUM],name='mask')
      self.delta_up = tf.mul(self.delta, self.mask)

      self.clipped_delta = tf.clip_by_value(self.delta_up, self.min_delta, self.max_delta, name='clipped_delta')

      self.global_step = tf.Variable(0, trainable=False)

      self.loss = tf.reduce_mean(tf.square(self.clipped_delta), name='loss')
      self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
      self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
          tf.train.exponential_decay(
              self.learning_rate,
              self.learning_rate_step,
              self.learning_rate_decay_step,
              self.learning_rate_decay,
              staircase=True))
      self.optim = tf.train.RMSPropOptimizer(
          self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)


      # Validation Statistics
      self.max_q_t_plus_1 = tf.placeholder('float32', [None, self.HEADSNUM], name='max_q_t_plus_1')
      avg_v_l = []
      q_acted_l_valid = []
      for k in range(0, self.HEADSNUM):
        action_one_hot_slice_valid = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name='action_one_hot_valid')
        q_slice_valid = tf.slice(self.q, [0, k * self.action_size], [-1, self.action_size])
        q_acted_slice_valid = tf.reduce_sum(q_slice_valid * action_one_hot_slice_valid, reduction_indices=1, name='q_acted_valid')
        q_acted_slice_valid = tf.reshape(q_acted_slice_valid, [self.valid_size, 1])
        max_q_t_plus_1_slice = tf.slice(self.q, [0, k], [-1, 1])
        avg_v_slice = tf.mul(q_acted_slice_valid,max_q_t_plus_1_slice)
        avg_v_l.append(tf.reduce_mean(avg_v_slice))
        q_acted_l_valid.append(q_acted_slice_valid)

      self.avg_v = avg_v_l
      q_acted_valid = tf.concat(1, q_acted_l_valid)

      self.delta_valid = self.target_q_t - q_acted_valid

      self.delta_up_valid = tf.mul(self.delta_valid, self.mask)

      self.clipped_delta_valid = tf.clip_by_value(self.delta_up_valid, self.min_delta, self.max_delta, name='clipped_delta_valid')


      self.loss_valid = tf.reduce_mean(tf.square(self.clipped_delta_valid), name='loss')





      self.comp_grads_and_vars = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01).compute_gradients(self.loss_valid)

      l1_grad_l1_norm = tf.reduce_mean(tf.abs(self.comp_grads_and_vars[0][0]),name='l1_grad_l1_norm')
      l2_grad_l1_norm = tf.reduce_mean(tf.abs(self.comp_grads_and_vars[2][0]),name='l2_grad_l1_norm')
      l3_grad_l1_norm = tf.reduce_mean(tf.abs(self.comp_grads_and_vars[4][0]),name='l3_grad_l1_norm')
      if self.config.ToyProblem:
        l4_grad_l1_norm = tf.constant(0)
      else:
        l4_grad_l1_norm = tf.reduce_mean(tf.abs(self.comp_grads_and_vars[6][0]),name='l4_grad_l1_norm')

      l1_l1_norm = tf.reduce_mean(tf.abs(self.comp_grads_and_vars[0][1]),name='l1_l1_norm')
      l2_l1_norm = tf.reduce_mean(tf.abs(self.comp_grads_and_vars[2][1]),name='l2_l1_norm')
      l3_l1_norm = tf.reduce_mean(tf.abs(self.comp_grads_and_vars[4][1]),name='l3_l1_norm')
      if self.config.ToyProblem:
        l4_l1_norm = tf.constant(0)
      else:
        l4_l1_norm = tf.reduce_mean(tf.abs(self.comp_grads_and_vars[6][1]), name='l4_l1_norm')

      self.grad_and_val_l1_norm = [l1_grad_l1_norm,l2_grad_l1_norm,l3_grad_l1_norm,l4_grad_l1_norm,l1_l1_norm,l2_l1_norm,l3_l1_norm,l4_l1_norm]


    with tf.variable_scope('summary'):
      scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
          'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.scalar_summary("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

      histogram_summary_tags = ['episode.rewards', 'episode.actions']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.histogram_summary(tag, self.summary_placeholders[tag])

      self.writer = tf.train.SummaryWriter('./logs/%s' % self.model_dir, self.sess.graph)

    tf.initialize_all_variables().run()

    self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)

    self.load_model()
    self.update_target_q_network()

  def update_target_q_network(self):
    for name in self.w.keys():
      self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

  def save_weight_to_pkl(self):
    if not os.path.exists(self.weight_dir):
      os.makedirs(self.weight_dir)

    for name in self.w.keys():
      save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

  def load_weight_from_pkl(self, cpu_mode=False):
    with tf.variable_scope('load_pred_from_pkl'):
      self.w_input = {}
      self.w_assign_op = {}

      for name in self.w.keys():
        self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
        self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

    for name in self.w.keys():
      self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})

    self.update_target_q_network()

  def inject_summary(self, tag_dict, step):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.step)

  def playMy(self, n_step=100000, _test_ep=0.01, render=False):

    screen, reward, action, terminal = self.env.new_random_game()
    ep_rewards = []
    ep_reward = 0

    for _ in range(self.history_length):
      self.history.add(screen)
    self.current_head = 0

    for estep in tqdm(range(n_step), ncols=70):
      # 1. predict
      action = self.predict(self.history.get(), self.current_head, test_ep=_test_ep, is_training=False)
      # 2. act
      env_action = 0
      if (action == 1):
        env_action = 3  # LEFT
      elif (action == 2):
        env_action = 4  # RIGHT

      screen, reward, terminal = self.env.act(env_action, is_training=False)
      self.history.add(screen)

      ep_reward += reward

      if terminal == True:
        screen, reward, action, terminal = self.env.new_random_game()
        ep_rewards.append(ep_reward)
        ep_reward = 0


      if estep % 1000 == 0 and len(ep_rewards) > 0:
        max_ep_reward = np.max(ep_rewards)
        min_ep_reward = np.min(ep_rewards)
        avg_ep_reward = np.mean(ep_rewards)

        print ('\nframes:' + str(estep)
               +'\nmax episode reward:'+str(max_ep_reward)
               +'\navg episode reward:' + str(avg_ep_reward)
               + '\nmin episode reward:' + str(min_ep_reward)
               )



  def play(self, n_step=10000, n_episode=100, test_ep=None, render=False):
    if test_ep == None:
      test_ep = self.ep_end

    test_history = History(self.config)

    if not self.display:
      gym_dir = '/tmp/%s-%s' % (self.env_name, get_time())
      self.env.env.monitor.start(gym_dir)

    best_reward, best_idx = 0, 0
    for idx in xrange(n_episode):
      screen, reward, action, terminal = self.env.new_random_game()
      current_reward = 0

      for _ in range(self.history_length):
        test_history.add(screen)

      for t in tqdm(range(n_step), ncols=70):
        # 1. predict
        action = self.predict(test_history.get(), test_ep,is_training=False)
        # 2. act
        screen, reward, terminal = self.env.act(action+1, is_training=False)
        # 3. observe
        test_history.add(screen)

        current_reward += reward
        if terminal:
          break

      if current_reward > best_reward:
        best_reward = current_reward
        best_idx = idx

      print "="*30
      print " [%d] Best reward : %d" % (best_idx, best_reward)
      print "="*30

    if not self.display:
      self.env.env.monitor.close()
      #gym.upload(gym_dir, writeup='https://github.com/devsisters/DQN-tensorflow', api_key='')
