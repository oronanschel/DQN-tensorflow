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

class Agent(BaseModel):
  def __init__(self, config, environment, sess):
    super(Agent, self).__init__(config)
    self.sess = sess
    self.weight_dir = 'weights'

    self.HEADSNUM = config.heads_num
    self.batch_size = config.batch_size

    self.env = environment
    self.history = History(self.config)
    self.memory = ReplayMemory(self.config, self.model_dir)

    with tf.variable_scope('step'):
      self.step_op = tf.Variable(0, trainable=False, name='step')
      self.step_input = tf.placeholder('int32', None, name='step_input')
      self.step_assign_op = self.step_op.assign(self.step_input)

    self.create_results_file()
    self.build_dqn()

  def create_results_file(self):
    tempdir = os.path.join(os.getcwd(), "models")
    self.create_dir(tempdir)
    folder_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    self.mydir = os.path.join(tempdir, folder_name)
    self.create_dir(self.mydir)
    self.prog_file = os.path.join(self.mydir, 'training_progress.csv')
    data_file = open(self.prog_file, 'wb')
    data_file.write('avg_reward,avg_loss,avg_q,avg_ep_reward,max_ep_reward,min_ep_reward,num_game,epsilon,'
                    'l1_grad_l1_norm,l2_grad_l1_norm,l3_grad_l1_norm,l4_grad_l1_norm,l1_l1_norm,l2_l1_norm,l3_l1_norm,l4_l1_norm'
                    ',succ_counts,lr,step'
                    ',q0_0,q1_0,q2_0,q3_0,q4_0,q5_0,q6_0,q7_0,q8_0,q9_0'
                    ',q0_1,q1_1,q2_1,q3_1,q4_1,q5_1,q6_1,q7_1,q8_1,q9_1\n')
    data_file.close()

  def update_results(self, avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game,ep,
                     l1_norm,succ_counts,lr,step,q):
      fd = open(self.prog_file,'a')
      fd.write('%f,%f,%f,%f,%f,%f,%f,%f,' % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game, ep))
      fd.write('%f,%f,%f,%f,%f,%f,%f,%f,' % ( l1_norm[0],l1_norm[1],l1_norm[2],l1_norm[3],l1_norm[4],l1_norm[5],l1_norm[6],l1_norm[7] ))
      fd.write('%d,%f,%d,' % (succ_counts,lr,step))
      for i in range(0,10):
        fd.write('%f,' % (q[i][0]))
      for i in range(0, 10):
        fd.write('%f,' % (q[i][1]))

      fd.write('\n' % (q[i][1]))
      fd.close()

  def create_dir(self,p):
    try:
      os.makedirs(p)
    except OSError, e:
      if e.errno != 17:
        raise  # This was not a "directory exist" error..

  def train(self):
    start_step = 0
    # start_time = time.time()

    num_game, self.update_count, ep_reward = 0, 0, 0.
    total_reward, self.total_loss, self.total_q = 0., 0., 0.
    max_avg_ep_reward = 0
    ep_rewards, actions = [], []
    max_ep_reward_count = 0
    num_game = 0

    screen, reward, action, terminal = self.env.new_random_game()

    for _ in range(self.history_length):
      self.history.add(screen)

    self.current_head = 0
    self.max_ep_reward_flag = False

    for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):

      # if self.step == self.learn_start:
      #   num_game, self.update_count, ep_reward = 0, 0, 0.
      #   total_reward, self.total_loss, self.total_q = 0., 0., 0.
      #   ep_rewards, actions = [], []

      # 1. predict
      action = self.predict(self.history.get(),self.current_head)
      # 2. act
      screen, reward, terminal = self.env.act(action, is_training=True)
      # 3. observe + learn
      mask = np.random.binomial(1,1,size=[self.HEADSNUM])

      self.observe(screen, reward, action, terminal,mask)

      if terminal:
        num_game +=1
        ep_reward += reward

        screen, reward, action, terminal = self.env.new_random_game()

        ep_rewards.append(ep_reward)
        ep_reward = 0.
        # replace playing head
        self.current_head = np.random.randint(self.HEADSNUM)

      else:
        ep_reward += reward

      # actions.append(action)
      total_reward += reward

      if self.step >= self.learn_start:
        if (self.step % self.test_step == self.test_step - 1):
          self.max_ep_reward_flag = False
          avg_reward = total_reward / self.test_step
          avg_loss = self.total_loss / self.update_count
          avg_q = self.total_q / self.update_count

          try:
            max_ep_reward = np.max(ep_rewards)
            min_ep_reward = np.min(ep_rewards)
            avg_ep_reward = np.mean(ep_rewards)
          except:
            max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

          print '\nmax_ep_r_cnt:%d  avg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
              % (max_ep_reward_count,avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game)

          if max_avg_ep_reward * 0.9 <= avg_ep_reward:
            self.step_assign_op.eval({self.step_input: self.step + 1})
            # self.save_model(self.step + 1)

            max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

          if self.step > 180:
            # gradients, weights sample
            g_w_l1_norm, q = self.gradient_weights_l1_norm()

            lr = self.learning_rate_op.eval({self.learning_rate_step: self.step})

            ep_rec = (self.ep_end +max(0., (self.ep_start - self.ep_end)* (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))
            self.update_results(avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game,
                                ep_rec,g_w_l1_norm,max_ep_reward_count,lr,self.step,q)

            plot(self.mydir)
            '''
                        self.inject_summary({
                'average.reward': avg_reward,
                'average.loss': avg_loss,
                'average.q': avg_q,
                'episode.max reward': max_ep_reward,
                'episode.min reward': min_ep_reward,
                'episode.avg reward': avg_ep_reward,
                'episode.num of game': num_game,
                'episode.rewards': ep_rewards,
                'episode.actions': actions,
                'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
              }, self.step)
            '''


          # num_game = 0
          # total_reward = 0.
          # self.total_loss = 0.
          # self.total_q = 0.
          # self.update_count = 0
          # ep_reward = 0.
          # ep_rewards = []

  def gradient_weights_l1_norm(self):
    if self.memory.count < self.history_length:
      return
    else:
      s_t, action, reward, s_t_plus_1, terminal ,mask = self.memory.sample()

      # q(s_1,a) - shape = [32,actions*HEADS]
      q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})
      q_zero_plus_1 = self.zero_q.eval({self.zero_s_t: s_t_plus_1})
      q_zero_s_t = self.zero_q.eval({self.zero_s_t: s_t})

      # t - shape [32]
      terminal = np.array(terminal) + 0.

      q_t_plus_1_slice = q_t_plus_1[:, [i for i in range(0, self.env.action_size)]]
      q_zero_plus_1_slice = q_zero_plus_1[:, [i for i in range(0, self.env.action_size)]]
      q_zero_slice = q_zero_s_t[:, [i for i in range(0, self.env.action_size)]]

      max_q_t_plus_1 = np.max(q_t_plus_1_slice, axis=1)
      action_t = np.argmax(q_t_plus_1_slice, axis=1)
      max_q_zero_plus_1 = q_zero_plus_1_slice[np.arange(self.batch_size), action_t]
      q_zero = q_zero_slice[np.arange(self.batch_size), action]

      target_q_t = (1. - terminal) * self.discount * (max_q_t_plus_1 - max_q_zero_plus_1) + reward - q_zero
      target_q_t = target_q_t.reshape(self.batch_size, 1)
      for k in range(1, self.HEADSNUM):
        q_t_plus_1_slice = q_t_plus_1[:, [i for i in range(self.env.action_size * k, self.env.action_size * (k + 1))]]
        q_zero_slice = q_zero_s_t[:, [i for i in range(self.env.action_size * k, self.env.action_size * (k + 1))]]

        max_q_t_plus_1 = np.max(q_t_plus_1_slice, axis=1)
        action_t = np.argmax(q_t_plus_1_slice, axis=0)
        max_q_zero_plus_1 = q_zero_plus_1_slice[np.arange(self.batch_size), action_t]
        q_zero = q_zero_slice[np.arange(self.batch_size), action]

        target_q_t_slice = (1. - terminal) * self.discount * (max_q_t_plus_1 - max_q_zero_plus_1) + reward - q_zero
        target_q_t_slice = target_q_t_slice.reshape(self.batch_size, 1)
        target_q_t = np.concatenate((target_q_t, target_q_t_slice), axis=1)

      grads_weights_samp =  self.sess.run([x for x in self.grad_and_val_l1_norm], {
        self.target_q_t: target_q_t,
        self.action: action,
        self.s_t: s_t,
        self.zero_s_t : s_t ,
        self.learning_rate_step: self.step,
        self.mask: mask
      })

      # q eval:
      s_t_test = np.zeros(shape=[10,1,10,1],dtype=np.float16)
      for i in range(0,10):
        s_t_test [i][0][i] = 1

      q_t_test = self.q.eval({self.s_t: s_t_test})
      q_zero_test = self.zero_q.eval({self.zero_s_t: s_t_test})
      '''
            print('---------')
            print(q_t_test)
            print(q_zero_test)
            print('---------')
      '''


      q = q_t_test - q_zero_test
      # q = q_t_test

      return grads_weights_samp , q

  def predict(self, s_t, current_head, test_ep=None):
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))
    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      action = self.q_action.eval({self.s_t: [s_t], self.zero_s_t: [s_t] , self.head:current_head})[0]

      #action = self.q_action.eval({self.s_t: [s_t]})[0]

    return action

  def observe(self, screen, reward, action, terminal,mask):
    reward = max(self.min_reward, min(self.max_reward, reward))

    self.history.add(screen)
    self.memory.add(screen, reward, action, terminal,mask)

    if self.step > self.learn_start:
      if self.step % self.train_frequency == 0:
        self.q_learning_mini_batch()

      if self.step % self.target_q_update_step == self.target_q_update_step - 1:
        self.update_target_q_network()

  def q_learning_mini_batch(self):
    if self.memory.count < self.history_length:
      return
    else:
      s_t, action, reward, s_t_plus_1, terminal, mask = self.memory.sample()


    # q(s_1,a) - shape = [32,actions*HEADS]
    q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})
    q_zero_plus_1 = self.zero_q.eval({self.zero_s_t: s_t_plus_1})
    q_zero_s_t = self.zero_q.eval({self.zero_s_t: s_t})

    # t - shape [32]
    terminal = np.array(terminal) + 0.

    q_t_plus_1_slice = q_t_plus_1[:, [i for i in range(0,self.env.action_size)]]
    q_zero_plus_1_slice = q_zero_plus_1[:, [i for i in range(0,self.env.action_size)]]
    q_zero_slice = q_zero_s_t[:, [i for i in range(0,self.env.action_size)]]

    max_q_t_plus_1 = np.max(q_t_plus_1_slice, axis=1)
    action_t = np.argmax(q_t_plus_1_slice, axis=1)
    max_q_zero_plus_1 = q_zero_plus_1_slice[ np.arange(self.batch_size) , action_t ]
    q_zero = q_zero_slice[ np.arange(self.batch_size) , action ]

    target_q_t = (1. - terminal) * self.discount * (max_q_t_plus_1 - max_q_zero_plus_1) + reward + q_zero
    target_q_t = target_q_t.reshape(self.batch_size,1)
    for k in range(1,self.HEADSNUM):
      q_t_plus_1_slice = q_t_plus_1[:, [i for i in range(self.env.action_size*k, self.env.action_size*(k+1))]]
      q_zero_slice = q_zero_s_t[:, [i for i in range(self.env.action_size*k, self.env.action_size*(k+1))]]

      max_q_t_plus_1 =  np.max( q_t_plus_1_slice, axis=1)
      action_t = np.argmax(q_t_plus_1_slice, axis=0)
      max_q_zero_plus_1 = q_zero_plus_1_slice[np.arange(self.batch_size), action_t]
      q_zero = q_zero_slice[np.arange(self.batch_size), action]

      target_q_t_slice =  (1. - terminal) * self.discount * (max_q_t_plus_1 - max_q_zero_plus_1) + reward + q_zero
      target_q_t_slice = target_q_t_slice.reshape(self.batch_size,1)
      target_q_t = np.concatenate(  (target_q_t,target_q_t_slice), axis=1)
      # TODO: Think about mixing the targets

      # max_q - shape = [32,5]

      #max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)



    zero_q_t, _, q_t, loss = self.sess.run([self.zero_q,self.optim, self.q, self.loss], {
      self.target_q_t: target_q_t,
      self.action: action,
      self.s_t: s_t,
      self.zero_s_t : s_t,
      self.learning_rate_step: self.step,
      self.mask : mask,
    })

    self.total_loss += loss
    self.total_q += (q_t - zero_q_t).mean()
    self.update_count += 1

  def build_dqn(self):
    self.w = {}
    self.t_w = {}
    self.zero_w = {}

    initializer = tf.truncated_normal_initializer(0, 0.02)
    # linear_std_init = 0.02
    # linear_bias_init = 0
    linear_std_init = 0.02
    linear_bias_init = 0
    activation_fn = tf.nn.relu

    # zero network
    with tf.variable_scope('zero'):
      if self.cnn_format == 'NHWC':
        self.zero_s_t = tf.placeholder('float32',
            [None, self.screen_width, self.screen_height, self.history_length], name='zero_s_t')
      else:
        self.zero_s_t = tf.placeholder('float32',
            [None, self.history_length, self.screen_width, self.screen_height], name='zero_s_t')

      if self.config.ToyProblem:
        shape = self.zero_s_t.get_shape().as_list()
        self.zero_s_t_flat = tf.reshape(self.zero_s_t, [-1, reduce(lambda x, y: x * y, shape[1:])])
        self.zero_l3, self.zero_w['l3_w'], self.zero_w['l3_b'] = linear(self.zero_s_t_flat,128,stddev=linear_std_init, bias_start=linear_bias_init, activation_fn=activation_fn, name='zero_l3')

      else:
        self.zero_l1, self.zero_w['l1_w'], self.zero_w['l1_b'] = conv2d(self.zero_s_t,
            32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='zero_l1')
        self.zero_l2, self.zero_w['l2_w'], self.zero_w['l2_b'] = conv2d(self.zero_l1,
            64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='zero_l2')
        self.zero_l3, self.zero_w['l3_w'], self.zero_w['l3_b'] = conv2d(self.zero_l2,
            64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='zero_l3')

      shape = self.zero_l3.get_shape().as_list()
      self.zero_l3_flat = tf.reshape(self.zero_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])


      self.zero_l4, self.zero_w['l4_w'], self.zero_w['l4_b'] = \
          linear(self.zero_l3_flat, 512,stddev=linear_std_init, bias_start=linear_bias_init, activation_fn=activation_fn, name='zero_l4')
      self.zero_q_tmp, self.zero_w['q_w'], self.zero_w['q_b'] = \
          linear(self.zero_l4, self.env.action_size*self.HEADSNUM,stddev=linear_std_init, bias_start=linear_bias_init, name='zero_q')


      q_slice_left = tf.slice(self.zero_q_tmp,[0,0],[-1,1])
      q_slice_right= tf.slice(self.zero_q_tmp,[0,1],[-1,1])


      self.zero_q_c = tf.concat(1,[q_slice_left,q_slice_right])
      self.zero_q = tf.scalar_mul(1,self.zero_q_c)

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
        self.l3, self.w['l3_w'], self.w['l3_b'] = linear(self.s_t_flat, 128,stddev=linear_std_init, bias_start=linear_bias_init, activation_fn=activation_fn, name='l3')

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
          linear(self.adv_hid, self.env.action_size, name='adv_out')

        # Average Dueling
        self.q = self.value + (self.advantage - 
          tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
      else:
        self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512,stddev=linear_std_init, bias_start=linear_bias_init, activation_fn=activation_fn, name='l4')
        self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.env.action_size*self.HEADSNUM,stddev=linear_std_init, bias_start=linear_bias_init, name='q')

      self.head = tf.placeholder('int32', name='head')
      #self.q_action = tf.argmax(self.q, dimension=1)
      tmp_q_slice = tf.slice(self.q, [0, self.head * self.env.action_size], [-1, self.env.action_size])
      tmp_q_zero_slice = tf.slice(self.zero_q, [0, self.head * self.env.action_size], [-1, self.env.action_size])
      self.q_action = tf.argmax(tmp_q_slice - tmp_q_zero_slice, dimension=1)

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
        self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = linear(self.target_s_t_flat, 128,stddev=linear_std_init, bias_start=linear_bias_init, activation_fn=activation_fn, name='target_l3')

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
          linear(self.t_adv_hid, self.env.action_size, name='target_adv_out')

        # Average Dueling
        self.target_q = self.t_value + (self.t_advantage - 
          tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
      else:
        self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
            linear(self.target_l3_flat, 512,stddev=linear_std_init, bias_start=linear_bias_init, activation_fn=activation_fn, name='target_l4')
        self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
            linear(self.target_l4, self.env.action_size*self.HEADSNUM,stddev=linear_std_init, bias_start=linear_bias_init, name='target_q')

      self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
      self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

    with tf.variable_scope('pred_to_target'):
      self.t_w_input = {}
      self.t_w_assign_op = {}

      for name in self.w.keys():
        self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
        self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])
    with tf.variable_scope('pred_to_zero'):
      self.zero_w_input = {}
      self.zero_w_assign_op = {}

      for name in self.w.keys():
        self.zero_w_input[name] = tf.placeholder('float32', self.zero_w[name].get_shape().as_list(), name=name)
        self.zero_w_assign_op[name] = self.zero_w[name].assign(self.zero_w_input[name])

    # optimizer
    with tf.variable_scope('optimizer'):
      self.target_q_t = tf.placeholder('float32', [self.batch_size, self.HEADSNUM], name='target_q_t')
      self.action = tf.placeholder('int64', [self.batch_size], name='action')

      q_acted_l = []
      for k in range(0,self.HEADSNUM):
        action_one_hot_slice = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')

        q_slice = tf.slice(self.q,[0,k*self.env.action_size],[-1,self.env.action_size])

        q_acted_slice = tf.reduce_sum(q_slice * action_one_hot_slice, reduction_indices=1, name='q_acted')

        q_acted_slice = tf.reshape(q_acted_slice,[self.batch_size,1])

        q_acted_l.append(q_acted_slice)

      self.q_acted = tf.concat(1,q_acted_l)

      self.delta = self.target_q_t - self.q_acted

      self.mask = tf.placeholder('float32',shape=[self.batch_size,self.HEADSNUM],name='mask')
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


      self.comp_grads_and_vars = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01).compute_gradients(self.loss)

      l1_grad_l1_norm = tf.reduce_mean(tf.abs(self.comp_grads_and_vars[0+6][0]),name='l1_grad_l1_norm')
      l2_grad_l1_norm = tf.reduce_mean(tf.abs(self.comp_grads_and_vars[2+6][0]),name='l2_grad_l1_norm')
      l3_grad_l1_norm = tf.reduce_mean(tf.abs(self.comp_grads_and_vars[4+6][0]),name='l3_grad_l1_norm')
      if self.config.ToyProblem:
        l4_grad_l1_norm = tf.constant(0)
      else:
        l4_grad_l1_norm = tf.reduce_mean(tf.abs(self.comp_grads_and_vars[6+6][0]),name='l4_grad_l1_norm')

      l1_l1_norm = tf.reduce_mean(tf.abs(self.comp_grads_and_vars[0+6][1]),name='l1_l1_norm')
      l2_l1_norm = tf.reduce_mean(tf.abs(self.comp_grads_and_vars[2+6][1]),name='l2_l1_norm')
      l3_l1_norm = tf.reduce_mean(tf.abs(self.comp_grads_and_vars[4+6][1]),name='l3_l1_norm')
      if self.config.ToyProblem:
        l4_l1_norm = tf.constant(0)
      else:
        l4_l1_norm = tf.reduce_mean(tf.abs(self.comp_grads_and_vars[6+6][1]), name='l4_l1_norm')

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

    # self.load_model()
    self.update_target_q_network()
    self.update_zero_q_network()

  def update_target_q_network(self):
    for name in self.w.keys():
      self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

  def update_zero_q_network(self):
    for name in self.w.keys():
      self.zero_w_assign_op[name].eval({self.zero_w_input[name]: self.w[name].eval()})

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
      print(summary_str)
      #self.writer.add_summary(summary_str, self.step)

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
        action = self.predict(test_history.get(), test_ep)
        # 2. act
        screen, reward, terminal = self.env.act(action, is_training=False)
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
