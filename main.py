import random
import tensorflow as tf
import os
from ALE.ale_python_interface.ale_python_interface import ALEInterface
import numpy as np
from dqn.agent import Agent

#from dqn.environment import GymEnvironment
from dqn.Myenvironment import MyGymEnvironment

from config import get_config
import time
import datetime

flags = tf.app.flags

# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
#flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Environment
# flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
# flags.DEFINE_string('env_name', 'MontezumaRevenge-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')

# Etc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/10', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
# flags.DEFINE_boolean('display', True, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')


import numpy as np
rand = np.random.randint(10,200)
print('rand seed:'+str(rand))
flags.DEFINE_integer('random_seed', rand, 'Value of random seed')

FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = idx/num
  print " [*] GPU : %.4f" % fraction
  return fraction

def main(_):
  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    config = get_config(FLAGS) or FLAGS

    exp_name = 'ddqn_10_head_target_update_3104'
    config.folder_name = exp_name + datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')

    # config.folder_name = 'ddqn_10_head_2016-08-29_15-01-09'

    if config.ToyProblem:
      env = MyGymEnvironment(config)
      print('TOY ENV')

    else:
      env = ALEInterface()
      rng = np.random.RandomState(123456) # DETERMINSTIC
      env.setInt('random_seed', rng.randint(0,1000))
      env.setBool('display_screen', config.display)
      env.setFloat('repeat_action_probability', 0.)
      env.setBool('color_averaging', True)
      rom_dir = os.path.join(os.getcwd(), "aleroms")
      rom_path = os.path.join(rom_dir, config.env_name)
      env.loadROM(rom_path)

    if not FLAGS.use_gpu:
      config.cnn_format = 'NHWC'

    agent = Agent(config, env, sess)

    if FLAGS.is_train:
      agent.train()
    else:
      agent.test()

if __name__ == '__main__':
  tf.app.run()
