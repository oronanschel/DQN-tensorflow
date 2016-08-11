import random
import tensorflow as tf
from tqdm import tqdm
from dqn.agent import Agent
import sys
from dqn.environment import GymEnvironment
from dqn.Myenvironment import MyGymEnvironment
from config import get_config
from tensorflow.python.framework import ops
flags = tf.app.flags

# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Environment
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')

# Etc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '9/10', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')

import numpy as np
rand = np.random.randint(10,200)
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

  fraction = 1 / (num - idx + 1)
  print " [*] GPU : %.4f" % fraction
  return fraction

def main(_):

  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    config = get_config(FLAGS) or FLAGS
    if not FLAGS.use_gpu:
      config.cnn_format = 'NHWC'


    print("args:"+str(sys.argv))
    config.folder_name = sys.argv[1]
    config.max_state = int(sys.argv[2])
    config.heads_num = int(sys.argv[3])
    config.p = float(sys.argv[4])
    config.screen_1_hot = sys.argv[5] == 'True'


    env = MyGymEnvironment(config)
    agent = Agent(config, env, sess)
    env = MyGymEnvironment(config)
    agent.train(env,config)



if __name__ == '__main__':
  tf.app.run()
