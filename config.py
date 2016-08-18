class AgentConfig(object):
  display = False

  max_step = 50*(10**6)
  memory_size = 10**6

  batch_size = 32
  random_start = 30
  cnn_format = 'NCHW'
  discount = 0.99
  target_q_update_step = 10**4
  learning_rate = 0.0006
  learning_rate_minimum = 0.00025
  learning_rate_decay = 0.96
  learning_rate_decay_step = 50000

  ep_end = 0.05
  ep_start = 1.

  # ep_end_t = 10**6
  ep_end_t = 10 ** 5

  history_length = 4
  train_frequency = 4

  # learn_start = 5*(10**4)
  learn_start = 100


  min_delta = -1
  max_delta = 1

  double_q = False
  dueling = False

  # eval_freq =  5*(10**4)
  eval_freq = 300

  # save_freq = 10000
  save_freq = 500

  # eval_steps = 10**4
  eval_steps = 100

  valid_size = 500

  # Bootstrap
  heads_num = 10
  p  = 0.5

  # ToyProblem
  ToyProblem = False
  max_ep_possible_reward = 10
  succ_max = 100

class EnvironmentConfig(object):
  env_name = 'Breakout-v0'
  # env_name = 'MontezumaRevenge-v0'
  ToyProblem = False
  if ToyProblem:
    screen_width  = 25
    screen_height = 1
  else:
    screen_width  = 84
    screen_height = 84

  max_reward = 1.
  min_reward = -1.

class DQNConfig(AgentConfig, EnvironmentConfig):
  model = ''
  pass

class M1(DQNConfig):
  backend = 'tf'
  env_type = 'detail'
  action_repeat = 1

def get_config(FLAGS):
  if FLAGS.model == 'm1':
    config = M1
  elif FLAGS.model == 'm2':
    config = M2

  for k, v in FLAGS.__dict__['__flags'].items():
    if k == 'gpu':
      if v == False:
        config.cnn_format = 'NHWC'
      else:
        config.cnn_format = 'NCHW'

    if hasattr(config, k):
      setattr(config, k, v)

  return config
