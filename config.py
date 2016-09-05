class AgentConfig(object):
  # ToyProblem
  ToyProblem = False

  if not ToyProblem:
    cropped = True
    screen_width  = 84
    screen_height = 84

    display = False

    max_step = 50*(10**6)
    memory_size = 10**6

    # frame_skip = 4
    frame_skip = 4
    death_ends_episode = True

    batch_size = 32 # cannot be changed for now
    random_start = 30
    cnn_format = 'NCHW'
    discount = 0.99
    target_q_update_step = 10**4
    # target_q_update_step =3* 10**4
    learning_rate = 0.00025
    learning_rate_minimum = 0.00025
    # learning_rate = 0.000025
    # learning_rate_minimum = 0.000025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 50000

    ep_end = 0.1
    ep_start = 1.

    ep_end_t = 1*10**6
    # ep_end_t = 10 ** 5

    history_length = 4
    train_frequency = 4
    # train_frequency = 2

    learn_start = 5*(10**4)
    # learn_start = 200


    min_delta = -1
    max_delta = 1

    double_q = True

    dueling = False

    eval_freq =  125*(10**3)
    # eval_freq = 15000

    # save_freq = 10000
    # save_freq = )500

    eval_steps = 10**4
    # eval_steps = 1000

    valid_size = 1000

    # Bootstrap
    heads_num = 1
    p  = 1
    test_policy = 'Ensemble'
    # test_policy = 'MaxQHead'
    # test_policy = 'MajorityVote'
    # test_policy = 'Standard'
  if ToyProblem:
    screen_width  = 15
    screen_height = 1

    display = False
    death_minus_reward = False
    death_ends_episode = False

    max_step = 5 * (10 ** 5)
    memory_size = 10 ** 4

    frame_skip = 1

    batch_size = 32  # cannot be changed for now
    random_start = 4
    cnn_format = 'NCHW'
    discount = 0.99
    target_q_update_step = 300
    learning_rate = 0.00025
    learning_rate_minimum = 0.00025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 50000

    ep_end = 0.1
    ep_start = 1.

    ep_end_t = 1 * 10 ** 4

    history_length = 2
    train_frequency = 4

    learn_start = 500


    min_delta = -1
    max_delta = 1

    double_q = True

    dueling = False

    eval_freq = 1250

    eval_steps = 500

    valid_size = 500
    # Bootstrap
    heads_num = 1
    p = 1
    test_policy = 'Ensemble'
    # test_policy = 'MaxQHead'
    # test_policy = 'MajorityVote'
    # test_policy = 'Standard'

class EnvironmentConfig(object):
  env_name = 'breakout.bin'
  # env_name = 'qbert.bin'
  # env_name = 'jamesbond.bin'
  # env_name = 'alien.bin'
  # env_name= 'battle_zone.bin'
  # env_name= 'beam_rider.bin'
  # ToyProblem = True
  # if ToyProblem:
  #   screen_width  = 10
  #   screen_height = 1
  # else:
  #   screen_width  = 84
  #   screen_height = 84

  max_reward = 1.
  min_reward = -1.

class DQNConfig(AgentConfig, EnvironmentConfig):
  model = ''
  pass

class M1(DQNConfig):
  backend = 'tf'
  env_type = 'detail'
  # action_repeat = 1

def get_config(FLAGS):
  if FLAGS.model == 'm1':
    config = M1

  for k, v in FLAGS.__dict__['__flags'].items():
    if k == 'gpu':
      if v == False:
        config.cnn_format = 'NHWC'
      else:
        config.cnn_format = 'NCHW'

    if hasattr(config, k):
      setattr(config, k, v)

  return config
