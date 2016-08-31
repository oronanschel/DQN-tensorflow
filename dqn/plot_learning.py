import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv, pdb, sys
from matplotlib.backends.backend_pdf import PdfPages
from config import get_config
import numpy as np



def plot(filename, pdf_loc="training.pdf", csv_loc="training_progress.csv",heads_num=10):


  avg_v_tag = []
  q_diff_tag = []
  p_diff_tag = []
  for i in range(0,heads_num):
      avg_v_tag.append('avg_v['+str(i)+']')
      q_diff_tag.append('q_diff['+str(i)+']')
      p_diff_tag.append('p_diff[' + str(i) + ']')



  lr_tag = {'lr'}
  reward_tags = {'avg_ep_reward', 'min_ep_reward','max_ep_reward'}
  grad_norm_tags = {'l1_grad_l1_norm','l2_grad_l1_norm','l3_grad_l1_norm','l4_grad_l1_norm'}
  weight_norm_tags = {'l1_l1_norm','l2_l1_norm','l3_l1_norm','l4_l1_norm'}
  tags_1 = {'epsilon','avg_loss','num_game'}

  filename = "../models/2016-08-17_15-26-59" if filename is None else filename


  open_loc = "%s/%s" % (filename, csv_loc)
  save_loc = "%s/%s" % (filename, pdf_loc)
  with open(open_loc) as csvfile:
    reader = csv.DictReader(csvfile)
    avg_v = []
    p_diff = []
    q_diff = []
    for i in range(0, heads_num):
        avg_v.append([])
        p_diff.append([])
        q_diff.append([])


    avg_ep_reward = []
    max_ep_reward = []
    min_ep_reward = []
    epsilon = []

    l1_grad_l1_norm = []
    l2_grad_l1_norm = []
    l3_grad_l1_norm = []
    l4_grad_l1_norm = []

    l1_l1_norm =[]
    l2_l1_norm =[]
    l3_l1_norm = []
    l4_l1_norm = []

    lr =[]
    step = []
    avg_loss = []
    num_game = []

    for row in reader:
        for i in range(0, heads_num):
            avg_v[i].append(float(row['avg_v['+str(i)+']']))
            p_diff[i].append(float(row['p_diff['+str(i)+']']))
            q_diff[i].append(float(row['q_diff[' + str(i) + ']']))
        avg_loss.append(float(row['avg_loss']))
        avg_ep_reward.append(float(row['avg_ep_reward']))
        max_ep_reward.append(float(row['max_ep_reward']))
        min_ep_reward.append(float(row['min_ep_reward']))
        num_game.append(row['num_game'])
        epsilon.append(row['epsilon'])

        l1_grad_l1_norm.append(row['l1_grad_l1_norm'])
        l2_grad_l1_norm.append(row['l2_grad_l1_norm'])
        l3_grad_l1_norm.append(row['l3_grad_l1_norm'])
        l4_grad_l1_norm.append(row['l4_grad_l1_norm'])

        l1_l1_norm.append(row['l1_l1_norm'])
        l2_l1_norm.append(row['l2_l1_norm'])
        l3_l1_norm.append(row['l3_l1_norm'])
        l4_l1_norm.append(row['l4_l1_norm'])

        lr.append(row['lr'])
        step.append(row['step'])

  pp = PdfPages(save_loc)

  h = filter()

  step = np.array([float(i) for i in step])
  step = step/10**6


  cm_subsection = np.linspace(0, 1, heads_num)
  color = [cm.jet(x) for x in cm_subsection]
  # print(avg_v)
  plt.figure()
  for i in range(0,heads_num):
      plt.plot(step,avg_v[i], label='avg_v['+str(i)+']',color=color[i])

  plt.title("avg v")
  plt.grid(True)
  plt.xlabel('frames [millions]')
  plt.legend(loc='best')
  plt.savefig(pp, format='pdf')
  plt.close()


  color_reward = {'avg_ep_reward':'g', 'min_ep_reward':'r','max_ep_reward':'b'}
  color_m_reward = {'avg_ep_reward':'go', 'min_ep_reward':'ro','max_ep_reward':'bo'}
  # print(avg_reward)
  plt.figure()
  plt.title("Reward Stats")
  for tag in reward_tags:
      smooth = np.convolve(eval(tag), h)
      plt.plot(step, smooth[len(h)  / 2:-len(h) / 2 + 1], color_reward[tag], label=tag, linewidth=2.0)
      plt.plot(step, eval(tag), color_m_reward[tag], alpha=.5)
  plt.grid(True)
  plt.xlabel('frames [millions]')
  plt.legend(loc='best')
  plt.savefig(pp, format='pdf')
  plt.close()

  # print(tags 1)

  for tag in tags_1:
      plt.figure()
      plt.title(tag)
      plt.plot(step,eval(tag), label=tag)
      plt.xlabel('frames [millions]')
      plt.legend(loc='best')
      plt.savefig(pp, format='pdf')
      plt.close()


  # print(diff_p)
  plt.figure()
  for i in range(0,heads_num):
      smooth = np.convolve(p_diff[i], h)
      plt.plot(step, smooth[len(h)  / 2:-len(h) / 2 + 1], color = color[i], label='q['+str(i)+']', linewidth=2.0)
      plt.plot(step, p_diff[i], color = color[i],marker='o',linestyle=' ', alpha=.5)
  plt.title("Avarage Policy Difference from Ensemble")
  plt.grid(True)
  plt.xlabel('frames [millions]')
  plt.legend(loc='best')
  plt.savefig(pp, format='pdf')
  plt.close()

  # print(diff_q)
  plt.figure()
  for i in range(0,heads_num):
      smooth = np.convolve(q_diff[i], h)
      plt.plot(step, smooth[len(h)  / 2:-len(h) / 2 + 1], color = color[i], label='q['+str(i)+']', linewidth=2.0)
      plt.plot(step, q_diff[i], color = color[i],marker='o',linestyle=' ', alpha=.5)

  plt.title("Avarage ||Q_k Q_{ensemble}||_2^2")
  plt.grid(True)
  plt.xlabel('frames [millions]')
  plt.legend(loc='best')
  plt.savefig(pp, format='pdf')
  plt.close()

  # print(grad norm 1)

  plt.figure()
  plt.title("Grad L1 norm")
  for tag in grad_norm_tags:
    plt.plot(step,eval(tag), label=tag)

  plt.xlabel('frames [millions]')
  plt.legend(loc='best')
  plt.savefig(pp, format='pdf')
  plt.close()

  # print(weights norm 1)

  plt.figure()
  plt.title("Weights L1 norm")
  for tag in weight_norm_tags:
    plt.plot(step,eval(tag), label=tag)

  plt.xlabel('frames [millions]')
  plt.legend(loc='best')
  plt.savefig(pp, format='pdf')
  plt.close()

  # learning rate
  for tag in lr_tag:
      plt.figure()
      plt.title(tag)
      plt.plot(step,eval(tag), label=tag)
      plt.xlabel('frames [millions]')
      plt.legend(loc='best')
      plt.savefig(pp, format='pdf')
      plt.close()


  pp.close()

def filter():
    fc = 0.1  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)

    # Compute sinc filter.
    h = np.sinc(2 * fc * (n - (N - 1) / 2.))

    # Compute Blackman window.
    w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
        0.08 * np.cos(4 * np.pi * n / (N - 1))

    # Multiply sinc filter with window.
    h = h * w

    # Normalize to get unity gain.
    h = h / np.sum(h)

    return h


if __name__ == "__main__":
  plot(sys.argv[1] if len(sys.argv) > 1 else None)