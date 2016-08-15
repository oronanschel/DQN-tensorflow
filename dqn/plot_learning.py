import matplotlib.pyplot as plt
import csv, pdb, sys
from matplotlib.backends.backend_pdf import PdfPages
from config import get_config
import numpy as np



def plot(filename, pdf_loc="training.pdf", csv_loc="training_progress.csv",heads_num=1):


  avg_v_tag = []
  for i in range(0,heads_num):
      avg_v_tag.append('avg_v['+str(i)+']')

  lr_tag = {'lr'}
  reward_tags = {'avg_ep_reward', 'min_ep_reward','max_ep_reward'}
  grad_norm_tags = {'l1_grad_l1_norm','l2_grad_l1_norm','l3_grad_l1_norm','l4_grad_l1_norm'}
  weight_norm_tags = {'l1_l1_norm','l2_l1_norm','l3_l1_norm','l4_l1_norm'}
  tags_1 = {'avg_loss','epsilon'}
  tags_2 = {'num_game'}

  filename = "../models/2016-07-18_15-37-45" if filename is None else filename
  open_loc = "%s/%s" % (filename, csv_loc)
  save_loc = "%s/%s" % (filename, pdf_loc)
  with open(open_loc) as csvfile:
    reader = csv.DictReader(csvfile)
    avg_v = []
    for i in range(0, heads_num):
        avg_v.append([])

    avg_ep_reward = []
    max_ep_reward = []
    min_ep_reward = []
    num_game = []
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

    for row in reader:
        for i in range(0, heads_num):
            avg_v[i].append(float(row['avg_v['+str(i)+']']))
        avg_loss.append(float(row['avg_loss']))
        avg_ep_reward.append(row['avg_ep_reward'])
        max_ep_reward.append(row['max_ep_reward'])
        min_ep_reward.append(row['min_ep_reward'])
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


  # print(avg_v)
  plt.figure()
  plt.title("avg v")
  for i in range(0,heads_num):
    plt.plot(step,avg_v[i], label='avg_v['+str(i)+']')
  plt.xlabel('step')
  plt.legend(loc='best')
  plt.savefig(pp, format='pdf')
  plt.close()


  # print(avg_reward)
  plt.figure()
  plt.title("Reward Stats")
  for tag in reward_tags:
    plt.plot(step,eval(tag), label=tag)

  plt.xlabel('step')
  plt.legend(loc='best')
  plt.savefig(pp, format='pdf')
  plt.close()

  # print(tags 1)

  for tag in tags_1:
      plt.figure()
      plt.title(tag)
      plt.plot(step,eval(tag), label=tag)
      plt.xlabel('step')
      plt.legend(loc='best')
      plt.savefig(pp, format='pdf')
      plt.close()


  # print(grad norm 1)

  plt.figure()
  plt.title("Grad L1 norm")
  for tag in grad_norm_tags:
    plt.plot(step,eval(tag), label=tag)

  plt.xlabel('step')
  plt.legend(loc='best')
  plt.savefig(pp, format='pdf')
  plt.close()

  # print(weights norm 1)

  plt.figure()
  plt.title("Weights L1 norm")
  for tag in weight_norm_tags:
    plt.plot(step,eval(tag), label=tag)

  plt.xlabel('step')
  plt.legend(loc='best')
  plt.savefig(pp, format='pdf')
  plt.close()

  # learning rate
  for tag in lr_tag:
      plt.figure()
      plt.title(tag)
      plt.plot(step,eval(tag), label=tag)
      plt.xlabel('step')
      plt.legend(loc='best')
      plt.savefig(pp, format='pdf')
      plt.close()








  pp.close()

if __name__ == "__main__":
  plot(sys.argv[1] if len(sys.argv) > 1 else None)