import matplotlib.pyplot as plt
import csv, pdb, sys
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np



def plot(filename, pdf_loc="training.pdf", csv_loc="training_progress.csv"):


  q_tags = {'q0_0','q1_0','q2_0','q3_0','q4_0','q5_0','q6_0','q7_0','q8_0','q9_0',
                    'q0_1','q1_1','q2_1','q3_1','q4_1','q5_1','q6_1','q7_1','q8_1','q9_1'}
  counts_tag = {'succ_counts'}
  lr_tag = {'lr'}
  reward_tags = {'avg_reward','avg_ep_reward', 'min_ep_reward','max_ep_reward'}
  grad_norm_tags = {'l1_grad_l1_norm','l2_grad_l1_norm','l3_grad_l1_norm','l4_grad_l1_norm'}
  weight_norm_tags = {'l1_l1_norm','l2_l1_norm','l3_l1_norm','l4_l1_norm'}
  tags_1 = {'avg_loss', 'avg_q','epsilon'}
  tags_2 = {'num_game'}

  filename = "../models/2016-07-18_15-37-45" if filename is None else filename
  open_loc = "%s/%s" % (filename, csv_loc)
  save_loc = "%s/%s" % (filename, pdf_loc)
  with open(open_loc) as csvfile:
    reader = csv.DictReader(csvfile)
    avg_reward = []
    avg_loss = []
    avg_q = []
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

    succ_counts =[]
    lr =[]
    step = []

    for tag in q_tags:
        tag_str = tag + ' = []'
        exec(tag_str)

    for row in reader:
        avg_reward.append(row['avg_reward'])
        avg_loss.append(float(row['avg_loss']))
        avg_q.append(row['avg_q'])
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
        for tag in q_tags:
            tag_str = tag +'.append(row[\''+ tag+'\'])'
            eval(tag_str)

        succ_counts.append(row['succ_counts'])
        lr.append(row['lr'])
        step.append(row['step'])
  pp = PdfPages(save_loc)
  # counts
  for tag in counts_tag:
      plt.figure()
      plt.title(tag)
      plt.plot(step,eval(tag), label=tag)
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

  number = 10
  cmap = plt.get_cmap('brg')
  colors = [cmap(i) for i in np.linspace(0, 1, number)]


  # q eval a=0
  plt.figure()
  plt.title('q(s,a=0)')
  for s in range(0,10):
      q_str = 'q'+str(s)+'_0'
      q_tag = 'q(s='+str(s)+',a=0)'
      plt.plot(step,eval(q_str),color=colors[s], label=q_tag)
      plt.xlabel('step')
  plt.legend(loc='best')
  plt.savefig(pp, format='pdf')
  plt.close()

  # q eval a=1
  plt.figure()
  plt.title('q(s,a=1)')
  for s in range(0,10):
      q_str = 'q'+str(s)+'_1'
      q_tag = 'q(s='+str(s)+',a=1)'
      plt.plot(step, eval(q_str), color=colors[s], label=q_tag)
      plt.xlabel('step')
  plt.legend(loc='best')
  plt.savefig(pp, format='pdf')
  plt.close()







  pp.close()

if __name__ == "__main__":
  plot(sys.argv[1] if len(sys.argv) > 1 else None)