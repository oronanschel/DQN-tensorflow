import matplotlib.pyplot as plt
import csv, pdb, sys
from matplotlib.backends.backend_pdf import PdfPages
import  numpy as np

# folder_name = 'p:1_heads_num:1_screen_1_hot:False'
# folder_name = 'p:1_heads_num:1_screen_1_hot:True'
# folder_name = 'p:0.5_heads_num:20_screen_1_hot:True'
folder_name = 'p:0.5_heads_num:20_screen_1_hot:False'

pdf_loc = "training.pdf"
csv_loc = "training_progress.csv"

filename = 'models/'+folder_name
open_loc = "%s/%s" % (filename, csv_loc)
save_loc = "%s/%s" % (filename, pdf_loc)

with open(open_loc) as csvfile:
    reader = csv.DictReader(csvfile)
    tags = {'chain_len','p','tot_ep_until_100','heads_num','feature_1_hot'}

    chain_len = []
    num_game = []
    p = []
    heads = []
    f_1_hot = []

    for row in reader:
        chain_len.append(row['chain_len'])
        num_game.append(row['tot_ep_until_100'])
        p.append(row['p'])
        heads.append(row['heads_num'])
        f_1_hot.append(row['feature_1_hot'])

    p = p[0]
    heads = heads[0]
    f_1_hot = f_1_hot[0]
    chain_len = np.array(map(int,chain_len))
    num_game = np.array(map(int,num_game))
    ind = np.argsort(chain_len,axis=0)
    chain_len = chain_len[np.array(ind)]
    num_game = num_game[np.array(ind)]

    chain_len_uniq = []
    game_num_mid = []
    game_num_vec = []
    for i in range(0,len(chain_len)):
        if i == len(chain_len)-1 or chain_len[i] < chain_len[i+1]:
            game_num_vec.append(num_game[i])
            game_num_mid.append(np.median(game_num_vec))
            chain_len_uniq.append(chain_len[i])
            game_num_vec = []
        else:
            game_num_vec.append(num_game[i])

    last_chain_len = chain_len_uniq[len(chain_len_uniq)-1]
    if(last_chain_len) <100:
        for i in range(last_chain_len+2,100,2):
            game_num_mid.append(2000)
            chain_len_uniq.append(i)



    pp = PdfPages(save_loc)

    if int(heads)>1:
        title_str = 'Bootstrapped DQN' + '\n(p='+p+', K='+heads+')'
    else:
        title_str = 'Standard DQN'

    if f_1_hot == 'True':
        title_str += ', features: 1_hot'
    else :
        title_str += ', features: unari'


    plt.figure()
    plt.grid = True
    plt.title(title_str)
    plt.stem(chain_len_uniq, game_num_mid)
    plt.xlabel('Chain length')
    plt.ylabel('Total episodes until 100 optimal episodes (median of 3 seeds)')
    plt.savefig(pp, format='pdf')
    plt.close()
    pp.close()