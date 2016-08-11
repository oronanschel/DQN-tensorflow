import subprocess




chain_len = range(10,30,2)
p = [1]
heads_num = [1]
scree_1_hot = ['True','False']

for _p in p:
    print(_p)
    for _heads_num in heads_num:
        print(_heads_num)
        if (_p==0.5) and _heads_num == 1:
            continue
        for _screen_1_hot in scree_1_hot:
            print(_screen_1_hot)
            folder_name = 'p:'         +str(_p)+'_'\
                         +'heads_num:' +str(_heads_num) + '_'\
                         +'screen_1_hot:'+str(_screen_1_hot)
            for _chain_len in chain_len:
                print(_chain_len)
                for _ in range(0,3):
                    subprocess.call(['python', 'main.py',folder_name,
                                             str(_chain_len),str(_heads_num)
                                             ,str(_p),_screen_1_hot])
