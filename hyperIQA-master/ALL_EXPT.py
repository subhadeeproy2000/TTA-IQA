import os
import time

def main(s1,s2):
    t = []
    t.append(time.time())
    os.system(f'{s1} {s2} --test')
    t.append(time.time() - t[0])
    os.system(f'{s1} {s2} --rank')
    t.append(time.time() - t[0])
    os.system(f'{s1} {s2} --group_contrastive')
    t.append(time.time() - t[0])
    os.system(f'{s1} {s2} --rank --group_contrastive')
    t.append(time.time() - t[0])
    os.system(f'{s1} {s2} --rotation')
    t.append(time.time() - t[0])

    print(f' Time Taken for all experiment : {t[1:]}')
    pass

if __name__ == '__main__':

    s1 = 'python train_test_IQA.py --seed 2021 --svpath ./weights  --fix_ssh --run 3'

    s2 = '--datapath DSLR --dataset dslr'
    main(s1, s2)

    s2 = '--datapath LIVE --dataset live'
    main(s1,s2)

    s2 = '--datapath CID2013 --dataset cidiq'
    main(s1,s2)

    s2 = '--datapath KONIQ --dataset koniq'
    main(s1,s2)

    s2 = '--datapath PIPAL --dataset pipal'
    main(s1,s2)

    s2 = '--datapath CLIVE --dataset clive'
    main(s1,s2)

    # s2 = '--datapath SPAQ --dataset spaq'
    # main(s1,s2)