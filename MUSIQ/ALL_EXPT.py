import os
import time

def main(s1,s2):
    t = []
    t.append(time.time())
    os.system(f'python test.py {s2}')
    t.append(time.time() - t[0])
    os.system(f'{s1} {s2} --rank')
    t.append(time.time() - t[0])
    os.system(f'{s1} {s2} --gc')
    t.append(time.time() - t[0])
    os.system(f'{s1} {s2} --rank --gc')
    t.append(time.time() - t[0])
    os.system(f'{s1} {s2} --rot')
    t.append(time.time() - t[0])

    print(f' Time Taken for all experiment : {t[1:]}')
    pass

if __name__ == '__main__':

    # Without transformer Test Time Adaptation

    s1 = 'python test_time_training.py'

    s2 = '--database dslr --datapath DSLR'
    main(s1, s2)

    s2 = '--database LIVE  --datapath LIVE'
    main(s1, s2)

    s2 = '--database cid  --datapath CID2013'
    main(s1, s2)

    s2 = '--database KonIQ-10k  --datapath KONIQ'
    main(s1, s2)

    s2 = '--database PIPAL  --datapath PIPAL'
    main(s1, s2)

    # With transformer Test Time Adaptation

    s1='python test_time_training_tra.py --bn --ln --cls'
    s2 = '--database LIVE  --datapath LIVE'
    main(s1,s2)

    s2 = '--database cid  --datapath CID2013'
    main(s1,s2)

    s2 = '--database KonIQ-10k  --datapath KONIQ'
    main(s1,s2)

    s2 = '--database PIPAL  --datapath PIPAL'
    main(s1,s2)
