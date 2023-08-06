import os
import time

def main(s1,s2):
    t = []
    t.append(time.time())
    os.system(f'{s1} {s2} --test_only')
    t.append(time.time() - t[0])
    # os.system(f'{s1} {s2} --comp')
    # t.append(time.time() - t[0])
    # os.system(f'{s1} {s2} --blur')
    # t.append(time.time() - t[0])
    # os.system(f'{s1} {s2} --nos')
    # t.append(time.time() - t[0])
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

    s1 = 'python test_adapt.py --run 1 --batch_size 8 --lr 0.001 --niter 3 --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network resnet50 --svpath ./weight --gpunum 0 --test_patch_num 1 --train_patch_num 1 --train_data live --fix_ssh'
    s2 = '--datapath DSLR --dataset dslr'
    main(s1, s2)

    # Without transformer Test Time Adaptation

    s1 = 'python test_adapt.py --run 3 --batch_size 8 --lr 0.001 --niter 3 --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network resnet50 --svpath ./weight --gpunum 0 --test_patch_num 1 --train_patch_num 1 --train_data fblive --fix_ssh'

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

    # With transformer Test Time Adaptation

    s1 = 'python test_adapt.py --tta_tf --ln --bn --batch_size 8 --lr 0.001 --niter 3 --num_encoder_layerst 2 --dim_feedforwardt 64 --nheadt 16 --network resnet50 --svpath ./weight --gpunum 0 --test_patch_num 1 --train_patch_num 1 --train_data fblive --fix_ssh'

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