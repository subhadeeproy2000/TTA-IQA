# TTA-IQA
Code for Test Time Adaptation in the Context of Blind Image Quality Assessment.
# Installation 
```
conda env create -f environment.yml
```
# Datasets
We have used mainly 6 datasets for evaluation ( [KonIQ-10k](http://database.mmsp-kn.de/koniq-10k-database.html) , [PIPAL](https://github.com/HaomingCai/PIPAL-dataset) , [CID2013](https://zenodo.org/record/2647033) , [LIVE-IQA](https://live.ece.utexas.edu/research/quality/subjective.htm) , [SPAQ](https://github.com/h4nwei/SPAQ) , [LIVEC](https://live.ece.utexas.edu/research/ChallengeDB/).
# Run Code for Four Different Models
TReS:
```
python3 TTA-IQA/ttt_cifar_IQA/ALL_EXPT.py
```
MUSIQ:
```
python3 TTA-IQA/MUSIQ-main/ALL_EXPT.py
```
HyperIQA:
```
python3 TTA-IQA/hyperIQA-master/ALL_EXPT.py
```
MetaIQA:
```
python3 TTA-IQA/MetaIQA-master/ALL_EXPT.py
```
# Generalize for any model
You need to write a code for developing the model. In the given demo, we write a code in TTA-IQA\General TTA\models.py to design the architecture of TReS model.Then one need to specify where model should be classified into feature extractor and regressor ( In our case, two examples are shown. If we do not include transformer as a part of feature extractor then we will use SSHead.py, otherwise we will use SSHead_tf.py. Rest all the codes are same. Just need to look up whether you have a csv file along with names of the images in 'image_name' column and their corresponding MOS score in 'MOS' column. After providing all these codes, you can run tta_inference.py . For example -
```
python3 TTA-IQA/General TTA/tta_inference.py --run 3 --batch_size 8 --lr 0.001 --niter 3 --svpath TTA-IQA/ttt_cifar_IQA/weight/fblive_TReS --gpunum 0 --test_patch_num 1 --fix_ssh --datapath DSLR --rank --group_contrastive
```

# Acknowledgement 
The main code for the model TReS, MUSIQ , HyperIQA , MetaIQA is borrowed from [TReS](https://github.com/isalirezag/TReS) [MUSIQ](https://github.com/anse3832/MUSIQ) [hyperIQA](https://github.com/SSL92/hyperIQA) [MetaIQA](https://github.com/zhuhancheng/MetaIQA) respectively.
