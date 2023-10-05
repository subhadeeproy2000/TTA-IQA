# Test Time Adaptation for Blind Image Quality Assessment (ICCV 2023)
This is the official project  repository containing the implementation Code for [Test Time Adaptation for Blind Image Quality Assessment (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Roy_Test_Time_Adaptation_for_Blind_Image_Quality_Assessment_ICCV_2023_paper.pdf) by Subhadeep Roy, Shankhanil Mitra, Soma Biswas, and Rajiv Soundararajan.  

Our paper addresses the issue of the distribution shift across various image-quality databases and proposes methods to adapt the pre-trained model at test time in the absence of source data. The source model needs to be updated based on a self-supervised auxiliary task to learn the distribution shift between train and test data. We formulate novel self-supervised auxiliary tasks using the rank and group contrastive losses, which can learn quality-aware information from the test data.
![Block diagram of a general architecture for test time adaptation](https://github.com/subhadeeproy2000/TTA-IQA/assets/64764444/9213d944-23ba-4cdd-b49d-cc3e8e12678b)
# Installation 
```
conda env create -f environment.yml
```
# Datasets
We have used mainly 6 datasets for evaluation ( [KonIQ-10k](http://database.mmsp-kn.de/koniq-10k-database.html) , [PIPAL](https://github.com/HaomingCai/PIPAL-dataset) , [CID2013](https://zenodo.org/record/2647033) , [LIVE-IQA](https://live.ece.utexas.edu/research/quality/subjective.htm) , [SPAQ](https://github.com/h4nwei/SPAQ) , [LIVEC](https://live.ece.utexas.edu/research/ChallengeDB/) ).
# Pretrained models
To download the pretrained model, run the following python scripts  

TReS:
```
python3 TTA-IQA/TReS/weight/download_fblive.py
```
MUSIQ:
```
python3 TTA-IQA/MUSIQ/weights/download_fblive.py
python3 TTA-IQA/MUSIQ/model/download_fblive_resnet.py
```
HyperIQA:
```
python3 TTA-IQA/hyperIQA/weight/download_fblive.py
```
MetaIQA:
```
python3 TTA-IQA/MetaIQA/model_IQA/download.py
```
# Run Code for Four Different Models
TReS:
```
python3 TTA-IQA/TReS/ALL_EXPT.py
```
MUSIQ:
```
python3 TTA-IQA/MUSIQ/ALL_EXPT.py
```
HyperIQA:
```
python3 TTA-IQA/hyperIQA/ALL_EXPT.py
```
MetaIQA:
```
python3 TTA-IQA/MetaIQA/ALL_EXPT.py
```
# Generalize for any model
You need to write a code for developing the model. In the given demo, we write a code in TTA-IQA\General TTA\models.py to design the architecture of TReS model.Then one need to specify where model should be classified into feature extractor and regressor ( In our case, two examples are shown. If we do not include transformer as a part of feature extractor then we will use SSHead.py, otherwise we will use SSHead_tf.py. Rest all the codes are same. Just need to look up whether you have a csv file along with names of the images in 'image_name' column and their corresponding MOS score in 'MOS' column. After providing all these codes, you can run tta_inference.py . For example -
```
python3 TTA-IQA/General TTA/tta_inference.py --run 3 --batch_size 8 --lr 0.001 --niter 3 --svpath TTA-IQA/ttt_cifar_IQA/weight/fblive_TReS --gpunum 0 --test_patch_num 1 --fix_ssh --datapath DSLR --rank --group_contrastive
```

# Acknowledgement 
The main code for the model TReS, MUSIQ , HyperIQA , MetaIQA is borrowed from [TReS](https://github.com/isalirezag/TReS), [MUSIQ](https://github.com/anse3832/MUSIQ), [hyperIQA](https://github.com/SSL92/hyperIQA), [MetaIQA](https://github.com/zhuhancheng/MetaIQA) respectively.

# Citation 
```
@InProceedings{Roy_2023_ICCV,
    author    = {Roy, Subhadeep and Mitra, Shankhanil and Biswas, Soma and Soundararajan, Rajiv},
    title     = {Test Time Adaptation for Blind Image Quality Assessment},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {16742-16751}
}
```
