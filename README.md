# CVFT + LPN

### Experiment Dataset
We use two existing dataset to do the experiments

- CVUSA datset: a dataset in America, with pairs of ground-level images and satellite images. All ground-level images are panoramic images.  
	The dataset can be accessed from https://github.com/viibridges/crossnet

- CVACT dataset: a dataset in Australia, with pairs of ground-level images and satellite images. All ground-level images are panoramic images.  
	The dataset can be accessed from https://github.com/Liumouliu/OriCNN


### Dataset Preparation
Please Download the two datasets from above links, and then put them under the director "Data/". The structure of the director "Data/" should be:
"Data/CVUSA/
 Data/ANU_data_small/"

### Models:

There is also an "Initialize" model for your own training step. The VGG16 part in the "Initialize_model" model is initialised by the online model and other parts are initialised randomly. 

Please put them under the director of "Model/" and then you can use them for training or evaluation.


### Codes

1. Training:

	CVUSA: python train_cvusa_lpn.py --multi_loss

	CVACT: python train_cvact_lpn.py --multi_loss

2. Evaluation:

	CVUSA: python test_cvusa.py --multi_loss
	
	CVACT: python test_cvact.py --multi_loss


### Reference  
[Optimal Feature Transport for Cross-View Image Geo-Localization](https://arxiv.org/pdf/1907.05021.pdf)

[github](https://github.com/shiyujiao/cross_view_localization_CVFT.git)

