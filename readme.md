# Action Recognition
In this repo we study the problem of action recognition(recognizing actions in videos) on UCF101 famous dataset.

Here, I reimplemented two-stream approach for action recognition using pre-trained Xception networks for both streams(Look at references).  

# Get started:
A full demo of the code in the repo can be found in **Action_Recognition_Walkthrough.ipynb** notebook.

Please clone **Action_Recognition_Walkthrough.ipynb** notebook to your drive account and run it on Google Colab on python3 GPU-enabled instance.

## Environment and requirements:
This code requires python 3.6,
 ```
 Tensorflow 1.11.0 (GPU enabled-the code uses keras associated with Tensorflow)
 Imgaug 0.2.6
 opencv 3.4.2.17
 numpy 1.14.1
 ```
All of these requirements are satisfied by (python3 Colab GPU-enabled instance) Just use it and the notebook **Action_Recognition_Walkthrough.ipynb** will install the rest :)


## Dataset:
I used UCF101 dataset originally found [here](http://crcv.ucf.edu/data/UCF101/UCF101.rar).

Also the dataset is processed and published by [feichtenhofer/twostreamfusion](https://github.com/feichtenhofer/twostreamfusion))
  * RGB images(single zip file split into three parts)
  ```
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003
  ```
  * Optical Flow u/v frames(single zip file split into three parts)
  ```
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.001
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.002
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.003
  ```
 
## Code Features:
* You have variety of models to exchange between them easily.
* Saves checkpoints on regular intervals and those checkpoints are synchronized to google drive using Drive API which means you can resume training anywhere for any Goggle Colab Instance.
* Accesses the public models on my drive and you can resume and fine-tune them at different time stamps.
Where the name of every checkpoint is as follows, **EPOCH.BEST_TOP_1_ACC.CURRENT_TOP_1_ACC**
for example [this](https://drive.google.com/open?id=1N697z8uvAHICBbFNOJyKn4nbT64rUTcB)
which is **300-0.84298-0.84166**.zip in folder **heavy-mot-xception-adam-1e-05-imnet**
at this checkpoint,
    * **epoch=300**
    * **best top 1 accuracy was 0.84298** (obtained in checkpoint before 300)
    * **the current accuracy is 0.84166** 
    * in the experiment **heavy-mot-xception-adam-1e-05-imnet**
## Models:
I used pre-trained models on imagenet provided by keras applications [here](https://keras.io/applications/).

The best results are obtained using Xception architecture.


 Network      | Top1-Acc  |
--------------|:-------:|
Spatial VGG19 stream    | ~75%  | 
Spatial Resnet50 stream | 81.2% | 
Spatial Xception stream | 86.04%|
------------------------|-------|
Motion Resnet50 stream  | ~75%  | 
Motion xception stream  | 84.4% | 
------------------------|-------|
Average fusion| **91.25%**  | 
------------------------|-------|
Recurrent network fusion| **91.7%** | 

## Pre-trained Model
All the pre-trained models could be found [here](https://drive.google.com/drive/folders/1B82anWV8Mb4iHYmOp9tIR9aOTlfllwsD).

It's the same drive folder accessed by the code while training and resuming training from a checkpoint.

## Reference Paper:
* [[1] Two-stream convolutional networks for action recognition in videos](https://arxiv.org/pdf/1406.2199.pdf)
* [[2] Real-time Action Recognition with Enhanced Motion Vector CNNs](https://arxiv.org/pdf/1604.07669.pdf)
* [[3] Towards Good Practices for Very Deep Two-Stream ConvNets](https://arxiv.org/pdf/1507.02159.pdf)


## Nice implementations of two-stream approach:
* [[1] Nice two-stream reimplementation using pytorch using resnets](https://github.com/jeffreyhuang1/two-stream-action-recognition)
My code is inspired by this repo.
* [[2] Two-stream-pytorch](https://github.com/bryanyzhu/two-stream-pytorch)
* [[3] Hidden-Two-Stream](https://github.com/bryanyzhu/Hidden-Two-Stream)


## Future directions: 
* [[1] Hidden-Two-stream](https://arxiv.org/pdf/1704.00389.pdf)
Which achieves real-time performance by using a deep neural net for generating the optical flow.
* [[2] Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?](https://arxiv.org/pdf/1711.09577.pdf)
Discuses how 3d convolutions is the perfect architecture for videos and Kinetics dataset pre-training could retrace imagenet pre-training.
* [[3] Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/pdf/1705.07750.pdf)

## Useful links:
* [[1] awesome-action-recognition](https://github.com/jinwchoi/awesome-action-recognition)
