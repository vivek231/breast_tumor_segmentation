# Breast tumor segmentation and shape classification in mammograms using generative adversarial and convolutional neural network
Mammogram inspection in search of breast tumors is a tough assignment that radiologists must carry out frequently. Therefore, image analysis methods are needed for the detection and delineation of breast tumors, which portray crucial morphological information that will support reliable diagnosis. In this paper, we proposed a conditional Generative Adversarial Network (cGAN) devised to segment a breast tumor within a region of interest (ROI) in a mammogram. The generative network learns to recognize the tumor area and to create the binary mask that outlines it. In turn, the adversarial network learns to distinguish between real (ground truth) and synthetic segmentations, thus enforcing the generative network to create binary masks as realistic as possible. The cGAN works well even when the number of training samples are limited. As a consequence, the proposed method outperforms several state-of-the-art approaches. Our working hypothesis is corroborated by diverse segmentation experiments performed on INbreast and a private in-house dataset. The proposed segmentation model, working on an image crop containing the tumor as well as a significant surrounding area of healthy tissue (loose frame ROI), provides a high Dice coefficient and Intersection over Union (IoU) of 94% and 87%, respectively. In addition, a shape descriptor based on a Convolutional Neural Network (CNN) is proposed to classify the generated masks into four tumor shapes: irregular, lobular, oval and round. The proposed shape descriptor was trained on DDSM, since it provides shape ground truth (while the other two datasets does not), yielding an overall accuracy of 80%, which outperforms the current state-of-the-art.

## Prerequisites

+ Linux
+ Python with numpy
+ NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
+ PyTorch
+ TorchVision

## Dataset Folder

+ Dataset path will follow the structure: Dataset/breast/train/a-b folders, Dataset/breast/test/a-b folders, 

+ Please put your images inside folder "a" and binary mask into "b" in both train and test folders.

## Train the model:

    python train --dataset breast --nEpochs 200 --cuda

## Test the model:

    python test.py --dataset breast --model checkpoint/breast/netG_model_epoch_200.pth --cuda

## Citation:
If you use the code in your work, please use the following citation:
```
@article{singh2020breast,
  title={Breast tumor segmentation and shape classification in mammograms using generative adversarial and convolutional neural network},
  author={Singh, Vivek Kumar and Rashwan, Hatem A and Romani, Santiago and Akram, Farhan and Pandey, Nidhi and Sarker, Md Mostafa Kamal and Saleh, Adel and Arenas, Meritxell and Arquez, Miguel and Puig, Domenec and others},
  journal={Expert Systems with Applications},
  volume={139},
  pages={112855},
  year={2020},
  publisher={Elsevier}
}
```
