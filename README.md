# breast_tumor_segmentation

## Prerequisites

+ Linux
+ Python with numpy
+ NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
+ pytorch
+ torchvision

+ Dataset Folder

Please put your image inside folder "a" and binary mask into "b" in both train and test folders.

+ Train the model:

    python train --dataset breast --nEpochs 200 --cuda

+ Test the model:

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
