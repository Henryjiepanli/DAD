
# DAD: Difference-Aware Decoder for Binary Segmentation

We are delighted to share that our paper has been successfully accepted by the IEEE Transactions on Circuits and Systems for Video Technology (TCSVT 2025).[Paper Link](https://ieeexplore.ieee.org/document/11175179).

**Abstract**: Inspired by the way human eyes detect objects, we propose a new unified dual-branch decoder paradigm, termed the Difference-Aware Decoder (DAD), designed to explore the differences between foreground and background effectively, thereby enhancing the separation of objects of interest in optical images. The DAD operates in two stages, leveraging multi-level features from the encoder. In the first stage, it achieves coarse detection of foreground objects by utilizing high-level semantic features, mimicking the initial rough observation typical of human vision. In the second stage, the decoder refines segmentation by examining differences in low-level features, guided by the coarse map generated in the first stage.

This repository contains the code for our paper:  
**[Towards Complex Backgrounds: A Unified Difference-Aware Decoder for Binary Segmentation](https://arxiv.org/abs/2210.15156)**.

## Training Instructions

To train the DAD model, follow these steps:

1. Set the task (COD/SOD/Poly/MSD), batch size, and specify the GPU for training. Execute the following commands:

   ```bash
   python train.py --gpu_id 0 --task COD --batchsize 8 --backbone resnet
   python train.py --gpu_id 0 --task COD --batchsize 8 --backbone res2net
   python train.py --gpu_id 0 --task COD --batchsize 8 --backbone v2_b2
   python train.py --gpu_id 0 --task COD --batchsize 8 --backbone v2_b4

   python train.py --gpu_id 0 --task SOD --batchsize 8 --backbone resnet
   python train.py --gpu_id 0 --task SOD --batchsize 8 --backbone res2net
   python train.py --gpu_id 0 --task SOD --batchsize 8 --backbone v2_b2

   python train.py --gpu_id 0 --task Poly --batchsize 8 --backbone v2_b2
   python train.py --gpu_id 0 --task MSD --batchsize 8 --backbone v2_b2
   ```

## Inference Code and Pretrained Models

We provide inference code along with pretrained and trained models. You can download them using the links below:

- **Pretrained Models**:
  - **ResNet-50**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)
  - **Res2Net-50**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)
  - **PVT-v2-b2**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)

- **Trained Models**:
  - **COD**:
    - **ResNet-50**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)
    - **Res2Net-50**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)
    - **PVT-v2-b2**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)
    - **PVT-v2-b4**: [Download]( https://pan.baidu.com/s/10skNCkRxHybFiygGMXSiMA?pwd=dadp)
  - **SOD**:
    - **ResNet-50**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)
    - **Res2Net-50**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)
    - **PVT-v2-b2**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)
  - **Polyp Segmentation**:
    - **PVT-v2-b2**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)
  - **Mirror Detection**:
    - **PVT-v2-b2**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)

To test the trained models, run the following command:

```bash
python test.py --task COD --backbone resnet --pth_path './Experiments/DAD/'
python test.py --task COD --backbone res2net --pth_path './Experiments/DAD/'
python test.py --task COD --backbone v2_b2 --pth_path './Experiments/DAD/'
python test.py --task COD --backbone v2_b4 --pth_path './Experiments/DAD/'
python test.py --task SOD --backbone resnet --pth_path './Experiments/DAD/'
python test.py --task SOD --backbone res2net --pth_path './Experiments/DAD/'
python test.py --task SOD --backbone v2_b2 --pth_path './Experiments/DAD/'
```

## Visual Results for Multiple Tasks and Backbones

We have released visual results for various tasks using different backbones. You can access them from the following links:

### Camouflaged Object Detection (COD)

- **ResNet-50**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)
- **Res2Net-50**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)
- **PVT-v2-b2**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)
- **PVT-v2-b4**: [Download]( https://pan.baidu.com/s/10skNCkRxHybFiygGMXSiMA?pwd=dadp)

### Salient Object Detection (SOD)

- **ResNet-50**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)
- **Res2Net-50**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)
- **PVT-v2-b2**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)

### Mirror Detection

- **PVT-v2-b2**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)

### Polyp Segmentation

- **PVT-v2-b2**: [Download](https://pan.baidu.com/s/1JmgYZXXWsU_6xfnO3tKApA?pwd=xnhz)

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@ARTICLE{11175179,
  author={Li, Jiepan and He, Wei and Lu, Fangxiao and Zhang, Hongyan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Toward Complex Backgrounds: A Unified Difference-Aware Decoder for Binary Segmentation}, 
  year={2026},
  volume={36},
  number={2},
  pages={2372-2386},
  doi={10.1109/TCSVT.2025.3612574}}

}
```
