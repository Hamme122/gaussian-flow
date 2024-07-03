# 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

## 1. Introduction

This repository is an unofficial implementation of "[Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle](https://arxiv.org/abs/2312.03431)", building on top of the [4D Gaussian Splatting for Real-Time Dynamic Scene Rendering](https://github.com/hustvl/4DGaussians) framework.

To-do list:

- [x] Poly_fourier class
- [x] Timestamp scaling
- [ ] Loss: We tried two loss implementations, but we didn't get good results



> Notice: This repository is based on the code from 4DGaussian and still retains some functions and parameters from 4DGaussian that have not been completely removed. I hope this does not cause any misunderstanding. You can use the debug mode to better understand how the poly_fourier and deformation model are handled in this repository. Thank you！

## 2. Result

We test the model on four datasets:

| PSNR           | dynerf/flame steak | dynerf/flame salmon | hypernerf/broom | hypernerf/chicken |
| -------------- | ------------------ | ------------------- | --------------- | ----------------- |
| Paper(30k)     | blank              | blank               | 22.5            | 29.4              |
| This repo(30k) | 32.2               | 28.6                | 21.8            | 26.3              |

We write a more detailed analysis of the results in part 9 Reproduction note.

## 3. Environmental Setups

Please follow the [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) to install the relative packages.

```bash
git clone https://github.com/hustvl/4DGaussians
cd 4DGaussians
git submodule update --init --recursive
conda create -n Gaussians4D python=3.7 
conda activate Gaussians4D

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

In our environment, we use pytorch=1.13.1+cu116.

## 4. Data Preparation

**For real dynamic scenes:**
The dataset provided in [HyperNeRF](https://github.com/google/hypernerf) is used. You can download scenes from [Hypernerf Dataset](https://github.com/google/hypernerf/releases/tag/v0.1) and organize them as [Nerfies](https://github.com/google/nerfies#datasets). 

Meanwhile, we still use [Dynerf](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0) dataset, his name has many variations, also known as the Plenoptic Video dataset and the Neural 3D Video Synthesis Dataset.

The code structure of the entire dataset is shown below, and in the Training section, we also explained how to preprocess these datasets.

```
├── data
│   | hypernerf
│     ├── interp
│     ├── misc
│     ├── virg
│   | dynerf
│     ├── cook_spinach
│       ├── cam00
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── 0002.png
│               ├── ...
│       ├── cam01
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── ...
│     ├── cut_roasted_beef
|     ├── ...
```


## 5. Training

For training dynerf scenes such as `cut_roasted_beef`, run

```python
# First, extract the frames of each video.
python scripts/preprocess_dynerf.py --datadir data/dynerf/cut_roasted_beef
# Second, generate point clouds from input data.
bash colmap.sh data/dynerf/cut_roasted_beef llff
# Third, downsample the point clouds generated in the second step.
python scripts/downsample_point.py data/dynerf/cut_roasted_beef/colmap/dense/workspace/fused.ply data/dynerf/cut_roasted_beef/points3D_downsample2.ply
# Finally, train.
python train.py -s data/dynerf/cut_roasted_beef --port 6017 --expname "dynerf/cut_roasted_beef" --configs arguments/dynerf/cut_roasted_beef.py 
```

For training hypernerf scenes such as `virg/broom`: Pregenerated point clouds by COLMAP are provided [here](https://drive.google.com/file/d/1fUHiSgimVjVQZ2OOzTFtz02E9EqCoWr5/view). Just download them and put them in to correspond folder, and you can skip the former two steps. Also, you can run the commands directly.

```python
# First, computing dense point clouds by COLMAP
bash colmap.sh data/hypernerf/virg/broom2 hypernerf
# Second, downsample the point clouds generated in the first step. 
python scripts/downsample_point.py data/hypernerf/virg/broom2/colmap/dense/workspace/fused.ply data/hypernerf/virg/broom2/points3D_downsample2.ply
# Finally, train.
python train.py -s  data/hypernerf/virg/broom2/ --port 6017 --expname "hypernerf/broom2" --configs arguments/hypernerf/broom2.py 
```

## 8. Citation

If you find this repository/work helpful in your research, welcome to cite these papers and give a ⭐.

```
@article{wu20234dgaussians,
  title={4D Gaussian Splatting for Real-Time Dynamic Scene Rendering},
  author={Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and Zhang, Xiaopeng and Wei Wei and Liu, Wenyu and Tian, Qi and Wang Xinggang},
  journal={arXiv preprint arXiv:2310.08528},
  year={2023}
}

@article{lin2023gaussian,
  title={Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle},
  author={Lin, Youtian and Dai, Zuozhuo and Zhu, Siyu and Yao, Yao},
  journal={arXiv preprint arXiv:2312.03431},
  year={2023}
}
```



## 9. Reproduction notes

> Here are some notes and issues I took during the repro process:

**The code structure of 4DGaussian:**

![The code structure of 4DGaussian](https://lzztypora.oss-cn-beijing.aliyuncs.com/202407022150193.png)

**The code structure of 3DGaussian:**

![Gaussian splatting](https://lzztypora.oss-cn-beijing.aliyuncs.com/202407022154724.png)

### 9.1 The role of each component

#### 9.1.1 Polyfourier

Based on the 4DGaussian framework, we changed the part of the deformation to be implemented with pooly_fourier. After changing only this section, we have listed the results below.

We followed the 4DGaussian practice of initializing 3000 times with 3DGaussian and then training 14000 times with the deformation model. The first line is the result of the 4DGaussian three-replicate trial. The second line is the result of the second phase of 14,000 iterations still using 3DGaussian. The third line is the results of three repeated tests adding only the poly_fourier module and we use init.uniform_() there.

| 4DGaussian splatting                      | 1                                                            | 2                                                            | 3                                                            | Mean       |
| ----------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------- |
| [ITER 14000] Evaluating test              | L1 0.015267772873973144 PSNR 33.31632389741785 [04/06 16:15:24] | L1 0.015912557623403913 PSNR 32.99937483843635 [24/06 22:23:27] | L1 0.01650691098150085 PSNR 32.79038821949678 [25/06 20:07:31] | PSNR: 33.0 |
| [ITER 14000] Evaluating train             | L1 0.009441451553036185 PSNR 35.70967663035673 [04/06 16:15:24] | L1 0.01016127282534452 PSNR 35.32027132370893 [24/06 22:23:27] | L1 0.009234793155508883 PSNR 35.91758840224322 [25/06 20:07:31] | PSNR: 35.6 |
| 3DGaussian splatting                      |                                                              |                                                              |                                                              |            |
| [ITER 14000] Evaluating test              | L1 0.02249857860014719 PSNR 28.73034712847541 [06/06 21:46:20] |                                                              |                                                              | PSNR: 28.7 |
| [ITER 14000] Evaluating train             | L1 0.01404128092176774 PSNR 29.94566872540642 [06/06 21:46:20] |                                                              |                                                              | PSNR: 29.9 |
| gaussian-flow+ Polyfourier(init.uniform_) |                                                              |                                                              |                                                              |            |
| [ITER 14000] Evaluating test:             | L1 0.02227828837931156 PSNR 30.207836712107937 [24/06 14:32:32] | L1 0.020741680746569353 PSNR 30.47933847763959 [24/06 21:15:11] | L1 0.01639221520984874 PSNR 31.640568677116843 [24/06 22:19:50] | PSNR: 30.8 |
| [ITER 14000] Evaluating train:            | L1 0.014463305254192912 PSNR 31.37994452083812 [24/06 14:32:33] | L1 0.014745961710372391 PSNR 31.32480856951545 [24/06 21:15:11] | L1 0.01534967650385464 PSNR 31.13037042056813 [24/06 22:19:50] | PSNR: 31.3 |

From the results, we can see that the addition of poly_fourier has a slight effect, but it is not very good

#### 9.1.2 Adaptive Timestemp Scaling

After completing this part, the model has been significantly improved.

| gaussian-flow  + Polyfourier(init.uniform_) + Adaptive Timestemp Scaling |                                                              |                                                              |                                                              | Mean       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------- |
| [ITER 14000] Evaluating test:                                | L1 0.018394825651365167 PSNR 31.200657900641946 [30/06 10:45:32] | L1 0.018614117823102894 PSNR 31.104160196640912 [29/06 22:27:39] | L1  0.01820581713143517 PSNR 31.3098243264591 [29/06 23:08:58] | PSNR: 31.2 |
| [ITER 14000] Evaluating train:                               | L1 0.012931905905990039 PSNR 32.26852719924029 [30/06 10:45:32] | L1 0.013267592088702847 PSNR 32.12875556945801 [29/06 22:27:39] | L1 0.013322525822064456 PSNR 32.18548774719238 [29/06 23:08:58] | PSNR: 32.2 |

Then, after adding the timestemp scaling, we changed the previous poly_fourier to the way the parameters were initialized, and found that the results were also significantly improved.

| gaussian-flow  + Polyfourier(nn.init.kaiming_normal_) + Adaptive Timestemp Scaling |                                                              |                                                              |                                                              |                                                              | Mean       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------- |
| [ITER 14000] Evaluating test:                                | L1 0.015813744681722978 PSNR 32.75542158239028 [03/07 12:49:49] | L1 0.017934783197501126 PSNR 31.938018013449277 [03/07 13:37:28] | L1 0.017712286837837276 PSNR 31.958147609935086 [03/07 13:58:10] | L1 0.017179548411684877 PSNR 32.08180719263413 [03/07 20:11:37] | PSNR: 32.2 |
| [ITER 14000] Evaluating train:                               | L1 0.010392635622445275 PSNR 34.162504757151886 [03/07 12:49:49] | L1 0.011557015426018658 PSNR 33.56630123362822 [03/07 13:37:28] | L1 0.010980667491607806 PSNR 33.88952838673311 [03/07 13:58:10] | L1 0.011120671406388283 PSNR 33.69730983060949 [03/07 20:11:37] | PSNR: 33.9 |

#### 9.1.3 Loss

1. Time smooth loss:

   We tried two approaches to handle the L2 norm calculation in time smooth: one is the L2 norm of the matrix, and the other is to compute the L2 norm for each Gaussian point, then sum and average them. We also attempted loss scaling to make the new loss and L1 loss of the same magnitude, but neither approach improved the model's performance😨😨.

   | smoothloss                     | *0.01                                                        |                                                              | matrix L2 norm                                               |                                                              |
   | ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | [ITER 14000] Evaluating test:  | L1 0.0174745248959345 PSNR 31.27042646969066 [02/07 20:42:42] | L1 0.02302054447286269 PSNR 29.289218678193933 [03/07 12:07:41] | L1 0.018887575934914982 PSNR 30.931609209846048 [02/07 21:30:09] | L1 0.01737015475245083 PSNR 31.317197350894702 [03/07 11:31:31] |
   | [ITER 14000] Evaluating train: | L1 0.01472287208718412 PSNR 31.62007219651166 [02/07 20:42:42] | L1 0.01496632769703865 PSNR 30.929186091703528 [03/07 12:07:41] | L1 0.014322300058077364 PSNR 31.767316593843347 [02/07 21:30:09] | L1 0.01497127438950188 PSNR 31.56536640840418 [03/07 11:31:31] |

2. KNN rigid loss

   For this part, we still used two methods. One was extracted from the paper "Gaussian Grouping: Segment and Edit Anything in 3D Scenes," but both approaches failed. At present, the PSNR on the test set is only around 32, which cannot be further improved.

   | KNNloss1                       |                                                              | *0.1                                                         |
   | ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | [ITER 14000] Evaluating test:  | L1 0.01581644031273968 PSNR 32.45481390111586 [03/07 17:02:45] | L1 0.017660899745190844 PSNR 31.88553294013528 [03/07 17:19:24] |
   | [ITER 14000] Evaluating train: | L1 0.014082024531329378 PSNR 32.73800086975098 [03/07 17:02:45] | L1 0.012039224617183208 PSNR 33.401686163509595 [03/07 17:19:24] |

   | KNNloss2 from Gaussian Grouping |                                                              |                                                              |
   | ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | [ITER 28000] Evaluating test:   | L1 0.01742225107462967 PSNR 31.908257316140567 [03/07 22:56:19] | L1 0.017317032222362125 PSNR 32.121478249044976 [03/07 23:54:04] |
   | [ITER 28000] Evaluating train:  | L1 0.01062560300616657 PSNR 34.034979651956 [03/07 22:56:20] | L1 0.010371644378584973 PSNR 34.14780381146599 [03/07 23:54:04] |

#### 9.1.4 Summary

1. In this implementation, we found that using Adaptive Timestemp Scaling is a very effective way to solve the problem of normalization of time in 4D Gaussian. 
2. In addition, the initialization method also has a great impact on the results
3. In 3DGaussian, the opacity is reset every 3000 times, but this method will have a bad effect in 4DGaussian