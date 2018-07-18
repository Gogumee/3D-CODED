# Template-Based Correspondences

This repository contains the source codes for the paper [Correspondences](http://imagine.enpc.fr/~groueixt/correspondences/). The task is to put 2 meshes in point-wise correspondence. Below, given 2 humans scans with holes, the reconstruction are in correspondence (suggested by color).

<img src="README/mesh8.ply.gif" style="zoom:80%" /><img src="README/8RecBestRotReg.ply.gif" style="zoom:80%" />
<img src="README/mesh25.ply.gif" style="zoom:80%" /><img src="README/25RecBestRotReg.ply.gif" style="zoom:80%" />

## Citing this work

If you find this work useful in your research, please consider citing:

```
@inproceedings{groueix2018b,
          title={{Shape correspondences from learnt template-based parametrization}},
          author={Groueix, Thibault and Fisher, Matthew and Kim, Vladimir G. and Russell, Bryan and Aubry, Mathieu},
          year={2018}
        }
```

## Project Page

The project page is available [http://imagine.enpc.fr/~groueixt/correspondences/](http://imagine.enpc.fr/~groueixt/correspondences/)

## Install


#### Clone the repo
```shell
## Download the repository
git clone git@github.com:ThibaultGROUEIX/template-based-correspondences.git
## Create python env with relevant packages
conda create --name pytorch-atlasnet --file aux/spec-file.txt
source activate pytorch-atlasnet
pip install pandas visdom tqdm
conda install pytorch=0.1.12 cuda80 -c soumith #Update cuda80 to cuda90 if relevant
conda install torchvision
```

This implementation uses [Pytorch](http://pytorch.org/). Please note that the Chamfer Distance code doesn't work on  [all versions of pytorch](http://pytorch.org/) because of some weird error with the batch norm layers. It has been tested on v1.12, v3 and a specific commit of v4.
#### Pytorch compatibility

| Python/[Pytorch](http://pytorch.org/) | v1.12           | v2  | v3.1  |  0.4.0a0+ea02833 | 0.4.x latest |
| ------------- |:-------------:| -----:|-----:|-----:| ------------- |
| 2.7 | :heavy_check_mark: :+1: :smiley: | :no_entry_sign: :thumbsdown: :disappointed: | :no_entry_sign: :thumbsdown: :disappointed: | :heavy_check_mark: :+1: :smiley: | 🚫 👎 😞 |
| 3.6 | :heavy_check_mark::+1: :smiley: | ? | ? | :no_entry_sign: :thumbsdown: :disappointed: | 🚫 👎 😞 |

**<u>Recommended</u>** : *Python* **2.7**, *Pytorch* **1.12**

**<u>If you need v4</u>** : 

```shell
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch ; git reset --hard ea02833 #Go to this specific commit that works fine for the chamfer distance
# Then follow pytorch install instruction as usual
```

Developped in python 2.7, so might need a few adjustements for python 3.6. 

#### Build chamfer distance

```shell
cd AtlasNet/nndistance/src
nvcc -c -o nnd_cuda.cu.o nnd_cuda.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ..
python build.py
python test.py
```





## Using the Trained models

The trained models and some corresponding results are also available online :

- [The trained_models](https://cloud.enpc.fr/s/TODO) go in ``` trained_models/```

#### On the demo meshes

Require 3 GB of RAM on the GPU and 17 sec to run (Titan X Pascal). 

```shell
python inference/correspondences.py --cuda 1
```
This script takes as input 2 meshes from ```data``` and compute correspondences in ```results```. Reconstruction are saved in ```data```

#### On your own meshes

You need to make sure your meshes are preprocessed correctly :

* The trunk axis is the Y axis (visualize your mesh against the mesh in ```data``` to make sure they are normalized in the same way). 
* the scale should be about 1.7 for a standing human (meaning the unit for the point cloud is the ```cm```). You can automatically scale them with the flag ```--scale 1```


#### Failure modes instruction :

- Sometimes the reconstruction is flipped, which break the correspondences. In the easiest case where you meshes are registered in the same orientation, you can just fix this angle in ```reconstruct.py```, to avoid the flipping problem.
- Check the presence of lonely outliers that break the Pointnet encoder. You could try to remove them with the ```--clean``` flag.

## Training the autoencoder

#### Data 

- TODO

#### Options

* TODO

#### Now you can start training

* First launch a visdom server :

```bash
python -m visdom.server -p 8888
```

* Launch the training. Check out all the options in ```./training/train_correspondences.py``` .

```shell
export CUDA_VISIBLE_DEVICES=0 #whichever you want
source activate pytorch-atlasnet
git pull
env=correspondences
python ./training/train_correspondences.py --env $env  |& tee ${env}.txt
```

* Monitor your training on http://localhost:8888/

![visdom](pictures/TODO.png)

## Acknowledgement

* The code for the Chamfer Loss was taken from Fei Xia'a repo : [PointGan](https://github.com/fxia22/pointGAN). Many thanks to him !
* The code for the Laplacian regularization comes from [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/) and [Shubham Tulsiani](https://people.eecs.berkeley.edu/~shubhtuls/). This was so helpful, thanks !
* Part of the SMPL parameters used in the training data comes from [Gül Varol](https://www.di.ens.fr/~varol/)'s repo : https://github.com/gulvarol/surreal But most of all, thanks for all the advices :)
* The FAUST Team for their prompt reaction in resolving a benchmark issue the week of the deadline, especially to [Federica Bogo](https://ps.is.tuebingen.mpg.de/person/fbogo) and Jonathan Williams.
* The efficient code for to compute geodesic errors comes from  https://github.com/zorah/KernelMatching. Thanks!
* The [SMAL](http://smalr.is.tue.mpg.de/) team, and [SCAPE](https://ai.stanford.edu/~drago/Projects/scape/scape.html) team for their help in generating the training data.
* [DeepFunctional Maps](https://arxiv.org/abs/1704.08686) authors for their fast reply the week of the rebuttal ! Many thanks.
* [Hiroharu Kato](http://hiroharu-kato.com/projects_en/neural_renderer.html) for his very clean neural renderer code, that I used for the gifs :-)
* [Pytorch developpers](https://github.com/pytorch/pytorch) for making DL code so easy.
* This work was funded by [Ecole Doctorale MSTIC](http://www.univ-paris-est.fr/fr/-ecole-doctorale-mathematiques-et-stic-mstic-ed-532/). Thanks !
* And last but not least, my great co-authors :  [Matthew Fisher](http://graphics.stanford.edu/~mdfisher/publications.html), [Vladimir G. Kim](http://vovakim.com/), [Bryan C. Russell](http://bryanrussell.org/), and [Mathieu Aubry](http://imagine.enpc.fr/~aubrym/cv.html)

## License

[MIT](https://github.com/ThibaultGROUEIX/AtlasNet/blob/master/license_MIT)

[![Analytics](https://ga-beacon.appspot.com/UA-91308638-2/github.com/ThibaultGROUEIX/template-based-correspondences/readme.md?pixel)](https://github.com/ThibaultGROUEIX/template-based-correspondences/)
