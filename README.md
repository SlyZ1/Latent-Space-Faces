# Latent-Space-Faces
This repository is intended for learning how to use and manipulate StyleGAN pre-trained models.

## Usage
### Install
Clone the repository.
```shell
git clone https://github.com/SlyZ1/Latent-Space-Faces.git
cd Latent-Space-Faces
```

Initialize the submodule (StyleGAN2).
```shell
git submodule update --init --recursive
```

### Setup
Create a virtual environment with your favorite Python version (tested with Python 3.11.15).
> NOTE: newer version of Python (3.13 or 3.14) might not work well with torchvision
```shell
python3 -m venv venv
```

Activate the virtual environment.
- On Linux/macOS
```shell
source venv/bin/activate
```
- On Windows
```shell
venv\Scripts\Activate.ps1
```

Install all the dependencies.
```shell
pip install -r requirements.txt
```

Download the pretained model ffhq for 512*512 images :
https://huggingface.co/DragGan/DragGan-Models/blob/20dedd0259ff3009fceefa531c3be8ae4f11cd82/stylegan2-ffhq-512x512.pkl


### Run default
Run the StyleGAN2 default projection algorithm.
```shell
python projector.py --outdir=out --target=~/mytargetimg.png --network=./models/stylegan2-ffhq-512x512.pkl
```

### Run custom
Explore our implementation of the projection method explained in https://arxiv.org/pdf/1912.04958 \
which is situated in projecting-demo.ipynb

Results:
|-|`out/seed0085.png`|`out/antonin2.png`|`out/antonin_frere.png`|
|-|-|-|-|
|target|![FFHQ(85)](out/seed0085.png)|![FFHQ(265)](out/antonin2.png)|![FFHQ(297)](out/antonin_frere.png)|
|result|![](projection%20results/z_imnet_v1_mse_features.png)|![FFHQ(265)](projection%20results/9_antonin_train_noise.jpg)|![FFHQ(297)](projection%20results/9_frere_anto_train_noise.png)|
