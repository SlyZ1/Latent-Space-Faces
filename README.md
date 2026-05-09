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
Create a virtual environment with your favorite Python version (tested with Python 3.14.4).
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

### Run
Run the StyleGAN2 test sampling.
```shell
python3 stylegan2/generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
```

Expected results

|`out/seed0085.png`|`out/seed0265.png`|`out/seed0297.png`|`out/seed0849.png`|
|-|-|-|-|
|![MetFace(85)](res/metfaces0085.png)|![MetFace(265)](res/metfaces0265.png)|![MetFace(297)](res/metfaces0297.png)|![MetFace(849)](res/metfaces0849.png)|

