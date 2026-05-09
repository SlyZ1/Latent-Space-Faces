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

Download a pretrained model amongst (will be saved as `./models/<MODEL NAME>.pkl`):
- ffhq
- metfaces
- afhqcat
- afhqdog
- afhqwild
- cifar10
- brecahad
```shell
./download-model.sh <MODEL NAME>
```


### Run
Run the StyleGAN2 test sampling.
```shell
python3 stylegan2/generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 --network=./models/<MODEL NAME>.pkl
```

Expected results for `FFHQ`:
|`out/seed0085.png`|`out/seed0265.png`|`out/seed0297.png`|`out/seed0849.png`|
|-|-|-|-|
|![FFHQ(85)](res/ffhq0085.png)|![FFHQ(265)](res/ffhq0265.png)|![FFHQ(297)](res/ffhq0297.png)|![FFHQ(849)](res/ffhq0849.png)|

Expected results for `MetFaces`:
|`out/seed0085.png`|`out/seed0265.png`|`out/seed0297.png`|`out/seed0849.png`|
|-|-|-|-|
|![MetFaces(85)](res/metfaces0085.png)|![MetFaces(265)](res/metfaces0265.png)|![MetFaces(297)](res/metfaces0297.png)|![MetFaces(849)](res/metfaces0849.png)|
