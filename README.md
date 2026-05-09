# Latent-Space-Faces
This repository is for the purpose of learning how to use and manipulate StyleGan pre-trained models

## Usage
Create a virtual environement with your favorite Python version (tested with 3.14.4)
```shell
python3 -m venv venv
source venv/bin/activate
```

Install all the dependencies
```shell
pip install -r requirements.txt
```

Run the StyleGAN2 test sampling
```shell
python3 stylegan2/generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
```
