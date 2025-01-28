# CV-Project-3

## Prerequisites
```sh
# Create a new conda environment
conda create -n colorization_env python=3.12 -y
conda activate colorization_env

# Install required packages
pip install -r requirements.txt

# Add DVC client secret
dvc remote modify myremote gdrive_client_secret 'THE_SECRET'

# Pull DVC dataset
dvc pull
```

## Training
```sh
python main.py
```
