# Applying MixViT to Video Prediction
This Document is a applying MixViT to Video Prediction with SimVP on Moving MNIST. The code is based on OpenSTL.

### Environement Setup
Install [OpenSTL](https://github.com/chengtan9907/OpenSTL) project with setup.py. _Note_ You need to run the installation command with sudo.
```shell
sudo python setup.py develop
```

### Prepare Dataset
Thanks to [OpenSTL](https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/docs/en/install.md), you can easily prepare the dataset from the project's website. You can download the dataset either from the [OpenSTL Github](https://github.com/chengtan9907/OpenSTL) or from the dedicated [download page](https://www.cs.toronto.edu/~nitish/unsupervised_video/) Alternatively, you can download the the daset with the following code

```shell
./tools/prepared_data/download_mmnist.sh
```

we prepared only MMNIST dataset for this experienment.

```
OpenSTL
├── configs
└── data
    |── moving_mnist
    |   ├── mnist_test_seq.npy
    |   ├── train-images-idx3-ubyte.gz
```

## Train
The code to train MixViT + SimVP models on OpenSTL
```shell
python ./tools/train.py -d mmnist -m SimVP --model_type mixvit --lr 1e-3 \
    -c /configs/mmnist/simvp/SimVP_MixVit.py
```

## Validate
The code to validate MixViT + SimVP models on OpenSTL
```shell
python ./tools/non_dist_test.py -d mmnist -m SimVP --model_type mixvit \
    --config_file ./configs/mmnist/simvp/SimVP_MixVit.py --test 
```

    
| Architecture |   Setting  | Params | FLOPs |  MSE  |  MAE  |  SSIM  |  PSNR | Download |
| :----------: | :--------: | :----: | :---: | :---: | :---: | :----: | :---: | :------: |
|   MixViT     |  200 epoch |  37.6M | 14.0G | 25.68 | 75.59 | 0.9317 | 38.38 |          |
|   MixViT     | 2000 epoch |  37.6M | 14.0G | 16.37 | 53.57 | 0.9579 | 39.26 |          |
