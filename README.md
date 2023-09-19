# Clustering-based Self-supervised Compressive Spectral Imaging using Positional-encoding Representation
This is the official implementation of the paper "Clustering-based Self-supervised Compressive Spectral Imaging using Positional-encoding Representation".

# 1. Create Environment

* Python 3
* NVIDIA GPU +CUDA
* Python packages:
```
    pip install -r requirements.txt
```

# 2. Dataset and Experiement

Download cave, KAIST, TSA_simu_data, and TSA_real_data (https://github.com/caiyuanhao1998/MST) for real and simulation experiments, and put them into corresponding folders of ```simulation```, and ```real_dataset```, respectively.
```
|--CEINR
  |--simulation
    |--cave
    |--KAIST
    |--TSA_simu_data
  |--real_dataset
    |--TSA_real_data
  |--train_code
  |--net_code
```

