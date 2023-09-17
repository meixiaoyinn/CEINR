# Clustering-based Self-supervised Compressive Spectral Imaging using Positional-encoding Representation
This is the official implementation of the paper "Clustering-based Self-supervised Compressive Spectral Imaging using Positional-encoding Representation".
# 1. Method overview

![network1](https://github.com/meixiaoyinn/CEINR/assets/93026157/f1bdc370-3d91-42ac-b31e-8fd728c82bd2)

# 2. Comparison with State-of-the-art Methods

## Quantitative Comparison on Simulation Dataset

![image](https://github.com/meixiaoyinn/CEINR/assets/93026157/758b80f6-a3df-4b84-a1ce-f7bf56467d50)

![image](https://github.com/meixiaoyinn/CEINR/assets/93026157/91dc7239-53cd-4f7d-8ce0-cced82789349)

The performance is reported on 10 scenes of the KAIST dataset. The test size of FLOPS is 256 x 256.

# 3. Create Environment

* Python 3
* NVIDIA GPU +CUDA
* Python packages:
```
    pip install -r requirements.txt
```

# 4. Dataset and Experiement

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

