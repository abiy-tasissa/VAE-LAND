# VAE-LAND
VAE-LAND is an algorithm that combines diffusion process on graphs with variational autoencoders (VAE). It enjoys the mathematical tractability of diffusion process on graphs and
the useful low dimensional features obtained from an optimized VAE. Formally, consider a hyperspectral image (HSI) for which we are interested to consider the task of clustering. VAE-LAND first applies VAE to extract latent features. The latent features are an input to the LAND
algorithm. LAND is a provable diffusion based algorithm that is robust to different cluster geometries and noise. 

## Files description

`Hyperspectral_VAE.py`: This is the script that extracts the latent features of HSI using a VAE. The user can change architecture as needed. 

`VAE-LAND.m`: This is the main script that runs LAND on the input data obtained from running `hyperspectral_VAE.py'.

## List of data files
* `salinas_train_dim40.csv`: By applying the VAE on the Salinas A dataset, with latent dimension set to 40, we obtain this dataset. 
* `salinas_train_label_dim40.csv`: This is the ground label of the Salinas A dataset. It can be used for evaluating the accuracy of VAE-LAND.

## Instructions

1. Download the LAND code from [here](https://jmurphy.math.tufts.edu/Code/). 

2. Run `Hyperspectral_VAE.py` on an HSI data of interest and obtain a latent HSI data. This will be an input to LAND. 

3. RUN `VAE-LAND.m`. It is necessary to specify location of LAND_Public folder. For example, `addpath(genpath('C:/home/abiyo\Desktop\LAND_Public\LAND_Public'))`

## References

If you find the code useful, please cite the following papers:

* Abiy Tasissa, Duc Nguyen, and James Murphy, Deep Diffusion Processes for Active Learning of Hyperspectral Images [Link](https://arxiv.org/abs/2101.03197)

* Maggioni, M., J.M. Murphy, Learning by Active Nonlinear Diffusion. Foundations of Data Science, 1(3), pp. 271-291. 2019


## Feedback

Email your feedback to <a href="mailto:abiy19@gmail.com">Abiy Tasissa</a>.
