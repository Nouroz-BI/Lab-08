# Lab 08 Two Ways to Generate

## Project Overview

In this lab, I worked with different generative models and compared the results they produced.

The experiments use the professor's starter code and the provided dataset of 8 × 8 handwritten digit images. Each image contains 64 input features.

I completed five parts:

1. Checked GAN loss calculations and compared generator gradients.
2. Compared a standard GAN with WGAN-GP.
3. Tested forward and reverse diffusion using a ring of Gaussians and handwritten digits.
4. Compared DDIM sampling with different numbers of steps.
5. Compared VAE, WGAN-GP, and diffusion models.

The purpose of this lab was to understand how these models work and see how their results differ.

## Project Structure

```text
Lab 08/
│
├── lab08.py
├── lab08_starter.py
├── README.md
├── .gitignore
│
├── figures/
│   ├── gaussian_discriminator.png
│   ├── ring_diffusion_seed_0.png
│   ├── diffusion_digits_seed_0.png
│   ├── diffusion_digits_seed_1.png
│   └── diffusion_digits_seed_2.png
│
└── results/
    ├── part2_gan_comparison.csv
    ├── part2_wgan_gp_monitoring.csv
    ├── part3_diffusion_digits.csv
    ├── part3_diffusion_ring.csv
    └── part5_three_family_comparison.csv
```

The `lab08.py` file contains my work for all five parts. I kept `lab08_starter.py` unchanged because it was provided by the professor.

The `figures` folder contains the plots and generated images. The `results` folder contains the experiment results in CSV format.

## Requirements and Environment

I ran the experiments using the following environment:

- Python 3.12
- PyTorch 2.14.0+cpu
- NumPy 2.5.3
- SciPy 1.18.1
- Scikit-learn 1.9.1
- Matplotlib 3.11.2
- Windows
- CPU execution
- CUDA was not available

I used the Anaconda `ml-env` environment to run the code.

## How to Run

Open PowerShell in the Lab 08 folder and run:

```powershell
& "C:\Anaconda3\envs\ml-env\python.exe" lab08.py
```

This command runs all five parts in order. The script also saves the plots and CSV files in the `figures` and `results` folders.

## Part 1 GAN Loss Verification

In this part, I checked the GAN loss calculations and compared the gradients produced by two generator loss functions: minimax and non-saturating.

I completed the following checks:

- Checked the discriminator loss.
- Checked the minimax generator loss.
- Checked the non-saturating generator loss.
- Trained a discriminator using a one-dimensional Gaussian example.
- Compared the generator gradients for both loss functions.

### Loss verification results

| Check | Actual Value | Expected Value | Difference | Result |
|---|---:|---:|---:|---|
| Discriminator loss | 1.662559 | 1.662559 | 0.000000 | PASS |
| Minimax generator loss | 0.871528 | 0.871528 | 0.000000 | PASS |
| Non-saturating generator loss | 0.716267 | 0.716267 | 0.000000 | PASS |

All three checks passed. The actual values matched the expected values.

### Gaussian discriminator

I also trained a discriminator using a one-dimensional Gaussian example.

The final mean squared error was:

```text
MSE = 0.000081
```

The plot is saved as:

```text
figures/gaussian_discriminator.png
```

### Generator gradient comparison

I compared the two generator loss functions over 100 batches.

| Measurement | Minimax | Non-saturating |
|---|---:|---:|
| Mean D(G(z)) | 0.0000000018 | 0.0000000018 |
| Gradient norm | 0.0000000001 | 0.0454602832 |

The minimax gradient was almost zero, while the non-saturating loss produced a larger gradient. This shows that the non-saturating loss can help the generator continue learning when the discriminator is very confident.

## Part 2 Standard GAN vs WGAN-GP

In this part, I compared a standard GAN with a Wasserstein GAN using Gradient Penalty (WGAN-GP).

I ran both models using three random seeds:

```text
Seeds: 0, 1, 2
```

I compared the results using Fréchet distance, precision, recall, and the number of generated classes.

### Results

| Model | Seed | Fréchet Distance | Precision | Recall | Classes |
|---|---:|---:|---:|---:|---:|
| Standard GAN | 0 | 140.0576 | 0.3844 | 0.0111 | 7 |
| Standard GAN | 1 | 141.8674 | 0.7103 | 0.0000 | 8 |
| Standard GAN | 2 | 289.9879 | 0.6072 | 0.0000 | 4 |
| WGAN-GP | 0 | 3.2710 | 0.7967 | 0.9025 | 10 |
| WGAN-GP | 1 | 4.2847 | 0.7799 | 0.9136 | 10 |
| WGAN-GP | 2 | 3.9563 | 0.7549 | 0.9053 | 10 |

### WGAN-GP training checks

I also checked the critic's gradient norm and score gap during training.

| Seed | First 100 Gradient Norm | First 100 Score Gap | Last 500 Gradient Norm | Last 500 Score Gap |
|---|---:|---:|---:|---:|
| 0 | 0.912958 | 1.080788 | 1.008310 | 0.296475 |
| 1 | 0.927622 | 1.284866 | 1.008710 | 0.301598 |
| 2 | 0.919847 | 1.131853 | 1.008588 | 0.283904 |

WGAN-GP performed better than the standard GAN in these experiments. Its Fréchet distance was much lower, and its recall was higher. It also generated all 10 classes in each run.

The results are saved in:

```text
results/part2_gan_comparison.csv
results/part2_wgan_gp_monitoring.csv
```

## Part 3 Diffusion Models

In this part, I worked with diffusion models to see how noise is added to data and then removed during the reverse process.

I tested the model using two examples:

1. A ring of Gaussian distributions.
2. Handwritten digit images.

### Forward diffusion verification

I compared the results from the iterative forward diffusion process with the closed-form calculation.

| Step | Iterative Mean | Closed-Form Mean | Iterative Std | Closed-Form Std |
|---|---:|---:|---:|---:|
| 10 | 1.4648 | 1.4645 | 0.2140 | 0.2144 |
| 50 | 0.7992 | 0.7951 | 0.8476 | 0.8436 |
| 100 | 0.1138 | 0.1104 | 1.0006 | 0.9992 |

The two calculations produced very similar results.

The final alpha-bar value was:

```text
0.00561876
```

### Ring of Gaussians

I used a ring of Gaussian distributions to check whether the model could learn all eight modes.

The results were:

- Modes covered: 8 out of 8.
- Samples near a mode: 0.9700.

The generated plot is saved here:

```text
figures/ring_diffusion_seed_0.png
```

### Handwritten digit diffusion results

I trained and tested the diffusion model using three random seeds.

| Seed | Fréchet Distance | Precision | Recall | First 100 Loss | Last 200 Loss |
|---|---:|---:|---:|---:|---:|
| 0 | 13.7157 | 0.7939 | 0.8106 | 0.6275 | 0.1850 |
| 1 | 14.5023 | 0.8078 | 0.8134 | 0.6311 | 0.1855 |
| 2 | 16.8160 | 0.7298 | 0.7688 | 0.6293 | 0.1853 |

All three runs generated 10 distinct classes. The loss also decreased during training.

The generated images are saved in the `figures` folder:

```text
figures/diffusion_digits_seed_0.png
figures/diffusion_digits_seed_1.png
figures/diffusion_digits_seed_2.png
```

The results are saved in:

```text
results/part3_diffusion_digits.csv
results/part3_diffusion_ring.csv
```

## Part 4 DDIM Sampling

In this part, I tested DDIM sampling using different numbers of steps. I also compared it with the 100-step ancestral sampler.

The DDIM configurations used were:

```text
5, 10, 25, 50, 100 steps
```

I used three random seeds for the comparison.

### Fréchet distance results

| Seed | 5 Steps | 10 Steps | 25 Steps | 50 Steps | 100 Steps |
|---|---:|---:|---:|---:|---:|
| 0 | 6.0029 | 6.0602 | 6.8210 | 7.0373 | 7.1487 |
| 1 | 12.5535 | 11.2997 | 12.3360 | 12.6373 | 12.7955 |
| 2 | 15.0473 | 14.0281 | 14.3735 | 14.5940 | 14.7249 |

All DDIM configurations generated 10 classes.

### DDIM sampling time

I also recorded how long each configuration took to generate samples.

| Seed | 5 Steps | 10 Steps | 25 Steps | 50 Steps | 100 Steps |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0116 s | 0.0297 s | 0.0642 s | 0.0687 s | 0.1427 s |
| 1 | 0.0093 s | 0.0155 s | 0.0381 s | 0.0946 s | 0.2154 s |
| 2 | 0.0091 s | 0.0426 s | 0.0459 s | 0.0969 s | 0.1947 s |

The ancestral 100-step sampling times were:

| Seed | Sampling Time |
|---|---:|
| 0 | 0.1468 s |
| 1 | 0.2336 s |
| 2 | 0.1370 s |

In these experiments, the 5-step DDIM configuration took less time than the 100-step configuration. The exact sampling time varied between seeds. The results show that DDIM can generate samples with fewer steps, which can reduce the time needed for sampling.

## Part 5 Comparing VAE, WGAN-GP, and Diffusion

In the final part, I compared three models:

- Variational Autoencoder (VAE)
- Wasserstein GAN with Gradient Penalty (WGAN-GP)
- Diffusion model

I used the same test data and evaluation feature network for all three models. Each model was tested with three random seeds.

### Results

| Model | Seed | Fréchet Distance | Precision | Recall | Nearest Train Mean | TV |
|---|---:|---:|---:|---:|---:|---:|
| VAE | 0 | 31.1435 | 0.8468 | 0.2006 | 1.2695 | 0.1281 |
| VAE | 1 | 31.5710 | 0.8329 | 0.2646 | 1.2758 | 0.1421 |
| VAE | 2 | 35.8467 | 0.8552 | 0.1671 | 1.2601 | 0.1532 |
| WGAN-GP | 0 | 3.2710 | 0.7967 | 0.9025 | 1.2844 | 0.0947 |
| WGAN-GP | 1 | 4.2847 | 0.7799 | 0.9136 | 1.3198 | 0.0696 |
| WGAN-GP | 2 | 3.9563 | 0.7549 | 0.9053 | 1.3186 | 0.0752 |
| Diffusion | 0 | 13.7157 | 0.7939 | 0.8106 | 1.4293 | 0.1421 |
| Diffusion | 1 | 14.5023 | 0.8078 | 0.8134 | 1.4767 | 0.1421 |
| Diffusion | 2 | 16.8160 | 0.7298 | 0.7688 | 1.4513 | 0.1643 |

### Average results across three seeds

| Model | Average Fréchet Distance | Average Precision | Average Recall | Average Nearest Train Mean | Average TV |
|---|---:|---:|---:|---:|---:|
| VAE | 32.8537 | 0.8449 | 0.2108 | 1.2685 | 0.1411 |
| WGAN-GP | 3.8373 | 0.7772 | 0.9071 | 1.3076 | 0.0799 |
| Diffusion | 15.0113 | 0.7772 | 0.7976 | 1.4524 | 0.1495 |

### Training time comparison

| Model | Average Training Time |
|---|---:|
| VAE | 1.30 seconds |
| WGAN-GP | 53.88 seconds |
| Diffusion | 13.99 seconds |

### Sampling time comparison

| Model | Average Sampling Time |
|---|---:|
| VAE | 0.0004 seconds |
| WGAN-GP | 0.0028 seconds |
| Diffusion | 0.2172 seconds |

WGAN-GP had the lowest average Fréchet distance and the highest average recall. The diffusion model also had good recall, while the VAE had the highest precision.

The VAE took the least time to train and generate samples. Diffusion took longer to generate samples because it uses multiple denoising steps.

The results are saved in:

```text
results/part5_three_family_comparison.csv
```

## Evaluation Metrics

I used the following metrics to compare the generated images:

- **Fréchet Distance:** Measures the difference between generated images and real images. A lower value is better.
- **Precision:** Measures how closely the generated samples match the real data.
- **Recall:** Measures how well the generated samples cover the real data.
- **Nearest Train Mean:** Measures the average distance to the nearest training samples.
- **TV:** Total variation measurement included in the model comparison.

Each metric looks at a different part of the results, so I used them together rather than relying on only one.

## Expected Results

Before running the experiments, I expected the following:

- The GAN loss calculations would match their expected values.
- The non-saturating generator loss would produce a stronger gradient than the minimax loss when the discriminator was very confident.
- WGAN-GP would perform better than the standard GAN.
- The diffusion model would learn to remove noise and generate images.
- DDIM with fewer steps would take less time to generate samples.
- VAE, WGAN-GP, and diffusion would have different results in image quality, sample coverage, and training time.

After running the experiments, I compared the results with these expectations.

## Limitations

There are a few limitations to this lab:

- The experiments use small 8 × 8 handwritten digit images.
- All experiments were run using a CPU.
- Each model was tested with three random seeds. More runs could give a better idea of how consistent the results are.
- Training and sampling times can change depending on the computer and its workload.
- The evaluation metrics do not show every detail of the generated images, so the images should also be checked visually.

## Reproducibility

I followed these steps to make the experiments easier to run again:

- Kept the professor's starter code unchanged.
- Used the same three random seeds: 0, 1, and 2.
- Saved the experiment results in CSV files.
- Saved the generated plots and images in the `figures` folder.
- Included the Python environment and run command in this README.
- Kept all five parts in one Python script.

The results and running times may be different on another computer or in another run.

## Conclusion

This lab helped me understand how GANs, WGAN-GP, VAEs, and diffusion models work.

In Part 1, all three GAN loss checks passed. In Part 2, WGAN-GP performed better than the standard GAN in the experiments. In Part 3, the diffusion model generated samples from both the ring of Gaussians and the handwritten digit dataset. In Part 4, I compared DDIM sampling with different numbers of steps and recorded the time taken for each configuration.

In the final comparison, WGAN-GP had the lowest average Fréchet distance and the highest average recall. The VAE was the fastest model to train and sample, while diffusion took longer to generate samples.

Overall, this lab gave me a chance to work with different generative models, check their results, and understand the differences between them.
