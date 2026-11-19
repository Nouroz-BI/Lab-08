# Lab 08 Two Ways to Generate

## Project Overview

In this lab, I explored different generative models and compared how well they can create handwritten digit images.

The experiments use the dataset provided in the professor's starter code. Each image is 8 × 8 pixels and is represented by 64 input features.

The lab is divided into five parts:

1. Checking GAN loss calculations and comparing generator gradients.
2. Comparing a standard GAN with WGAN-GP.
3. Implementing forward diffusion and training a diffusion model to generate digit images.
4. Comparing DDIM sampling with different numbers of steps.
5. Comparing VAE, WGAN-GP, and diffusion models.

For the experiments that require multiple runs, I used three random seeds: 0, 1, and 2. This helped me see how the results changed between runs.

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
│   ├── diffusion_digits_seed_2.png
│   └── additional sampling and model comparison figures
│
└── results/
    ├── part2_gan_comparison.csv
    ├── part2_wgan_gp_monitoring.csv
    ├── part3_diffusion_digits.csv
    ├── part3_diffusion_ring.csv
    ├── part4_sampling_comparison.csv
    └── part5_three_family_comparison.csv
```

## Requirements and Environment

The experiments were completed using:

- Python 3.12
- Anaconda environment: `ml-env`
- PyTorch
- CPU
- The professor-provided `lab08_starter.py` file

The starter file contains the models, dataset loading functions, training functions, and evaluation methods used in this lab.

## How to Run

Open PowerShell in the Lab 08 folder and run:

```powershell
& "C:\Anaconda3\envs\ml-env\python.exe" lab08.py
```

The script runs all five parts in order.

- CSV results are saved in the `results/` folder.
- Figures and generated image samples are saved in the `figures/` folder.

**Important:** Run the complete script from the beginning. Parts 4 and 5 use models trained in earlier parts, so they depend on those earlier experiments.

## Part 1 GAN Verification

In this part, I checked the GAN loss calculations and compared the gradients produced by two generator loss functions.

The experiments included:

- Checking the discriminator loss.
- Checking the minimax generator loss.
- Checking the non-saturating generator loss.
- Training a discriminator on a one-dimensional Gaussian dataset.
- Comparing the gradients from the minimax and non-saturating losses.

### Results

| Verification | Result |
|---|---:|
| Discriminator loss difference | 0.00000000 |
| Minimax generator loss difference | 0.00000000 |
| Non-saturating generator loss difference | 0.00000000 |
| Gaussian discriminator MSE | 0.000081 |
| Minimax output gradient norm | 0.0000000001 |
| Non-saturating output gradient norm | 0.0454602832 |

All three loss verification checks passed.

The non-saturating loss produced a much stronger generator gradient than the minimax loss in this experiment. This shows why the choice of loss function can make a difference during GAN training.

## Part 2 Standard GAN vs WGAN-GP

In this part, I compared a standard GAN with a Wasserstein GAN using Gradient Penalty (WGAN-GP).

Both models were tested with three random seeds: 0, 1, and 2. Each experiment used 4,000 training steps and generated 359 samples for evaluation.

I also checked the critic's gradient norms and score gaps during WGAN-GP training.

### Results

| Model | Seed | Fréchet Distance | Precision | Recall | Digit Classes |
|---|---:|---:|---:|---:|---:|
| Standard GAN | 0 | 140.0576 | 0.3844 | 0.0111 | 7 |
| Standard GAN | 1 | 141.8674 | 0.7103 | 0.0000 | 8 |
| Standard GAN | 2 | 289.9879 | 0.6072 | 0.0000 | 4 |
| WGAN-GP | 0 | 3.2710 | 0.7967 | 0.9025 | 10 |
| WGAN-GP | 1 | 4.2847 | 0.7799 | 0.9136 | 10 |
| WGAN-GP | 2 | 3.9563 | 0.7549 | 0.9053 | 10 |

### What I Observed

WGAN-GP performed better than the standard GAN in all three runs.

It had much lower Fréchet distances and higher recall. It also generated samples covering all ten digit classes in every run.

The standard GAN had difficulty covering the digit classes, especially in seeds 1 and 2, where its recall was zero.

Based on these results, WGAN-GP gave better results and more consistent digit coverage in this experiment.

## Part 3 Diffusion Model

In this part, I implemented forward diffusion and trained a model to predict the noise added to the images.

The experiments included:

- Comparing iterative and closed-form forward diffusion.
- Testing diffusion on a ring of Gaussian distributions.
- Training a diffusion model to generate handwritten digits.
- Evaluating the results using three random seeds.

The digit diffusion model used 100 diffusion steps and generated 359 samples for each seed.

### Forward Diffusion Results

| Step | Iterative Mean | Closed-form Mean | Iterative Std | Closed-form Std |
|---:|---:|---:|---:|---:|
| 10 | 1.4648 | 1.4645 | 0.2140 | 0.2144 |
| 50 | 0.7992 | 0.7951 | 0.8476 | 0.8436 |
| 100 | 0.1138 | 0.1104 | 1.0006 | 0.9992 |

Final alpha-bar: `0.00561876`

### Digit Diffusion Results

| Seed | Fréchet Distance | Precision | Recall | First 100 Loss | Last 200 Loss |
|---:|---:|---:|---:|---:|---:|
| 0 | 13.7157 | 0.7939 | 0.8106 | 0.6275 | 0.1850 |
| 1 | 14.5023 | 0.8078 | 0.8134 | 0.6311 | 0.1855 |
| 2 | 16.8160 | 0.7298 | 0.7688 | 0.6293 | 0.1853 |

### Ring Diffusion Results

- Modes covered: 8 out of 8.
- Generated samples near a mode: 0.9700.

### What I Observed

The noise prediction loss decreased during training, which shows that the model improved at predicting the added noise.

The digit generation experiments also covered all ten digit classes across the three runs.

In the ring experiment, the model covered all eight modes, and 97% of generated samples were near one of the modes.

## Part 4 Diffusion Sampling Comparison

In this part, I compared deterministic DDIM sampling using different numbers of steps.

The tested step counts were:

- 5 steps
- 10 steps
- 25 steps
- 50 steps
- 100 steps

I also compared DDIM with the ancestral sampler using 100 steps.

Each experiment used three random seeds and generated 359 samples.

### Deterministic DDIM Results

| Seed | 5 Steps | 10 Steps | 25 Steps | 50 Steps | 100 Steps |
|---:|---:|---:|---:|---:|---:|
| 0 | 6.0029 | 6.0602 | 6.8210 | 7.0373 | 7.1487 |
| 1 | 12.5535 | 11.2997 | 12.3360 | 12.6373 | 12.7955 |
| 2 | 15.0473 | 14.0281 | 14.3735 | 14.5940 | 14.7249 |

The values represent Fréchet distance. Lower values indicate better results.

### What I Observed

Using fewer DDIM steps can reduce the time needed to generate samples.

However, the best result was not the same for every seed. Five steps gave the lowest Fréchet distance for seed 0, while ten steps gave the lowest result for seeds 1 and 2.

This means that fewer steps can work well, but the results depend on the run. The comparison also showed that DDIM and ancestral sampling can produce different results, even when both use 100 steps.

## Part 5 VAE vs WGAN-GP vs Diffusion

In the final part, I compared three different models:

- Variational Autoencoder (VAE)
- Wasserstein GAN with Gradient Penalty (WGAN-GP)
- Diffusion model

Each model was tested using seeds 0, 1, and 2, with 359 generated samples per run.

For each seed, I used the same feature network and test data to evaluate the three models.

### Average Results Across Three Seeds

| Metric | VAE | WGAN-GP | Diffusion |
|---|---:|---:|---:|
| Fréchet Distance | 32.8537 | 3.8373 | 15.0113 |
| Precision | 0.8449 | 0.7772 | 0.7772 |
| Recall | 0.2108 | 0.9071 | 0.7976 |
| Nearest Train Mean | 1.2685 | 1.3076 | 1.4524 |
| TV Distance | 0.1411 | 0.0799 | 0.1495 |
| Average Training Time (seconds) | 1.04 | 126.57 | 11.82 |

### What I Observed

WGAN-GP achieved the lowest average Fréchet distance and TV distance. It also had the highest recall, meaning it covered more of the real data distribution.

VAE had the highest precision, but its recall was much lower. This means its generated samples were closer to the real data in some respects, but it did not cover as much variety.

The diffusion model had better recall than VAE, but its Fréchet distance was higher than WGAN-GP's.

WGAN-GP took much longer to train than the other two models.

Overall, WGAN-GP gave the best results across most of the evaluation metrics, although it required more training time.

## Evaluation Metrics

I used the following metrics to compare the generated samples:

- **Fréchet Distance:** Compares the feature distributions of real and generated samples. Lower values are better.
- **Precision:** Measures how many generated samples are close to the real-data distribution. Higher values are better.
- **Recall:** Measures how well generated samples cover the real-data distribution. Higher values are better.
- **Nearest Training Distance:** Measures the distance between generated samples and their closest training samples.
- **Distinct Digit Classes:** Counts how many different digit classes appear in the generated samples.
- **Total Variation (TV) Distance:** Measures the difference between the real and generated digit-class distributions. Lower values are better.
- **Training Time:** Measures how long the model takes to train.
- **Sampling Time:** Measures how long it takes to generate samples.

The TV distance is calculated as:

\[
TV(P,Q)=\frac{1}{2}\sum_i |P(i)-Q(i)|
\]

## Predictions Before Experiments

Before running the experiments, I expected the following results:

1. The non-saturating generator loss would produce stronger gradients than the minimax loss when the discriminator became confident.
2. WGAN-GP would have more stable training and better digit coverage than the standard GAN.
3. The diffusion training loss would decrease as the model learned to predict the added noise.
4. Using fewer DDIM steps would reduce sampling time, although the generated image quality might change.
5. VAE, WGAN-GP, and diffusion would have different results in terms of image quality, sample variety, and training time.

After running the experiments, I compared these expectations with the actual results. Most of the results followed the expected patterns, especially the stronger performance of WGAN-GP compared with the standard GAN.

## Limitations

There are some limitations to these experiments:

- The dataset contains small 8 × 8 handwritten digit images, so the results may be different with larger images.
- I used only three random seeds. More runs would give a better idea of how much the results can change.
- Training and sampling times depend on the computer and hardware used.
- Evaluation metrics are useful for comparing models, but they cannot fully measure how good the generated images look.
- The predicted digit classes depend on the trained feature network, so classification errors can affect the results.

## Reproducibility

I followed these steps to make the experiments easier to run again:

- Kept the professor-provided `lab08_starter.py` file unchanged.
- Put all five parts in `lab08.py`.
- Used random seeds 0, 1, and 2 for the multi-run experiments.
- Saved the experiment results as CSV files in the `results/` folder.
- Saved generated image grids and plots in the `figures/` folder.
- Added a `.gitignore` file to exclude unnecessary files.

To run the experiments again, use the same Python environment and run the command in the **How to Run** section.

## Conclusion

In this lab, I worked with GANs, WGAN-GP, diffusion models, DDIM sampling, and VAEs.

I learned how different loss functions affect generator training and how the number of diffusion steps affects sampling. I also compared the models using the same evaluation approach.

Based on my results, WGAN-GP performed best overall in the final comparison. VAE achieved higher precision, while diffusion provided a different balance between sample coverage and training time.

The experiments also showed that results can change between random seeds, so it is important to compare more than one run.

All five parts were executed successfully, and the results and figures were saved in their respective folders.
