# Lab 08 – Two Ways to Generate

## Project Overview

This lab explores different generative models and compares how they generate handwritten digit images.

The experiments use the professor's starter code and the provided dataset of 8 × 8 handwritten digit images. Each image is represented by 64 input features.

The lab covers five main parts:

1. GAN loss verification and generator gradient comparison.
2. Standard GAN versus Wasserstein GAN with Gradient Penalty (WGAN-GP).
3. Forward diffusion and digit generation using a diffusion model.
4. Comparison of deterministic DDIM sampling with different numbers of sampling steps.
5. Comparison of VAE, WGAN-GP, and diffusion models.

The experiments use three random seeds (0, 1, and 2) to evaluate how results vary across runs.

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

The experiments were run using:

- Python 3.12
- Anaconda environment: `ml-env`
- PyTorch
- CPU execution
- The professor-provided `lab08_starter.py`

The starter file provides the models, dataset loading functions, training utilities, and evaluation functions used in the experiments.

## How to Run

Open PowerShell in the Lab 08 folder.

Run the following command:

```powershell
& "C:\Anaconda3\envs\ml-env\python.exe" lab08.py
```

The script executes all five parts in sequence.

The generated CSV files are saved in the `results/` folder, and experiment figures are saved in the `figures/` folder.

**Important:** Run the complete script from the beginning. Parts 4 and 5 reuse models trained in earlier parts, so running those parts independently will not reproduce the complete workflow.

## Part 1 – GAN Verification

This part verifies the mathematical implementation of GAN losses and compares generator gradients.

Experiments include:

- Discriminator loss verification.
- Minimax generator loss verification.
- Non-saturating generator loss verification.
- One-dimensional Gaussian discriminator training.
- Comparison of minimax and non-saturating generator gradients.

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

The non-saturating generator loss produced a substantially stronger gradient than the minimax loss in this experiment.

## Part 2 – Standard GAN vs WGAN-GP

This part compares a standard GAN with a Wasserstein GAN using Gradient Penalty.

Both models were evaluated using seeds 0, 1, and 2.

The experiments use 4,000 generator training steps and generate 359 samples for evaluation.

The WGAN-GP experiment also monitors critic gradient norms and score gaps.

### Results

| Model | Seed | Fréchet Distance | Precision | Recall | Digit Classes |
|---|---:|---:|---:|---:|---:|
| Standard GAN | 0 | 140.0576 | 0.3844 | 0.0111 | 7 |
| Standard GAN | 1 | 141.8674 | 0.7103 | 0.0000 | 8 |
| Standard GAN | 2 | 289.9879 | 0.6072 | 0.0000 | 4 |
| WGAN-GP | 0 | 3.2710 | 0.7967 | 0.9025 | 10 |
| WGAN-GP | 1 | 4.2847 | 0.7799 | 0.9136 | 10 |
| WGAN-GP | 2 | 3.9563 | 0.7549 | 0.9053 | 10 |

### Observation

WGAN-GP achieved substantially lower Fréchet distances and higher recall than the standard GAN across all three seeds.

It also generated samples covering all ten digit classes in each run.

The results suggest that WGAN-GP provided more stable generation and better coverage in these experiments.

## Part 3 – Diffusion Model

This part implements forward diffusion and trains a noise-prediction model.

The experiments include:

- Verification of iterative and closed-form forward diffusion distributions.
- A diffusion experiment using a ring of Gaussian distributions.
- Digit generation using a diffusion model.
- Evaluation across three random seeds.

The digit diffusion model uses 100 diffusion steps and generates 359 samples per seed.

### Forward Diffusion Results

| Step | Iterative Mean | Closed-form Mean | Iterative Std | Closed-form Std |
|---:|---:|---:|---:|---:|
| 10 | 1.4648 | 1.4645 | 0.2140 | 0.2144 |
| 50 | 0.7992 | 0.7951 | 0.8476 | 0.8436 |
| 100 | 0.1138 | 0.1104 | 1.0006 | 0.9992 |

Final alpha-bar: 0.00561876.

### Digit Diffusion Results

| Seed | Fréchet Distance | Precision | Recall | First 100 Loss | Last 200 Loss |
|---:|---:|---:|---:|---:|---:|
| 0 | 13.7157 | 0.7939 | 0.8106 | 0.6275 | 0.1850 |
| 1 | 14.5023 | 0.8078 | 0.8134 | 0.6311 | 0.1855 |
| 2 | 16.8160 | 0.7298 | 0.7688 | 0.6293 | 0.1853 |

### Ring Diffusion Results

- Modes covered: 8 out of 8.
- Share of generated samples near a mode: 0.9700.

### Observation

The noise-prediction loss decreased considerably during training.

The model also generated samples covering all ten digit classes across the three seeds.

The ring experiment covered all eight modes, indicating good coverage of the target distribution.

## Part 4 – Diffusion Sampling Comparison

This part compares deterministic DDIM sampling using different numbers of steps.

The tested sampling step counts are:

- 5
- 10
- 25
- 50
- 100

The deterministic sampler is also compared with the ancestral 100-step sampler.

All comparisons use three random seeds and 359 generated samples per experiment.

### Deterministic DDIM Results

| Seed | 5 Steps | 10 Steps | 25 Steps | 50 Steps | 100 Steps |
|---:|---:|---:|---:|---:|---:|
| 0 | 6.0029 | 6.0602 | 6.8210 | 7.0373 | 7.1487 |
| 1 | 12.5535 | 11.2997 | 12.3360 | 12.6373 | 12.7955 |
| 2 | 15.0473 | 14.0281 | 14.3735 | 14.5940 | 14.7249 |

Values represent Fréchet distance. Lower values indicate better results.

### Observation

The results show that fewer DDIM sampling steps can reduce sampling time considerably.

The best Fréchet result differed across seeds. Therefore, the results do not establish that one sampling step count is always the best choice.

The comparison also shows that deterministic DDIM and ancestral sampling can produce different evaluation results, even when both use 100 steps.

## Part 5 – VAE vs WGAN-GP vs Diffusion

This part compares three generative model families:

- Variational Autoencoder (VAE)
- Wasserstein GAN with Gradient Penalty (WGAN-GP)
- Diffusion model

The comparison uses seeds 0, 1, and 2, with 359 generated samples per model and seed.

The same feature-network evaluation approach is used for the three models within each seed.

### Average Results Across Three Seeds

| Metric | VAE | WGAN-GP | Diffusion |
|---|---:|---:|---:|
| Fréchet Distance | 32.8537 | 3.8373 | 15.0113 |
| Precision | 0.8449 | 0.7772 | 0.7772 |
| Recall | 0.2108 | 0.9071 | 0.7976 |
| Nearest Train Mean | 1.2685 | 1.3076 | 1.4524 |
| TV Distance | 0.1411 | 0.0799 | 0.1495 |
| Average Training Time (seconds) | 1.04 | 126.57 | 11.82 |

Lower Fréchet distance, TV distance, and nearest-training distance indicate better results for those metrics. Higher precision and recall are generally preferred.

### Observation

WGAN-GP achieved the lowest average Fréchet distance and TV distance, along with the highest recall.

VAE achieved the highest precision but had considerably lower recall.

The diffusion model achieved better recall than VAE but did not match WGAN-GP on Fréchet distance in these experiments.

WGAN-GP required more training time than the other two models.

Overall, WGAN-GP provided the strongest balance of sample quality and diversity in this comparison.

## Evaluation Metrics

The experiments use the following evaluation metrics:

- **Fréchet Distance:** Measures the difference between feature distributions of real and generated samples. Lower is better.
- **Precision:** Measures how many generated samples fall within the real-data feature distribution. Higher is better.
- **Recall:** Measures how well generated samples cover the real-data feature distribution. Higher is better.
- **Nearest Training Distance:** Measures the distance between generated samples and their nearest training samples.
- **Distinct Digit Classes:** Counts the number of different predicted digit classes among generated samples.
- **Total Variation (TV) Distance:** Measures the difference between real and generated digit-class distributions. Lower is better.
- **Training Time:** Measures the time taken to train each model.
- **Sampling Time:** Measures the time required to generate samples.

For TV distance, the calculation is:

\[
TV(P,Q)=\frac{1}{2}\sum_i |P(i)-Q(i)|
\]

## Predictions Before Experiments

The following were the expected outcomes before comparing the models:

1. The non-saturating generator loss was expected to provide stronger gradients than the minimax loss when the discriminator became confident.
2. WGAN-GP was expected to provide more stable training and better sample coverage than the standard GAN.
3. Diffusion training loss was expected to decrease as the model learned to predict added noise.
4. DDIM sampling with fewer steps was expected to reduce sampling time, although image quality could change.
5. WGAN-GP, VAE, and diffusion were expected to show different trade-offs between sample quality, diversity, and computational cost.

These expectations were compared with the experimental results.

## Limitations

- Experiments were conducted on a small 8 × 8 handwritten digit dataset.
- Results may differ on larger and more complex image datasets.
- Only three random seeds were used, so the experiments do not capture every possible training outcome.
- Training and sampling times depend on the execution environment.
- Evaluation metrics provide useful comparisons but do not fully represent visual image quality.
- Generated digit classes are predicted using a trained feature network, so class-distribution results depend on that network's predictions.

## Reproducibility

- The professor-provided `lab08_starter.py` file was kept unchanged.
- All five parts are executed through `lab08.py`.
- Random seeds 0, 1, and 2 are used for the multi-seed experiments.
- Experiment results are saved as CSV files.
- Generated sample grids and plots are saved in the `figures/` folder.
- The `.gitignore` file excludes files that should not be committed.

To reproduce the experiments, use the same Python environment and run the command provided in the setup section.

## Conclusion

This lab explored GANs, WGAN-GP, diffusion models, DDIM sampling, and VAEs.

The experiments demonstrated the importance of loss functions, gradient behaviour, sampling methods, and model architecture in generative modelling.

Across the three model families tested in Part 5, WGAN-GP achieved the strongest overall evaluation results, while VAE showed higher precision and diffusion provided a different balance between diversity and generation time.

The experiments also showed that sampling speed and sample quality can vary with the number of diffusion steps.

All five parts were executed successfully, and the results and figures were saved for review.
