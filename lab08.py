from pathlib import Path
import csv
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

# Use a non-interactive backend so figures can be saved without opening windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the models and helper functions provided in the professor's starter file.
# The starter file is kept separate so the original professor code remains unchanged.
from lab08_starter import (
    Generator,
    Discriminator,
    Denoiser,
    BETAS,
    ring_of_gaussians,
    mode_coverage,
    load_task,
    train_feature_net,
    get_features,
    score_samples,
    save_grid,
    VAE,
    train_vae,
    nearest_train_distance,
)


# GENERAL SETUP

# Use the same starting seed to make the experiments repeatable.
SEED = 0

# Run the experiments on the CPU.
DEVICE = torch.device("cpu")

# Keep generated figures and result files in separate folders.
FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")

# Create these folders automatically if they do not already exist.
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Use three seeds to check whether the results stay similar across runs.
PART2_SEEDS = [0, 1, 2]

# Set the initial random seeds for Python, NumPy, and PyTorch.
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def set_seed(seed):
    """Set the random seeds before an experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# PART 1: GAN LOSS FUNCTIONS

# The discriminator tries to give high scores to real samples
# and low scores to generated samples.
def discriminator_loss(real_logits, fake_logits):
    real_loss = F.softplus(-real_logits).mean()
    fake_loss = F.softplus(fake_logits).mean()
    return real_loss + fake_loss


# The minimax generator loss is the original generator objective.
# It can produce very small gradients when the discriminator is strong.
def generator_minimax_loss(fake_logits):
    return F.softplus(fake_logits).mean()


# The non-saturating loss is an alternative that helps the generator
# receive stronger gradients when its generated samples are poor.
def generator_nonsaturating_loss(fake_logits):
    return F.softplus(-fake_logits).mean()


# CHECK 1: Verify the GAN loss functions

def check_gan_losses():

    print("\n" + "=" * 65)
    print("CHECK 1: GAN LOSS VERIFICATION")
    print("=" * 65)

    torch.manual_seed(SEED)

    # Create sample real and fake discriminator outputs.
    real_logits = torch.randn(100)
    fake_logits = torch.randn(100)

    # Calculate the losses using our functions.
    actual_d = discriminator_loss(real_logits, fake_logits)
    actual_minimax = generator_minimax_loss(fake_logits)
    actual_nonsaturating = generator_nonsaturating_loss(fake_logits)

    # Calculate the same losses using the log-sigmoid formulas.
    # These values give us a reference for checking our implementations.
    expected_d = (
        -F.logsigmoid(real_logits).mean()
        - F.logsigmoid(-fake_logits).mean()
    )

    expected_minimax = -F.logsigmoid(-fake_logits).mean()
    expected_nonsaturating = -F.logsigmoid(fake_logits).mean()

    # Compare the two calculations for each loss.
    checks = [
        ("Discriminator", actual_d, expected_d),
        ("Minimax generator", actual_minimax, expected_minimax),
        ("Non-saturating generator", actual_nonsaturating, expected_nonsaturating),
    ]

    for name, actual, expected in checks:

        difference = abs(actual.item() - expected.item())

        print(
            f"{name}: actual={actual.item():.6f}, "
            f"expected={expected.item():.6f}, "
            f"difference={difference:.8f}"
        )

        # A very small difference means the implementation is correct.
        print("  PASS" if difference < 1e-5 else "  FAIL")

    print("Check 1 completed.")


# CHECK 2: One-dimensional Gaussian discriminator

def check_gaussian_discriminator():

    print("\n" + "=" * 65)
    print("CHECK 2: ONE-DIMENSIONAL GAUSSIAN DISCRIMINATOR")
    print("=" * 65)

    torch.manual_seed(SEED)

    batch_size = 256
    learning_rate = 3e-3
    training_steps = 3000

    # Create a discriminator that takes one input value.
    discriminator = Discriminator(
        d_in=1,
        d_hidden=32
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=learning_rate
    )

    # Train the discriminator to tell apart two Gaussian distributions.
    # Real samples come from N(0, 1), while fake samples come from N(1.5, 1).
    for step in range(training_steps):

        real = torch.randn(batch_size, 1, device=DEVICE)
        fake = torch.randn(batch_size, 1, device=DEVICE) + 1.5

        real_logits = discriminator(real)
        fake_logits = discriminator(fake)

        loss = discriminator_loss(real_logits, fake_logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss every 500 steps to follow the training progress.
        if (step + 1) % 500 == 0:
            print(
                f"Step {step + 1}/{training_steps} "
                f"| Loss: {loss.item():.6f}"
            )

    # Check the discriminator's predictions over a range of input values.
    x_values = torch.linspace(-2, 3.5, 500).reshape(-1, 1)

    with torch.no_grad():
        learned_probabilities = torch.sigmoid(
            discriminator(x_values)
        ).squeeze().numpy()

    x_numpy = x_values.squeeze().numpy()

    # Calculate the probability densities of the real and fake distributions.
    # These are used to find the theoretical optimal discriminator.
    real_density = (
        1.0 / (2.0 * torch.pi) ** 0.5
    ) * torch.exp(-0.5 * x_values.squeeze() ** 2)

    fake_density = (
        1.0 / (2.0 * torch.pi) ** 0.5
    ) * torch.exp(-0.5 * (x_values.squeeze() - 1.5) ** 2)

    # The optimal discriminator is the real density divided by
    # the combined real and fake densities.
    optimal_probabilities = (
        real_density / (real_density + fake_density)
    ).numpy()

    # Measure how close the learned discriminator is to the theoretical one.
    mse = (
        (learned_probabilities - optimal_probabilities) ** 2
    ).mean()

    print(f"\nMean squared error: {mse:.6f}")

    # Plot both curves so we can visually compare the results.
    plt.figure(figsize=(8, 5))

    plt.plot(
        x_numpy,
        optimal_probabilities,
        label="Theoretical optimal D*(x)",
        linewidth=2
    )

    plt.plot(
        x_numpy,
        learned_probabilities,
        label="Learned discriminator",
        linewidth=2,
        linestyle="--"
    )

    plt.xlabel("x")
    plt.ylabel("Discriminator probability")
    plt.title("Learned vs Theoretical Gaussian Discriminator")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the comparison figure for the lab report.
    figure_path = FIGURES_DIR / "gaussian_discriminator.png"
    plt.savefig(figure_path, dpi=150)
    plt.close()

    print(f"Figure saved: {figure_path}")
    print("Check 2 completed.")


# CHECK 3: Generator gradient comparison

def check_generator_gradients():

    print("\n" + "=" * 65)
    print("CHECK 3: GENERATOR GRADIENT COMPARISON")
    print("=" * 65)

    torch.manual_seed(SEED)

    batch_size = 128
    discriminator_steps = 300
    measurement_batches = 100

    # Create a generator that produces two-dimensional samples.
    generator = Generator(
        d_noise=8,
        d_out=2,
        d_hidden=64
    ).to(DEVICE)

    # The discriminator receives the two-dimensional samples.
    discriminator = Discriminator(
        d_in=2,
        d_hidden=64
    ).to(DEVICE)

    # Move the generator's starting output away from the real data.
    # This helps demonstrate what happens when the discriminator
    # becomes confident that generated samples are fake.
    with torch.no_grad():
        generator.net[-1].bias.add_(6.0)

    optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=2e-3,
        betas=(0.5, 0.9)
    )

    # Train the discriminator while keeping the generator fixed.
    for step in range(discriminator_steps):

        real, _ = ring_of_gaussians(batch_size, seed=step)
        real = real.to(DEVICE)

        z = torch.randn(batch_size, generator.d_noise, device=DEVICE)

        with torch.no_grad():
            fake = generator(z)

        real_logits = discriminator(real)
        fake_logits = discriminator(fake)

        loss = discriminator_loss(real_logits, fake_logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Store gradient sizes from several batches so the comparison
    # is not based on only one random batch.
    minimax_gradients = []
    nonsaturating_gradients = []
    probabilities = []

    for _ in range(measurement_batches):

        z = torch.randn(batch_size, generator.d_noise, device=DEVICE)

        # We calculate gradients with respect to the generated samples.
        fake = generator(z).detach().requires_grad_(True)

        fake_logits = discriminator(fake)
        fake_probabilities = torch.sigmoid(fake_logits)

        minimax_loss = generator_minimax_loss(fake_logits)
        nonsaturating_loss = generator_nonsaturating_loss(fake_logits)

        # Find the gradient for each generator loss.
        minimax_gradient = torch.autograd.grad(
            minimax_loss,
            fake,
            retain_graph=True
        )[0]

        nonsaturating_gradient = torch.autograd.grad(
            nonsaturating_loss,
            fake
        )[0]

        # Save the average gradient size for each batch.
        minimax_gradients.append(
            torch.linalg.vector_norm(minimax_gradient, dim=1).mean().item()
        )

        nonsaturating_gradients.append(
            torch.linalg.vector_norm(nonsaturating_gradient, dim=1).mean().item()
        )

        probabilities.append(fake_probabilities.mean().item())

    mean_probability = float(np.mean(probabilities))

    print("\nResults averaged over 100 batches:")

    print("\nMinimax generator loss:")
    print(f"Mean D(G(z)): {mean_probability:.10f}")
    print(f"Mean output gradient norm: {np.mean(minimax_gradients):.10f}")

    print("\nNon-saturating generator loss:")
    print(f"Mean D(G(z)): {mean_probability:.10f}")
    print(f"Mean output gradient norm: {np.mean(nonsaturating_gradients):.10f}")

    print("Check 3 completed.")


# PART 2: STANDARD GAN AND WGAN-GP

# These settings are used for both GAN models so their results
# can be compared using the same data dimensions and batch size.
LATENT_DIM = 16
HIDDEN_DIM = 128
BATCH_SIZE = 128
GENERATOR_STEPS = 4000
NUMBER_OF_SAMPLES = 359

# Keep the trained WGAN-GP models and their training times.
# Part 5 uses these models again instead of training them a second time.
PART2_WGAN_MODELS = {}
PART2_WGAN_TRAINING_TIMES = {}


# CHECK 4: Gradient penalty verification

def gradient_penalty_from_norms(gradient_norms):

    # The penalty is zero when the gradient norm is 1.
    # It increases when the norm moves away from 1.
    return ((gradient_norms - 1.0) ** 2).mean()


def gradient_penalty(critic, real, fake):

    batch_size = real.size(0)

    # Choose a random point between each real and generated sample.
    # The critic's gradient is measured at these interpolated points.
    epsilon = torch.rand(
        batch_size,
        1,
        device=real.device
    )

    interpolated = (
        epsilon * real + (1.0 - epsilon) * fake
    ).requires_grad_(True)

    critic_output = critic(interpolated)

    # Calculate how the critic output changes with its input.
    gradients = torch.autograd.grad(
        outputs=critic_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Calculate the gradient length for each sample.
    gradient_norms = torch.linalg.vector_norm(
        gradients,
        dim=1
    )

    # Penalise gradient lengths that are different from 1.
    penalty = gradient_penalty_from_norms(gradient_norms)

    return penalty, gradient_norms.detach()


def check_gradient_penalty():

    print("\n" + "=" * 65)
    print("CHECK 4: WGAN-GP GRADIENT PENALTY VERIFICATION")
    print("=" * 65)

    # Test two known gradient lengths.
    # A norm of 1 should give a penalty of 0.
    # A norm of 3 should give a penalty of 4.
    test_norms = torch.tensor([1.0, 3.0])

    penalties = (test_norms - 1.0) ** 2

    print(f"Gradient norm 1: penalty = {penalties[0].item():.6f}")
    print(f"Gradient norm 3: penalty = {penalties[1].item():.6f}")

    # Stop the experiment if either expected value is incorrect.
    assert torch.isclose(penalties[0], torch.tensor(0.0))
    assert torch.isclose(penalties[1], torch.tensor(4.0))

    print("PASS: Both gradient penalty checks passed.")


# Create digit GAN models

def create_gan_models():

    # The generator takes random noise and creates a 64-value image.
    # Sigmoid keeps the generated pixel values between 0 and 1.
    generator = Generator(
        d_noise=LATENT_DIM,
        d_out=64,
        d_hidden=HIDDEN_DIM,
        out_act="sigmoid"
    ).to(DEVICE)

    # The discriminator receives a flattened 64-value image
    # and returns a score for that image.
    discriminator = Discriminator(
        d_in=64,
        d_hidden=HIDDEN_DIM
    ).to(DEVICE)

    return generator, discriminator


# Train standard GAN

def train_standard_gan(X_train, seed):

    print(f"\nTraining Standard GAN | Seed {seed}")

    set_seed(seed)

    generator, discriminator = create_gan_models()

    # Use separate optimizers because the generator and discriminator
    # have different learning rates.
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=5e-4,
        betas=(0.5, 0.999)
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=2e-4,
        betas=(0.5, 0.999)
    )

    start_time = time.perf_counter()

    for step in range(GENERATOR_STEPS):

        # Randomly select a batch of real training images.
        indices = torch.randint(
            0,
            len(X_train),
            (BATCH_SIZE,)
        )

        real = X_train[indices].to(DEVICE)

        # First update the discriminator.
        # The generator does not receive gradients during this step.
        z = torch.randn(BATCH_SIZE, LATENT_DIM, device=DEVICE)

        with torch.no_grad():
            fake = generator(z)

        real_logits = discriminator(real)
        fake_logits = discriminator(fake)

        d_loss = discriminator_loss(real_logits, fake_logits)

        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # Next update the generator.
        # It tries to create samples that the discriminator accepts as real.
        z = torch.randn(BATCH_SIZE, LATENT_DIM, device=DEVICE)

        fake = generator(z)
        fake_logits = discriminator(fake)

        g_loss = generator_nonsaturating_loss(fake_logits)

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

        # Print both losses regularly to check how training is progressing.
        if (step + 1) % 500 == 0:
            print(
                f"Step {step + 1}/{GENERATOR_STEPS} "
                f"| D Loss: {d_loss.item():.4f} "
                f"| G Loss: {g_loss.item():.4f}"
            )

    training_time = time.perf_counter() - start_time

    print(f"Standard GAN training time: {training_time:.2f} seconds")

    return generator, training_time


# Train WGAN-GP

def train_wgan_gp(X_train, seed):

    print(f"\nTraining WGAN-GP | Seed {seed}")

    set_seed(seed)

    generator, critic = create_gan_models()

    # The critic and generator use separate optimizers.
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=5e-4,
        betas=(0.5, 0.999)
    )

    optimizer_c = torch.optim.Adam(
        critic.parameters(),
        lr=5e-4,
        betas=(0.5, 0.999)
    )

    # The critic is updated three times for every generator update.
    critic_updates_per_generator = 3
    gp_weight = 10.0

    # Keep the gradient norms and score gaps so we can compare
    # the early and later stages of critic training.
    first_100_gradient_norms = []
    first_100_score_gaps = []

    last_500_gradient_norms = []
    last_500_score_gaps = []

    total_critic_updates = GENERATOR_STEPS * critic_updates_per_generator

    start_time = time.perf_counter()

    critic_update = 0

    for step in range(GENERATOR_STEPS):

        # Train the critic three times before updating the generator.
        for _ in range(critic_updates_per_generator):

            indices = torch.randint(
                0,
                len(X_train),
                (BATCH_SIZE,)
            )

            real = X_train[indices].to(DEVICE)

            z = torch.randn(BATCH_SIZE, LATENT_DIM, device=DEVICE)

            with torch.no_grad():
                fake = generator(z)

            real_scores = critic(real)
            fake_scores = critic(fake)

            # Calculate the gradient penalty using points between
            # real and generated samples.
            gp, gradient_norms = gradient_penalty(
                critic,
                real,
                fake
            )

            # The critic tries to give higher scores to real samples.
            # The gradient penalty is added to keep its gradients controlled.
            critic_loss = (
                fake_scores.mean()
                - real_scores.mean()
                + gp_weight * gp
            )

            optimizer_c.zero_grad()
            critic_loss.backward()
            optimizer_c.step()

            # Record the gradient norm and the difference between
            # the average real and fake critic scores.
            mean_gradient_norm = gradient_norms.mean().item()

            score_gap = (
                real_scores.mean() - fake_scores.mean()
            ).item()

            critic_update += 1

            # Save measurements from the first 100 critic updates.
            if critic_update <= 100:
                first_100_gradient_norms.append(mean_gradient_norm)
                first_100_score_gaps.append(score_gap)

            # Save measurements from the final 500 critic updates.
            if critic_update > total_critic_updates - 500:
                last_500_gradient_norms.append(mean_gradient_norm)
                last_500_score_gaps.append(score_gap)

        # Update the generator after the three critic updates.
        z = torch.randn(BATCH_SIZE, LATENT_DIM, device=DEVICE)

        fake = generator(z)

        generator_loss = -critic(fake).mean()

        optimizer_g.zero_grad()
        generator_loss.backward()
        optimizer_g.step()

        if (step + 1) % 500 == 0:
            print(
                f"Step {step + 1}/{GENERATOR_STEPS} "
                f"| Critic Loss: {critic_loss.item():.4f} "
                f"| Generator Loss: {generator_loss.item():.4f} "
                f"| GP: {gp.item():.4f}"
            )

    training_time = time.perf_counter() - start_time

    print(f"WGAN-GP training time: {training_time:.2f} seconds")

    # Calculate average values for the early and later training periods.
    monitoring = {
        "first_100_gradient_norm": float(np.mean(first_100_gradient_norms)),
        "first_100_score_gap": float(np.mean(first_100_score_gaps)),
        "last_500_gradient_norm": float(np.mean(last_500_gradient_norms)),
        "last_500_score_gap": float(np.mean(last_500_score_gaps)),
    }

    print("\nWGAN-GP critic monitoring:")
    for name, value in monitoring.items():
        print(f"{name}: {value:.6f}")

    return generator, training_time, monitoring


# Generate and evaluate samples

@torch.no_grad()
def generate_samples(generator, seed):

    generator.eval()

    # Use a separate seed for sample generation.
    # This keeps the sampling process repeatable.
    set_seed(seed + 1000)

    z = torch.randn(
        NUMBER_OF_SAMPLES,
        LATENT_DIM,
        device=DEVICE
    )

    samples = generator(z)

    # Keep the generated pixel values in the expected range.
    return samples.clamp(0.0, 1.0).cpu()


def evaluate_gan(
    model_name,
    generator,
    training_time,
    X_train,
    X_test,
    y_train,
    y_test,
    seed,
    feature_net
):

    # Generate the same number of samples for each GAN experiment.
    samples = generate_samples(generator, seed)

    # Extract features from the real test images.
    test_features = get_features(feature_net, X_test)

    # Calculate the quality and coverage measurements.
    metrics = score_samples(
        samples,
        test_features,
        feature_net,
        X_train
    )

    # Use the feature network to estimate the digit class of each sample.
    with torch.no_grad():
        predicted_classes = feature_net(samples).argmax(dim=1)

    distinct_classes = int(torch.unique(predicted_classes).numel())

    # Add experiment details to the evaluation results.
    metrics["distinct_classes"] = distinct_classes
    metrics["training_time_seconds"] = training_time
    metrics["seed"] = seed
    metrics["model"] = model_name

    # Save a grid of generated images so their appearance can be checked.
    figure_path = (
        FIGURES_DIR /
        f"{model_name.lower().replace('-', '_')}_seed_{seed}.png"
    )

    save_grid(
        samples[:30],
        figure_path,
        ncols=10,
        title=f"{model_name} samples - seed {seed}"
    )

    print(f"\n{model_name} evaluation | Seed {seed}")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    return metrics


# Save Part 2 results

def save_part2_results(results):

    # Store the Standard GAN and WGAN-GP results in one CSV file.
    csv_path = RESULTS_DIR / "part2_gan_comparison.csv"

    fieldnames = [
        "model",
        "seed",
        "frechet",
        "precision",
        "recall",
        "distinct_classes",
        "training_time_seconds"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for result in results:

            writer.writerow({
                key: result[key]
                for key in fieldnames
            })

    print(f"\nPart 2 results saved: {csv_path}")


def save_wgan_monitoring(monitoring_results):

    # Save the critic measurements separately from the main comparison.
    csv_path = RESULTS_DIR / "part2_wgan_gp_monitoring.csv"

    fieldnames = [
        "seed",
        "first_100_gradient_norm",
        "first_100_score_gap",
        "last_500_gradient_norm",
        "last_500_score_gap"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for result in monitoring_results:
            writer.writerow(result)

    print(f"WGAN-GP monitoring saved: {csv_path}")


# Run Part 2 for seeds 0, 1 and 2

def run_part2():

    print("\n" + "=" * 65)
    print("PART 2: STANDARD GAN VS WGAN-GP")
    print("=" * 65)

    all_results = []
    monitoring_results = []

    # Remove any old saved models before starting a new run.
    PART2_WGAN_MODELS.clear()
    PART2_WGAN_TRAINING_TIMES.clear()

    for seed in PART2_SEEDS:

        print("\n" + "#" * 65)
        print(f"EXPERIMENT SEED: {seed}")
        print("#" * 65)

        set_seed(seed)

        # Load the training and test data for this seed.
        X_train, y_train, X_test, y_test = load_task(seed=seed)

        # Train one feature network to evaluate both GAN models.
        feature_net = train_feature_net(
            X_train,
            y_train,
            seed=seed
        )

        feature_net.eval()

        # Train and evaluate the Standard GAN.
        standard_generator, standard_time = train_standard_gan(
            X_train,
            seed
        )

        standard_result = evaluate_gan(
            "Standard-GAN",
            standard_generator,
            standard_time,
            X_train,
            X_test,
            y_train,
            y_test,
            seed,
            feature_net
        )

        all_results.append(standard_result)

        # Train and evaluate WGAN-GP using the same data split.
        wgan_generator, wgan_time, monitoring = train_wgan_gp(
            X_train,
            seed
        )

        # Keep the trained WGAN-GP model and training time.
        # Part 5 will use this model again.
        PART2_WGAN_MODELS[seed] = wgan_generator
        PART2_WGAN_TRAINING_TIMES[seed] = wgan_time

        wgan_result = evaluate_gan(
            "WGAN-GP",
            wgan_generator,
            wgan_time,
            X_train,
            X_test,
            y_train,
            y_test,
            seed,
            feature_net
        )

        all_results.append(wgan_result)

        monitoring["seed"] = seed
        monitoring_results.append(monitoring)

        # The Standard GAN model is not needed after evaluation.
        # Keep the WGAN-GP model because Part 5 needs it.
        del standard_generator
        del feature_net

    # Save the results after all three seeds have been completed.
    save_part2_results(all_results)
    save_wgan_monitoring(monitoring_results)

    print("\n" + "=" * 65)
    print("PART 2 EXPERIMENTS COMPLETED")
    print("=" * 65)

    print("\nFinal comparison:")

    for result in all_results:

        print(
            f"{result['model']} | "
            f"Seed {result['seed']} | "
            f"Frechet: {result['frechet']:.4f} | "
            f"Precision: {result['precision']:.4f} | "
            f"Recall: {result['recall']:.4f} | "
            f"Classes: {result['distinct_classes']}"
        )


# PART 3: DIFFUSION MODEL

# The diffusion process gradually adds noise over 100 steps.
DIFFUSION_STEPS = 100
DIFFUSION_BATCH_SIZE = 128
DIFFUSION_SAMPLES = 359
DIFFUSION_SEEDS = [0, 1, 2]


# Create the diffusion schedule

def create_diffusion_schedule():

    # BETAS contains the noise amounts for each step.
    betas = BETAS.to(DEVICE)

    # Alpha is the amount of information kept at each step.
    alphas = 1.0 - betas

    # Alpha bar is the product of alpha values up to the current step.
    alpha_bars = torch.cumprod(alphas, dim=0)

    return betas, alphas, alpha_bars


# Forward diffusion: add noise to clean data

def forward_diffusion(x0, timesteps, alpha_bars):

    # Select the alpha bar value for each sample's timestep.
    alpha_bar_t = alpha_bars[timesteps].view(-1, 1)

    # Generate random noise with the same shape as the input.
    noise = torch.randn_like(x0)

    # Combine the original data and noise using the diffusion formula.
    xt = (
        torch.sqrt(alpha_bar_t) * x0
        + torch.sqrt(1.0 - alpha_bar_t) * noise
    )

    return xt, noise


# CHECK 5: Verify forward diffusion

def check_forward_diffusion():

    print("\n" + "=" * 65)
    print("CHECK 5: FORWARD DIFFUSION VERIFICATION")
    print("=" * 65)

    torch.manual_seed(SEED)

    betas, alphas, alpha_bars = create_diffusion_schedule()

    # Use many values so the mean and standard deviation
    # give a useful comparison between the two methods.
    number_of_values = 100000

    x0 = torch.full((number_of_values, 1), 1.5)

    # First, apply the forward process one step at a time.
    iterative_x = x0.clone()

    iterative_results = {}

    with torch.no_grad():

        for t in range(DIFFUSION_STEPS):

            noise = torch.randn_like(iterative_x)

            iterative_x = (
                torch.sqrt(alphas[t]) * iterative_x
                + torch.sqrt(betas[t]) * noise
            )

            # Keep the results at steps 10, 50 and 100.
            if t in [9, 49, 99]:
                iterative_results[t] = iterative_x.clone()

        # Next, calculate the same steps directly using the closed-form formula.
        # New random noise is used here, so we compare the distributions
        # rather than expecting every individual value to be identical.
        closed_form_results = {}

        for t in [9, 49, 99]:

            noise = torch.randn_like(x0)

            closed_form_x = (
                torch.sqrt(alpha_bars[t]) * x0
                + torch.sqrt(1.0 - alpha_bars[t]) * noise
            )

            closed_form_results[t] = closed_form_x

    print("\nComparison of iterative and closed-form distributions:")

    print(
        f"{'Step':<10}"
        f"{'Iterative Mean':<20}"
        f"{'Closed Mean':<20}"
        f"{'Iterative Std':<20}"
        f"{'Closed Std':<20}"
    )

    for t in [9, 49, 99]:

        iterative = iterative_results[t]
        closed = closed_form_results[t]

        print(
            f"{t + 1:<10}"
            f"{iterative.mean().item():<20.4f}"
            f"{closed.mean().item():<20.4f}"
            f"{iterative.std().item():<20.4f}"
            f"{closed.std().item():<20.4f}"
        )

    print(
        "\nFinal alpha_bar:",
        f"{alpha_bars[-1].item():.8f}"
    )

    print("Forward diffusion verification completed.")

    return betas, alphas, alpha_bars


# Noise prediction training loss

def diffusion_training_loss(model, x0, alpha_bars):

    batch_size = x0.size(0)

    # Pick a random timestep for each image in the batch.
    timesteps = torch.randint(
        0,
        DIFFUSION_STEPS,
        (batch_size,),
        device=DEVICE
    )

    # Add noise to the original images at the selected timesteps.
    xt, noise = forward_diffusion(
        x0,
        timesteps,
        alpha_bars
    )

    # Ask the model to predict the noise that was added.
    predicted_noise = model(xt, timesteps)

    # Measure the difference between the real and predicted noise.
    loss = F.mse_loss(
        predicted_noise,
        noise
    )

    return loss


# Ancestral reverse diffusion sampler

@torch.no_grad()
def ancestral_sample(model, number_of_samples, data_dimension, seed=0):

    model.eval()

    set_seed(seed)

    betas, alphas, alpha_bars = create_diffusion_schedule()

    # Start with random noise. The model will gradually remove it.
    x = torch.randn(
        number_of_samples,
        data_dimension,
        device=DEVICE
    )

    # Move backwards from the last diffusion step to the first.
    for t in reversed(range(DIFFUSION_STEPS)):

        timesteps = torch.full(
            (number_of_samples,),
            t,
            device=DEVICE,
            dtype=torch.long
        )

        # Predict the noise present at the current step.
        predicted_noise = model(x, timesteps)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]

        # Calculate the mean of the reverse distribution.
        mean = (
            x
            - (beta_t / torch.sqrt(1.0 - alpha_bar_t))
            * predicted_noise
        ) / torch.sqrt(alpha_t)

        if t > 0:

            alpha_bar_previous = alpha_bars[t - 1]

            # Calculate the amount of random noise to add at this step.
            sigma = torch.sqrt(
                beta_t
                * (1.0 - alpha_bar_previous)
                / (1.0 - alpha_bar_t)
            )

            noise = torch.randn_like(x)

            x = mean + sigma * noise

        else:

            # Do not add more noise after reaching the final step.
            x = mean

    return x


# Train diffusion model on ring-of-Gaussians data

def train_ring_diffusion(seed=0):

    print(f"\nTraining ring diffusion | Seed {seed}")

    set_seed(seed)

    # This model learns to predict noise in two-dimensional samples.
    model = Denoiser(
        d_data=2,
        d_hidden=128,
        T=DIFFUSION_STEPS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=2e-3
    )

    _, _, alpha_bars = create_diffusion_schedule()

    training_steps = 4000

    start_time = time.perf_counter()

    for step in range(training_steps):

        # Generate a batch of real samples arranged in eight Gaussian modes.
        real, _ = ring_of_gaussians(
            DIFFUSION_BATCH_SIZE,
            seed=seed * training_steps + step
        )

        real = real.to(DEVICE)

        # Train the model to predict the noise added to these samples.
        loss = diffusion_training_loss(
            model,
            real,
            alpha_bars
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 500 == 0:

            print(
                f"Step {step + 1}/{training_steps} "
                f"| Noise MSE: {loss.item():.6f}"
            )

    training_time = time.perf_counter() - start_time

    print(
        f"Ring diffusion training time: "
        f"{training_time:.2f} seconds"
    )

    return model, training_time


# Evaluate ring diffusion

def evaluate_ring_diffusion(model, seed=0):

    # Generate 2,000 samples by starting from random noise.
    samples = ancestral_sample(
        model,
        number_of_samples=2000,
        data_dimension=2,
        seed=seed + 5000
    )

    # Check how many of the eight modes are represented in the samples.
    covered_modes, near_share = mode_coverage(
        samples.cpu()
    )

    print("\nRing diffusion evaluation:")
    print(f"Modes covered: {covered_modes}/8")
    print(f"Share near a mode: {near_share:.4f}")

    # Plot the generated samples to see how well they cover the ring.
    plt.figure(figsize=(7, 7))

    plt.scatter(
        samples[:, 0].cpu().numpy(),
        samples[:, 1].cpu().numpy(),
        s=8,
        alpha=0.5
    )

    plt.title(f"Ring Diffusion Samples - Seed {seed}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    figure_path = FIGURES_DIR / f"ring_diffusion_seed_{seed}.png"

    plt.savefig(figure_path, dpi=150)
    plt.close()

    print(f"Figure saved: {figure_path}")

    return {
        "seed": seed,
        "modes_covered": covered_modes,
        "share_near_mode": near_share
    }


# Train diffusion model on digit images

def train_digit_diffusion(X_train, seed=0):

    print(f"\nTraining digit diffusion | Seed {seed}")

    set_seed(seed)

    # The input has 64 values because each digit image is 8 by 8 pixels.
    model = Denoiser(
        d_data=64,
        d_hidden=256,
        T=DIFFUSION_STEPS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    # Reduce the learning rate gradually during training.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=4000,
        eta_min=0.0
    )

    _, _, alpha_bars = create_diffusion_schedule()

    training_steps = 4000

    # Keep the early and later losses to see how training changes.
    first_100_losses = []
    last_200_losses = []

    start_time = time.perf_counter()

    for step in range(training_steps):

        # Select a random batch of training images.
        indices = torch.randint(
            0,
            len(X_train),
            (DIFFUSION_BATCH_SIZE,)
        )

        real = X_train[indices].to(DEVICE)

        # The diffusion model is trained with pixel values between -1 and 1.
        real = real * 2.0 - 1.0

        loss = diffusion_training_loss(
            model,
            real,
            alpha_bars
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_value = loss.item()

        # Record the first 100 losses.
        if step < 100:
            first_100_losses.append(loss_value)

        # Record the final 200 losses.
        if step >= training_steps - 200:
            last_200_losses.append(loss_value)

        if (step + 1) % 500 == 0:

            print(
                f"Step {step + 1}/{training_steps} "
                f"| Noise MSE: {loss_value:.6f} "
                f"| LR: {scheduler.get_last_lr()[0]:.7f}"
            )

    training_time = time.perf_counter() - start_time

    # Save average losses and training time for later comparison.
    training_metrics = {
        "first_100_loss": float(np.mean(first_100_losses)),
        "last_200_loss": float(np.mean(last_200_losses)),
        "training_time_seconds": training_time
    }

    print("\nDigit diffusion training summary:")

    for name, value in training_metrics.items():
        print(f"{name}: {value:.6f}")

    return model, training_metrics


# Generate digit images from the trained diffusion model

@torch.no_grad()
def generate_diffusion_digits(model, seed=0):

    samples = ancestral_sample(
        model,
        number_of_samples=DIFFUSION_SAMPLES,
        data_dimension=64,
        seed=seed + 6000
    )

    # Convert the generated pixel values from [-1, 1] back to [0, 1].
    samples = ((samples + 1.0) / 2.0).clamp(0.0, 1.0)

    return samples.cpu()


# Evaluate digit diffusion

def evaluate_digit_diffusion(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    seed,
    feature_net,
    training_metrics
):

    samples = generate_diffusion_digits(
        model,
        seed
    )

    # Use the feature network to compare generated images with test images.
    test_features = get_features(
        feature_net,
        X_test
    )

    metrics = score_samples(
        samples,
        test_features,
        feature_net,
        X_train
    )

    # Predict the digit class for each generated image.
    with torch.no_grad():

        predicted_classes = feature_net(samples).argmax(dim=1)

    distinct_classes = int(
        torch.unique(predicted_classes).numel()
    )

    # Include the training measurements with the sample evaluation results.
    metrics.update(training_metrics)

    metrics["distinct_classes"] = distinct_classes
    metrics["seed"] = seed
    metrics["model"] = "Diffusion"

    # Save a grid of generated digit images for visual inspection.
    figure_path = (
        FIGURES_DIR / f"diffusion_digits_seed_{seed}.png"
    )

    save_grid(
        samples[:30],
        figure_path,
        ncols=10,
        title=f"Diffusion digit samples - seed {seed}"
    )

    print(f"\nDigit diffusion evaluation | Seed {seed}")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    return metrics


# Save Part 3 results

def save_part3_results(digit_results, ring_results):

    # Save the digit experiment results in a CSV file.
    digit_path = RESULTS_DIR / "part3_diffusion_digits.csv"

    if digit_results:

        with open(
            digit_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(digit_results[0].keys())
            )

            writer.writeheader()
            writer.writerows(digit_results)

        print(f"\nDigit diffusion results saved: {digit_path}")

    # Save the ring experiment results in a separate CSV file.
    ring_path = RESULTS_DIR / "part3_diffusion_ring.csv"

    if ring_results:

        with open(
            ring_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(ring_results[0].keys())
            )

            writer.writeheader()
            writer.writerows(ring_results)

        print(f"Ring diffusion results saved: {ring_path}")


# Run all Part 3 experiments

# Keep the trained digit models and their results.
# Part 4 and Part 5 use these models again.
PART3_DIGIT_MODELS = {}
PART3_DIGIT_RESULTS = []
PART3_RING_RESULTS = []


def run_part3():

    print("\n" + "=" * 65)
    print("PART 3: DIFFUSION MODEL")
    print("=" * 65)

    # Check that the forward diffusion calculations are working.
    check_forward_diffusion()

    # First, train and evaluate diffusion on the ring data.
    ring_model, ring_training_time = train_ring_diffusion(seed=0)

    ring_result = evaluate_ring_diffusion(
        ring_model,
        seed=0
    )

    ring_result["training_time_seconds"] = ring_training_time

    PART3_RING_RESULTS.append(ring_result)

    del ring_model

    # Next, train and evaluate a digit diffusion model for each seed.
    for seed in DIFFUSION_SEEDS:

        print("\n" + "#" * 65)
        print(f"DIGIT DIFFUSION EXPERIMENT | SEED {seed}")
        print("#" * 65)

        set_seed(seed)

        X_train, y_train, X_test, y_test = load_task(
            seed=seed
        )

        # Train a feature network to measure the generated image quality.
        feature_net = train_feature_net(
            X_train,
            y_train,
            seed=seed
        )

        feature_net.eval()

        model, training_metrics = train_digit_diffusion(
            X_train,
            seed=seed
        )

        result = evaluate_digit_diffusion(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            seed,
            feature_net,
            training_metrics
        )

        # Save the model and its results for use in later parts.
        PART3_DIGIT_MODELS[seed] = model
        PART3_DIGIT_RESULTS.append(result)

        del feature_net

    save_part3_results(
        PART3_DIGIT_RESULTS,
        PART3_RING_RESULTS
    )

    print("\n" + "=" * 65)
    print("PART 3 EXPERIMENTS COMPLETED")
    print("=" * 65)


# PART 4: DIFFUSION SAMPLING COMPARISON

# Test different numbers of steps with the deterministic sampler.
SAMPLING_STEPS = [5, 10, 25, 50, 100]


# Deterministic diffusion sampler

@torch.no_grad()
def deterministic_sample(
    model,
    number_of_samples,
    data_dimension,
    sampling_steps,
    seed=0
):

    model.eval()

    set_seed(seed)

    betas, alphas, alpha_bars = create_diffusion_schedule()

    # Start with random noise. The same seed makes the starting noise
    # repeatable when comparing different sampling methods.
    x = torch.randn(
        number_of_samples,
        data_dimension,
        device=DEVICE
    )

    # Select evenly spaced timesteps from 99 down to 0.
    # This allows the sampler to use fewer than 100 steps.
    timesteps_list = np.linspace(
        DIFFUSION_STEPS - 1,
        0,
        sampling_steps
    ).round().astype(int)

    timesteps_list = list(dict.fromkeys(timesteps_list.tolist()))

    for index, t in enumerate(timesteps_list):

        timesteps = torch.full(
            (number_of_samples,),
            t,
            device=DEVICE,
            dtype=torch.long
        )

        predicted_noise = model(x, timesteps)

        alpha_bar_t = alpha_bars[t]

        # Estimate the original clean sample from the noisy input
        # and the noise predicted by the model.
        predicted_x0 = (
            x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
        ) / torch.sqrt(alpha_bar_t)

        # Find the next timestep in the selected list.
        if index + 1 < len(timesteps_list):

            next_t = timesteps_list[index + 1]
            alpha_bar_next = alpha_bars[next_t]

        else:

            # At the final step, use alpha bar 1 to return the clean estimate.
            alpha_bar_next = torch.tensor(
                1.0,
                device=DEVICE
            )

        # Update the sample without adding new random noise.
        x = (
            torch.sqrt(alpha_bar_next) * predicted_x0
            + torch.sqrt(1.0 - alpha_bar_next) * predicted_noise
        )

    return x


# Evaluate one sampling method

def evaluate_sampling_method(
    model,
    method_name,
    sampling_steps,
    X_train,
    X_test,
    feature_net,
    seed
):

    start_time = time.perf_counter()

    # Run the regular ancestral sampler using all 100 steps.
    if method_name == "Ancestral":

        samples = ancestral_sample(
            model,
            number_of_samples=DIFFUSION_SAMPLES,
            data_dimension=64,
            seed=seed + 7000
        )

        network_evaluations = DIFFUSION_STEPS

    else:

        # Run the deterministic sampler using the requested number of steps.
        samples = deterministic_sample(
            model,
            number_of_samples=DIFFUSION_SAMPLES,
            data_dimension=64,
            sampling_steps=sampling_steps,
            seed=seed + 7000
        )

        network_evaluations = len(
            np.linspace(
                DIFFUSION_STEPS - 1,
                0,
                sampling_steps
            ).round().astype(int)
        )

    sampling_time = time.perf_counter() - start_time

    # Convert generated images from [-1, 1] to [0, 1].
    samples = (
        (samples + 1.0) / 2.0
    ).clamp(0.0, 1.0).cpu()

    # Use the same feature network to evaluate the generated samples.
    test_features = get_features(
        feature_net,
        X_test
    )

    metrics = score_samples(
        samples,
        test_features,
        feature_net,
        X_train
    )

    # Count how many different digit classes appear in the generated images.
    with torch.no_grad():

        predicted_classes = feature_net(samples).argmax(dim=1)

    distinct_classes = int(
        torch.unique(predicted_classes).numel()
    )

    # Keep the main quality measurements and sampling time.
    result = {
        "seed": seed,
        "method": method_name,
        "sampling_steps": network_evaluations,
        "frechet": metrics["frechet"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "nearest_train": metrics["nearest_train"],
        "distinct_classes": distinct_classes,
        "sampling_time_seconds": sampling_time
    }

    # Save a sample grid for each method and seed.
    if method_name == "Ancestral":

        filename = f"part4_ancestral_seed_{seed}.png"

    else:

        filename = (
            f"part4_deterministic_{sampling_steps}_steps_seed_{seed}.png"
        )

    figure_path = FIGURES_DIR / filename

    save_grid(
        samples[:30],
        figure_path,
        ncols=10,
        title=f"{method_name} sampling - {network_evaluations} steps - seed {seed}"
    )

    print(
        f"\n{method_name} | "
        f"Steps: {network_evaluations} | Seed: {seed}"
    )

    for name, value in result.items():
        print(f"{name}: {value}")

    return result


# Save Part 4 results

def save_part4_results(results):

    csv_path = RESULTS_DIR / "part4_sampling_comparison.csv"

    fieldnames = [
        "seed",
        "method",
        "sampling_steps",
        "frechet",
        "precision",
        "recall",
        "nearest_train",
        "distinct_classes",
        "sampling_time_seconds"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nPart 4 results saved: {csv_path}")


# Run all Part 4 experiments

def run_part4():

    print("\n" + "=" * 65)
    print("PART 4: DIFFUSION SAMPLING COMPARISON")
    print("=" * 65)

    all_results = []

    for seed in DIFFUSION_SEEDS:

        print("\n" + "#" * 65)
        print(f"SAMPLING EXPERIMENT | SEED {seed}")
        print("#" * 65)

        # Reuse the diffusion model trained in Part 3.
        if seed not in PART3_DIGIT_MODELS:

            raise RuntimeError(
                f"Missing trained diffusion model for seed {seed}. "
                "Run Part 3 before Part 4."
            )

        model = PART3_DIGIT_MODELS[seed]
        model.eval()

        # Load the same data split and train one feature network
        # for all sampling methods tested with this seed.
        set_seed(seed)

        X_train, y_train, X_test, y_test = load_task(
            seed=seed
        )

        feature_net = train_feature_net(
            X_train,
            y_train,
            seed=seed
        )

        feature_net.eval()

        # Compare deterministic sampling with different step counts.
        for number_of_steps in SAMPLING_STEPS:

            result = evaluate_sampling_method(
                model=model,
                method_name="Deterministic",
                sampling_steps=number_of_steps,
                X_train=X_train,
                X_test=X_test,
                feature_net=feature_net,
                seed=seed
            )

            all_results.append(result)

        # Also test the ancestral sampler using all 100 steps.
        result = evaluate_sampling_method(
            model=model,
            method_name="Ancestral",
            sampling_steps=100,
            X_train=X_train,
            X_test=X_test,
            feature_net=feature_net,
            seed=seed
        )

        all_results.append(result)

        del feature_net

    save_part4_results(all_results)

    print("\n" + "=" * 65)
    print("PART 4 EXPERIMENTS COMPLETED")
    print("=" * 65)

    print("\nFinal sampling comparison:")

    for result in all_results:

        print(
            f"Seed {result['seed']} | "
            f"{result['method']} | "
            f"Steps: {result['sampling_steps']} | "
            f"Frechet: {result['frechet']:.4f} | "
            f"Precision: {result['precision']:.4f} | "
            f"Recall: {result['recall']:.4f}"
        )


# PART 5: VAE VS WGAN-GP VS DIFFUSION

# Use the same three seeds for the final comparison.
PART5_SEEDS = [0, 1, 2]

# Use the same number of training epochs for each VAE experiment.
VAE_EPOCHS = 30


# Calculate total variation distance

def calculate_tv_distance(real_labels, generated_labels):

    # Convert the labels to integer values before counting them.
    real_labels = torch.as_tensor(real_labels).long()
    generated_labels = torch.as_tensor(generated_labels).long()

    # Count how often each digit appears in the real and generated data.
    real_counts = torch.bincount(real_labels, minlength=10).float()
    generated_counts = torch.bincount(
        generated_labels,
        minlength=10
    ).float()

    # Turn the counts into probability distributions.
    real_distribution = real_counts / real_counts.sum()
    generated_distribution = (
        generated_counts / generated_counts.sum()
    )

    # TV distance measures the difference between the two distributions.
    # A smaller value means the distributions are closer.
    return (
        0.5 * torch.abs(
            real_distribution - generated_distribution
        ).sum().item()
    )


# Evaluate a model for Part 5

def evaluate_part5_model(
    model_name,
    samples,
    training_time,
    sampling_time,
    X_train,
    X_test,
    y_test,
    feature_net,
    seed,
    network_forward_calls,
    sample_level_network_evaluations
):

    # Make sure all samples are on the CPU and have pixel values from 0 to 1.
    samples = samples.detach().cpu().clamp(0.0, 1.0)

    # Extract features from the real test images.
    test_features = get_features(feature_net, X_test)

    # Calculate the main sample quality measurements.
    metrics = score_samples(
        samples,
        test_features,
        feature_net,
        X_train
    )

    # Measure how close generated samples are to their nearest training samples.
    distances = nearest_train_distance(samples, X_train)

    nearest_mean = distances.mean().item()
    nearest_minimum = distances.min().item()

    # Predict the digit class of each generated image.
    with torch.no_grad():
        generated_labels = feature_net(samples).argmax(dim=1)

    distinct_classes = int(
        torch.unique(generated_labels).numel()
    )

    # Compare the real and generated digit class distributions.
    tv_distance = calculate_tv_distance(
        y_test,
        generated_labels
    )

    # Collect all measurements in one result dictionary.
    result = {
        "seed": seed,
        "model": model_name,
        "frechet": metrics["frechet"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "nearest_train_mean": nearest_mean,
        "nearest_train_minimum": nearest_minimum,
        "tv_distance": tv_distance,
        "distinct_classes": distinct_classes,
        "training_time_seconds": training_time,
        "sampling_time_seconds": sampling_time,
        "network_forward_calls": network_forward_calls,
        "sample_level_network_evaluations": (
            sample_level_network_evaluations
        )
    }

    # Save a grid of images generated by this model.
    filename = (
        f"part5_{model_name.lower().replace('-', '_')}"
        f"_seed_{seed}.png"
    )

    figure_path = FIGURES_DIR / filename

    save_grid(
        samples[:30],
        figure_path,
        ncols=10,
        title=f"{model_name} samples - seed {seed}"
    )

    print(f"\nPart 5 evaluation: {model_name} | Seed {seed}")

    for name, value in result.items():
        print(f"{name}: {value}")

    return result


# VAE

def run_part5_vae(
    X_train,
    X_test,
    y_test,
    feature_net,
    seed
):

    print(f"\nTraining VAE | Seed {seed}")

    set_seed(seed)

    # Create and train the VAE using the helper function from the starter file.
    vae = VAE()

    start_time = time.perf_counter()

    vae = train_vae(
        vae,
        X_train,
        epochs=VAE_EPOCHS,
        seed=seed
    )

    training_time = time.perf_counter() - start_time

    vae.eval()

    # Generate the same number of images used by the other models.
    set_seed(seed + 8000)

    start_time = time.perf_counter()

    with torch.no_grad():
        samples = vae.sample(NUMBER_OF_SAMPLES)

    sampling_time = time.perf_counter() - start_time

    # Evaluate the VAE using the same measurements as the other models.
    result = evaluate_part5_model(
        model_name="VAE",
        samples=samples,
        training_time=training_time,
        sampling_time=sampling_time,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        feature_net=feature_net,
        seed=seed,
        network_forward_calls=1,
        sample_level_network_evaluations=NUMBER_OF_SAMPLES
    )

    del vae

    return result


# WGAN-GP: reuse Part 2 models

def run_part5_wgan(
    X_train,
    X_test,
    y_test,
    feature_net,
    seed
):

    print(f"\nEvaluating WGAN-GP | Seed {seed}")

    # Part 2 should have already trained and saved this model.
    # Reusing it avoids training WGAN-GP again.
    if seed not in PART2_WGAN_MODELS:
        raise RuntimeError(
            f"Missing WGAN-GP model for seed {seed}. "
            "Check the model-saving code in run_part2()."
        )

    generator = PART2_WGAN_MODELS[seed]
    generator.eval()

    training_time = PART2_WGAN_TRAINING_TIMES[seed]

    # Measure the time needed to generate the samples.
    start_time = time.perf_counter()

    samples = generate_samples(generator, seed)

    sampling_time = time.perf_counter() - start_time

    return evaluate_part5_model(
        model_name="WGAN-GP",
        samples=samples,
        training_time=training_time,
        sampling_time=sampling_time,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        feature_net=feature_net,
        seed=seed,
        network_forward_calls=1,
        sample_level_network_evaluations=NUMBER_OF_SAMPLES
    )


# Diffusion: reuse Part 3 models

def run_part5_diffusion(
    X_train,
    X_test,
    y_test,
    feature_net,
    seed
):

    print(f"\nEvaluating Diffusion | Seed {seed}")

    # Reuse the digit diffusion model trained in Part 3.
    if seed not in PART3_DIGIT_MODELS:
        raise RuntimeError(
            f"Missing diffusion model for seed {seed}. "
            "Run Part 3 before Part 5."
        )

    model = PART3_DIGIT_MODELS[seed]
    model.eval()

    # Get the training time recorded during Part 3.
    training_result = next(
        item for item in PART3_DIGIT_RESULTS
        if item["seed"] == seed
    )

    training_time = training_result["training_time_seconds"]

    # Generate samples and record how long sampling takes.
    start_time = time.perf_counter()

    samples = generate_diffusion_digits(model, seed)

    sampling_time = time.perf_counter() - start_time

    return evaluate_part5_model(
        model_name="Diffusion",
        samples=samples,
        training_time=training_time,
        sampling_time=sampling_time,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        feature_net=feature_net,
        seed=seed,
        network_forward_calls=DIFFUSION_STEPS,
        sample_level_network_evaluations=(
            NUMBER_OF_SAMPLES * DIFFUSION_STEPS
        )
    )


# Save Part 5 results

def save_part5_results(results):

    # Save the final comparison of VAE, WGAN-GP and diffusion.
    csv_path = (
        RESULTS_DIR / "part5_three_family_comparison.csv"
    )

    fieldnames = [
        "seed",
        "model",
        "frechet",
        "precision",
        "recall",
        "nearest_train_mean",
        "nearest_train_minimum",
        "tv_distance",
        "distinct_classes",
        "training_time_seconds",
        "sampling_time_seconds",
        "network_forward_calls",
        "sample_level_network_evaluations"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nPart 5 results saved: {csv_path}")


# Run Part 5

def run_part5():

    print("\n" + "=" * 65)
    print("PART 5: VAE VS WGAN-GP VS DIFFUSION")
    print("=" * 65)

    all_results = []

    for seed in PART5_SEEDS:

        print("\n" + "#" * 65)
        print(f"PART 5 COMPARISON | SEED {seed}")
        print("#" * 65)

        set_seed(seed)

        # Load the same data split for all three models with this seed.
        X_train, y_train, X_test, y_test = load_task(
            seed=seed
        )

        # Train one feature network and use it to evaluate all three models.
        feature_net = train_feature_net(
            X_train,
            y_train,
            seed=seed
        )

        feature_net.eval()

        # Train and evaluate the VAE.
        vae_result = run_part5_vae(
            X_train,
            X_test,
            y_test,
            feature_net,
            seed
        )

        all_results.append(vae_result)

        # Evaluate the WGAN-GP model saved from Part 2.
        wgan_result = run_part5_wgan(
            X_train,
            X_test,
            y_test,
            feature_net,
            seed
        )

        all_results.append(wgan_result)

        # Evaluate the diffusion model saved from Part 3.
        diffusion_result = run_part5_diffusion(
            X_train,
            X_test,
            y_test,
            feature_net,
            seed
        )

        all_results.append(diffusion_result)

        del feature_net

    # Save the results after all seeds and models have been evaluated.
    save_part5_results(all_results)

    print("\n" + "=" * 65)
    print("PART 5 EXPERIMENTS COMPLETED")
    print("=" * 65)

    print("\nFINAL THREE-MODEL COMPARISON")

    for result in all_results:

        print(
            f"Seed {result['seed']} | "
            f"{result['model']} | "
            f"Frechet: {result['frechet']:.4f} | "
            f"Precision: {result['precision']:.4f} | "
            f"Recall: {result['recall']:.4f} | "
            f"Nearest mean: {result['nearest_train_mean']:.4f} | "
            f"TV distance: {result['tv_distance']:.4f}"
        )


# MAIN: RUN ALL FIVE PARTS

if __name__ == "__main__":

    print("=" * 65)
    print("LAB 08 - TWO WAYS TO GENERATE")
    print("=" * 65)

    # Part 1 checks the GAN loss formulas, discriminator,
    # and generator gradients.
    print("\nRUNNING PART 1")

    check_gan_losses()
    check_gaussian_discriminator()
    check_generator_gradients()

    print("\nPART 1 COMPLETED")

    # Part 2 trains and compares Standard GAN and WGAN-GP.
    print("\nRUNNING PART 2")

    check_gradient_penalty()
    run_part2()

    print("\nPART 2 COMPLETED")

    # Part 3 checks the forward diffusion process and trains
    # diffusion models on ring data and digit images.
    print("\nRUNNING PART 3")

    run_part3()

    print("\nPART 3 COMPLETED")

    # Part 4 compares deterministic sampling with different
    # numbers of steps and the regular ancestral sampler.
    print("\nRUNNING PART 4")

    run_part4()

    print("\nPART 4 COMPLETED")

    # Part 5 compares the VAE, WGAN-GP and diffusion models
    # using the same evaluation measurements.
    print("\nRUNNING PART 5")

    run_part5()

    print("\nPART 5 COMPLETED")

    print("\n" + "=" * 65)
    print("ALL FIVE PARTS COMPLETED")
    print("=" * 65)
