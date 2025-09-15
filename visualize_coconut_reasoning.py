import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import random
import os
from train_mnist_simple import load_mnist_data
from mnist_coconut import create_mnist_coconut_from_pretrained, load_trained_coconut_model
import argparse


class CoconutWithTrajectoryCapture:
    """Wrapper to capture coconut's autoregressive reasoning trajectory"""

    def __init__(self, coconut_model):
        self.coconut_model = coconut_model
        self.reasoning_trajectory = []

    def capture_reasoning_trajectory(self, images, max_reasoning_steps=10):
        """Capture the step-by-step autoregressive reasoning process with growing sequences"""
        batch_size = images.shape[0]
        device = images.device
        self.reasoning_trajectory = []

        # Create reasoning sequence like in coconut generate()
        initial_sequence = []
        initial_sequence.append(self.coconut_model.start_latent_id)  # start token
        for _ in range(max_reasoning_steps):
            initial_sequence.append(self.coconut_model.latent_token_id)
        initial_sequence.append(self.coconut_model.end_latent_id)  # end token
        reasoning_sequence = torch.tensor([initial_sequence] * batch_size, device=device)

        # Get image embeddings
        image_embeds = self.coconut_model._get_image_embeddings(images)

        # Find latent positions for reference
        latent_indices = (reasoning_sequence == self.coconut_model.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(batch_size)
        ]
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0

        # Initialize growing reasoning sequence with start token
        growing_reasoning_embeds = self.coconut_model._get_sequence_embeddings(
            reasoning_sequence[:, :1], image_embeds[:, 0]  # Only start token initially
        )

        # Store initial state
        self.reasoning_trajectory.append({
            'step': 0,
            'sequence_embeds': growing_reasoning_embeds.clone().detach(),
            'reasoning_embeds': growing_reasoning_embeds.clone().detach(),
            'hidden_states': None,
            'sequence_length': growing_reasoning_embeds.shape[1],
            'description': 'Initial: [start]'
        })

        # Growing autoregressive reasoning loop
        for reasoning_step in range(max_n_latents + 1):
            # Create combined input: image patches + GROWING reasoning sequence
            combined_input = torch.cat([image_embeds, growing_reasoning_embeds], dim=1)

            # Add positional embeddings for current sequence length
            extended_pos_embed = self.coconut_model._get_extended_positional_embeddings(
                combined_input.shape[1]
            )
            combined_input = combined_input + extended_pos_embed
            combined_input = self.coconut_model.base_vision_model.dropout(combined_input)

            # Forward through transformer blocks
            x = combined_input
            for block in self.coconut_model.base_vision_model.blocks:
                x = block(x)
            x = self.coconut_model.base_vision_model.norm(x)

            # Extract reasoning hidden states (growing each step!)
            reasoning_hidden_states = x[:, image_embeds.shape[1]:, :]

            # Store current step with growing sequence info
            tokens_desc = f"[start" + ", latent" * (reasoning_hidden_states.shape[1] - 1) + "]"
            if reasoning_step == max_n_latents:
                tokens_desc = tokens_desc[:-1] + ", end]"

            self.reasoning_trajectory.append({
                'step': reasoning_step + 1,
                'sequence_embeds': growing_reasoning_embeds.clone().detach(),
                'reasoning_embeds': reasoning_hidden_states.clone().detach(),
                'hidden_states': reasoning_hidden_states.clone().detach(),
                'sequence_length': reasoning_hidden_states.shape[1],
                'description': f'Step {reasoning_step + 1}: {tokens_desc}'
            })

            # GROWING AUTOREGRESSIVE: Add next reasoning token to sequence
            if reasoning_step < max_n_latents:
                # Get the next latent token embedding
                next_token_pos = reasoning_step + 1  # +1 because we started with start token
                if next_token_pos < reasoning_sequence.shape[1]:
                    next_token_embeds = self.coconut_model._get_sequence_embeddings(
                        reasoning_sequence[:, next_token_pos:next_token_pos+1],
                        reasoning_hidden_states[:, -1, :]  # Use last hidden state as context
                    )

                    # GROW the sequence by appending next token
                    growing_reasoning_embeds = torch.cat([growing_reasoning_embeds, next_token_embeds], dim=1)
            else:
                # Final step: add end token
                end_token_embeds = self.coconut_model._get_sequence_embeddings(
                    reasoning_sequence[:, -1:], image_embeds[:, 0]  # End token
                )
                growing_reasoning_embeds = torch.cat([growing_reasoning_embeds, end_token_embeds], dim=1)

        # Generate final prediction
        final_logits = self.coconut_model.base_vision_model.head(reasoning_hidden_states)
        final_class_logits = final_logits[:, -1, :]  # Last position
        predictions = torch.argmax(final_class_logits, dim=1)

        return predictions, self.reasoning_trajectory


def extract_coconut_trajectories(coconut_wrapper, images, labels, max_reasoning_steps=10):
    """Extract coconut reasoning trajectories for visualization"""
    trajectories = []

    with torch.no_grad():
        for i, (image, label) in enumerate(zip(images, labels)):
            image_batch = image.unsqueeze(0)

            # Capture reasoning trajectory
            prediction, trajectory = coconut_wrapper.capture_reasoning_trajectory(
                image_batch, max_reasoning_steps
            )

            # Extract key embeddings from each step
            step_embeddings = []
            step_descriptions = []

            for step_data in trajectory:
                # Use the mean of reasoning embeddings as the representative point
                reasoning_embeds = step_data['reasoning_embeds']
                if reasoning_embeds is not None:
                    # Take mean across sequence length for this step
                    step_repr = reasoning_embeds.squeeze(0).mean(dim=0).cpu().numpy()
                else:
                    # For initial step, use sequence embeddings
                    step_repr = step_data['sequence_embeds'].squeeze(0).mean(dim=0).cpu().numpy()

                step_embeddings.append(step_repr)
                step_descriptions.append(step_data['description'])

            trajectories.append({
                'embeddings': step_embeddings,
                'descriptions': step_descriptions,
                'true_label': label.item(),
                'predicted_label': prediction.item(),
                'image_idx': i,
                'num_steps': len(step_embeddings)
            })

    return trajectories


def reduce_trajectories_to_2d(trajectories, method='pca'):
    """Reduce all trajectory points to 2D space"""
    # Collect all embeddings
    all_embeddings = []
    trajectory_info = []

    for traj in trajectories:
        for step_idx, embedding in enumerate(traj['embeddings']):
            all_embeddings.append(embedding)
            trajectory_info.append({
                'image_idx': traj['image_idx'],
                'step_idx': step_idx,
                'description': traj['descriptions'][step_idx],
                'true_label': traj['true_label'],
                'predicted_label': traj['predicted_label']
            })

    all_embeddings = np.array(all_embeddings)

    reducer = PCA(n_components=2, random_state=42)

    embeddings_2d = reducer.fit_transform(all_embeddings)

    # Reorganize back into trajectories
    trajectories_2d = []
    point_idx = 0

    for traj in trajectories:
        num_steps = len(traj['embeddings'])
        step_points = embeddings_2d[point_idx:point_idx + num_steps]

        trajectories_2d.append({
            'points_2d': step_points,
            'descriptions': traj['descriptions'],
            'true_label': traj['true_label'],
            'predicted_label': traj['predicted_label'],
            'image_idx': traj['image_idx'],
            'num_steps': num_steps
        })

        point_idx += num_steps

    return trajectories_2d


def plot_coconut_reasoning_trajectories(trajectories_2d, save_path="coconut_reasoning.png"):
    """Plot coconut's autoregressive reasoning trajectories"""

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.suptitle("Coconut MNIST: Reasoning Trajectories",
                 fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Get plot bounds for density estimation
    all_points = np.vstack([traj['points_2d'] for traj in trajectories_2d])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

    # Expand bounds slightly
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range

    # Plot all trajectories
    for traj in trajectories_2d:
        points = traj['points_2d']
        true_label = traj['true_label']
        pred_label = traj['predicted_label']
        is_correct = true_label == pred_label

        # Use solid line for hits, dotted for misses
        linestyle = '-' if is_correct else ':'
        alpha = 0.8 if is_correct else 0.5
        linewidth = 2 if is_correct else 1.5

        # Plot trajectory
        ax.plot(points[:, 0], points[:, 1],
               color=colors[true_label], alpha=alpha, linewidth=linewidth, linestyle=linestyle)

        # Mark start and end points
        ax.scatter(points[0, 0], points[0, 1], color=colors[true_label], s=80,
                  marker='o', edgecolors='black', linewidth=1, alpha=0.9)
        ax.scatter(points[-1, 0], points[-1, 1], color=colors[true_label], s=120,
                  marker='*' if is_correct else 'X', edgecolors='black', linewidth=1, alpha=1.0)

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("First Principal Component", fontsize=12)
    ax.set_ylabel("Second Principal Component", fontsize=12)

    # Create legend
    legend_elements = []

    # Add digit color legend
    for i in range(10):
        legend_elements.append(plt.Line2D([0], [0], color=colors[i], lw=3, label=f'Digit {i}'))

    # Add hit/miss legend
    legend_elements.append(plt.Line2D([0], [0], color='black', lw=2, linestyle='-', label='Correct'))
    legend_elements.append(plt.Line2D([0], [0], color='black', lw=2, linestyle=':', label='Incorrect'))

    ax.legend(handles=legend_elements, loc='best', title="Legend", ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print analysis
    correct_count = sum(1 for traj in trajectories_2d if traj['true_label'] == traj['predicted_label'])

    print(f"\nCoconut Reasoning Analysis:")
    print(f"Total images: {len(trajectories_2d)}")
    print(f"Ground truth labels: {[traj['true_label'] for traj in trajectories_2d]}")
    print(f"Predicted labels:   {[traj['predicted_label'] for traj in trajectories_2d]}")
    print(f"Correct predictions: {correct_count}/{len(trajectories_2d)} ({correct_count/len(trajectories_2d)*100:.1f}%)")
    print(f"Solid lines = correct predictions, Dotted lines = incorrect predictions")
    print(f"Plot saved to: {save_path}")

    # Print per-image details
    print(f"\nPer-image breakdown:")
    for traj in trajectories_2d:
        status = "✓" if traj['true_label'] == traj['predicted_label'] else "✗"
        print(f"  Image #{traj['image_idx']}: GT={traj['true_label']}, Pred={traj['predicted_label']} {status}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Coconut's autoregressive reasoning process")
    parser.add_argument("--model_path", default="mnist_coconut_checkpoint",
                       help="Path to trained Coconut model")
    parser.add_argument("--data_path", default="data", help="Path to MNIST data")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to analyze")
    parser.add_argument("--max_reasoning_steps", type=int, default=10, help="Max coconut reasoning steps")
    parser.add_argument("--reduction_method", choices=['pca', 'tsne'], default='pca',
                       help="Dimensionality reduction method")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_path", default="coconut_reasoning.png", help="Output plot path")

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Check model files
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        print("Please train the base Vision Transformer first using train_mnist_simple.py")
        return

    # Load data
    print("Loading MNIST test data...")
    _, _, test_images, test_labels = load_mnist_data(args.data_path)

    # Select diverse images
    print(f"Selecting {args.num_images} diverse images...")
    selected_indices = []

    # Try to get one image per digit
    for digit in range(min(10, args.num_images)):
        digit_indices = torch.where(test_labels == digit)[0]
        if len(digit_indices) > 0:
            idx = random.choice(digit_indices.tolist())
            selected_indices.append(idx)

    # Fill remaining slots randomly
    while len(selected_indices) < args.num_images:
        idx = random.randint(0, len(test_images) - 1)
        if idx not in selected_indices:
            selected_indices.append(idx)

    selected_images = test_images[selected_indices]
    selected_labels = test_labels[selected_indices]

    print(f"Selected images with labels: {selected_labels.tolist()}")

    # Load coconut model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # First try to load a trained coconut model
        if os.path.exists(os.path.join(args.model_path, "coconut_config.pth")):
            print("Loading trained Coconut model...")
            coconut_model, config = load_trained_coconut_model(args.model_path, device=device)
            print(f"Trained Coconut model loaded successfully on {device}")
        else:
            print("Loading Coconut model from pretrained Vision Transformer...")
            coconut_model, config = create_mnist_coconut_from_pretrained(args.model_path, device=device)
            print(f"Coconut model created from pretrained ViT on {device}")

        coconut_wrapper = CoconutWithTrajectoryCapture(coconut_model)
    except Exception as e:
        print(f"Error loading coconut model: {e}")
        return

    # Extract reasoning trajectories
    print("Extracting coconut autoregressive reasoning trajectories...")
    selected_images = selected_images.to(device)
    trajectories = extract_coconut_trajectories(
        coconut_wrapper, selected_images, selected_labels, args.max_reasoning_steps
    )

    # Reduce to 2D
    print(f"Reducing dimensionality using {args.reduction_method.upper()}...")
    trajectories_2d = reduce_trajectories_to_2d(trajectories, method=args.reduction_method)

    # Create visualization
    print("Creating coconut reasoning visualization...")
    plot_coconut_reasoning_trajectories(trajectories_2d, save_path=args.save_path)


if __name__ == "__main__":
    main()