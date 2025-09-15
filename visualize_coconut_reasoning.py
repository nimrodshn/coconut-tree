import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde, wasserstein_distance
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.linalg import svd
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


def estimate_intrinsic_dimensionality(points, k_neighbors=5, method='mle'):
    """
    Estimate intrinsic dimensionality of point cloud using various methods

    Args:
        points: np.array of shape (n_points, n_features)
        k_neighbors: number of neighbors for local analysis
        method: 'mle' (Maximum Likelihood), 'pca_ratio', or 'correlation'

    Returns:
        estimated intrinsic dimension
    """
    n_points, n_features = points.shape

    if method == 'mle':
        # Levina-Bickel MLE estimator
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(points)
        distances, indices = nbrs.kneighbors(points)

        # Remove self-distance (first column)
        distances = distances[:, 1:]

        # MLE estimation
        dims = []
        for i in range(n_points):
            r = distances[i]
            if np.min(r) > 0:  # Avoid log(0)
                # MLE formula: d = (k-1) / sum(log(r_k / r_j)) for j < k
                log_ratios = np.log(r[-1] / r[:-1])
                if np.sum(log_ratios) > 0:
                    dim_est = (k_neighbors - 1) / np.sum(log_ratios)
                    dims.append(max(1, min(dim_est, n_features)))

        return np.mean(dims) if dims else n_features

    elif method == 'pca_ratio':
        # PCA-based: count components explaining 95% variance
        pca = PCA()
        pca.fit(points)
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = np.argmax(cumsum_variance >= 0.95) + 1
        return min(intrinsic_dim, n_features)

    elif method == 'correlation':
        # Correlation dimension method
        pca = PCA()
        pca.fit(points)
        eigenvals = pca.explained_variance_
        # Participation ratio
        part_ratio = (np.sum(eigenvals) ** 2) / np.sum(eigenvals ** 2)
        return min(part_ratio, n_features)

    return n_features


def test_local_linearity(points, k_neighbors=10, threshold=0.1):
    """
    Test if points lie on a locally linear manifold

    Args:
        points: np.array of shape (n_points, n_features)
        k_neighbors: number of neighbors for local analysis
        threshold: threshold for linearity (reconstruction error)

    Returns:
        dict with linearity metrics
    """
    n_points = points.shape[0]

    # Use LLE reconstruction weights as linearity test
    try:
        lle = LocallyLinearEmbedding(n_neighbors=k_neighbors, n_components=2,
                                   reg=1e-3, method='standard')
        lle.fit(points)

        # Reconstruction error indicates local linearity
        reconstruction_errors = []
        nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(points)

        for i in range(min(n_points, 100)):  # Sample for efficiency
            distances, indices = nbrs.kneighbors([points[i]])
            neighbor_indices = indices[0][1:]  # Exclude self

            if len(neighbor_indices) >= 3:
                neighbors = points[neighbor_indices]

                # Solve for reconstruction weights
                gram = np.dot(neighbors, neighbors.T)
                gram += 1e-3 * np.eye(len(neighbors))  # Regularization

                try:
                    weights = np.linalg.solve(gram, np.ones(len(neighbors)))
                    weights = weights / np.sum(weights)

                    # Reconstruction error
                    reconstruction = np.dot(weights, neighbors)
                    error = np.linalg.norm(points[i] - reconstruction)
                    reconstruction_errors.append(error)
                except:
                    continue

        mean_error = np.mean(reconstruction_errors) if reconstruction_errors else float('inf')

        return {
            'mean_reconstruction_error': mean_error,
            'is_locally_linear': mean_error < threshold,
            'linearity_score': max(0, 1 - mean_error / threshold)
        }

    except Exception as e:
        return {
            'mean_reconstruction_error': float('inf'),
            'is_locally_linear': False,
            'linearity_score': 0.0,
            'error': str(e)
        }


def estimate_curvature(points, k_neighbors=5):
    """
    Estimate local curvature of trajectory points

    Args:
        points: np.array of shape (n_points, n_features)
        k_neighbors: number of neighbors for curvature estimation

    Returns:
        dict with curvature metrics
    """
    n_points = points.shape[0]

    if n_points < 3:
        return {'mean_curvature': 0, 'max_curvature': 0, 'curvature_values': []}

    # Use PCA on local neighborhoods to estimate curvature
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, n_points)).fit(points)
    curvatures = []

    for i in range(n_points):
        distances, indices = nbrs.kneighbors([points[i]])
        neighbor_indices = indices[0]

        if len(neighbor_indices) >= 3:
            local_points = points[neighbor_indices]

            # Center the points
            centered = local_points - np.mean(local_points, axis=0)

            # SVD to find principal directions
            try:
                U, s, Vt = svd(centered, full_matrices=False)

                # Curvature estimate: ratio of smallest to largest singular values
                if len(s) >= 2 and s[0] > 1e-10:
                    curvature = s[-1] / s[0]  # Condition number inverse
                    curvatures.append(curvature)
            except:
                continue

    if curvatures:
        return {
            'mean_curvature': np.mean(curvatures),
            'max_curvature': np.max(curvatures),
            'std_curvature': np.std(curvatures),
            'curvature_values': curvatures
        }
    else:
        return {'mean_curvature': 0, 'max_curvature': 0, 'std_curvature': 0, 'curvature_values': []}


def analyze_submanifold_structure(trajectories, verbose=True):
    """
    Comprehensive analysis of whether reasoning steps lie on a submanifold

    Args:
        trajectories: list of trajectory dictionaries with 'embeddings' key
        verbose: whether to print detailed analysis

    Returns:
        dict with comprehensive submanifold analysis
    """
    # Collect all embeddings
    all_embeddings = []
    trajectory_embeddings = []

    for traj in trajectories:
        traj_embeds = np.array(traj['embeddings'])
        all_embeddings.extend(traj_embeds)
        trajectory_embeddings.append(traj_embeds)

    all_embeddings = np.array(all_embeddings)
    n_points, n_features = all_embeddings.shape

    if verbose:
        print(f"\nSubmanifold Analysis:")
        print(f"Total points: {n_points}")
        print(f"Ambient dimension: {n_features}")

    # 1. Intrinsic dimensionality estimation
    intrinsic_dims = {}
    for method in ['mle', 'pca_ratio', 'correlation']:
        try:
            dim = estimate_intrinsic_dimensionality(all_embeddings, method=method)
            intrinsic_dims[method] = dim
        except Exception as e:
            intrinsic_dims[method] = None
            if verbose:
                print(f"Warning: {method} estimation failed: {e}")

    # 2. Local linearity test
    linearity_result = test_local_linearity(all_embeddings)

    # 3. Curvature analysis for individual trajectories
    trajectory_curvatures = []
    for i, traj_embeds in enumerate(trajectory_embeddings):
        if len(traj_embeds) >= 3:
            curv = estimate_curvature(traj_embeds)
            trajectory_curvatures.append(curv)
        else:
            trajectory_curvatures.append({'mean_curvature': 0, 'max_curvature': 0})

    # 4. Overall curvature analysis
    overall_curvature = estimate_curvature(all_embeddings)

    # 5. Variance analysis along principal components
    pca = PCA()
    pca.fit(all_embeddings)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Find effective dimensionality (95% variance)
    effective_dim = np.argmax(cumulative_variance >= 0.95) + 1

    results = {
        'intrinsic_dimensions': intrinsic_dims,
        'linearity': linearity_result,
        'overall_curvature': overall_curvature,
        'trajectory_curvatures': trajectory_curvatures,
        'pca_analysis': {
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'effective_dimension': effective_dim,
            'first_10_components': explained_variance_ratio[:10].tolist()
        },
        'geometry_summary': {
            'ambient_dimension': n_features,
            'total_points': n_points,
            'is_likely_submanifold': effective_dim < n_features * 0.5,
            'is_locally_linear': linearity_result.get('is_locally_linear', False),
            'mean_trajectory_curvature': np.mean([tc.get('mean_curvature', 0) for tc in trajectory_curvatures])
        }
    }

    if verbose:
        print(f"\nIntrinsic Dimensionality Estimates:")
        for method, dim in intrinsic_dims.items():
            if dim is not None:
                print(f"  {method.upper()}: {dim:.2f}")

        print(f"\nPCA Analysis:")
        print(f"  Effective dimension (95% variance): {effective_dim}")
        print(f"  First 5 components explain: {cumulative_variance[4]:.3f} of variance")
        print(f"  First 10 components: {explained_variance_ratio[:10]}")

        print(f"\nLocal Linearity:")
        print(f"  Reconstruction error: {linearity_result.get('mean_reconstruction_error', 'N/A'):.4f}")
        print(f"  Is locally linear: {linearity_result.get('is_locally_linear', False)}")
        print(f"  Linearity score: {linearity_result.get('linearity_score', 0):.3f}")

        print(f"\nCurvature Analysis:")
        print(f"  Overall mean curvature: {overall_curvature.get('mean_curvature', 0):.4f}")
        print(f"  Mean trajectory curvature: {results['geometry_summary']['mean_trajectory_curvature']:.4f}")

        print(f"\nSubmanifold Assessment:")
        summary = results['geometry_summary']
        print(f"  Likely on submanifold: {summary['is_likely_submanifold']}")
        print(f"  Locally linear: {summary['is_locally_linear']}")
        print(f"  Effective dim / Ambient dim: {effective_dim}/{n_features} = {effective_dim/n_features:.3f}")

    return results


def analyze_label_specific_submanifolds(trajectories, verbose=True):
    """
    Analyze whether different labels occupy different submanifolds

    Args:
        trajectories: list of trajectory dictionaries with 'embeddings' and 'true_label'
        verbose: whether to print detailed analysis

    Returns:
        dict with per-label submanifold analysis and separation metrics
    """
    # Group trajectories by label
    label_groups = {}
    for traj in trajectories:
        label = traj['true_label']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(traj)

    available_labels = sorted(label_groups.keys())

    if verbose:
        print(f"\nLabel-Specific Submanifold Analysis:")
        print(f"Available labels: {available_labels}")
        print(f"Trajectories per label: {[len(label_groups[label]) for label in available_labels]}")

    # Analyze each label's submanifold separately
    label_analyses = {}
    label_embeddings = {}

    for label in available_labels:
        label_trajs = label_groups[label]

        # Collect embeddings for this label
        label_embeds = []
        for traj in label_trajs:
            label_embeds.extend(traj['embeddings'])

        if len(label_embeds) >= 3:  # Need minimum points for analysis
            label_embeds = np.array(label_embeds)
            label_embeddings[label] = label_embeds

            # Analyze this label's submanifold
            analysis = {}

            # Intrinsic dimensionality
            try:
                analysis['intrinsic_dim_mle'] = estimate_intrinsic_dimensionality(label_embeds, method='mle')
                analysis['intrinsic_dim_pca'] = estimate_intrinsic_dimensionality(label_embeds, method='pca_ratio')
            except:
                analysis['intrinsic_dim_mle'] = None
                analysis['intrinsic_dim_pca'] = None

            # Local linearity
            analysis['linearity'] = test_local_linearity(label_embeds)

            # Curvature
            analysis['curvature'] = estimate_curvature(label_embeds)

            # PCA analysis
            if len(label_embeds) >= 2:
                pca = PCA()
                pca.fit(label_embeds)
                explained_var = pca.explained_variance_ratio_
                cumulative_var = np.cumsum(explained_var)
                effective_dim = np.argmax(cumulative_var >= 0.95) + 1

                analysis['pca'] = {
                    'explained_variance_ratio': explained_var,
                    'effective_dimension': effective_dim,
                    'first_3_components_var': cumulative_var[2] if len(cumulative_var) > 2 else cumulative_var[-1]
                }

            label_analyses[label] = analysis

            if verbose:
                print(f"\nLabel {label} ({len(label_trajs)} trajectories, {len(label_embeds)} points):")
                if analysis['intrinsic_dim_mle'] is not None:
                    print(f"  Intrinsic dim (MLE): {analysis['intrinsic_dim_mle']:.2f}")
                if analysis['intrinsic_dim_pca'] is not None:
                    print(f"  Intrinsic dim (PCA): {analysis['intrinsic_dim_pca']:.2f}")
                if 'pca' in analysis:
                    print(f"  Effective dim (95% var): {analysis['pca']['effective_dimension']}")
                    print(f"  First 3 components: {analysis['pca']['first_3_components_var']:.3f} variance")
                print(f"  Local linearity: {analysis['linearity'].get('is_locally_linear', False)}")
                print(f"  Mean curvature: {analysis['curvature'].get('mean_curvature', 0):.4f}")

    # Analyze separation between labels
    separation_metrics = analyze_manifold_separation(label_embeddings, verbose=verbose)

    return {
        'label_analyses': label_analyses,
        'separation_metrics': separation_metrics,
        'available_labels': available_labels,
        'label_counts': {label: len(label_groups[label]) for label in available_labels}
    }


def analyze_manifold_separation(label_embeddings, verbose=True):
    """
    Analyze how well separated different labels' manifolds are

    Args:
        label_embeddings: dict mapping labels to their embedding arrays
        verbose: whether to print analysis

    Returns:
        dict with separation metrics
    """
    labels = list(label_embeddings.keys())
    n_labels = len(labels)

    if n_labels < 2:
        return {'insufficient_labels': True}

    # Compute pairwise distances between label manifolds
    pairwise_distances = {}
    hausdorff_distances = {}
    wasserstein_distances = {}
    pca_angle_differences = {}

    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            if i < j:  # Only compute upper triangle
                emb1 = label_embeddings[label1]
                emb2 = label_embeddings[label2]

                # 1. Average minimum distance (manifold proximity)
                dist_matrix = cdist(emb1, emb2)
                min_dists_1_to_2 = np.min(dist_matrix, axis=1)
                min_dists_2_to_1 = np.min(dist_matrix, axis=0)
                avg_min_dist = (np.mean(min_dists_1_to_2) + np.mean(min_dists_2_to_1)) / 2
                pairwise_distances[(label1, label2)] = avg_min_dist

                # 2. Hausdorff distance (maximum separation)
                hausdorff_dist = max(np.max(min_dists_1_to_2), np.max(min_dists_2_to_1))
                hausdorff_distances[(label1, label2)] = hausdorff_dist

                # 3. Wasserstein distance along first principal component
                try:
                    # Project both sets onto their first PCA component
                    pca1 = PCA(n_components=1)
                    pca2 = PCA(n_components=1)
                    proj1 = pca1.fit_transform(emb1).flatten()
                    proj2 = pca2.fit_transform(emb2).flatten()
                    wass_dist = wasserstein_distance(proj1, proj2)
                    wasserstein_distances[(label1, label2)] = wass_dist
                except:
                    wasserstein_distances[(label1, label2)] = np.nan

                # 4. Principal component angle difference
                try:
                    pca1 = PCA()
                    pca2 = PCA()
                    pca1.fit(emb1)
                    pca2.fit(emb2)

                    # Compute angle between first principal components
                    pc1 = pca1.components_[0]
                    pc2 = pca2.components_[0]
                    cos_angle = np.abs(np.dot(pc1, pc2)) / (np.linalg.norm(pc1) * np.linalg.norm(pc2))
                    angle_deg = np.arccos(np.clip(cos_angle, 0, 1)) * 180 / np.pi
                    pca_angle_differences[(label1, label2)] = angle_deg
                except:
                    pca_angle_differences[(label1, label2)] = np.nan

    # Compute overall separation metrics
    avg_pairwise_dist = np.mean(list(pairwise_distances.values()))
    avg_hausdorff_dist = np.mean(list(hausdorff_distances.values()))
    avg_wasserstein_dist = np.nanmean(list(wasserstein_distances.values()))
    avg_pca_angle = np.nanmean(list(pca_angle_differences.values()))

    # Separation quality assessment
    separation_score = avg_pairwise_dist  # Simple metric - could be more sophisticated

    results = {
        'pairwise_distances': pairwise_distances,
        'hausdorff_distances': hausdorff_distances,
        'wasserstein_distances': wasserstein_distances,
        'pca_angle_differences': pca_angle_differences,
        'summary': {
            'avg_pairwise_distance': avg_pairwise_dist,
            'avg_hausdorff_distance': avg_hausdorff_dist,
            'avg_wasserstein_distance': avg_wasserstein_dist,
            'avg_pca_angle_difference': avg_pca_angle,
            'separation_score': separation_score,
            'well_separated': separation_score > 1.0  # Threshold-based assessment
        }
    }

    if verbose:
        print(f"\nManifold Separation Analysis:")
        print(f"Average pairwise distance: {avg_pairwise_dist:.4f}")
        print(f"Average Hausdorff distance: {avg_hausdorff_dist:.4f}")
        print(f"Average Wasserstein distance: {avg_wasserstein_dist:.4f}")
        print(f"Average PCA angle difference: {avg_pca_angle:.2f}°")
        print(f"Well separated: {results['summary']['well_separated']}")

        print(f"\nPairwise label distances:")
        for (l1, l2), dist in pairwise_distances.items():
            angle = pca_angle_differences.get((l1, l2), np.nan)
            print(f"  Labels {l1}-{l2}: dist={dist:.3f}, angle={angle:.1f}°")

    return results


def plot_label_manifold_visualization(trajectories_2d, submanifold_analysis, save_path="label_manifolds.png"):
    """
    Enhanced visualization showing label-specific manifold structure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("COCONUT Label-Specific Manifold Analysis", fontsize=16, fontweight='bold')

    # Group trajectories by label
    label_groups = {}
    for traj in trajectories_2d:
        label = traj['true_label']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(traj)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Plot 1: All trajectories with label separation highlighted
    ax1 = axes[0, 0]
    ax1.set_title("Reasoning Trajectories by Label")

    for label, trajs in label_groups.items():
        for traj in trajs:
            points = traj['points_2d']
            is_correct = traj['true_label'] == traj['predicted_label']
            alpha = 0.8 if is_correct else 0.4
            ax1.plot(points[:, 0], points[:, 1], color=colors[label], alpha=alpha, linewidth=1.5)
            ax1.scatter(points[0, 0], points[0, 1], color=colors[label], s=50, alpha=0.8)

    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")

    # Plot 2: Label centroids and spread
    ax2 = axes[0, 1]
    ax2.set_title("Label Centroids and Dispersion")

    for label, trajs in label_groups.items():
        all_points = np.vstack([traj['points_2d'] for traj in trajs])
        centroid = np.mean(all_points, axis=0)

        # Plot all points for this label
        ax2.scatter(all_points[:, 0], all_points[:, 1], color=colors[label], alpha=0.3, s=20)

        # Plot centroid
        ax2.scatter(centroid[0], centroid[1], color=colors[label], s=200, marker='*',
                   edgecolors='black', linewidth=2, label=f'Label {label}')

        # Plot dispersion ellipse
        try:
            cov = np.cov(all_points.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1]))
            width, height = 2 * np.sqrt(eigenvals)
            ellipse = plt.matplotlib.patches.Ellipse(centroid, width, height, angle=angle,
                                                   fill=False, color=colors[label], linewidth=2)
            ax2.add_patch(ellipse)
        except:
            pass

    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.legend()

    # Plot 3: Dimensionality comparison
    ax3 = axes[1, 0]
    ax3.set_title("Intrinsic Dimensionality by Label")

    if 'label_analyses' in submanifold_analysis:
        labels_list = []
        mle_dims = []
        pca_dims = []

        for label, analysis in submanifold_analysis['label_analyses'].items():
            labels_list.append(label)
            mle_dims.append(analysis.get('intrinsic_dim_mle', 0) or 0)
            pca_dims.append(analysis.get('intrinsic_dim_pca', 0) or 0)

        x = np.arange(len(labels_list))
        width = 0.35

        ax3.bar(x - width/2, mle_dims, width, label='MLE Estimate', alpha=0.8)
        ax3.bar(x + width/2, pca_dims, width, label='PCA Estimate', alpha=0.8)

        ax3.set_xlabel('Label')
        ax3.set_ylabel('Estimated Intrinsic Dimension')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels_list)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Separation metrics heatmap
    ax4 = axes[1, 1]
    ax4.set_title("Pairwise Label Separation")

    if 'separation_metrics' in submanifold_analysis and 'pairwise_distances' in submanifold_analysis['separation_metrics']:
        all_labels = sorted(label_groups.keys())
        n_labels = len(all_labels)

        if n_labels > 1:
            # Create distance matrix
            dist_matrix = np.zeros((n_labels, n_labels))

            for i, label1 in enumerate(all_labels):
                for j, label2 in enumerate(all_labels):
                    if i == j:
                        dist_matrix[i, j] = 0
                    else:
                        # Find distance in either direction
                        key1 = (label1, label2) if (label1, label2) in submanifold_analysis['separation_metrics']['pairwise_distances'] else (label2, label1)
                        if key1 in submanifold_analysis['separation_metrics']['pairwise_distances']:
                            dist_matrix[i, j] = submanifold_analysis['separation_metrics']['pairwise_distances'][key1]

            im = ax4.imshow(dist_matrix, cmap='viridis')
            ax4.set_xticks(range(n_labels))
            ax4.set_yticks(range(n_labels))
            ax4.set_xticklabels(all_labels)
            ax4.set_yticklabels(all_labels)

            # Add text annotations
            for i in range(n_labels):
                for j in range(n_labels):
                    text = ax4.text(j, i, f'{dist_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="w")

            plt.colorbar(im, ax=ax4, label='Average Distance')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return fig


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
    parser.add_argument("--analyze_submanifold", action='store_true',
                       help="Perform detailed submanifold analysis")

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

    # Analyze submanifold structure (optional)
    label_analysis = None
    if args.analyze_submanifold:
        print("Analyzing submanifold structure of reasoning trajectories...")
        analyze_submanifold_structure(trajectories, verbose=True)

        print("Analyzing label-specific submanifolds...")
        label_analysis = analyze_label_specific_submanifolds(trajectories, verbose=True)

    # Reduce to 2D
    print(f"Reducing dimensionality using {args.reduction_method.upper()}...")
    trajectories_2d = reduce_trajectories_to_2d(trajectories, method=args.reduction_method)

    # Create visualization
    print("Creating coconut reasoning visualization...")
    plot_coconut_reasoning_trajectories(trajectories_2d, save_path=args.save_path)

    # Create label-specific manifold visualization if analysis was performed
    if label_analysis is not None:
        print("Creating label-specific manifold visualization...")
        label_viz_path = args.save_path.replace('.png', '_label_manifolds.png')
        plot_label_manifold_visualization(trajectories_2d, label_analysis, save_path=label_viz_path)



if __name__ == "__main__":
    main()