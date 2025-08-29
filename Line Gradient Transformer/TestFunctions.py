# TestFunctions.py
import numpy as np
import matplotlib.pyplot as plt
import torch
from itertools import combinations
import cv2
from math import atan2, degrees
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import torch
import random
from itertools import combinations
from ViT_PatchTokenTransformerEmbeding import *

# --- Drawing and plotting utilities --- #
def draw_line_cells(A, B, grid_size=8):
    """
    Returns a grid_size x grid_size array with 1s where the line from A to B passes.
    """
    img = np.zeros((grid_size, grid_size), dtype=np.uint8)
    cv2.line(img, A, B, color=1, thickness=1)
    return img


def plot_line_critical_with_class(A, B, model, grid_size=8, num_classes=16, points=None):
    """
    Visualizes the pixels in a grid-based line image whose flipping changes the 
    predicted class of a given model. 

    Specifically, the function:
    1. Draws a line between points A and B on a grid of size `grid_size` x `grid_size`.
       Alternatively, a custom set of points can be provided.
    2. Computes the true class of the line using `compute_angle_class`.
    3. Passes the image through the model to get the original predicted class.
    4. Iteratively flips each pixel (0 ↔ 1) and checks if the model's prediction changes.
    5. Marks in red the new predicted class for any pixel whose flipping changes the output.
    6. Plots the grid with the original line and the critical pixels annotated.

    Args:
        A (tuple): Coordinates of the first endpoint of the line.
        B (tuple): Coordinates of the second endpoint of the line.
        model (torch.nn.Module): The trained PyTorch model for classification.
        grid_size (int): Size of the square grid (default 8).
        num_classes (int): Number of possible classes (default 16).
        points (list of tuples, optional): Custom points to draw instead of using A and B.

    Returns:
        None. Displays a matplotlib figure showing the line and critical pixels.
    """
    device = next(model.parameters()).device
    model = model.to(device)
    model.eval()

    if points is None:
        img = draw_line_cells(A, B, grid_size)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        true_class = compute_angle_class(A, B, num_classes)
    else:
        img = np.zeros((grid_size, grid_size), dtype=np.uint8)
        for point in points:
            img[point[1], point[0]] = 255
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        true_class = compute_angle_class(points[0], points[1], num_classes)

    with torch.no_grad():
        out = model(img_tensor)
        if isinstance(out, tuple):
            out = out[0]
        pred_orig = out.argmax(dim=1).item()

    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='Blues', origin='upper', extent=[0, grid_size, 0, grid_size])

    for i in range(grid_size):
        for j in range(grid_size):
            img_tensor[0,0,i,j] = 1 - img_tensor[0,0,i,j]
            with torch.no_grad():
                out = model(img_tensor)
                if isinstance(out, tuple):
                    out = out[0]
                pred_new = out.argmax(dim=1).item()
            img_tensor[0,0,i,j] = 1 - img_tensor[0,0,i,j]

            if pred_new != pred_orig:
                plt.text(j + 0.5, i + 0.5, str(pred_new),
                         color='red', ha='center', va='center', fontsize=12, fontweight='bold')

    for x in range(grid_size + 1):
        plt.axhline(x, color='gray', linewidth=0.5)
        plt.axvline(x, color='gray', linewidth=0.5)

    plt.xticks([])
    plt.yticks([])
    plt.title(f"Line A={A} B={B} | True class={true_class} | Red numbers = new predicted class if pixel flipped")
    plt.show()


# --- Robustness testing --- #
def robustness_score_exclude_AB_random(model, image_size=8, num_classes=16, num_samples=20):
    """
    Compute fraction of all lines (all pairs of points) for which flipping
    any single random pixel (except the two line points A and B) does NOT change the model prediction.
    """
    device = next(model.parameters()).device
    model = model.to(device)
    model.eval()

    coords = [(x, y) for x in range(image_size) for y in range(image_size)]
    pairs = list(combinations(coords, 2))

    safe_lines = 0
    total_lines = 0

    with torch.no_grad():
        for A, B in pairs:
            # Create line image
            img_tensor = draw_line_image(A, B, image_size=image_size).unsqueeze(0).to(device)

            # Original prediction
            out = model(img_tensor)
            if isinstance(out, tuple):
                out = out[0]
            pred_orig = out.argmax(dim=1).item()

            if pred_orig != compute_angle_class(A, B, num_classes):
                continue
            total_lines += 1
            # Get all pixels excluding A and B
            all_pixels = [(i, j) for i in range(image_size) for j in range(image_size) if (j, i) != A and (j, i) != B]

            # Sample random pixels
            sampled_pixels = random.sample(all_pixels, min(num_samples, len(all_pixels)))

            changed = False
            for i, j in sampled_pixels:
                # Flip pixel
                if (i,j) == A or (i,j) == B:
                    continue
                img_tensor[0, 0, i, j] = 1 - img_tensor[0, 0, i, j]

                # Prediction after flip
                out_new = model(img_tensor)
                if isinstance(out_new, tuple):
                    out_new = out_new[0]
                pred_new = out_new.argmax(dim=1).item()

                # Restore pixel
                img_tensor[0, 0, i, j] = 1 - img_tensor[0, 0, i, j]

                if pred_new != pred_orig:
                    changed = True
                    break

            if not changed:
                safe_lines += 1

    total_lines = len(pairs)
    robustness_percentage = 100.0 * safe_lines / total_lines
    print(f"The robustness against random pixel flips (20 per line) is: {robustness_percentage:.4f}%")


def translation_robustness_global_unique(model, grid_size=8, num_classes=16):
    """
    Test all translations for each pair of points without repeating globally.
    Returns overall accuracy and per-pair fraction of correctly predicted translations.
    """
    device = next(model.parameters()).device
    model = model.to(device)
    model.eval()

    coords = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    pairs = list(combinations(coords, 2))

    global_seen_hashes = set()
    total_unique = 0
    correct_unique = 0
    pair_accuracy = {}

    with torch.no_grad():
        for A, B in pairs:
            img_orig = draw_line_image(A, B, grid_size)
            orig_hash = hash(img_orig.tobytes())
            if orig_hash in global_seen_hashes:
                continue

            true_class = compute_angle_class(A, B, num_classes)
            correct_for_pair = 0
            total_for_pair = 0
            local_hashes = set()

            max_dx = grid_size - max(A[0], B[0]) - 1
            min_dx = -min(A[0], B[0])
            max_dy = grid_size - max(A[1], B[1]) - 1
            min_dy = -min(A[1], B[1])

            for dx in range(min_dx, max_dx + 1):
                for dy in range(min_dy, max_dy + 1):
                    A_trans = (A[0] + dx, A[1] + dy)
                    B_trans = (B[0] + dx, B[1] + dy)
                    img = draw_line_image(A_trans, B_trans, grid_size)
                    img_hash = hash(img.tobytes())
                    if img_hash in global_seen_hashes:
                        continue
                    local_hashes.add(img_hash)

                    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    out = model(img_tensor)
                    if isinstance(out, tuple):
                        out = out[0]
                    pred_class = out.argmax(dim=1).item()

                    if pred_class == true_class:
                        correct_for_pair += 1
                        correct_unique += 1
                    total_for_pair += 1
                    total_unique += 1

            global_seen_hashes.update(local_hashes)
            if total_for_pair > 0:
                pair_accuracy[(A, B)] = correct_for_pair / total_for_pair

    overall_accuracy = correct_unique / total_unique if total_unique > 0 else 1.0
    return overall_accuracy, pair_accuracy


# --- Visualization utilities of embeding space --- #
def visualize_line_embeddings(model, grid_size=16, num_angle_classes=16):
    """
    Generates all line images on a grid, computes angle classes, passes through a model,
    performs PCA, and visualizes embeddings in an interactive scatter plot.

    Args:
        model: torch.nn.Module, model that takes images_tensor [B,1,H,W] and returns embeddings [B,D]
        grid_size: int, size of the 2D grid
        num_angle_classes: int, number of discrete angle classes
    """

    def compute_angle_class(p1, p2, num_classes=16):
        """Compute a discrete angle class between two points."""
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        angle = degrees(atan2(dy, dx)) % 360
        cls = int(angle / (360 / num_classes))
        return cls

    # --- Generate points and images ---
    points = [(x, y) for y in range(grid_size) for x in range(grid_size)]
    images, angle_classes, pairs = [], [], []

    for i in range(len(points)):
        for j in range(i+1, len(points)):
            # Draw line image
            img = np.zeros((grid_size, grid_size), dtype=np.uint8)
            cv2.line(img, points[i], points[j], color=255, thickness=1)
            images.append(img.astype(np.float32) / 255.0)

            # Angle class
            angle_classes.append(compute_angle_class(points[i], points[j], num_angle_classes))
            pairs.append((points[i], points[j]))

    images = np.stack(images)
    images_tensor = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # [N,1,H,W]

    # --- Pass through model ---
    model.eval()
    with torch.no_grad():
        out, _ = model(images_tensor)  # assuming model returns (embedding, something)
    out_np = out.detach().cpu().numpy()

    # --- PCA ---
    pca = PCA(n_components=2)
    out_2d = pca.fit_transform(out_np)

    # --- Prepare DataFrame for interactive plot ---
    labels = [f"A: {A}, B: {B}" for (A, B) in pairs]
    df = pd.DataFrame({
        "PC1": out_2d[:,0],
        "PC2": out_2d[:,1],
        "AngleClass": angle_classes,
        "Label": labels
    })

    # --- Plot ---
    fig = px.scatter(
        df, x="PC1", y="PC2", color="AngleClass",
        hover_data=["Label"], color_continuous_scale=px.colors.qualitative.Plotly
    )
    fig.update_layout(
        title="PCA of Lines Colored by Angle Class",
        xaxis_title="PC1",
        yaxis_title="PC2",
        width=700, height=700
    )
    fig.show()

# --- Saliency map utilities --- #
def generate_saliency_map(model, img, label):

    """
    Generate a saliency map for the given image and model's output.
    """
    model.eval()
    img.requires_grad_()  # Ensure gradients are tracked

    # Perform a forward pass
    output, _ = model(img.unsqueeze(0))
    print('prediction')
    print(output.argmax().item())
    print(output)
    # Get the class score/logit for the correct label
    output_class = output[0, label]

    # Compute gradients with respect to the input image
    model.zero_grad()  # Clear any previous gradients
    output_class.backward()  # Backpropagate to compute gradients

    # Get the absolute value of the gradients (saliency map)
    saliency_map = img.grad.abs()

    # Convert the saliency map to numpy and return
    saliency_map = saliency_map.squeeze().cpu().detach().numpy()
    return saliency_map


def flip_pixel(img_01, pix):
    """Flip exactly one pixel (0<->1) in a binary image [1,H,W]."""
    out = img_01.clone()
    out[0, pix[1], pix[0]] = 1.0 - out[0, pix[1], pix[0]]
    return out


def saliency_plot(model, A, B, image_size=16, flip = []):
    """
    Generates and visualizes a saliency map for a line image, highlighting which 
    pixels most influence the model's prediction.

    The function performs the following steps:
    1. Draws a binary line image between points A and B on a grid of size `image_size`.
       Optional pixels in `flip` can be toggled (0 ↔ 1) before analysis.
    2. Computes the true class of the line using `compute_angle_class`.
    3. Generates a saliency map using `generate_saliency_map`, indicating pixel-wise 
       importance for the model's prediction.
    4. Plots the saliency map using a heatmap, overlaying the original line pixels 
       as white dots for reference.
    5. Prints the line angle and its corresponding discretized class (for debugging/inspection).

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        A (tuple): Coordinates of the first endpoint of the line.
        B (tuple): Coordinates of the second endpoint of the line.
        image_size (int): Size of the square image (default 16).
        flip (list of tuples): Optional list of pixel coordinates to flip before generating 
                               the saliency map.

    Returns:
        None. Displays a matplotlib figure with the saliency map overlaid with line pixels.
    """
    base = draw_line_cells(A, B, image_size=image_size)  # [1, H, W]
    for pix in flip:
        base = flip_pixel(base, pix)
    true_cls = compute_angle_class(A, B, 16)
    saliency_map = generate_saliency_map(model, base, true_cls)

    # Squeeze the base image and saliency map to remove the batch dimension
    base = base.squeeze()  # Shape [H, W]
    saliency_map = saliency_map  # Shape [H, W]

    dx, dy = B[0] - A[0], B[1] - A[1]
    angle = degrees(atan2(dy, dx)) % 180
    print(angle)
    print('Real')
    print( int(angle // (180 / 16)))

    # Create the figure
    plt.figure(figsize=(7, 5))

    # Plot the saliency map
    plt.imshow(saliency_map, cmap='hot_r', alpha=1.0)  # Using 'hot_r' for reverse colormap with transparency
    plt.title("Saliency Map with Binary Image Overlay")
    # Add color bar for saliency map
    plt.colorbar()

    # Find the coordinates where the binary image has value 1
    coords = np.argwhere(base == 1)
    # Overlay dots at the locations where the binary image has value 1
    for i in range(coords.shape[1]):
        y, x = coords[0, i], coords[1, i] # Unpack the coordinates
        plt.scatter(x, y, color='white', s=20, edgecolors='black', marker='o')  # Plot dots on the saliency map



    # Hide axes for cleaner view
    plt.axis('off')

    # Display the plot
    plt.tight_layout()
    plt.show()


def saliency_plot_custom(model, image_size=16, points = []):
    """
    Same as the previous function but with custom points instead of A and B.
    Line gradient is calculated between the first two points in the list.

    """
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    for point in points:
        img[point[1], point[0]] = 255
    base = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0  # [1, H, W]
    true_cls = compute_angle_class(points[0], points[1], 16)
    saliency_map = generate_saliency_map(model, base, true_cls)

    # Squeeze the base image and saliency map to remove the batch dimension
    base = base.squeeze()  # Shape [H, W]
    saliency_map = saliency_map  # Shape [H, W]
    A = points[0]
    B = points[1]
    dx, dy = B[0] - A[0], B[1] - A[1]
    angle = degrees(atan2(dy, dx)) % 180
    print(angle)
    print('Real')
    print( int(angle // (180 / 16)))

    # Create the figure
    plt.figure(figsize=(7, 5))

    # Plot the saliency map
    plt.imshow(saliency_map, cmap='hot_r', alpha=1.0)  # Using 'hot_r' for reverse colormap with transparency
    plt.title("Saliency Map with Binary Image Overlay")
    # Add color bar for saliency map
    plt.colorbar()

    # Find the coordinates where the binary image has value 1
    coords = np.argwhere(base == 1)
    # Overlay dots at the locations where the binary image has value 1
    for i in range(coords.shape[1]):
        y, x = coords[0, i], coords[1, i] # Unpack the coordinates
        plt.scatter(x, y, color='white', s=20, edgecolors='black', marker='o')  # Plot dots on the saliency map



    # Hide axes for cleaner view
    plt.axis('off')

    # Display the plot
    plt.tight_layout()
    plt.show()


# --- Attention map utilities --- #
def attention_plot(model, A, B, image_size=16, flip = []):
    """
    Visualizes the attention map of a model for a binary line image.

    Draws a line between points A and B, optionally flips pixels, 
    then extracts and normalizes attention weights from the first layer 
    and first head of the model. Plots the attention map with grid lines 
    for clarity.
    """
    base = draw_line_cells(A, B, image_size=image_size)  # [1, H, W]
    for pix in flip:
        base = flip_pixel(base, pix)
    true_cls = compute_angle_class(A, B, 16)

    # Perform a forward pass through the model to get attention weights
    output, attention_weights = model(base.unsqueeze(0))  # Add batch dimension to the image

    # Extract the attention weights from the first layer and the first head
    attention_map = attention_weights[0][0]  # Shape [num_patches, num_patches] (first layer, first head)

    # Normalize the attention map for better visualization
    attention_map = attention_map.cpu().detach().numpy()
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())  # Normalize

    # Create a figure
    plt.figure(figsize=(8, 8))

    # Plot the saliency map
    plt.imshow(attention_map, cmap='hot_r', alpha=1.0)

    # Draw black grid lines to separate cells
    num_cells = attention_map.shape[0]
    for i in range(1, num_cells):
        plt.axhline(i - 0.5, color='black', linewidth=1)  # horizontal lines
        plt.axvline(i - 0.5, color='black', linewidth=1)  # vertical lines

    plt.colorbar()
    plt.title("Attention Map with Binary Image Overlay")

    # Hide axes ticks but keep the grid visible
    plt.xticks([])
    plt.yticks([])

    # Display the plot
    plt.tight_layout()
    plt.show()


# --- Rank reduction ---
def do_low_rank(weight, k, debug=False, niter=2):
    assert weight.ndim == 2

    max_rank = min(weight.shape[0], weight.shape[1])
    desired_rank = int(max_rank * k)

    if debug:
        print(f"Shape is {weight.shape} and shape is {weight.dtype} => desired rank {desired_rank}")

    results = torch.svd_lowrank(weight,
                                q=desired_rank,
                                niter=niter)
    weight_approx = results[0] @ torch.diag(results[1]) @ results[2].T

    if debug:
        print(f"New matrix has shape {weight_approx.shape}")

    assert weight_approx.shape[0] == weight.shape[0] and weight_approx.shape[1] == weight.shape[1]
    weight_approx = torch.nn.Parameter(weight_approx)

    return weight_approx


def accuracy_for_weight(model, weight_tensor, k, set_weight_fn):
    """Apply low-rank to a weight and test accuracy."""
    low_rank_weight = do_low_rank(weight_tensor.clone(), k)
    set_weight_fn(model, low_rank_weight)
    acc, _ = test_all_pairs(model, image_size=16)
    return acc


def sweep_all_weights(kpercentages):
    """
    For each weight in the model, sweep through kpercentages to do low-rank approximation
    and test accuracy. Plot all results in one figure.

    """
    weight_info = {
        #"pre_MLP": lambda m: m.blocks[0].mlp[0].weight,
        #"pos_MLP": lambda m: m.blocks[0].mlp[2].weight
        #"classifier_W": lambda m: m.classifier.weight,
        "patch_poss_MLP": lambda m: m.patch_embed.transformer.mlp[2].weight,
        "patch_pre_MLP": lambda m: m.patch_embed.transformer.mlp[0].weight
        #"patch_projection": lambda m: m.patch_embed.transformer.proj.weight
    }

    set_weight_fns = {
        #"pre_MLP": lambda m, w: setattr(m.blocks[0].mlp[0], 'weight', w),
        #"pos_MLP": lambda m, w: setattr(m.blocks[0].mlp[2], 'weight', w)
        #"classifier_W": lambda m, w: setattr(m.classifier, 'weight', w),
        "patch_poss_MLP": lambda m, w: setattr(m.patch_embed.transformer.mlp[2], 'weight', w),
        "patch_pre_MLP": lambda m, w: setattr(m.patch_embed.transformer.mlp[0], 'weight', w)
        #"patch_projection": lambda m, w: setattr(m.patch_embed.transformer.proj, 'weight', w)
    }

    accuracies_all = {name: [] for name in weight_info}

    for name in weight_info:
        print(f"==== Sweeping {name} ====")
        # Load fresh model each time
        for k in kpercentages:
            model = VisionTransformer(img_size=16, patch_size=4, embed_dim=256, num_classes=16)
            model.load_state_dict(torch.load("vit_line_angle_model-doubleAtt-nosiy.pth"))
            pos_MLP = model.blocks[0].mlp[2].weight.data.clone()
            pos_MLP_low = do_low_rank(pos_MLP, 0.15)
            model.blocks[0].mlp[2].weight = pos_MLP_low
            weight_tensor = weight_info[name](model).data.clone()
            acc = accuracy_for_weight(model, weight_tensor, k, set_weight_fns[name])
            accuracies_all[name].append(acc)
            print(f"{name} | k={k:.2f} -> acc={acc:.4f}")

    # Plotting
    plt.figure(figsize=(10,6))
    for name, acc_list in accuracies_all.items():
        plt.plot(kpercentages, acc_list, marker='o', label=name)
    plt.xlabel("Rank Reduction Percentage (k)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Low-Rank Approximation for All Weights")
    plt.grid(True)
    plt.legend()
    plt.show()

    return accuracies_all


def visualize_patch_transformer(model, patch_size=4, num_angle_classes=16):
    """
    Generates all line images on a patch, computes angle classes,
    passes through the model's patch transformer, performs PCA, 
    and creates an interactive scatter plot.

    Args:
        model: torch.nn.Module with a patch_embed.transformer attribute
        patch_size: int, size of the patch (H=W)
        num_angle_classes: int, number of discrete angle classes
    """

    # --- Helper to compute angle class ---
    def compute_angle_class(p1, p2, num_classes=num_angle_classes):
        angle = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
        angle = (angle + np.pi) / (2*np.pi)  # normalize to [0,1)
        cls = int(angle * num_classes) % num_classes
        return cls

    # --- Generate points and line images ---
    points = [(x, y) for x in range(patch_size) for y in range(patch_size)]
    images, angle_classes, pairs = [], [], []

    for i in range(len(points)):
        for j in range(i+1, len(points)):
            # Draw line image
            img = np.zeros((patch_size, patch_size), dtype=np.float32)
            cv2.line(img, points[i], points[j], color=1.0, thickness=1)
            images.append(img.flatten())  # flatten for transformer

            # Angle class
            angle_classes.append(compute_angle_class(points[i], points[j], num_angle_classes))
            pairs.append((points[i], points[j]))

    images = np.stack(images)  # [num_lines, patch_dim]
    images_tensor = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # [num_lines, 1, patch_dim]

    # --- Pass through patch transformer ---
    model.eval()
    with torch.no_grad():
        embeddings = model.patch_embed.transformer(images_tensor)  # [num_lines, 1, embed_dim]
        embeddings = embeddings.squeeze(1).cpu().numpy()  # [num_lines, embed_dim]

    # --- PCA ---
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # --- Prepare DataFrame for interactive plot ---
    labels = [f"A: {A}, B: {B}" for (A, B) in pairs]
    df = pd.DataFrame({
        "PC1": embeddings_2d[:,0],
        "PC2": embeddings_2d[:,1],
        "AngleClass": angle_classes,
        "Label": labels
    })

    # --- Interactive Plot ---
    fig = px.scatter(
        df, x="PC1", y="PC2", color="AngleClass",
        hover_data=["Label"], color_continuous_scale=px.colors.qualitative.Plotly
    )
    fig.update_layout(
        title=f"PCA of {patch_size}x{patch_size} Patch Lines Colored by Angle Class",
        xaxis_title="PC1",
        yaxis_title="PC2",
        width=700, height=700
    )
    fig.show()

    return df, embeddings_2d








