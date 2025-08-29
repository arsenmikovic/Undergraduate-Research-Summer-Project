#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import cv2
import random
import argparse
import math
from math import atan2, degrees
from itertools import combinations
import matplotlib.pyplot as plt

"""

A Vision Transformer (ViT) implementation with Transformer patch embedding and positional encoding.
This model is designed to classify the angle of a line drawn between two points in a small image.

"""



class SinCos2DPositionalEncoding(nn.Module):
    """
    Standard 2D sinusoidal positional encoding for Vision Transformers.
    Produces a [num_patches, embed_dim] positional encoding.
    """
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Number of patches ({num_patches}) must be a perfect square.")

        # Create normalized grid coordinates
        coords = torch.stack(torch.meshgrid(
            torch.arange(grid_size), torch.arange(grid_size), indexing='ij'
        ), dim=-1).float()  # shape: [grid_size, grid_size, 2]
        coords /= (grid_size - 1)  # normalize to [0,1]
        coords = coords.view(-1, 2)  # shape: [num_patches, 2]

        # Compute the sinusoidal embedding
        self.register_buffer('pos_embed', self.build_sincos_embedding(coords, embed_dim))

    def build_sincos_embedding(self, coords, embed_dim):
        """
        coords: [num_patches, 2] normalized to [0,1]
        """
        num_patches = coords.shape[0]
        pe = torch.zeros(num_patches, embed_dim)

        # Split the embedding dimension into two halves: x and y
        dim_half = embed_dim // 2
        div_term = torch.exp(torch.arange(0, dim_half, 2, dtype=torch.float32) * -(math.log(10000.0) / dim_half))

        # Apply sin/cos to x and y separately
        for i in range(num_patches):
            x, y = coords[i]
            pe[i, 0::4] = torch.sin(x * div_term)
            pe[i, 1::4] = torch.cos(x * div_term)
            pe[i, 2::4] = torch.sin(y * div_term)
            pe[i, 3::4] = torch.cos(y * div_term)

        # If embed_dim is odd, pad last dimension
        if embed_dim % 4 != 0:
            pe = torch.cat([pe, torch.zeros(num_patches, embed_dim - pe.shape[1])], dim=1)

        return pe.unsqueeze(0)  # [1, num_patches, embed_dim]

    def forward(self, x):
        # x: [B, num_patches, embed_dim]
        return x + self.pos_embed



# --- 1. Small Transformer for patches ---
class PatchTransformer(nn.Module):
    def __init__(self, patch_dim, embed_dim=256, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(patch_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(patch_dim)
        hidden_dim = int(patch_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, patch_dim)
        )
        self.norm2 = nn.LayerNorm(patch_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(patch_dim, embed_dim)
        

    def forward(self, x):
        # x: [B, num_patches, patch_dim]
        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        mlp_out = self.mlp(x)
        x = x + self.dropout(mlp_out)
        x = self.norm2(x)
        # Project to embed_dim if needed
        x = self.proj(x)
        return x  # [B, num_patches, embed_dim] if proj applied







# --- 1. Patch Embedding ---
class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_chans=1, embed_dim=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_chans * patch_size * patch_size
        self.transformer = PatchTransformer(patch_dim=self.patch_dim, embed_dim=embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).reshape(B, -1, self.patch_dim)  # [B, num_patches, patch_dim]
        # Now maybe add a transformer here?
        return self.transformer(x)

# --- 3. Trasnformer Block---
class ViTBlock(nn.Module):
    def __init__(self, embed_dim=16, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # Multi-head attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

        # Separate LayerNorms for pre-norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # --- Attention with pre-norm ---
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output

        # --- MLP with pre-norm ---
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)

        return x, attn_weights

# --- 4. Vision Transformer ---
class VisionTransformer(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_chans=1, embed_dim=16,
                 depth=1, num_heads=4, mlp_ratio=4.0, num_classes=8):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = SinCos2DPositionalEncoding(num_patches, embed_dim)
        self.blocks = nn.Sequential(*[
            ViTBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.classifier = nn.Linear(num_patches * embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)

        attn_weights_list = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attn_weights_list.append(attn_weights)

        x = self.classifier(x.flatten(1))
        return x, attn_weights_list

# --- Training ---
def train(model, train_data, test_data, epochs=20000, lr=1e-4, weight_decay=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x_train, y_train = train_data[0].to(device), train_data[1].to(device)
    x_test, y_test = test_data[0].to(device), test_data[1].to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()

        logits, _ = model(x_train)
        loss = loss_fn(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # not sure this will help but it might
        #scheduler.step()

        if epoch % 500 == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                train_preds = logits.argmax(dim=1)
                train_acc = 100 * (train_preds == y_train).float().mean().item()
                train_loss = loss.item()

                test_logits, _ = model(x_test)
                test_loss = loss_fn(test_logits, y_test).item()
                test_acc = 100 * (test_logits.argmax(dim=1) == y_test).float().mean().item()

            print(f"Epoch {epoch:4d}: "
                  f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f} | "
                  f"Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.2f}")

    torch.save(model.state_dict(), "vit_line_angle_model.pth")
    print(" Model saved to vit_line_angle_model.pth")


# --- Drawing and Data ---
def compute_angle_class(A, B, num_classes=16):
    dx, dy = B[0] - A[0], B[1] - A[1]
    #angle = degrees(atan2(dy, dx)) % 180
    angle = (degrees(atan2(dy, dx)) + 90/num_classes) % 180
    return int(angle // (180 / num_classes))

def draw_line_image(A, B, image_size=12):
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.line(img, A, B, color=255, thickness=1)  # white line on black
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0  # [1, H, W]
    return tensor

def blind_side(image, A, B):
    """
    image: torch tensor [1, H, W], already has a line drawn
    x, y: row indices (inclusive) to potentially blank out
    A, B: endpoints of the line (tuples)
    """
    img = image.clone()

    # Check if line endpoints are in the row range
    if not (abs(A[1]-B[1]) >= 2):
        img[0, min(A[1]+1, B[1]+1):max(A[1], B[1]), :] = 0.0  # set rows between x and y to zero

    return img


# --- Modify generate_train_test_line_data to add flipped points (outliers) ---
def generate_train_test_line_data(
    image_size=8,
    num_classes=16,
    train_ratio=0.8,
    test_ratio=1.0,
    outlier_prob=0.1,
    flip_num=0,
    blind=False,
    point_only_frac=0.1  # fraction of train pairs that are "points only"
):
    coords = [(x, y) for x in range(image_size) for y in range(image_size)]
    pairs = list(combinations(coords, 2))
    random.shuffle(pairs)
    total = int(len(pairs) * train_ratio)

    def make_data(pair_list, point_only_frac=0.0):
        images, labels = [], []
        for A, B in pair_list:
            # Decide if this pair will be "points only"
            point_only = (random.random() < point_only_frac)
            
            if point_only:
                img = torch.zeros(1, image_size, image_size)  # empty image
                img[0, A[1], A[0]] = 1.0
                img[0, B[1], B[0]] = 1.0
            else:
                img = draw_line_image(A, B, image_size)
            
            if blind and random.random() < outlier_prob:
                img = blind_side(img, A, B)
            
            label = compute_angle_class(A, B, num_classes)

            # Add flipped outliers if needed
            for _ in range(flip_num):
                if random.random() < outlier_prob:
                    flipped_point = random.choice([p for p in coords if p != A and p != B])
                    img[0, flipped_point[1], flipped_point[0]] = 1.0 - img[0, flipped_point[1], flipped_point[0]]

            images.append(img)
            labels.append(label)
        x = torch.stack(images)
        y = torch.tensor(labels)
        return x, y

    train_pairs = pairs[:total]
    test_pairs = pairs[total:int(len(pairs) * test_ratio)]
    train_data = make_data(train_pairs, point_only_frac=point_only_frac)
    test_data = make_data(test_pairs, point_only_frac=0.0)  # keep test set all normal
    return train_data, test_data


# --- Evaluation ---
def test_all_pairs(model, image_size=12, num_classes=16):
    device = next(model.parameters()).device
    coords = [(x, y) for x in range(image_size) for y in range(image_size)]
    all_pairs = list(combinations(coords, 2))

    total = 0
    correct = 0
    misclassified = []

    model.eval()
    with torch.no_grad():
        for A, B in all_pairs:
            img = draw_line_image(A, B, image_size=image_size).unsqueeze(0).to(device)
            pred_class = model(img)[0].argmax(dim=1).item()
            true_class = compute_angle_class(A, B, num_classes)

            total += 1
            if pred_class == true_class:
                correct += 1
            else:
                misclassified.append((A, B, pred_class, true_class))

    accuracy = correct / total

    return round(accuracy*100, 4), misclassified
    # print(f"\n Overall Accuracy on All Point Pairs: {accuracy*100:.2f}%")
    # print(f" Misclassified {len(misclassified)} out of {total} cases\n")

    # if len(misclassified) > 0:
    #     print(" Misclassified Examples (A → B):")
    #     for i, (A, B, pred, true) in enumerate(misclassified[:20]):
    #         print(f"{i+1:2d}. {A} → {B} | Pred: {pred}, True: {true}")
    #     if len(misclassified) > 20:
    #         print("... (truncated)")



# --- Randomly check for robustness of the clasification ---
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
 


# --- Main Execution ---
def main():

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Using CUDA: {device_name} ({total_mem:.2f} GB)")
    else:
        print("CUDA is NOT available. Using CPU.")

    parser = argparse.ArgumentParser(description="Vision Transformer Line Angle Classifier")
    parser.add_argument("--image_size", type=int, default=8, help="Size of the generated images")
    parser.add_argument("--patch_size", type=int, default=1, help="Patch size for ViT")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Fraction of data for training")
    parser.add_argument("--test_ratio", type=float, default=1.0, help="Fraction of remaining data for testing")
    parser.add_argument("--epochs", type=int, default=30000, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=1.0, help="Weight decay for AdamW")
    parser.add_argument("--embed_size", type=int, default=256, help="Size of embeded space")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--outlier_prob", type=float, default=0.1, help="Probability of implementing augmentation")
    parser.add_argument("--flip_num", type=int, default=0, help="Maximum number of flip points per image")
    parser.add_argument("--depth", type=int, default=1, help="Number of Transformer layers")
    parser.add_argument("--blind", type=bool, default=False, help="Do we cover up rows 4-7?")
    parser.add_argument("--num_classes", type=int, default=16, help="Resolution of the angle classification")
    parser.add_argument("--point_only_frac", type=float, default=0.0, help="Fraction of training pairs that are 'points only'")
    



    args = parser.parse_args()

    # Generate datasets
    train_data, test_data = generate_train_test_line_data(
        image_size=args.image_size,
        num_classes=args.num_classes,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        outlier_prob=args.outlier_prob,
        flip_num=args.flip_num,
        blind=args.blind,
        point_only_frac=args.point_only_frac
    )
    print(f"Train Data Size: {len(train_data[0])}")
    print(f"Test Data Size: {len(test_data[0])}")

    # Build model
    model = VisionTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_size,
        num_classes=args.num_classes,
        depth=args.depth
    )

    # Train and evaluate
    train(model, train_data, test_data, epochs=args.epochs, weight_decay=args.weight_decay, lr=args.lr)
    test_all_pairs(model, image_size=args.image_size, num_classes=16)

    robustness_score_exclude_AB_random(model, image_size=args.image_size, num_classes=args.num_classes)


if __name__ == "__main__":
    main()
