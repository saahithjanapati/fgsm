import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np





# Function to load the MNIST model
def load_mnist_model():
    class MNISTClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(784, 512)
            self.fc2 = torch.nn.Linear(512, 512)
            self.fc3 = torch.nn.Linear(512, 256)
            self.fc4 = torch.nn.Linear(256, 128)
            self.fc5 = torch.nn.Linear(128, 10)
            self.dropout = torch.nn.Dropout(0.1)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            x = self.dropout(x)
            x = self.relu(self.fc4(x))
            return self.fc5(x)

    model = MNISTClassifier()
    state_dict = torch.load("mnist_state_dict.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    return model


# Standardize the scale for visualization
def standardize_image(image, global_min, global_max):
    image = (image - global_min) / (global_max - global_min)
    image = np.clip(image, 0, 1)  # Ensure the values are between 0 and 1
    return image


# FGSM attack function for MNIST
def run_mnist_fgsm(model, x, y, epsilon):
    model.eval()
    x = x.view(-1, 784).requires_grad_(True)
    out = model(x)
    original_prob = F.softmax(out, dim=-1)[0, y.item()].item()
    loss = F.cross_entropy(out, y)
    loss.backward()
    grad_sign = torch.sign(x.grad)
    modified_images = []
    diffs = []
    probs = []

    for eps in epsilon:
        x_mod = x + eps * grad_sign
        prob = F.softmax(model(x_mod), dim=-1)[0, y.item()].item()
        modified_images.append(x_mod.detach())
        diffs.append((x_mod - x).detach())
        probs.append(prob)

    return original_prob, modified_images, diffs, probs


# FGSM attack function for Imagenette
def run_imagenet_fgsm(model, x, y, epsilon):
    
    """
    # target_classes = [
    "tench", "English Springer Spaniel", "cassette player", "chainsaw",
    "church", "French horn", "garbage truck", "gas pump",
    "golf ball", "parachute"
    """
    
    class_map = {0: 0, 1: 217, 2: 482, 3: 491, 4: 497, 5: 566, 6: 569, 7: 571, 8: 574, 9: 701}
    
    y = torch.tensor([class_map[y.item()]])
    print(f"y_mod: {y}")


    model.eval()
    x = x.requires_grad_(True)
    out = model(x)
    original_prob = F.softmax(out, dim=-1)[0, y.item()].item()
    loss = F.cross_entropy(out, y)
    loss.backward()
    grad_sign = torch.sign(x.grad)
    modified_images = []
    diffs = []
    probs = []

    for eps in epsilon:
        x_mod = x + eps * grad_sign
        prob = F.softmax(model(x_mod), dim=-1)[0, y.item()].item()
        modified_images.append(x_mod.squeeze().detach())
        diffs.append((x_mod - x).squeeze().detach())
        probs.append(prob)

    return original_prob, modified_images, diffs, probs


def main(dataset_name, epsilon_values):
    epsilon_values = list(map(float, epsilon_values.split(",")))

    if dataset_name == "MNIST":
        model = load_mnist_model()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST(root="./mnist_data", train=False, transform=transform, download=True)
        loader = DataLoader(dataset, shuffle=True, batch_size=1)
        x, y = next(iter(loader))
        original_prob, modified_images, diffs, probs = run_mnist_fgsm(model, x, y, epsilon_values)
        original_image = x[0].view(28, 28).detach().cpu().numpy()
    
    
    elif dataset_name == "Imagenette":
        model = models.resnet50(pretrained=True)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        dataset = datasets.Imagenette(root="./imagenette", split='train', size='full', transform = transform)
        loader = DataLoader(dataset, shuffle=True, batch_size=1)
        x, y = next(iter(loader))
        
        # print(f"x.size: {x.size()}")
        # print(f"y.size: {y.size()}")
        # print(f"y: {y}")
        
        original_prob, modified_images, diffs, probs = run_imagenet_fgsm(model, x, y, epsilon_values)
        original_image = x[0].permute(1, 2, 0).detach().cpu().numpy()

    # Determine global min and max for standardization
    all_images = [original_image] + [img.view(28, 28).cpu().numpy() if dataset_name == "MNIST" else img.permute(1, 2, 0).cpu().numpy() for img in modified_images]
    global_min, global_max = np.min(all_images), np.max(all_images)

    # Create a single output figure
    fig, axes = plt.subplots(2, len(epsilon_values) + 1, figsize=(15, 6))

    # Original Image (row 1, column 0)
    axes[0, 0].imshow(standardize_image(original_image, global_min, global_max), cmap="gray" if dataset_name == "MNIST" else None)
    axes[0, 0].set_title(f"Probability of correct class\n{original_prob:.2e}")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    # Modified Images and Differences
    for i, eps in enumerate(epsilon_values):
        mod_img = modified_images[i].view(28, 28).cpu().numpy() if dataset_name == "MNIST" else modified_images[i].permute(1, 2, 0).cpu().numpy()
        diff_img = diffs[i].view(28, 28).cpu().numpy() if dataset_name == "MNIST" else diffs[i].permute(1, 2, 0).cpu().numpy()

        # Modified Image
        axes[0, i + 1].imshow(standardize_image(mod_img, global_min, global_max), cmap="gray" if dataset_name == "MNIST" else None)
        axes[0, i + 1].set_title(f"Eps={eps}\nProb={probs[i]:.2e}")
        axes[0, i + 1].axis("off")

        # Difference Image
        axes[1, i + 1].imshow(standardize_image(diff_img, global_min, global_max), cmap="gray" if dataset_name == "MNIST" else None)
        axes[1, i + 1].set_title(f"Noise with Eps={eps}")
        axes[1, i + 1].axis("off")

    plt.tight_layout()
    plt.savefig("output.png")
    print("Saved output figure as output.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform FGSM Adversarial Attacks on MNIST or Imagenette.")
    parser.add_argument("--dataset", type=str, choices=["MNIST", "Imagenette"], required=True, help="Dataset to use (MNIST or Imagenette).")
    parser.add_argument("--epsilon", type=str, required=True, help="Comma-separated list of epsilon values (e.g., '0.05,0.1,0.2').")
    args = parser.parse_args()

    main(args.dataset, args.epsilon)