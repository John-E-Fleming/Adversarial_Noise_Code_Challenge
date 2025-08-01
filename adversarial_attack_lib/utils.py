import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np
from matplotlib import pyplot as plt

def get_confidence(model, image_tensor, target_class):
    """
    Computes the softmax confidence score for a specific target class.

    Args:
        model (torch.nn.Module): Pre-trained classification model.
        image_tensor (torch.Tensor): Input image tensor (1 x C x H x W).
        target_class (int): Index of the target class.

    Returns:
        float: Confidence score for the target class (0 to 1).
    """
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence = probabilities[0, target_class].item()
    return confidence

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Reverses ImageNet normalization on a tensor image.

    Args:
        tensor (torch.Tensor): Tensor with shape (C x H x W) or (1 x C x H x W)
        mean (list): Mean used during normalization.
        std (list): Std used during normalization.

    Returns:
        torch.Tensor: Denormalized image tensor in [0, 1] range.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.clone().detach().cpu()
    for c in range(tensor.shape[0]):
        tensor[c] = tensor[c] * std[c] + mean[c]
    return tensor.clamp(0, 1)

def tensor_to_pil(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Converts a normalized torch Tensor to a PIL Image.

    Args:
        tensor (torch.Tensor): Normalized image tensor (1 x C x H x W) or (C x H x W)
        mean (list): Mean used during normalization.
        std (list): Std used during normalization.

    Returns:
        PIL.Image.Image: Denormalized PIL image.
    """
    assert tensor.dim() in (3, 4), "Expected 3D or 4D tensor"
    denorm = denormalize(tensor, mean, std)
    if denorm.dim() == 4:
        denorm = denorm.squeeze(0)
    return to_pil_image(denorm)


def visualize_attack_result(result, adv_label="target"):
    """
    Styled visualization of adversarial attack showing:
    - Original image with label + confidence
    - Perturbation noise
    - Adversarial image with label + confidence

    Args:
        result (dict): Output from run_attack(). Must contain:
            - 'original_tensor': original image (1 x C x H x W)
            - 'adversarial_tensor': adversarial image (1 x C x H x W)
            - 'target_class': name of target class (string)
            - 'adv_confidence': float
            - 'orig_confidence': float (optional)
            - 'orig_class': name of original class (optional)
    """
    orig = result["original_tensor"].cpu().squeeze()
    adv = result["adversarial_tensor"].cpu().squeeze()
    noise = (adv - orig)

    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    def denorm(img): return torch.clamp(img * std + mean, 0, 1)
    orig_img = denorm(orig).permute(1, 2, 0).numpy()
    adv_img = denorm(adv).permute(1, 2, 0).numpy()
    noise_img = (noise / (2 * noise.abs().max()) + 0.5).permute(1, 2, 0).numpy()

    # Text Labels
    orig_class = result.get("orig_class", adv_label)
    target_class = result.get("target_class", adv_label)
    orig_conf = result.get("original_confidence", None)
    adv_conf = result.get("adversarial_confidence", None)

    orig_title = f'Original Prediction\n“{adv_label}”\n{orig_conf*100:.1f}% confidence' if orig_conf is not None else f'“{orig_class}”'
    adv_title = f'Updated Prediction\n“{target_class}”\n{adv_conf*100:.1f}% confidence' if adv_conf is not None else f'“{target_class}”'

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    axes[0].imshow(orig_img)
    axes[0].axis('off')
    axes[0].set_title(orig_title, fontsize=11)

    axes[1].imshow(noise_img)
    axes[1].axis('off')
    axes[1].set_title(f'Adverserial Noise\n(+ ϵ)', fontsize=11)

    axes[2].imshow(adv_img)
    axes[2].axis('off')
    axes[2].set_title(adv_title, fontsize=11)

    # Equal spacing
    plt.subplots_adjust(wspace=0.3)
    plt.show()