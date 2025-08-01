import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np

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