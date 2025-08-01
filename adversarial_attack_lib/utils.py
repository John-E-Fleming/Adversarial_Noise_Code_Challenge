import torch

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
