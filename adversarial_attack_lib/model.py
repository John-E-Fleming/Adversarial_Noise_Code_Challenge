import torch
from torchvision import models, transforms
from torchvision.models import get_model_weights

def load_model(model_name="resnet18", device=None):
    """
    Loads a pre-trained image classification model from torchvision.

    Args:
        model_name (str): Model architecture to load (e.g., 'resnet18').
        device (str or torch.device): 'cpu' or 'cuda'. Defaults to 'cuda' if available.

    Returns:
        model (torch.nn.Module): Loaded model in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_fn = getattr(models, model_name)
    weights = get_model_weights(model_name).DEFAULT
    model = model_fn(weights=weights).eval().to(device)
    return model