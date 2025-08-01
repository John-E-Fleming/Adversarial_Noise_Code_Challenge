import torch
from torchvision import models, transforms
from torchvision.models import get_model_weights
from PIL import Image

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

def load_imagenet_classes(model_name="resnet18"):
    """
    Dynamically loads ImageNet class labels from the model's metadata.

    Args:
        model_name (str): Name of the torchvision model.

    Returns:
        list of str: Class labels indexed by class ID.
    """
    try:
        weights = get_model_weights(model_name).DEFAULT
        return weights.meta["categories"]
    except Exception:
        return [str(i) for i in range(1000)]  # fallback

def preprocess_image(image_path, device=None):
    """
    Loads and preprocesses an image to match model input requirements.

    Args:
        image_path (str): Path to the image file.
        device (str or torch.device): 'cpu' or 'cuda'.

    Returns:
        torch.Tensor: Preprocessed image tensor (1 x C x H x W).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Check if the image exists and is valid
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        return image_tensor
    except Exception as e:
        raise ValueError(f"Failed to load or preprocess image '{image_path}': {e}")