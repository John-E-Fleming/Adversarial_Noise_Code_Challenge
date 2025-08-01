from .attack import FGSMAttack, PGDAttack
from .model import load_model, preprocess_image, load_imagenet_classes
from .utils import get_confidence, tensor_to_pil

import os
import torch

def run_attack(image_path, target_class, attack_type="fgsm", epsilon=0.03, alpha=0.005, steps=10, model_name="resnet18", save_output=False, save_dir="results"):
    """
    Executes a targeted adversarial attack (FGSM or PGD) on a given input image.

    Args:
        image_path (str): Path to the input image file.
        target_class (str): Name of the ImageNet target class to misclassify as.
        attack_type (str): Either 'fgsm' or 'pgd'. Specifies the attack method to use.
        epsilon (float): Maximum pixel-wise perturbation. Controls the attack strength.
        alpha (float): Step size for PGD (ignored for FGSM).
        steps (int): Number of iterations for PGD (ignored for FGSM).
        model_name (str): Name of the model architecture to use (e.g., 'resnet18').
        save_output (bool): Whether to save the original and adversarial images.
        save_dir (str): Directory where output images will be saved.

    Returns:
        dict: A dictionary containing:
            - 'original_confidence' (float)
            - 'adversarial_confidence' (float)
            - 'original_tensor' (Tensor)
            - 'adversarial_tensor' (Tensor)
            - 'target_index' (int)
    """

    # Validate the input parameters are correct
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if attack_type not in {"fgsm", "pgd"}:
        raise ValueError(f"Invalid attack_type '{attack_type}'. Choose 'fgsm' or 'pgd'.")

    if not (0.0 < epsilon <= 1.0):
        raise ValueError(f"epsilon should be between 0 and 1. Got: {epsilon}")

    if attack_type == "pgd":
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha should be between 0 and 1 for PGD. Got: {alpha}")
        if not (steps > 0 and isinstance(steps, int)):
            raise ValueError(f"steps should be a positive integer. Got: {steps}")

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Load the model and ImageNet classes
    model = load_model(model_name)
    classes = load_imagenet_classes(model_name)

    # Preprocess the input image
    img_tensor = preprocess_image(image_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    img_tensor = img_tensor.to(device)

    # Cofirm target class is valid
    if target_class not in classes:
        raise ValueError(f"Target class '{target_class}' not found in ImageNet labels.")

    # Get the index of the target class
    target_idx = classes.index(target_class)

    # Check what the original confidence is for the target class
    orig_conf = get_confidence(model, img_tensor, target_idx)
    print(f"Original confidence for '{target_class}': {orig_conf:.4f}")

    # Implement the attack to generate adversarial examples, either FGSM or PGD algorithms
    if attack_type == "fgsm":
        attack = FGSMAttack(epsilon=epsilon)
    else:
        attack = PGDAttack(epsilon=epsilon, alpha=alpha, steps=steps)

    # Generate adversarial image
    adv_tensor = attack.generate(model, img_tensor, target_idx)

    # Check the confidence of the adversarial image for the target class
    adv_conf = get_confidence(model, adv_tensor, target_idx)
    print(f"Adversarial confidence for '{target_class}': {adv_conf:.4f}")

    # Save the original and adversarial images if requested
    if save_output:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving images to '{save_dir}/'")
        tensor_to_pil(img_tensor).save(os.path.join(save_dir, "original_image.png"))
        tensor_to_pil(adv_tensor).save(os.path.join(save_dir, "adversarial_image.png"))
        print(f"Saved images to '{save_dir}/'")

    # Return the results as a dictionary
    return {
        "original_confidence": orig_conf,
        "adversarial_confidence": adv_conf,
        "original_tensor": img_tensor,
        "adversarial_tensor": adv_tensor,
        "target_index": target_idx,
    }