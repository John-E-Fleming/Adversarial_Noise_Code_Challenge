import torch
import torch.nn.functional as F

class AdversarialAttack:
    """
    Abstract base class for adversarial attacks.

    Subclasses of this class must implement the `generate()` method, which
    takes an input image and perturbs it in a way that causes a model to 
    predict a specified target class instead of the true label.

    This interface should allow multiple attack strategies (e.g. FGSM, PGD) to be
    used interchangeably.
    """
    def generate(self, model, image_tensor, target_class):
        """
        Generate an adversarial image that fools the model into predicting
        the target class.

        Args:
            model (torch.nn.Module): Pre-trained image classification model.
            image_tensor (torch.Tensor): Input image tensor (shape: 1 x C x H x W).
            target_class (int): The index of the target class to misclassify as.

        Returns:
            torch.Tensor: The adversarial image.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class FGSMAttack(AdversarialAttack):
    """
    Implements the Fast Gradient Sign Method (FGSM) adversarial attack.

    FGSM is a one-step method that perturbs the input image in the direction
    of the gradient of the loss (sign of the gradient), scaled by a small factor ε.
    """
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon

    def generate(self, model, image_tensor, target_class):
        # Clone and prepare image for gradient computation
        image = image_tensor.clone().detach().requires_grad_(True)
        target = torch.tensor([target_class], device=image.device)

        # Forward pass
        output = model(image)

        # For a targeted attack, maximize the target class score ⇒ negate loss
        loss = -F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()

        # Apply FGSM perturbation
        perturb = self.epsilon * image.grad.sign()
        adv_image = image + perturb

        # Clamp adversarial image to valid normalized range
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
        min_pixel = (0 - mean) / std
        max_pixel = (1 - mean) / std
        adv_image = torch.clamp(adv_image, min=min_pixel, max=max_pixel)

        return adv_image.detach()
    

class PGDAttack(AdversarialAttack):
    """
    Implements the Projected Gradient Descent (PGD) adversarial attack.

    PGD is an iterative version of FGSM where multiple small perturbation
    steps are taken, and the result is projected back into a valid ε-ball
    around the original image after each step.

    This method is considered one of the most effective first-order attacks.
    """
    def __init__(self, epsilon=0.03, alpha=0.005, steps=10):
        """
        Args:
            epsilon (float): Maximum total perturbation allowed.
            alpha (float): Step size per iteration.
            steps (int): Number of attack iterations.
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps

    def generate(self, model, image_tensor, target_class):
        """
        Generates an adversarial image using PGD.

        Args:
            model (torch.nn.Module): Pre-trained classification model.
            image_tensor (torch.Tensor): Input image tensor (1 x C x H x W).
            target_class (int): Index of the desired target class.

        Returns:
            torch.Tensor: Adversarial image tensor.
        """
        # Clone original image for projection constraint
        orig = image_tensor.clone().detach()

        # Initialize perturbed image with small random noise
        perturbed = orig.clone().detach() + 0.001 * torch.randn_like(orig)
        
        # Get the target class tensor
        target = torch.tensor([target_class], device=orig.device)

        for _ in range(self.steps):
            # Forward pass and compute loss (maximize target class score)
            perturbed.requires_grad = True
            output = model(perturbed)
            
            # Targeted PGD: minimize loss toward the target class
            loss = F.cross_entropy(output, target)
            
            model.zero_grad()
            loss.backward()

            # Apply gradient ascent step
            grad_sign = perturbed.grad.detach().sign()
            perturbed = perturbed - self.alpha * grad_sign

            # Project back to ε-ball around the original image (in norm constrained space)
            delta = torch.clamp(perturbed - orig, min=-self.epsilon, max=self.epsilon)
            perturbed = (orig + delta).detach()

        return perturbed.detach()

