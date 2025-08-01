# 🧠 Adversarial Image Attacks with FGSM and PGD

This project demonstrates how to generate adversarial examples using Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) to fool pretrained image classification models (e.g., ResNet18). The goal is to perturb input images so that a neural network misclassifies them into a specified target class, while the perturbation remains imperceptible to the human eye.

## 📸 Example

<p align="center">
  <img src="results/pgd_algorithm_result_overview.png" width="600" />
</p>

An image of a “Maltese dog” is perturbed to be classified as a “goldfish” by the model.

---

## 📦 Features

- 🔍 Support for FGSM and PGD attacks
- 🖼️ Visualizations of original image, perturbation, and adversarial result
- 📊 Confidence scores for original and adversarial predictions
- 🧪 Jupyter notebook demo and CLI interface
- 🧠 Works with any torchvision-compatible model (e.g., ResNet, ViT)

---

## 📁 Project Structure

```
adversarial_noise_code_challenge/
├── adversarial_attack_lib/
│   ├── __init__.py           
│   ├── attack.py             # Algorithms to generate adversarial images
│   ├── model.py              # Load models and preprocess image data helper functions
│   ├── runner.py             # Helper function to run adversarial image generation function 
│   ├── utils.py              # Helper functions (get prediction confidence and visualize outputs)
│
│   main.py                   # CLI main entry point
├── examples/
│   └── original_images/      # Sample input images
│   └── AdversarialDemo.ipynb # Interactive demo with visualizations
├── results/                  # Output images (original, adversarial)
|
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Run from Python Script

```bash
python main.py \
  --image examples/original_images/dog.JPEG \
  --target_class goldfish \
  --attack fgsm \
  --epsilon 0.03 \
  --model resnet18
```

### 3. Run from Jupyter Notebook

Use `examples/AdversarialDemo.ipynb` to experiment interactively.

---

## ✏️ Example Usage (Python)

```python
from adversarial_attack_lib.runner import run_attack

result = run_attack(
    image_path="examples/original_images/dog.JPEG",
    target_class="goldfish",
    attack_type="pgd",
    epsilon=0.03,
    alpha=0.005,
    steps=10,
    model_name="resnet18",
    save_output=True
)

# Visualize result
from adversarial_attack_lib.utils import visualize_attack
visualize_attack(result)
```

---

## ✅ Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- Pillow

---

## 🧠 Author

Created by John E. Fleming 