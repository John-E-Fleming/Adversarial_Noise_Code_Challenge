import os
import argparse
from adversarial_attack_lib.model import load_model, preprocess_image, load_imagenet_classes
from adversarial_attack_lib.utils import get_confidence, tensor_to_pil
from adversarial_attack_lib.attack import FGSMAttack

def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial Attack Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--target_class", type=str, required=True, help="Target class name (e.g. 'goldfish')")
    parser.add_argument("--epsilon", type=float, default=0.03, help="Attack strength ε")
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture (e.g. resnet18, resnet50)")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Load model and class labels
    model = load_model(args.model)
    classes = load_imagenet_classes(args.model)

    # Preprocess input
    img_tensor = preprocess_image(args.image)
    
    # Resolve target class index
    if args.target_class in classes:
        target_idx = classes.index(args.target_class)
    else:
        raise ValueError(f"Class '{args.target_class}' not found in ImageNet labels")

    # Original confidence
    orig_conf = get_confidence(model, img_tensor, target_idx)
    print(f"Original confidence for '{args.target_class}': {orig_conf:.4f}")

    # Run Fast Gradient Sign Method attack, i.e. generate adversarial image
    attack = FGSMAttack(epsilon=args.epsilon)
    adv_tensor = attack.generate(model, img_tensor, target_idx)

    # Get confidence of adversarial image
    adv_conf = get_confidence(model, adv_tensor, target_idx)
    print(f"Adversarial confidence for '{args.target_class}': {adv_conf:.4f}")

    # Save outputs
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    orig_pil = tensor_to_pil(img_tensor)
    adv_pil = tensor_to_pil(adv_tensor)
    orig_pil.save(os.path.join(out_dir, "original.png"))
    adv_pil.save(os.path.join(out_dir, "adversarial.png"))
    print(f"Saved original and adversarial images to '{out_dir}/'")

if __name__ == "__main__":
    main()
