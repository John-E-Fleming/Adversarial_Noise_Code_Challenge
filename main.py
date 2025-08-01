import argparse
from adversarial_attack_lib.model import load_model, preprocess_image, load_imagenet_classes
from adversarial_attack_lib.utils import get_confidence

def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial Attack Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--target_class", type=str, required=True, help="Target class name (e.g. 'airliner')")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Load model and class labels
    model = load_model("resnet18")
    classes = load_imagenet_classes("resnet18")

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


if __name__ == "__main__":
    main()
