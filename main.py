import argparse
from adversarial_attack_lib.runner import run_attack

def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial Attack Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--target_class", type=str, required=True, help="Target class name (e.g. 'goldfish')")
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture (e.g. resnet18, resnet50)")
    parser.add_argument("--attack", choices=["fgsm", "pgd"], default="fgsm", help="Type of attack algorithm to implement")
    parser.add_argument("--epsilon", type=float, default=0.03, help="Attack strength ε")
    parser.add_argument("--alpha", type=float, default=0.005, help="PGD step size")
    parser.add_argument("--steps", type=int, default=10, help="PGD number of steps")

    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Main runner function to generate adversarial image to be misclassified as target class
    run_attack(
        image_path=args.image,
        target_class=args.target_class,
        attack_type=args.attack,
        epsilon=args.epsilon,
        alpha=args.alpha,
        steps=args.steps,
        model_name=args.model, 
        save_output=True,
    )

if __name__ == "__main__":
    main()
