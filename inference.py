# inference.py
# Classifies a single traffic sign image using the trained model.
#
# Usage:
#   python inference.py --image path\to\your\image.jpg

import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.model import TrafficSignNet
from src.utils import CLASS_NAMES, get_device

# Must exactly match the transforms used during training.
# Any mismatch here = poor predictions. Common real-world bug.
TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.3337, 0.3064, 0.3171),
        std=(0.2672, 0.2564, 0.2629)
    ),
])


def load_model(checkpoint_path, device):
    """
    Loads the trained model from a saved checkpoint.
    Always call model.eval() before inference — disables dropout
    so predictions are deterministic.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No model found at '{checkpoint_path}'\n"
            "Run 'python main.py' first to train the model."
        )

    model = TrafficSignNet(num_classes=43)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def predict(image_path, model, device, top_k=3):
    """
    Runs inference on a single image.

    Pipeline:
      Load image → preprocess → add batch dim → forward pass
      → softmax → return top-k predictions

    Args:
        image_path : path to any JPG/PNG image
        model      : loaded TrafficSignNet in eval mode
        device     : cpu or cuda
        top_k      : how many top predictions to return

    Returns:
        list of dicts with class_id, name, confidence
    """
    image = Image.open(image_path).convert('RGB')
    tensor = TRANSFORM(image)
    tensor = tensor.unsqueeze(0).to(device)   # [1, 3, 32, 32]

    with torch.no_grad():
        logits = model(tensor)                        # [1, 43]
        probs = F.softmax(logits, dim=1)             # → probabilities

    top_probs, top_idx = torch.topk(probs, k=top_k, dim=1)

    results = []
    for i in range(top_k):
        class_id = top_idx[0][i].item()
        confidence = top_probs[0][i].item()
        results.append({
            'rank':       i + 1,
            'class_id':   class_id,
            'name':       CLASS_NAMES[class_id],
            'confidence': confidence,
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Sign Classifier — Single Image Inference"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--model",
        default="./models/best_model.pth",
        help="Path to model checkpoint (default: ./models/best_model.pth)"
    )
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.model, device)

    predictions = predict(args.image, model, device, top_k=3)

    print()
    print("=" * 50)
    print("  PREDICTION RESULT")
    print("=" * 50)
    for p in predictions:
        bar = "█" * int(p['confidence'] * 40)
        print(f"  #{p['rank']} {p['confidence']*100:5.1f}%  {p['name']}")
        print(f"       {bar}")
        print()
    print("=" * 50)
    print(f"  Prediction : {predictions[0]['name']}")
    print(f"  Confidence : {predictions[0]['confidence']*100:.1f}%")
    print("=" * 50)
    print()


if __name__ == "__main__":
    main()
