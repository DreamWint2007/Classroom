
import argparse

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "microsoft/resnet-18"
BATCH_SIZE = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="number of MNIST test samples to evaluate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval().to(device)

    def preprocess(images):
        # MNIST is 28x28 grayscale; convert to RGB, then the image processor
        # resizes to 224x224 and normalizes as the model expects.
        inputs = processor([img.convert("RGB") for img in images], return_tensors="pt")
        return inputs["pixel_values"]

    test_set = datasets.MNIST(root="./data", train=False, download=True)
    num = min(args.num_samples, len(test_set))
    subset = Subset(test_set, range(num))

    def collate(batch):
        images = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])
        return preprocess(images), labels

    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate)

    correct = 0
    printed = 0
    with torch.no_grad():
        for pixel_values, labels in loader:
            logits = model(pixel_values.to(device)).logits
            preds = logits.argmax(dim=-1).cpu()
            correct += (preds == labels).sum().item()
            if printed < 5:  # show a few predictions for sanity
                for pred, label in zip(preds, labels):
                    name = model.config.id2label[pred.item()]
                    print(f"true digit: {label.item()}, predicted ImageNet class: {name}")
                    printed += 1
                    if printed >= 5:
                        break

    accuracy = correct / num
    print(f"\nEvaluated {num} samples")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{num})")


if __name__ == "__main__":
    main()
