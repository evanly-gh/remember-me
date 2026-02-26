import argparse
import csv
import torch
from torchvision import transforms
from PIL import Image
from basic_cnn import CelebACNN, DEVICE, IMG_SIZE, NUM_ATTRS, ATTR_CSV

MODEL_PATH = "celeba_cnn.pth"


def load_attr_names(attr_csv=ATTR_CSV):
    with open(attr_csv, "r") as f:
        header = next(csv.reader(f))
    return [name.replace("_", " ") for name in header[1:]]

transform = transforms.Compose([
    transforms.CenterCrop(min(178, 178)),  # match training crop
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])


def predict(image_path, model_path=MODEL_PATH, device=DEVICE):
    device = torch.device(device)

    model = CelebACNN(num_attrs=NUM_ATTRS)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze()

    detected = []
    absent = []
    attr_names = load_attr_names()
    for name, prob in zip(attr_names, probs):
        if prob > 0.5:
            detected.append((name, prob.item()))
        else:
            absent.append((name, prob.item()))

    detected.sort(key=lambda x: -x[1])
    absent.sort(key=lambda x: x[1])

    print("Detected attributes:")
    for name, conf in detected:
        print(f"{name:<25} {conf:.0%} confident")

    print()
    print("Absent attributes:")
    for name, conf in absent:
        print(f"{name:<25} {1 - conf:.0%} confident")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict face attributes from an image")
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to model weights")
    parser.add_argument("--device", default=DEVICE, help="Device (mps, cpu, cuda)")
    args = parser.parse_args()
    predict(args.image, args.model, args.device)
