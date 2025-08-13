import torch
import json
from timm.data import create_transform
from torchvision import datasets
from pathlib import Path

def load_model(model_path, model_name, num_classes=2, device=None):
    # num_classes should be 2: hybrid and non-hybrid
    import timm
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model, checkpoint.get('class_to_idx', None)

def infer_image(model, img_path, img_size, device):
    transform = create_transform(input_size=img_size)
    from PIL import Image
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        pred = outputs.argmax(dim=1).item()
    # 0: hybrid, 1: non-hybrid (or use class_to_idx for mapping)
    return pred


# Example usage
if __name__ == "__main__":
    model, class_to_idx = load_model("model.pth", "model_name", device="cuda")
    result = infer_image(model, "image.jpg", 224, device="cuda")
    print(result)