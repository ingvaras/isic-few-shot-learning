import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)
text = clip.tokenize(["This is Squamous cell carcinoma", "This is other skin cancer type than Squamous cell carcinoma"]).to(device)

def run_over_class(directory_path, inverse=False):
    counter = 0
    correct = 0
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
        counter += 1

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        correct += probs[0][0] > probs[0][1] and not inverse or probs[0][0] < probs[0][1] and inverse

    print("Accuracy:", correct / counter)

run_over_class("data/val/SCC")
run_over_class("data/val/AK", True)
run_over_class("data/val/BCC", True)
run_over_class("data/val/BKL", True)
run_over_class("data/val/DF", True)
run_over_class("data/val/MEL", True)
run_over_class("data/val/NV", True)
run_over_class("data/val/VASC", True)