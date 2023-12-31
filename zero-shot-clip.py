import numpy as np
import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)
text = clip.tokenize(["This is Squamous cell carcinoma", "This is Melanoma", "This is Melanocytic nevus", "This is Basal cell carcinoma", "This is Actinic keratosis", "This is Benign keratosis", "This is Dermatofibroma", "This is Vascular lesion"]).to(device)

true_positives = 0
false_positives = 0
false_negatives = 0
for category in ['SCC', 'AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']:
    directory_path = os.path.join('data/val', category)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            if category == 'SCC':
                true_positives += np.argmax(probs[0]) == 0
                false_negatives += np.argmax(probs[0]) != 0
            else:
                false_positives += np.argmax(probs[0]) == 0

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)
print('zero-shot F1 score: ' + str(f1_score))
