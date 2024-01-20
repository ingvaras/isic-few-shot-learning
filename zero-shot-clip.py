import numpy as np
import torch
import clip
from PIL import Image
import os
from utils import accuracy, f1, balanced_accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)
text = clip.tokenize(["This is Squamous cell carcinoma", "This is Melanoma", "This is Melanocytic nevus", "This is Basal cell carcinoma", "This is Actinic keratosis", "This is Benign keratosis", "This is Dermatofibroma", "This is Vascular lesion"]).to(device)

true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0
positives = 0
negatives = 0
for category in ['SCC', 'AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']:
    directory_path = os.path.join('data/val', category)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            if category == 'SCC':
                positives += 1
                true_positives += np.argmax(probs[0]) == 0
                false_negatives += np.argmax(probs[0]) != 0
            else:
                negatives += 1
                false_positives += np.argmax(probs[0]) == 0
                true_negatives += np.argmax(probs[0]) != 0

print('zero-shot F1 score: ' + str(f1(true_positives, false_positives, false_negatives)))
print('zero-shot accuracy: ' + str(accuracy(true_positives, true_negatives, false_positives, false_negatives)))
print('zero-shot balanced accuracy: ' + str(balanced_accuracy(true_positives, true_negatives, positives, negatives)))
