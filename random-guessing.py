import os
from utils import accuracy, f1

true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0
for category in ['SCC', 'AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']:
    directory_path = os.path.join('data/test', category)
    for _ in os.listdir(directory_path):
        if category == 'SCC':
            true_positives += 1/8
            false_negatives += 7/8
        else:
            false_positives += 1/8
            true_negatives += 7/8

print('random-guessing F1 score: ' + str(f1(true_positives, false_positives, false_negatives)))
print('random-guessing accuracy: ' + str(accuracy(true_positives, true_negatives, false_positives, false_negatives)))