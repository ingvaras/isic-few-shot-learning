import os

true_positives = 0
false_positives = 0
false_negatives = 0
for category in ['SCC', 'AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']:
    directory_path = os.path.join('data/val', category)
    for _ in os.listdir(directory_path):
        if category == 'SCC':
            true_positives += 1/8
            false_negatives += 7/8
        else:
            false_positives += 1/8

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)
print('random-guessing F1 score: ' + str(f1_score))