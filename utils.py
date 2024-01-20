def f1(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return round(2 * (precision * recall) / (precision + recall), 3)


def balanced_accuracy(tp, tn, p, n):
    return round((tp/p + tn/n)/2, 3)


def accuracy(tp, tn, fp, fn):
    return round((tp + tn)/(tp + tn + fp + fn), 3)
