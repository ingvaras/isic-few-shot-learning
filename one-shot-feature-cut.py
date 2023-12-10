# 
# pasidaryti auksto lygio pozymiu istraukima -
# https://stackoverflow.com/questions/56206330/how-to-extract-features-from-a-layer-of-the-pretrained-resnet-model-keras
# paduodam nauja nuotrauka xnew (is naujos klases), tada pozymiai f = model_cut(x_new),
# paimkim ir suskaiciuojam f1, f2, ..., fk nuo visu pirmos apmokytos klases vaizdu ir suskaiciuojam f*1, kuris yra ju vidurkis. Taip darom visom klasem
# gavome vidutinius vektorius f*1, f*2, ..., f*8 ir naujos klases viena vektoriu f
# tada testvimas vyksta, imame savo naujos klases visas likiusias nuotraukas testavimui ir leidziam pro modeli ir gaunam f1, f2, ..., fm
# reikia kiekvienam f1, f2, .., fm palyginti su vienu is 9 kitu vektoriu pvz cosine similiarity. lyginam su f*1, f*2, ..., f*8 ir su f. Kazkuris bus didziausias. Parenkam ta klase kuriai ir priklauso
#

# 1 do base model of 7 classes
# 2 cut out last layer, pick high level features
# 3 calculate them for all base model classes, average features of all examples
# 4 do one for new class. 
# 5 go over all other new class examples, calculate last layer of fetures and compare (consine similiarity) with all 8 feature cuts. measure how often it is the most similar to one-shot (from step 4)
import os
from math import *
from decimal import Decimal
from functools import partial
import numpy as np
from numpy.linalg import norm
import torch
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

#const
IMAGE_SIZE = 128
DB_SUB_DIRS = ['train', 'val', 'test']
ONE_SHOT_BASE_SAMPLE_NAME = 'ISIC_0024329.jpg'

# Load base model for 7 classes out of 8
model = load_model('models/base-7-classes.h5')
model_cut = Model(inputs=model.inputs, outputs=model.layers[-1].input)

def feature_cut(file_path):
    img = image.load_img(file_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return model_cut.predict(img_array, verbose=0)

def collect_class_average(class_name):
    feature_samples = []
    for directory in DB_SUB_DIRS:
        sample_dir = os.path.join('data', directory, class_name)
        for filename in os.listdir(sample_dir):
            file_path = os.path.join(sample_dir, filename)
            features = feature_cut(file_path)
            feature_samples.append(features[0])
    feature_samples = np.array(feature_samples)
    return np.average(feature_samples, axis=0)

high_lvl_features_file_path = 'models/high_lvl_features.npy'

def collect_known_classes_averages():
    known_classes_featues_averages = np.array([
        collect_class_average("AK"),
        collect_class_average("BCC"),
        collect_class_average("BKL"),
        collect_class_average("DF"),
        collect_class_average("MEL"),
        collect_class_average("NV"),
        collect_class_average("VASC")
    ])
    np.save(high_lvl_features_file_path, known_classes_featues_averages)    

def square_rooted(x):
    return round(sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)

def euclidean_distance(x,y):
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def manhattan_distance(x,y):
    return sum(abs(a-b) for a,b in zip(x,y))

def nth_root(value, n_root):
    root_value = 1/float(n_root)
    return round (Decimal(value) ** Decimal(root_value),3)

def minkowski_distance(x,y,p_value=3):
    return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)

def jaccard_similarity(x,y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

def one_shot_class_sample(file_name):
    new_calss_baseline_sample_file_path = os.path.join('data1', 'train', 'SCC', file_name)
    return feature_cut(new_calss_baseline_sample_file_path)

def is_separable(sim_func, known_classes, one_shot_base, sample):
    sim_func_partial = partial(sim_func, sample[0])
    similiarities_to_known_classes = np.apply_along_axis(sim_func_partial, 1, known_classes)
    similiarity_to_new_class_baseline = sim_func(sample[0], one_shot_base[0])
    return np.all(similiarity_to_new_class_baseline > similiarities_to_known_classes)

def test(sim_func):
    count = 0
    positives = 0
    known_classes = np.load(high_lvl_features_file_path)
    one_shot_base = one_shot_class_sample(ONE_SHOT_BASE_SAMPLE_NAME)

    for directory in DB_SUB_DIRS:
        sample_dir = os.path.join('data1', directory, 'SCC')
        for sample_file_name in os.listdir(sample_dir):
            if sample_file_name == ONE_SHOT_BASE_SAMPLE_NAME:
                continue
            file_path = os.path.join(sample_dir, sample_file_name)
            sample_features = feature_cut(file_path)

            if is_separable(sim_func, known_classes, one_shot_base, sample_features):
                positives += 1
            count += 1
    print(f"One shot accuracy is {positives/count}, using {sim_func.__name__} as feature similarity check")

if not os.path.exists(high_lvl_features_file_path):
    print('Getting averaged high level features of known classes..')
    collect_known_classes_averages()

test(cosine_similarity)
test(euclidean_distance)
test(manhattan_distance)
test(minkowski_distance)
test(jaccard_similarity)

## conclusion - results vary extremely, depending on similarity evaluation method
## cosine - 0.06379
## euclidean - 0.8181818
## manhattan - 0.5598086124401914
## minkowski - 0.82296650
## jaccard -  0.9936