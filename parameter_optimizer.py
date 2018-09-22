from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

import lang_modeler as q
import perplexer as p
import os
import pickle
import sys


def fileread(method, size):
    print(" Please enter 1. ")
    model_list = []
    # creating a model for the training data and storing it.
    for filename in os.listdir(os.getcwd() + '/811_a1_dev/'):
        with open('./811_a1_dev/' + filename, "r") as f:
            model = q.LangModeler(filename, f.read().replace('\n', ''), size,method)
            model_list.append(model)
    with open('./models/' + method + str(size) + '.pkl', 'wb') as f:
        pickle.dump(model_list, f)

    # Using the stored model to test the dev data
    tofile = ''
    for filename in os.listdir(os.getcwd() + '/811_a1_train/'):
        with open('./811_a1_train/' + filename, "r") as f:
            min_perplexity = sys.maxsize
            for model_obj in model_list:
                perplexity = p.pp(model_obj, f.read(), size)
                if perplexity < min_perplexity :
                    min_perplexity = perplexity
                    possible_filename = model_obj.name
                    tofile += filename + '  ' + possible_filename + '   ' + str(round(min_perplexity,4) ) + '  ' + str(size) + '\n'
    with open('./perplexity/' + method + '_' + str(size), "w") as file1:
        file1.write(tofile)
        file1.close()


method_list = ['unsmoothed', 'laplace', 'interpolation']
for i in range(1, 6):
    for method in method_list:
        fileread(method, i)


def best_ngram_sizer(method):
    best_size = 0
    best_fscore = -1
    for filename in os.listdir(os.getcwd() + '/models/'):
        if not filename.startswith(method):
            continue
        size = int(filename.partition(' ')[3])

        X_labels = []
        y_labels = []
        with open('./models/' + filename, 'r') as infile:
            for line in infile:
                dev, tra, p, s = line.split()
                dev = dev.partition('-')[3].partition('.')[1]
                tra = tra.partition('-')[3].partition('.')[1]
                X_labels.append(dev)
                y_labels.append(tra)
        le = LabelEncoder()
        le.fit(X_labels)
        f_s = f1_score(le.transform(X_labels), le.transform(y_labels), average='macro')

        if f_s > best_fscore:
            best_fscore = f_s
            best_size = size
    print('best size for', method, 'is', best_size, 'with f-score of', best_fscore)
