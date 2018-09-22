import pickle
from lang_modeler import LangModeler
import perplexer as p
import os
import sys

TRAIN_PATH = './811_a1_train/'


class LangIder:
    def __init__(self):
        self.models = dict()
        for method in ['unsmoothed', 'laplace', 'interpolation']:
            with open('./models/' + method + '.pkl', 'rb') as infile:
                self.models[method] = pickle.load(infile)
        # # build all models
        # self.models = []
        # for filename in os.listdir(TRAIN_PATH):
        #     with open(TRAIN_PATH + filename, 'r') as f:
        #         model = LangModeler(filename, f.read().replace('\n', ' '), ngram_size, method)
        #         self.models.append(model)

    def predict(self, path, method):
        output = ''
        size = self.models[method][0].n

        # read all test cases in the given path
        for filename in os.listdir(path):
            with open(path + filename, 'r') as infile:
                test = infile.read().replace('\n', ' ')
            min_perplexity = sys.maxsize
            best_match = ''
            # find the min perplexity among all language models of the given smoothing method
            for model in self.models[method]:
                perplexity = p.pp(model=model, testcase=test, ngram_size=size)
                if perplexity < min_perplexity:
                    min_perplexity = perplexity
                    best_match = model.name
            # add the most similar model name and the input test case to the output
            output += filename + '  ' + best_match + '   ' + str(round(min_perplexity, 4)) + '  ' + str(size) + '\n'

        # write the list of all input test cases and their most similar language
        with open('./perplexity/' + method + '_' + str(size), "w") as outfile:
            outfile.write(output)
