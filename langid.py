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
        for filename in os.listdir(path):
            with open(path + filename, 'r') as f:
                min_perplexity = sys.maxsize
                i = 0
                for model in self.models[method]:
                    i += 1
                    perplexity = p.pp(model=model, testcase=f.read().replace('\n', ' '), ngram_size=model.n)
                    if perplexity < min_perplexity:
                        min_perplexity = perplexity
                        best_lang = model.name