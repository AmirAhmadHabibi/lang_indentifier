import pickle
from time import time
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from lang_modeler import LangModeler
import os
import sys

TRAIN_PATH = './811_a1_train/'


class LangIder:
    def __init__(self, max_n):
        """
        :param int max_n:  maximum size of the ngrams
        """
        self.max_n = max_n
        self.models = []
        self.smoothing_methods = ['unsmoothed', 'laplace', 'interpolation']
        if os.path.isfile('./models/models_' + str(max_n) + '.pkl'):
            with open('./models/models_' + str(max_n) + '.pkl', 'rb') as infile:
                self.models = pickle.load(infile)
            print('[log] language models loaded')
        else:
            self._train_models(max_n)

        if os.path.isfile('./predictions/opt_parameters.pkl'):
            with open('./predictions/opt_parameters.pkl', 'rb') as infile:
                self.opt_param = pickle.load(infile)
            print('[log] optimised parameters loaded')
        else:
            self._optimize_parameter()

    def _train_models(self, max_ngram_size):
        """ creates an instance of LangModeler for each language file in TRAIN_PATH
        and saves them in a ./models/ for later use

        :param int max_ngram_size:
        """
        print('[log] training...')
        s_time = time()

        # build models for all languages in the directory and each n value up to max_ngram_size
        for filename in sorted(os.listdir(TRAIN_PATH)):
            with open(TRAIN_PATH + filename, 'r') as infile:
                model = LangModeler(name=filename, corpus=infile.read().replace('\n', ' '), n=max_ngram_size)
                self.models.append(model)

        os.makedirs('./models/', exist_ok=True)
        with open('./models/models_' + str(max_ngram_size) + '.pkl', 'wb') as outfile:
            pickle.dump(self.models, outfile)

        print('[log] training done in', round(time() - s_time), 's')

    def _optimize_parameter(self):
        """ iterates over all three smoothing methods and all different sizes up to n_max
        and tries to find the best size for each method based on their f1-score
        """
        # find the best ngram size for each method
        self.opt_param = dict()
        for method in self.smoothing_methods:
            # predict for each size
            best_size = 0
            best_fscore = -1
            for size in range(1, self.max_n + 1):
                print('[log] predicting', method, size)
                self.predict('./811_a1_dev/', method, size)

                # evaluate the prediction
                f_s = LangIder.f_scorer(path='./predictions/' + method + '_' + str(size) + '.txt')
                if f_s >= best_fscore:
                    best_fscore = f_s
                    best_size = size

            print('[log] best size for', method, 'is', best_size, 'with f-score of', best_fscore)
            self.opt_param[method] = best_size

        with open('./predictions/opt_parameters.pkl', 'wb') as outfile:
            pickle.dump(self.opt_param, outfile)

    def predict(self, path, method, size=None, o_file_name=None):
        """predicts the language of the files in the input path with the specified smoothing method and
        saves the results in file
        if the size parameter is not specified, it would use the optimised parameter for that smoothing method
        if the output file name is not specified the file would be saved with default naming format

        :param str path: directory of the test cases
        :param str method: smoothing method {"unsmoothed", "laplace", "interpolation"}
        :param int size: ngram size
        :param str o_file_name: customized output file name
        """
        if size is None:
            size = self.opt_param[method]
        output = ''
        # for each files in the given path find the lowest perplexity
        for filename in sorted(os.listdir(path)):
            with open(path + filename, 'r') as infile:
                test = infile.read().replace('\n', ' ')
            min_perplexity = sys.maxsize
            best_match = ''
            # find the minimum perplexity among all language models of the given smoothing method
            for model in self.models:
                perplexity = self._perplex(model=model, testcase=test, ngram_size=size, smoothing=method)
                if perplexity < min_perplexity:
                    min_perplexity = perplexity
                    best_match = model.name
            # add the most similar model name and the input test case to the output
            output += filename + '\t' + best_match + '\t' + str(round(min_perplexity, 2)) + '\t' + str(size) + '\n'

        # write the list of all input test cases and their most similar language
        os.makedirs('./predictions/', exist_ok=True)
        if o_file_name is None:
            o_file_name = method + '_' + str(size) + '.txt'
        with open('./predictions/' + o_file_name, 'w') as outfile:
            outfile.write(output)

    @staticmethod
    def f_scorer(path):
        """computes the f_score of a prediction result in the specified format

        :param str path: path to the prediction result file
        :returns: f1-score value
        :rtype: float
        """
        y_true = []
        y_pred = []
        with open(path, 'r') as infile:
            for line in infile:
                dev, tra, p, s = line.split()
                dev = dev.partition('-')[2].partition('.')[0]
                tra = tra.partition('-')[2].partition('.')[0]
                y_true.append(dev)
                y_pred.append(tra)
        le = LabelEncoder()
        le.fit(y_true)
        f_s = f1_score(le.transform(y_true), le.transform(y_pred), average='macro')
        print('[log] F-score for', path, 'is', f_s)
        return f_s

    @staticmethod
    def _perplex(model, testcase, ngram_size, smoothing):
        """computes the perplexity of the testcase given the model with a specific ngram size and smoothing method

        :param LangModeler model:
        :param str testcase:
        :param int ngram_size: size of ngrams to chunk the testcase
        :param str smoothing: {"unsmoothed", "laplace", "interpolation"}
        :returns: perplexity value
        :rtype: float
        """
        N = 0
        pp = 1
        testcase = ' ' * ngram_size + testcase + ' ' * ngram_size
        for j in range(0, len(testcase) - ngram_size):
            curr_ngram = testcase[j:j + ngram_size]
            s = model.p(curr_ngram, smoothing)
            if s != 0:
                N += 1
                pp = pp * (1 / s)
        if N != 0:
            pp = pow(pp, 1 / N)
        return pp


lid = LangIder(10)
lid.predict(path='./test/', method='unsmoothed', o_file_name='results_test_unsmoothed.txt')
lid.predict(path='./test/', method='laplace', o_file_name='results_test_laplace.txt')
lid.predict(path='./test/', method='interpolation', o_file_name='results_test_interpolation.txt')
