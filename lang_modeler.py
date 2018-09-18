class LangModeler:
    def __init__(self, corpus_name, corpus, n=4, smoothing='unsmoothed'):
        """
        :param corpus_name: the name of the corpus for future reference
        :param corpus: the text of the corpus
        :param n: size of the context in n-gram
        :param smoothing: one of the following methods: {"unsmoothed", "laplace", "interpolation"}
        """
        self.name = corpus_name
        self.corpus = ' ' * n + corpus + ' ' * n
        self.n = n
        self.ngrams = {i: dict() for i in range(1, n + 1)}
        self._count_ngrams()
        self.smoothing = smoothing

    def _count_ngrams(self):
        """
        this method counts the number of n-grams from 1 to n for the given corpus
        and stores them into a dictionary mapping the numbers k (1 to n) to the
        dictionaries of k-grams and their counts
        the resulting dictionary would be like this:
            self.ngrams[3]['the']=12
        """
        for i in range(self.n, len(self.corpus) - self.n):
            for j in range(1, self.n + 1):
                curr_ngram = self.corpus[i:i + j]
                if curr_ngram not in self.ngrams[j]:
                    self.ngrams[j][curr_ngram] = 1
                else:
                    self.ngrams[j][curr_ngram] += 1

    def p(self, ngram):
        if self.smoothing == 'unsmoothed':
            # TODO
            pass
        elif self.smoothing == 'laplace':
            # TODO
            pass
        elif self.smoothing == 'interpolation':
            # TODO
            pass
        else:
            raise Exception("Smoothing method not defined!")
        pass
