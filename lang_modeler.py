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
        self.corpus_len = len(corpus)
        self.counts = {i: dict() for i in range(1, n + 1)}
        self._count_ngrams()
        self.smoothing = smoothing
        self.v = 0  # length of the vocabulary

    def _count_ngrams(self):
        """
        this method counts the number of n-grams from 1 to n for the given corpus
        and stores them into a dictionary mapping the numbers k (1 to n) to the
        dictionaries of k-grams and their counts
        the resulting dictionary would be like this:
            self.counts[3]['the']=12
        """
        for i in range(self.n, len(self.corpus) - self.n):
            for j in range(1, self.n + 1):
                curr_ngram = self.corpus[i:i + j]
                if curr_ngram not in self.counts[j]:
                    self.counts[j][curr_ngram] = 1
                else:
                    self.counts[j][curr_ngram] += 1

        self.v = len(self.counts[1])

    def p(self, ngram):
        # TODO : zero counts
        if self.smoothing == 'unsmoothed':
            if self.n == 1:
                return self.counts[self.n][ngram] / self.corpus_len
            else:
                return self.counts[self.n][ngram] / self.counts[self.n - 1][ngram[:-1]]
        elif self.smoothing == 'laplace':
            if self.n == 1:
                return (self.counts[self.n][ngram] + 1) / (self.corpus_len + self.v)
            else:
                return (self.counts[self.n][ngram] + 1) / (self.counts[self.n - 1][ngram[:-1]] + self.v)
        elif self.smoothing == 'interpolation':
            # TODO
            pass
        else:
            raise Exception("Smoothing method not defined!")
        pass
