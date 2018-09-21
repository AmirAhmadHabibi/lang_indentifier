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
        self.counts = [dict()] * (n + 1)
        self.smoothing = smoothing
        self.v = 0  # length of the vocabulary
        self.l = [0] * (n + 1)  # lambda values for deleted interpolation alg

        self._count_ngrams()
        self._set_lambdas()

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

    def _set_lambdas(self):
        """
        calculate the lambda values for deleted interpolation algorithm.
        """
        # for all the ngrams of size n
        for ngram in self.counts[self.n].keys():
            max_val = 0
            max_i = 0
            for i in range(self.n):
                sub_ngram = ngram[i:]
                try:
                    if len(sub_ngram) == 1:
                        val = (self.counts[self.n][sub_ngram] - 1) / (self.corpus_len - 1)
                    else:
                        val = (self.counts[self.n][sub_ngram] - 1) / (self.counts[self.n - 1][sub_ngram[:-1]] - 1)
                except ZeroDivisionError:
                    val = 0

                if val > max_val:
                    max_val = val
                    max_i = i

            self.l[self.n - max_i] += self.counts[self.n][ngram]

        # normalise lambdas
        sum_l = sum(self.l)
        self.l = [el / sum_l for el in self.l]

    def _calculate_prob(self, ngram, k=0):
        # TODO : zero counts
        try:
            # if its a unigram divide by the size of the vocabulary otherwise divide by count of the previous tokens
            if self.n == 1:
                return (self.counts[self.n][ngram] + k) / (self.corpus_len + (k * self.v))
            else:
                return (self.counts[self.n][ngram] + k) / (self.counts[self.n - 1][ngram[:-1]] + (k * self.v))
        except ZeroDivisionError:
            return 0

    def p(self, ngram):
        """
        returns the probability of last token of the input ngram assuming
        the previous tokens had happened.
        """
        # TODO: log probability?
        if len(ngram) != self.n:
            raise Exception("Ngram size mismatch!")

        prob = 0
        if self.smoothing == 'unsmoothed':
            prob = self._calculate_prob(ngram)
        elif self.smoothing == 'laplace':
            prob = self._calculate_prob(ngram, k=1)
        elif self.smoothing == 'interpolation':
            for i in range(self.n):
                prob += self._calculate_prob(ngram[i:]) * self.l[i+1]
        else:
            raise Exception("Smoothing method not defined!")

        return prob

