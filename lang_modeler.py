class LangModeler:
    def __init__(self, name, corpus, n=4):
        """
        :param name: the name of the corpus for future reference
        :param corpus: the text of the corpus
        :param n: size of the context in n-gram
        """
        self.name = name
        self.corpus = ' ' * n + corpus + ' ' * n
        self.n = n
        self.corpus_len = len(corpus)
        self.counts = [dict() for _ in range(n + 1)]
        self.v = 0  # length of the vocabulary
        self.l = [[]] * (n + 1)  # lambda values for deleted interpolation alg

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
        calculate the lambda values for deleted interpolation algorithm. for each max ngram size
        """
        for size in range(1, self.n + 1):
            lmbd = [0] * (size + 1)
            for ngram in self.counts[size].keys():
                max_val = 0
                max_i = 0
                for i in range(size):
                    sub_ngram = ngram[i:]
                    try:
                        if len(sub_ngram) == 1:
                            val = (self.counts[size - i][sub_ngram] - 1) / (self.corpus_len - 1)
                        else:
                            val = (self.counts[size - i][sub_ngram] - 1) / (
                                    self.counts[size - i - 1][sub_ngram[:-1]] - 1)
                    except ZeroDivisionError:
                        val = 0
                    except KeyError:  # Need to investiage
                        return 0

                    if val > max_val:
                        max_val = val
                        max_i = i

                lmbd[size - max_i] += self.counts[size][ngram]

            # normalise lambdas
            sum_l = sum(lmbd)
            self.l[size] = [el / sum_l for el in lmbd]

    def _calculate_prob(self, ngram, k=0):
        # TODO : zero counts
        try:
            size = len(ngram)
            # if its a unigram divide by the size of the vocabulary otherwise divide by count of the previous tokens
            if size == 1:
                return (self.counts[size][ngram] + k) / (self.corpus_len + (k * self.v))
            else:
                return (self.counts[size][ngram] + k) / (self.counts[size - 1][ngram[:-1]] + (k * self.v))
        except ZeroDivisionError:
            return 0
        except KeyError:
            return 0

    def p(self, ngram, smoothing='unsmoothed'):
        """
        returns the probability of last token of the input ngram assuming
        the previous tokens had happened.
        """
        # TODO: log probability?

        prob = 0
        if smoothing == 'unsmoothed':
            prob = self._calculate_prob(ngram)
        elif smoothing == 'laplace':
            prob = self._calculate_prob(ngram, k=1)
        elif smoothing == 'interpolation':
            size = len(ngram)
            for i in range(size):
                prob += self._calculate_prob(ngram[i:]) * self.l[size][i + 1]
        else:
            raise Exception("Smoothing method not defined!")

        return prob
