def pp(model, testcase, ngram_size):
    """
    :param model: language model class
    :param testcase: test corpus
    :param ngram_size: the ngram number ex: bigram =2 , trigram =3

    """
    N = 0
    pp = 1
    testcase = ' ' * ngram_size + testcase + ' ' * ngram_size

    for j in range(0, len(testcase) - ngram_size):
        curr_ngram = testcase[j:j + ngram_size]
        s = model.p(curr_ngram)
        # for unsmoothed value skipping the word
        if s != 0:
            N += 1
            pp = pp * (1 / s)
    pp = pow(pp, 1 / N)
    return pp
