
def pp(model,testcase, n_gram_value):
    """
    :param testcase: test corpus
    :param n_gram_value: the ngram number ex: bigram =2 , trigram =3

    """
    N = 0
    pp = 1

    for j in range(0, len(testcase) ):
        curr_ngram = testcase[j:j+n_gram_value]
        s = model.p(curr_ngram)
        # for unsmoothed value skipping the word
        if s != 0 :
            N += 1
            pp = pp * (1/ s)
    pp = pow(pp,1/N)
    return pp

