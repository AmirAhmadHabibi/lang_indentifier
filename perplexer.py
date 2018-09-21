#def perplexity(lang_model, test_set):
 #   perp = 0
    # TODO
 #   return perp

def pp(self,testcase, n_gram_value):
    """
    :param testcase: test corpus
    :param n_gram_value: the ngram number ex: bigram =2 , trigram =3
    
    """
    N = 0
    pp = 1
    for i in range(1,n_gram_value+1):
        for j in range(j, len(testcase) - i):
            N += 1
            curr_ngram = testcase[j:j+i]
            pp[i] = pp[i] * (1/ self.prob[curr_ngram] )
        pp[i] = pow(pp[i],1/N)
    return pp
