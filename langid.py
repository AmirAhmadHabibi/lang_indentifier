import lang_modeler as q
import perplexer as p
import os
import sys

def fileread():
    print(" Please enter 1. ")
    model_obj = []
    #creating a model for the training data and storing it.
    for filename in os.listdir(os.getcwd() + '/811_a1_dev/'):
        with open( filename, "r") as f:
            model = q.LangModeler(filename,f.read().replace('\n', ''))
            model_obj.append(model)

    #Using the stored model to test the dev data
    for filename in os.listdir(os.getcwd()+'/811_a1_train/'):
        with open( filename, "r") as f:
            low_perplexer = sys.maxsize
            i = 0
            while i < len(model_obj):
                i += 1
                perplexer = p.pp(model_obj[i], f.read(), n_gram_value=3)
                if perplexer < low_perplexer :
                    low_perplexer = perplexer
                    possible_filename = model_obj[i].name

                    #TODO Precision and Recall method need tobe called from here

fileread()
