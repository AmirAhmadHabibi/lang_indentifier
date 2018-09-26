#LangId
You can run the langid.py file and give it the parameters, smoothing method and the path to the directory containing the 
test files. In the first run it would need some time to train the models and optimise parameters but then the files 
would be saved on disk and the next runs would not take time for training again. Without the input values the default 
method would be "_unsmoothed_" and the default path would be "_./811_a1_test_final/_".

A sample run of the programme would be like this:
```
> python langid.py laplace
```

or
```
> python langid.py interpolation "./test/"
```