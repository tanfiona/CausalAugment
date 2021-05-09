# CausalAugment
This repository holds the supplementary materials for the paper "Causal Counterfactual Augmentation for Causal Sentence Classification". [Submission to ACL Rolling Review 2021].

# About the code
* process: Performs counterfactual augments (creates and appends edits)
* model: Holds main models
* src: Holds data files and train/test steps

# Dependencies
For generating augmentations (negations and strengthening edits):
```
numpy, pandas, nltk, pattern, spacy, collections
```

For running training and testing pipeline:
```
pytorch, cudatoolkit, IPython, tensorflow, transformers, scikit-learn, scipy
```

# Running the code
### Creating the augments
If you wish to create the negations and strengthening edits from scratch, run the edits generator at 'process/edit_doc.py'.<br> 
We have saved the main edits featured in the paper under 'src/data/.' for convenience.<br>
From here, to create copies of original+edits, run the code under 'process/process_edits.py'.<br>

### Training and testing
```
cd model
pip install .
cd ..
cd src
python main.py
```

# Main Code and Data References:
* [bert-sklearn-with-class-weight](https://github.com/junwang4/bert-sklearn-with-class-weight)
* [causal-language-use-in-science](https://github.com/junwang4/causal-language-use-in-science)
* [supervised contrastive loss](https://github.com/HobbitLong/SupContrast)
* [scite dataset](https://github.com/Das-Boot/scite/tree/master/corpus)
* [altlex dataset](https://github.com/chridey/altlex/tree/master/data)
