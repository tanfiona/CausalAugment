# CausalAugment
This repository holds the supplementary materials for the paper "Causal Counterfactual Augmentation for Causal Sentence Classification". [Submission to ACL Rolling Review 2021].

# About the code
* process: Performs counterfactual augments (creates and appends edits)
* model: Holds main models
* src: Holds data files and train/test steps

# Dependencies
For generating augmentations (negations and strengthening edits):
```
numpy, pandas, nltk, pattern, spacy, transformers, collections
```

For running training and testing pipeline:
```
pytorch, cudatoolkit, IPython, tensorflow, transformers, scikit-learn, scipy, joblib
```
We recommend using virtual environment to install the dependencies.

# Running the code
### Creating the augments
If you wish to create the negations and strengthening edits from scratch, run the edits generator at 'process/edit_doc.py' under '__main__'. For example, to create Negation`*`Regular edits, run ```process_and_keep_edits(dataset='pubmed')```. To create edits with heuristics, for example Negation`*`Shorten, run ```process_and_keep_edits(dataset='pubmed', extensions='_shorten')```. To run Strengthen`*`Regular edits, run ```process_and_keep_edits(dataset='pubmed_2to1')```.<br>
We have saved the main edits featured in the paper under 'src/data/.' for convenience.<br>

With the respective edits saved in csv, you can create any combination of augmented datasets built from "Original + Edits". Simply run the code under 'process/process_edits.py' by amending the function under '__main__' as follows. For example, to create our proposed augment of "Original + Negation`*`Shorten + Strengthen`*`Regular", use the following function: <br>
```
  run_one_full_round(
      directory_name = "D:/50 CausalCF/data/",
      edits_file_name = ["pubmed_edits_shorten.csv", "pubmed_2to1_edits.csv"],
      filter_examples_by = None,
      dataset_name = 'pubmed',
      edits_name = '_mix01_edits',
      extensions = '',
      base_extensions = ''
      )
```
The final augmented dataset that will be randomly sampled and deduplicated would be named as "pubmed_mix01_edits_4t_rs.csv" and saved under the same data folder.

### Training and testing
Before running the BERT+MLP or BERT+MLP+SVM pipeline, you need to install our model into the system. Enter into the model folder to pip install:
```
cd model
pip install .
cd ..
```

Subsequently, you may run the main training and testing script as follows:
```
python main.py
```
Some options are available from command line, the important ones highlighted as follows:
```
'--learning_rate', type=float, default=0.05,
'--random_state', type = int, default = 0
'--cuda_device', type = str, default = '1'
```
Other configurations, like which dataset to apply on during out-of-domain testing, is amendable within '__main__'. For example, in experiments not shown in our paper, we also train on AltLex dataset instead:
```
    mt = ModelTrainTester(
        bert_model_name='biobert', classifier_name='mlp', dataset_name='altlex', 
        train_file='altlex_base', k=5, epochs=5, data_dir='data',
        label_name = {0:'not_causal', 1:'causal'}, label_list={0: 0, 1: 1})
    mt.main(task = 'train_kfold')
```
If you already have trained your model and just want to predict the model on other datasets, after setting up the Model object as above, you could directly run:
```
    mt.fpath_unseen_data = <<YOUR TEST PATH NAME>>
    mt.main(task = 'apply_KFold_model_to_new_sentences')
```

# Main Code and Data References:
* [bert-sklearn-with-class-weight](https://github.com/junwang4/bert-sklearn-with-class-weight)
* [causal-language-use-in-science](https://github.com/junwang4/causal-language-use-in-science)
* [supervised contrastive loss](https://github.com/HobbitLong/SupContrast)
* [scite dataset](https://github.com/Das-Boot/scite/tree/master/corpus)
* [altlex dataset](https://github.com/chridey/altlex/tree/master/data)
