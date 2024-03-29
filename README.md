# CausalAugment
This repository holds the supplementary materials for the paper ["Causal Augmentation for Causal Sentence Classification"](https://aclanthology.org/2021.cinlp-1.1/).

# Abstract
Scarcity of corpora with annotated causal texts can lead to poor robustness when training state-of-the-art (SOTA) language models for causal sentence classification. In particular, we find that SOTA models misclassify on augmented sentences that have been negated or strengthened in terms of their causal meaning. This is worrying because minor linguistic changes in causal sentences can have disparate meaning. To resolve these issues, we propose a rule-based augmentation of causal sentences for creating contrast sets. Interestingly, introducing simple heuristics (like sentence shortening or multiplying key causal terms) to emphasize semantically important keywords to the model can improve classification performance. We demonstrate these findings on different training setups and across two out-of-domain corpora. Our proposed mixture of augmented edits consistently achieves improved performance compared to baseline across two models and both within and out of corpus' domain, suggesting our proposed augmentation also helps the model generalize.

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
If you wish to create the negations and strengthening edits from scratch, run the edits generator at 'process/edit_doc.py' under '__main__'. For example, to create Negation\*Regular edits, run ```process_and_keep_edits(dataset='pubmed')```. To create edits with heuristics, for example Negation\*Shorten, run ```process_and_keep_edits(dataset='pubmed', extensions='_shorten')```. To run Strengthen\*Regular edits, run ```process_and_keep_edits(dataset='pubmed_2to1')```.<br>
We have saved the main edits featured in the paper under 'src/data/.' for convenience.<br>

With the respective edits saved in csv, you can create any combination of augmented datasets built from "Original + Edits". Simply run the code under 'process/process_edits.py' by amending the function under '__main__' as follows. For example, to create our proposed augment of "Original + Negation\*Shorten + Strengthen\*Regular", use the following function: <br>
```
  run_one_full_round(
      directory_name = "src/data/",
      edits_file_name = ["pubmed_edits_shorten.csv", "pubmed_2to1_edits.csv"],
      filter_examples_by = None,
      dataset_name = 'pubmed',
      edits_name = '_mix01_edits',
      extensions = '',
      base_extensions = ''
      )
```
The final augmented dataset that will be randomly sampled and deduplicated would be named as "pubmed_mix01_edits_4t_rs.csv" and saved under the same data folder.
Alternatively, you may also explore our contribution to [NL-Augmenter](https://github.com/GEM-benchmark/NL-Augmenter) and apply the transformations under `negate_strengthen` in their framework.


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
--learning_rate', type=float, default=0.05,
--random_state', type = int, default = 0
--cuda_device', type = str, default = '1'
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

# Cite Us
Accepted to CI+NLP @ EMNLP 2021.
```
@inproceedings{tan-etal-2021-causal,
    title = "Causal Augmentation for Causal Sentence Classification",
    author = "Tan, Fiona Anting  and
      Hazarika, Devamanyu  and
      Ng, See-Kiong  and
      Poria, Soujanya  and
      Zimmermann, Roger",
    booktitle = "Proceedings of the First Workshop on Causal Inference and NLP",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.cinlp-1.1",
    doi = "10.18653/v1/2021.cinlp-1.1",
    pages = "1--20"
}
```
