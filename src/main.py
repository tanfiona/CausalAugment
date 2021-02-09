import os, sys, time, re, json, argparse
import numpy as np
import pandas as pd
from scipy import stats
from IPython.display import display
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from bert_sklearn import BertClassifier, load_model
from joblib import dump, load
from sklearn.svm import SVC

# print options
pd.options.display.max_colwidth = 500
pd.options.display.width = 1000
pd.options.display.precision = 3
np.set_printoptions(precision=3)

# arguments
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # causal science
    parser.add_argument('--random_state', type = int, default = 0, help = 'Set random state of the model')
    parser.add_argument('--cuda_device', type = str, default = '1', help = 'Set which cuda device to run on (0 or 1)')
    parser.add_argument('--classifier_name', type = str, default = 'mlp',
                        choices=['mlp', 'svm', 'supcon'], 
                        help = 'Set which cuda device to run on (0 or 1)')
    # supcontrast
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='pubmed',
                        choices=['pubmed'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt
opt = parse_option()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda_device)
RANDOM_STATE = opt.random_state

class ModelTrainTester(object):
    def __init__(
        self, bert_model_name, classifier_name, dataset_name, label_name, 
        k, epochs, data_dir, train_file, learning_rate = None, label_list = None, 
        supcon_incl_negs = False, supcon_edit_name=None, supcon_aug_method=None, loss_method=None):
        BERT_NAME_2_MODEL = {
            'bert' : 'bert-base-cased',
            'biobert' : 'biobert-base-cased'
        }
        self.bert_model_name = bert_model_name
        self.bert_model = BERT_NAME_2_MODEL[bert_model_name]
        self.classifier_name = classifier_name
        self.loss_method = loss_method
        self.dataset_name = dataset_name
        self.folder_name = str(dataset_name) + '_' + str(classifier_name).upper() # project name
        self.label_name = label_name
        self.num_classes = len(label_name)
        self.k = k
        self.epochs = epochs
        self.supcon_incl_negs = supcon_incl_negs
        if supcon_incl_negs:
            self.supcon_edit_name = supcon_edit_name
        else:
            self.supcon_edit_name = ''
        self.data_dir = data_dir
        self.train_file = train_file
        self.extensions = '_'.join(train_file.split('_')[1:])
        self.model_dir = f'model/{self.folder_name}_{bert_model_name}'
        self.fpath_unseen_data = "data/scite_base.csv"
        self.label_list = label_list
        if learning_rate is None:
            self.learning_rate = 2e-5
        else:
            self.learning_rate = learning_rate
        self.supcon_aug_method = self.format_ext_string(supcon_aug_method)
        self.manual_naming_changes = ''
        self.manual_predname_changes = '' #''

    def get_train_data_csv_fpath(self, extensions=''): 
        return f'{self.data_dir}/{self.train_file}{extensions}.csv'
    
    def read_train_data(self, train_data_csv_path = None):
        if train_data_csv_path is None:
            train_data_csv_path = self.get_train_data_csv_fpath()
        # build_list = [i*(round(3000/100,0)) for i in range(0,100)]
        return pd.read_csv(
            train_data_csv_path, 
            usecols=['sentence', 'label', 'pmid'], 
            encoding = 'utf8', 
            keep_default_na=False
            # , skiprows = lambda x: x not in build_list
            )
    def clean_str(self, s): return s.strip() # BioBert or cased-Bert works better with cased letters, so don't use s.lower()

    def get_class_weight(self, labels):
        class_weight = [x for x in compute_class_weight(
            class_weight="balanced", classes=range(len(set(labels))), y=labels)]
        print('- auto-computed class weight:', class_weight)
        return class_weight

    def get_model_bin_file(self, fold=0):
        directory = f'{self.model_dir}/{self.extensions}{self.supcon_edit_name}{self.format_ext_string(self.loss_method)}{self.supcon_aug_method}{self.manual_naming_changes}/{RANDOM_STATE}'
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'\ncreate a new folder for storing BERT model: "{directory}/{RANDOM_STATE}"\n')
        if fold>=0:
            return f'{directory}/K{self.k}_epochs{self.epochs}_{fold}.bin'
        elif fold==-1:
            return f'{directory}/K{self.k}_epochs{self.epochs}_{self.epochs-1}.bin'
        else:
            print('Wrong value of fold:', fold)
            sys.exit()

    def format_ext_string(self, in_string):
        if in_string is None:
            out = ''
        elif in_string != '':
            out = '_'+in_string
        else:
            out = ''
        return out

    def get_pred_csv_file(self, mode='train'):
        pred_folder = f'./pred/{self.folder_name}_{self.bert_model_name}_{mode}/{RANDOM_STATE}'
        if not os.path.exists(f'{pred_folder}'):
            os.makedirs(pred_folder)
            print(f'\ncreate new folder for prediction results: "{pred_folder}"\n')
        if mode == 'train':
            return f'{pred_folder}/K{self.k}_epochs{self.epochs}{self.format_ext_string(self.extensions)}{self.supcon_edit_name}{self.format_ext_string(self.loss_method)}{self.supcon_aug_method}{self.manual_naming_changes}{self.manual_predname_changes}.csv'
        elif mode == 'apply':
            return f'{pred_folder}/epochs{self.epochs}{self.format_ext_string(self.extensions)}{self.supcon_edit_name}{self.format_ext_string(self.loss_method)}{self.supcon_aug_method}{self.manual_naming_changes}{self.manual_predname_changes}.csv'
        else:
            print('- wrong mode:', mode, '\n')

    def get_train_test_data(self, df, fold=0):
        df['sentence'] = df.sentence.apply(self.clean_str)
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=RANDOM_STATE)
        for i, (train_index, test_index) in enumerate(skf.split(df.sentence, df.label)):
            if i == fold:
                break
        train = df.iloc[train_index]
        test = df.iloc[test_index]

        print(f"ALL: {len(df)}   TRAIN: {len(train)}   TEST: {len(test)}")
        label_list = np.unique(train.label)
        return train, test, label_list

    def get_x_y_data(self, data, data_type='train'):
        if self.classifier_name == 'supcon':
            if self.supcon_incl_negs:
                if data_type == 'test':
                    df_neg = data[['_neg', '_neg_aug', 'label_neg']].drop_duplicates()
                    df_neg = df_neg[df_neg['_neg']!='']
                    df_neg.columns = ['sentence', '_aug', 'label']
                    X = pd.concat([
                        data[['sentence', '_aug']], 
                        df_neg[['sentence', '_aug']]
                        ], ignore_index=True)
                    y = pd.concat([data['label'], df_neg['label']], ignore_index=True)
                else:
                    X = data[['sentence', '_aug', '_neg', '_neg_aug']]
                    y = data[['label', 'label_neg']]
            else:
                X = data[['sentence', '_aug']]
                y = data['label']
        else:
            X = data['sentence']
            y = data['label']
        return X, y.astype(int)

    def train_model(self, train, model_file_to_save, epochs=3, val_frac=0.1, class_weight=None, classifier_name='mlp'):
        X_train, y_train = self.get_x_y_data(data=train)
        max_seq_length, train_batch_size = 128, 32
        model = BertClassifier(bert_model=self.bert_model, random_state=RANDOM_STATE, \
                                class_weight=class_weight, max_seq_length=max_seq_length, \
                                train_batch_size=train_batch_size, learning_rate=self.learning_rate, \
                                epochs=epochs, validation_fraction=val_frac, 
                                classifier_name=classifier_name, loss_method=self.loss_method,
                                label_list=self.label_list)
        model.fit(X_train, y_train)
        model.save(model_file_to_save)
        print(f'\n- model saved to: {model_file_to_save}\n')

        if classifier_name == 'supcon':
            X_train, y_train = self.get_x_y_data(data=train, data_type='test')

        if classifier_name == 'svm' or classifier_name == 'supcon':
            # svm model outputs mid_outputs
            y_pred, mid_output = model.predict(X_train)
            # print(f'y_train.shape: {y_train.shape} | y_pred.shape: {y_pred.shape} | mid_output.shape: {mid_output.shape}')
            # print('Example y_pred: {} and mid_output: {}'.format(y_pred[0], mid_output[0]))
            print('np.any(np.isnan(mid_output))', np.any(np.isnan(mid_output)))
            print('np.any(np.isnan(y_train))', np.any(np.isnan(y_train)))
            svm = SVC(kernel='linear', C=1e-2)
            svm.fit(mid_output, y_train)
            model_file_to_save_2 = model_file_to_save[:-4]+'.joblib'
            dump(svm, model_file_to_save_2)
            print(f'\n- model saved to: {model_file_to_save_2}\n')
            return model, svm
        else:
            return model

    def train_one_full_model(self):
        df_train = self.read_train_data()
        class_weight = self.get_class_weight(df_train['label'])

        model_file_to_save = self.get_model_bin_file(fold=-1) # -1: for one full model
        self.train_model(df_train, model_file_to_save, epochs=self.epochs, val_frac=0.15, class_weight=None, classifier_name=CLS_NAME)

    def train_KFold_model(self):

        df = self.read_train_data()
        print('- label value counts:', df.label.value_counts())

        def append_by_pmid(df, aug, aug_name):
            # discard bases
            aug['pmid'] = aug['pmid'].apply(lambda x: '' if '_' not in x else str(x).split('_')[0])
            # merge on base
            df = df.merge(aug[['sentence', 'pmid', 'label']].rename(columns={"sentence": aug_name}), \
                on='pmid', how='left', suffixes=['',aug_name])
            return df

        if self.classifier_name == 'supcon':
            df['pmid'] = df['pmid'].apply(lambda x: str(x))
            aug = self.read_train_data(f'{self.data_dir}/{self.train_file}{self.supcon_aug_method}.csv')
            df = append_by_pmid(df, aug, '_aug')
            if self.supcon_incl_negs:
                neg = self.read_train_data(f'{self.data_dir}/{self.dataset_name}{self.supcon_edit_name}.csv')
                df = append_by_pmid(df, neg, '_neg')
                neg_aug = self.read_train_data(f'{self.data_dir}/{self.dataset_name}{self.supcon_edit_name}{self.supcon_aug_method}.csv')
                df = append_by_pmid(df, neg_aug, '_neg_aug')
                df['label_neg'] = df['label_neg'].fillna(0) # '' text equiv is 0 tag (dropped eventually)
            df = df.fillna('')
            df = df.drop_duplicates(subset='sentence')

        y_test_all, y_pred_all = [], []
        results = []
        df_out_proba = None

        for fold in range(self.k):
            train_data, test_data, label_list = self.get_train_test_data(df, fold)

            model_file = self.get_model_bin_file(fold)
            use_class_weight_for_unbalanced_data = True
            if self.supcon_incl_negs:
                labels = list(df['label']) + list(df[df['_neg']!='']['label_neg'])
            else:
                labels = df['label']
            class_weight = self.get_class_weight(labels) if use_class_weight_for_unbalanced_data else None

            val_frac = 0.05
            model = self.train_model(train_data, model_file, epochs=self.epochs, val_frac=val_frac, class_weight=class_weight, classifier_name=self.classifier_name)
            X_test, y_test = self.get_x_y_data(data=test_data, data_type='test')

            if self.classifier_name == 'svm' or self.classifier_name == 'supcon':
                # for svm, tuple of model is returned from train_model
                model, svm = model
                # svm model outputs mid_outputs
                y_proba, mid_output = model.predict_proba(X_test)
                y_pred = svm.predict(mid_output)
                del mid_output
            else:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
            
            del model
            y_pred_all += y_pred.tolist()
            # print('y_proba:', y_proba)
            # print('y_proba.shape', y_proba.shape)
            # print('y_pred.shape', y_pred.shape)
            # print('X_test.shape', X_test.shape)
            # print('y_test.shape', y_test.shape)
            """
            y_proba.shape (1754, 5)
            y_pred.shape (1754,)
            X_test.shape (1754, 4)
            y_test.shape (1754,)
            y_pred.shape: (1754,)
            y_test.shape: (1754,)
            """

            tmp = pd.DataFrame(data=y_proba, columns=[f'c{i}' for i in range(self.num_classes)])
            tmp['confidence'] = tmp.max(axis=1)
            tmp['winner'] = tmp.idxmax(axis=1)

            if self.classifier_name == 'svm' or self.classifier_name == 'supcon':
                tmp['winner_svm'] = y_pred

            if self.classifier_name == 'supcon':
                tmp['sentence'] = X_test['sentence'].tolist()
                tmp['label'] = y_test.tolist()
                tmp = tmp.merge(df[['sentence', 'pmid']], on='sentence', how='left')
                tmp = tmp.merge(aug[['sentence', 'pmid']], on='sentence', how='left', suffixes=['','_aug'])
                tmp['pmid'] = tmp['pmid'].fillna(tmp['pmid_aug'])
                del(tmp['pmid_aug'])
                # add neg below

                # tmp['sentence'] = pd.concat(
                #     [X_test.iloc[:,0], X_test.iloc[:,1]], 
                #     ignore_index=True).tolist()
                # tmp['label'] = y_test.tolist()
                # if 'pmid' in test_data.columns:
                #     top = test_data['pmid'].astype(str).tolist() 
                #     bottom = [str(s) + '_mask' for s in top]
                #     tmp['pmid'] = top + bottom
                # else:
                #     tmp['pmid'] = None
            else:
                tmp['sentence'] = X_test.tolist()
                tmp['label'] = y_test.tolist()
                tmp['pmid'] = test_data['pmid'].tolist() if 'pmid' in test_data.columns else None

            df_out_proba = tmp if df_out_proba is None else pd.concat((df_out_proba, tmp))
            y_test_all += y_test.tolist()

            acc = accuracy_score(y_pred, y_test)
            res = precision_recall_fscore_support(y_test, y_pred, average='macro')
            print(f'\nAcc: {acc:.3f}      F1:{res[2]:.3f}       P: {res[0]:.3f}   R: {res[1]:.3f} \n')

            item = {'Acc': acc, 'weight': len(test_data)/len(df), 'size': len(test_data)}
            item.update({'P':res[0], 'R':res[1], 'F1':res[2]})
            for cls in np.unique(y_test):
                res = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[cls])
                for i, scoring in enumerate('P R F1'.split()):
                    item['{}_{}'.format(scoring, cls)] = res[i][0]
            results.append(item)

            acc_all = np.mean(np.array(y_pred_all) == np.array(y_test_all))
            res = precision_recall_fscore_support(y_test_all, y_pred_all, average='macro')
            print( f'\nAVG of {fold+1} folds  |  Acc: {acc_all:.3f}    F1:{res[2]:.3f}       P: {res[0]:.3f}   R: {res[1]:.3f} \n')

        # show an overview of the performance
        df_2 = pd.DataFrame(list(results)).transpose()
        df_2['avg'] = df_2.mean(axis=1)
        df_2 = df_2.transpose()
        df_2['size'] = df_2['size'].astype(int)
        display(df_2)

        # put together the results of all 5-fold tests and save
        output_pred_csv_file_train = self.get_pred_csv_file(mode='train')
        df_out_proba.to_csv(output_pred_csv_file_train, index=False, float_format="%.3f")
        print(f'\noutput all {self.k}-fold test results to: "{output_pred_csv_file_train}"\n')

    def apply_KFold_model_to_new_sentences(self, mode='apply', unseen_data_file_name = None):

        columns = ['pmid', 'sentence', 'label'] # unique index
        _avail_class = [f'c{i}' for i in range(self.num_classes)]

        if unseen_data_file_name is None:
            unseen_data_file_name = self.fpath_unseen_data
        df = pd.read_csv(unseen_data_file_name)
        _avail_columns = [i for i in columns if i in df.columns]
        df = df[_avail_columns]
        print(f'all: {len(df):,}    unique sentences: {len(df.sentence.unique()):,}     papers: {len(df.pmid.unique()):,}')

        output_pred_file = self.get_pred_csv_file(mode=mode)
        print(f'generating >>> {output_pred_file}')

        y_test_all, y_pred_all = [], []
        results = []
        df_out_proba = None
        _averages = {col_name: np.mean for col_name in _avail_class}
        _aggs = {'confidence': np.mean,'winner': lambda x: stats.mode(x)[0]}

        for fold in range(self.k):
            model_file = self.get_model_bin_file(fold=fold)  # -1: indicating this is the model trained on all data
            print(f'\n- use trained model: {model_file}\n')
            model = load_model(model_file)
            model.eval_batch_size = 32

            if self.classifier_name == 'svm' or self.classifier_name == 'supcon':
                model_file_2 = model_file[:-4]+'.joblib'
                svm = load(model_file_2)
                y_proba, mid_output = model.predict_proba(df.sentence)
                y_pred = svm.predict(mid_output)
                del(mid_output)
            else:
                y_pred = model.predict(df.sentence)
                y_proba = model.predict_proba(df.sentence)
            del model

            # print(y_pred)

            y_pred_all += y_pred.tolist()

            tmp = pd.DataFrame(data=y_proba, columns=_avail_class)
            tmp['confidence'] = tmp.max(axis=1)
            tmp['winner'] = tmp.idxmax(axis=1)

            if self.classifier_name == 'svm' or self.classifier_name == 'supcon':
                tmp['winner_svm'] = y_pred
                _aggs.update({'winner_svm': lambda x: stats.mode(x)[0]})

            tmp['sentence'] = df['sentence'].tolist()
            tmp['label'] = df['label'] if 'label' in df.columns else [None]*len(df)
            tmp['pmid'] = df['pmid'].tolist() if 'pmid' in df.columns else None
            df_out_proba = tmp if df_out_proba is None else pd.concat((df_out_proba, tmp))

        # put together the results of all 5-fold tests and save
        _aggs.update(_averages)
        df_out = df_out_proba.groupby(_avail_columns).agg(_aggs).reset_index()

        df_out['winner_avg'] = df_out[_avail_class].idxmax(axis=1)
        df_out['winner_mode'] = df_out['winner']
        df_out['winner'] = [mode if type(mode)==str else avg for mode, avg in zip(df_out['winner_mode'], df_out['winner_avg'])]
        df_out.to_csv(output_pred_file, index=False, float_format="%.3f")
        print(f'\noutput prediction for {self.k}-fold test results to: "{output_pred_file}"\n')

    def apply_one_full_model_to_new_sentences(self, unseen_data_file_name = None):
        
        columns = ['pmid', 'sentence']
        if unseen_data_file_name is None:
            unseen_data_file_name = self.fpath_unseen_data
        df = pd.read_csv(unseen_data_file_name, usecols=columns)

        print(f'all: {len(df):,}    unique sentences: {len(df.sentence.unique()):,}     papers: {len(df.pmid.unique()):,}')

        output_pred_file = self.get_pred_csv_file('apply')
        print(f'generating >>> {output_pred_file}')

        model_file = self.get_model_bin_file(fold=-1)  # -1: indicating this is the model trained on all data
        print(f'\n- use trained model: {model_file}\n')
        model = load_model(model_file)
        model.eval_batch_size = 32

        if self.classifier_name == 'svm' or self.classifier_name == 'supcon':
            model_file_2 = model_file[:-4]+'.joblib'
            svm = load(model_file_2)
            y_prob, mid_output = model.predict_proba(df.sentence)
            y_pred = svm.predict(mid_output)
            del(mid_output)
        else:
            y_prob = model.predict_proba(df.sentence)

        df_out = pd.DataFrame(data=y_prob, columns=[f'c{i}' for i in range(self.num_classes)])
        df_out['confidence'] = df_out.max(axis=1)
        df_out['winner'] = df_out.idxmax(axis=1)

        if self.classifier_name == 'svm' or classifier_name == 'supcon':
            df_out['winner_svm'] = y_pred

        for col in columns:
            df_out[col] = df[col]

        df_out.to_csv(output_pred_file, index=False, float_format="%.3f")
        print(f'\n- output prediction to: {output_pred_file}\n')

    def evaluate_and_error_analysis(self):
        df = pd.read_csv(self.get_pred_csv_file(mode='train')) # -2: a flag indicating putting together the results on all folds
        df['pred'] = df['winner'].apply(lambda x:int(x[1])) # from c0->0, c1->1, c2->2, c3->3

        print('\nConfusion Matrix:\n')
        cm = confusion_matrix(df.label, df.pred)
        print(cm)

        print('\n\nClassification Report:\n')
        print(classification_report(df.label, df.pred))

        out = ["""
        <style>
            * {font-family:arial}
            body {width:900px;margin:auto}
            .wrong {color:red;}
            .hi1 {font-weight:bold}
        </style>
        <table cellpadding=10>
        """]

        row = f'<tr><th><th><th colspan=4>Predicted</tr>\n<tr><td><td>'
        for i in range(self.num_classes):
            row += f"<th>{self.label_name[i]}"
        for i in range(self.num_classes):
            row += f'''\n<tr>{'<th rowspan=4>Actual' if i==0 else ''}<th align=right>{self.label_name[i]}'''
            for j in range(self.num_classes):
                row += f'''<td align=right><a href='#link{i}{j}'>{cm[i][j]}</a></td>'''
        out.append(row + "</table>")

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                row = f"<div id=link{i}{j}><h2>{self.label_name[i]} => {self.label_name[j]}</h2><table cellpadding=10>"
                row += f'<tr><th><th>Sentence<th>Label<th>{self.label_name[0]}<th>{self.label_name[1]}<th>{self.label_name[2]}<th>{self.label_name[3]}<th>mark</tr>'
                out.append(row)

                df_ = df[(df.label==i) & (df.pred==j)]
                df_ = df_.sort_values('confidence', ascending=False)

                cnt = 0

                if self.num_classes == 4:
                    for c0, c1, c2, c3, sentence, label, pred in zip(df_.c0, df_.c1, df_.c2, df_.c3, df_.sentence, df_.label, df_.pred):
                        cnt += 1
                        mark = "" if label == pred else "<span class=wrong>oops</span>"
                        item = f"""<tr><th valign=top>{cnt}.
                                <td valign=top width=70%>{sentence}
                                <td valign=top>{self.label_name[label]}
                                <td valign=top class=hi{int(c0>max(c1,c2,c3))}>{c0:.2f}
                                <td valign=top class=hi{int(c1>max(c0,c2,c3))}>{c1:.2f}
                                <td valign=top class=hi{int(c2>max(c0,c1,c3))}>{c2:.2f}
                                <td valign=top class=hi{int(c3>max(c0,c1,c2))}>{c3:.2f}
                                <td valign=top>{mark}</tr>"""
                        out.append(item)
                elif self.num_classes == 5:
                    for c0, c1, c2, c3, c4, sentence, label, pred in zip(df_.c0, df_.c1, df_.c2, df_.c3, df_.c4, df_.sentence, df_.label, df_.pred):
                        cnt += 1
                        mark = "" if label == pred else "<span class=wrong>oops</span>"
                        item = f"""<tr><th valign=top>{cnt}.
                                <td valign=top width=70%>{sentence}
                                <td valign=top>{label_name[label]}
                                <td valign=top class=hi{int(c0>max(c1,c2,c3,c4))}>{c0:.2f}
                                <td valign=top class=hi{int(c1>max(c0,c2,c3,c4))}>{c1:.2f}
                                <td valign=top class=hi{int(c2>max(c0,c1,c3,c4))}>{c2:.2f}
                                <td valign=top class=hi{int(c3>max(c0,c1,c2,c4))}>{c3:.2f}
                                <td valign=top class=hi{int(c4>max(c0,c1,c2,c3))}>{c4:.2f}
                                <td valign=top>{mark}</tr>"""
                        out.append(item)
                elif self.num_classes == 2:
                    for c0, c1, sentence, label, pred in zip(df_.c0, df_.c1, df_.sentence, df_.label, df_.pred):
                        cnt += 1
                        mark = "" if label == pred else "<span class=wrong>oops</span>"
                        item = f"""<tr><th valign=top>{cnt}.
                                <td valign=top width=70%>{sentence}
                                <td valign=top>{label_name[label]}
                                <td valign=top class=hi{int(c0>max(c1))}>{c0:.2f}
                                <td valign=top class=hi{int(c1>max(c0))}>{c1:.2f}
                                <td valign=top>{mark}</tr>"""
                        out.append(item)
                else:
                    print('Currently error analysis only supports n=2,4,5 classes, please edit code.')

                out.append('</table></div>')

        html_file_output = '/var/www/html/a.html'
        html_file_output = '/tmp/a.html'
        with open(html_file_output, 'w') as fout:
            fout.write('\n'.join(out))
            print(f'\n- HTML file output to: "{html_file_output}"\n')

    def main(self, task):
        task_func = {
            'train_kfold': self.train_KFold_model,
            'train_one_full_model': self.train_one_full_model,
            'evaluate_and_error_analysis': self.evaluate_and_error_analysis,
            'apply_one_full_model_to_new_sentences': self.apply_one_full_model_to_new_sentences,
            'apply_KFold_model_to_new_sentences': self.apply_KFold_model_to_new_sentences
        }

        task_func[task]()


def run_one_full_round(bert_model_name, classifier_name, dataset_name, edits_name, data_dir, \
    k, epochs, extensions = '', run_base=False, run_5t=True, run_4t=True, 
    run_4t_rs=True, apply_on=None, train=True, test=True):

    if run_base:
        mt = ModelTrainTester(
            bert_model_name=bert_model_name, classifier_name=classifier_name, 
            dataset_name=dataset_name, train_file=f'{dataset_name}_base{extensions}',
            label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr'}, 
            k=k, epochs=epochs, data_dir=data_dir)
        if train: mt.main(task = 'train_kfold')
        if apply_on is not None: mt.fpath_unseen_data = apply_on
        if test: mt.main(task = 'apply_KFold_model_to_new_sentences')

    if run_5t:
        mt = ModelTrainTester(
            bert_model_name=bert_model_name, classifier_name=classifier_name, 
            dataset_name=dataset_name, train_file=f'{dataset_name}_{edits_name}_5t{extensions}',
            label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr', 4:'no_caus'}, 
            k=k, epochs=epochs, data_dir=data_dir)
        if train: mt.main(task = 'train_kfold')
        if apply_on is not None: mt.fpath_unseen_data = apply_on
        if test: mt.main(task = 'apply_KFold_model_to_new_sentences')

    if run_4t:
        mt = ModelTrainTester(
            bert_model_name=bert_model_name, classifier_name=classifier_name, 
            dataset_name=dataset_name, train_file=f'{dataset_name}_{edits_name}_4t{extensions}',
            label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr'}, 
            k=k, epochs=epochs, data_dir=data_dir)
        if train: mt.main(task = 'train_kfold')
        if apply_on is not None: mt.fpath_unseen_data = apply_on
        if test: mt.main(task = 'apply_KFold_model_to_new_sentences')

    if run_4t_rs:
        mt = ModelTrainTester(
            bert_model_name=bert_model_name, classifier_name=classifier_name, 
            dataset_name=dataset_name, train_file=f'{dataset_name}_{edits_name}_4t_rs{extensions}',
            label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr'}, 
            k=k, epochs=epochs, data_dir=data_dir)
        if train: mt.main(task = 'train_kfold')
        if apply_on is not None: mt.fpath_unseen_data = apply_on
        if test: mt.main(task = 'apply_KFold_model_to_new_sentences')


if __name__ == "__main__":
    tic = time.time()

    classifier_name='mlp' #

    ##### regular edits #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='edits', data_dir='data', extensions = '', 
        k=5, epochs=5, run_base=True
    )

    ##### shorten originals + shorten edits #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='oriedits', data_dir='data', extensions = '_shorten', 
        k=5, epochs=5, run_base=True
    )

    ##### multiples edits #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='edits', data_dir='data', extensions = '_multiples', 
        k=5, epochs=5, run_base=False
    )

    ##### shorten edits #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='edits', data_dir='data', extensions = '_shorten', 
        k=5, epochs=5, run_base=False
    )

    ##### t5para edits #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='edits', data_dir='data', extensions = '_t5para', 
        k=5, epochs=5, run_base=False
    )

    ##### synonyms edits #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='edits', data_dir='data', extensions = '_synonyms', 
        k=5, epochs=5, run_base=False
    )

    ##### mask originals + mask edits #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='oriedits', data_dir='data', extensions = '_mask', 
        k=5, epochs=5, run_base=True
    )

    ##### base on negation #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='oriedits', data_dir='data', extensions = '_removed', 
        k=5, epochs=5, run_base=True, apply_on = "data/pubmed_edits_removed_test.csv"
    )
    
    ##### regular edits (2to1) #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='2to1_edits', data_dir='data', extensions = '', 
        k=5, epochs=5, run_base=False, run_5t=False
    )

    ##### shorten edits (2to1) #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='2to1_edits', data_dir='data', extensions = '_shorten', 
        k=5, epochs=5, run_base=False, run_5t=False
    )

    ##### shorten originals + shorten edits (2to1) #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='2to1_oriedits', data_dir='data', extensions = '_shorten', 
        k=5, epochs=5, run_base=False, run_5t=False
    )

    ##### multiples edits (2to1) #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='2to1_edits', data_dir='data', extensions = '_multiples', 
        k=5, epochs=5, run_base=False, run_5t=False
    )

    ##### mask originals + mask edits (2to1) #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='2to1_oriedits', data_dir='data', extensions = '_mask', 
        k=5, epochs=5, run_base=False, run_5t=False
    )

    ##### t5para edits (2to1) #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='2to1_edits', data_dir='data', extensions = '_t5para', 
        k=5, epochs=5, run_base=False, run_5t=False
    )

    ##### synonyms edits (2to1) #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='2to1_edits', data_dir='data', extensions = '_synonyms', 
        k=5, epochs=5, run_base=False, run_5t=False
    )

    ##### base on 2to1 #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='2to1_oriedits', data_dir='data', extensions = '_removed', 
        k=5, epochs=5, run_base=True, run_5t=False,
        apply_on = "data/pubmed_2to1_edits_removed_test.csv"
    )

    ##### regular edits (all) #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='all_edits', data_dir='data', extensions = '', 
        k=5, epochs=5, run_base=False
    )

    ##### regular edits (mix) #####

    run_one_full_round(
        bert_model_name='biobert', classifier_name=classifier_name, 
        dataset_name='pubmed', edits_name='mix01_edits', data_dir='data', extensions = '', 
        k=5, epochs=5, run_base=False
    )

    # ##### shorten edits (mix) #####

    # run_one_full_round(
    #     bert_model_name='biobert', classifier_name=classifier_name, 
    #     dataset_name='pubmed', edits_name='mix02_oriedits', data_dir='data', extensions = '_shorten', 
    #     k=5, epochs=5, run_base=False, run_5t=False, run_4t=False
    # )

    # ##### predict base/edit labels #####
    # mt = ModelTrainTester(
    #     bert_model_name='biobert', classifier_name='mlp', dataset_name='pubmed', 
    #     train_file='pubmed_mix01_edits_4t_rs_betype',
    #     label_name = {0:'base', 1:'edit'}, 
    #     k=5, epochs=5, data_dir='data', label_list={0: 0, 1: 1})
    # mt.main(task = 'train_kfold')
    # mt.fpath_unseen_data = "data/pubmed_mix01_edits_4t_rs_betype_test.csv"
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # ##### ood_testing #####
    # ##### regular edits #####
    classifier_name = 'svm'
    dataset = 'altlex'
    bert_model_name = 'biobert'

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, dataset_name=dataset, 
    #     train_file=f'{dataset}_base', k=5, epochs=5, data_dir='data',
    #     label_name = {0:'not_causal', 1:'causal'}, label_list={0: 0, 1: 1})
    # mt.main(task = 'train_kfold')
    # mt.fpath_unseen_data = "data/altlex_base.csv"
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, dataset_name=dataset, 
    #     train_file=f'{dataset}_edits_4t', k=5, epochs=5, data_dir='data',
    #     label_name = {0:'not_causal', 1:'causal'}, label_list={0: 0, 1: 1})
    # mt.main(task = 'train_kfold')
    # mt.fpath_unseen_data = "data/altlex_base.csv"
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, dataset_name=dataset, 
    #     train_file=f'{dataset}_edits_4t_rs', k=5, epochs=5, data_dir='data',
    #     label_name = {0:'not_causal', 1:'causal'}, label_list={0: 0, 1: 1})
    # mt.main(task = 'train_kfold')
    # mt.fpath_unseen_data = "data/altlex_base.csv"
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # ##### shorten edits #####

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, dataset_name=dataset, 
    #     train_file=f'{dataset}_edits_4t_shorten', k=5, epochs=5, data_dir='data',
    #     label_name = {0:'not_causal', 1:'causal'}, label_list={0: 0, 1: 1})
    # mt.main(task = 'train_kfold')
    # mt.fpath_unseen_data = "data/altlex_base.csv"
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, dataset_name=dataset, 
    #     train_file=f'{dataset}_edits_4t_rs_shorten', k=5, epochs=5, data_dir='data',
    #     label_name = {0:'not_causal', 1:'causal'}, label_list={0: 0, 1: 1})
    # mt.main(task = 'train_kfold')
    # mt.fpath_unseen_data = "data/altlex_base.csv"
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # ##### multiples edits #####

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, dataset_name=dataset, 
    #     train_file=f'{dataset}_edits_4t_multiples', k=5, epochs=5, data_dir='data',
    #     label_name = {0:'not_causal', 1:'causal'}, label_list={0: 0, 1: 1})
    # mt.main(task = 'train_kfold')
    # # mt.fpath_unseen_data = "data/altlex_base.csv"
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, dataset_name=dataset, 
    #     train_file=f'{dataset}_edits_4t_rs_multiples', k=5, epochs=5, data_dir='data',
    #     label_name = {0:'not_causal', 1:'causal'}, label_list={0: 0, 1: 1})
    # mt.main(task = 'train_kfold')
    # # mt.fpath_unseen_data = "data/altlex_base.csv"
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    ##### regular edits (supcon) #####

    bert_model_name='biobert'
    classifier_name='supcon'
    dataset_name='pubmed'
    data_dir='data'
    extensions = ''
    k=5
    epochs=20
    supcon_aug_method = 't5para' #'synonyms' # 'shorten'

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, 
    #     dataset_name=dataset_name, train_file=f'{dataset_name}_base{extensions}',
    #     label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr'}, 
    #     k=k, epochs=epochs, data_dir=data_dir,
    #     label_list={0: 0, 1: 1, 2: 2, 3: 3}, supcon_incl_negs=False, 
    #     supcon_aug_method=supcon_aug_method)
    # mt.main(task = 'train_kfold')
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # req eps != 0 = 1e-2
    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, 
    #     dataset_name=dataset_name, train_file=f'{dataset_name}_base{extensions}',
    #     label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr'}, 
    #     k=k, epochs=epochs, data_dir=data_dir,
    #     label_list={0: 0, 1: 1, 2: 2, 3: 3}, supcon_incl_negs=False, 
    #     loss_method='ce+supcon', supcon_aug_method=supcon_aug_method)
    # mt.main(task = 'train_kfold')
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, 
    #     dataset_name=dataset_name, train_file=f'{dataset_name}_base{extensions}',
    #     label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr', 4:'no_caus'}, 
    #     k=k, epochs=epochs, data_dir=data_dir,
    #     label_list={0: 0, 1: 1, 2: 2, 3: 3, 4: 4}, supcon_incl_negs=True, 
    #     supcon_edit_name='_edits_5t', supcon_aug_method=supcon_aug_method)
    # mt.main(task = 'train_kfold')
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, 
    #     dataset_name=dataset_name, train_file=f'{dataset_name}_base{extensions}',
    #     label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr', 4:'no_caus'}, 
    #     k=k, epochs=epochs, data_dir=data_dir, loss_method='ce+triplet',
    #     label_list={0: 0, 1: 1, 2: 2, 3: 3, 4: 4}, supcon_incl_negs=True, 
    #     supcon_edit_name='_edits_5t', supcon_aug_method=supcon_aug_method)
    # mt.main(task = 'train_kfold')
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, 
    #     dataset_name=dataset_name, train_file=f'{dataset_name}_base{extensions}',
    #     label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr'}, 
    #     k=k, epochs=epochs, data_dir=data_dir,
    #     label_list={0: 0, 1: 1, 2: 2, 3: 3}, supcon_incl_negs=True, 
    #     supcon_edit_name='_2to1_edits_4t', supcon_aug_method=supcon_aug_method)
    # mt.main(task = 'train_kfold')
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, 
    #     dataset_name=dataset_name, train_file=f'{dataset_name}_base{extensions}',
    #     label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr'}, 
    #     k=k, epochs=epochs, data_dir=data_dir, loss_method='ce+triplet',
    #     label_list={0: 0, 1: 1, 2: 2, 3: 3}, supcon_incl_negs=True, 
    #     supcon_edit_name='_2to1_edits_4t', supcon_aug_method=supcon_aug_method)
    # mt.main(task = 'train_kfold')
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    #####

    supcon_aug_method = 'shorten'
    supcon_aug_method = 'synonyms'

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, 
    #     dataset_name=dataset_name, train_file=f'{dataset_name}_base{extensions}',
    #     label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr'}, 
    #     k=k, epochs=epochs, data_dir=data_dir,
    #     label_list={0: 0, 1: 1, 2: 2, 3: 3}, supcon_incl_negs=False, 
    #     supcon_aug_method=supcon_aug_method)
    # mt.main(task = 'train_kfold')
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, 
    #     dataset_name=dataset_name, train_file=f'{dataset_name}_base{extensions}',
    #     label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr'}, 
    #     k=k, epochs=epochs, data_dir=data_dir,
    #     label_list={0: 0, 1: 1, 2: 2, 3: 3}, supcon_incl_negs=False, 
    #     loss_method='ce+supcon', supcon_aug_method=supcon_aug_method)
    # mt.main(task = 'train_kfold')
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, 
    #     dataset_name=dataset_name, train_file=f'{dataset_name}_base{extensions}',
    #     label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr', 4:'no_caus'}, 
    #     k=k, epochs=epochs, data_dir=data_dir,
    #     label_list={0: 0, 1: 1, 2: 2, 3: 3, 4: 4}, supcon_incl_negs=True, 
    #     supcon_edit_name='_edits_5t', supcon_aug_method=supcon_aug_method)
    # mt.main(task = 'train_kfold')
    # mt.main(task = 'apply_KFold_model_to_new_sentences')
    
    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, 
    #     dataset_name=dataset_name, train_file=f'{dataset_name}_base{extensions}',
    #     label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr', 4:'no_caus'}, 
    #     k=k, epochs=epochs, data_dir=data_dir, loss_method='ce+triplet',
    #     label_list={0: 0, 1: 1, 2: 2, 3: 3, 4: 4}, supcon_incl_negs=True, 
    #     supcon_edit_name='_edits_5t', supcon_aug_method=supcon_aug_method)
    # mt.main(task = 'train_kfold')
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, 
    #     dataset_name=dataset_name, train_file=f'{dataset_name}_base{extensions}',
    #     label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr', 4:'no_caus'}, 
    #     k=k, epochs=epochs, data_dir=data_dir, loss_method='ce+supcon',
    #     label_list={0: 0, 1: 1, 2: 2, 3: 3, 4: 4}, supcon_incl_negs=True, 
    #     supcon_edit_name='_edits_5t', supcon_aug_method=supcon_aug_method)
    # mt.main(task = 'train_kfold')
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, 
    #     dataset_name=dataset_name, train_file=f'{dataset_name}_base{extensions}',
    #     label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr'}, 
    #     k=k, epochs=epochs, data_dir=data_dir,
    #     label_list={0: 0, 1: 1, 2: 2, 3: 3}, supcon_incl_negs=True, 
    #     supcon_edit_name='_2to1_edits_4t', supcon_aug_method=supcon_aug_method)
    # mt.main(task = 'train_kfold')
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, 
    #     dataset_name=dataset_name, train_file=f'{dataset_name}_base{extensions}',
    #     label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr'}, 
    #     k=k, epochs=epochs, data_dir=data_dir, loss_method='ce+triplet',
    #     label_list={0: 0, 1: 1, 2: 2, 3: 3}, supcon_incl_negs=True, 
    #     supcon_edit_name='_2to1_edits_4t', supcon_aug_method=supcon_aug_method)
    # mt.main(task = 'train_kfold')
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    # mt = ModelTrainTester(
    #     bert_model_name=bert_model_name, classifier_name=classifier_name, 
    #     dataset_name=dataset_name, train_file=f'{dataset_name}_base{extensions}',
    #     label_name = {0:'none', 1:'causal', 2:'cond', 3:'corr'}, 
    #     k=k, epochs=epochs, data_dir=data_dir, loss_method='ce+supcon',
    #     label_list={0: 0, 1: 1, 2: 2, 3: 3}, supcon_incl_negs=True, 
    #     supcon_edit_name='_2to1_edits_4t', supcon_aug_method=supcon_aug_method)
    # mt.main(task = 'train_kfold')
    # mt.main(task = 'apply_KFold_model_to_new_sentences')

    print(f'time used: {time.time()-tic:.0f} seconds')
