import numpy as np
import pandas as pd
import re, time, os, random, pickle, csv
from collections import Counter
SEED = 1234
np.random.seed(SEED)
random.seed(SEED)
pd.options.mode.chained_assignment = None  # default='warn'


class DataGenerator(object):
    def __init__(
        self, directory_name, edits_file_name, dataset_name, 
        edits_name = '_edits', num_labels_name = '', extensions_name = '', base_extensions_name = '',
        filter_examples_by = None, verbose = False):

        # labels naming format
        self.no_relat_label = 0
        self.causal_label = 1
        self.cond_causal_label = 2
        self.correlational_label = 3
        self.not_causal = 4

        # options
        self.verbose = verbose

        # file naming format
        self.directory_name = directory_name
        self.dataset_name = dataset_name
        self.edits_name = edits_name
        self.num_labels_name = num_labels_name
        self.num_labels = int(re.findall('\d+',num_labels_name)[0])
        self.extensions_name = extensions_name
        
        self.base_csv_file_name = "{}{}_base{}.csv".format(directory_name, dataset_name, base_extensions_name)
        if edits_file_name is None:
            self.edits_csv_file_name = None
        elif type(edits_file_name)==list:
            self.edits_csv_file_name = [directory_name + i for i in edits_file_name]
        else:
            self.edits_csv_file_name = directory_name + edits_file_name
        self.data_csv_file_name = "{}{}{}{}{}.csv".format(directory_name, dataset_name, edits_name, num_labels_name, extensions_name)

        # filters
        self.filter_examples_by = filter_examples_by


    def get_fill_label(self, edits_csv_file_name):
        edit_conv_type = re.findall('(\d+to\d+)', edits_csv_file_name)
        if len(edit_conv_type)==0:
            fill_value = self.not_causal
        else:
            fill_value = int(edit_conv_type[0][-1])
        
        # if error
        if fill_value >= self.num_labels:
            if fill_value == self.not_causal and self.num_labels == 4:
                fill_value = 0
            else:
                fill_value = input(f'For {edit_conv_type}, what new label value to replace by? (Note {self.num_labels_name})')
        return fill_value


    def combine_edits_and_original(self):

        print('generating >>> {}{}{}{} ...'.format(
            self.dataset_name, self.edits_name, self.num_labels_name, self.extensions_name))

        base = pd.read_csv(self.base_csv_file_name)
        base['source'] = 'base'

        if self.edits_csv_file_name is None:
            data = base
        else:
            if type(self.edits_csv_file_name)==list:
                for i, pth in enumerate(self.edits_csv_file_name):
                    print(f'appending {pth} ...')
                    _edits = pd.read_csv(pth)
                    _edits['label'] = self.get_fill_label(pth)
                    if i==0:
                        edits = _edits
                    else:
                        edits = pd.concat([edits, _edits], axis=0)
                    del(_edits)
            else:
                edits = pd.read_csv(self.edits_csv_file_name)
                edits['label'] = self.get_fill_label(self.edits_csv_file_name)
            edits['source'] = 'edits'
            data = pd.concat([base, edits], axis=0)

        # remove clean duplicates
        data = data.drop_duplicates(subset=['sentence', 'label'])
        # resolve duplicates with differring labels
        if self.filter_examples_by is None:    
            data = self.custom_remove_duplicates(data)
        else:
            filter_by_df = pd.read_csv(self.directory_name + self.filter_examples_by, usecols=['pmid'])
            # filter original base data by deduplicated index
            data['pmid_base'] = data['pmid'].apply(lambda x: str(x).split('_')[0])
            data = data[
                ((data['pmid_base'].astype(str).isin(filter_by_df['pmid'].astype(str))) 
                & (data['source'] == 'base')) 
                | (data['source'] == 'edits')
                ]
            if type(self.edits_csv_file_name)==list:
                # remove duplicates amongst edits
                data = self.custom_remove_duplicates(data)

        # random sample distribution
        if '_rs' in self.extensions_name:
            data = self.random_sample_distribution(data, base)

        data.to_csv(self.data_csv_file_name, index=False)


    def custom_remove_duplicates(self, data):

        data = data.reset_index(drop=True).copy()
        data['pmid'] = data['pmid'].astype(str)

        # manual: for pubmed
        dupl_ids = [1864]
        for del_id in dupl_ids:
            data = data[data['pmid']!= del_id]

        # with rule: e.g. not-causal > causal
        duplicates = data[data.duplicated(subset=['sentence'], keep=False)]
        print('number of duplicate examples to remove: {}'.format(len(duplicates)+len(dupl_ids)))

        duplicates['count'] = 1
        duplicates = pd.pivot_table(duplicates[['count','sentence','label']], index=["sentence"], columns=["label"]).reset_index()

        # set column names as 'sentence', 'labels...'
        keep_columns = ['sentence']+list(duplicates.columns.get_level_values(1))[1:]
        duplicates.columns = keep_columns
        missing_columns = [i for i in range(0,self.num_labels+1) if i not in duplicates.columns]
        for i in missing_columns:
            duplicates[i] = 0
        manually_check = []
        duplicates = duplicates.fillna(0)
        
        for ix in range(0, len(duplicates)):
            """
            causal > cond_causal
            causal < not_causal
            causal < edits_w_no_relation  
            """
            if (int(duplicates.loc[ix, self.causal_label]) >= 1) and \
            ((int(duplicates.loc[ix, self.cond_causal_label]) >= 1) or \
            (int(duplicates.loc[ix, self.not_causal]) >= 1)):

                dupl_ids = data[data['sentence'] == duplicates.loc[ix, 'sentence']].index
                # keep id from back (i.e. edits "not causal" tag)
                for del_id in dupl_ids[:-1]:
                    data = data[data.index!= del_id]

            elif (int(duplicates.loc[ix, self.causal_label]) >= 1) and \
            (('4t' in self.num_labels_name) and (int(duplicates.loc[ix, self.no_relat_label]) >= 1)):

                dupl_ids = data[data['sentence'] == duplicates.loc[ix, 'sentence']].index
                # check that duplicate is indeed an edit, else skip duplicate removal
                if any(st in data.loc[dupl_ids[-1], 'pmid'] for st in ['alt', 'edt']):
                    # keep id from back (i.e. edits "not causal" tag)
                    for del_id in dupl_ids[:-1]:
                        data = data[data.index!= del_id]

            else:
                manually_check.append(ix)

        # check no more duplicates
        # print(data[data.duplicated(subset=['sentence'], keep=False)])
        try:
            assert(len(data[data.duplicated(subset=['sentence'], keep=False)])==0)
            assert(len(manually_check)==0)
            print('removed all duplicates with no exceptions')
        except:
            print('unable to deduplicate the following:')
            print(data[data.duplicated(subset=['sentence'], keep=False)])
        return data

    
    def random_sample_distribution(self, data, base):
        data_label_counts = Counter(data['label'])
        base_label_counts = Counter(base['label'])
        for i in data_label_counts:
            # if number of labels differ, take random sample
            if data_label_counts[i]>base_label_counts[i]:
                print('rs: converting label "{}" from n={} to {}'.format(i, data_label_counts[i], base_label_counts[i]))
                replace = data[data['label']==i].sample(n=base_label_counts[i], random_state=SEED)
                data = data[data['label']!=i]
                data = data.append(replace)
        #     elif data_label_counts[i]<base_label_counts[i]:
        #         print('Converting label "{}" from n={} to {}'.format(i, data_label_counts[i], base_label_counts[i]))
        #         replace = data[data['label']==i].sample(n=base_label_counts[i]-data_label_counts[i], random_state=SEED)
        #         data = data.append(replace)
        return data


def run_one_full_round(directory_name, edits_file_name, dataset_name, edits_name, \
    extensions, base_extensions, filter_examples_by, run_5t=True, run_4t=True, run_4t_rs=True):

    if run_5t:
        dg = DataGenerator(
            directory_name = directory_name, edits_file_name = edits_file_name, 
            dataset_name = dataset_name, edits_name = edits_name, num_labels_name = '_5t', 
            extensions_name = f'{extensions}', base_extensions_name = base_extensions,
            filter_examples_by = filter_examples_by, verbose = False)
        dg.combine_edits_and_original()

        if filter_examples_by is None:
            filter_examples_by = os.path.basename(dg.data_csv_file_name)
            print(f'updating filter_examples_by: {filter_examples_by}')

    if run_4t:
        dg = DataGenerator(
            directory_name = directory_name, edits_file_name = edits_file_name, 
            dataset_name = dataset_name, edits_name = edits_name, num_labels_name = '_4t', 
            extensions_name = f'{extensions}', base_extensions_name = base_extensions,
            filter_examples_by = filter_examples_by, verbose = False)
        dg.combine_edits_and_original()

        if filter_examples_by is None:
            filter_examples_by = os.path.basename(dg.data_csv_file_name)
            print(f'updating filter_examples_by: {filter_examples_by}')

    if run_4t_rs:
        dg = DataGenerator(
            directory_name = directory_name, edits_file_name = edits_file_name, 
            dataset_name = dataset_name, edits_name = edits_name, num_labels_name = '_4t', 
            extensions_name = f'_rs{extensions}', base_extensions_name = base_extensions,
            filter_examples_by = filter_examples_by, verbose = False)
        dg.combine_edits_and_original()


if __name__ == '__main__':
    tic = time.time()

    # ##### regular edits #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "pubmed_edits.csv",
    #     filter_examples_by = None,
    #     dataset_name = 'pubmed',
    #     edits_name = '_edits',
    #     extensions = '',
    #     base_extensions = ''
    #     )

    # ##### multiple edits #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "pubmed_edits_multiples.csv",
    #     filter_examples_by = "pubmed_edits_5t.csv",
    #     dataset_name = 'pubmed',
    #     edits_name = '_edits',
    #     extensions = '_multiples',
    #     base_extensions = ''
    #     )

    # ##### shorten edits #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "pubmed_edits_shorten.csv",
    #     filter_examples_by = "pubmed_edits_5t.csv",
    #     dataset_name = 'pubmed',
    #     edits_name = '_edits',
    #     extensions = '_shorten',
    #     base_extensions = ''
    #     )

    # ##### synonyms edits #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "pubmed_edits_synonyms.csv",
    #     filter_examples_by = "pubmed_edits_5t.csv",
    #     dataset_name = 'pubmed',
    #     edits_name = '_edits',
    #     extensions = '_synonyms',
    #     base_extensions = ''
    #     )

    # ##### t5para edits #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "pubmed_edits_t5para.csv",
    #     filter_examples_by = "pubmed_edits_5t.csv",
    #     dataset_name = 'pubmed',
    #     edits_name = '_edits',
    #     extensions = '_t5para',
    #     base_extensions = ''
    #     )

    # ##### shorten originals + shorten edits #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "pubmed_edits_shorten.csv",
    #     filter_examples_by = "pubmed_edits_5t.csv",
    #     dataset_name = 'pubmed',
    #     edits_name = '_oriedits',
    #     extensions = '_shorten',
    #     base_extensions = '_shorten'
    #     )
    
    # ##### mask originals + mask edits #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "pubmed_edits_mask.csv",
    #     filter_examples_by = "pubmed_edits_5t.csv",
    #     dataset_name = 'pubmed',
    #     edits_name = '_oriedits',
    #     extensions = '_mask',
    #     base_extensions = '_mask'
    #     )

    # ##### removed originals + removed edits #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "pubmed_edits_removed.csv",
    #     filter_examples_by = None,
    #     dataset_name = 'pubmed',
    #     edits_name = '_oriedits',
    #     extensions = '_removed',
    #     base_extensions = '_removed'
    #     )

    ##### regular edits (2to4) #####
    run_one_full_round(
        directory_name = "D:/50 CausalCF/data/",
        edits_file_name = "pubmed_2to4_edits.csv",
        filter_examples_by = None,
        dataset_name = 'pubmed',
        edits_name = '_2to4_edits',
        extensions = '',
        base_extensions = '',
        run_5t = False
        )

    # ##### multiple edits (2to1) #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "pubmed_2to1_edits_multiples.csv",
    #     filter_examples_by = "pubmed_2to1_edits_4t.csv",
    #     dataset_name = 'pubmed',
    #     edits_name = '_2to1_edits',
    #     extensions = '_multiples',
    #     base_extensions = '',
    #     run_5t = False
    #     )

    # ##### shorten edits (2to1) #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "pubmed_2to1_edits_shorten.csv",
    #     filter_examples_by = "pubmed_2to1_edits_4t.csv",
    #     dataset_name = 'pubmed',
    #     edits_name = '_2to1_edits',
    #     extensions = '_shorten',
    #     base_extensions = '',
    #     run_5t = False
    #     )

    # ##### synonyms edits (2to1) #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "pubmed_2to1_edits_synonyms.csv",
    #     filter_examples_by = "pubmed_2to1_edits_4t.csv",
    #     dataset_name = 'pubmed',
    #     edits_name = '_2to1_edits',
    #     extensions = '_synonyms',
    #     base_extensions = '',
    #     run_5t = False
    #     )
    
    # ##### t5para edits (2to1) #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "pubmed_2to1_edits_t5para.csv",
    #     filter_examples_by = "pubmed_2to1_edits_4t.csv",
    #     dataset_name = 'pubmed',
    #     edits_name = '_2to1_edits',
    #     extensions = '_t5para',
    #     base_extensions = '',
    #     run_5t = False
    #     )
    
    # ##### shorten originals + shorten edits (2to1) #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "pubmed_2to1_edits_shorten.csv",
    #     filter_examples_by = "pubmed_2to1_edits_4t.csv",
    #     dataset_name = 'pubmed',
    #     edits_name = '_2to1_oriedits',
    #     extensions = '_shorten',
    #     base_extensions = '_shorten',
    #     run_5t = False
    #     )

    # ##### mix: shorten base + 1to4 shorten + 2to1 shorten #####

    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = ["pubmed_edits_shorten.csv", "pubmed_2to1_edits_shorten.csv"],
    #     filter_examples_by = "pubmed_2to1_edits_4t.csv",
    #     dataset_name = 'pubmed',
    #     edits_name = '_mix02_oriedits',
    #     extensions = '_shorten',
    #     base_extensions = '_shorten'
    #     )
    
    # ##### mask originals + mask edits (2to1) #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "pubmed_2to1_edits_mask.csv",
    #     filter_examples_by = "pubmed_2to1_edits_4t.csv",
    #     dataset_name = 'pubmed',
    #     edits_name = '_2to1_oriedits',
    #     extensions = '_mask',
    #     base_extensions = '_mask',
    #     run_5t = False
    #     )

    # ##### removed originals + removed edits #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "pubmed_2to1_edits_removed.csv",
    #     filter_examples_by = None,
    #     dataset_name = 'pubmed',
    #     edits_name = '_2to1_oriedits',
    #     extensions = '_removed',
    #     base_extensions = '_removed'
    #     )

    ##### all: base + 1to4 + 2to1 + 2to4 edits #####

    run_one_full_round(
        directory_name = "D:/50 CausalCF/data/",
        edits_file_name = ["pubmed_edits.csv", "pubmed_2to1_edits.csv", "pubmed_2to4_edits.csv"],
        filter_examples_by = None,
        dataset_name = 'pubmed',
        edits_name = '_all_edits',
        extensions = '',
        base_extensions = ''
        )

    # ##### mix: base + 1to4 shorten + 2to1 edits #####

    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = ["pubmed_edits_shorten.csv", "pubmed_2to1_edits.csv"],
    #     filter_examples_by = None,
    #     dataset_name = 'pubmed',
    #     edits_name = '_mix01_edits',
    #     extensions = '',
    #     base_extensions = ''
    #     )

    

    ##### other datasets #####
    # ##### regular edits #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "scite_edits.csv",
    #     filter_examples_by = None,
    #     dataset_name = 'scite',
    #     edits_name = '_edits',
    #     extensions = '',
    #     base_extensions = '',
    #     run_5t=False
    #     )

    # #### shorten edits #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "scite_edits_shorten.csv",
    #     filter_examples_by = "scite_edits_4t.csv",
    #     dataset_name = 'scite',
    #     edits_name = '_edits',
    #     extensions = '_shorten',
    #     base_extensions = ''
    #     )

    # #### multiples edits #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "scite_edits_multiples.csv",
    #     filter_examples_by = "scite_edits_4t.csv",
    #     dataset_name = 'scite',
    #     edits_name = '_edits',
    #     extensions = '_multiples',
    #     base_extensions = ''
    #     )

    # ##### regular edits #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "altlex_edits.csv",
    #     filter_examples_by = None,
    #     dataset_name = 'altlex',
    #     edits_name = '_edits',
    #     extensions = '',
    #     base_extensions = '',
    #     run_5t=False
    #     )

    # #### shorten edits #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "altlex_edits_shorten.csv",
    #     filter_examples_by = "altlex_edits_4t.csv",
    #     dataset_name = 'altlex',
    #     edits_name = '_edits',
    #     extensions = '_shorten',
    #     base_extensions = ''
    #     )

    # #### multiples edits #####
    # run_one_full_round(
    #     directory_name = "D:/50 CausalCF/data/",
    #     edits_file_name = "altlex_edits_multiples.csv",
    #     filter_examples_by = "altlex_edits_4t.csv",
    #     dataset_name = 'altlex',
    #     edits_name = '_edits',
    #     extensions = '_multiples',
    #     base_extensions = ''
    #     )

    print(f'time used: {time.time()-tic:.0f} seconds')
