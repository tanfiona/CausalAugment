import os, sys, time, re, pickle, random
import numpy as np
import pandas as pd
from pandas.core.common import flatten
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag, stem, tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.metrics import edit_distance
import spacy
from spacy.tokenizer import Tokenizer
from pattern.en import conjugate as conjugate_en
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
random.seed(123)

# WIP TO DO: Split codes into different scripts

# load defaults
verbose = False
nlp = spacy.load('en_core_web_sm')
def custom_tokenizer(nlp, prefix_re): return Tokenizer(nlp.vocab, prefix_search=prefix_re.search)
nlp.tokenizer = custom_tokenizer(nlp, re.compile('''^\$[a-zA-Z0-9]'''))
wnl = WordNetLemmatizer()
_RE_COMBINE_WHITESPACE = re.compile(r"\s+")
modal_verb_dict = {
    'could': 'would',
    'should': 'would',
    'would': 'would',
    'can': 'will',
    'may': 'will',
    'might': 'will',
    'will': 'will'
}

# not sure why, first round of pattern.en always fails
try:
    conjugate_en(verb='testing',tense='present',number='singular')
except:
    pass

# for paraphrasers, comment out if not using
MAX_LENGTH = 256
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(123)

model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_paraphraser')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f'using device: {device}')
model = model.to(device)


########## negation functions

def if_consequetive(l):
    return sorted(l) == list(range(min(l), max(l)+1))
    

def rework_start_end_ids_by_sentid(_start, _end, itemdict_text, sentid2tid):
    if sentid2tid is not None:
        # print('start:{} | end:{}'.format(_start, _end))
        start_id = int(itemdict_text[_start-1][1])
        end_id = int(itemdict_text[_end-1][1])
        _start = sentid2tid[start_id]
        if (start_id+1 == len(sentid2tid)) or (end_id+1 == len(sentid2tid)): # 0,1,2 , len=3, 
            _end = len(itemdict_text)+1
        else:    
            _end = sentid2tid[end_id+1]
        # print('by sentences - start:{} | end:{}'.format(_start, _end))
    return _start, _end


def show_plot_link_causes(dictionary, item, num_words=10, verbose=True, sentid2tid=None, return_all=False):
    
    if verbose:
        print('-'*10, item, '-'*10)

    itemdict = dictionary[item]
    plot = itemdict['plotlink_cse']
    outdict = {}
    
    for ix, ((source, target), items) in enumerate(plot.items()):

        text = [i[3] for i in itemdict['text']]
        source = [int(i) for i in source.split('_')]
        target = [int(i) for i in target.split('_')]

        if if_consequetive(source):
            source = [min(source), max(source)]
        if if_consequetive(target):
            target = [min(target), max(target)]
        
        text[source[0]-1] = "<SOURCE> " + text[source[0]-1]
        text[source[1]-1] = text[source[1]-1] + " </SOURCE>"
        text[target[0]-1] = "<TARGET> " + text[target[0]-1]
        text[target[1]-1] = text[target[1]-1] + " </TARGET>"
        
        signal = items[1]
        if signal == '' or signal == 'null':
            signal = [None]
            _start = max(min(source+target)-num_words,0)
            _end = min(max(source+target)+num_words, len(text))
            _start, _end = rework_start_end_ids_by_sentid(_start, _end, itemdict['text'], sentid2tid)

            if return_all:
                outdict[ix] = {
                    'signal_id': signal,
                    'source_id': source,
                    'target_id': target,
                    'eg_text': ' '.join(text[_start-1:_end-1])
                    }

        else:
            signal = itemdict['m2t'][signal]
            signal = [int(i) for i in signal.split('_')]
            if if_consequetive(signal):
                signal = [min(signal), max(signal)]
            text[signal[0]-1] = "<SIGNAL> " + text[signal[0]-1]
            text[signal[1]-1] = text[signal[1]-1] + " </SIGNAL>"
            _start = max(min(source+target+signal)-num_words,0)
            _end = min(max(source+target+signal)+num_words, len(text))
            _start, _end = rework_start_end_ids_by_sentid(_start, _end, itemdict['text'], sentid2tid)
            
            outdict[ix] = {
                'signal_id': signal,
                'source_id': source,
                'target_id': target,
                'eg_text': ' '.join(text[_start-1:_end-1])
            }
        
        if verbose:
            print(' '.join(text[_start-1:_end-1]))
        
    return outdict


def show_action_causative(dictionary, item, num_words=10, verbose=True, sentid2tid=None):
    
    if verbose:
        print('-'*10, item, '-'*10)

    itemdict = dictionary[item]
    mark = itemdict['markables_cse']
    _text = [i[3] for i in itemdict['text']]
    
    for keys, items in mark.items():

        text = _text.copy()
        
        if len(items)>1:
            source = [int(i) for i in items]
            if if_consequetive(source):
                source = [min(source), max(source)]
        else:
            source = [int(items[0]), int(items[0])]
        
        _start = max(min(source)-num_words,0)
        _end = min(max(source)+num_words, len(text))
        _start, _end = rework_start_end_ids_by_sentid(_start, _end, itemdict['text'], sentid2tid)

        text[source[0]-1] = "<ACT_CAUSE> " + text[source[0]-1]
        text[source[1]-1] = text[source[1]-1] + " </ACT_CAUSE>"

        if verbose:
            print(' '.join(text[_start-1:_end-1]))


def negation_rules(tgx, itemdict, text, pos, sentid2tid, method=[], num_tries=2, curr_try=0, verbose=True):
    """
    current problems: 
    some valid *did not*s not included yet
    should I change 'can not' into 'cannot'?
    """

    curr_try += 1
    edit_id = tgx

    if verbose:
        print('before {}, actual {}, after {}'.format(pos[tgx-2], pos[tgx-1], pos[tgx]))
    if curr_try > num_tries:
        if verbose:
            print('Max {} tries hit...'.format(num_tries))
        method.append(None)
        edit_id = None
        text = None
    elif pos[tgx-1][1][0:2] =='VB':
        # if actual word is a modal type
        if wnl.lemmatize(text[tgx-1],'v') in ['be', 'to', 'have', 'get', 'do'] \
        or (pos[tgx-1][1] in ['MD']):
            # if word is last word
            if tgx>=len(text):
                method.append('VB_1_1')
                text[tgx-1] =  "*not* " + text[tgx-1]
            # if word after is determiner of sorts
            elif pos[tgx][1][0:2] in ['CD', 'DT']:
                method.append('VB_1_2')
                text[tgx] = "*no*"
            # if word after is noun or description of a noun
            elif pos[tgx][1][0:2] in ['JJ', 'NN', 'PR', 'WP', 'WD', 'DT']:
                method.append('VB_1_3')
                text[tgx-1] = text[tgx-1] + " *not*"
            # if word after is VB
            elif pos[tgx][1][0:2] =='VB':
                method.append('VB_1_4')
                text[tgx-1] = text[tgx-1] + " *no*"
        # if word is the first word
        elif tgx-1==0:
            method.append('VB_2_1')
            text[tgx-1] = "*Not* " + text[tgx-1].lower()
        # if word before target is noun type
        elif pos[tgx-2][1][0:2] in ['NN', 'PR', 'WP', 'WD', 'DT']:
            method.append('VB_3_1')
            text[tgx-1] = "*did not* " + wnl.lemmatize(text[tgx-1],'v')
        # if word is last word
        elif tgx>=len(text):
            method.append('VB_4_1')
            text[tgx-1] =  "*not* " + text[tgx-1]
        # if word before target is AUX | if word after target is IN/TO
        elif wnl.lemmatize(text[tgx-2],'v') in ['be', 'to', 'have', 'get', 'do'] \
        or (pos[tgx-2][1] in ['MD'])\
        or (pos[tgx][1] in ['IN', 'TO']):
            method.append('VB_5_1')
            text[tgx-1] = "*not* " + text[tgx-1]
        else:
            if verbose:
                print('Unable to find VB_X_X rule...')
            method.append(None)
            edit_id = None
            text = None
            # text[tgx-1] = "*did not* " + wnl.lemmatize(text[tgx-1],'v')
    # if target is noun (e.g. as a result of)
    elif pos[tgx-1][1][0:2] =='NN':
        method.append('NN_1_1')
        sent_id = int(itemdict['text'][tgx-1][1])
        loc_id = tgx-sentid2tid[sent_id]

        if sent_id+1 < len(sentid2tid):
            doc = nlp(u' '.join(text[sentid2tid[sent_id]-1:sentid2tid[sent_id+1]-1]))
        else:
            doc = nlp(u' '.join(text[sentid2tid[sent_id]-1:]))

        # print('Check if loc_id {} makes sense in doc: {}'.format(loc_id, doc))
        dep_dict = {}
        for ix, token in enumerate(doc):
            dep_dict[token.idx] = [ix, token.text, token.head.text, token.head.idx]
            if ix==loc_id:
                spacy_loc_id = token.idx

        # print(text[sentid2tid[sent_id]-1:sentid2tid[sent_id+1]-1])
        assert(dep_dict[spacy_loc_id][1]==text[tgx-1])
        edit_id = sentid2tid[sent_id] + dep_dict[dep_dict[spacy_loc_id][3]][0]
        # print('Editing root word edx "{}" | orx "{}"...'.format(text[edit_id], dep_dict[spacy_loc_id][2]))
        text, method, edit_id = negation_rules(
            edit_id, itemdict, text, pos, sentid2tid, method=method, 
            num_tries=num_tries, curr_try=curr_try, verbose=verbose)
    # if actual word is an adjective
    elif pos[tgx-1][1][0:2] =='JJ':
        # if adjective is last word
        if tgx>=len(text):
            method.append('JJ_1_1')
            text[tgx-1] = "*not* " + text[tgx-1]
        # if word after is positive conjunctions
        elif text[tgx] in ['and', 'or']:
            method.append('JJ_1_2')
            text[tgx-1] = "*not* " + text[tgx-1]
            text[tgx] = "*nor*"
        else:
            method.append('JJ_1_3')
            text[tgx-1] = "*not* " + text[tgx-1]
    # if actual word is an subornating conjunction (E.g. because, before, of)
    elif pos[tgx-1][1][0:2] =='IN':
        method.append('IN_1_1')
        text[tgx-1] = "*not* " + text[tgx-1]
    else:
        if verbose:
            print('Unable to find rule...')
        method.append(None)
        edit_id = None
        text = None
        # text[tgx-1] = "*did not* " + wnl.lemmatize(text[tgx-1],'v')
    
    return text, method, edit_id


def get_synonyms_antonyms(word):

    # include self dictionary
    cause_terms = ['cause', 'induce', 'trigger', 'affect', 'spark', 'incite', 'set']
    opp_cause_terms = ['deter', 'defuse', 'impede', 'block']

    # use wordnet dictionary
    synonyms = [] 
    antonyms = [] 

    for syn in wordnet.synsets(word): 
        for l in syn.lemmas():
            synonyms.append(l.name()) 
            if l.antonyms(): 
                antonyms.append(l.antonyms()[0].name()) 
    
    syn, ant = list(set(synonyms)), list(set(antonyms))

    if (wnl.lemmatize(word,'v').lower() in cause_terms):
        if (len(ant)==0):
            ant = opp_cause_terms
        if (len(syn)==0):
            syn = cause_terms

    return syn, ant


def improve_negation_flow(text, edit_id, edit_method, verbose=True):
    """
    current problems: 
    might be using too many packages (can consider streamlining spacy, nltk, pattern)
    """
    tense = None
    word = text[edit_id-1]
    # check grammar of word
    if conjugate_en(verb=word,tense='past',number='singular')==word:
        tense = 'past'
    elif conjugate_en(verb=word,tense='present',number='singular')==word:
        tense = 'present'
    elif conjugate_en(verb=word,tense='participle',number='singular')==word:
        tense = 'participle'

    syn, ant = get_synonyms_antonyms(word)
    if len(ant)>0:
        edit_word = random.choice(ant)
        if tense is not None:
            edit_word = conjugate_en(verb=edit_word,tense=tense,number='singular')
        else:
            if verbose:
                print('Caution: Unsure of tense of "{}" to "{}"...'.format(word, edit_word))
        text[edit_id-1] = "*" + edit_word + "*"
    else:
        if verbose:
            print('Unable to find an antonym...')
        text = None
        edit_word = None
    return text, edit_word


def format_print_text(text, source, target, signal, num_words=10, itemdict_text=None, sentid2tid=None, _start=None, _end=None):

    if text is None:
        return None, _start, _end
    else:
        text[source[0]-1] = "<SOURCE> " + text[source[0]-1]
        text[source[1]-1] = text[source[1]-1] + " </SOURCE>"
        text[target[1]-1] = "<TARGET> " + text[target[0]-1]
        text[target[1]-1] = text[target[1]-1] + " </TARGET>"

        if None not in signal:
            text[signal[0]-1] = "<SIGNAL> " + text[signal[0]-1]
            text[signal[1]-1] = text[signal[1]-1] + " </SIGNAL>"
            if _start is None:
                _start = max(min(source+target+signal)-num_words,0)
            if _end is None:
                _end = min(max(source+target+signal)+num_words, len(text))
        else:
            if _start is None:
                _start = max(min(source+target)-num_words,0)
            if _end is None:
                _end = min(max(source+target)+num_words, len(text))
        
        if (itemdict_text is not None) and (sentid2tid is not None):
            _start, _end = rework_start_end_ids_by_sentid(_start, _end, itemdict_text, sentid2tid)
        
        return ' '.join(text[_start-1:_end-1]), _start, _end


def negate_causes(dictionary, item, edit_which, num_words=10, verbose=True):

    if verbose:
        print('-'*10, item, '-'*10)
    itemdict = dictionary[item]
    plot = itemdict['plotlink_cse']
    _text = []
    sentid2tid = {}
    for i in itemdict['text']:
        _text.append(i[3])
        if i[2]=="0": # start of new sentence
            sentid2tid[int(i[1])] = int(i[0])
    pos = pos_tag(_text)
    outdict = {}

    for ix, ((source, target), items) in enumerate(plot.items()):
        
        # format input
        text = _text.copy()
        source = [int(i) for i in source.split('_')]
        target = [int(i) for i in target.split('_')]

        if if_consequetive(source):
            source = [min(source), max(source)]
        if if_consequetive(target):
            target = [min(target), max(target)]

        signal = items[1]
        if signal == '' or signal == 'null':
            signal = [None]
            method = [None]
        else:
            signal = itemdict['m2t'][signal]
            signal = [int(i) for i in signal.split('_')]
            if if_consequetive(signal):
                signal = [min(signal), max(signal)]
        
        # negation rules
        if edit_which == 'target':
            text, method, edit_id = negation_rules(
                target[0], itemdict, text, pos, sentid2tid, 
                method=[], num_tries=2, verbose=verbose)
        elif edit_which == 'source':
            text, method, edit_id = negation_rules(
                source[0], itemdict, text, pos, sentid2tid, 
                method=[], num_tries=2, verbose=verbose)
        elif (edit_which == 'signal') and (None not in signal):
            text, method, edit_id = negation_rules(
                signal[0], itemdict, text, pos, sentid2tid, 
                method=[], num_tries=2, verbose=verbose)

        # improve text flow
        if None not in method and len(method)>0:
            alt_text, alt_word = improve_negation_flow(_text.copy(), edit_id, method, verbose=verbose)
            # if method is nor format
            if alt_text is not None:
                if method[-1]=='JJ_1_2':
                    alt_text, alt_word_2 = improve_negation_flow(alt_text.copy(), edit_id+2, method, verbose=verbose)
                    alt_word = [alt_word, alt_word_2]
        else:
            alt_word = None
            alt_text = None

        # format print text
        print_edit, _start, _end = format_print_text(
            text, source, target, signal, num_words=num_words, 
            itemdict_text=itemdict['text'], sentid2tid=sentid2tid, _start=None, _end=None)
        print_alt, _start, _end = format_print_text(
            alt_text, source, target, signal, num_words=num_words, 
            itemdict_text=None, sentid2tid=None, _start=_start, _end=_end)

        # keep only successful edits
        if None not in method:

            outdict[ix] = {
                'signal_id': signal,
                'source_id': source,
                'target_id': target,
                'print_range': [_start, _end],
                'edit_text': print_edit,
                'edit_id': edit_id,
                'edit_method': method,
                'alt_word': alt_word,
                'alt_text': print_alt
            }
        
    return outdict


########## main negation steps


def main(dataset):
    # load data
    with open(f'./src/data/{dataset}_document_raw.pickle', 'rb') as f:
        data_dict = pickle.load(f)
    
    # filter data
    list_of_cs_dirs = []
    for i in data_dict:
        if len(data_dict[i]['markables_cse'])+len(data_dict[i]['plotlink_cse'])>0:
            list_of_cs_dirs.append(i)

    outdict = {}
    for i in list_of_cs_dirs:
        show_action_causative(data_dict, i, verbose=verbose, num_words=0)
        t_dict = show_plot_link_causes(data_dict, i, verbose=verbose, num_words=0)
        if len(t_dict)>0:
            outdict[i] = t_dict

    counter = 0
    for i in outdict:
        for ix in outdict[i]:
            counter += 1
    print('number of orig examples to work with: {}'.format(counter))
    
    # intervene
    edit_signal_dict = {}
    for i in outdict:
        t_dict = negate_causes(data_dict, i, 'signal', verbose=verbose, num_words=0)
        if len(t_dict)>0:
            edit_signal_dict[i] = t_dict

    # number of sucessful interventions
    counter = 0
    for i in edit_signal_dict:
        for ix in edit_signal_dict[i]:
            counter += 1
    print('number of intervened signal to work with: {}'.format(counter))

    # intervene
    edit_target_dict = {}
    for i in outdict:
        t_dict = negate_causes(data_dict, i, 'target', verbose=verbose, num_words=0)
        if len(t_dict)>0:
            edit_target_dict[i] = t_dict

    # number of sucessful interventions
    counter = 0
    for i in edit_target_dict:
        for ix in edit_target_dict[i]:
            counter += 1
    print('number of intervened target to work with: {}'.format(counter))

    # save file
    edits_dict = {
        'ed_signal': edit_signal_dict,
        'ed_target': edit_target_dict
    }
    with open(f'./src/data/{dataset}_document_edits.pickle', 'wb') as f:
        pickle.dump(edits_dict, f, pickle.HIGHEST_PROTOCOL)


def text_to_spacy_table_info(_text):

    doc = nlp(_text)

    table = []
    # current_length = 0
    letters2word = {}
    
    # future to do: keep only relevant items
    # future to do: replicate for ESL corpus in main()
    for idx, token in enumerate(doc):
        letters2word[token.idx] = idx
        table.append([token.text, token.lemma_, token.pos_, token.tag_, 
                    token.dep_, token.shape_, token.is_alpha, token.is_stop,
                    token.head.text, token.head.idx, token.head.pos_, 
                    [child for child in token.children]])
        # current_length += len(token)+1

    table = pd.DataFrame(table)
    table.columns = ['text', 'lemma', 'pos', 'tag', 'dep', 'shape', 'is_alpha', 'is_stop', 
                    'head_text', 'head_id', 'head_pos', 'children']
    table['head_id'] = [letters2word[i] for i in table['head_id']] # convert idx to word level

    return table


def text_to_negated_edits(_text, sentid2tid, get_roots=None):
    # clean multiple whitespaces
    _text = _RE_COMBINE_WHITESPACE.sub(" ", _text).strip()
    text = _text.split(' ')

    # start with root verb of sentence
    table = text_to_spacy_table_info(_text)
    if get_roots is None:
        get_roots = table[(table['pos'] == 'VERB') & (table['dep'] == 'ROOT')].index
    
    # format into ESC format for convenience (not efficient)
    itemdict = {'text': [(int(idx+1),0,int(idx),t) for idx, t in enumerate(text)]}

    t_dict = {}
    for jx, root in enumerate(get_roots):
        # check if root has acomp
        if table.loc[root,'lemma'] in ['be', 'to', 'have', 'get', 'do'] or (table.loc[root,'tag'] in ['MD']):
            if len(table[(table['dep'].isin(['acomp','attr'])) & (table['head_id']==root)])>0:
                root = table[(table['dep'].isin(['acomp','attr'])) & (table['head_id']==root)].index[0]
                if verbose:
                    print('new root: {}'.format(table.loc[root, 'text']))
                # to do: check if new root has conj, negate them too
        
        edit_text, method, edit_id = negation_rules(
            root+1, itemdict, text.copy(), pos_tag(text), sentid2tid, 
            method=[], num_tries=2, verbose=verbose)

        # improve text flow
        if None not in method and len(method)>0:
            alt_text, alt_word = improve_negation_flow(_text.split(' '), edit_id, method, verbose=verbose)
            # if method is nor format
            if alt_text is not None:
                if method[-1]=='JJ_1_2':
                    alt_text, alt_word_2 = improve_negation_flow(alt_text.copy(), edit_id+2, method, verbose=verbose)
                    alt_word = [alt_word, alt_word_2]
        else:
            alt_word = None
            alt_text = None

        # keep only successful edits
        if None not in method:
            t_dict[jx] = {
                'edit_text': edit_text,
                'edit_id': edit_id,
                'edit_method': method,
                'alt_word': alt_word,
                'alt_text': alt_text
            }

    return t_dict


def text_to_stronger_edits(_text, sentid2tid):
    # clean multiple whitespaces
    _text = _RE_COMBINE_WHITESPACE.sub(" ", _text).strip()
    text = _text.split(' ')

    # get weaker words for converting
    table = text_to_spacy_table_info(_text)
    get_roots = table[table['lemma'].isin(modal_verb_dict.keys())].index
    
    # format into ESC format for convenience (not efficient)
    itemdict = {'text': [(int(idx+1),0,int(idx),t) for idx, t in enumerate(text)]}

    t_dict = {}
    for jx, root in enumerate(get_roots):
        
        if table.loc[root+1,'lemma']=='be':
            text[root] = "*was*"
            text[root+1] = ""
            method = ['MOD_2_1']
        elif  table.loc[root+1,'lemma']=='have':
            if table.loc[root+2,'lemma']=='be':
                text[root] = "*was*"
                text[root+1] = ""
                text[root+2] = ""
                method = ['MOD_3_2']
            else:
                text[root] = "*had*"
                text[root+1] = ""
                method = ['MOD_3_1'] 
        elif (table.loc[root,'tag']=='MD') and (table.loc[root+1,'tag']=='RB'):
            text[root] = '*'+ modal_verb_dict[text[root]] + '*'
            text[root+1] = ""
            # text[root+2] = conjugate_en(verb=text[root+2],tense='past',number='singular')
            method = ['MOD_4_1']
        else:
            text[root] = '*'+ modal_verb_dict[text[root]] + '*'
            method = ['MOD_1_1']
        
        edit_text = text
        edit_id = root

        # keep only successful edits
        if None not in method:
            t_dict[jx] = {
                'edit_text': edit_text,
                'edit_id': edit_id,
                'edit_method': method,
                'alt_word': None,
                'alt_text': None
            }

    return t_dict


def main_csci(file_name, dataset, ed_type = '1to4'):
    """Main function to create counterfactuals for CSci corpus 
    (Data used, "pubmed_causal_language_use.csv", can be downloaded from Github link 
    "https://github.com/junwang4/causal-language-use-in-science/tree/master/data")

    Args:
        file_name ([str]): Path to input data
        dataset ([str]): Name of dataset, will be how edits output file is named
        ed_type (str, optional): Edit type, supports only '1to4' or '2to1' currently. 
        Defaults to '1to4', which is equivalent to the baseline case.'
    """

    if ed_type == '1to4':
        _ed_type = ''
    else:
        _ed_type = '_'+ed_type
    
    label_filter = ed_type[0]

    base_file_name = './src/data/{}_base.csv'.format(dataset)
    edits_file_name = './src/data/{}{}_document_edits.pickle'.format(dataset, _ed_type)
    print('generating >>> {} ...'.format(edits_file_name))

    # load file
    sents = pd.read_csv(file_name)

    # altlex processing
    if dataset=='altlex':
        sents = sents.sort_values(by=['label'], ascending=[False])
        drop_index = sents[
            (sents.duplicated(subset=['sentence'], keep='first'))
            & (sents['label'].astype(str)=='0')
            ].index
        sents.drop(drop_index, inplace=True)
    
    # save base
    if 'pmid' not in sents.columns:
        sents['pmid'] = sents.index
    if 'label' not in sents.columns:
        print(f'label column is missing... using "{ed_type[0]}" as default label')
        sents['label'] = ed_type[0]
    if not os.path.isfile(base_file_name):
        print('generating >>> {} ...'.format(base_file_name))
        sents.to_csv(base_file_name, index=False)

    # create edits
    sentid2tid = {0: 1} # fixed since only one-sentence per doc
    edits_dict = {}
    counter = 0

    for ix, _text in zip(sents.loc[sents['label'].astype(str) == str(label_filter), 'pmid'], \
    sents.loc[sents['label'].astype(str) == str(label_filter), 'sentence']):
        counter += 1
        if ed_type[-3:]=='to4':
            if dataset=='altlex':
                edit_id = [sents[sents['pmid']==ix]['signal_id'].item()]
                t_dict = text_to_negated_edits(_text, sentid2tid, edit_id)
            else:
                t_dict = text_to_negated_edits(_text, sentid2tid)
        elif ed_type[-3:]=='to1':
            t_dict = text_to_stronger_edits(_text, sentid2tid)
        else:
            t_dict = {}
        if len(t_dict)>0:
            edits_dict[ix] = t_dict
            
    # number of sucessful interventions
    print('number of original examples to work with: {}'.format(counter))
    counter = 0
    for i in edits_dict:
        for ix in edits_dict[i]:
            counter += 1
    print('number of intervened examples to work with: {}'.format(counter))

    # save file
    with open(edits_file_name, 'wb') as f:
        pickle.dump(edits_dict, f, pickle.HIGHEST_PROTOCOL)


########## select edits

def fuzzy_match(s1, s2, min_prop=0.7):
    if (s2 is None) or (s1 is None):
        return False
    elif s1 in s2:
        return True
    else:
        max_dist = round(max(len(s1),len(s2))*(1-min_prop),0)
        return edit_distance(s1,s2) <= max_dist


def format_text_output_group(text, edit_id = None, edit_text = None, extensions=''):
    """
    text : list
    edit_id : int
    edit_text : str
    extensions: str
    """
    if 'multiple' in extensions:
        return format_text_output_multiples(text = text, edit_id = edit_id, mult_range = 1)
    elif 'shorten' in extensions:
        return find_root_phrase_from_doc(text = format_text_output(text), edit_text = edit_text, edit_id = edit_id)
    elif 'mask' in extensions:
        return format_text_output(mask_nouns_of_doc(text = text, edit_id = edit_id))
    elif 'synonyms' in extensions:
        return get_synonym_paraphrased_text(text = format_text_output(text).split(' '))
    elif 't5para' in extensions:
        return get_t5_paraphrased_text(text = format_text_output(text))
    else:
        return format_text_output(text)


def mask_nouns_of_doc(text, edit_id):
    MASK_TOKEN = '[MASK]'
    doc = nlp(' '.join(text))
    masked_text = []
    for idx, token in enumerate(doc):
        if (token.tag_[0:2] == 'NN') and (idx!= edit_id):
            word = MASK_TOKEN
        else:
            word = token.text
        # print(f'token.tag_ {token.tag} | word {word}')
        masked_text.append(word)
    return masked_text


def format_text_output(text):
    text = ' '.join(text)
    return re.sub(' +', ' ',re.sub('(\*)', '', text))


def format_text_output_multiples(text, edit_id, mult_range=1):
    mult_length = round(len(text)/(mult_range*2+1),0)
    text = text[int(edit_id-mult_range):int(edit_id+mult_range)]*int(mult_length)
    return format_text_output(text)


def loop_till_phrase_found(focus_id, table, t_phrase = []):
    if focus_id == table.loc[focus_id,'head_id']:
        focus_id = table.loc[focus_id,'head_id']
        all_children = table[table['head_id']==focus_id].index
        t_phrase = list(set(t_phrase+list(all_children)))
        if verbose: print('end')
    else:
        focus_id = table.loc[focus_id,'head_id']
        t_phrase.append(focus_id)
        # # if close to root word, get childrens
        # if len(t_phrase)<3:
        #     all_children = table[table['head_id']==focus_id].index
        #     t_phrase = list(set(t_phrase+list(all_children)))
        t_phrase = loop_till_phrase_found(focus_id, table, t_phrase)
    return t_phrase


def find_root_phrase_from_doc(text, edit_id, edit_text):
    
    n_edits = len(edit_text.split(' '))
    edit_ids = range(edit_id, edit_id+n_edits)
    table = text_to_spacy_table_info(text)
    final_phrase = []
    for ed in edit_ids:
        t_phrase = loop_till_phrase_found(ed, table, t_phrase = [ed])
        final_phrase.extend(t_phrase)
    final_phrase = list(set(final_phrase))
    
    return ' '.join([table.loc[i, 'text'] for i in final_phrase])


def process_and_keep_edits(dataset, extensions=''):

    print('generating >>> {}_edits{} ...'.format(dataset, extensions))
    edits_file_name = './src/data/{}_document_edits.pickle'.format(dataset) # in
    csv_file_name = './src/data/{}_edits{}.csv'.format(dataset, extensions) # out

    wnl = WordNetLemmatizer()
    with open(edits_file_name, 'rb') as f:
        edits_dict = pickle.load(f)
        
    """
    {'edit_text': edit_text,
    'edit_id': edit_id,
    'edit_method': method,
    'alt_word': alt_word,
    'alt_text': alt_text}
    """

    store_examples = []

    for i in edits_dict:

        for j in edits_dict[i]:
            
            if len(edits_dict[i][j]['edit_method'])==0:
                # no edits involved, skip to next item
                continue

            if isinstance(edits_dict[i][j]['edit_id'], (int, np.integer)):
                edits = [edits_dict[i][j]['edit_id']]
            elif isinstance(edits_dict[i][j]['edit_id'], list):
                edits = edits_dict[i][j]['edit_id']

            for jx, edit_id in enumerate(edits):
                
                if edits_dict[i][j]['alt_text'] is not None:
                    alt_word = edits_dict[i][j]['alt_text'][edit_id-1]
                    alt_word = wnl.lemmatize(re.sub('(\*)', '', alt_word))
                else:
                    alt_word = None
                not_word = edits_dict[i][j]['edit_text'][edit_id-1]
                orig_word = wnl.lemmatize(re.sub('(\s)*(\*)(.)+(\*)(\s)*', '', not_word))
                not_word = re.sub('(\*)', '', not_word)

                if verbose:
                    print('orig_word: {} | not_word: {} | alt_word: {}'.format(orig_word, not_word, alt_word))

                if edits_dict[i][j]['edit_method'][-1] == 'VB_1_2':
                    edit_id = edits_dict[i][j]['edit_id']
                else:
                    edit_id = edits_dict[i][j]['edit_id']-1

                if fuzzy_match(orig_word, alt_word, min_prop=0.7):
                    store_examples.append([
                        '{}_{}_{}{}'.format(i, j, 'alt', extensions),
                        format_text_output_group(
                            text = edits_dict[i][j]['alt_text'], 
                            edit_id = edit_id, edit_text = alt_word, 
                            extensions=extensions)
                        ])
                else:
                    store_examples.append([
                        '{}_{}_{}{}'.format(i, j, 'edt', extensions), 
                        format_text_output_group(
                            text = edits_dict[i][j]['edit_text'], 
                            edit_id = edit_id, edit_text = not_word, 
                            extensions=extensions)
                        ])

    # export
    store_examples = pd.DataFrame(store_examples)
    store_examples.columns = ['pmid', 'sentence']
    store_examples = store_examples.drop_duplicates(subset=['sentence'])
    store_examples.to_csv(csv_file_name, index=False)


def edit_originals(dataset, extensions='', base_file_name=None, csv_file_name=None):

    # load file
    if base_file_name is None:
        base_file_name = './src/data/{}_base.csv'.format(dataset)
    if csv_file_name is None:
        csv_file_name = './src/data/{}_base{}.csv'.format(dataset, extensions)
    print('generating >>> {} ...'.format(csv_file_name.split('/')[-1]))

    sents = pd.read_csv(base_file_name)

    # create edits
    base_dict = {}
    store_examples = []

    for ix, _text, label in zip(sents['pmid'], sents['sentence'], sents['label']):

        # clean multiple whitespaces
        _text = _RE_COMBINE_WHITESPACE.sub(" ", _text).strip()

        if 'aug' in extensions:
            # aug_text = paraphrase(_text)
            aug_text = _text
            store_examples.append(['{}_{}{}'.format(ix, 0, extensions), aug_text, label])
        else:
            text = _text.split(' ')

            # start with root verb of sentence
            table = text_to_spacy_table_info(_text)
            get_roots = table[(table['pos'] == 'VERB') & (table['dep'] == 'ROOT')].index

            for jx, root in enumerate(get_roots):
                store_examples.append(['{}_{}{}'.format(ix, jx, extensions), \
                format_text_output_group(text = text, edit_id = root, edit_text = 'xxx', extensions=extensions),
                label])

    # export
    store_examples = pd.DataFrame(store_examples)
    store_examples.columns = ['pmid', 'sentence', 'label']
    store_examples.to_csv(csv_file_name, index=False)


########## create paraphrases


def check_tense(word, default = None):
    # check grammar of word
    if conjugate_en(verb=word,tense='past',number='singular')==word:
        tense = 'past'
    elif conjugate_en(verb=word,tense='present',number='singular')==word:
        tense = 'present'
    elif conjugate_en(verb=word,tense='participle',number='singular')==word:
        tense = 'participle'
    else:
        tense = default
    return tense


def check_number(word, tense = None, default = None):
    if tense is None:
        tense = check_tense(word, 'present')
    # check grammar of word
    if conjugate_en(verb=word,tense=tense,number='singular')==word:
        number = 'singular'
    elif conjugate_en(verb=word,tense=tense,number='plural')==word:
        number = 'plural'
    else:
        number = default
    return number


def morphify(word,org_pos,target_pos):
    """ 
    morph a word 
    code reference: https://stackoverflow.com/questions/27852969/how-to-list-all-the-forms-of-a-word-using-nltk-in-python
    """
    synsets = wordnet.synsets(word, pos=org_pos)

    # Word not found
    if not synsets:
        return []

    # Get all  lemmas of the word
    lemmas = [l for s in synsets \
                   for l in s.lemmas() if s.name().split('.')[1] == org_pos]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) \
                                    for l in    lemmas]

    # filter only the targeted pos
    related_lemmas = [l for drf in derivationally_related_forms \
                           for l in drf[1] if l.synset().name().split('.')[1] == target_pos]

    # Extract the words from the lemmas
    words = [l.name() for l in related_lemmas]
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w))/len_words) for w in set(words)]
    result.sort(key=lambda w: -w[1])

    # return all the possibilities sorted by probability
    return result


def morph_if_possible(orig_word,orig_pos,to_pos):
    edit_word = morphify(orig_word,orig_pos,to_pos)
    if len(edit_word)>0:
        return edit_word[0][0]
    else:
        return orig_word


def get_synonym_paraphrased_text(text, max_edits = 5):
    change_dict = {}

    pos = pos_tag(text)
    edits = list(range(len(text)))
    random.shuffle(edits)

    # loop
    for jx in edits:

        orig_word = text[jx]
        orig_pos = pos[jx][1]
        
        if (orig_pos[0:2] in ['DT','IN', 'EX', 'CC', 'MD', 'WP', 'WD', 'WR', 'UH', 'RP', 'SY', 'PO']) \
        or (jx>0 and orig_word.istitle()) or orig_word.isupper():
            # skip common words
            # skip titles and abvs
            continue
            
        tense = check_tense(orig_word, default='present')
        number = check_number(orig_word, tense=tense, default='singular')

        syn, ant = get_synonyms_antonyms(orig_word)
        syn = [i for i in syn if i!=orig_word]
        random.shuffle(syn)

        if len(syn)>0:

            for sx, edit_word in enumerate(syn):
                
                if edit_word[0].isupper():
                    continue
                    
                edit_words = None
                if '_' in edit_word:
                    edit_word = re.sub('_', ' ', edit_word)
                    edit_words = edit_word.split(' ')
                    edits_len = len(edit_words)
                    _text = text[0:jx]+edit_words+text[jx+1:]
                    _text = format_text_output(_text).split(' ')
                    _pos = pos_tag(_text)

                    edit_words = []
                    plural_status = True
                    for ed in range(jx,jx+edits_len):
                        _edit_word = _text[ed]
                        if _pos[ed][1][0:2]=='NN' and orig_pos[0:2]!='NN':
                            sub_edit_word = morph_if_possible(_edit_word,'n','v')
                        elif _pos[ed][1][0:2]!='NN' and orig_pos[0:2]=='NN':
                            sub_edit_word = morph_if_possible(_edit_word,'v','n')
                        else:
                            sub_edit_word = _edit_word

                        if orig_pos[0:2]!='NN' and plural_status:
                            sub_edit_word = conjugate_en(verb=sub_edit_word,tense=tense,number=number)
                            plural_status = False
                        edit_words.append(sub_edit_word)
                else:
                    if orig_pos[0:2]!='NN':
                        edit_word = morph_if_possible(edit_word,'n','v')
                        edit_word = conjugate_en(verb=edit_word,tense=tense,number=number)
                    else:
                        edit_word = morph_if_possible(edit_word,'v','n')
                    edit_words = [edit_word]

                if edit_words is not None:
                    change_dict[jx] = {
                        'orig_text': orig_word,
                        'edit_text': edit_words
                    }
                    break
                else:
                    continue
                    
    edits = list(change_dict.keys())
    random.shuffle(edits)
    edits = edits[0:max_edits]

    final_text = [change_dict[ix]['edit_text'] if ix in edits else word for ix, word in enumerate(text)]
    final_text = ' '.join(list(flatten(final_text)))

    return final_text


def t5_paraphraser(model, text, input_ids, attention_masks, counter=0):
    
    beam_outputs = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_masks,
        do_sample=True,
        max_length=MAX_LENGTH,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=1
    )

    edit_text = tokenizer.decode(beam_outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
    if (edit_text.lower() != text.lower()) and ('?' not in edit_text[-3:]):
        return edit_text
    elif counter > 30: # max_tries
        return None
    else:
        counter += 1
        return t5_paraphraser(model, text, input_ids, attention_masks, counter=counter)


def get_t5_paraphrased_text(text):
    _text =  "paraphrase: " + text + " </s>"
    encoding = tokenizer.encode_plus(_text, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    return t5_paraphraser(model, text, input_ids, attention_masks)


if __name__ == '__main__':
    tic = time.time()
    ### generate c1->c4 edits
    file_name = './src/data/pubmed_base.csv' # contains duplicates
    main_csci(file_name=file_name, dataset='pubmed')

    ### select/render edits
    process_and_keep_edits(dataset='pubmed')
    process_and_keep_edits(dataset='pubmed', extensions='_multiples')
    process_and_keep_edits(dataset='pubmed', extensions='_shorten')
    process_and_keep_edits(dataset='pubmed', extensions='_mask')
    process_and_keep_edits(dataset='pubmed', extensions='_synonyms')
    process_and_keep_edits(dataset='pubmed', extensions='_t5para')

    # ### ext: on originals
    # edit_originals(dataset='pubmed', extensions='_shorten')
    # edit_originals(dataset='pubmed', extensions='_mask')
    # edit_originals(dataset='pubmed', extensions='_synonyms')
    # edit_originals(dataset='pubmed', extensions='_t5para')
 
    ### generate c2->c1 edits
    file_name = './src/data/pubmed_base.csv'
    main_csci(file_name=file_name, dataset='pubmed', ed_type='2to1')
    process_and_keep_edits(dataset='pubmed_2to1')
    process_and_keep_edits(dataset='pubmed_2to1', extensions='_shorten')
    process_and_keep_edits(dataset='pubmed_2to1', extensions='_multiples')
    process_and_keep_edits(dataset='pubmed_2to1', extensions='_mask')
    process_and_keep_edits(dataset='pubmed_2to1', extensions='_synonyms')
    process_and_keep_edits(dataset='pubmed_2to1', extensions='_t5para')

    ### generate c2->c1->c4 edits
    file_name = './src/data/pubmed_2to1_edits.csv'
    main_csci(file_name=file_name, dataset='pubmed', ed_type='2to4')
    process_and_keep_edits(dataset='pubmed_2to4')

    print(f'time used: {time.time()-tic:.0f} seconds')
