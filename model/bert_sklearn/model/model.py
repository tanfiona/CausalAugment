import torch
import torch.nn as nn

from .pytorch_pretrained import BertModel
from .pytorch_pretrained import BertPreTrainedModel

import sys
from .losses import SupConLoss


def LinearBlock(H1, H2, p):
    return nn.Sequential(
        nn.Linear(H1, H2),
        nn.BatchNorm1d(H2),
        nn.ReLU(),
        nn.Dropout(p))

def MLP(D, n, H, K, p):
    """
    MLP w batchnorm and dropout.

    Parameters
    ----------
    D : int, size of input layer
    n : int, number of hidden layers
    H : int, size of hidden layer
    K : int, size of output layer
    p : float, dropout probability
    """

    if n == 0:
        print("Defaulting to linear classifier/regressor")
        return nn.Linear(D, K)
    else:
        print("Using mlp with D=%d,H=%d,K=%d,n=%d"%(D, H, K, n))
        layers = [nn.BatchNorm1d(D),
                  LinearBlock(D, H, p)]
        for _ in range(n-1):
            layers.append(LinearBlock(H, H, p))
        layers.append(nn.Linear(H, K))
        return torch.nn.Sequential(*layers)


class BertPlusMLP(BertPreTrainedModel):
    """
    Bert model with MLP classifier/regressor head.

    Based on pytorch_pretrained_bert.modeling.BertForSequenceClassification

    Parameters
    ----------
    config : BertConfig
        stores configuration of BertModel

    model_type : string
         'text_classifier' | 'text_regressor' | 'token_classifier'

    num_labels : int
        For a classifier, this is the number of distinct classes.
        For a regressor his will be 1.

    num_mlp_layers : int
        the number of mlp layers. If set to 0, then defualts
        to the linear classifier/regresor in the original Google paper and code.

    num_mlp_hiddens : int
        the number of hidden neurons in each layer of the mlp.
    """

    def __init__(self, config,
                 model_type="text_classifier",
                 num_labels=2,
                 num_mlp_layers=2,
                 class_weight=[1.,1.],
                 num_mlp_hiddens=500):

        super(BertPlusMLP, self).__init__(config)
        self.model_type = model_type
        self.num_labels = num_labels
        self.num_mlp_layers = num_mlp_layers
        self.num_mlp_hiddens = num_mlp_hiddens
        self.class_weight = class_weight

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert = BertModel(config)
        self.input_dim = config.hidden_size

        self.mlp = MLP(D=self.input_dim,
                       n=self.num_mlp_layers,
                       H=self.num_mlp_hiddens,
                       K=self.num_labels,
                       p=config.hidden_dropout_prob)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids=None, input_mask=None, labels=None):

        hidden, pooled_output = self.bert(input_ids,
                                          segment_ids,
                                          input_mask,
                                          output_all_encoded_layers=False)

        if self.model_type == "token_classifier":
            output = hidden
        else:
            output = pooled_output
            output = self.dropout(output)

        output = self.mlp(output)

        if labels is not None:
            if self.model_type == "text_classifier":
                loss_criterion = nn.CrossEntropyLoss(reduction='none', weight=self.class_weight)
                loss = loss_criterion(output.view(-1, output.shape[-1]), labels.view(-1))
            elif self.model_type == "text_regressor":
                loss_criterion = nn.MSELoss(reduction='none')
                output = torch.squeeze(output)
                loss = loss_criterion(output, labels)
            elif self.model_type == "token_classifier":
                loss_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1, weight=self.class_weight)
                loss = loss_criterion(output.view(-1, output.shape[-1]), labels.view(-1))
            return loss, output
        else:
            return output


class BertPlusSVM(BertPreTrainedModel):
    """
    Bert model with SVM classifier/regressor head.

    Based on pytorch_pretrained_bert.modeling.BertForSequenceClassification

    Parameters
    ----------
    config : BertConfig
        stores configuration of BertModel

    model_type : string
         'text_classifier' | 'text_regressor' | 'token_classifier'

    num_labels : int
        For a classifier, this is the number of distinct classes.
        For a regressor his will be 1.

    num_mlp_layers : int
        the number of mlp layers. If set to 0, then defualts
        to the linear classifier/regresor in the original Google paper and code.

    num_mlp_hiddens : int
        the number of hidden neurons in each layer of the mlp.
    """

    def __init__(self, config,
                 model_type="text_classifier",
                 num_labels=2,
                 num_mlp_layers=0,
                 class_weight=[1.,1.],
                 num_mlp_hiddens=24):

        super(BertPlusSVM, self).__init__(config)
        self.model_type = model_type
        self.num_labels = num_labels
        self.num_mlp_layers = 0 # overwrite default in util
        self.num_mlp_hiddens = 24 # overwrite default in util
        self.class_weight = class_weight

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert = BertModel(config)
        self.input_dim = config.hidden_size
        
        self.mlp_1 = MLP(D=self.input_dim, n=0, H=0, K=self.num_mlp_hiddens,p=config.hidden_dropout_prob)
        self.mlp_2 = MLP(D=self.num_mlp_hiddens, n=0, H=0, K=self.num_labels,p=config.hidden_dropout_prob)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids=None, input_mask=None, labels=None):

        hidden, pooled_output = self.bert(input_ids,
                                          segment_ids,
                                          input_mask,
                                          output_all_encoded_layers=False)

        if self.model_type == "token_classifier":
            mid_output = hidden
        else:
            mid_output = pooled_output
            mid_output = self.dropout(mid_output)

        mid_output = self.mlp_1(mid_output)
        output = self.mlp_2(mid_output)

        if labels is not None:
            if self.model_type == "text_classifier":
                loss_criterion = nn.CrossEntropyLoss(reduction='none', weight=self.class_weight)
                loss = loss_criterion(output.view(-1, output.shape[-1]), labels.view(-1))
            elif self.model_type == "text_regressor":
                loss_criterion = nn.MSELoss(reduction='none')
                output = torch.squeeze(output)
                loss = loss_criterion(output, labels)
            elif self.model_type == "token_classifier":
                loss_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1, weight=self.class_weight)
                loss = loss_criterion(output.view(-1, output.shape[-1]), labels.view(-1))
            return loss, output, mid_output
        else:
            return output, mid_output


class BertPlusSupCon(BertPreTrainedModel):
    """
    Bert model with SupContrast classifier/regressor head.

    Based on pytorch_pretrained_bert.modeling.BertForSequenceClassification

    Parameters
    ----------
    config : BertConfig
        stores configuration of BertModel

    model_type : string
         'text_classifier' | 'text_regressor' | 'token_classifier'

    num_labels : int
        For a classifier, this is the number of distinct classes.
        For a regressor his will be 1.

    num_mlp_layers : int
        the number of mlp layers. If set to 0, then defualts
        to the linear classifier/regresor in the original Google paper and code.

    num_mlp_hiddens : int
        the number of hidden neurons in each layer of the mlp.
    """

    def __init__(self, config,
                 model_type="text_classifier",
                 num_labels=2,
                 num_mlp_layers=2,
                 class_weight=[1.,1.],
                 num_mlp_hiddens=500
                 ):

        super(BertPlusSupCon, self).__init__(config)
        self.model_type = model_type
        self.num_labels = num_labels
        self.num_mlp_layers = 0 # overwrite default in util
        self.num_mlp_hiddens = 64 # overwrite default in util
        self.class_weight = class_weight

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert = BertModel(config)
        self.input_dim = config.hidden_size
        
        self.mlp_1 = MLP(D=self.input_dim, n=0, H=0, K=self.num_mlp_hiddens,p=config.hidden_dropout_prob)
        self.mlp_2 = MLP(D=self.num_mlp_hiddens, n=0, H=0, K=self.num_labels,p=config.hidden_dropout_prob)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids=None, input_mask=None, labels=None, loss_method=None):
        """
        csci train: label pairs, output fours
        csci test (validation): labels singular, input_ids pairs
        scite test (blind): NO labels, input_ids singular (no formatting required)
        """
        supcon_incl_negs = False
        neg_ids = None
        if loss_method is None: loss_method=''

        # print(f'Example input_ids.shape[1]:{input_ids.shape[1]}')
        # print(f'Example loss_method:{loss_method}')
        # print(f'Example input_ids[0]:{input_ids[0]}')
        # print(f'Example input_ids[0].shape:{input_ids[0].shape}')
        # print(f'Example labels:{labels}')
        if input_ids.shape[1] == 4:
            supcon_incl_negs = True
            seps = torch.LongTensor([101, 102]).to('cuda') # empty string = [CLS][SEP]
            f1,f2,f3,f4 = torch.split(input_ids, [1,1,1,1], dim=1)
            if f3.shape[0] == 1: # if bs=1
                if torch.sum(f3.squeeze()[0:2] == seps,0)<2:
                    neg_ids = torch.LongTensor([0]).to('cuda')
                else:
                    neg_ids = torch.LongTensor([]).to('cuda')
            else:
                neg_ids = torch.nonzero(torch.sum(f3.squeeze()[:,0:2] == seps,1)<2)[:,0]
            neg_bsz = neg_ids.shape[0] if neg_ids is not None else 0
            input_ids = torch.cat([f1,f3[neg_ids],f2,f4[neg_ids]],dim=0).squeeze()
            f1,f2,f3,f4 = torch.split(segment_ids, [1,1,1,1], dim=1)
            segment_ids = torch.cat([f1,f3[neg_ids],f2,f4[neg_ids]],dim=0).squeeze()
            f1,f2,f3,f4 = torch.split(input_mask, [1,1,1,1], dim=1)
            input_mask = torch.cat([f1,f3[neg_ids],f2,f4[neg_ids]],dim=0).squeeze()
        elif input_ids.shape[1] == 2:
            f1,f2 = torch.split(input_ids, [1,1], dim=1)
            input_ids = torch.cat([f1, f2], dim=0).squeeze()
            # print('input_ids2.shape:', input_ids.shape)
            f1,f2 = torch.split(segment_ids, [1,1], dim=1)
            segment_ids = torch.cat([f1, f2], dim=0).squeeze()
            f1,f2 = torch.split(input_mask, [1,1], dim=1)
            input_mask = torch.cat([f1, f2], dim=0).squeeze()

        hidden, pooled_output = self.bert(input_ids,
                                          segment_ids,
                                          input_mask,
                                          output_all_encoded_layers=False)

        if self.model_type == "token_classifier":
            mid_output = hidden
        else:
            mid_output = pooled_output
            mid_output = self.dropout(mid_output)

        mid_output = self.mlp_1(mid_output)
        
        # complete head
        output = self.mlp_2(mid_output)

        if labels is not None:
            # print('labels.shape:', labels.shape)
            loss_triplet = 0
            bsz = labels.shape[0]
            
            if supcon_incl_negs and ('triplet' not in loss_method):
                f1,f3 = torch.split(labels, [1,1], dim=1)
                labels = torch.cat([f1,f3[neg_ids]], dim=0)
            
            if 'triplet' in loss_method:
                labels,f3 = torch.split(labels, [1,1], dim=1)
                f1,f3,f2,f4 = torch.split(mid_output, [bsz, neg_bsz, bsz, neg_bsz], dim=0)
                mid_output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                output = output[0:bsz]
                if neg_bsz>0: # get triplet loss iff neg exists
                    anchor, positive, negative = f1[neg_ids], f2[neg_ids], f3
                    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
                    loss_triplet = criterion(anchor, positive, negative)
            elif supcon_incl_negs and neg_bsz>0:
                f1,f3,f2,f4 = torch.split(mid_output, [bsz, neg_bsz, bsz, neg_bsz], dim=0)
                mid_output = torch.cat([
                    torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1),
                    torch.cat([f3.unsqueeze(1), f4.unsqueeze(1)], dim=1)], 
                    dim=0)
                output = output[0:bsz+neg_bsz]
            else:
                f1,f2 = torch.split(mid_output, [bsz, bsz], dim=0)
                mid_output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                output = output[0:bsz]
            
            # to do: check magnitude if need to scale
            if 'triplet' not in loss_method:
                criterion = SupConLoss(temperature=0.7, base_temperature=3)
                loss_supcon = criterion(mid_output, labels.view(-1))

            ce_criterion = nn.CrossEntropyLoss(reduction='none', weight=self.class_weight)
            loss_ce = ce_criterion(output.view(-1, output.shape[-1]), labels.view(-1))

            if loss_method == 'ce+supcon' or loss_method == 'supcon+ce':
                # supcon_weight = 40
                # print(f'loss_supcon: {loss_supcon} | loss_ce: {loss_ce.mean()}')
                loss = loss_supcon + loss_ce.mean()
            elif loss_method == 'ce+triplet' or loss_method == 'triplet+ce':
                triplet_weight = 4
                # print(f'loss_triplet: {loss_triplet} | loss_ce: {loss_ce.mean()}')
                loss = loss_triplet*triplet_weight + loss_ce.mean()
            elif loss_method == 'ce':
                loss = loss_ce
            elif loss_method == 'triplet':
                loss = loss_triplet
            else:
                loss = loss_supcon

            return loss, output, mid_output[:,1], labels
        else:
            return output, mid_output

