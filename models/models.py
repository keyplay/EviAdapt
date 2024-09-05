import torch
from torch import nn
import math
from torch.autograd import Function
from torch.nn.utils import weight_norm
import torch.nn.functional as F

class Model(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs, backbone):
        super(Model, self).__init__()
        backbone_fe = get_backbone_class(backbone)
        self.feature_extractor = backbone_fe(configs)
        if configs.evidential:
            self.regressor = evi_regressor(configs)
        else:
            self.regressor = regressor(configs)
               
    def forward(self, x_in):
        feature = self.feature_extractor(x_in)
        pred_rul = self.regressor(feature)
        return pred_rul, feature
        
def get_model(backbone, configs):
    backbone_fe = get_backbone_class(backbone)
    feature_extractor = backbone_fe(configs)
    classifier = regressor(configs)
    network = nn.Sequential(feature_extractor, classifier)
    
    return network

def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


##################################################
##########  BACKBONE NETWORKS  ###################
##################################################
class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM, self).__init__()

        self.encoder = nn.LSTM(configs.input_channels, configs.lstm_hid, configs.lstm_n_layers, dropout=configs.dropout, batch_first=True, bidirectional=configs.lstm_bid)
        
    def forward(self, x_in):
        encoder_outputs, (hidden, cell) = self.encoder(x_in)
        features = encoder_outputs[:, -1:].squeeze()

        return features


class regressor(nn.Module):
    def __init__(self, configs):
        super(regressor, self).__init__()
        #self.logits = nn.Linear(configs.features_len, 1)
        self.logits= nn.Sequential(
            #nn.Linear(64, 64),
            #nn.ReLU(),
            nn.Linear(configs.features_len, configs.features_len),   
            #nn.Linear(configs.features_len * configs.final_out_channels, configs.features_len * configs.final_out_channels//2),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.features_len, configs.features_len),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.features_len, 1) )
        self.configs = configs

    def forward(self, x):

        predictions = self.logits(x)

        return predictions

class evi_regressor(nn.Module):
    def __init__(self, configs):
        super(evi_regressor, self).__init__()
        self.logits= nn.Sequential(
            nn.Linear(configs.features_len, configs.features_len),   
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.features_len, configs.features_len),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            DenseNormalGamma(configs.features_len, len(configs.quantiles)) )
        self.configs = configs

    def forward(self, x):

        predictions = self.logits(x)

        return predictions

class DenseNormalGamma(nn.Module):
    def __init__(self, input_dim, units):
        super(DenseNormalGamma, self).__init__()
        self.units = int(units)
        self.dense = nn.Linear(input_dim, 4 * self.units)

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(output, self.units, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta

    def get_config(self):
        return {'units': self.units}


torch.backends.cudnn.benchmark = True  # might be required to fasten TCN