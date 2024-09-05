import torch
from utils import scoring_func, quant_evi_loss

def train(model, train_dl, optimizer, criterion,config,device):
    model.train()
    epoch_loss = 0
    epoch_score = 0
    for inputs, labels in train_dl:
        src = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        output, feat = model(src)
        rul_loss = 0
        if config.evidential:
            mu, v, alpha, beta = output
            pred = torch.mean(mu.detach(), dim=1)
            for i, q in enumerate(config.quantiles):
                rul_loss += quant_evi_loss(labels.unsqueeze(-1), mu[:, i].unsqueeze(-1), v[:, i].unsqueeze(-1),
                               alpha[:, i].unsqueeze(-1), beta[:, i].unsqueeze(-1), q, coeff=3e-1)

        else:
            pred = output
            rul_loss = criterion(pred.squeeze(), labels)
        
        #denormalization
        pred  = pred * config.max_rul
        labels = labels * config.max_rul
        score = scoring_func(pred.squeeze() - labels)

        rul_loss.backward()
        if (type(model.feature_extractor).__name__=='LSTM'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # only for LSTM models
        optimizer.step()

        epoch_loss += rul_loss.item()
        epoch_score += score
    return epoch_loss / len(train_dl), epoch_score, pred, labels



def evaluate(model, test_dl, criterion, config, device, denorm_flag=True):
    model.eval()
    total_feas=[];total_labels=[]
    epoch_loss = 0
    epoch_score = 0
    predicted_rul = []
    true_labels = []
    var_list = []
    sigma_list = []
    with torch.no_grad():
        for inputs, labels in test_dl:
            src = inputs.to(device)          
            labels = labels.to(device)

            test_output, feat = model(src)
            if config.evidential:
                mu, v, alpha, beta = test_output
                var = torch.sqrt((beta /(v*(alpha - 1))))
                sigma = beta/(alpha-1.0)
                pred = torch.mean(mu, dim=1)
                var_list.append(var)
                sigma_list.append(sigma)
            else:
                pred = test_output
                
            # denormalize predictions
            if denorm_flag:
                pred = pred * config.max_rul
                if labels.max() <= 1:
                    labels = labels * config.max_rul
            rul_loss = criterion(pred.squeeze(), labels)
            score = scoring_func(pred.squeeze() - labels)
            epoch_loss += rul_loss.item()
            epoch_score += score
            total_feas.append(feat)
            total_labels.append(labels)

            predicted_rul += (pred.squeeze().tolist())
            true_labels += labels.tolist()
    
    if len(var_list) > 0:
        var_list = torch.cat(var_list, dim=0)
        sigma_list = torch.cat(sigma_list, dim=0)

    model.train()
    return epoch_loss / len(test_dl), epoch_score, torch.cat(total_feas), [var_list, sigma_list], predicted_rul,true_labels
