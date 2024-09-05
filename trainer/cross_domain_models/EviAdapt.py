import sys

sys.path.append("..")
from utils import *
from data.mydataset import data_generator, Load_Dataset, stage_dataset_generator
import torch
from torch import nn
from trainer.train_eval import evaluate
import copy
import numpy as np
import time
from models.models import get_backbone_class, Model
import os

def cross_domain_train(device, dataset, dataset_configs, hparams, backbone, data_path, src_id, tgt_id, run_id, logger):

    dataset_file = torch.load(os.path.join(data_path, f"train_{tgt_id}.pt"))
    tgt_dataset = Load_Dataset(dataset_file, dataset_configs)
    tgt_train_dl = torch.utils.data.DataLoader(dataset=tgt_dataset, batch_size=hparams["batch_size"], shuffle=False, drop_last=False, num_workers=0)    
    tgt_trainX = tgt_dataset.x_data
    
    dataset_file = torch.load(os.path.join(data_path, f"train_{src_id}.pt"))
    src_dataset = Load_Dataset(dataset_file, dataset_configs)
    src_trainX, src_trainY = src_dataset.x_data, src_dataset.y_data
    src_trainY_cluster = src_trainY.clone()
    src_trainY_cluster[src_trainY <= 0.15] = 0 
    src_trainY_cluster[src_trainY > 0.66] = 2
    src_trainY_cluster[(src_trainY > 0.15) & (src_trainY <= 0.66 )] = 1
    src_trainY_cluster = src_trainY_cluster.cpu().numpy()

    tgt_test_dl = data_generator(data_path, tgt_id, dataset_configs, hparams, "test")

    logger.info('Restore source pre_trained model...')

    checkpoint = torch.load(f'./trained_models/{dataset}/single_domain/pretrained_{backbone}_EVI_{src_id}.pt')


    # pretrained source model
    source_model = Model(dataset_configs, backbone).to(device) 

    logger.info('=' * 89)
 
    if hparams['pretrain']:
        source_model.load_state_dict(checkpoint['state_dict'])
        source_model.eval()
        set_requires_grad(source_model, requires_grad=False)
        
        # initialize target model
        target_model = Model(dataset_configs, backbone).to(device)
        target_model.load_state_dict(checkpoint['state_dict'])
        target_encoder = target_model.feature_extractor
        target_encoder.train()
        set_requires_grad(target_encoder, requires_grad=True)
        set_requires_grad(target_model.regressor, requires_grad=False)
    else:
        source_model.train()
        target_model = source_model

    
    # criterion
    criterion = RMSELoss()
    same_align_criterion = Stage_Wise_Alignment()
    # optimizer
    target_optim = torch.optim.AdamW(target_encoder.parameters(), lr=hparams['learning_rate'], betas=(0.5, 0.9))   
    
    best_score, best_loss = 1e10, 1e10       
    for epoch in range(1, hparams['num_epochs'] + 1):
        if epoch % 5 == 1:
            _, _, _, [var_list, sigma_list], pred_tgt_label, _ = evaluate(target_model, tgt_train_dl, criterion, dataset_configs, device, False)
            pred_tgt_label = np.array(pred_tgt_label)
            median = np.quantile(pred_tgt_label, 0.3)

            tgt_trainY_cluster = pred_tgt_label.copy()
            tgt_trainY_cluster[pred_tgt_label > median] = 2
            tgt_trainY_cluster[pred_tgt_label <= median] = 1
     
        stage_train_dl = stage_dataset_generator(src_trainX, src_trainY, src_trainY_cluster, tgt_trainX, pred_tgt_label, tgt_trainY_cluster, random_flag=True, batch_size=hparams['batch_size'])
                
        total_loss = 0
        start_time = time.time()
        for step, (target_x, source_pos_x, tgt_y, pos_y) in enumerate(stage_train_dl):
            target_optim.zero_grad()

            source_pos_x, target_x = source_pos_x.to(device), target_x.to(device) 
            tgt_y, pos_y = tgt_y.to(device), pos_y.to(device)
            
            
            src_pos_outputs, src_pos_fea = source_model(source_pos_x)                 
            tgt_outputs, tgt_fea = target_model(target_x)
              
            src_pos_m, src_pos_v, src_pos_alpha, src_pos_beta = src_pos_outputs
            tgt_m, tgt_v, tgt_alpha, tgt_beta = tgt_outputs
              
            src_pos_DER = torch.cat((src_pos_v, src_pos_alpha, src_pos_beta), dim=1)
            tgt_DER = torch.cat((tgt_v, tgt_alpha, tgt_beta), dim=1)

            tgt_batch_cluster = tgt_y.clone()
            tgt_batch_cluster[tgt_y > median] = 2
            tgt_batch_cluster[tgt_y <= median] = 1


            loss = 0
            for stage in range(1, 3):
                stage_index = tgt_batch_cluster==stage               
                loss += same_align_criterion(tgt_DER[stage_index], src_pos_DER[stage_index])


            loss.backward()
            target_optim.step()
            total_loss += loss.item()
                
        mean_loss = total_loss / (step+1)
        
        logger.info(f'Epoch: {epoch:02}')
        logger.info(f'target_loss:{mean_loss} ')
        if epoch % 1 == 0:
            src_only_loss, src_only_score, _, _, _, _ = evaluate(source_model, tgt_test_dl, criterion, dataset_configs,device)
            test_loss, test_score, _, _, _, _ = evaluate(target_model, tgt_test_dl, criterion, dataset_configs,device)
            if best_score > test_score:
                best_loss, best_score = test_loss, test_score
            logger.info(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
            logger.info(f'DA RMSE:{test_loss} \t DA Score:{test_score}')
            
            
    src_only_loss, src_only_score, src_only_fea, _, pred_labels, true_labels = evaluate(source_model, tgt_test_dl, criterion, dataset_configs,device)
    test_loss, test_score, target_fea, _, pred_labels_DA, true_labels_DA = evaluate(target_model, tgt_test_dl, criterion, dataset_configs,device)
       
    
    logger.info(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
    logger.info(f'After DA RMSE:{test_loss} \t After DA Score:{test_score}')
    
    return src_only_loss, src_only_score, test_loss, test_score, best_loss, best_score
