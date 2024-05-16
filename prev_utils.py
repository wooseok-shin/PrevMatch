import torch
import random
import numpy as np
from copy import deepcopy

def update_previous_list(prev_model_list, model, list_max_len=5):
    prev_model = deepcopy(model)
    for p in prev_model.parameters():
        p.requires_grad = False 

    if len(prev_model_list) < list_max_len:
        prev_model_list.append(prev_model)

    elif len(prev_model_list) == list_max_len:
        del(prev_model_list[0])                    # Delete the oldest model
        prev_model_list.append(prev_model)
    else:
        raise ValueError('Prev list length must be less than or equal to list_max_len')
        
def get_previous_logits(prev_model_list, img_u_w, max_num=1, random_select=True):    
    if random_select:
        num = np.random.randint(1, high=max_num + 1)   # select (1~K)
    else:
        num = max_num                                  # select K

    prev_models = random.sample(prev_model_list, k=min(num, len(prev_model_list)))       # select k (1~K) models
    weight_values = np.random.dirichlet(np.ones(len(prev_models)))                       # sample weight values from dirichlet distribution for randomized ensemble

    # Random Ensemble
    with torch.no_grad():
        for i, model in enumerate(prev_models):
            model.eval()
            if i == 0:
                prev_pred_u_w = model(img_u_w) * weight_values[i]
            else:
                prev_pred_u_w += model(img_u_w) * weight_values[i]
        
    return prev_pred_u_w.detach()
