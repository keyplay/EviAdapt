import numpy as np
import os
import torch
domain = 'DS01'
win = 50
str = 10
smp = 100
max_rul = 88
keep_ratio=1


def load_array(sample_dir_path, unit_num, win_len, stride, sampling, train = True):
    if train:
        filename =  'Unit%s_win%s_str%s_smp%s_train.npz' %(int(unit_num), win_len, stride, sampling)
    else:
        filename = 'Unit%s_win%s_str%s_smp%s_test.npz' % (int(unit_num), win_len, stride, sampling)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)


    return loaded['sample'].transpose(2, 0, 1), loaded['label']

root_path = f'./data_set/Samples_whole_{domain}_{keep_ratio}'

unit_num_train = np.load(os.path.join(root_path,'unit_num_train.npy'), allow_pickle=True)
unit_num_test = np.load(os.path.join(root_path,'unit_num_test.npy'), allow_pickle=True)


def data_stack(root_path, unit_num, win, str, smp, train):
    samples = []
    labels = []
    for unit in unit_num:
        sample_array, label_array = load_array(root_path, unit, win, str, smp, train)
        # print(sample_array.shape)
        # print(label_array.shape)
        samples.append(sample_array)
        labels.append(label_array)

    samples = np.concatenate(samples,0)
    labels = np.concatenate(labels,0)
    index = np.arange(samples.shape[0])
    np.random.shuffle(index)
    samples = samples[index]
    labels = labels[index]

    samples = torch.Tensor(samples)
    labels = torch.Tensor(labels)
    if train:
        for i, v in enumerate(labels):
            if v>max_rul:
                labels[i] = 1
            else:
                labels[i] = v/max_rul

    return samples, labels
train_samples, train_labels = data_stack(root_path, unit_num_train, win, str, smp, train=True)
test_samples, test_labels = data_stack(root_path, unit_num_test, win, str, smp, train=False)

print(train_labels)
print(test_labels)

torch.save({'samples':train_samples, 'labels':train_labels},os.path.join('NCMAPSS', f'{domain}_{keep_ratio}_train.pt'))
torch.save({'samples':test_samples, 'labels':test_labels},os.path.join('NCMAPSS', f'{domain}_{keep_ratio}_test.pt'))
