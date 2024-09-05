## The cuurent hyper-parameters values are not necessarily the best ones for a specific risk.
def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class CMAPSS():
    def __init__(self):
        super(CMAPSS, self).__init__()
        self.train_params = {
            'num_epochs': 20,
            'batch_size': 256,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0, #0.5
            'pretrain': True,
            'save': True

        }
        self.alg_hparams = {
            'NO_ADAPT': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1, 'pretrain_epochs': 30},
            'TARGET_ONLY': {'learning_rate': 1e-3, 'trg_cls_loss_wt': 1},
            "EviAdapt": {
                "learning_rate": 5e-5,
            },
        }


class NCMAPSS():
    def __init__(self):
        super(NCMAPSS, self).__init__()
        self.train_params = {
            'num_epochs': 20,
            'batch_size': 256,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,
            'pretrain': True,
            'save': True
        }
        self.alg_hparams = {
            'NO_ADAPT': {'learning_rate': 1e-3, 'src_cls_loss_wt': 1, 'pretrain_epochs': 30},
            'TARGET_ONLY': {'learning_rate': 1e-3, 'trg_cls_loss_wt': 1},
            "EviAdapt": {
                "learning_rate": 5e-5,
            },
        }