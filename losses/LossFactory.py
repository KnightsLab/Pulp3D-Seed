import torch

class LossFactory:
    def __init__(self, names, classes, weights=None):
        self.names = names
        if not isinstance(self.names, list):
            self.names = [self.names]

        print(f'Losses used: {self.names}')
        self.classes = classes
        self.weights = weights
        self.losses = {}
        for name in self.names:
            loss = self.get_loss(name)
            self.losses[name] = loss

    def get_loss(self, name):
        if name == 'Jaccard':
            from losses.JaccardLoss import JaccardLoss
            loss_fn = JaccardLoss(weight=self.weights)
        else:
            raise Exception(f"Loss function {name} can't be found.")

        return loss_fn

    def __call__(self, pred, gt, partition_weights):
        """
        SHAPE MUST BE Bx1xHxW
        :param pred:
        :param gt:
        :return:
        """
        assert pred.device == gt.device
        assert gt.device != 'cpu'

        cur_loss = []
        for loss_name in self.losses.keys():
            loss = self.losses[loss_name](pred, gt)
            if torch.isnan(loss.sum()):
                raise ValueError(f'Loss {loss_name} has some NaN')
            loss = loss * partition_weights
            cur_loss.append(loss.mean())
        return torch.sum(torch.stack(cur_loss))
