import torch
import torch.nn.functional as F
from statistics import mean
from medpy.metric.binary import hd95

class Eval:
    def __init__(self, config, classes=1):
        self.iou_list = []
        self.dice_list = []
        self.hd95_list = []
        self.num_classes=classes
        self.config = config

    def reset_eval(self):
        self.iou_list.clear()
        self.dice_list.clear()
        self.hd95_list.clear()

    def compute_metrics(self, pred, gt, print_val=False):
        pred = pred.detach()
        gt = gt.detach()

        if torch.cuda.is_available():
            pred = pred.cuda()
            gt = gt.cuda()

            pred = pred.to(torch.uint8)
            gt = gt.to(torch.uint8)

            pred = pred[None, ...] if pred.ndim == 3 else pred
            gt = gt[None, ...] if gt.ndim == 3 else gt

            pred_count = torch.sum(pred == 1, dim=list(range(1, pred.ndim)))
            gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
            iou, dice = self.iou_and_dice(pred, gt)
            
            self.iou_list.append(iou)
            self.dice_list.append(dice)

            hd95_val = 0
            if torch.sum(pred_count) > 0 and torch.sum(gt_count) > 0:
                hd95_val = self.compute_hd95(pred.squeeze(), gt.squeeze())
                self.hd95_list.append(hd95_val)

        if print_val:
            print(f'iou={iou}, dice={dice}, h95={hd95_val}')
        
    def iou_and_dice(self, pred, gt):
        eps = 1e-6
        intersection = (pred & gt).sum()
        dice_union = pred.sum() + gt.sum()
        iou_union = dice_union - intersection

        iou = (intersection + eps) / (iou_union + eps)
        dice = (2 * intersection + eps) / (dice_union + eps)

        return iou.item(), dice.item()

    def compute_hd95(self, pred, gt):
        return hd95(pred.cpu().numpy(), gt.cpu().numpy())
    
    def mean_metric(self):
        iou = 0 if len(self.iou_list) == 0 else mean(self.iou_list)
        dice = 0 if len(self.dice_list) == 0 else mean(self.dice_list)
        hd95_val = 0 if len(self.hd95_list) == 0 else mean(self.hd95_list)

        self.reset_eval()
        return iou, dice, hd95_val
