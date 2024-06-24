import os
import logging
import logging.config
import torch
import logging
import torchio as tio
import wandb

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.Pulpy import Pulpy
from losses.LossFactory import LossFactory
from models.ModelFactory import ModelFactory
from optimizers.OptimizerFactory import OptimizerFactory
from schedulers.SchedulerFactory import SchedulerFactory
from eval import Eval as Evaluator

class Experiment:
  def __init__(self, config, debug=False):
    self.config = config
    self.debug = debug
    self.epoch = 0
    self.eps = 1e-10
    self.metrics = {}

    self.num_classes = 1
    # load model
    model_name = self.config.model.name
    in_ch = 1
    emb_shape = [dim // 8 for dim in self.config.data_loader.patch_shape]

    self.model = ModelFactory(model_name, self.num_classes, in_ch, emb_shape).get()
    if torch.cuda.is_available():
      self.model = self.model.cuda()

    self.model = nn.DataParallel(self.model)
    wandb.watch(self.model, log_freq=10)

    # load optimizer
    optim_name = self.config.optimizer.name
    train_params = self.model.parameters()
    lr = self.config.optimizer.learning_rate

    self.optimizer = OptimizerFactory(optim_name, train_params, lr).get()

    # load scheduler
    sched_name = self.config.lr_scheduler.name
    sched_milestones = self.config.lr_scheduler.get('milestones', None)
    sched_gamma = self.config.lr_scheduler.get('factor', None)
    sched_patience = self.config.lr_scheduler.get('patience', None)

    self.scheduler = SchedulerFactory(
      sched_name,
      self.optimizer,
      milestones=sched_milestones,
      gamma=sched_gamma,
      mode='max',
      verbose=True,
      patience=sched_patience
    ).get()

    # load loss
    self.loss = LossFactory(self.config.loss.name, classes=self.num_classes)

    # load evaluator
    self.evaluator = Evaluator(self.config, classes=self.num_classes)

    # Accumlator
    self.accumlation_iter = self.config.data_loader.accumlation_iter

    self.train_dataset = Pulpy(
      root=self.config.data_loader.dataset,
      config=self.config.data_loader,
      splits='train',
      transform=tio.Compose([
        tio.CropOrPad(self.config.data_loader.resize_shape, padding_mode=0),
        self.config.data_loader.preprocessing,
        self.config.data_loader.augmentations,
      ]),
    )
    self.val_dataset = Pulpy(
      root=self.config.data_loader.dataset,
      config=self.config.data_loader,
      splits='val',
      transform=tio.Compose([
        self.config.data_loader.preprocessing,
      ])
    )
    self.test_dataset = Pulpy(
      root=self.config.data_loader.dataset,
      config=self.config.data_loader,
      splits='test',
      transform=tio.Compose([
        self.config.data_loader.preprocessing,
      ])
    )

    # queue start loading when used, not when instantiated
    self.train_loader = self.train_dataset.get_loader(self.config.data_loader)

    if self.config.trainer.reload:
      self.load()

  def save(self, name):
    if '.pth' not in name:
      name = name + '.pth'
    path = os.path.join(self.config.project_dir, self.config.title, 'checkpoints', name)
    logging.info(f'Saving checkpoint at {path}')
    state = {
      'title': self.config.title,
      'epoch': self.epoch,
      'state_dict': self.model.state_dict(),
      'optimizer': self.optimizer.state_dict(),
      'metrics': self.metrics,
    }
    torch.save(state, path)

  def load(self):
    path = self.config.trainer.checkpoint
    logging.info(f'Loading checkpoint from {path}')
    state = torch.load(path)

    if 'title' in state.keys():
      # check that the title headers (without the hash) is the same
      self_title_header = self.config.title[:-11]
      load_title_header = state['title'][:-11]
      if self_title_header == load_title_header:
        self.config.title = state['title']
    
    if state.get('optimizer', None) is not None:
      self.optimizer.load_state_dict(state['optimizer'])
    self.model.load_state_dict(state['state_dict'])
    self.epoch = state.get('epoch', 0) + 1

    if 'metrics' in state.keys():
      self.metrics = state['metrics']

  def extract_data_from_patch(self, patch):
    images = patch['data'][tio.DATA].float().cuda()
    gt = patch['gt'][tio.DATA].float().cuda()
    emb_codes = torch.cat((
      patch[tio.LOCATION][:,:3],
      patch[tio.LOCATION][:,:3] + torch.as_tensor(images.shape[-3:])
    ), dim=1).float().cuda()

    return images, gt, emb_codes

  def train(self):
    self.model.train()
    self.evaluator.reset_eval()

    data_loader = self.train_loader

    losses = []
    self.optimizer.zero_grad()
    for i, d in tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Train epoch {str(self.epoch)}'):
      images, gt, emb_codes = self.extract_data_from_patch(d)
      partition_weights = 1

      if self.num_classes == 1:
        gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
      else:
        gt_count = torch.sum(gt > 0, dim=list(range(1, gt.ndim)))

      if torch.sum(gt_count) == 0: continue
      partition_weights = (self.eps + gt_count) / torch.max(gt_count)

      preds = self.model(images, emb_codes)
      if self.num_classes == 1:
        assert preds.ndim == gt.ndim, f'Gt and output dimensions are not the same before loss. {preds.ndim} vs {gt.ndim}'
      
      loss = self.loss(preds, gt, partition_weights) 
      losses.append(loss.item() / self.accumlation_iter)
      loss.backward()

      if ((i + 1) % self.accumlation_iter == 0) or (i + 1 == len(data_loader)):
        self.optimizer.step()
        self.optimizer.zero_grad()
      
      if self.num_classes == 1:
        preds = (preds > 0.5)

      self.evaluator.compute_metrics(preds, gt)

    epoch_train_loss = sum(losses) / len(losses)
    epoch_iou, epoch_dice, epoch_hd95 = self.evaluator.mean_metric()

    self.metrics['Train'] = {
      'iou': epoch_iou,
      'dice': epoch_dice,
      'hd95': epoch_hd95,
    }

    wandb.log({
      f'Epoch': self.epoch,
      f'Train/Loss': epoch_train_loss,
      f'Train/Dice': epoch_dice,
      f'Train/IoU': epoch_iou,
      f'Train/HD95': epoch_hd95,
      f'Train/Lr': self.optimizer.param_groups[0]['lr']
    })

    return epoch_train_loss, epoch_iou, epoch_hd95


  def test(self, phase):
    self.model.eval()
    with torch.inference_mode():
      self.evaluator.reset_eval()
      losses = []

      if phase == 'Test':
        dataset = self.test_dataset
      elif phase == 'Validation':
        dataset = self.val_dataset

      for _, subject in tqdm(enumerate(dataset), total=len(dataset), desc=f'{phase} epoch {str(self.epoch)}'):
        sampler = tio.inference.GridSampler(
          subject,
          self.config.data_loader.patch_shape,
          0
        )
        loader = DataLoader(sampler, batch_size=self.config.data_loader.batch_size)
        aggregator = tio.inference.GridAggregator(sampler)
        gt_aggregator = tio.inference.GridAggregator(sampler)

        for _, patch in enumerate(loader):
          images, gt, emb_codes = self.extract_data_from_patch(patch)

          preds = self.model(images, emb_codes)
          aggregator.add_batch(preds, patch[tio.LOCATION])
          gt_aggregator.add_batch(gt, patch[tio.LOCATION])

        output = aggregator.get_output_tensor()
        gt = gt_aggregator.get_output_tensor()
        partition_weights = 1

        if self.num_classes == 1:
          gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
        else:
          gt_count = torch.sum(gt > 0, dim=list(range(1, gt.ndim)))
          
        if torch.sum(gt_count) != 0:
          partition_weights = (self.eps + gt_count) / (self.eps + torch.max(gt_count))

        loss = self.loss(output.unsqueeze(0), gt.unsqueeze(0), partition_weights)
        losses.append(loss.item())

        if self.num_classes == 1:
          output = output.squeeze(0)
          output = (output > 0.5)

        self.evaluator.compute_metrics(output.unsqueeze(0), gt.unsqueeze(0))

      epoch_loss = sum(losses) / len(losses)
      epoch_iou, epoch_dice, epoch_hd95 = self.evaluator.mean_metric()

      wandb.log({
          f'Epoch': self.epoch,
          f'{phase}/Loss': epoch_loss,
          f'{phase}/Dice': epoch_dice,
          f'{phase}/IoU': epoch_iou,
          f'{phase}/HD95': epoch_hd95
      })

      return epoch_iou, epoch_dice, epoch_hd95
