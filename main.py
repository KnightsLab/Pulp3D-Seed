import sys
import os
import argparse
import logging
import logging.config
import shutil
import yaml
import random
import time
import torch
import wandb
import numpy as np

from os import path
from hashlib import shake_256
from munch import munchify, unmunchify

from dataloader.AugFactory import AugFactory
from experiments.ExperimentFactory import ExperimentFactory


# used to generate random names that will be appended to the
# experiment name
def timehash():
  t = time.time()
  t = str(t).encode()
  h = shake_256(t)
  h = h.hexdigest(5) # output len: 2*5=10
  return h.upper()

def setup(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  # Parse Arguments
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("-c", "--config", default="config.yaml", help="the config file to be used to run the experiment", required=True)
  arg_parser.add_argument("--verbose", action='store_true', help="Log also to stdout")
  arg_parser.add_argument("--debug", action='store_true', help="debug, no wandb")
  args = arg_parser.parse_args()

  # Read Configuration File
  if not os.path.exists(args.config):
    logging.info("Config file does not exist: {}".format(args.config))
    raise SystemExit

  # Munchify the dict to access entries with both dot notation and ['name']
  logging.info(f'Loading the config file...')
  config = yaml.load(open(args.config, "r"), yaml.FullLoader)
  config = munchify(config)

  # Deterministic Setup
  logging.info(f'setup to be deterministic')
  setup(config.seed)

  # WANDB
  # os.environ['WANDB_API_KEY'] = ''
  
  if args.debug:
    os.environ['WANDB_DISABLED'] = 'true'

  wandb.init(
    project="",
    entity="",
    config=unmunchify(config),
  )

  # Check if project_dir exists
  if not os.path.exists(config.project_dir):
    logging.error("Project_dir does not exist: {}".format(config.project_dir))
    os.makedirs(config.project_dir)

  # Preprocessing
  logging.info(f'loading preprocessing')
  if config.data_loader.preprocessing is None:
    preproc = []
  elif not os.path.exists(config.data_loader.preprocessing):
    logging.error("Preprocessing file does not exist: {}".format(config.data_loader.preprocessing))
    preproc = []
  else:
    with open(config.data_loader.preprocessing, 'r') as preproc_file:
      preproc = yaml.load(preproc_file, yaml.FullLoader)
  config.data_loader.preprocessing = AugFactory(preproc).get_transform()

  # Augmentations
  logging.info(f'loading augmentations')
  if config.data_loader.augmentations is None:
    aug = []
  elif not os.path.exists(config.data_loader.augmentations):
    logging.warning(f'Augmentations file does not exist: {config.augmentations}')
    aug = []
  else:
    with open(config.data_loader.augmentations) as aug_file:
      aug = yaml.load(aug_file, yaml.FullLoader)
  config.data_loader.augmentations = AugFactory(aug).get_transform()

  # make title unique to avoid overriding
  config.title = f'{config.title}_{timehash()}'

  logging.info(f'Instantiation of the experiment')

  # Experiment initiation
  experiment = ExperimentFactory(config, args.debug).get()
  logging.info(f'experiment title: {experiment.config.title}')

  project_dir_title = os.path.join(experiment.config.project_dir, experiment.config.title)
  os.makedirs(project_dir_title, exist_ok=True)
  logging.info(f'project directory: {project_dir_title}')

  # Setup logger's handlers
  file_handler = logging.FileHandler(os.path.join(project_dir_title, 'output.log'))
  log_format = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
  file_handler.setFormatter(log_format)
  logger.addHandler(file_handler)

  if args.verbose:
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(log_format)
    logger.addHandler(stdout_handler)

  # Copy config file to project_dir, to be able to reproduce the experiment
  copy_config_path = os.path.join(project_dir_title, 'config.yaml')
  shutil.copy(args.config, copy_config_path)

  if not os.path.exists(experiment.config.data_loader.dataset):
    logging.error("Dataset path does not exist: {}".format(experiment.config.data_loader.dataset))
    raise SystemExit

  # pre-calculate the checkpoints path
  checkpoints_path = path.join(project_dir_title, 'checkpoints')

  if not os.path.exists(checkpoints_path):
    os.makedirs(checkpoints_path)

  if experiment.config.trainer.reload and not os.path.exists(experiment.config.trainer.checkpoint):
    logging.error(f'Checkpoint file does not exist: {experiment.config.trainer.checkpoint}')
    raise SystemExit


  best_val = float('-inf')
  best_test = { 'value': float('-inf'), 'epoch': -1 }

  # Train the model
  if config.trainer.do_train:
    for epoch in range(experiment.epoch, config.trainer.epochs+1):
      train_iou, train_dice, train_hd95 = experiment.train()
      logging.info(f'Epoch {epoch} Train IoU: {train_iou}')
      logging.info(f'Epoch {epoch} Train Dice: {train_dice}')
      logging.info(f'Epoch {epoch} Train HD95: {train_hd95}')

      val_iou, val_dice, val_hd95 = experiment.test(phase="Validation")
      logging.info(f'Epoch {epoch} Val IoU: {val_iou}')
      logging.info(f'Epoch {epoch} Val Dice: {val_dice}')
      logging.info(f'Epoch {epoch} Val HD95: {val_hd95}')

      if val_iou < 1e-05 and experiment.epoch > 15:
        logging.warning('WARNING: drop in performances detected.')

      optim_name = experiment.optimizer.name
      sched_name = experiment.scheduler.name

      if experiment.scheduler is not None:
        if optim_name == 'SGD' and sched_name == 'Plateau':
          experiment.scheduler.step(val_dice)
        else:
          experiment.scheduler.step(epoch)

      if epoch % 5 == 0:
        test_iou, test_dice, test_hd95 = experiment.test(phase="Test")
        logging.info(f'Epoch {epoch} Test IoU: {test_iou}')
        logging.info(f'Epoch {epoch} Test Dice: {test_dice}')
        logging.info(f'Epoch {epoch} Test HD95: {test_hd95}')

        if test_dice > best_test['value']:
          best_test['value'] = test_dice
          best_test['epoch'] = epoch
          experiment.save(f'test__{epoch}.pth')  

      experiment.save('last.pth')

      if val_dice > best_val:
        best_val = val_dice
        experiment.save('best.pth')

      experiment.epoch += 1

      logging.info(f"Best test Dice found: {best_test['value']} at epoch: {best_test['epoch']}")

  # Test the model
  if config.trainer.do_test:
    logging.info('Testing the model...')
    experiment.load()
    test_iou, test_dice, test_hd95 = experiment.test(phase="Test")
    logging.info(f'Test results IoU: {test_iou}\nDice: {test_dice}\nHD95: {test_hd95}')
