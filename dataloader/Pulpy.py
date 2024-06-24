import json
from pathlib import Path

import numpy as np
import torch
import torchio as tio

from torch.utils.data import DataLoader

class Pulpy(tio.SubjectsDataset):
  def __init__(self, root, splits, config, transform=None, **kwargs):
    self.config = config 

    if not isinstance(splits, list):
      splits = [splits]

    root = Path(root)
    if not isinstance(splits, list):
      splits = [splits]

    subjects_list = self._get_subjects_list(root, splits)
    super().__init__(subjects_list, transform, **kwargs)

  def _get_subjects_list(self, root, splits):
    subjects_dir = root / 'Dataset'
    splits_path = root / "splits.json"

    with open(splits_path) as splits_file:
      json_splits = json.load(splits_file)
    
    subjects = []
    for split in splits:
      for patient in json_splits[split]:
        data_path = subjects_dir / patient / 'data.nii.gz'
        if not data_path.is_file():
          raise ValueError(f'Missing data file for patient {patient} ({data_path})')
        
        subject_dict = {
          'partition': split,
          'patient': patient,
          'data': tio.ScalarImage(data_path),
        }

        gt_path = subjects_dir / patient / f'{self.config.gt}.nii.gz'
        if not gt_path.is_file():
          raise ValueError(f'Missing gt file for patient {patient} ({gt_path})')
        
        subject_dict['gt'] = tio.LabelMap(gt_path)

        subjects.append(tio.Subject(**subject_dict))
      print(f"Loaded {len(subjects)} patients for split {split}")
    return subjects
  

  def get_queue(self, config):
    if config.sampler_type == 'label': 
      label_props = {0:1, 1: 3}
      sampler = tio.LabelSampler(patch_size=config.patch_shape, label_name='gt', label_probabilities=label_props)
    else:
      sampler = tio.UniformSampler(patch_size=config.patch_shape)
    queue = tio.Queue(
      subjects_dataset=self,
      max_length=config.max_length,
      samples_per_volume=config.samples_per_volume,
      sampler=sampler,
      num_workers=config.num_workers,
      shuffle_subjects=True,
      shuffle_patches=True,
      start_background=True,
    )
    return queue

  def get_loader(self, config, drop_last=False):
    queue = self.get_queue(config)    
    loader = DataLoader(queue, batch_size=config.batch_size, num_workers=0, pin_memory=True, drop_last=drop_last)
    return loader
