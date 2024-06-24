import logging
import torchio as tio

class AugFactory:
  def __init__(self, aug_list):
    self.aug_list = aug_list
    self.transforms = self.factory(self.aug_list, [])

  def log(self):
    """
    save the list of aug for this experiment to the default log file
    :param path:
    :return:
    """
    logging.info('going to use the following augmentations:: %s', self.aug_list)

  def factory(self, auglist, transforms):
    for aug in auglist:
      if aug == 'OneOf':
        transforms.append(tio.OneOf(self.factory(auglist[aug], [])))
      else:
        try:
          kwargs = {}
          for param, value in auglist[aug].items():
            kwargs[param] = value
          transforms.append(getattr(tio, aug)(**kwargs))
        except:
          raise Exception(f"this transform is not valid: {aug}")
    return transforms

  def get_transform(self):
    """
    return the transform object
    :return:
    """
    return tio.Compose(self.transforms)
