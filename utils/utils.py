
from sklearn.neighbors import NearestNeighbors
import numpy as np

class AggregateAvg(object):
  """Bookkeeping of <key, value> pairs.

  The registered keys are expected to be passed later.
  """

  def __init__(self, key_list=None):
    super(AggregateAvg, self).__init__()
    self.steps = 0
    self.container = dict()
    if key_list is not None:
      self.reset_keys(key_list)

  def add(self, values):
    assert isinstance(values, dict), "<key, value> pairs expected."

    if not self.container:
      self.reset_keys(values)

    for key, value in self.container.items():
      if isinstance(value, list) or isinstance(value, np.ndarray):
        self.container[key].extend(values[key])
      else:
        self.container[key] = value + values[key]
    self.steps += 1

  def summary(self):
    summary_dict = dict()
    for key, value in self.container.items():
      if isinstance(value, list):
        summary_dict[key] = np.array(value).mean()
      else:
        summary_dict[key] = value/self.steps
    return summary_dict

  def summary_and_reset(self):
    summary_dict = dict()
    steps = max(self.steps, 1)
    for key, value in self.container.items():
      if isinstance(value, list):
        summary_dict[key] = np.array(value).mean()
        self.container[key] = list()
      else:
        summary_dict[key] = value / self.steps
        self.container[key] = 0.0
    self.steps = 0
    return summary_dict, steps

  def reset_keys(self, val_dict):
    for key, value in val_dict.items():
      if isinstance(value, list) or isinstance(value, np.ndarray):
        self.container[key] = list()
      else:
        self.container[key] = 0.0

  def reset(self):
    for key, value in self.container.items():
      if isinstance(value, list):
        self.container[key] = list()
      else:
        self.container[key] = 0.0
    self.steps = 0

def evaluate_chamfer(targets, predictions, return_all=True, to_origin = True, ignore_pen_step = True, ignore_pen = True):
    targets_ = list()
    predictions_ = list()
    for t_, p_ in zip(targets, predictions):
        if to_origin:
            t_ = t_ - t_[0, :]
            p_ = p_ - p_[0, :]
        if ignore_pen and ignore_pen_step:
            t_ = t_[:-1, 0:2]
            p_ = p_[:-1, 0:2]
        elif ignore_pen:
            t_ = t_[:, 0:2]
            p_ = p_[:, 0:2]
        elif ignore_pen_step:
            t_ = t_[:-1]
            p_ = p_[:-1]
        targets_.append(t_)
        predictions_.append(p_)
    results = [chamfer_distance_np_var_len_normalized((gt, pred)) for gt, pred in zip(targets, predictions)]
    if not return_all:
        results = np.array(res).mean()
    return results

def chamfer_distance_np_var_len_normalized(arrays):
    """Chamfer distance in numpy supporting arrays with different lengths.

    Args:
    arrays:
    Returns:
    """
    x, y = arrays
    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(x)
    min_y_to_x = x_nn.kneighbors(y)[0]
    y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(y)
    min_x_to_y = y_nn.kneighbors(x)[0]
    dist_y_to_x = np.mean(min_y_to_x)
    dist_x_to_y = np.mean(min_x_to_y)
    return dist_y_to_x + dist_x_to_y#, dist_y_to_x, dist_x_to_y