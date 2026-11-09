import torch
import torch.nn.functional as F
import math
import numpy as np

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)

def distribution_error(pred, target, range_min=1, range_max=8, bin_width=1):

    hist_pred = torch.histc(pred, bins=(range_max - range_min) // bin_width + 1, min=range_min, max=range_max)
    hist_true = torch.histc(target, bins=(range_max - range_min) // bin_width + 1, min=range_min, max=range_max)


    hist_pred = F.normalize(hist_pred, p=1, dim=0)
    hist_true = F.normalize(hist_true, p=1, dim=0)


    dist_matrix = torch.cdist(hist_pred.unsqueeze(0), hist_true.unsqueeze(0), p=1)
    error = torch.mean(torch.min(dist_matrix, dim=1).values)


    max_possible_error = (range_max - range_min) / bin_width
    normalized_error = error / max_possible_error

    accuracy = 1 - normalized_error

    return accuracy.item()

class Result(object):
    def __init__(self):
        self.distribution_error = 0
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def set_to_worst(self):
        self.distribution_error = np.inf
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, gpu_time, data_time):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target):
        valid_mask = target>0
        if not valid_mask.any():
            raise ValueError("Metrics require at least one positive target")
        output = output[valid_mask]
        target = target[valid_mask]

        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        positive_output = output.clamp_min(torch.finfo(output.dtype).eps)
        self.lg10 = float((log10(positive_output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(positive_output / target, target / positive_output)
        self.delta1 = float((maxRatio < 1.50).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / positive_output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())
        self.distribution_error = distribution_error(output, target)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_distribution_error = 0
        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_distribution_error += n * result.distribution_error
        self.sum_irmse += n*result.irmse
        self.sum_imae += n*result.imae
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_lg10 += n*result.lg10
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3
        self.sum_data_time += n*data_time
        self.sum_gpu_time += n*gpu_time

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count, 
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count, self.sum_data_time / self.count)
        avg.distribution_error = self.sum_distribution_error / self.count
        return avg