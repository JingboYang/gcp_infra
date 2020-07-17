import copy
import re

from collections import defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix as skcm
import torch

from utils.tb_plot_helper import plot_simple, plot_confusion_matrix, plot_tsne
from utils.tb_logger import Logger

def detach_tensor(tensor):
    if type(tensor) != np.ndarray:
        if type(tensor) == list:
            return np.ndarray(tensor)
        else:
            return tensor.cpu().detach().numpy()
    return tensor

def cash_to_tensor(tensor):
    if type(tensor) == np.ndarray:
        return torch.Tensor(tensor)
    return tensor

# https://stackoverflow.com/questions/32461246/how-to-get-top-3-or-top-n-predictions-using-sklearns-sgdclassifier
def topk(*args):
    tag_mode = args[0]
    k = args[1]
    info_dict = args[2]

    ground_truth = info_dict['target']
    output = info_dict['output']

    best_k = np.argsort(output, axis=1)[:, -k:]
    # ts = np.argmax(ground_truth, axis=1)
    ts = ground_truth
    successes = 0
    for i in range(len(ts)):
      if ts[i] in best_k[i,:]:
        successes += 1

    acc = float(successes) / len(ts)

    return 'scalar', f'TopK@{k}', acc

def confusion_matrix(*args):
    tag_mode = args[0]
    info_dict = args[1]

    ground_truth = info_dict['target']
    output = info_dict['output']
    predicted = np.argmax(output, axis=1)
    
    cm = skcm(ground_truth, predicted)
    cm_plot = plot_confusion_matrix(cm)

    return 'image', f'Confusion-Matrix', cm_plot

def tsne(*args):
    tag_mode = args[0]
    perplexity = args[1]
    info_dict = args[2]

    ground_truth = info_dict['target']
    embedding = info_dict['embedding']

    tsne_plot = plot_tsne(embedding, ground_truth, perplexity=perplexity)

    return 'image', f't-SNE@perplexity={perplexity}', tsne_plot


class Evaluator:

    def __init__(self, eval_list):
        self.result_storage = defaultdict(lambda : defaultdict(list))
        self.eval_list = eval_list

    def store(self, tag_mode, tag_name, data):
        data = detach_tensor(data)
        self.result_storage[tag_mode][tag_name].extend(data)
    
    def clear(self, tag_mode):
        self.result_storage[tag_mode] = defaultdict(list)

    def evaluate(self, tag_mode):

        scalar_results = {}
        image_results = {}

        arg_0 = tag_mode
        for eval_name, params in self.eval_list:
            args = [arg_0]
            args.extend(params)
            args.append(self.result_storage[tag_mode])

            result_type, eval_name, value = globals()[eval_name](*args)

            if result_type == 'scalar':
                scalar_results[f'{tag_mode}:{eval_name}'] = value
            elif result_type == 'image':
                image_results[f'{tag_mode}:{eval_name}'] = value
            else:
                raise NotImplementedError
        
        return scalar_results, image_results

    def report(self, epoch, epoch_step, total_steps,
                scalar_results, image_results):

        logger = Logger.get_unique_logger()
        logger.log_scalars(scalar_results,
                           total_steps,
                           epoch_step,
                           epoch)

        logger.log_images(image_results,
                            total_steps,
                            epoch_step,
                            epoch)