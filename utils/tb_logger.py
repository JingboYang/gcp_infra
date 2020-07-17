'''
Originally created for Stanford Spring 2019 CS341
Jingbo Yang, Ruge Zhao, Meixian Zhu
Pytorch-specific implementation

Adapted from Stanford AI for Healthcare Bootcamp deep learning infrastructure
'''

import datetime
import getpass
import json
import os
from pathlib import Path
import pickle
import sys
import socket
import shutil
import tempfile

import torch
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from utils.gc_storage import GCStorage, GCOpen


def find_tf_event(name_suffix, local_cache_folder):
    for f in os.listdir(local_cache_folder):
        last = f.split('.')[-1]
        if name_suffix == last:
            return f

def makezip(fname, file_dir, remove_original=False):

    tmp_dir = tempfile.mkdtemp()
    shutil.make_archive(tmp_dir + '/' + fname, 'zip', file_dir, '.')

    # Do not remove original if they are the actual original
    if remove_original:
        shutil.rmtree(file_dir)

    return tmp_dir + '/' + fname + '.zip'

def copy_to_temp(root_folder, ignore_files):
    tmp_dir = tempfile.mkdtemp()
    os.rmdir(tmp_dir)       # Welp let's remove it first then copy stuff there again
    shutil.copytree(root_folder, tmp_dir, ignore=shutil.ignore_patterns(*ignore_files))

    return tmp_dir, os.path.basename(os.path.abspath(root_folder))

def get_ignore_files(root_folder):
    ignore_files = ['.git', '.vscode',]
    with open(f'{root_folder}/.gitignore', 'r') as f:
        lines = f.readlines()

        for l in lines:
            l = l.strip()
            if l and '#' != l[0]:
                if l[-1] == '/' :
                    l = l[:-1]
                ignore_files.append(l)
    
    return ignore_files


class Logger(object):
    """Class for logging output."""

    unique_logger = None
    @staticmethod
    def get_unique_logger():
        if Logger.unique_logger is None:
            raise ValueError('Unable to find a unique logger.')
        return Logger.unique_logger

    def __init__(self, gc, exp_folder):
        """Both general_log_ath and ouptuts_folder
           should be in heavy logging.
        """
        try:
            logger = None
            logger = Logger.get_unique_logger()
        except ValueError:
            # If error then no logger has been created yet
            pass

        if logger is not None:
            raise ValueError('There should only be 1 global unique logger')

        self.gc = gc

        # self.general_log_file = general_log_path.open('w')
        self.general_log_file = GCOpen(exp_folder / 'log.txt', 'w', self.gc)
        self.general_log_file.open()

        self.exp_folder = exp_folder
        self.file_outputs_dir = exp_folder / 'output_files'
        self.local_outputs_dir = self.gc.local_cache / exp_folder / 'output_files'
        # self.file_outputs_dir.mkdir(exist_ok=True)

        exp_name = str(exp_folder).split('/')[-1]

        self.summary_writer = SummaryWriter(log_dir=self.local_outputs_dir,
                                            filename_suffix='.' + exp_name)
        tf_filename = find_tf_event(exp_name, self.local_outputs_dir)
        self.sw_local_path = Path(self.local_outputs_dir) / tf_filename
        self.sw_gc_path = exp_folder / tf_filename

        self.log("Starting new experiment at " +
                 datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.log("User: " + getpass.getuser())
        self.log("Host: " + socket.gethostname())

        Logger.unique_logger = self

    def sync_summary_writer(self):
        self.summary_writer.flush()
        self.gc.upload(self.sw_local_path, self.sw_gc_path)

    def log(self, *args):
        """Writes args to the log file."""
        self.log_stdout(*args)
        print(*args, file=self.general_log_file.file)
        self.general_log_file.flush()

    def log_stdout(self, *args):
        """Writes args to the console."""
        print(*args, file=sys.stdout)
        sys.stdout.flush()

    def log_scalars(self, scalar_dict,
                    iterations, steps_per_epoch=None,
                    step_in_epoch=None, cur_epoch=None,
                    print_to_stdout=True):
        """Log all values in a dict as scalars to TensorBoard."""
        if len(scalar_dict) != 0:
            thing1 = f'/{step_in_epoch}/{steps_per_epoch}' \
                                        if step_in_epoch is not None and \
                                           steps_per_epoch is not None \
                                        else ""
            thing2 = f'@Epoch {cur_epoch}' if cur_epoch is not None else ""
            self.log_stdout(f'Step {iterations}{thing1}{thing2} Scalars')

        for k, v in scalar_dict.items():
            if print_to_stdout:
                temp_k = k.replace(':', '/')
                self.log_stdout(f'\t[{temp_k}: {v:.5g}]')
            k = k.replace(':', '/')  # Group in TensorBoard.
            self.summary_writer.add_scalar(k, v, iterations)

        self.sync_summary_writer()

    def log_images(self, image_dict,
                   iterations, step_in_epoch=None, cur_epoch=None,
                   save_to_outputs=True, include_iter=False):
        """Log all images in a dict as images to TensorBoard."""
        if len(image_dict) != 0:
            thing1 = f'/{step_in_epoch}' if step_in_epoch is not None else ""
            thing2 = f'@Epoch {cur_epoch}' if cur_epoch is not None else ""
            self.log_stdout(f'Step {iterations}{thing1}{thing2} Images')

        for k, v in image_dict.items():
            np_image, plt_figure = v
            temp_k = k.replace(':', '/')

            if save_to_outputs:
                if include_iter:
                    img_name = k.replace(':', '_') + f'_{iterations}.png'
                else:
                    img_name = k.replace(':', '_') + '.png'
                self.log_image(plt_figure, img_name)
                self.log_stdout(f'\t[{temp_k}: Saved]')

            self.summary_writer.add_image(temp_k,
                                          np_image,
                                          iterations,
                                          dataformats='HWC')
        self.sync_summary_writer()

    def log_text(self, text_dict,
                 iterations, step_in_epoch=None, cur_epoch=None,
                 print_to_stdout=True):
        """Log all text in a dict to TensorBoard."""
        if len(text_dict) != 0:
            thing1 = f'/{step_in_epoch}' if step_in_epoch is not None else ""
            thing2 = f'@Epoch {cur_epoch}' if cur_epoch is not None else ""
            self.log_stdout(f'Step {iterations}{thing1}{thing2} Texts')

        for k, v in text_dict.items():
            if print_to_stdout:
                temp_k = k.replace(':', '/')
                self.log_stdout(f'\t[{temp_k}: {v}]')
            k = k.replace(':', '/')  # Group in TensorBoard.

            self.summary_writer.add_text(k,
                                         v,
                                         iterations)
        self.sync_summary_writer()

    def log_numpy(self, output, filename):
        with GCOpen(self.file_outputs_dir / (filename + '.npy'), 'wb', self.gc) as f:
            np.save(f, output)

    def log_data(self, data, filename):
        with GCOpen(self.file_outputs_dir / (filename + '.pickle'), 'wb', self.gc) as f:
            pickle.dump(data, f)

    def log_image(self, fig, filename):
        with GCOpen(self.file_outputs_dir / filename, 'wb', self.gc) as f:
            fig.savefig(f)

    def log_json(self, dictionary, filename):
        with GCOpen(self.file_outputs_dir / (filename + '.json'), 'w', self.gc) as f:
            json.dump(dictionary, f)

    def log_sourcecode(self, main_file_dir):
        
        ignore_files = get_ignore_files(main_file_dir)
        copy_folder, copy_base = copy_to_temp(main_file_dir, ignore_files)
        output_zip = makezip(copy_base, copy_folder, remove_original=True)  # Remove temp folder
        self.gc.upload(output_zip, self.exp_folder / 'source.zip')

    def log_model_state(self, state, epoch, step, save_fn):
        tmp_dir = tempfile.mkdtemp()
        temp_fname = f'model_state_{epoch}_{step}.ckpt'
        filename = f'{tmp_dir}/{temp_fname}'
        save_fn(state, filename)

        self.gc.upload(filename, self.exp_folder / 'output_files' / filename)
        shutil.rmtree(tmp_dir)

    def close(self):
        '''Closes the log file'''
        self.general_log_file.close()

