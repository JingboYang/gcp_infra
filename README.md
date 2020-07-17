# Infrastructure for Machine Learning Projects on Google Cloud Platform

Author: Jingbo Yang, Ruge Zhao, Meixian Zhu
Additional credits to: [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) and participants of Stanford AI for Healtcare Bootcamp

Concept for this repository orignated from CheXpert, then was adapted for [ElectreeScore](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2767367)(requires JAMA subscription). Migration toward GCP support was added after Stanford's 2019 Spring CS 341 [report](https://web.stanford.edu/class/cs341/project/Zhao-Zhu-Yang_report.pdf) offering then further strengthend during subsequent paper review process and for Stanford's 2019 Fall CS224W offering [report](https://web.stanford.edu/class/cs224w/project/26425124.pdf).

**WARNING!** You do have to read source code to fully understand this repository. It is not designed to be an installable package. Perhaps you can view this as a poor man's version of experiment logging and source control.

## Bash Tools

See [bash helper](bash_scripts/bash_helper.sh). 

Requires GCP's [command line tools](https://cloud.google.com/sdk/gcloud/).

Usages are the following

```
gc_info

gc_ssh pg-example
gc_ssh pg-example "ls /etc"

gc_put pg-example local_source_folder $GCHOME
gc_get pg-example $GCHOME/source_code .
gc_jupyter pg-jupyter $GCHOME 8965
```

## Python Tools

A moderately modified [ImageNet training script](https://github.com/pytorch/examples/tree/master/imagenet) has been provided to serve as a minimium working example. See [main file](main_train_model.py). Most of the tools here are in [gcp bucket API](utils/gc_storage.py) and [logger](utils/tb_logger.py).

Example usage is as follows
```
python main_train_model.py --exp_name demonstration --train_data ~/gc_cache/[PATH NAME]/mnist_png/training --val_data ~/gc_cache/[PATH NAME]/mnist_png/testing
```

### GCP Bucket API

Allows you to use Python to interface with GCP. Provides a fast way to download a folder from GCP with local cache. This is NOT a replacement for [Google Cloud Stoage FUSE](https://cloud.google.com/storage/docs/gcs-fuse) but allows direct interface with GCP Bucket, which you might want.

### Tensorboard Logging Tools

Logger here supports the following functionalities

* Logging of text
* Logging of text
* Logging of images
   * Custom wrapper to allow logging of matplotlib images (TensorboardX only logs Tensor-represented images)
* Duplication of source code
    * Stores a zip file of all source codes. Uses `.gitignore` to filter unwanted files
* Storage of model checkpoints