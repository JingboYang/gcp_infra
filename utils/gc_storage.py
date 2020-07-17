'''
Originally created for Stanford Winter 2019 CS224W
Jingbo Yang, Ruge Zhao, Meixian Zhu

Local cache and mini-file system support for GCP storage bucket
'''

import collections
import copy
import os
from pathlib import Path
from multiprocessing import Pool, Process
import pprint as pp

import tqdm
import base64
import time

from google.cloud import storage
# from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2 import service_account

import crcmod.predefined

# Nicely rephrased version of the following
# https://stackoverflow.com/questions/37003862/google-cloud-storage-how-to-upload-a-file-from-python-3

# Useful page describing what to do
# https://hackersandslackers.com/manage-files-in-google-cloud-storage-with-python/


def path_to_str(function):
    def wrapper(*args, **kwargs):
        all_args = [args[0]]
        for a in args[1:]:
            all_args.append(str(a))
        result = function(*all_args, **kwargs)
        return result
    return wrapper

def vprint(*args, verbose=False):
    if verbose:
        print(*args)

# Flatten a dictionary, acquired from
# https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten(d, parent_key='', sep='/'):

    items = []
    for k, v in d.items():

        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            next_level_items = flatten(v, new_key, sep=sep).items()
            if len(next_level_items) == 0:
                items.append((new_key, None))
            else:
                items.extend(next_level_items)
        else:
            items.append((new_key, v))
    return dict(items)

# https://stackoverflow.com/questions/3431825/generating-an-crc32c-checksum-of-a-file
def compute_crc32c(fname):
    # hash_crc32c = hashlib.crc32c(digest_size=10)
    crc32 = crcmod.predefined.Crc('crc-32c')
    with open(fname, "rb") as f:
        buf = f.read()
        crc32.update(buf)
    return crc32.digest()

def download_helper(args):
    '''Download file from GCP bucket'''
    local_path, cloud_path, cloud_crc32c, credential_path, project_name, bucket_name, verbose = args
    vprint(f'Preparing download for => {cloud_path}', verbose=verbose)

    client = storage.Client(
                            credentials=GCStorage.get_credentials(credential_path),
                            project=project_name)
    bucket = client.get_bucket(bucket_name)    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    local_crc32c = None
    if os.path.exists(local_path):
        vprint(f'Computing hash => {cloud_path}', verbose=verbose)
        local_crc32c = compute_crc32c(local_path)
        local_crc32c = base64.b64encode(local_crc32c).decode('utf-8')

    if local_crc32c != cloud_crc32c:
        vprint(f'Hash mismatch! Local ({local_crc32c}) != Cloud ({cloud_crc32c})', verbose=verbose)
        vprint(f'Downloading => {local_path}', verbose=verbose)
        blob = bucket.blob(cloud_path)
        blob.download_to_filename(local_path)
        vprint(f'Download complete => {local_path}', verbose=verbose)
    else:
        vprint(f'Download skipped => {cloud_path}', verbose=verbose)

    if os.path.exists(local_path):
        vprint(f'Verifying hash => {cloud_path}', verbose=verbose)
        local_crc32c = compute_crc32c(local_path)
        local_crc32c = base64.b64encode(local_crc32c).decode('utf-8')

        assert local_crc32c == cloud_crc32c, f'Downloaded {local_path} was incorrect hash Local ({local_crc32c}) != Cloud ({cloud_crc32c})'
    else:
        raise ValueError(f'File {local_path} was not downloaded?!')


def upload_helper(args):
    local_path, cloud_path, cloud_crc32c, credential_path, project_name, bucket_name, verbose, verify = args
    client = storage.Client(
                            credentials=GCStorage.get_credentials(credential_path),
                            project=project_name)
    bucket = client.get_bucket(bucket_name)    

    local_crc32c = None
    if os.path.exists(local_path) and verify:
        vprint(f'Computing hash => {cloud_path}', verbose=verbose)
        local_crc32c = compute_crc32c(local_path)
        local_crc32c = base64.b64encode(local_crc32c).decode('utf-8')

    if local_crc32c != cloud_crc32c or cloud_crc32c is None:
        vprint(f'Hash mismatch! Local ({local_crc32c}) != Cloud ({cloud_crc32c})', verbose=verbose)
        vprint(f'Uploading => {local_path}', verbose=verbose)
        blob = bucket.blob(cloud_path)
        blob.upload_from_filename(local_path)
        vprint(f'Uploading complete => {local_path}', verbose=verbose)
    else:
        vprint(f'Uploading skipped => {cloud_path}', verbose=verbose)

    if verify:
        try:
            cloud_file = bucket.list_blobs(prefix=cloud_path).__iter__().__next__()
            new_cloud_crc32c = cloud_file.__dict__['_properties']['crc32c']
            assert local_crc32c == new_cloud_crc32c, f'Uploaded {local_path} was incorrect hash Local ({local_crc32c}) != Cloud ({new_cloud_crc32c})'
        except:
            raise ValueError(f'File {local_path} was not uploaded?!')


class GCFile:
    def __init__(self, gc_file):
        self.__dict__.update(gc_file.__dict__)

        self.is_folder = '//' in self.__dict__['_properties']['id']
        self.cloud_levels = [x for x in self.__dict__['_properties']['id'].split('/')[:-1] if len(x.strip()) > 0]
        self.bucket_levels = [x for x in self.__dict__['name'].split('/') if len(x.strip()) > 0]
    
    def get_crc32chash(self):
        # return self.__dict__['_properties']['crc32cHash']
        return self.__dict__['_properties']['crc32c']
    
    def get_cloud_path(self):
        return '/'.join(self.bucket_levels)

    def __repr__(self):
        return f'path=> {self.get_cloud_path()}, crc32c=> {self.get_crc32chash()}'


class GCStorage:
    '''Utility class for interaction with Google Cloud storage

        project name = google cloud project id
        credential = JSON file for service account
        bucket name = well, bucket name
    '''
    
    MANAGER = {}

    @staticmethod
    def get_credentials(credential_path):
        return service_account.Credentials.\
                    from_service_account_file(credential_path)

    @staticmethod
    def get_CloudFS(*args, **kwargs):
        bucket_name = args[1] if len(args) >= 2 else kwargs['bucket_name']
        if bucket_name not in GCStorage.MANAGER:
            storage = GCStorage()
            storage._initialize(*args, **kwargs)
            GCStorage.MANAGER[bucket_name] = storage

        return GCStorage.MANAGER[bucket_name]

    def _initialize(self, project_name, bucket_name, credential_path, local_cache, verbose=False):
        self.client = storage.Client(
            credentials=GCStorage.get_credentials(credential_path),
            project=project_name
            )
        self.verbose = verbose
        self.credential_path = credential_path
        self.project_name = project_name
        self.bucket = self.client.get_bucket(bucket_name)
        self.bucket_name = bucket_name
        self.local_cache = Path(local_cache) / self.project_name / self.bucket_name
        self.local_cache.mkdir(parents=True, exist_ok=True)
        
        vprint(f'Created GCStorage@{self.project_name}/{self.bucket_name}\nLocal cache=> {self.local_cache}',
                verbose=self.verbose)

    def get_bucket_root(self):
        return f'gs://{self.bucket_name}/'

    def get(self, cloud_path, num_workers=1, progress=False):
        local_path = self.local_cache / cloud_path

        full_path, local_folder = self.list_files(cloud_path)
        vprint('Download structure => ', verbose=self.verbose)
        vprint(pp.pformat(full_path, indent=4), verbose=self.verbose)

        if local_folder is None:      # Then this is simply a file
            # self.download(local_path, cloud_path)
            pass
        else:
            flat = flatten(full_path)
            download_arguments = []
            for f in flat:
                local_path = self.local_cache / flat[f].get_cloud_path()
                cloud_path = flat[f].get_cloud_path()
                crc32c = flat[f].get_crc32chash()

                download_arguments.append((local_path, cloud_path, crc32c,
                                           self.credential_path, self.project_name, self.bucket_name,
                                           self.verbose))
            
            # pp.pprint(download_arguments)
            # https://stackoverflow.com/questions/41920124/multiprocessing-use-tqdm-to-display-a-progress-bar
            with Pool(processes=num_workers) as p:
                max_ = len(download_arguments)
                if progress:
                    with tqdm.tqdm(total=max_) as pbar:
                        for i, _ in enumerate(p.imap_unordered(download_helper, download_arguments)):
                            pbar.update()
                else:
                    p.map(download_helper, download_arguments)

            return download_arguments

    def put(self, cloud_path, num_workers=1, progress=False, verify=True):
        
        full_path, local_folder = self.list_files(cloud_path)

        vprint('Upload structure => ', verbose=self.verbose)
        vprint(pp.pformat(full_path, indent=4), verbose=self.verbose)

        if full_path is not None:
            flat = flatten(full_path)
        else:
            flat = None

        upload_arguments = []
        for root, dirs, files in os.walk(f'{self.local_cache}/{cloud_path}'):
            for f in files:
                local_path = f'{root}/{f}'
                cloud_full = cloud_path + '/' + local_path.split(cloud_path)[-1]
                cloud_full = cloud_full.replace('//', '/')
                crc32c = flat[cloud_full].get_crc32chash() if flat is not None and cloud_full in flat else None
                
                upload_arguments.append((local_path, cloud_full, crc32c,
                                         self.credential_path, self.project_name, self.bucket_name,
                                         self.verbose, verify))

        with Pool(processes=num_workers) as p:
            max_ = len(upload_arguments)
            if progress:
                with tqdm.tqdm(total=max_) as pbar:
                    for i, _ in enumerate(p.imap_unordered(upload_helper, upload_arguments)):
                        pbar.update()
            else:
                p.map(upload_helper, upload_arguments)

        return upload_arguments

    @path_to_str
    def download(self, local_path, cloud_path):
        '''Download file from GCP bucket'''
        blob = self.bucket.blob(cloud_path)
        blob.download_to_filename(local_path)
        vprint(f'{cloud_path} downloaded from bucket.', verbose=self.verbose)
        return local_path
    
    @path_to_str
    def upload(self, local_path, cloud_path):
        '''Upload file to GCP bucket'''
        blob = self.bucket.blob(cloud_path)
        blob.upload_from_filename(local_path)
        vprint( f'Uploaded {local_path} to "{cloud_path}" bucket.', verbose=self.verbose)
        return cloud_path

    @path_to_str
    def list_files(self, storage_path, delimiter='/'):
        '''List all files in GCP bucket'''
        files = self.bucket.list_blobs(prefix=storage_path)
        # file_list = [f.name for f in files]
        directory = dict()

        entered = False
        for f in files:
            entered = True
            gc_file = GCFile(f)

            cur_dir = directory
            for i, l in enumerate(gc_file.bucket_levels):
                if l not in cur_dir:
                    if i == len(gc_file.bucket_levels) - 1:
                        if gc_file.is_folder:
                            cur_dir[l] = dict()
                        else:
                            cur_dir[l] = gc_file
                    else:
                        cur_dir[l] = dict()
                cur_dir = cur_dir[l]

        if entered:
            cur_dir = directory
            for l in [x for x in storage_path.split(delimiter) if len(x.strip()) > 0]:
                cur_dir = cur_dir[l]

            return directory, cur_dir
        else:
            print(f'Cloud path {storage_path} does not exist')
            return None, None

    @path_to_str
    def delete(self, cloud_path):
        '''Delete file from GCP bucket'''
        self.bucket.delete_blob(cloud_path)
        return f'{cloud_path} deleted from bucket.'

    @path_to_str
    def rename(self, cloud_path_orig, cloud_path_new):
        '''Rename file in GCP bucket'''
        blob = self.bucket.blob(cloud_path_orig)
        self.bucket.rename_blob(blob, new_name=cloud_path_new)
        return f'{cloud_path_orig} renamed to {cloud_path_new}.'


class GCOpen():

    def __init__(self, cloud_path, file_mode, gc, use_cloud=True):
        
        self.cloud_path = cloud_path
        self.file_mode = file_mode
        self.use_cloud = use_cloud
        self.gc = gc

        self.local_path = f'{self.gc.local_cache}/{cloud_path}'

    def open(self):
        if 'w' in self.file_mode:
            filename = str(self.cloud_path).split('/')[-1]
            cloud_folder = '/'.join(str(self.cloud_path).split('/')[:-1])
            temp_folder = f'{self.gc.local_cache}/{cloud_folder}'
            os.makedirs(temp_folder, exist_ok=True)
        else:
            if self.use_cloud:
                self.gc.download(self.local_path, self.cloud_path)
            else:
                pass

        self.file = open(self.local_path, mode=self.file_mode)
        return self.file

    def __enter__(self):
        return self.open() 

    def __exit__(self, type, value, traceback):
        self.close()

    def send_to_cloud(self):
        if self.use_cloud:
            if 'w' in self.file_mode:
                self.gc.upload(self.local_path, self.cloud_path)
            elif 'r' in self.file_mode:
                pass

    def flush(self):
        self.file.flush()
        self.send_to_cloud()

    def close(self):
        self.file.close()
        self.send_to_cloud()