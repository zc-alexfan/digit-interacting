import subprocess
import os
import pickle
import argparse
from glob import glob
import pygit2
from email.message import EmailMessage
import smtplib
import pyarrow as pa
from tqdm import tqdm
import numpy as np
import lmdb
import os.path as op
import cv2 as cv


def get_branch():
    return pygit2.Repository('.').head.shorthand


def get_commit_hash():
    return pygit2.Repository('.').head.target

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def mkdir_p(exp_path):
    os.makedirs(exp_path, exist_ok=True)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(path, obj):
    with (open(path, 'wb')) as f:
        pickle.dump(obj, f)

def count_files(path):
    """
    Non-recursively count number of files in a folder.
    """
    files = glob(path)
    return len(files)

def get_host_name():
    results = subprocess.run(['cat', '/etc/hostname'], stdout=subprocess.PIPE)
    return results.stdout.decode("utf-8").rstrip()

class Email():
    def __init__(self, address, password, default_recipient):
        self.email_address = address
        self.email_password = password
        self.default_recipient = default_recipient

    def create_email_message(self, from_address, to_address, subject, body):
        msg = EmailMessage()
        msg['From'] = from_address
        msg['To'] = to_address
        msg['Subject'] = subject
        msg.set_content(body)
        return msg

    def notify(self, subject, body):
        if 'bdb.BdbQuit' in body:
            return
        self.send_msg(subject, body, self.default_recipient)

    def send_msg(self, subject, body, to_address):
        msg = self.create_email_message(
            from_address=self.email_address,
            to_address=to_address,
            subject=subject,
            body=body,
        )

        with smtplib.SMTP('smtp.gmail.com', port=587) as smtp_server:
            smtp_server.ehlo()
            smtp_server.starttls()
            smtp_server.login(self.email_address, self.email_password)
            smtp_server.send_message(msg)
        print("Email sent.")


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def fetch_lmdb_reader(db_path):
    env = lmdb.open(db_path, subdir=op.isdir(db_path),
                    readonly=True, lock=False,
                    readahead=False, meminit=False)
    txn = env.begin(write=False)
    return txn


def read_lmdb_image(txn, fname):
    image_bin = txn.get(fname.encode('ascii'))
    if image_bin is None:
        return image_bin
    image = np.fromstring(image_bin, dtype=np.uint8)
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    return image


def package_lmdb(lmdb_name, map_size, fnames, keys, write_frequency=5000):
    """
    Package image files into a lmdb database.
    fnames are the paths to each file and also the key to fetch the images.
    lmdb_name is the name of the lmdb database file
    map_size: recommended to set to len(fnames)*num_types_per_image*10
    keys: the key of each image in dict
    """
    assert len(fnames) == len(keys)
    db = lmdb.open(lmdb_name, map_size=map_size)
    txn = db.begin(write=True)
    for idx, (fname, key) in tqdm(enumerate(zip(fnames, keys)), total=len(fnames)):
        img = cv.imread(fname)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        status, encoded_image = cv.imencode(
            ".png", img, [cv.IMWRITE_JPEG_QUALITY, 100]
        )
        assert status
        txn.put(key.encode('ascii'), encoded_image.tostring())

        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(fnames))
        txn.put(b'__len__', dumps_pyarrow(len(fnames)))
    db.sync()
    db.close()
