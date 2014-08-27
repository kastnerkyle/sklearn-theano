"""Base datasets functionality."""
# Authors: Kyle Kastner
#          Michael Eickenberg
# License: BSD 3 Clause

import os
import numpy as np
from PIL import Image
from sklearn.datasets.base import Bunch
try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib


def get_dataset_dir(dataset_name, data_dir=None, folder=None, create_dir=True):
    if not data_dir:
        data_dir = os.getenv("SKLEARN_THEANO_DATA", os.path.join(
            os.path.expanduser("~"), "sklearn_theano_data"))
    if folder is None:
        data_dir = os.path.join(data_dir, dataset_name)
    else:
        data_dir = os.path.join(data_dir, folder)
    if not os.path.exists(data_dir) and create_dir:
        os.makedirs(data_dir)
    return data_dir


def download(url, server_fname, local_fname=None, progress_update_percentage=5):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            print("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        print("Downloading: %s Bytes: %s" % (server_fname, file_size))
        file_size_dl = 0
        block_sz = int(1E7)
        p = 0
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if (file_size_dl * 100. / file_size) > p:
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl *
                                               100. / file_size)
                print(status)
                p += progress_update_percentage


def load_sample_images(read_dir=None, resize_shape=None):
    """Load sample images for image manipulation.
    Loads ``sloth``, ``sloth_closeup``, and ``cat and dog``.

    Parameters
    ----------
    read_dir : None or string
        If None, read in data from sklearn_theano's datasets/images path.
        Otherwise, read a set of .jpg images from the specified path.

    resize_shape : None or tuple, shape = [width, height]
        If None, read in each image in its native size. If a tuple is passed,
        resize all images to the shape given by resize_shape.

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'images', the sample images, 'filenames', the file
        names for the images, and 'DESCR'
        the full description of the dataset.
    """
    if read_dir is None:
        module_path = os.path.join(os.path.dirname(__file__), "images")
        with open(os.path.join(module_path, 'README.txt')) as f:
            descr = f.read()
    else:
        module_path = os.path.join(read_dir)
        descr = "Specially loaded dataset from %s" % module_path
    filenames = [os.path.join(module_path, filename)
                 for filename in os.listdir(module_path)
                 if filename.endswith(".jpg")]
    # Load image data for each image in the source folder.
    if resize_shape is None:
        images = [np.array(Image.open(filename, 'r')) for filename in filenames]
    else:
        images = [np.array(Image.open(filename, 'r').resize(resize_shape))
                  for filename in filenames]

    return Bunch(images=images,
                 filenames=filenames,
                 DESCR=descr)


def load_sample_image(image_name, read_dir=None, resize_shape=None):
    """Load the numpy array of a single sample image

    Parameters
    -----------
    image_name : {`sloth.jpg`, `sloth_closeup.jpg`, `cat_and_dog.jpg`}
        The name of the sample image loaded

    read_dir : None or string
        If None, read in data from sklearn_theano's datasets/images path.
        Otherwise, read a .jpg image from the specified path.

    resize_shape : None or tuple, shape = [width, height]
        If None, read in the image in its native size. If a tuple is passed,
        resize the selected image to the shape given by resize_shape.

    Returns
    -------
    img : 3D array
        The image as a numpy array: height x width x color

    """
    images = load_sample_images(read_dir=read_dir, resize_shape=resize_shape)
    index = None
    for i, filename in enumerate(images.filenames):
        if filename.endswith(image_name):
            index = i
            break
    if index is None:
        raise AttributeError("Cannot find sample image: %s" % image_name)
    return images.images[index]
