# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import numpy as np
import panoramic

def _selective_search_IJCV_top_k(split, year, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb

# panoramic
for name in ['pano_2018_train', 'pano_2018_val', 'pano_2018_test']:
    __sets[name] = (lambda split=name, year='2018': panoramic.Panoramic(split, year))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        print (list_imdbs())
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
