# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""PASCAL VOC dataset evaluation interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import os
import cv2

from core.config import cfg
from datasets.dataset_catalog import DATASETS
from datasets.dataset_catalog import DATA_DIR
from datasets.dataset_catalog import IM_SET
from datasets.my_eval import voc_eval
from utils.io import save_object

logger = logging.getLogger(__name__)


def evaluate_boxes(
    json_dataset,
    all_boxes,
    output_dir
):
    _write_voc_results_files(json_dataset, all_boxes)
    _do_python_eval(json_dataset, output_dir)
    return None


def _write_voc_results_files(json_dataset, all_boxes):
    info = voc_info(json_dataset)
    data_path = info['data_path']
    image_set = info['image_set']
    image_set_index_file = os.path.join(data_path, image_set + '.txt')
    assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
    with open(image_set_index_file) as f:
        image_index = [x.strip() for x in f.readlines()]
    # Sanity check that order of images in json dataset matches order in the
    # image set
    roidb = json_dataset.get_roidb()
    for i, entry in enumerate(roidb):
        index = os.path.splitext(os.path.split(entry['image'])[1])[0]
        assert index == image_index[i]
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        logger.info('Writing VOC results for: {}'.format(cls))
        filename = _get_voc_results_file_template(json_dataset).format(cls)
        assert len(all_boxes[cls_ind]) == len(image_index)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(image_index):
                dets = all_boxes[cls_ind][im_ind]
                if len(dets) == 0:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3]))


def _get_voc_results_file_template(json_dataset):
    info = voc_info(json_dataset)
    data_path = info['data_path']
    image_set = info['image_set']
    filename = 'det_' + image_set + '_{:s}.txt'
    return os.path.join(data_path, 'results', filename)


def _do_python_eval(json_dataset, output_dir='output'):
    info = voc_info(json_dataset)
    data_path = info['data_path']
    image_set = info['image_set']
    imagesetfile = os.path.join(data_path, image_set + '.txt')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(json_dataset, imagename)
        if i % 100 == 0:
            print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))

    aps = []
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for _, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        filename = _get_voc_results_file_template(json_dataset).format(cls)
        rec, prec, ap = voc_eval(filename, imagenames, cls, recs, ovthresh=0.5)
        aps += [ap]
        logger.info('AP for {} = {:.4f}'.format(cls, ap))
        res_file = os.path.join(output_dir, cls + '_pr.pkl')
        save_object({'rec': rec, 'prec': prec, 'ap': ap}, res_file)
    A = np.array(aps)
    m_ap = A[~np.isnan(A)].mean()
    logger.info('Mean AP@0.5 = {:.4f}'.format(m_ap))


def parse_rec(json_dataset, index):
    """ Parse a annotation file """
    info = voc_info(json_dataset)
    data_path = info['data_path']
    image_file = os.path.join(data_path, 'images', index + '.jpg')
    assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)

    height, width = cv2.imread(image_file).shape[:2]
    annopath = os.path.join(data_path, 'annotations', '{:s}.txt')
    filename = annopath.format(index)
    rotate = 0
    objects = []
    with open(filename) as f:
        line = f.readline()
        while line:
            parts = line.split()
            if parts[0] == 'rotate':
                rotate = int(parts[1])
                assert rotate == 0
            else:
                obj_struct = {'name': parts[0]}
                x1 = min(max(int(parts[1]), 0), width - 1)
                y1 = min(max(int(parts[2]), 0), height - 1)
                x2 = min(max(int(parts[3]), 0), width - 1)
                y2 = min(max(int(parts[4]), 0), height - 1)
                obj_struct['bbox'] = [x1, y1, x2, y2]
                obj_struct['truncated'] = int(parts[5])
                obj_struct['difficult'] = 0
                objects.append(obj_struct)
            line = f.readline()

    return objects


def voc_info(json_dataset):
    data_path = DATASETS[json_dataset.name][DATA_DIR]
    image_set = DATASETS[json_dataset.name][IM_SET]
    return dict(
        data_path=data_path,
        image_set=image_set)
