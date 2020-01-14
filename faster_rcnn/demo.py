import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob
import json
import collections

import PIL
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

this_dir = osp.dirname(__file__)
sys.path.insert(0, this_dir + '/..')
print(this_dir)

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer

CLASSES = ('__background__',
            'car', 'person', 'sign', 'line')

color = {'car': 'magenta', 'person': 'red', 'sign': 'green', 'line': 'cyan'}


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=False):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.

  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  fonts_heights = np.mean(display_str_heights)
  display_str_widths = [font.getsize(ds)[0] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
  total_display_str_width = (1 + 2 * 0.05) * sum(display_str_widths)

  # if top > total_display_str_height:
  #   text_bottom = top
  # else:
  #   text_bottom = bottom + total_display_str_height
  if left > total_display_str_width:
    text_right = left
  else:
    text_right = right + total_display_str_height

  for display_str in display_str_list:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, top - fonts_heights - 2 * margin), (left + text_width, top)],
        fill=color)
    draw.text(
        (left + margin, top - fonts_heights - 2 * margin),
        display_str,
        fill='black',
        font=font)
    left = left + text_width


def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        draw_bounding_box_on_image(im, bbox[1], bbox[0], bbox[3], bbox[2],
                                   color=color[class_name], display_str_list=class_name)
    return im

def demo(sess, net, image_name, thresh=0.05):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    image = PIL.Image.open(image_name)
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()

    im_num = os.path.split(image_name)[1].split('.')[0]
    scores, boxes = im_detect(sess, net, im, save_feature=True, feature_path='./data/conv.npy')
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.7
    NMS_THRESH = 0.3
    results = []
    name = image_name.split('/')[-1]
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        cls_lables = np.full_like(cls_scores, cls_ind)
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis], cls_lables[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -2] > thresh)[0]
        dets = dets[inds]
        for i in range(dets.shape[0]):
            name = str(name)
            category = int(dets[i, -1])
            bbox = list(map(float, dets[i, :4]))
            bbox = [round(b, 2) for b in bbox]
            score = float(dets[i, -2])
            dic = collections.OrderedDict()
            dic['name'] = str(name)
            dic['category'] = int(category)
            dic['bbox'] = bbox
            dic['score'] = float(score)
            results.append(dic)
        im = vis_detections(image, cls, dets, ax=None, thresh=CONF_THRESH)

    out_path = './data/detection_result'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_path = os.path.join(out_path, os.path.split(image_name)[-1])
    image.save(out_path)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default='./output/pano_2018_train/VGGnet_fast_rcnn_iter_30000.ckpt')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    print ('Loading network {:s}... '.format(args.demo_net)),
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    print (' done.')

    img_path = './data/*.jpg'
    im_names = glob.glob(img_path)
    submit_results = []
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {:s}'.format(im_name)
        demo(sess, net, im_name)
    print 'done'
    # plt.show()

