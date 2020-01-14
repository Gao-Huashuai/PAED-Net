import tensorflow as tf
from network import Network
from ..fast_rcnn.config import cfg


class VGGnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self.gt_ishard = tf.placeholder(tf.int32, shape=[None], name='gt_ishard')
        self.dontcare_areas = tf.placeholder(tf.float32, shape=[None, 4], name='dontcare_areas')
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes,\
                            'gt_ishard': self.gt_ishard, 'dontcare_areas': self.dontcare_areas})
        self.trainable = trainable
        self.setup()

    def setup(self):
        n_classes = cfg.NCLASSES
        anchor_scales = cfg.ANCHOR_SCALES
        aspect_ratio = cfg.ASPECT_RATIO
        _feat_stride = [4, ]
        # encoder-decoder network
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='SAME', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='SAME', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, padding='SAME', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, padding='SAME', name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3'))
        # conv3
        (self.feed('conv3_3')
             .conv(1, 1, 256, 1, 1, name='conv3_3_1', relu=False)
             .batch_normalization(name='bn3_3_1', relu=True, is_training=True))
        # conv4
        (self.feed('conv4_3')
             .conv(1, 1, 256, 1, 1, name='conv4_3_1', relu=False)
             .batch_normalization(name='bn4_3_1', relu=True, is_training=True))
        (self.feed('conv5_3')
             .conv(1, 1, 256, 1, 1, name='conv6_1', relu=False)
             .batch_normalization(name='bn6_1', relu=True, is_training=True)
             .deform_rfb_block(is_training=False, rate=2, name='block6')
             .upconv(tf.shape(self.layers['conv4_3']), 256, 2, 2, name='conv6_2', relu=False)
             .batch_normalization(name='bn6_2', relu=True, is_training=True))
        (self.feed('bn4_3_1', 'bn6_2')
             .concat(axis=3, name='input_7')
             .conv(1, 1, 256, 1, 1, name='conv7_1', relu=False)
             .batch_normalization(name='bn7_1', relu=True, is_training=True)
             .deform_rfb_block(is_training=False, rate=2, name='block7')
             .upconv(tf.shape(self.layers['conv3_3']), 256, 2, 2, name='conv7_2', relu=False)
             .batch_normalization(name='bn7_2', relu=True, is_training=True))
        (self.feed('bn7_2', 'bn3_3_1')
             .concat(axis=3, name='input_8')
             .conv(1, 1, 512, 1, 1, name='conv8_1', relu=False)
             .batch_normalization(name='final_feature', relu=True, is_training=True))

        #========= RPN ============
        (self.feed('final_feature')
             .conv(3,3,512,1,1, name='rpn_conv/3x3'))

        # Loss of rpn_cls & rpn_boxes
        # shape is (1, H, W, A x 4) and (1, H, W, A x 2)
        (self.feed('rpn_conv/3x3')
             .conv(1,1,len(anchor_scales) * len(aspect_ratio) * 4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred'))
        (self.feed('rpn_conv/3x3')
             .conv(1, 1, len(anchor_scales) * len(aspect_ratio) * 2, 1, 1, padding='VALID', relu=False, name='rpn_cls_score'))

        # generating training labels on the fly
        # output: rpn_labels(HxWxA, 2) rpn_bbox_targets(HxWxA, 4) rpn_bbox_inside_weights rpn_bbox_outside_weights
        (self.feed('rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info')
             .anchor_target_layer(_feat_stride, anchor_scales, aspect_ratio, name = 'rpn-data' ))

        # shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        (self.feed('rpn_cls_score')
             .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape')
             .spatial_softmax(name='rpn_cls_prob'))

        # shape is (1, H, WxA, 2) -> (1, H, W, Ax2)
        (self.feed('rpn_cls_prob')
             .spatial_reshape_layer(len(anchor_scales)*len(aspect_ratio)*2, name = 'rpn_cls_prob_reshape'))

        # ========= RoI Proposal ============
        # add the delta(output) to anchors then
        # choose some reasonabel boxes, considering scores, ratios, size and iou
        # rpn_rois <- (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]
        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
             .proposal_layer(_feat_stride, anchor_scales, aspect_ratio,  'TRAIN', name = 'rpn_rois'))

        # matching boxes and groundtruth,
        # and randomlyt sample some rois and labels for RCNN
        (self.feed('rpn_rois','gt_boxes', 'gt_ishard', 'dontcare_areas')
             .proposal_target_layer(n_classes,name = 'roi-data'))

        #========= RCNN ============
        # multi-scale part-attention module
        (self.feed('bn6_1', 'rois')
             .roi_pool(3, 5, 1.0/16, name='pool_5')
             .lstm(name='lstm1', is_training=True)
             .conv_reshape(name='pool5_reshape'))
        (self.feed('bn7_1', 'rois')
             .roi_pool(5, 7, 1.0/8, name='pool_6')
             .lstm(name='lstm2', is_training=True)
             .conv_reshape(name='pool6_reshape'))
        (self.feed('final_feature', 'rois')
             .roi_pool(7, 7, 1.0/4, name='pool_7')
             .lstm(name='lstm3', is_training=True)
             .conv_reshape(name='pool7_reshape'))

        (self.feed('pool5_reshape', 'pool6_reshape', 'pool7_reshape')
             .concat(axis=-1, name='pool_concat')
             .fc(4096, name='fc6')
             .dropout(0.5, name='drop6')
             .fc(4096, name='fc7')
             .dropout(0.5, name='drop7')
             .fc(n_classes, relu=False, name='cls_score')
             .softmax(name='cls_prob'))
    
        (self.feed('drop7')
             .fc(n_classes*4, relu=False, name='bbox_pred'))


