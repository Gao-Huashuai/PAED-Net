import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

from ..fast_rcnn.config import cfg
from ..roi_pooling_layer import roi_pooling_op as roi_pool_op
from ..rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from ..rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from ..rpn_msr.proposal_target_layer_tf import proposal_target_layer as proposal_target_layer_py
# FCN pooling
from ..psroi_pooling_layer import psroi_pooling_op as psroi_pooling_op
from ..deform_psroi_pooling_layer import deform_psroi_pooling_op as deform_psroi_pooling_op
from ..deform_conv_layer import deform_conv_op as deform_conv_op


DEFAULT_PADDING = 'SAME'

def include_original(dec):
    """ Meta decorator, which make the original function callable (via f._original() )"""
    def meta_decorator(f):
        decorated = dec(f)
        decorated._original = f
        return decorated
    return meta_decorator

@include_original
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path, allow_pickle=True).item()
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print "assign pretrain model "+subkey+ " to "+key
                    except ValueError:
                        print "ignore "+key
                        if not ignore_missing:

                            raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    """
    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [c_o], init_biases, trainable)

            if group==1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)
            """

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True,relu=True, padding=DEFAULT_PADDING, trainable=True):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(conv, biases)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.relu(conv)
                return conv

    @layer
    def conv2(self, input, k_h, k_w, c_o, s_h, s_w, name, rate=1, biased=True, relu=True, padding=DEFAULT_PADDING,
             trainable=True, initializer=None):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.convolution(
            i, k, padding=padding, strides=[s_h, s_w], dilation_rate=[rate, rate])
        with tf.variable_scope(name) as scope:

            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            init_weights = tf.zeros_initializer() if initializer is 'zeros' else tf.contrib.layers.variance_scaling_initializer(
                factor=0.01, mode='FAN_AVG', uniform=False)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable,
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(conv, biases)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.relu(conv)
                return conv

    @layer
    def deform_conv(self, input, k_h, k_w, c_o, s_h, s_w, num_deform_group, name, num_groups=1, rate=1, biased=True,
                    relu=True,
                    padding=DEFAULT_PADDING, trainable=True, initializer=None):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        data = input[0]
        offset = input[1]
        c_i = data.get_shape()[-1]
        trans2NCHW = lambda x: tf.transpose(x, [0, 3, 1, 2])
        trans2NHWC = lambda x: tf.transpose(x, [0, 2, 3, 1])
        # deform conv only supports NCHW
        data = trans2NCHW(data)
        offset = trans2NCHW(offset)
        dconvolve = lambda i, k, o: deform_conv_op.deform_conv_op(
            i, k, o, strides=[1, 1, s_h, s_w], rates=[1, 1, rate, rate], padding=padding, num_groups=num_groups,
            deformable_group=num_deform_group)
        with tf.variable_scope(name) as scope:

            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            init_weights = tf.zeros_initializer() if initializer is 'zeros' else tf.contrib.layers.variance_scaling_initializer(
                factor=0.01, mode='FAN_AVG', uniform=False)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [c_o, c_i, k_h, k_w], init_weights, trainable,
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            print(data, kernel, offset)
            dconv = trans2NHWC(dconvolve(data, kernel, offset))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                if relu:
                    bias = tf.nn.bias_add(dconv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(dconv, biases)
            else:
                if relu:
                    return tf.nn.relu(dconv)
                return dconv
    
    @layer
    def atrous_conv(self, input, k_h, k_w, c_o, name, rate = 2, biased=True, relu=True, padding=DEFAULT_PADDING, trainable=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        aconv = lambda i, k: tf.nn.atrous_conv2d(i, k, rate, padding=padding)
        with tf.variable_scope(name) as scope:
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable,
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = aconv(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(conv, biases)
            else:
                conv = aconv(input, kernel)
                if relu:
                    return tf.nn.relu(conv)
                return conv

    @layer
    def upconv(self, input, shape, c_o, ksize=4, stride = 2, name = 'upconv', biased=False, relu=True, padding=DEFAULT_PADDING,
             trainable=True):
        """ up-conv"""
        self.validate_padding(padding)

        c_in = input.get_shape()[3].value
        in_shape = tf.shape(input)
        if shape is None:
            # h = ((in_shape[1] - 1) * stride) + 1
            # w = ((in_shape[2] - 1) * stride) + 1
            h = ((in_shape[1] ) * stride)
            w = ((in_shape[2] ) * stride)
            new_shape = [in_shape[0], h, w, c_o]
        else:
            new_shape = [in_shape[0], shape[1], shape[2], c_o]
        output_shape = tf.stack(new_shape)

        filter_shape = [ksize, ksize, c_o, c_in]

        with tf.variable_scope(name) as scope:
            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            filters = self.make_var('weights', filter_shape, init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            deconv = tf.nn.conv2d_transpose(input, filters, output_shape,
                                            strides=[1, stride, stride, 1], padding=DEFAULT_PADDING, name=scope.name)
            # coz de-conv losses shape info, use reshape to re-gain shape
            deconv = tf.reshape(deconv, new_shape)

            if biased:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                if relu:
                    bias = tf.nn.bias_add(deconv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(deconv, biases)
            else:
                if relu:
                    return tf.nn.relu(deconv)
                return deconv

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def sigmoid(self, input, name):
        return tf.nn.sigmoid(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name):
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]

        print input
        return roi_pool_op.roi_pool(input[0], input[1],
                                    pooled_height,
                                    pooled_width,
                                    spatial_scale,
                                    name=name)[0]

    @layer
    def psroi_pool(self, input, output_dim, group_size, spatial_scale, name):
        """contribution by miraclebiu"""
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]

        return psroi_pooling_op.psroi_pool(input[0], input[1],
                                           output_dim=output_dim,
                                           group_size=group_size,
                                           spatial_scale=spatial_scale,
                                           name=name)[0]

    @layer
    def deform_psroi_pool(self, input, output_dim, group_size, spatial_scale, pooled_size, part_size, sample_per_part,
                          trans_std, no_trans, name):
        """contribution by Zardinality"""
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]
        if no_trans:
            assert len(input) < 3
            num_rois = tf.shape(input[1])[0]
            input.append(tf.zeros([num_rois, pooled_size, pooled_size, 2]))
        if isinstance(input[2], tuple):
            input[2] = input[2][0]
        trans2NCHW = lambda x: tf.transpose(x, [0, 3, 1, 2])
        trans2NHWC = lambda x: tf.transpose(x, [0, 2, 3, 1])
        input[0] = trans2NCHW(input[0])
        # input[1] = trans2NCHW(input[1])
        input[2] = trans2NCHW(input[2])
        return deform_psroi_pooling_op.deform_psroi_pool(input[0], input[1], input[2],
                                                         output_dim=output_dim,
                                                         group_size=group_size,
                                                         spatial_scale=spatial_scale,
                                                         pooled_size=pooled_size,
                                                         part_size=part_size,
                                                         sample_per_part=sample_per_part,
                                                         trans_std=trans_std,
                                                         no_trans=no_trans,
                                                         name=name)[0]

    @layer
    def proposal_layer(self, input, _feat_stride, anchor_scales, aspect_ratio, cfg_key, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
            # input[0] shape is (1, H, W, Ax2)
            # rpn_rois <- (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        return tf.reshape(tf.py_func(proposal_layer_py,\
                                     [input[0],input[1],input[2], cfg_key, _feat_stride, anchor_scales, aspect_ratio],\
                                     [tf.float32]),
                          [-1,5],name =name)

    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, aspect_ratio, name, anchor_thresh=8):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:
            # 'rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info'
            rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = \
                tf.py_func(anchor_target_layer_py,
                           [input[0],input[1],input[2],input[3],input[4], _feat_stride, anchor_scales, aspect_ratio, anchor_thresh],
                           [tf.float32,tf.float32,tf.float32,tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels') # shape is (1 x H x W x A, 2)
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets') # shape is (1 x H x W x A, 4)
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights') # shape is (1 x H x W x A, 4)
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights') # shape is (1 x H x W x A, 4)


            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


    @layer
    def proposal_target_layer(self, input, classes, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:
            #inputs: 'rpn_rois','gt_boxes', 'gt_ishard', 'dontcare_areas'
            rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights \
                = tf.py_func(proposal_target_layer_py,
                             [input[0],input[1],input[2],input[3],classes],
                             [tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])
            # rois <- (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]
            # rois = tf.convert_to_tensor(rois, name='rois')
            rois = tf.reshape(rois, [-1, 5], name='rois') # goes to roi_pooling
            labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels') # goes to FRCNN loss
            bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'bbox_targets') # goes to FRCNN loss
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name = 'bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name = 'bbox_outside_weights')

            self.layers['rois'] = rois

            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


    @layer
    def reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
            #
            # transpose: (1, AxH, W, 2) -> (1, 2, AxH, W)
            # reshape: (1, 2xA, H, W)
            # transpose: -> (1, H, W, 2xA)
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                            [   input_shape[0],
                                                int(d),
                                                tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),
                                                input_shape[2]
                                            ]),
                                 [0,2,3,1],name=name)
        else:
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                        [   input_shape[0],
                                            int(d),
                                            tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),
                                            input_shape[2]
                                        ]),
                                 [0,2,3,1],name=name)

    @layer
    def spatial_reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
        return tf.reshape(input,\
                               [input_shape[0],\
                                input_shape[1], \
                                -1,\
                                int(d)])

    # @layer
    # def feature_extrapolating(self, input, scales_base, num_scale_base, num_per_octave, name):
    #     return feature_extrapolating_op.feature_extrapolating(input,
    #                           scales_base,
    #                           num_scale_base,
    #                           num_per_octave,
    #                           name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def reduce_mean(self, inputs, name):
        return tf.reduce_mean(inputs, [1, 2], keep_dims=True, name=name)

    @layer
    def multiply(self, inputs, name):
        return tf.multiply(inputs[0], inputs[1], name=name)

    @layer
    def mul_feature(self, inputs, idx, number, name):
        features1 = inputs[0]
        features2 = inputs[1]
        features3 = inputs[2]
        conv_layer3 = inputs[3]
        conv_layer2 = inputs[4]
        conv_layer1 = inputs[5]

        feature3_split = tf.split(features3, number, axis=3)
        _ = tf.multiply(feature3_split[idx], conv_layer3)
        feature2_split = tf.split(features2, number, axis=3)
        _ = tf.multiply(feature2_split[idx], conv_layer2)
        feature1_split = tf.split(features1, number, axis=3)
        out = tf.multiply(feature1_split[idx], conv_layer1, name=name)
        return out


    @layer
    def conv_reshape(self, input, name):
        input_shape = input.get_shape()
        assert input_shape.ndims == 4, 'The input shape is not 4'
        dim = 1
        for d in input_shape[1:].as_list():
            dim *= d
        output = tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [-1, dim], name=name)
        return output

    @layer
    def lstm(self, input, name, keep_droup=0.7, hidden_num=128, data_format="NHWC", num_layers=2, is_training=True):
        with tf.variable_scope(name):
            if isinstance(input, tuple):
                input = input[0]
            if data_format == "NCHW":
                input = tf.transpose(input, [0, 2, 3, 1])
            input_shape = input.get_shape().as_list()
            input_re = tf.reshape(input, [-1, input_shape[1]*input_shape[2], input_shape[3]])
            print("the input of lstm is {}".format(input_re.get_shape().as_list()))
            # seq_len = tf.fill(input_shape[0], input_re.get_shape().as_list()[1])
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_num, state_is_tuple=True)
            if is_training:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=keep_droup)
            # lstm_cell1 = tf.nn.rnn_cell.LSTMCell(hidden_num, state_is_tuple=True)
            # if is_training:
            #     lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell1, output_keep_prob=keep_droup)
            # cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell, lstm_cell1], state_is_tuple=True)
            # initial_state = cell.zero_state(input_shape[0], dtype=tf.float32)
            # The second output is the last state and we will not use that
            _, out = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=input_re,
                # sequence_length=seq_len,
                # initial_state=initial_state,
                dtype=tf.float32,
                time_major=False
            )  # [batch_size, max_stepsize, hidden_num]
            # Reshaping to apply the same weights over the timesteps
            c_out = tf.reshape(out[0], [-1, hidden_num])  # [batch_size * max_stepsize, hidden_num]
            h_out = tf.reshape(out[1], [-1, hidden_num])
            outputs = tf.concat([c_out, h_out], -1)

            W = tf.get_variable(name='W_out',
                                shape=[hidden_num*2, input_shape[1]*input_shape[2]],
                                dtype=tf.float32,
                                initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer
            b = tf.get_variable(name='b_out',
                                shape=[input_shape[1]*input_shape[2]],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())
            logits = tf.sigmoid(tf.matmul(outputs, W) + b)
            # Reshaping back to the original shape
            # logits = tf.reshape(logits, [-1, input_shape[1]*input_shape[2], input_shape[3]])
            # Time major
            # logits = tf.transpose(logits, (1, 0, 2))

            # expand logits to [bach_size, m*n, feature_num]
            logits = tf.expand_dims(logits, -1)
            weights = logits
            for i in range(1, input_shape[3]):
                weights = tf.concat([weights, logits], -1)
            rois_feature = tf.multiply(input_re, weights)
            rois_feature = tf.reshape(rois_feature, [-1, input_shape[1], input_shape[2], input_shape[3]])
            return rois_feature

    @layer
    def fc(self, input, num_out, name, relu=True, data_format = "NCHW", trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                if data_format == "NCHW":
                    input = tf.transpose(input, [0, 2, 3, 1])
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable, \
                                    regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input,name=name)

    @layer
    def spatial_softmax(self, input, name):
        input_shape = tf.shape(input)
        # d = input.get_shape()[-1]
        return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                          [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)

    @layer
    def reshape(self, input, shape, name):
        return tf.reshape(input, shape=shape, name=name)

    @layer
    def add(self,input,name):
        """contribution by miraclebiu"""
        return tf.add(input[0],input[1], name=name)

    @layer
    def batch_normalization(self,input,name,relu=True, is_training=False):
        """contribution by miraclebiu"""
        if relu:
            temp_layer=tf.layers.batch_normalization(input,scale=True,center=True,training=is_training, name=name)
            return tf.nn.relu(temp_layer)
        else:
            return tf.layers.batch_normalization(input,scale=True,center=True,training=is_training,name=name)
    @layer
#    def l2_normalization(self, input, name):
#        return tf.nn.l2_normalize(input, -1, name=name)
    def l2_normalization(self, input, name, trainable=True, scale=20):
        n_channels = input.get_shape().as_list()[-1]
        l2_norm = tf.nn.l2_normalize(input, [3], epsilon=1e-12)
        with tf.variable_scope(name):
            gamma = tf.get_variable("gamma", shape=[n_channels, ], dtype=tf.float32,
                                initializer=tf.constant_initializer(scale),
                                trainable=trainable)
        return l2_norm * gamma

    @layer
    def GroupNormalization(inputs, 
                       group, 
                       N_axis=0, 
                       C_axis=-1, 
                       momentum=0.9,
                       epsilon=1e-3,
                       training=False,
                       name=None):
        """ Group normalization implementation with tensorflow.

                As descriped in Wu's paper(http://arxiv.org/abs/1803.08494), we can implement a
                group norm with existed batch norm routine.

                The tensorflow code in this paper:
                ```python
                def GroupNorm(x, gamma, beta, G, eps=1e-5):
                    # x: input features with shape [N,C,H,W]
                    # gamma, beta: scale and offset, with shape [1,C,1,1]
                    # G: number of groups for GN
                    N, C, H, W = x.shape
                    x = tf.reshape(x, [N, G, C // G, H, W])
                    mean, var = tf.nn.moments(x, [2, 3, 4], keep dims=True)
                    x = (x - mean) / tf.sqrt(var + eps)
                    x = tf.reshape(x, [N, C, H, W])
                    return x * gamma + beta
                ```

                It is easy to know that when `x` is reshaped as [N, G, C//G, H, W], we can implement
                it with batch norm in tensorflow:
                ```python
                tf.layers.batch_normalization(x, axis=[0, 1])
                ```

                Params
                ------
                `inputs`: tensor input
                `group`: number of groups in group norm
                `N_axis`: axis number of batch axis
                `C_axis`: axis number of channel axis
                `momentum`: momentum used in moving average mean and moving average variance
                `epsilon`: a small value to prevent divided by zero
                `training`: either a Python boolean, or a Tensorflow boolean scalar tensor (e.g. a
                placeholder). Whether to return the output in training mode or in inference mode.
                **Note:** make sure to set this parameter correctly, or else your training/inference
                will not work properly.
                `name`: string, the name of the layer

                Returns
                -------
                Output tensor.
                """
        with tf.variable_scope(name, "GroupNorm"):
            input_shape = inputs.get_shape().as_list()
            ndims = len(input_shape)
            if not ndims in [4, 5]:
                raise ValueError("Not supported input dimension. Only 3 or 4")

            if not isinstance(C_axis, int):
                raise ValueError('`C_axis` must be an integer. Now it is {}'.format(C_axis))

            # Check axis
            if C_axis < 0:
                C_axis = ndims + C_axis
            if C_axis < 0 or C_axis >= ndims:
                raise ValueError('Invalid axis: %d' % C_axis)
            if N_axis < 0:
                N_axis = ndims + N_axis
            if N_axis < 0 or N_axis >= ndims:
                raise ValueError('Invalid axis: %d' % N_axis)

            # Require C % G == 0
            if input_shape[C_axis] % group != 0 or input_shape[C_axis] < group:
                raise ValueError('`group` should less than C_shape and be dividable '
                                 'by C_shape. `group` is %d and C_shape is %d'
                                 % (group, input_shape[C_axis]))

            permutation = [N_axis, C_axis] + [i for i in range(ndims) if i != C_axis and i != N_axis]
            inputs = tf.transpose(inputs, perm=permutation)

            old_shape = tf.shape(inputs)
            old_shape_val = inputs.get_shape().as_list()
            if ndims == 4:
                new_shape = [old_shape_val[0], group, old_shape_val[1] // group, old_shape[2], old_shape[3]]
            elif ndims == 5:
                new_shape = [old_shape_val[0], group, old_shape_val[1] // group, old_shape[2], old_shape[3],
                             old_shape[4]]

            inputs = tf.reshape(inputs, shape=new_shape)

            outputs = tf.layers.batch_normalization(inputs,
                                                    axis=[0, 1],
                                                    momentum=momentum,
                                                    epsilon=epsilon,
                                                    training=training)

            outputs = tf.reshape(outputs, shape=old_shape)

            reverse_permutation = permutation[:]
            for i, idx in enumerate(permutation):
                reverse_permutation[idx] = i

            outputs = tf.transpose(outputs, perm=reverse_permutation)

            return outputs


 


    @layer
    def negation(self, input, name):
        """ simply multiplies -1 to the tensor"""
        return tf.multiply(input, -1.0, name=name)

    @layer
    def bn_scale_combo(self, input, c_in, name, relu=True):
        """ PVA net BN -> Scale -> Relu"""
        with tf.variable_scope(name) as scope:
            bn = self.batch_normalization._original(self, input, name='bn', relu=False, is_training=False)
            # alpha = tf.get_variable('bn_scale/alpha', shape=[c_in, ], dtype=tf.float32,
            #                     initializer=tf.constant_initializer(1.0), trainable=True,
            #                     regularizer=self.l2_regularizer(0.00001))
            # beta = tf.get_variable('bn_scale/beta', shape=[c_in, ], dtype=tf.float32,
            #                    initializer=tf.constant_initializer(0.0), trainable=True,
            #                    regularizer=self.l2_regularizer(0.00001))
            # bn = tf.add(tf.mul(bn, alpha), beta)
            if relu:
                bn = tf.nn.relu(bn, name='relu')
            return bn

    @layer
    def res_block(self, input, is_training, name):
        with tf.variable_scope(name) as scope:
            conv = self.conv._original(self, input, 1, 1, 64, 1, 1, relu=False, name='0/conv')
            conv = self.batch_normalization._original(self, conv, name='0/norm0', relu=True, is_training=is_training)
            conv = self.conv._original(self, conv, 3, 3, 64, 1, 1, relu=False, name='1/conv')
            conv = self.batch_normalization._original(self, conv, name='1/norm1', relu=True, is_training=is_training)
            conv = self.conv._original(self, conv, 1, 1, 256, 1, 1, relu=False, name='2/conv')
            conv = self.batch_normalization._original(self, conv, name='2/norm2', relu=False, is_training=is_training)
            out = tf.nn.relu(tf.add(input, conv))
        return out

    @layer
    def se_block(self, input, name):
        with tf.variable_scope(name) as scope:
            out = tf.reduce_mean(input, [1, 2], keep_dims=False)
            out = self.fc._original(self, out, 256//6, name='se_fc1', relu=True)
            out = self.fc._original(self, out, 256, name='se_fc2', relu=False)
            out = self.sigmoid._original(self, out, name=name)
            out = tf.reshape(out, [-1, 1, 1, 256])
            out = tf.multiply(out, input)
        return out

    # deformable residual block
    @layer
    def deform_rfb_block(self, input, is_training, rate, name):
        with tf.variable_scope(name) as scope:
            conv_l = self.conv._original(self, input, 1, 1, 64, 1, 1, relu=False, name='l/conv')
            conv_l = self.batch_normalization._original(self, conv_l, name='l/norm', relu=True, is_training=is_training)
            conv_off_l = self.conv2._original(self, conv_l, 3, 3, 72, 1, 1, name='l/conv_off', biased=True, rate=rate,
                                              relu=False, padding='SAME', initializer='zeros')
            conv_deform_l = self.deform_conv._original(self, (conv_l, conv_off_l), 3, 3, 64, 1, 1, biased=False, rate=rate,
                                                     relu=False, num_deform_group=4, name='l_1/conv')
            conv_deform_l = self.batch_normalization._original(self, conv_deform_l, name='l_1/norm', relu=True, is_training=is_training)
            conv_deform_l = self.conv._original(self, conv_deform_l, 1, 1, 256, 1, 1, relu=False, name='l_2/conv')
            conv_deform_l = self.batch_normalization._original(self, conv_deform_l, name='l_2/norm', relu=True, is_training=is_training)


            conv_m = self.conv._original(self, input, 1, 1, 64, 1, 1, relu=False, name='m/conv')
            conv_m = self.batch_normalization._original(self, conv_m, name='m/norm0', relu=True,
                                                        is_training=is_training)
            conv_m = self.conv._original(self, conv_m, 3, 3, 64, 1, 1, relu=False, name='m_1/conv')
            conv_m = self.batch_normalization._original(self, conv_m, name='m_1/norm0', relu=True,
                                                        is_training=is_training)
            conv_m = self.conv._original(self, conv_m, 1, 1, 256, 1, 1, relu=False, name='m_2/conv')
            conv_m = self.batch_normalization._original(self, conv_m, name='m_2/norm0', relu=True,
                                                        is_training=is_training)

            conv_cat = tf.concat([conv_deform_l, conv_m], axis=-1)
            conv_cat = self.conv._original(self, conv_cat, 1, 1, 256, 1, 1, relu=False, name='conv_cat/conv')
            out = tf.nn.relu(tf.add(input, conv_cat))
            return out


    @layer
    def pva_negation_block(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True, padding=DEFAULT_PADDING, trainable=True,
                           scale = True, negation = True):
        """ for PVA net, Conv -> BN -> Neg -> Concat -> Scale -> Relu"""
        with tf.variable_scope(name) as scope:
            conv = self.conv._original(self, input, k_h, k_w, c_o, s_h, s_w, biased=biased, relu=False, name='conv', padding=padding, trainable=trainable)
            conv = self.batch_normalization._original(self, conv, name='bn', relu=False, is_training=False)
            c_in = c_o
            if negation:
                conv_neg = self.negation._original(self, conv, name='neg')
                conv = tf.concat(axis=3, values=[conv, conv_neg], name='concat')
                c_in += c_in
            if scale:
                # y = \alpha * x + \beta
                alpha = tf.get_variable('scale/alpha', shape=[c_in,], dtype=tf.float32,
                                        initializer=tf.constant_initializer(1.0), trainable=True, regularizer=self.l2_regularizer(0.00001))
                beta = tf.get_variable('scale/beta', shape=[c_in, ], dtype=tf.float32,
                                        initializer=tf.constant_initializer(0.0), trainable=True, regularizer=self.l2_regularizer(0.00001))
                # conv = conv * alpha + beta
                conv = tf.add(tf.multiply(conv, alpha), beta)
            return tf.nn.relu(conv, name='relu')

    @layer
    def pva_negation_block_v2(self, input, k_h, k_w, c_o, s_h, s_w, c_in, name, biased=True, padding=DEFAULT_PADDING, trainable=True,
                           scale = True, negation = True):
        """ for PVA net, BN -> [Neg -> Concat ->] Scale -> Relu -> Conv"""
        with tf.variable_scope(name) as scope:
            bn = self.batch_normalization._original(self, input, name='bn', relu=False, is_training=False)
            if negation:
                bn_neg = self.negation._original(self, bn, name='neg')
                bn = tf.concat(axis=3, values=[bn, bn_neg], name='concat')
                c_in += c_in
                # y = \alpha * x + \beta
                alpha = tf.get_variable('scale/alpha', shape=[c_in,], dtype=tf.float32,
                                        initializer=tf.constant_initializer(1.0), trainable=True, regularizer=self.l2_regularizer(0.00004))
                beta = tf.get_variable('scale/beta', shape=[c_in, ], dtype=tf.float32,
                                        initializer=tf.constant_initializer(0.0), trainable=True, regularizer=self.l2_regularizer(0.00004))
                bn = tf.add(tf.multiply(bn, alpha), beta)
            bn = tf.nn.relu(bn, name='relu')
            if name == 'conv3_1/1': self.layers['conv3_1/1/relu'] = bn

            conv = self.conv._original(self, bn, k_h, k_w, c_o, s_h, s_w, biased=biased, relu=False, name='conv', padding=padding,
                         trainable=trainable)
            return conv

    @layer
    def pva_inception_res_stack(self, input, c_in, name, block_start = False, type = 'a'):

        if type == 'a':
            (c_0, c_1, c_2, c_pool, c_out) = (64, 64, 24, 128, 256)
        elif type == 'b':
            (c_0, c_1, c_2, c_pool, c_out) = (64, 96, 32, 128, 384)
        else:
            raise ('Unexpected inception-res type')
        if block_start:
            stride = 2
        else:
            stride = 1
        with tf.variable_scope(name+'/incep') as scope:
            bn = self.batch_normalization._original(self, input, name='bn', relu=False, is_training=False)
            bn_scale = self.scale._original(self, bn, c_in, name='bn_scale')
            ## 1 x 1

            conv = self.conv._original(self, bn_scale, 1, 1, c_0, stride, stride, name='0/conv', biased = False, relu=False)
            conv_0 = self.bn_scale_combo._original(self, conv, c_in=c_0, name ='0', relu=True)

            ## 3 x 3
            bn_relu = tf.nn.relu(bn_scale, name='relu')
            if name == 'conv4_1': tmp_c = c_1; c_1 = 48
            conv = self.conv._original(self, bn_relu, 1, 1, c_1, stride, stride, name='1_reduce/conv', biased = False, relu=False)
            conv = self.bn_scale_combo._original(self, conv, c_in=c_1, name='1_reduce', relu=True)
            if name == 'conv4_1': c_1 = tmp_c
            conv = self.conv._original(self, conv, 3, 3, c_1 * 2, 1, 1, name='1_0/conv', biased = False, relu=False)
            conv_1 = self.bn_scale_combo._original(self, conv, c_in=c_1 * 2, name='1_0', relu=True)

            ## 5 x 5
            conv = self.conv._original(self, bn_scale, 1, 1, c_2, stride, stride, name='2_reduce/conv', biased = False, relu=False)
            conv = self.bn_scale_combo._original(self, conv, c_in=c_2, name='2_reduce', relu=True)
            conv = self.conv._original(self, conv, 3, 3, c_2 * 2, 1, 1, name='2_0/conv', biased = False, relu=False)
            conv = self.bn_scale_combo._original(self, conv, c_in=c_2 * 2, name='2_0', relu=True)
            conv = self.conv._original(self, conv, 3, 3, c_2 * 2, 1, 1, name='2_1/conv', biased = False, relu=False)
            conv_2 = self.bn_scale_combo._original(self, conv, c_in=c_2 * 2, name='2_1', relu=True)

            ## pool
            if block_start:
                pool = self.max_pool._original(self, bn_scale, 3, 3, 2, 2, padding=DEFAULT_PADDING, name='pool')
                pool = self.conv._original(self, pool, 1, 1, c_pool, 1, 1, name='poolproj/conv', biased = False, relu=False)
                pool = self.bn_scale_combo._original(self, pool, c_in=c_pool, name='poolproj', relu=True)

        with tf.variable_scope(name) as scope:
            if block_start:
                concat = tf.concat(axis=3, values=[conv_0, conv_1, conv_2, pool], name='concat')
                proj = self.conv._original(self, input, 1, 1, c_out, 2, 2, name='proj', biased=True,
                                           relu=False)
            else:
                concat = tf.concat(axis=3, values=[conv_0, conv_1, conv_2], name='concat')
                proj = input

            conv = self.conv._original(self, concat, 1, 1, c_out, 1, 1, name='out/conv', relu=False)
            if name == 'conv5_4':
                conv = self.bn_scale_combo._original(self, conv, c_in=c_out, name='out', relu=False)
            conv = self.add._original(self, [conv, proj], name='sum')
        return  conv

    @layer
    def pva_inception_res_block(self, input, name, name_prefix = 'conv4_', type = 'a'):
        """build inception block"""
        node = input
        if type == 'a':
            c_ins = (128, 256, 256, 256, 256, )
        else:
            c_ins = (256, 384, 384, 384, 384, )
        for i in range(1, 5):
            node = self.pva_inception_res_stack._original(self, node, c_in = c_ins[i-1],
                                                          name = name_prefix + str(i), block_start=(i==1), type=type)
        return node

    @layer
    def scale(self, input, c_in, name):
        with tf.variable_scope(name) as scope:

            alpha = tf.get_variable('alpha', shape=[c_in, ], dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0), trainable=False,
                                    regularizer=self.l2_regularizer(0.00001))
            beta = tf.get_variable('beta', shape=[c_in, ], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0), trainable=False,
                                   regularizer=self.l2_regularizer(0.00001))
            return tf.add(tf.multiply(input, alpha), beta)






    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight_decay')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer

    def smooth_l1_dist(self, deltas, sigma2=9.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name) as scope:
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                        (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)


    def build_loss(self, ohem=False):
        ############# RPN
        # classification loss
        rpn_cls_score = tf.reshape(self.get_output('rpn_cls_score_reshape'), [-1, 2])  # shape (HxWxA, 2)
        rpn_label = tf.reshape(self.get_output('rpn-data')[0], [-1])  # shape (HxWxA)
        # ignore_label(-1)
        fg_keep = tf.equal(rpn_label, 1)
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1))
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_keep), [-1, 2]) # shape (N, 2)
        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_keep), [-1])
        rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label)
        rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)

        # box loss
        rpn_bbox_pred = self.get_output('rpn_bbox_pred') # shape (1, H, W, Ax4)
        rpn_bbox_targets = self.get_output('rpn-data')[1]
        rpn_bbox_inside_weights = self.get_output('rpn-data')[2]
        rpn_bbox_outside_weights = self.get_output('rpn-data')[3]
        rpn_bbox_pred = tf.reshape(tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), rpn_keep), [-1, 4]) # shape (N, 4)
        rpn_bbox_targets = tf.reshape(tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep), [-1, 4])
        rpn_bbox_inside_weights = tf.reshape(tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_keep), [-1, 4])
        rpn_bbox_outside_weights = tf.reshape(tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_keep), [-1, 4])

        rpn_loss_box_n = tf.reduce_sum(self.smooth_l1_dist(
            rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), axis=[1])

        # rpn_loss_n = tf.reshape(rpn_cross_entropy_n + rpn_loss_box_n * 5, [-1])

        if ohem:
            # k = tf.minimum(tf.shape(rpn_cross_entropy_n)[0] / 2, 300)
            # # k = tf.shape(rpn_loss_n)[0] / 2
            # rpn_loss_n, top_k_indices = tf.nn.top_k(rpn_cross_entropy_n, k=k, sorted=False)
            # rpn_cross_entropy_n = tf.gather(rpn_cross_entropy_n, top_k_indices)
            # rpn_loss_box_n = tf.gather(rpn_loss_box_n, top_k_indices)

            # strategy: keeps all the positive samples
            fg_ = tf.equal(rpn_label, 1)
            bg_ = tf.equal(rpn_label, 0)
            pos_inds = tf.where(fg_)
            neg_inds = tf.where(bg_)
            rpn_cross_entropy_n_pos = tf.reshape(tf.gather(rpn_cross_entropy_n, pos_inds), [-1])
            rpn_cross_entropy_n_neg = tf.reshape(tf.gather(rpn_cross_entropy_n, neg_inds), [-1])
            top_k = tf.cast(tf.minimum(tf.shape(rpn_cross_entropy_n_neg)[0], 300), tf.int32)
            rpn_cross_entropy_n_neg, _ = tf.nn.top_k(rpn_cross_entropy_n_neg, k=top_k)
            rpn_cross_entropy = tf.reduce_sum(rpn_cross_entropy_n_neg) / (tf.reduce_sum(tf.cast(bg_, tf.float32)) + 1.0) \
                                + tf.reduce_sum(rpn_cross_entropy_n_pos) / (tf.reduce_sum(tf.cast(fg_, tf.float32)) + 1.0)

            rpn_loss_box_n = tf.reshape(tf.gather(rpn_loss_box_n, pos_inds), [-1])
            # rpn_cross_entropy_n = tf.concat(0, (rpn_cross_entropy_n_pos, rpn_cross_entropy_n_neg))

        # rpn_loss_box = 1 * tf.reduce_mean(rpn_loss_box_n)
        rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1.0)

        ############# R-CNN
        # classification loss
        cls_score = self.get_output('cls_score') # (R, C+1)
        label = tf.reshape(self.get_output('roi-data')[1], [-1]) # (R)
        cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label)

        # bounding box regression L1 loss
        bbox_pred = self.get_output('bbox_pred') # (R, (C+1)x4)
        bbox_targets = self.get_output('roi-data')[2] # (R, (C+1)x4)
        # each element is {0, 1}, represents background (0), objects (1)
        bbox_inside_weights = self.get_output('roi-data')[3] # (R, (C+1)x4)
        bbox_outside_weights = self.get_output('roi-data')[4] # (R, (C+1)x4)

        loss_box_n = tf.reduce_sum( \
            bbox_outside_weights * self.smooth_l1_dist(bbox_inside_weights * (bbox_pred - bbox_targets)), \
            axis=[1])

        loss_n = loss_box_n + cross_entropy_n
        loss_n = tf.reshape(loss_n, [-1])

        # if ohem:
        #     # top_k = 100
        #     top_k = tf.minimum(tf.shape(loss_n)[0] / 2, 500)
        #     loss_n, top_k_indices = tf.nn.top_k(loss_n, k=top_k, sorted=False)
        #     loss_box_n = tf.gather(loss_box_n, top_k_indices)
        #     cross_entropy_n = tf.gather(cross_entropy_n, top_k_indices)

        loss_box = tf.reduce_mean(loss_box_n)
        cross_entropy = tf.reduce_mean(cross_entropy_n)

        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

        # add regularizer
        if cfg.TRAIN.WEIGHT_DECAY > 0:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n(regularization_losses) + loss

        return loss, cross_entropy, loss_box, rpn_cross_entropy, rpn_loss_box


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))


    return tf.reduce_sum(per_entry_cross_ent)


def convert_to_one_hot(labels, num_classes):
    num_labels = labels.shape[0]
    labels_one_hot = np.zeros((num_labels, num_classes))
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot.flat[index_offset + labels] = 1
    return labels_one_hot