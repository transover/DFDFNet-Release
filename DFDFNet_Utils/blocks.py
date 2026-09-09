"""
@project : DFDFNet-Release
@file    : DFDFNet_Utils/blocks.py
@author  : Hang Yu
@contact : transover@buaa.edu.cn
@paper   : Dual-Stream Feature-Map Dynamic Fusion Network for Multiclass sEMG Gesture Recognition
@venue   : Pattern Recognition
@license : MIT License 
@update  : 2026-09-09 21:23:00
@description : 
    Reusable network building blocks.
----------------------------------------------------------------------------------------------------
    Implements the custom Keras layers used by DFDFNet, including
    normalization, squeeze-and-excitation, inverted residual blocks,
    temporal convolution blocks, decision heads, and learnable fusion layers.

"""

from __future__ import annotations
from collections.abc import Sequence
from typing import Iterable, Optional, Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

__all__ = [
    "NormLayer",
    "SwitchableNormalization",
    "SqueezeExcitation",
    "InvertedResidualBlock",
    "TemporalConvBlock",
    "DecisionScoreHead_sEMGNet",
    "DynamicRoutingLayer",
    "DecisionScoreHead_FeatureNet",
    "LearnableFusion",
    "LearnableWeightedSum",
    "FeedForwardRefBlock",
    "FeedForwardFeaBlock"
]
def _make_norm(method: Optional[str], epsilon: float = 1e-6) -> layers.Layer:
    # Construct a normalization layer for use after convolution
    return NormLayer(method=method or "", epsilon=epsilon)

def _apply_norm(norm_layer: layers.Layer, x: tf.Tensor, training: Optional[bool]) -> tf.Tensor:
    # Apply the normalization layer and forward the training flag
    return norm_layer(x, training=training)

class DRBLayer(layers.Layer):
    # Dynamic Routing Layer (DRB) with iterative routing between capsules
    def __init__(self,
                 num_DRB,
                 dim_DRB,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(DRBLayer, self).__init__(**kwargs)  # inherit **kwargs parameters
        self.num_DRB = num_DRB
        self.dim_DRB = dim_DRB
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = self.squash
        else:
            self.activation = activation.get(activation)  # resolve the activation function
        self.squash_factor = kwargs.get('squash_factor')

    def build(self, input_shape):
        # Define the routing kernel weights
        super(DRBLayer, self).build(input_shape)  # must call the parent build method
        input_dim_DRB = input_shape[-1]
        if self.share_weights:
            # Custom weights: [1, input_dim_DRB, num_DRB * dim_DRB]
            self.kernel = self.add_weight(
                name='DRB_kernel',
                shape=(1, input_dim_DRB,
                       self.num_DRB * self.dim_DRB),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_DRB = input_shape[-2]
            self.kernel = self.add_weight(
                name='DRB_kernel',
                shape=(input_num_DRB, input_dim_DRB,
                       self.num_DRB * self.dim_DRB),
                initializer='glorot_uniform',
                trainable=True)

    def softmax(self, x, axis=-1):
        # Softmax along the specified axis (K.softmax cannot select an axis)
        ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
        result = ex / K.sum(ex, axis=axis, keepdims=True)
        return result

    def squash(self, x, axis=-1, factor=0.25):
        # Squash nonlinearity: x' = ||x||^2 / (factor + ||x||^2) * x / ||x||
        if self.squash_factor:
            factor = self.squash_factor
        x_Euclidean_Norm_Square = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()  # ||x||^2
        scale = K.sqrt(x_Euclidean_Norm_Square) / (factor + x_Euclidean_Norm_Square)  # scaling factor
        result = scale * x
        return result

    def call(self, inputs):
        # Core routing logic for inputs [batch, input_num_DRB, input_dim_DRB]
        if self.share_weights:
            # inputs: [batch, input_num_DRB, input_dim_DRB]; kernel: [1, input_dim_DRB, num_DRB*dim_DRB]
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_DRB = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_DRB, self.num_DRB, self.dim_DRB))
        # Expand dims: hat_inputs [batch, input_num_DRB, num_DRB, dim_DRB]
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))
        # Transpose dims: hat_inputs [batch, num_DRB, input_num_DRB, dim_DRB]
        b = K.zeros_like(hat_inputs[:, :, :, 0])
        # b: [batch, num_DRB, input_num_DRB]

        output = None
        for i in range(self.routings):
            c = self.softmax(b, 1)  # shape = [None, num_DRB, input_num_DRB]
            s = tf.einsum('bin,binj->bij', c, hat_inputs)  # contraction [batch, num_DRB, dim_DRB]
            if i < self.routings - 1:
                s = K.l2_normalize(s, -1)  # [batch, num_DRB, dim_DRB]
                b = tf.einsum('bij,binj->bin', s, hat_inputs)  # [batch, num_DRB, input_num_DRB]
            output = s
        return output

    def compute_output_shape(self, input_shape):  # infer the output shape
        return (None, self.num_DRB, self.dim_DRB)

class SwitchableNormalization(keras.layers.Layer):
    # Switchable Normalization (SN): learnable fusion of BN / LN / IN statistics
    def __init__(self, axis: int = -1, momentum: float = 0.99, epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.axis = int(axis)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)

    def build(self, input_shape: tf.TensorShape) -> None:
        channels = int(input_shape[-1])
        self.gamma = self.add_weight(name="gamma", shape=(channels,), initializer="ones", trainable=True)
        self.beta = self.add_weight(name="beta", shape=(channels,), initializer="zeros", trainable=True)
        self.moving_mean = self.add_weight(
            name="moving_mean", shape=(1, 1, channels), initializer="zeros", trainable=False
        )
        self.moving_variance = self.add_weight(
            name="moving_variance", shape=(1, 1, channels), initializer="ones", trainable=False
        )
        self.mean_weights = self.add_weight(name="mean_weights", shape=(3,), initializer="ones", trainable=True)
        self.variance_weights = self.add_weight(
            name="variance_weights", shape=(3,), initializer="ones", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        training = bool(training)
        if training:
            bn_mean, bn_var = tf.nn.moments(inputs, axes=[0, 1], keepdims=True)
        else:
            bn_mean, bn_var = self.moving_mean, self.moving_variance
        ln_mean, ln_var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        in_mean, in_var = tf.nn.moments(inputs, axes=[1], keepdims=True)

        mean_weights = tf.nn.softmax(self.mean_weights)
        variance_weights = tf.nn.softmax(self.variance_weights)
        mean = mean_weights[0] * bn_mean + mean_weights[1] * ln_mean + mean_weights[2] * in_mean
        variance = variance_weights[0] * bn_var + variance_weights[1] * ln_var + variance_weights[2] * in_var

        outputs = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        if training:
            self.moving_mean.assign(self.momentum * self.moving_mean + (1.0 - self.momentum) * bn_mean)
            self.moving_variance.assign(self.momentum * self.moving_variance + (1.0 - self.momentum) * bn_var)
        return outputs * self.gamma + self.beta

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"axis": self.axis, "momentum": self.momentum, "epsilon": self.epsilon})
        return config

class NormLayer(keras.layers.Layer):
    # Normalization layer dispatching BN / LN / GN / SN / UN / RMS by method name
    def __init__(
        self,
        method: str = "BN",
        groups: int = 32,
        axis: int = -1,
        epsilon: float = 1e-3,
        momentum: float = 0.99,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.method = (method or "").upper()
        self.groups = int(groups)
        self.axis = int(axis)
        self.epsilon = float(epsilon)
        self.momentum = float(momentum)

    def build(self, input_shape: tf.TensorShape) -> None:
        method = self.method
        if method == "BN":
            norm = keras.layers.BatchNormalization(axis=self.axis, momentum=self.momentum, epsilon=self.epsilon)
        elif method == "LN":
            norm = keras.layers.LayerNormalization(axis=self.axis, epsilon=self.epsilon)
        elif method == "GN":
            norm = keras.layers.GroupNormalization(groups=self.groups, axis=self.axis, epsilon=self.epsilon)
        elif method == "SN":
            norm = SwitchableNormalization(axis=self.axis, momentum=self.momentum, epsilon=self.epsilon)
        elif method == "UN":
            norm = keras.layers.UnitNormalization(axis=self.axis)
        elif method == "RMS":
            rms_cls = getattr(keras.layers, "RMSNormalization", None)
            norm = (
                rms_cls(axis=self.axis, epsilon=self.epsilon)
                if rms_cls is not None
                else keras.layers.LayerNormalization(axis=self.axis, epsilon=self.epsilon)
            )
        else:
            norm = keras.layers.Identity()
        self._norm = norm
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        if isinstance(self._norm, (keras.layers.BatchNormalization, SwitchableNormalization)):
            return self._norm(inputs, training=training)
        return self._norm(inputs)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "method": self.method,
                "groups": self.groups,
                "axis": self.axis,
                "epsilon": self.epsilon,
                "momentum": self.momentum,
            }
        )
        return config

class SqueezeExcitation(keras.layers.Layer):
    # Squeeze-and-Excitation channel attention module
    def __init__(self, ratio: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.ratio = int(ratio)

    def build(self, input_shape: tf.TensorShape) -> None:
        channels = int(input_shape[-1])
        self._fc1 = layers.Dense(
            max(1, channels // self.ratio),
            activation="relu",
            kernel_initializer="he_normal",
            name="se_fc1",
        )
        self._fc2 = layers.Dense(
            channels,
            activation="sigmoid",
            kernel_initializer="he_normal",
            name="se_fc2",
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        rank = inputs.shape.rank
        axes = list(range(1, rank - 1)) or [1]
        pooled = tf.reduce_mean(inputs, axis=axes, keepdims=True)
        excitation = self._fc2(self._fc1(pooled))
        return inputs * excitation

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

class InvertedResidualBlock(keras.layers.Layer):
    # Inverted Residual Block: pointwise expand -> depthwise conv -> pointwise conv (or DSC)
    def __init__(
        self,
        expanded_channels: int,
        output_channels: int,
        strides: int = 1,
        kernel_size: int = 6,
        norm: str = "SN",
        activation: Optional[str] = None,
        dropout: Optional[float] = None,
        is_use_expand_conv: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.expanded_channels = int(expanded_channels)
        self.output_channels = int(output_channels)
        self.depthwise_strides = int(strides)
        self.depthwise_kernel_size = int(kernel_size)
        self.norm = norm
        self.activation = activation
        self.dropout = float(dropout) if dropout else None
        self.is_use_expand_conv = bool(is_use_expand_conv)
        self.depth_multiplier = kwargs.get("depth_multiplier", 1)
        self.data_format = kwargs.get("data_format", "channels_last")
        self.padding_mode = kwargs.get("padding_mode", "same")
        self.expand_conv_size = kwargs.get("expand_conv_size", 1)
        self.expand_conv_stride = kwargs.get("expand_conv_stride", 1)
        self.expand_conv_padding_mode = kwargs.get("expand_conv_padding_mode", self.padding_mode)
        self.pointwise_padding_mode = kwargs.get("pointwise_padding_mode", self.padding_mode)
        self.pointwise_stride = kwargs.get("pointwise_stride", 1)
        self.pointwise_kernel_size = kwargs.get("pointwise_kernel_size", 1)

    def build(self, input_shape: tf.TensorShape) -> None:
        self.in_channels = int(input_shape[-1])
        self._pw1 = layers.Conv1D(self.expanded_channels, self.expand_conv_size, strides=self.expand_conv_stride, padding=self.expand_conv_padding_mode, use_bias=False)
        self._dw = layers.DepthwiseConv1D(
            self.depthwise_kernel_size, self.depthwise_strides, padding=self.padding_mode if self.depthwise_strides == 1 else "valid", use_bias=False, activation=None,
            depth_multiplier=self.depth_multiplier,
            data_format=self.data_format
        )  # 1D depthwise convolution
        self._pw2 = layers.Conv1D(self.output_channels, self.pointwise_kernel_size, strides=self.pointwise_stride, padding=self.pointwise_padding_mode, use_bias=False)
        self._norm1 = _make_norm(self.norm)
        self._norm2 = _make_norm(self.norm)
        self._norm3 = _make_norm(self.norm)
        self._act = layers.Activation(self.activation) if self.activation else layers.Identity()
        self._dropout = layers.Dropout(self.dropout) if self.dropout else None
        self._add = layers.Add()
        self._use_shortcut = self.depthwise_strides == 1 and self.in_channels == self.output_channels
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        shortcut = inputs
        if self.is_use_expand_conv:
            x = _apply_norm(self._norm1, self._pw1(inputs), training)
            x = self._act(x)
            if self._dropout is not None:
                x = self._dropout(x, training=training)
        else:
            x = inputs
        x = _apply_norm(self._norm2, self._dw(x), training)
        x = self._act(x)
        if self._dropout is not None:
            x = self._dropout(x, training=training)
        x = _apply_norm(self._norm3, self._pw2(x), training)
        x = self._act(x)
        if self._use_shortcut:
            x = self._add([x, shortcut])
        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "expanded_channels": self.expanded_channels,
                "output_channels": self.output_channels,
                "strides": self.depthwise_strides,
                "kernel_size": self.depthwise_kernel_size,
                "norm": self.norm,
                "activation": self.activation,
                "dropout": self.dropout,
                "is_use_expand_conv": self.is_use_expand_conv,
                "depth_multiplier": self.depth_multiplier,
                "data_format": self.data_format,
                "padding_mode": self.padding_mode,
                "expand_conv_size": self.expand_conv_size,
                "expand_conv_stride": self.expand_conv_stride,
                "expand_conv_padding_mode": self.expand_conv_padding_mode,
                "pointwise_padding_mode": self.pointwise_padding_mode,
                "pointwise_stride": self.pointwise_stride,
                "pointwise_kernel_size": self.pointwise_kernel_size,
            }
        )
        return config


class TemporalConvBlock(keras.layers.Layer):
    # Temporal Convolution Block: stack of conv + norm + activation + pooling
    def __init__(
        self,
        filters: Union[int, Sequence[int]],
        kernel_size: int = 3,
        pool_size: int = 1,
        norm: str = "",
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = list(filters) if isinstance(filters, (tuple, list)) else [int(filters)]
        self.kernel_size = int(kernel_size)
        self.pool_size = int(pool_size)
        self.norm = norm or ""
        self.activation = activation

    def build(self, input_shape: tf.TensorShape) -> None:
        self._convs, self._norms, self._acts, self._pools = [], [], [], []
        for i, filters_num in enumerate(self.filters, start=1):
            conv = layers.Conv1D(int(filters_num), self.kernel_size, padding="same", name=f"tcnn_conv{i}")
            pool = (
                layers.MaxPooling1D(self.pool_size, name=f"tcnn_pool{i}")
                if self.pool_size > 1
                else layers.Identity()
            )
            self._convs.append(conv)
            self._norms.append(_make_norm(self.norm))
            self._acts.append(layers.Activation(self.activation))
            self._pools.append(pool)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = inputs
        for conv, norm, act, pool in zip(self._convs, self._norms, self._acts, self._pools):
            x = _apply_norm(norm, conv(x), training)
            x = act(x)
            x = pool(x)
        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "pool_size": self.pool_size,
                "norm": self.norm,
                "activation": self.activation,
            }
        )
        return config

class DecisionScoreHead_sEMGNet(keras.layers.Layer):
    # DecisionScoreHead for the sEMG branch
    def __init__(
        self,
        num_classes: int,
        multiple: tuple[int, int] = (25, 5),
        dropout: float = 0.5,
        use_hidden: bool = True,
        activation = "swish",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.multiple = tuple(multiple)
        self.dropout = float(dropout)
        self.use_hidden = bool(use_hidden)
        self.activation = activation
        self.output_activation = "linear"
        self.seed = kwargs.get("seed", 2024)

    def build(self, input_shape: tf.TensorShape) -> None:
        self._flatten = layers.Flatten()
        self._dropout = layers.Dropout(self.dropout, seed=self.seed)
        self._decision_primary_dense = layers.Dense(self.num_classes * self.multiple[0], activation=self.activation, name="ffn_dense1")
        self._decision_hidden_dense = layers.Dense(self.num_classes * self.multiple[1], activation=self.activation, name="ffn_dense2")
        self._decision_head = layers.Dense(self.num_classes, activation=self.output_activation, name="ffn_head")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = self._flatten(inputs)
        x = self._decision_primary_dense(x)
        if self.use_hidden:
            x = self._decision_hidden_dense(x)
        x = self._dropout(x, training=training)
        return self._decision_head(x)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "multiple": self.multiple,
                "dropout": self.dropout,
                "use_hidden": self.use_hidden,
                "activation": self.activation,
            }
        )
        return config

class DynamicRoutingLayer(keras.layers.Layer):
    # Dynamic routing layer
    def __init__(
        self,
        num_caps: int,
        dim_caps: int,
        routings: int = 3,
        share_weights: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_caps = int(num_caps)
        self.dim_caps = int(dim_caps)
        self.routings = int(routings)
        self.share_weights = bool(share_weights)

    def build(self, input_shape: tf.TensorShape) -> None:
        input_dim = int(input_shape[-1])
        if self.share_weights:
            kernel_shape = (1, input_dim, self.num_caps * self.dim_caps)
        else:
            input_num = int(input_shape[-2])
            kernel_shape = (input_num, input_dim, self.num_caps * self.dim_caps)
        self.kernel = self.add_weight(
            name="drb_kernel", shape=kernel_shape, initializer="glorot_uniform", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.share_weights:
            hat_inputs = tf.nn.conv1d(inputs, self.kernel, stride=1, padding="VALID")
        else:
            hat_inputs = tf.einsum("bni,nio->bno", inputs, self.kernel)

        batch_size = tf.shape(inputs)[0]
        input_num = tf.shape(inputs)[1]
        hat_inputs = tf.reshape(hat_inputs, (batch_size, input_num, self.num_caps, self.dim_caps))
        hat_inputs = tf.transpose(hat_inputs, (0, 2, 1, 3))
        b = tf.zeros_like(hat_inputs[:, :, :, 0])

        output = None
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            s = tf.einsum("bin,binj->bij", c, hat_inputs)
            if i < self.routings - 1:
                s = tf.nn.l2_normalize(s, axis=-1)
                b = tf.einsum("bij,binj->bin", s, hat_inputs)
            output = s
        return output

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "num_caps": self.num_caps,
                "dim_caps": self.dim_caps,
                "routings": self.routings,
                "share_weights": self.share_weights,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_caps, self.dim_caps)

class DecisionScoreHead_FeatureNet(keras.layers.Layer):
    # DecisionScoreHead for the feature branch
    def __init__(
        self,
        num_classes: int,
        filters: int = 32,
        kernel_size: int = 5,
        dim_caps: int = 30,
        routings: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.dim_caps = int(dim_caps)
        self.routings = int(routings)

    def build(self, input_shape: tf.TensorShape) -> None:
        self._conv = layers.Conv1D(self.filters, self.kernel_size, padding="same", activation="relu", name="drb_conv")
        self._pool = layers.MaxPooling1D(pool_size=2, name="drb_pool")
        self._routing = DynamicRoutingLayer(
            num_caps=self.num_classes, dim_caps=self.dim_caps, routings=self.routings, name="drb_routing"
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self._conv(inputs)
        x = self._pool(x)
        x = self._routing(x)
        # Take the vector norm (Euclidean length) of each capsule as the output
        return tf.sqrt(tf.reduce_sum(tf.square(x), axis=2))

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "dim_caps": self.dim_caps,
                "routings": self.routings,
            }
        )
        return config


class LearnableFusion(keras.layers.Layer):
    # Learnable weighted fusion with concatenation along a specified axis
    def __init__(self, axis: int = -1, init: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.axis = int(axis)
        self.init = float(init)

    def build(self, input_shapes: Sequence[tf.TensorShape]) -> None:
        shape1, shape2 = input_shapes
        self.lambda_weight = self.add_weight(
            name="lambda_weight",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.init),
            trainable=True,
        )
        self._align1, self._align2 = layers.Identity(), layers.Identity()
        if self.axis == -1:
            t1, t2 = int(shape1[1]), int(shape2[1])
            if t1 < t2:
                self._align1 = layers.ZeroPadding1D((0, t2 - t1))
            elif t1 > t2:
                self._align2 = layers.ZeroPadding1D((0, t1 - t2))
        elif self.axis == -2:
            c1, c2 = int(shape1[-1]), int(shape2[-1])
            if c1 < c2:
                self._align1 = layers.Dense(c2)
            elif c1 > c2:
                self._align2 = layers.Dense(c1)
        super().build(input_shapes)

    def call(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        x1 = self._align1(inputs[0])
        x2 = self._align2(inputs[1])
        weight = self.lambda_weight
        return tf.concat([weight * x1, (1.0 - weight) * x2], axis=self.axis)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"axis": self.axis, "init": self.init})
        return config


class LearnableWeightedSum(keras.layers.Layer):
    # Learnable weighted sum (Lambda-s) for decision-level fusion
    def __init__(self, init: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.init = float(init)

    def build(self, input_shapes: Sequence[tf.TensorShape]) -> None:
        self.alpha = self.add_weight(
            name="alpha_weight",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.init),
            trainable=True,
        )
        super().build(input_shapes)

    def call(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        return self.alpha * inputs[0] + (1.0 - self.alpha) * inputs[1]

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"init": self.init})
        return config

class FeedForwardRefBlock(keras.layers.Layer):
    # Feed-Forward Network: flatten then hidden dense mapping to class probabilities
    def __init__(
        self,
        num_classes: int,
        middle_hidden_units = 25,
        use_hidden: bool = True,
        use_softmax: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.middle_hidden_units_multiple = int(middle_hidden_units)
        self.dropout = float(0.2)
        self.use_hidden = bool(use_hidden)
        self.use_softmax = bool(use_softmax)
        self.seed = kwargs.get("seed", 2024)
        self.output_activation = "softmax" if self.use_softmax else "linear"

    def build(self, input_shape: tf.TensorShape) -> None:
        self._flatten = layers.Flatten()
        self._dropout = layers.Dropout(self.dropout, seed=self.seed)
        self._middle_hidden_dense = layers.Dense(self.num_classes * self.middle_hidden_units_multiple, activation='relu', use_bias = True, kernel_initializer = "glorot_uniform", name="ffn_hidden_dense")
        self._head = layers.Dense(self.num_classes, activation=self.output_activation, name="ffn_head")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = self._flatten(inputs)
        x = self._middle_hidden_dense(x)
        x = self._dropout(x, training=training)
        return self._head(x)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "middle_hidden_units": self.middle_hidden_units_multiple,
                "dropout": self.dropout,
                "use_softmax": self.use_softmax,
                "use_hidden": self.use_hidden,
            }
        )
        return config

class FeedForwardFeaBlock(keras.layers.Layer):
    # Feed-Forward Network: flatten then hidden dense mapping to class probabilities
    def __init__(
        self,
        num_classes: int,
        middle_hidden_units = 25,
        use_hidden: bool = True,
        use_softmax: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.middle_hidden_units_multiple = int(middle_hidden_units)
        self.dropout = float(0.6)
        self.use_hidden = bool(use_hidden)
        self.use_softmax = bool(use_softmax)
        self.seed = kwargs.get("seed", 2024)
        self.output_activation = "softmax" if self.use_softmax else "linear"

    def build(self, input_shape: tf.TensorShape) -> None:
        self._flatten = layers.Flatten()
        self._dropout = layers.Dropout(self.dropout, seed=self.seed)
        self._middle_hidden_dense = layers.Dense(self.num_classes * self.middle_hidden_units_multiple, activation='relu', use_bias = True, kernel_initializer = "glorot_uniform", name="ffn_hidden_dense")
        self._head = layers.Dense(self.num_classes, activation=self.output_activation, name="ffn_head")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = self._flatten(inputs)
        x = self._middle_hidden_dense(x)
        x = self._dropout(x, training=training)
        return self._head(x)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "middle_hidden_units": self.middle_hidden_units_multiple,
                "dropout": self.dropout,
                "use_softmax": self.use_softmax,
                "use_hidden": self.use_hidden,
            }
        )
        return config
