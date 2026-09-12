# Auto-derived from the validated V7 three-modality inference architecture.
# Source basis: train_v7_three_modality_fixed.py
# This file contains model construction only; it does not train models.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

IMAGE_SIZE = 170
L2_ALPHA = 0.0005
TCIC_DROPOUT = 0.50


def cbam(x, reduction=16, name="cbam"):
    c = int(x.shape[-1])
    hidden = max(c // reduction, 1)

    avg = layers.GlobalAveragePooling2D(name=name + "_avg")(x)
    mx = layers.GlobalMaxPooling2D(name=name + "_max")(x)

    d1 = layers.Dense(hidden, activation="relu", name=name + "_d1")
    d2 = layers.Dense(c, name=name + "_d2")

    ca = layers.Add(name=name + "_cadd")([d2(d1(avg)), d2(d1(mx))])
    ca = layers.Activation("sigmoid", name=name + "_csig")(ca)
    ca = layers.Reshape((1, 1, c), name=name + "_reshape")(ca)
    x = layers.Multiply(name=name + "_cmul")([x, ca])

    avs = layers.Lambda(
        lambda z: tf.reduce_mean(z, axis=-1, keepdims=True),
        name=name + "_sav",
    )(x)
    mxs = layers.Lambda(
        lambda z: tf.reduce_max(z, axis=-1, keepdims=True),
        name=name + "_smx",
    )(x)

    sa = layers.Concatenate(name=name + "_sconcat")([avs, mxs])
    sa = layers.Conv2D(
        1,
        7,
        padding="same",
        activation="sigmoid",
        name=name + "_sconv",
    )(sa)

    return layers.Multiply(name=name + "_smul")([x, sa])

def build_paper_tcic(modality):
    """
    Direct functional implementation of the intended TCIC structure:

    input -> Inception-ResNet-v2 stem -> 5 A + CBAM -> Reduction-A
          -> 10 B + CBAM -> Reduction-B -> 5 C + CBAM
          -> final conv -> GAP -> dropout -> 3-class softmax

    This does not extract submodels from keras.applications and therefore
    avoids the disconnected-graph errors caused by get_layer('block35_1').
    """

    inp = keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
        name=f"{modality}_input",
    )

    # Repeat the ONE modality into three identical planes.
    x = layers.Concatenate(
        name=f"{modality}_single_to_three"
    )([inp, inp, inp])

    def conv_bn(
        x,
        filters,
        kernel_size,
        strides=1,
        padding="same",
        activation="relu",
        name=None,
    ):
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=True,
            name=None if name is None else name + "_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=-1,
            name=None if name is None else name + "_bn",
        )(x)
        if activation is not None:
            x = layers.Activation(
                activation,
                name=None if name is None else name + "_ac",
            )(x)
        return x

    # --------------------------------------------------------
    # Stem
    # --------------------------------------------------------
    x = conv_bn(
        x, 32, 3, strides=2, padding="valid",
        name=f"{modality}_stem1",
    )
    x = conv_bn(
        x, 32, 3, padding="valid",
        name=f"{modality}_stem2",
    )
    x = conv_bn(
        x, 64, 3, padding="same",
        name=f"{modality}_stem3",
    )
    x = layers.MaxPooling2D(
        3, 2, padding="valid",
        name=f"{modality}_stem_pool1",
    )(x)
    x = conv_bn(
        x, 80, 1, padding="valid",
        name=f"{modality}_stem4",
    )
    x = conv_bn(
        x, 192, 3, padding="valid",
        name=f"{modality}_stem5",
    )
    x = layers.MaxPooling2D(
        3, 2, padding="valid",
        name=f"{modality}_stem_pool2",
    )(x)

    # --------------------------------------------------------
    # Mixed 5b -> 320 channels
    # --------------------------------------------------------
    b0 = conv_bn(
        x, 96, 1,
        name=f"{modality}_mixed5b_b0",
    )

    b1 = conv_bn(
        x, 48, 1,
        name=f"{modality}_mixed5b_b1_1",
    )
    b1 = conv_bn(
        b1, 64, 5,
        name=f"{modality}_mixed5b_b1_5",
    )

    b2 = conv_bn(
        x, 64, 1,
        name=f"{modality}_mixed5b_b2_1",
    )
    b2 = conv_bn(
        b2, 96, 3,
        name=f"{modality}_mixed5b_b2_3a",
    )
    b2 = conv_bn(
        b2, 96, 3,
        name=f"{modality}_mixed5b_b2_3b",
    )

    b3 = layers.AveragePooling2D(
        3, 1, padding="same",
        name=f"{modality}_mixed5b_pool",
    )(x)
    b3 = conv_bn(
        b3, 64, 1,
        name=f"{modality}_mixed5b_b3",
    )

    x = layers.Concatenate(
        axis=-1,
        name=f"{modality}_mixed5b",
    )([b0, b1, b2, b3])

    # --------------------------------------------------------
    # 5 x Inception-ResNet-A + CBAM
    # --------------------------------------------------------
    def block35(x, i):
        shortcut = x

        b0 = conv_bn(
            x, 32, 1,
            name=f"{modality}_A{i}_b0_1x1",
        )

        b1 = conv_bn(
            x, 32, 1,
            name=f"{modality}_A{i}_b1_1x1",
        )
        b1 = conv_bn(
            b1, 32, 3,
            name=f"{modality}_A{i}_b1_3x3",
        )

        b2 = conv_bn(
            x, 32, 1,
            name=f"{modality}_A{i}_b2_1x1",
        )
        b2 = conv_bn(
            b2, 48, 3,
            name=f"{modality}_A{i}_b2_3x3a",
        )
        b2 = conv_bn(
            b2, 64, 3,
            name=f"{modality}_A{i}_b2_3x3b",
        )

        mixed = layers.Concatenate(
            axis=-1,
            name=f"{modality}_A{i}_mixed",
        )([b0, b1, b2])

        up = layers.Conv2D(
            320, 1,
            padding="same",
            use_bias=True,
            name=f"{modality}_A{i}_up",
        )(mixed)

        up = layers.Lambda(
            lambda z: z * 0.17,
            name=f"{modality}_A{i}_scale",
        )(up)

        x = layers.Add(
            name=f"{modality}_A{i}_add",
        )([shortcut, up])

        x = layers.Activation(
            "relu",
            name=f"{modality}_A{i}_relu",
        )(x)

        return x

    for i in range(1, 6):
        x = block35(x, i)
        x = cbam(
            x,
            reduction=16,
            name=f"{modality}_A{i}_CBAM",
        )

    # --------------------------------------------------------
    # Reduction-A -> 1088 channels
    # --------------------------------------------------------
    b0 = conv_bn(
        x, 384, 3, strides=2, padding="valid",
        name=f"{modality}_reductionA_b0",
    )

    b1 = conv_bn(
        x, 256, 1,
        name=f"{modality}_reductionA_b1_1",
    )
    b1 = conv_bn(
        b1, 256, 3,
        name=f"{modality}_reductionA_b1_3",
    )
    b1 = conv_bn(
        b1, 384, 3, strides=2, padding="valid",
        name=f"{modality}_reductionA_b1_3s",
    )

    b2 = layers.MaxPooling2D(
        3, 2, padding="valid",
        name=f"{modality}_reductionA_pool",
    )(x)

    x = layers.Concatenate(
        axis=-1,
        name=f"{modality}_mixed6a",
    )([b0, b1, b2])

    # --------------------------------------------------------
    # 10 x Inception-ResNet-B + CBAM
    # --------------------------------------------------------
    def block17(x, i):
        shortcut = x

        b0 = conv_bn(
            x, 192, 1,
            name=f"{modality}_B{i}_b0_1x1",
        )

        b1 = conv_bn(
            x, 128, 1,
            name=f"{modality}_B{i}_b1_1x1",
        )
        b1 = conv_bn(
            b1, 160, (1, 7),
            name=f"{modality}_B{i}_b1_1x7",
        )
        b1 = conv_bn(
            b1, 192, (7, 1),
            name=f"{modality}_B{i}_b1_7x1",
        )

        mixed = layers.Concatenate(
            axis=-1,
            name=f"{modality}_B{i}_mixed",
        )([b0, b1])

        up = layers.Conv2D(
            1088, 1,
            padding="same",
            use_bias=True,
            name=f"{modality}_B{i}_up",
        )(mixed)

        up = layers.Lambda(
            lambda z: z * 0.10,
            name=f"{modality}_B{i}_scale",
        )(up)

        x = layers.Add(
            name=f"{modality}_B{i}_add",
        )([shortcut, up])

        x = layers.Activation(
            "relu",
            name=f"{modality}_B{i}_relu",
        )(x)

        return x

    for i in range(1, 11):
        x = block17(x, i)
        x = cbam(
            x,
            reduction=16,
            name=f"{modality}_B{i}_CBAM",
        )

    # --------------------------------------------------------
    # Reduction-B -> 2080 channels
    # --------------------------------------------------------
    b0 = conv_bn(
        x, 256, 1,
        name=f"{modality}_reductionB_b0_1",
    )
    b0 = conv_bn(
        b0, 384, 3, strides=2, padding="valid",
        name=f"{modality}_reductionB_b0_3",
    )

    b1 = conv_bn(
        x, 256, 1,
        name=f"{modality}_reductionB_b1_1",
    )
    b1 = conv_bn(
        b1, 288, 3, strides=2, padding="valid",
        name=f"{modality}_reductionB_b1_3",
    )

    b2 = conv_bn(
        x, 256, 1,
        name=f"{modality}_reductionB_b2_1",
    )
    b2 = conv_bn(
        b2, 288, 3,
        name=f"{modality}_reductionB_b2_3a",
    )
    b2 = conv_bn(
        b2, 320, 3, strides=2, padding="valid",
        name=f"{modality}_reductionB_b2_3b",
    )

    b3 = layers.MaxPooling2D(
        3, 2, padding="valid",
        name=f"{modality}_reductionB_pool",
    )(x)

    x = layers.Concatenate(
        axis=-1,
        name=f"{modality}_mixed7a",
    )([b0, b1, b2, b3])

    # --------------------------------------------------------
    # 5 x Inception-ResNet-C + CBAM
    # --------------------------------------------------------
    def block8(x, i):
        shortcut = x

        b0 = conv_bn(
            x, 192, 1,
            name=f"{modality}_C{i}_b0_1x1",
        )

        b1 = conv_bn(
            x, 192, 1,
            name=f"{modality}_C{i}_b1_1x1",
        )
        b1 = conv_bn(
            b1, 224, (1, 3),
            name=f"{modality}_C{i}_b1_1x3",
        )
        b1 = conv_bn(
            b1, 256, (3, 1),
            name=f"{modality}_C{i}_b1_3x1",
        )

        mixed = layers.Concatenate(
            axis=-1,
            name=f"{modality}_C{i}_mixed",
        )([b0, b1])

        up = layers.Conv2D(
            2080, 1,
            padding="same",
            use_bias=True,
            name=f"{modality}_C{i}_up",
        )(mixed)

        up = layers.Lambda(
            lambda z: z * 0.20,
            name=f"{modality}_C{i}_scale",
        )(up)

        x = layers.Add(
            name=f"{modality}_C{i}_add",
        )([shortcut, up])

        x = layers.Activation(
            "relu",
            name=f"{modality}_C{i}_relu",
        )(x)

        return x

    for i in range(1, 6):
        x = block8(x, i)
        x = cbam(
            x,
            reduction=16,
            name=f"{modality}_C{i}_CBAM",
        )

    # --------------------------------------------------------
    # Final convolution + head
    # --------------------------------------------------------
    x = conv_bn(
        x,
        1536,
        1,
        name=f"{modality}_conv7b",
    )

    x = layers.GlobalAveragePooling2D(
        name=f"{modality}_gap"
    )(x)

    x = layers.Dropout(
        TCIC_DROPOUT,
        name=f"{modality}_dropout",
    )(x)

    x = layers.Dense(
        1,
        kernel_regularizer=regularizers.l2(L2_ALPHA),
        name=f"{modality}_paper_fc",
    )(x)

    out = layers.Dense(
        3,
        activation="softmax",
        name=f"{modality}_class_output",
    )(x)

    return keras.Model(
        inputs=inp,
        outputs=out,
        name=f"Paper_TCIC_{modality}",
    )

def build_paper_tcie(modality, cls):
    inp = keras.Input(
        (IMAGE_SIZE, IMAGE_SIZE, 1),
        name=f"{modality}_c{cls}_input",
    )

    reg = regularizers.l2(L2_ALPHA)

    def conv(x, filters, kernel, name):
        x = layers.Conv2D(
            filters,
            kernel,
            padding="same",
            kernel_regularizer=reg,
            name=name,
        )(x)
        return layers.LeakyReLU(
            negative_slope=0.1,
            name=name + "_lrelu",
        )(x)

    x = conv(inp, 8, 5, f"{modality}_c{cls}_conv1")
    x = layers.MaxPooling2D(
        3, 2, padding="same",
        name=f"{modality}_c{cls}_pool1",
    )(x)

    x = conv(x, 32, 3, f"{modality}_c{cls}_conv2")
    x = conv(x, 32, 3, f"{modality}_c{cls}_conv3")
    x = conv(x, 32, 3, f"{modality}_c{cls}_conv4")

    x = layers.MaxPooling2D(
        3, 2, padding="same",
        name=f"{modality}_c{cls}_pool2",
    )(x)

    x = conv(x, 64, 3, f"{modality}_c{cls}_conv5")
    x = conv(x, 64, 3, f"{modality}_c{cls}_conv6")

    x = layers.Flatten(
        name=f"{modality}_c{cls}_flatten"
    )(x)

    x = layers.Dense(
        512,
        kernel_regularizer=reg,
        name=f"{modality}_c{cls}_fc1",
    )(x)
    x = layers.LeakyReLU(
        negative_slope=0.1,
        name=f"{modality}_c{cls}_fc1_lrelu",
    )(x)

    x = layers.Dense(
        16,
        kernel_regularizer=reg,
        name=f"{modality}_c{cls}_fc2",
    )(x)
    x = layers.LeakyReLU(
        negative_slope=0.1,
        name=f"{modality}_c{cls}_fc2_lrelu",
    )(x)

    out = layers.Dense(
        1,
        name=f"{modality}_c{cls}_vmax",
    )(x)

    return keras.Model(
        inp,
        out,
        name=f"Paper_TCIE_{modality}_class{cls}",
    )
