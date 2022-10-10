import os
import keras
import tensorflow as tf
from tensorflow.python.keras import layers, Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BasicBlock(layers.Layer):
    """
    实现残差网络的短接层
    """

    def __init__(self, filter_num, stride=1):
        """
        filter_num: 输入的channel,转化为流动的一个channel,通道数量,
        stride:采样步长，默认为1，不影响图片的大小，为2的话就是二分采样
        """
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, kernel_size=[3, 3], strides=stride, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = layers.Activation("relu")

        self.conv2 = layers.Conv2D(filter_num, kernel_size=[3, 3], strides=1, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, kernel_size=(1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        """
        前向传播
        """
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        identity = self.downsample(inputs)

        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


class ResNet(keras.Model):
    """
    ResNet18
    由1个预处理层
     4个Res-Block 1个Res-Block4层  分别对应2个短接层  1个短接层是2层分别对应2次卷积
     1个全链接层用于分类
    """

    def __init__(self, layer_dims, num_classes=3):
        super(ResNet, self).__init__()

        """
        预处理层，第一层
        """
        self.stem = Sequential([layers.Conv2D(128, kernel_size=[3, 3], strides=(1, 1)),
                                tf.keras.layers.BatchNormalization(),
                                layers.Activation("relu"),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="same")
                                ])
        """
        创建4个Res-Block中间层
        """
        self.layer1 = self.build_resblock(128, layer_dims[0], stride=1)
        self.layer2 = self.build_resblock(256, layer_dims[1], stride=2)  # stride 图像大小减少1倍
        self.layer3 = self.build_resblock(512, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        """
        我们没有办法知道最后一层Res-Block的输出的图像大小，我们就自适应输出
        """
        self.avgpool = layers.GlobalAveragePooling2D()
        """
        全链接层
        """
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        """
        前向预算
        """
        x = self.stem(inputs, training=training)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)

        logic = self.fc(x)

        return logic

    def build_resblock(self, filter_num, blocks, stride=1):
        """
        残差网络的基本单元Res-Block
        """
        res_block = Sequential()
        # 下采样  当调用时，是有可能stride不为1的，有可能不需要下采样，在开始就进行一次下采样，然后后面的都不能下采样就行了
        res_block.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_block.add(BasicBlock(filter_num, stride=1))

        return res_block


# 定义层数
def resnet18():
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 3])
