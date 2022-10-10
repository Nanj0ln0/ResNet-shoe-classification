import os
import tensorflow as tf
from picture import get_file
from ResNet import resnet18, resnet34

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255. - 0.5  # 给到-1~1范围
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x_all, y_all) = get_file()     # (15000, 100, 100, 3) (15000, 3)
y_all = tf.one_hot(y_all, depth=3)
print(x_all.shape, y_all.shape)

x_train, x_val = tf.split(x_all, num_or_size_splits=[12000, 3000])
y_train, y_val = tf.split(y_all, num_or_size_splits=[12000, 3000])


batch_size = 16
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.map(preprocess).shuffle(10000).batch(batch_size)
db_test = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_test = db_test.map(preprocess).batch(batch_size)


def main():
    model = resnet18()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer,
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(db_train,
              epochs=50,
              validation_data=db_test,
              validation_steps=2)


if __name__ == '__main__':
    main()

