import tensorflow as tf




if __name__ == "__main__":

    size = 1
    stride = 1
    padding = "SAME"
    image = tf.random.normal((2, 16, 16, 5))
    channels = int(image.shape[-1])
    print(image)
    print(image.shape)

    kernel = tf.reshape(
    tf.eye(size * size * channels, dtype=image.dtype),
    (size, size, channels, channels * size * size))
    print(kernel)
    print(kernel.shape)

    out = tf.nn.conv2d(image, kernel, strides=stride, padding=padding)
    print(out.shape)