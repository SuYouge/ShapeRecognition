import tensorflow as tf
import os
from PIL import Image


IMAGE_SIZE = 30
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 3
epoch = 20

train_filename = "C:\\Users\YogurtSuee\PycharmProjects\shape_recognition\data\\tfrecords\\train.tfrecords"
test_filename = "C:\\Users\YogurtSuee\PycharmProjects\shape_recognition\data\\tfrecords\\test.tfrecords"

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 制作训练集
def createTFRecord_train(filename, mapfile):
    class_map = {}
    data_dir = 'C:\\Users\YogurtSuee\PycharmProjects\shape_recognition\data\origin\\train'
    """
    输入训练集的地址
    """
    classes = {'cir', 'rec','tri'}
    # 输出TFRecord文件的地址

    writer = tf.python_io.TFRecordWriter(filename)
    #for i in range(epoch):
    for index, name in enumerate(classes):
        class_path = data_dir + '\\' + name + '\\'
        class_map[index] = name
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name  # 每个图片的地址
            img = Image.open(img_path)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_raw = img.tobytes()  # 将图片转化成二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(index),
                'image_raw': _bytes_feature(img_raw)
            }))
            #for i in range(epoch):
            writer.write(example.SerializeToString())
    writer.close()

    txtfile = open(mapfile, 'w+')
    for key in class_map.keys():
        txtfile.writelines(str(key) + ":" + class_map[key] + "\n")
    txtfile.close()

# 制作测试集
def createTFRecord_test(filename, mapfile):
    class_map = {}
    data_dir = 'C:\\Users\YogurtSuee\PycharmProjects\shape_recognition\data\origin\\test'
    """
    输入测试集的地址
    """
    classes = {'cir', 'rec','tri'}
    # 输出TFRecord文件的地址

    writer = tf.python_io.TFRecordWriter(filename)

    for index, name in enumerate(classes):
        class_path = data_dir + '\\' + name + '\\'
        class_map[index] = name
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name  # 每个图片的地址
            img = Image.open(img_path)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_raw = img.tobytes()  # 将图片转化成二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(index),
                'image_raw': _bytes_feature(img_raw)
            }))
            writer.write(example.SerializeToString())
    writer.close()

    txtfile = open(mapfile, 'w+')
    for key in class_map.keys():
        txtfile.writelines(str(key) + ":" + class_map[key] + "\n")
    txtfile.close()

# 读取 tfrecords 中的数据
def read_and_decode(filename):
    # 创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    # 创建一个队列来维护输入文件列表
    filename_queue = tf.train.string_input_producer([filename], shuffle=True, num_epochs=epoch,seed=2)
    # 从文件中读出一个样例，也可以使用read_up_to一次读取多个样例
    _, serialized_example = reader.read(filename_queue)
    #     print _,serialized_example

    # 解析读入的一个样例，如果需要解析多个，可以用parse_example
    features = tf.parse_single_example(
        serialized_example,
        features={'label': tf.FixedLenFeature([], tf.int64),
                  'image_raw': tf.FixedLenFeature([], tf.string), })
    # 将字符串解析成图像对应的像素数组
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [IMAGE_SIZE, IMAGE_SIZE, 3])  # reshape
    img = tf.image.per_image_standardization(img)
    labels = tf.cast(features['label'], tf.int32)
    return img, labels

# 从数据集创建Batch
def createBatch(filename, batchsize):
    images, labels = read_and_decode(filename)
    min_after_dequeue = epoch * 1000
    capacity = min_after_dequeue + 3 * batchsize
    #capacity = 200
    #print(images.shape)
    image_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                      batch_size=batchsize,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue
                                                      )
    #深度为3的独热编码
    label_batch = tf.one_hot(label_batch, depth=3)
    return image_batch, label_batch

if __name__ == "__main__":
    # 训练图片两张为一个batch,进行训练，测试图片一起进行测试
    mapfile = "C:\\Users\YogurtSuee\PycharmProjects\shape_recognition\data\\tfrecords\classmap.txt"

    createTFRecord_train(train_filename, mapfile)
    image_batch, label_batch = createBatch(filename=train_filename, batchsize = 128)
    #createTFRecord_test(test_filename, mapfile)
    # test_images, test_labels = createBatch(filename=test_filename, batchsize = 64)

    with tf.Session() as sess:
        initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(initop)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # try:
        #     step = 0
        #     while 1:
        #         _test_images, _test_labels = sess.run([test_images, test_labels])
        #         step += 1
        #         #print(step)
        #         #print(_test_images.shape)
        #         #print (_test_images.shape)
        #         #print(_test_labels)
        #         #correct_pred = tf.equal(tf.argmax(_test_labels, 1), tf.argmax(_test_labels, 1))
        #         #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #         #print(accuracy)
        # except tf.errors.OutOfRangeError:
        #     print("testData done!")

        try:
            step = 0
            while 1:
                _image_batch, _label_batch = sess.run([image_batch, label_batch])
                step += 1
                #print(step)
                #print(_image_batch.shape)
                #print(_label_batch.shape)
        except tf.errors.OutOfRangeError:
            print(step * 128)
            print("trainData done!")
        coord.request_stop()
        coord.join(threads)