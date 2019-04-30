import model
import tensorflow as tf
import numpy as np
from PIL import Image
import  matplotlib.pyplot as plt
import  os
import input
# tensorboard --logdir=C:\Users\YogurtSuee\PycharmProjects\shape_recognition\board
IMAGE_SIZE = input.IMAGE_SIZE
result = 'result'

# 测试图片
def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 3

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 30, 30, 3])

        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[30, 30, 3])

        # you need to change the directories to yours.
        logs_train_dir = result

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('这是四边形的可能性为： %.6f' % prediction[:, 0])
            elif max_index == 1:
                print('这是三角的可能性为： %.6f' % prediction[:, 1])
            elif max_index == 2:
                print ('这是圆形的可能性为： %.6f' % prediction[:, 2])
            else:
                print('网络不能判断')
            return max_index

def evaluate_all_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 3

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 30, 30, 3])

        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[30, 30, 3])

        # you need to change the directories to yours.
        logs_train_dir = result

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            # if max_index == 0:
            #     print('这是四边形的可能性为： %.6f' % prediction[:, 0])
            # elif max_index == 1:
            #     print('这是三角的可能性为： %.6f' % prediction[:, 1])
            # elif max_index == 2:
            #     print ('这是圆形的可能性为： %.6f' % prediction[:, 2])
            # else:
            #     print('网络不能判断')
            return max_index


# ------------------------------------------------------------------------

def test_all():
    data_dir = 'C:\\Users\YogurtSuee\PycharmProjects\shape_recognition\data\origin\\test'
    """
    输入测试集的地址
    """
    classes = {'cir', 'rec','tri'}
    shapes = ['rec','tri','cir']
    # rec = 0
    # tri = 1
    # cir = 2

    acc = 0
    for index, name in enumerate(classes):
        class_path = data_dir + '\\' + name + '\\'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name  # 每个图片的地址
            img = Image.open(img_path)
            imag = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            image = np.array(imag)
            shape = evaluate_all_image(image)
            if shapes[shape] == name:
                acc+=1
            else:
                format_str = ('%s: 处的%s将识别为 %s')
                print(format_str %(img_path,name, shapes[shape]))
    acc = acc/200
    print("对测试集的准确率为%.2f"%acc)

if __name__ == '__main__':
    #img = Image.open('C:\\Users\YogurtSuee\PycharmProjects\shape_recognition\data\origin\\test\\rec\\val_111.jpg')
    #img = Image.open('C:\\Users\YogurtSuee\PycharmProjects\shape_recognition\data\origin\\test\\tri\\val_8.jpg')
    #img = Image.open('C:\\Users\YogurtSuee\PycharmProjects\shape_recognition\data\origin\\test\\rec\\val_102.jpg')
    #img = Image.open('C:\\Users\YogurtSuee\PycharmProjects\shape_recognition\data\origin\\test\\rec\\val_132.jpg')
    #img = Image.open('C:\\Users\YogurtSuee\PycharmProjects\shape_recognition\data\origin\\test\\rec\\val_182.jpg')
    #img = Image.open('C:\\Users\YogurtSuee\PycharmProjects\shape_recognition\data\origin\\test\\rec\\val_196.jpg')
    # img = Image.open('C:\\Users\YogurtSuee\PycharmProjects\shape_recognition\data\origin\\test\\rec\\val_93.jpg')
    # plt.imshow(img)
    # plt.show()
    # imag = img.resize([30, 30])
    # image = np.array(imag)
    # evaluate_one_image(image)
    test_all()
