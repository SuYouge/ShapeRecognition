import  model
import  input
import  tensorflow as tf
import time
from datetime import datetime


# 注意每次修改batch的大小后要清空result
tfrecords_name = input.train_filename
batch_size = 128
learning_rate = 0.001
train_dir = 'C:\\Users\YogurtSuee\PycharmProjects\shape_recognition\\result'
board_dir = 'C:\\Users\YogurtSuee\PycharmProjects\shape_recognition\\board'
max_steps = 10000
log_frequency = 5
n_classes = 3
test_name = input.test_filename
batch_size_test = 200
#epoch = 20

def train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        images, labels = input.createBatch(tfrecords_name,batch_size)
        logits = model.inference(images,batch_size,n_classes)
        loss = model.loss(logits, labels)
        accuracy = model.evaluation(logits, labels)
        train_op = model.trainning(loss, learning_rate,global_step)
        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                #if self._step % log_frequency == 0:
                    # print(self.run(accuracy))
                    # print("step %d, accuracy = %.2f"%(self._step ,accuracy))
                return tf.train.SessionRunArgs([loss,accuracy])  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time
                    [loss_value,accuracy_value] = run_values.results
                    #accuracy_value = run_context.accuracy
                    examples_per_sec = log_frequency * batch_size / duration
                    sec_per_batch = float(duration / log_frequency)
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))
                    print('Accuracy = %.2f'%accuracy_value)
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir = train_dir,
                hooks=[tf.train.StopAtStepHook(last_step = max_steps),
                       tf.train.NanTensorHook(loss),
                       tf.train.SummarySaverHook(save_steps=5,output_dir=board_dir,summary_op=tf.summary.merge_all()),
                       _LoggerHook()]
                ) as mon_sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=mon_sess, coord=coord)
            while not mon_sess.should_stop():
                mon_sess.run(train_op)
                #print('dont stop')
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)
    train()