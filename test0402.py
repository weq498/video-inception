import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import sys
import cv2
import time
import threading

tf.app.flags.DEFINE_string('train_directory', '/home/weq498/movie/train',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/home/weq498/movie/validation',
                           'Validation data directory.')
tf.app.flags.DEFINE_string('output_directory', '/tmp',
                           'Output data directory')
tf.app.flags.DEFINE_integer('train_shards', 5,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 5,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 5,
                            'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_string('labels_file', '/home/weq498/movie/labels.txt'
                           , 'Labels file')

FLAGS = tf.app.flags.FLAGS


class ImageCoder(object):
    def __init__(self):
        self._sess = tf.Session()
        self._frame_data = tf.placeholder(dtype=tf.uint8, shape=(None, None, 3), name='Frame_data')
        self._resize_data = tf.image.resize_images(self._frame_data, (1280, 720), method=tf.image.ResizeMethod.BILINEAR)
        self._encode_jpeg = tf.image.encode_jpeg(self._frame_data, format='rgb', quality=100)


    def encode_jpeg(self, image_data):
        return self._sess.run(self._encode_jpeg,
                              feed_dict={self._frame_data: image_data})

    def resize_frame(self, image_data):
        return self._sess.run(self._resize_data,
                              feed_dict={self._frame_data: image_data})


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width):
    colorspace = 'RGB'
    channels = 3
    image_format = 'jpeg'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(text),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example


def _process_image(frame_image, coder):
    """ original function:_process_image()
    Process frame image to encode jpeg
    Args:
        frame_image: integer, frame image buffer.
        coder: instance of ImageCoder to provide Tensorflow image coding utils.
    Returns:
        image_buffer: string, JPEG encoding of RGB image.
    """
    _image_buffer = coder.resize_frame(frame_image)
    # print(len(_image_buffer[0]))
    # Resize each frame and return data.
    _image_buffer = coder.encode_jpeg(_image_buffer)
    return _image_buffer


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
    frame_counter = 0
    counter = 0
    num_threads = len(ranges)
    num_shards_per_batch = int(num_shards / num_threads)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    # Use flags to recognize shards.
    shard_counter = 0
    shard = thread_index * num_shards_per_batch
    output_filename = '%s-%0.5d-of-%0.5d' % (name, shard, num_shards)
    output_directory = FLAGS.output_directory + '/' + name
    if not tf.gfile.IsDirectory(output_directory):
        tf.gfile.MkDir(output_directory)
    output_file = os.path.join(output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)
    lost_frame = 0
    for i in range(len(filenames)):
        filename = filenames[i]
        label = labels[i]
        text = texts[i]
        print filename

        videoCount = 1
        video = cv2.VideoCapture(filename)
        while (video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT) > videoCount ):
            videoCount += 1

            if ranges[thread_index][0] <= frame_counter < ranges[thread_index][1]:
                # if counter < num_files_in_thread:
                # Getting height and width.
                height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
                width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
                ret, frame = video.read()
                # Do buffer works.
                if frame is None:
                    lost_frame += 1
                    continue
                image_buffer = _process_image(frame, coder)
                # example = _convert_to_example(filename, image_buffer, label,
                #                               text, height, width)
                # writer.write(example.SerializeToString())
                counter += 1
                shard_counter += 1
                # print counter
                # if not counter % 1000:
                #     print('%s [thread %d]: Processed %d images in thread batch.' %
                #           (datetime.now(), thread_index, counter, num_files_in_thread))
                #     sys.stdout.flush()
            frame_counter += 1
    # because opencv function have some issue, you need to notice when loss frame was too big. or change film camera.
    print('%s [thread %d]:loss frame: %d'%(datetime.now(), thread_index, lost_frame))
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, counter, output_file))
    sys.stdout.flush()
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_movie_files(name, filenames, labels, texts, num_shards):
    """"original function: _process_image_files
    Process movie and save list of images as TFRecord of Example protos.
    Args:
        filenames: list of strings; each string is a path to an image file
        texts: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
    """
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)
    count_frame_length = 0
    for filename in filenames:
        video = cv2.VideoCapture(filename)
        if not video.isOpened():
            print("could not open :", filename)
        frame_length = int(video.get(7))
        count_frame_length += frame_length
    # 30fps video file.
    spacing = np.linspace(0, count_frame_length, FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    coder = ImageCoder()
    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames, texts, labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)
    # _process_image_files_batch(coder, 4, ranges, name, filenames, texts,
    #                            labels, num_shards)
    coord.join(threads)
    print('%s: Finished writing all %d videos(%d frames) in data set.' % (
            datetime.now(), len(filenames), count_frame_length))
    sys.stdout.flush()


def _find_image_files(data_dir, labels_file):
    print('Determining list of input files and labels from %s.' % data_dir)
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(
        labels_file, 'r').readlines()]

    filenames = []
    labels = []
    texts = []

    label_index = 1

    for text in unique_labels:
        jpeg_file_path = '%s/%s/*' % (data_dir, text)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            print('Finished finding files in %d of %d classes.' % (
                label_index, len(labels)))
        label_index += 1

        # shuffled_index = range(len(filenames))
        # random.seed(12345)
        # random.shuffle(shuffled_index)

        # filenames = [filenames[i] for i in shuffled_index]
        # texts = [texts[i] for i in shuffled_index]
        # labels = [labels[i] for i in shuffled_index]
    return filenames, labels, texts


# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def _process_dataset(name, directory, num_shards, labels_file):
    filenames, labels, texts = _find_image_files(directory, labels_file)
    _process_movie_files(name, filenames, labels, texts, num_shards)


def main(unused_argv):
    _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards, FLAGS.labels_file)
    _process_dataset('validation', FLAGS.validation_directory, FLAGS.validation_shards, FLAGS.labels_file)


if __name__ == '__main__':
    tf.app.run()
