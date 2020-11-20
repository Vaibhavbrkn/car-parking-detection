from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import numpy as np
import cv2
import os
from PIL import Image
from tensorflow.python.saved_model import tag_constants
from core.yolov4 import filter_boxes
import core.utils as utils
from absl.flags import FLAGS
from absl import app, flags, logging
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')

flags.DEFINE_string('video', './data/video/video.mp4',
                    'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID',
                    'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')


def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes[0]):

        if class_ids[0][i] in [2, 7, 5]:
            car_boxes.append(box)

    return np.array(car_boxes)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    print(boxes1.shape)
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * \
        (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * \
        (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    parked_car_boxes = None

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(
            FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None
    counter = 0
    free_space_frames = 0
    count = 0

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        if counter < 40:

            return_value, frame2 = vid.read()
            d = cv2.absdiff(frame, frame2)
            grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(grey, (1, 1), 0)
            ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

            # perform these morphological transformations to erode the car which is moving so that it is not detected by MASKRCNN. Take the eorsion levels to be high.
            dilated = cv2.dilate(th, np.ones((30, 30), np.uint8), iterations=1)
            eroded = cv2.erode(dilated, np.ones(
                (30, 30), np.uint8), iterations=1)

            # fill the contours for even a better morphing of the vehicle
            c, h = cv2.findContours(
                eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            frame2 = cv2.drawContours(frame2, c, -1, (0, 0, 0), cv2.FILLED)

            counter = counter + 1
            continue

            # Converting the image from BGR color used by OpenCV to RGB color.
            if counter == 40:
                rgb_image = frame2[:, :, ::-1]
                counter += 1
            else:
                rgb_image = frame[:, :, ::-1]

            frame = rgb_image

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(
                output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(),
                     valid_detections.numpy()]

        if parked_car_boxes is None:
            print("going in to mark vehicles. Frame number:  ", counter)
            # video_capture = cv2.VideoCapture(VIDEO_SOURCE)
        # This is the first frame of video - assume all the cars detected are in parking spaces.
        # Save the location of each car as a parking space box and go to the next frame of video.
            parked_car_boxes = get_car_boxes(boxes.numpy(), classes.numpy())

        else:
            # Get where cars are currently located in the frame
            car_boxes = get_car_boxes(boxes.numpy(), classes.numpy())
            overlaps = compute_overlaps(
                parked_car_boxes, car_boxes)
            free_space = False

            for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):

                max_IoU_overlap = np.max(overlap_areas)


            # Get the top-left and bottom-right coordinates of the parking area
                y1, x1, y2, x2 = parking_area

                y1, y2 = y1 * frame.shape[0], y2 * frame.shape[0]
                x1, x2 = x1 * frame.shape[1], x2 * frame.shape[1]
                
                if (int(x2-x1) > int(frame.shape[1]/2) or int(y2-y1) > int(frame.shape[0]/2)):
                    continue
            # Check if the parking space is occupied by seeing if any car overlaps
            # it by more than 0.15 using IoU
                if max_IoU_overlap < 0.05:
                    # Parking space not occupied! Draw a green box around it
                    cv2.rectangle(frame, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (0, 255, 0), 3)
                # Flag that we have seen at least one open space
                    free_space = True
                else:
                    # Parking space is still occupied - draw a red box around it
                    cv2.rectangle(frame, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (0, 0, 255), 1)

                cv2.imshow('image', frame)

            # Write the IoU measurement inside the box

                # font = cv2.FONT_HERSHEY_DUPLEX
                # cv2.putText(int(frame), str(max_IoU_overlap),
                #             (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))

            if free_space:
                free_space_frames += 1
            else:
                # If no spots are free, reset the count.
                free_space_frames = 0
            if free_space_frames > 140:
                # Write SPACE AVAILABLE!! at the top of the screen
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, f"SPACE AVAILABLE!", (10, 150),
                            font, 3.0, (0, 255, 0), 2, cv2.FILLED)

        name = str(count) + ".jpg"
        name = os.path.join('./detect', name)
        cv2.imwrite(name, frame)
        count += 1

        if FLAGS.output:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass