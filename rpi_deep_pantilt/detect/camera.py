# Python
import logging
import time
import picamera
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np

from threading import Thread

logging.basicConfig()
LOGLEVEL = logging.getLogger().getEffectiveLevel()

RESOLUTION = (224, 224)

logging.basicConfig()

# https://github.com/dtreskunov/rpi-sensorium/commit/40c6f3646931bf0735c5fe4579fa89947e96aed7


def run_pantilt_detect(center_x, center_y, labels, model_cls, rotation, resolution=RESOLUTION):
    '''
        Updates center_x and center_y coordinates with centroid of detected class's bounding box
        Overlay is only rendered around the tracked object
    '''
    model = model_cls()

    capture_manager = PiCameraStream(resolution=resolution, rotation=rotation)
    capture_manager.start()
    capture_manager.start_overlay()

    label_idxs = model.label_to_category_index(labels)
    start_time = time.time()
    fps_counter = 0
    while not capture_manager.stopped:
        if capture_manager.frame is not None:
            frame = capture_manager.read()
            prediction = model.predict(frame)

            if not len(prediction.get('detection_boxes')):
                continue

            if any(item in label_idxs for item in prediction.get('detection_classes')):

                tracked = (
                    (i, x) for i, x in
                    enumerate(prediction.get('detection_classes'))
                    if x in label_idxs
                )
                tracked_idxs, tracked_classes = zip(*tracked)

                track_target = prediction.get('detection_boxes')[
                    tracked_idxs[0]]
                # [ymin, xmin, ymax, xmax]
                y = int(
                    RESOLUTION[1] - ((np.take(track_target, [0, 2])).mean() * RESOLUTION[1]))
                center_y.value = y
                x = int(
                    RESOLUTION[0] - ((np.take(track_target, [1, 3])).mean() * RESOLUTION[0]))
                center_x.value = x

                display_name = model.category_index[tracked_classes[0]]['name']
                logging.info(
                    f'Tracking {display_name} center_x {x} center_y {y}')

            overlay = model.create_overlay(frame, prediction)
            capture_manager.overlay_buff = overlay
            if LOGLEVEL is logging.DEBUG and (time.time() - start_time) > 1:
                fps_counter += 1
                fps = fps_counter / (time.time() - start_time)
                logging.debug(f'FPS: {fps}')
                fps_counter = 0
                start_time = time.time()


def run_stationary_detect(labels, model_cls, model_path, rotation, draw_boxes, log_csv, video_path, save_frame_path, save_frame_freq):
    '''
        Overlay is rendered around all tracked objects
    '''
    model = model_cls()
    model.set_model_path(model_path)

    if log_csv:
        import csv
        csv_file = open('/home/ssl/Documents/detection_csv/' 
                        + ((video_path.split('.')[0].split('/')[-1] + '_') if video_path else '')
                        + ('overlay_' if draw_boxes else '')
                        + time.strftime("%d_%m_%Y-%H_%M_%S", time.localtime()) + '.csv', 'w')
        csv_writer = csv.writer(csv_file)
        
        csv_writer.writerow([item for sublist in [f'ymin{i};xmin{i};ymax{i};xmax{i}'.split(';') for i in range(10)] for item in sublist]
                            + [f'class{i}' for i in range(10)]
                            + [f'score{i}' for i in range(10)]
                            + ['detection_time_ms'])


    label_idxs = model.label_to_category_index(labels)
    start_time = time.time()
    fps_counter = 0

    if video_path:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if draw_boxes:
            out = cv2.VideoWriter(video_path.split('.')[0] + '_detect.' + video_path.split('.')[1],
                                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (224, 224))
        if save_frame_path:
            import os
            full_path = os.path.join(save_frame_path, video_path.split('.')[0].split('/')[-1])
            if not os.path.exists(full_path):
                os.mkdir(full_path)
            frames_path = os.path.join(full_path, 'frames')
            if not os.path.exists(frames_path):
                os.mkdir(frames_path)
            detection_path = os.path.join(full_path, 'detection')
            if not os.path.exists(detection_path):
                os.mkdir(detection_path)
            frame_count = -1

    else: 
        capture_manager = PiCameraStream(resolution=RESOLUTION, rotation=rotation, framerate=90)
        capture_manager.start()
        if draw_boxes:
            capture_manager.start_overlay()

    try:
        while (not video_path and not capture_manager.stopped) or (video_path and cap.isOpened()):
            if video_path or capture_manager.frame is not None:
                if video_path:
                    _, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_count = frame_count + 1
                else:
                    frame = capture_manager.read()

                prediction = model.predict(frame)

                if not len(prediction.get('detection_boxes')):
                    continue
                if any(item in label_idxs for item in prediction.get('detection_classes')):
                    
                    # Not all models will need to implement a filter_tracked() interface
                    # For example, FaceSSD only allows you to track 1 class (faces) and does not implement this method
                    try:
                        filtered_prediction = model.filter_tracked(
                        prediction, label_idxs)
                    except AttributeError:
                        filtered_prediction = prediction
                    if draw_boxes:
                        overlay = model.create_overlay(frame, filtered_prediction, video_path)
                        if not video_path:
                            capture_manager.overlay_buff = overlay
                    # for class_name in prediction.get('detection_classes'):
                    #     logging.info(
                    #         f'Tracking {class_name}')

                detection_time_ms = (time.time() - start_time) * 1000
                logging.info(f'det - time: {detection_time_ms}ms')

                if video_path:
                    if draw_boxes:
                        output_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        out.write(output_rgb)
                    if save_frame_path and not (frame_count % save_frame_freq):
                        count_string = f'{frame_count:05d}'
                        cv2.imwrite(os.path.join(frames_path, count_string + '.jpg'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        det_frame_file = open(os.path.join(detection_path, count_string + '.txt'), 'w')
                        for qtd_detection in range(10):
                            if filtered_prediction['detection_scores'][qtd_detection] < 0.5:
                                break
                            det_frame_file.write(str(filtered_prediction['detection_classes'][qtd_detection] - 1) + ' ')
                            # bb_line = " ".join([f'{abs(item):.6f}' for item in filtered_prediction['detection_boxes'][qtd_detection]])
                            box_det = filtered_prediction['detection_boxes'][qtd_detection]
                            bb_line = (f'{((box_det[1] + box_det[3]) / 2):.6f}' + ' ' + \
                                        f'{((box_det[0] + box_det[2]) / 2):.6f}' + ' ' + \
                                        f'{(box_det[3] - box_det[1]):.6f}' + ' ' + \
                                        f'{(box_det[2] - box_det[0]):.6f}')
                            det_frame_file.write(bb_line + '\n')
                        det_frame_file.close()

                if log_csv:
                    csv_writer.writerow(np.concatenate((np.array(filtered_prediction['detection_boxes']).ravel(),
                                        np.array(filtered_prediction['detection_classes']),
                                        np.array(filtered_prediction['detection_scores']),
                                        np.array([detection_time_ms]))))
                # logging.info(f'det - FPS: {1 / (time.time() - start_time)}')
                start_time = time.time()
    except KeyboardInterrupt:
        if video_path:
            cap.release()
            if draw_boxes:
                out.release()
        else:
            capture_manager.stop()


def _monkey_patch_picamera(overlay):
    original_send_buffer = picamera.mmalobj.MMALPortPool.send_buffer

    def silent_send_buffer(zelf, *args, **kwargs):
        try:
            original_send_buffer(zelf, *args, **kwargs)
        except picamera.exc.PiCameraMMALError as error:
            # Only silence MMAL_EAGAIN for our target instance.
            our_target = overlay.renderer.inputs[0].pool == zelf
            if not our_target or error.status != 14:
                raise error

    picamera.mmalobj.MMALPortPool.send_buffer = silent_send_buffer


class PiCameraStream(object):
    """
      Continuously capture video frames, and optionally render with an overlay

      Arguments
      resolution - tuple (x, y) size 
      framerate - int 
      vflip - reflect capture on x-axis
      hflip - reflect capture on y-axis

    """

    def __init__(self,
                 resolution=(320, 240),
                 framerate=24,
                 vflip=False,
                 hflip=False,
                 rotation=0,
                 max_workers=2
                 ):

        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.vflip = vflip
        self.camera.hflip = hflip
        self.camera.rotation = rotation
        self.overlay = None

        self.data_container = PiRGBArray(self.camera, size=resolution)

        self.stream = self.camera.capture_continuous(
            self.data_container, format="rgb", use_video_port=True
        )

        self.overlay_buff = None
        self.frame = None
        self.stopped = False
        logging.info('starting camera preview')
        self.camera.start_preview()
        # self.camera.start_recording('/home/ssl/Documents/detection_videos/' 
        #                 + time.strftime("%d_%m_%Y-%H_%M_%S", time.localtime()) + '.h264')
        # time.sleep(1)

    def render_overlay(self):
        while True:
            if self.overlay and self.overlay_buff:
                self.overlay.update(self.overlay_buff)
            elif not self.overlay and self.overlay_buff:
                self.overlay = self.camera.add_overlay(
                    self.overlay_buff, layer=3, size=self.camera.resolution)
                _monkey_patch_picamera(self.overlay)

    def start_overlay(self):
        Thread(target=self.render_overlay, args=()).start()
        return self

    def start(self):
        '''Begin handling frame stream in a separate thread'''
        Thread(target=self.flush, args=()).start()
        return self

    def flush(self):
        # looping until self.stopped flag is flipped
        # for now, grab the first frame in buffer, then empty buffer
        for f in self.stream:
            self.frame = f.array
            self.data_container.truncate(0)

            if self.stopped:
                self.stream.close()
                self.data_container.close()
                self.camera.close()
                return

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        # self.camera.stop_recording()
