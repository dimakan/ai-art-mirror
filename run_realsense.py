from __future__ import print_function
from __future__ import division

import sys
sys.path.insert(0, 'src')
import argparse
import numpy as np
import transform, vgg, pdb, os
import tensorflow as tf
import cv2
from datetime import datetime
# First import the library
import pyrealsense2 as rs

models_all=[{"ckpt":"models/ckpt_cubist_b20_e4_cw05/fns.ckpt", "style":"styles/cubist-landscape-justineivu-geanina.jpg"},
    {"ckpt":"models/ckpt_hokusai_b20_e4_cw15/fns.ckpt", "style":"styles/hokusai.jpg"},
    {"ckpt":"models/wave/wave.ckpt", "style":"styles/hokusai.jpg"},
    {"ckpt":"models/ckpt_kandinsky_b20_e4_cw05/fns.ckpt", "style":"styles/kandinsky2.jpg"},
    {"ckpt":"models/ckpt_liechtenstein_b20_e4_cw15/fns.ckpt", "style":"styles/liechtenstein.jpg"},
    {"ckpt":"models/ckpt_maps3_b5_e2_cw10_tv1_02/fns.ckpt", "style":"styles/maps3.jpg"},
    {"ckpt":"models/ckpt_wu_b20_e4_cw15/fns.ckpt", "style":"styles/wu4.jpg"},
    {"ckpt":"models/ckpt_elsalahi_b20_e4_cw05/fns.ckpt", "style":"styles/elsalahi2.jpg"},
    {"ckpt":"models/scream/scream.ckpt", "style":"styles/the_scream.jpg"},
    {"ckpt":"models/udnie/udnie.ckpt", "style":"styles/udnie.jpg"},
    {"ckpt":"models/ckpt_clouds_b5_e2_cw05_tv1_04/fns.ckpt", "style":"styles/clouds.jpg"}]


models=[{"ckpt":"models/ckpt_cubist_b20_e4_cw05/fns.ckpt", "style":"styles/cubist-landscape-justineivu-geanina.jpg"},
    {"ckpt":"models/ckpt_hokusai_b20_e4_cw15/fns.ckpt", "style":"styles/hokusai.jpg"},
    {"ckpt":"models/ckpt_kandinsky_b20_e4_cw05/fns.ckpt", "style":"styles/kandinsky2.jpg"},
    {"ckpt":"models/ckpt_liechtenstein_b20_e4_cw15/fns.ckpt", "style":"styles/liechtenstein.jpg"},
    {"ckpt":"models/ckpt_wu_b20_e4_cw15/fns.ckpt", "style":"styles/wu4.jpg"},
    {"ckpt":"models/ckpt_elsalahi_b20_e4_cw05/fns.ckpt", "style":"styles/elsalahi2.jpg"},
#    {"ckpt":"models/scream/scream.ckpt", "style":"styles/the_scream.jpg"},
#    {"ckpt":"models/udnie/udnie.ckpt", "style":"styles/udnie.jpg"},
    {"ckpt":"models/ckpt_maps3_b5_e2_cw10_tv1_02/fns.ckpt", "style":"styles/maps3.jpg"},
    {"ckpt": "models/ckpt_rain_princess_b20_e4_cw05/fns.ckpt", "style": "styles/rain_princess.jpg"},
    {"ckpt": "models/ckpt_la_muse_b20_e4_cw05/fns.ckpt", "style": "styles/la_muse.jpg"}]

#models=[
#    {"ckpt": "checkpoints/fns.ckpt", "style": "styles/la_muse.jpg"},
#    {"ckpt": "models/ckpt_la_muse_b20_e4_cw05/fns.ckpt", "style": "styles/la_muse.jpg"}]

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--device_id', type=int, help='camera device id (default 0)', required=False, default=0)
parser.add_argument('--width', type=int, help='width to resize camera feed to (default 320)', required=False, default=640)
parser.add_argument('--disp_width', type=int, help='width to display output (default 640)', required=False, default=1200)
parser.add_argument('--disp_source', type=int, help='whether to display content and style images next to output, default 1', required=False, default=1)
parser.add_argument('--horizontal', type=int, help='whether to concatenate horizontally (1) or vertically(0)', required=False, default=1)
parser.add_argument('--num_sec', type=int, help='number of seconds to hold current model before going to next (-1 to disable)', required=False, default=-1)


dim_low = (320, 240)
dim_med = (640, 480)
dim_hd = (1280, 720)
dim = dim_med
# perform the actual resizing of the image and show it


lower_white = np.array([220, 220, 220], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)




# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, 30)
config.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, 30)


# Start streaming
profile = pipeline.start(config)



# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)




def load_checkpoint(checkpoint, sess):
    saver = tf.train.Saver()
    try:
        saver.restore(sess, checkpoint)
        style = cv2.imread(checkpoint)
        return True
    except:
        print("checkpoint %s not loaded correctly" % checkpoint)
        return False


def get_camera_shape(cam):
    """ use a different syntax to get video size in OpenCV 2 and OpenCV 3 """
    cv_version_major, _, _ = cv2.__version__.split('.')
    if cv_version_major == '3':
        return cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    else:
        return cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
  
  
def make_triptych(disp_width, frame, style, output, horizontal=True):
    ch, cw, _ = frame.shape
    sh, sw, _ = style.shape
    oh, ow, _ = output.shape
    disp_height = int(disp_width * oh / ow)
    h = int(ch * disp_width * 0.5 / cw)
    w = int(cw * disp_height * 0.5 / ch)
    if horizontal:
        full_img = np.concatenate([
            cv2.resize(frame, (int(w), int(0.5*disp_height))),
            cv2.resize(style, (int(w), int(0.5*disp_height)))], axis=0)
        full_img = np.concatenate([full_img, cv2.resize(output, (disp_width, disp_height))], axis=1)
    else:
        full_img = np.concatenate([
            cv2.resize(frame, (int(0.5 * disp_width), h)),
            cv2.resize(style, (int(0.5 * disp_width), h))], axis=1)
        full_img = np.concatenate([full_img, cv2.resize(output, (disp_width, disp_width * oh // ow))], axis=0)
    return full_img


def main(device_id, width, disp_width, disp_source, horizontal, num_sec):
    t1 = datetime.now()
    idx_model = 0
    device_t='/gpu:0'
    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), tf.Session(config=soft_config) as sess:
        #cam = cv2.VideoCapture(device_id)
        #cam_width, cam_height = get_camera_shape(cam)
        cam_width, cam_height = dim
        cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, 1)
        width = width if width % 4 == 0 else width + 4 - (width % 4) # must be divisible by 4
        height = int(width * float(cam_height/cam_width))
        height = height if height % 4 == 0 else height + 4 - (height % 4) # must be divisible by 4
        img_shape = (height, width, 3)
        batch_shape = (1,) + img_shape
        print("batch shape", batch_shape)
        print("disp source is ", disp_source)
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
        preds = transform.net(img_placeholder)

        # load checkpoint
        load_checkpoint(models[idx_model]["ckpt"], sess)
        style = cv2.imread(models[idx_model]["style"])
        style_resized = cv2.resize(style, dim, interpolation=cv2.INTER_AREA)

        # enter cam loop
        while True:
            #ret, frame = cam.read()
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())


            frame = cv2.resize(color_image, (width, height))
            frame = cv2.flip(frame, 1)
            X = np.zeros(batch_shape, dtype=np.float32)
            X[0] = frame

            output = sess.run(preds, feed_dict={img_placeholder:X})
            output = output[:, :, :, [2,1,0]].reshape(img_shape)
            output = np.clip(output, 0, 255).astype(np.uint8)
            output = cv2.resize(output, (width, height))

            if disp_source:
                full_img = make_triptych(disp_width, frame, style, output, horizontal)
                cv2.imshow('frame', full_img)
            else:
                # Remove background - Set pixels further than clipping_distance to grey
                depth_image_3d = np.dstack(
                    (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
                replace_color = 255
                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), replace_color,
                                      color_image)
                replace_color = 0
                mask_black = np.where(((depth_image_3d < clipping_distance) & (depth_image_3d > 0)), replace_color,
                                      bg_removed)
                # "erase" the small white points in the resulting mask
                mask = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
                mask = cv2.inRange(mask, lower_white, upper_white)  # fixes dimensions.
                mask = cv2.flip(mask, 1)
                # get masked background, mask must be inverted
                mask_inv = cv2.bitwise_not(mask)
                bk_masked = cv2.bitwise_and(style_resized, style_resized, mask=mask)
                output_resized = cv2.resize(output, dim)
                fg_masked = cv2.bitwise_and(output_resized, output_resized, mask=mask_inv)
                output = cv2.bitwise_or(fg_masked, bk_masked)

                oh, ow, _ = output.shape
                output = cv2.resize(output, (disp_width, int(oh * disp_width / ow)))
                cv2.imshow('frame', output)

            key_ = cv2.waitKey(1)
            if key_ == 27:
                break
            elif key_ == ord('a'):
                idx_model = (idx_model + len(models) - 1) % len(models)
                print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
                load_checkpoint(models[idx_model]["ckpt"], sess)
                style = cv2.imread(models[idx_model]["style"])
            elif key_ == ord('s'):
                idx_model = (idx_model + 1) % len(models)
                print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
                load_checkpoint(models[idx_model]["ckpt"], sess)
                style = cv2.imread(models[idx_model]["style"])

            t2 = datetime.now()
            dt = t2-t1
            if num_sec>0 and dt.seconds > num_sec:
                t1 = datetime.now()
                idx_model = (idx_model + 1) % len(models)
                print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
                load_checkpoint(models[idx_model]["ckpt"], sess)
                style = cv2.imread(models[idx_model]["style"])
                style_resized = cv2.resize(style, dim, interpolation=cv2.INTER_AREA)

        # done
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    opts = parser.parse_args()
    main(opts.device_id, opts.width, opts.disp_width, opts.disp_source==1, opts.horizontal==1, opts.num_sec),

