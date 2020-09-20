#####################################################
##               Read bag from file                ##
#####################################################


# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
from datetime import datetime
parser = argparse.ArgumentParser()
parser.add_argument("-i","--input", type=int, help="Path to the bag file")
args = parser.parse_args()
"""
# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream resolution, fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
parser.add_argument("-oc", "--outputcolor", type=str, help="Path to save color images")
parser.add_argument("-od", "--outputdepth", type=str, help="Path to save depth images")
# Parse the command line arguments to an object
args = parser.parse_args()
# Safety if no parameter have been given
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()
# Check if output path exists
if not os.path.exists(args.outputcolor):
    os.mkdir(args.outputcolor)
if not os.path.exists(args.outputdepth):
    os.mkdir(args.outputdepth)
"""
#colorizer = rs.colorizer()
#colorizer.set_option(rs.option.color_scheme, 0);
if args.input==1:
    W=640
    H=480
elif args.input==2:
    W=640
    H=480
try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()
	
    # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
    #rs.config.enable_device_from_file(config, args.input)
    # Configure the pipeline to stream the depth stream
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, W, H, rs.format.rgb8, 30)

    # Start streaming from file
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
    depth_sensor.set_option(rs.option.visual_preset, 0)
    depth_scale = depth_sensor.get_depth_scale()
    
    # Create an align object to align depth frames to color frames
    align = rs.align(rs.stream.color)
    
    
    
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 2)
    folder = datetime.now().strftime("%Y%m%d_%H%M%S")
    """
    threshold = rs.threshold_filter()
    threshold.set_option(rs.option.filter_min_distance, 0.1)
    threshold.set_option(rs.option.filter_max_distance, 4)
    """
    """
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)
"""
    if args.input==2:
        if not os.path.exists(folder):
            os.mkdir(folder)
        if not os.path.exists(folder+'/color'):
            os.mkdir(folder+'/color')
        #if not os.path.exists(folder+'/depth'):
        #    os.mkdir(folder+'/depth')
        if not os.path.exists(folder+'/depth_meter'):
            os.mkdir(folder+'/depth_meter')
    # Streaming loop
    SHOW=1
    if args.input==1:
        while True:
            frames = pipeline.wait_for_frames()
            frame_id = frames.get_frame_number()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            #depth_frame = decimation.process(depth_frame)
            #depth_frame = threshold.process(depth_frame)
            '''
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)

            depth_image = (depth_image.astype(float) * depth_scale).astype(np.uint8)
            '''
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            #depth_image = np.asanyarray(depth_frame.get_data())
            print(depth_scale)
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_meter = depth_image.astype(float) * 0.001
            color_frame = aligned_frames.get_color_frame()
            """
            print (depth_image.dtype)
            print (depth_image.shape)
            print (np.max(depth_image))
            print (np.min(depth_image))
            """
            color_image = np.asanyarray(color_frame.get_data())
            """
            print (color_image.dtype)
            print (color_image.shape)
            print (np.max(color_image))
            print (np.min(color_image))
            """
            if SHOW:
                #color_map=cv2.resize(color_map,(640,480))
				#depth_colormap=
                images = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR), depth_colormap))

                cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)

                cv2.imshow('Align Example', images)

                cv2.waitKey(1)

        #cv2.imshow('111', color_image)
        #cv2.waitKey(0.0)
        #cv2.imshow('222', depth_colormap)
        #np.save("depth_meter/%05d.npy"%(frame_id),depth_meter)

    elif args.input==2:
        while True:
            frames = pipeline.wait_for_frames()
            frame_id = frames.get_frame_number()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            #depth_frame = decimation.process(depth_frame)
            #depth_frame = threshold.process(depth_frame)
            '''
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            
            depth_image = (depth_image.astype(float) * depth_scale).astype(np.uint8)
            '''
            #depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            #depth_image = np.asanyarray(depth_frame.get_data())
            print(depth_scale)
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_meter = depth_image.astype(float) * 0.001
            color_frame = aligned_frames.get_color_frame()
            """
            print (depth_image.dtype)
            print (depth_image.shape)
            print (np.max(depth_image))
            print (np.min(depth_image))
            """
            color_image = np.asanyarray(color_frame.get_data())
            """
            print (color_image.dtype)
            print (color_image.shape)
            print (np.max(color_image))
            print (np.min(color_image))
            """

            # Visualize
            images = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Align Example', images)
            cv2.waitKey(1)

            # Save
            cv2.imwrite(folder+"/color/color_%05d.jpg"%(frame_id), cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
            #cv2.imwrite(folder+"/depth/depth_%05d.jpg"%(frame_id), depth_colormap)
            depth_meter.tofile(folder+"/depth_meter/%05d.bin"%(frame_id))
            #np.save("depth_meter/%05d.npy"%(frame_id),depth_meter)
            #break
        

finally:
    
    pass
