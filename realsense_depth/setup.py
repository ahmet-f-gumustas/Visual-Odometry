import pyrealsense2 as rs
import numpy as np
import cv2

# Create a Realsense pipeline object
pipline = rs.pipeline()

# Create a Realsense Configuration object and set the resolution and frame rate
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline and get the color and depth sensor objects
pipline.start(config)

color_sensor = pipline.get_active_profile().get_device().query_sensors()[1]

# Set the exposure and gain values for the color sensor
color_sensor.set_option(rs.option.exposure, 1000)
color_sensor.set_option(rs.option.gain, 32)

# Create aligment object
align = rs.align(rs.stream.color)

# Loop through frames and get color and depth data
while True:
    # wait for Realsense data streams
    frames = pipline.wait_for_frames()
    aligned_frames = align.process(frames)

    # Get color and depth frames
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    # If there are no valid frames, continue waiting for the next one
    if not color_frame or not depth_frame:
        continue

    # Convert color and depth data to NumPy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Apply color map to depth image for visualization
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)

    # Display RGB and depth images
    cv2.imshow('Color Frame', color_image)
    cv2.imshow('Depth Frame', depth_image)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break


# Release resources and stop the pipeline
cv2.destroyAllWindows()
pipline.stop()