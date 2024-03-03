from realsense_camera import *
import cv2
from mask_rcnn import *

# Load Realsense camera
realsense_camera = RealsenseCamera()
mask_rcnn = MaskRCNN()

while True:
	# Get frame in real time from Realsense camera
	ret, bgr_frame, depth_frame = realsense_camera.get_frame_stream()

	# Get object mask
	boxes, classes, contours, centers = mask_rcnn.detect_objects_mask(bgr_frame)

	# Draw Object Mask
	mask_rcnn.draw_object_mask(bgr_frame)

	# Show depth info of the objects
	mask_rcnn.draw_object_info(bgr_frame, depth_frame)

	cv2.imshow("Depth Frame", depth_frame)
	cv2.imshow("Bgr Frame", bgr_frame)

	key = cv2.waitKey(1)
	if key == 27:
		break

cv2.destroyAllWindows()
