import os
import cv2
import sys
import numpy as np


def select_region(image):
	# Grab the x and y size and make a copy of the image
	ysize = image.shape[0]
	xsize = image.shape[1]
	color_select = np.copy(image)
	line_image = np.copy(image)
	# print xsize, ysize

	# Define color selection criteria
	# MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
	red_threshold = 256
	green_threshold = 256
	blue_threshold = 256

	rgb_threshold = [red_threshold, green_threshold, blue_threshold]

	# Define the vertices of a triangular mask.
	# Keep in mind the origin (x=0, y=0) is in the upper left
	# MODIFY THESE VALUES TO ISOLATE THE REGION 
	# WHERE THE LANE LINES ARE IN THE IMAGE
	left_bottom = [0, ysize-1]
	right_bottom = [xsize-1, ysize-1]
	apex = [xsize/2, 0]

	# Perform a linear fit (y=Ax+B) to each of the three sides of the triangle
	# np.polyfit returns the coefficients [A, B] of the fit
	fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
	fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
	fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

	# Mask pixels below the threshold
	color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
	                    (image[:,:,1] < rgb_threshold[1]) | \
	                    (image[:,:,2] < rgb_threshold[2])

	# Find the region inside the lines
	XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
	region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
	                    (YY > (XX*fit_right[0] + fit_right[1])) & \
	                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))
	                    
	# Mask color and region selection
	color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]
	# Color pixels red where both color and region selections met
	line_image[~color_thresholds & region_thresholds] = [255, 0, 0]
	return line_image

def detect_image(image):
	cv2.imshow("image", image)

	gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

	kernel_size = 5
	gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size),0)

	gray_select = np.copy(gray)

	# Define our color selection criteria
	# Note: if you run this code, you'll find these are not sensible values!!
	# But you'll get a chance to play with them soon in a quiz
	gray_threshold = 120

	# Identify pixels below the threshold
	thresholds = gray[:] < gray_threshold

	gray_select[thresholds] = 0
	# cv2.imshow("gray_select", gray_select)


	# Define a kernel size and apply Gaussian smoothing
	kernel_size = 3
	blur_gray = cv2.GaussianBlur(gray_select, (kernel_size, kernel_size),0)

	# Define our parameters for Canny and apply
	low_threshold = 50
	high_threshold = 150
	edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
	cv2.imshow("canny", edges)

	# Next we'll create a masked edges image using cv2.fillPoly()
	mask = np.zeros_like(edges)   
	ignore_mask_color = 255   

	# This time we are defining a four sided polygon to mask
	imshape = image.shape
	vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_edges = cv2.bitwise_and(edges, mask)

	# Define the Hough transform parameters
	# Make a blank the same size as our image to draw on
	rho = 1 # distance resolution in pixels of the Hough annotatedgrid
	theta = np.pi/180 # angular resolution in radians of the Hough grid
	threshold = 40     # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 10 #minimum number of pixels making up a line
	max_line_gap = 10    # maximum gap in pixels between connectable line segments
	line_image = np.copy(image)*0 # creating a blank to draw lines on

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
	                            min_line_length, max_line_gap)

	# Iterate over the output "lines" and draw lines on a blank image
	line_angle_threshold = [30, 60]
	if lines is not None:
		for line in lines:
		    for x1,y1,x2,y2 in line:
		    	dy = y2 - y1
		    	dx = x2 - x1
		    	if dy**2 + dx**2 < 80:
		    		continue
		    	if line_angle_threshold[0] < abs(np.arctan2(dy, dx)) * 180 / 2 / np.pi < line_angle_threshold[1]:
		    		cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
		    	else:
		    		pass
	line_image = select_region(line_image)
	# Create a "color" binary image to combine with line image
	color_edges = np.dstack((edges, edges, edges)) 

	cv2.imshow("line_image", line_image)

	# Draw the lines on the edge image
	lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
	cv2.imshow("lines_edges", lines_edges)
	key = cv2.waitKey(100) & 255
	if key in [ord('q'), ord('Q')]:
		return True
	elif key in [ord('s'), ord('S')]:
		cv2.imwrite("hard.png", image)
		return False
	return False

def detect_video(video_fn):
	print "detect video:", video_fn 
	cap = cv2.VideoCapture(video_fn)
	frame_count = 0
	while True:
		success, image = cap.read()
		print "\tframe_index=%d" % frame_count
		frame_count += 1
		if not success:
			break
		if frame_count < 100:
			continue
		# skip = detect_image(image)
		skip = setlect_region_test(image)
		if skip:
			return
	pass
def setlect_region_test(image):
	def mouse_callback(event,x,y,flags,param):
		if(event==cv2.EVENT_LBUTTONDOWN):
			print "pos:(%3d, %3d), hsv(%s), gray(%s), rgb(%s)" % (y, x, hsv[y, x, :], gray[y,x], image[y, x, :])

	kernel_size = 3
	image = cv2.GaussianBlur(image, (kernel_size, kernel_size),0)

	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

	hsv_write_threshold = [(0, 180), (0, 30), (150, 255)]

	white_thresholds = ((hsv[:,:,0] >= hsv_write_threshold[0][0]) & (hsv[:,:,0] <= hsv_write_threshold[0][1])) & \
			((hsv[:,:,1] >= hsv_write_threshold[1][0]) & (hsv[:,:,1] <= hsv_write_threshold[1][1])) & \
			((hsv[:,:,2] >= hsv_write_threshold[2][0]) & (hsv[:,:,2] <= hsv_write_threshold[2][1]))

	hsv_yellow_threshold = [(90, 100), (43, 255), (46, 255)]
	yellow_thresholds = ((hsv[:,:,0] >= hsv_yellow_threshold[0][0]) & (hsv[:,:,0] <= hsv_yellow_threshold[0][1])) & \
			((hsv[:,:,1] >= hsv_yellow_threshold[1][0]) & (hsv[:,:,1] <= hsv_yellow_threshold[1][1])) & \
			((hsv[:,:,2] >= hsv_yellow_threshold[2][0]) & (hsv[:,:,2] <= hsv_yellow_threshold[2][1]))
 
	rgb_threshold = [(0, 200), (0, 200), (0, 200)]
	rgb_thresholds = ((image[:,:,0] >= rgb_threshold[0][0]) & (image[:,:,0] <= rgb_threshold[0][1])) & \
			((image[:,:,1] >= rgb_threshold[1][0]) & (image[:,:,1] <= rgb_threshold[1][1])) & \
			((image[:,:,2] >= rgb_threshold[2][0]) & (image[:,:,2] <= rgb_threshold[2][1]))

	gray_threshold = (0, 255)
	gray_thresholds = (gray >= gray_threshold[0]) & (gray <= gray_threshold[1])

	# Define the vertices of a triangular mask.
	# Keep in mind the origin (x=0, y=0) is in the upper left
	# MODIFY THESE VALUES TO ISOLATE THE REGION 
	# WHERE THE LANE LINES ARE IN THE IMAGE
	h, w = image.shape[:2]
	left_bottom = [h-1, 0]
	right_bottom = [h-1, w-1]
	apex = [0, w/2]
	print left_bottom, right_bottom, apex

	# Perform a linear fit (y=Ax+B) to each of the three sides of the triangle
	# np.polyfit returns the coefficients [A, B] of the fit
	fit_left = np.polyfit((left_bottom[1], apex[1]), (left_bottom[0], apex[0]), 1)
	fit_right = np.polyfit((right_bottom[1], apex[1]), (right_bottom[0], apex[0]), 1)
	fit_bottom = np.polyfit((left_bottom[1], right_bottom[1]), (left_bottom[0], right_bottom[0]), 1)

	# Find the region inside the lines
	XX, YY = np.meshgrid(np.arange(0, w), np.arange(0, h))
	region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
	                    (YY > (XX*fit_right[0] + fit_right[1])) & \
	                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

	thresholds = (yellow_thresholds | white_thresholds) & gray_thresholds & region_thresholds
	print "\tthresholds.shape:", thresholds.shape

	region = np.zeros(image.shape, dtype=np.uint8)
	region[thresholds] = [255,255,255]
	mor_kernel = np.ones((3, 3), dtype=np.uint8) * 255
	mor_kernel[0, 0] = mor_kernel[0, 2] = mor_kernel[2, 0] = mor_kernel[2, 2] = 0
	region = cv2.dilate(region, mor_kernel)


	# Define our parameters for Canny and apply
	low_threshold = 50
	high_threshold = 150
	edges = cv2.Canny(region, low_threshold, high_threshold)
	cv2.imshow("canny", edges)

	# Define the Hough transform parameters
	# Make a blank the same size as our image to draw on
	rho = 1 # distance resolution in pixels of the Hough annotatedgrid
	theta = np.pi/180 # angular resolution in radians of the Hough grid
	threshold = 40     # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 20 #minimum number of pixels making up a line
	max_line_gap = 10    # maximum gap in pixels between connectable line segments
	line_image = np.zeros(image.shape, dtype=np.uint8) # creating a blank to draw lines on

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
	                            min_line_length, max_line_gap)

	# Iterate over the output "lines" and draw lines on a blank image
	if lines is not None:
		for line in lines:
		    for x1,y1,x2,y2 in line:
		    	dy = y2 - y1
		    	dx = x2 - x1
		    	if dy**2 + dx**2 < 80:
		    		continue
		    	if 10 < abs(np.arctan2(dy, dx)) * 180 / 2 / np.pi < 60:
		    		cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
		    	else:
		    		pass

	# cv2.imshow("hsv", hsv)
	cv2.imshow("origin", image)
	cv2.imshow("lines", line_image)
	# cv2.imshow("gray", gray)
	cv2.imshow("region", region)
	# cv2.setMouseCallback('hsv', mouse_callback)
	cv2.setMouseCallback('region', mouse_callback)
	cv2.setMouseCallback('origin', mouse_callback)
	key = cv2.waitKey(0) & 255
	if key in [ord('q'), ord('Q')]:
		return True
	pass

def main(args):
	image_root = "./test_images"
	for fn in os.listdir(image_root):
		# break
		# fn = "solidYellowCurve.jpg"
		fn = os.path.join(image_root, fn)
		# fn = "hard.png"
		image = cv2.imread(fn, 1)
		print image.shape, fn
		# image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
		# detect_image(image)
		setlect_region_test(image)
		# break
	video_root = "./test_videos"
	for index, fn in enumerate(os.listdir(video_root)):
		# break
		if index != 1:
			continue
		fn = os.path.join(video_root, fn)
		detect_video(fn)
	pass

if __name__ == '__main__':
	main(sys.argv)
	