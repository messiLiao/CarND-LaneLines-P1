from __future__ import print_function
import os
import cv2
import sys
import numpy as np

def save_sample_image_from_video(video_fn):
	cap = cv2.VideoCapture(video_fn)
	frame_count = 1
	d, fn = os.path.split(video_fn)
	f, _ = os.path.splitext(fn)	

	while True:
		if cap is None:
			break
		success, image = cap.read()
		if not success:
			break
		cv2.imshow(video_fn, image)
		key = cv2.waitKey(0) & 255
		if key in [ord('q'), ord('Q'), 27]:
			break
		elif key in [ord('s'), ord('S')]:
			cv2.imwrite('./test_images/%s_%03d.jpg' % (f, frame_count), image)
		frame_count += 1

	pass

def detect_video(video_fn, output_fn, debug=False):
	print("detect video: {0}".format(video_fn) )
	cap = cv2.VideoCapture(video_fn)
	if output_fn is not None:
		# get fps and frame_size
		fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)  
		size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),   
		        int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))  
		out = cv2.VideoWriter(output_fn, cv2.cv.CV_FOURCC('D', 'I', 'V', '3'), fps, size, 1) 
	else:
		out = None
	frame_count = 0
	while True:
		success, image = cap.read()
		print("\tframe_index={0}".format(frame_count))
		frame_count += 1
		if not success:
			break

		skip, image = detect_image(image, debug)
		if out is not None:
			out.write(image)
			pass
		if skip:
			break
	pass

def detect_image(image, debug=False):
	'''
	steps:
	1. using HSV color space to select yello and write region
	2. detect Hough Lines in the region picture
	3. remove the lines which not in the right area and the slope not in the right section.
	4. fit the lines with LMS method.
	5. draw the fitted line.
	'''
	def mouse_callback(event,x,y,flags,param):
		if(event==cv2.EVENT_LBUTTONDOWN):
			print("pos:(%3d, %3d), hsv(%s), gray(%s), rgb(%s)" % (y, x, hsv[y, x, :], gray[y,x], image[y, x, :]))

	# let image more smooth
	kernel_size = 5
	image = cv2.GaussianBlur(image, (kernel_size, kernel_size),0)

	# gray space 
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	# use hsv space to detect yello line
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

	# white color section in HSV space:[(h_min, h_mAX), (s_min, s_max), (v_min, v_max)]
	hsv_write_threshold = [(0, 180), (0, 30), (150, 255)]

	# select white region
	white_thresholds = ((hsv[:,:,0] >= hsv_write_threshold[0][0]) & (hsv[:,:,0] <= hsv_write_threshold[0][1])) & \
			((hsv[:,:,1] >= hsv_write_threshold[1][0]) & (hsv[:,:,1] <= hsv_write_threshold[1][1])) & \
			((hsv[:,:,2] >= hsv_write_threshold[2][0]) & (hsv[:,:,2] <= hsv_write_threshold[2][1]))

	# yello color section in HSV space
	hsv_yellow_threshold = [(90, 100), (43, 255), (46, 255)]
	yellow_thresholds = ((hsv[:,:,0] >= hsv_yellow_threshold[0][0]) & (hsv[:,:,0] <= hsv_yellow_threshold[0][1])) & \
			((hsv[:,:,1] >= hsv_yellow_threshold[1][0]) & (hsv[:,:,1] <= hsv_yellow_threshold[1][1])) & \
			((hsv[:,:,2] >= hsv_yellow_threshold[2][0]) & (hsv[:,:,2] <= hsv_yellow_threshold[2][1]))

	# Define the vertices of a triangular mask.
	# Keep in mind the origin (x=0, y=0) is in the upper left
	# MODIFY THESE VALUES TO ISOLATE THE REGION 
	# WHERE THE LANE LINES ARE IN THE IMAGE
	h, w = image.shape[:2]

	# use two triangle to devide lane-line area and none-lane-line area
	polygon_1 = np.array([[(w*0.5, h*0.5), (w*0.1, h), (w*0.99, h-1)]], dtype=np.int32)
	region_thresholds = np.zeros(image[:,:,0].shape, dtype=np.uint8)
	polygon_2 = np.array([[(w*0.5, h*0.75), (w*0.3, h-1), (w*0.8, h-1)]], dtype=np.int32)
	cv2.fillPoly(region_thresholds, polygon_1, 255)
	cv2.fillPoly(region_thresholds, polygon_2, 0)

	# add the yello region and write region.
	thresholds = yellow_thresholds | white_thresholds	
	if debug:
		print("\tthresholds.shape:{0}".format(thresholds.shape))

	region = np.zeros(image.shape, dtype=np.uint8)
	region[thresholds] = [255,255,255]
	region = region[:,:,0]

	# Define our parameters for Canny and apply
	low_threshold = 50
	high_threshold = 150
	edges = cv2.Canny(region, low_threshold, high_threshold)

	# Define the Hough transform parameters
	# Make a blank the same size as our image to draw on
	rho = 1 # distance resolution in pixels of the Hough annotatedgrid
	theta = np.pi/90 # angular resolution in radians of the Hough grid
	threshold = 30     # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 1 #minimum number of pixels making up a line

	# lane line width is nearly 20 pixels
	max_line_gap = 20    # maximum gap in pixels between connectable line segments

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
	                            min_line_length, max_line_gap)

	# left lane line angle is in [10, 60]
	# right lane line angle is in [-60, -10]
	left_lines = []
	right_lines = []
	line_image = np.zeros(image.shape[:2], dtype=np.uint8) # creating a blank to draw lines on
	all_line_image = np.zeros(image.shape[:2], dtype=np.uint8) # creating a blank to draw lines on
	if lines is not None:
		for line in lines:
			for x1,y1,x2,y2 in line:
				dy = y2 - y1
				dx = x2 - x1

				# cx, xy is the middle of the line
				# cx < w / 2 means the line is on the left, otherwise on the right.
				cx = (x1 + x2) / 2
				cy = (y1 + y2) / 2
				if (((cx < w/2) and (-60 < (np.arctan2(dy, dx) * 180.0 / 2 / np.pi) < -10)) or (cx >= w/2) and (10 < (np.arctan2(dy, dx) * 180 / 2 / np.pi) < 60))\
				and region_thresholds[y1, x1] and region_thresholds[y2, x2]:
					if dx*dy > 0:
						right_lines.append((x1, y1, x2, y2))
					else:
						left_lines.append((x1, y1, x2, y2))
					cv2.line(line_image,(x1,y1),(x2,y2),255,1)
				else:
					pass
				cv2.line(all_line_image,(x1,y1),(x2,y2),255,1)
	
	if debug:
		print("\tright lines count:%d, left lines count:%d" % (len(right_lines), len(left_lines)))
	# creating a blank to draw fit lines on.
	# there will be more wanted points after draw lines.
	fit_line_image = np.zeros(image.shape[:2], dtype=np.uint8) 
	for x1, y1, x2, y2 in right_lines:
		cv2.line(fit_line_image,(x1,y1),(x2,y2),255,1)

	points = np.where(fit_line_image>0)
	if len(points[0]) < 2:
		# not enough line to polyfit. create a "outside" line.
		points = [(-1, -1), (1,2)]
	# the fitted line formular is x = y*a + b
	right_fit_line= np.polyfit(points[0],points[1],1)

	fit_line_image = np.zeros(image.shape[:2], dtype=np.uint8) # creating a blank to draw lines on
	for x1, y1, x2, y2 in left_lines:
		cv2.line(fit_line_image,(x1,y1),(x2,y2), 255,2)
	points = np.where(fit_line_image>0)
	if len(points) < 2:
		points = [(-1, -1), (1,2)]
	left_fit_line= np.polyfit(points[0], points[1],1)
	if debug:
		print("\tright fit line: y = %f x + %f" % (right_fit_line[0], right_fit_line[1]))
		print("\tleft fit line : y = %f x + %f" % (left_fit_line[0], left_fit_line[1]))
	
	fit_line_image = np.zeros(image.shape, dtype=np.uint8) # creating a blank to draw lines on

	# the fomular is x = a*y + b. y is in [0.6h h].
	cv2.line(fit_line_image, (int(right_fit_line[0]*h*0.6 + right_fit_line[1]), int(h*0.6)), (int(right_fit_line[0]*h + right_fit_line[1]), h), (0, 0, 255), 10)
	cv2.line(fit_line_image, (int(left_fit_line[0]*h*0.6 + left_fit_line[1]), int(h*0.6)), (int(left_fit_line[0]*h + left_fit_line[1]), h), (0, 0, 255), 10)

	output_image = cv2.addWeighted(image, 0.8, fit_line_image, 1, 0)
	if debug:
		# cv2.imshow("output",output_image)
		# cv2.imshow("origin",output_image)
		cv2.imshow("hsv", hsv)
		cv2.imshow("thresholds", cv2.addWeighted(gray, 0.2, region_thresholds, 0.8, 0) )
		cv2.imshow("lines", line_image)
		cv2.imshow("region", region)
		cv2.imshow("gray", gray)
		cv2.setMouseCallback('region', mouse_callback)
		cv2.setMouseCallback('origin', mouse_callback)
		cv2.setMouseCallback('output', mouse_callback)
		cv2.setMouseCallback('thresholds', mouse_callback)
		key = cv2.waitKey(0) & 255
		if key in [ord('q'), ord('Q')]:
			return True, output_image
		elif key in [ord('s'), ord('S')]:
			cv2.imwrite('challenge_107.jpg', image)
			cv2.imwrite('challenge_107_hsv.jpg', hsv)
			cv2.imwrite('challenge_107_gray.jpg', gray)
			cv2.imwrite('challenge_107_region.jpg', region)
			cv2.imwrite('challenge_107_edges.jpg', edges)
			cv2.imwrite('challenge_107_lines.jpg', line_image)
			cv2.imwrite('challenge_107_all_lines.jpg', all_line_image)
			cv2.imwrite('challenge_107_output.jpg', output_image)
			cv2.imwrite('challenge_107_region_thresholds.jpg', region_thresholds)
	pass
	return False, output_image

def main(args, video_output=False, debug=True):
	image_root = "./test_images"
	for fn in os.listdir(image_root):
		fn = os.path.join(image_root, fn)		
		image = cv2.imread(fn, 1)
		print("detect image:{0}".format(fn))
		skip, _ = detect_image(image, debug)
		if skip:
			break
	video_root = "./test_videos"
	video_output_root = "./test_videos_output"
	for index, fn in enumerate(os.listdir(video_root)):
		fn = "challenge.mp4"		
		f, ext = os.path.splitext(fn)
		if video_output:
			output_fn = os.path.join(video_output_root, "%s_output%s" % (f, '.avi'))
		else:
			output_fn = None
		fn = os.path.join(video_root, fn)
		detect_video(fn, output_fn, debug)
		break
	pass

if __name__ == '__main__':
	debug = True
	video_write = False
	main(sys.argv, video_write, debug)
	# save_sample_image_from_video("./test_videos/challenge.mp4")
	