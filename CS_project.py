## Code for the Hand Tracking and Detection project for CS: Exercises - Keio University
# Author: Vinícius Ferreira Bandeira do Nascimento
# Student ID: 82323700
# Date: 18/10/2023

#--------------------------------------------------------------------------------------
#==========================================================
# =================== Importing modules ==================
#==========================================================

import cv2 as cv
import numpy as np
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import mediapipe as mp
from scipy.interpolate import RectBivariateSpline
import tkinter as tk # for the buttons
import time
import math
import torch
from torchvision import transforms

import monodepth2_master.networks as nw
from monodepth2_master.utils import download_model_if_doesnt_exist
import matplotlib.pyplot as plt
import PIL.Image as pil

#==========================================================
# ================= Setting up variables =================
#==========================================================

# rect_calib() and hist_hand()
rectangle_num_calib = 9 # Number of rectangles to be used on the calibration
rectangle_tl = None # Top-left coordinate for the calibration rectangle
rectangle_bl = None # Bottom-left coordinate for the calibration rectangle
rectangle_tr = None # Top-right coordinate for the calibration rectangle
rectangle_br = None # Bottom-right coordinate for the calibration rectangle

# hist_hand()
histogram_hand = None # Initial value for the hand histogram

# finger_tracking()
far_point_array = []

'''# main_hand_tracking()
prev_time = 0
cur_time = 0'''

#==========================================================
# ================= Setting up functions =================
#==========================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          Marker detection and depth recognition
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''def depth_est():
    model_name = "mono_640x192"

    download_model_if_doesnt_exist(model_name)
    encoder_path = os.path.join("models", model_name, "encoder.pth")
    depth_decoder_path = os.path.join("models", model_name, "depth.pth")

    encoder = nw.ResnetEncoder(18, False)
    depth_decoder = nw.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    encoder.eval()
    depth_decoder.eval()
    
    image_path = "monodepth2_master/assets/test_image.jpg" # CHANGE IT TO LIVE FEED WITH cv.VideoCapture(0)

    input_image = pil.open(image_path).convert('RGB')
    original_width, original_height = input_image.size

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

    input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
    
    with torch.no_grad():
        features = encoder(input_image_pytorch)
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    
    disp_resized = torch.nn.functional.interpolate(disp,
    (original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)

    plt.figure(figsize=(10, 10))
    plt.subplot(211)
    plt.imshow(input_image)
    plt.title("Input", fontsize=22)
    plt.axis('off')

    plt.subplot(212)
    plt.imshow(disp_resized_np, cmap='magma', vmax=vmax)
    plt.title("Disparity prediction", fontsize=22)
    plt.axis('off')
    plt.show()'''
    
'''def depth_est_2():

    #Live depth estimation with MiDaS Neural Network

    #model_type = "DPT_Hybrid"
    model_type = "MiDaS_small"
    dpt_model = torch.hub.load("intel-isl/MiDaS", model_type)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dpt_model.to(device)
    dpt_model.eval()
    
    dpt_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    #transform = dpt_transforms.dpt_transform # if using DPT_Hybrid model
    transform = dpt_transforms.small_transform # if using MiDaS_small
    
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        success, img = cap.read()
        start = time.time()
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        input_batch = transform(img).to(device)
        with torch.no_grad():
            prediction = dpt_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        depth_map = cv.normalize(depth_map, None, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
        end = time.time()
        fps = 1/(end - start)
        
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        depth_map = (depth_map*255).astype(np.uint8)
        depth_map = cv.applyColorMap(depth_map, cv.COLORMAP_MAGMA)
        
        img = cv.flip(img, 1)
        cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
        cv.imshow("Live feed", img)
        cv.imshow("Depth map", cv.flip(depth_map,1))
        
        if cv.waitKey(5) & 0xFF == 27:
            break
        
    cap.release()'''
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                     Color tracking
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def color_track(): 
    '''
    This function is implemented to change the colorspace of the image/video feed,
    extracting the colored object from the overall image.
    
    https://docs.opencv.org/3.4.2/df/d9d/tutorial_py_colorspaces.html 18/10/2023
    
    For now, the chosen color is blue.
    '''

    capture = cv.VideoCapture(0) # Captures the video feed
    while(1):
        _, live_feed = capture.read() # Take each frame
        
        HSV_fun = cv.cvtColor(live_feed, cv.COLOR_BGR2HSV) # Convert BGR to HSV
        
        # Define range of blue color in HSV
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
       
        mask = cv.inRange(HSV_fun, lower_blue, upper_blue) # Threshold the HSV image to get only blue colors    
        out_img = cv.bitwise_and(live_feed, live_feed, mask=mask) # Bitwise-AND mask and original image
        
        cv.imshow('Live feed', live_feed)
        cv.imshow('Video mask', mask)
        cv.imshow('Output', out_img)
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break

    cv.destroyAllWindows()
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#            Finger detection and tracking
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def rect_calib(live_feed):
    '''
    This function creates an area with rectangles in which
    the user place their hand inside and the program will
    recognize the skin color and create a histogram, in 
    order to create a proper hand masking.
    
    Variables:
        live_feed - Image/video that is used for the calibration
    '''
    rows, columns, _ = live_feed.shape
    global rectangle_num_calib, rectangle_tl, rectangle_bl, rectangle_tr, rectangle_br
    
    rectangle_tl = np.array(
        [6*rows/20, 6*rows/20, 6*rows/20, 9*rows/20, 9*rows/20, 9*rows/20, 12*rows/20, 12*rows/20, 12*rows/20], 
        dtype=np.uint32
        )

    rectangle_bl = np.array(
        [9*columns/20, 10*columns/20, 11*columns/20, 9*columns/20, 10*columns/20, 11*columns/20, 9*columns/20, 10*columns/20, 11*columns/20], 
        dtype=np.uint32
        )

    rectangle_tr = rectangle_tl + 10
    rectangle_br = rectangle_bl + 10

    for i in range(rectangle_num_calib):
        cv.rectangle(live_feed, 
                     (rectangle_bl[i], rectangle_tl[i]), 
                     (rectangle_br[i], rectangle_tr[i]), 
                     (0, 255, 0), 
                     1)
        
    return live_feed

def hist_hand(live_feed):
    '''
    Function to extract the pixels from rect_calib()
    
    Variables:
        live_feed - Image/video that is used for the calibration
    '''
    global rectangle_tl, rectangle_bl
    
    HSV_fun = cv.cvtColor(live_feed, cv.COLOR_BGR2HSV) # Convert BGR to HSV
    region_interest = np.zeros([90, 10, 3], dtype=HSV_fun.dtype) # region of interest matrix that saves the skin color details
    
    for i in range(rectangle_num_calib): # Makes the inside of the rectangles, the region of interest to check the pixels
        region_interest[i*10:i*10 + 10, 0:10] = HSV_fun[rectangle_tl[i]:rectangle_tl[i] + 10,
                                                        rectangle_bl[i]:rectangle_bl[i] + 10]
    
    histogram_hand = cv.calcHist([region_interest], # Creates a histogram using the region_interest variable
                                 [0,1],
                                 None,
                                 [180,256],
                                 [0,180,0,256]) 
    return_fun = cv.normalize(histogram_hand, # Function for the normalization of the matrix
                              histogram_hand,
                              0,
                              255,
                              cv.NORM_MINMAX)
    
    return return_fun
    
def mask_hand(live_feed, histogram):
    '''
    Function that creates a mask around the hand for contour
    creation on function finger_tracking()
    
    Variables:
        live_feed - Image/video that is used for the calibration
        histogram - Histogram for the hand color
    '''
    HSV_fun = cv.cvtColor(live_feed, cv.COLOR_BGR2HSV) # Convert BGR to HSV
    back_proj = cv.calcBackProject([HSV_fun], # Calculate the histogram model of a feature and then use it to find this feature in an image
                                   [0, 1],
                                   histogram,
                                   [0, 180, 0, 256],
                                   1)

    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (31, 31))
    cv.filter2D(back_proj, -1, disc, back_proj) # Create the filter for the background removal/hand masking

    _, threshold_smooth = cv.threshold(back_proj, 150, 255, cv.THRESH_BINARY) # Selects a threshold for the masking process, to make the process smoother
    threshold_smooth = cv.merge((threshold_smooth, threshold_smooth, threshold_smooth))

    return_fun = cv.bitwise_and(live_feed, threshold_smooth) # Contains the processed image hand with the mask applied

    return return_fun
    
def finger_tracking(live_feed, histogram):
    '''
    Function that finds the contour of the hand, calculates the 
    centroid and finds the farthest distance from it (tip of the finger)
    
    https://theailearner.com/tag/cv2-convexitydefects/
    https://docs.opencv.org/3.4/d0/d49/tutorial_moments.html
    
    Variables:
        live_feed - Image/video that is used for the calibration
        histogram - Histogram for the hand color
    '''
    hand_mask_image = mask_hand(live_feed, histogram) # Uses the function mask_hand() to create the hand mask, which is then post-processed to a smoother mask
    hand_mask_image = cv.erode(hand_mask_image, None, iterations=2)
    hand_mask_image = cv.dilate(hand_mask_image, None, iterations=2)
    
    _, threshold_contour = cv.threshold(cv.cvtColor(hand_mask_image, cv.COLOR_BGR2GRAY), 0, 255, 0) # Sets a value for the contour of the hand, following the mask of the hand done previously
    hand_contour, _ = cv.findContours(threshold_contour, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    contour_border = max(hand_contour, key=cv.contourArea)

    if cv.moments(contour_border)['m00'] != 0: # Gets the centroid coordinates of the image contour
        centroid_x = int(cv.moments(contour_border)['m10']/cv.moments(contour_border)['m00'])
        centroid_y = int(cv.moments(contour_border)['m01']/cv.moments(contour_border)['m00'])
    else:
        centroid_x = None
        centroid_y = None
    
    cv.circle(live_feed, [centroid_x, centroid_y], 5, [255, 0, 255], -1) # Draws a magenta point at the centroid location
    cv.circle(live_feed, [centroid_x, centroid_y], 5, [0, 0, 0], 1) 

    if contour_border is not None: # If a hand contour exists
        convex_hull = cv.convexHull(contour_border, returnPoints=False) # Calculate the convex hull of the hand contour
        convexity_defects = cv.convexityDefects(contour_border, convex_hull) # Find convexity defects

        if convexity_defects is not None and [centroid_x, centroid_y] is not None: # Identify the farthest point from the centroid
            defect_points = convexity_defects[:,0][:,0]
            x_coords = np.array(contour_border[defect_points][:, 0][:, 0], dtype=np.float64)
            y_coords = np.array(contour_border[defect_points][:, 0][:, 1], dtype=np.float64)
            
            squared_distances_x = cv.pow(cv.subtract(x_coords, centroid_x), 2)
            squared_distances_y = cv.pow(cv.subtract(y_coords, centroid_y), 2)
            distances = cv.sqrt(cv.add(squared_distances_x, squared_distances_y))
            farthest_point_idx = np.argmax(distances)
            if farthest_point_idx < len(defect_points):
                farthest_defect = defect_points[farthest_point_idx]
                farthest_point = tuple(contour_border[farthest_defect][0])
        
        print("Centroid: " + str([centroid_x, centroid_y]) + ", Farthest Point: " + str(farthest_point)) # Display the centroid and farthest point
        cv.circle(live_feed, farthest_point, 5, [0, 0, 255], -1) # Draw a red circle at the farthest point
        cv.circle(live_feed, farthest_point, 5, [0, 0, 0], 1) # Draw a black disk around the red one at the farthest point

        if len(far_point_array) < 25: # Maintain a list of the last 25 farthest points for tracking
            far_point_array.append(farthest_point)
        else:
            far_point_array.pop(0)
            far_point_array.append(farthest_point)

        if far_point_array is not None: # Draw circles to visualize the tracked points
            for point in far_point_array:
                cv.circle(live_feed, point, 5, [0, 0, 255], -1)
                cv.circle(live_feed, point, 5, [0, 0, 0], 1)
                
        cv.imshow('mask', hand_mask_image)
        # cv.imshow('contour', contour_border)


#==========================================================
# ====================== Main code =======================
#==========================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#            Finger detection and tracking
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main_finger_tracking():
    '''
    This is the main function for hand detection and finger tracking in a live video feed
    '''
    global histogram_hand # Global variable to store the hand histogram
    is_hand_hist_created = False # Flag to track if the hand histogram is created
    capture = cv.VideoCapture(0) # Initialize video capture from the default camera

    while capture.isOpened(): # Enter the main loop while the capture is open
        pressed_key = cv.waitKey(1) # Capture keyboard input

        _, live_feed = capture.read() # Read a frame from the video capture
        live_feed = cv.flip(live_feed, 1) # Flip the frame horizontally (mirror effect)

        if pressed_key & 0xFF == ord('s'): # If the 's' key is pressed
            is_hand_hist_created = True # Set the flag to indicate that the hand histogram is created
            histogram_hand = hist_hand(live_feed) # Create the hand histogram using the current frame

        if is_hand_hist_created: # If the hand histogram is already created
            finger_tracking(live_feed, histogram_hand) # Apply image processing operations using the hand histogram

        else: # If the hand histogram is not yet created
            live_feed = rect_calib(live_feed) # Draw a rectangle to help the user calibrate and create the hand histogram

        cv.imshow("Live Feed", live_feed) # Display the live video feed with any applied modifications

        if pressed_key == 27: # If the 'Esc' key is pressed
            break 

    cv.destroyAllWindows() # Close any open OpenCV windows
    capture.release() # Release the video capture

def run_main_fingers():
    main_finger_tracking()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                      Hand tracking
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main_hand_tracking():
    '''
    This function captures video from a camera, applies hand tracking using the
    MediaPipe library, and displays hand landmarks and connections in real-time
    '''
    cur_time = 0
    prev_time = 0
    
    capture = cv.VideoCapture(0)
    with mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while capture.isOpened():
            success, image = capture.read()

            image.flags.writeable = False # Stopping the write part of the image to decrease the resource use
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True # Draw the hand annotations on the image
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
#            '''
            # Landmark is the part of the hand, e.g. tip, palm, wrist, ...
            finger_count = 0 # Setting up the initial finger count
            hand_landmarks_pos = [] # Setting up the x,y positions of the hand marks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_index = results.multi_hand_landmarks.index(hand_landmarks) # To get the hand index for label checks
                    hand_label = results.multi_handedness[hand_index].classification[0].label
                    
                    for landmarks in hand_landmarks.landmark: # Filling the variable hand_landmarks_pos with x and y positions of each landmark
                        hand_landmarks_pos.append([landmarks.x, landmarks.y])
                    
                    # Counting fingers
                    if hand_label == "Left" and hand_landmarks_pos[4][0] > hand_landmarks_pos[3][0]:
                        finger_count = finger_count+1
                    elif hand_label == "Right" and hand_landmarks_pos[4][0] < hand_landmarks_pos[3][0]:
                        finger_count = finger_count+1

                    if hand_landmarks_pos[8][1] < hand_landmarks_pos[6][1]:       # Index finger
                        finger_count = finger_count+1
                    if hand_landmarks_pos[12][1] < hand_landmarks_pos[10][1]:     # Middle finger
                        finger_count = finger_count+1
                    if hand_landmarks_pos[16][1] < hand_landmarks_pos[14][1]:     # Ring finger
                        finger_count = finger_count+1
                    if hand_landmarks_pos[20][1] < hand_landmarks_pos[18][1]:     # Pinky
                        finger_count = finger_count+1
                    
                    mp.solutions.drawing_utils.draw_landmarks(image,
                                                              hand_landmarks,
                                                              mp.solutions.hands.HAND_CONNECTIONS,                                                              
                                                              mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2))
                                                              #mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                                                              #mp.solutions.drawing_styles.get_default_hand_connections_style())

            image = cv.flip(image, 1)

            #cv.putText(image, str(finger_count), (10, 80), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            # Frame per second
            cur_time = time.time()
            fps = 1/(cur_time - prev_time)
            prev_time = cur_time
            cv.putText(image, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
            
            cv.imshow('Hands', image) # Flip the image horizontally for a selfie-view display
#            '''
            # For the fingertips only below
            '''
            fingertip_positions = []  # List to store fingertip positions

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                        if landmark_id in [4, 8, 12, 16, 20]:  # Landmarks corresponding to fingertips
                            x, y = int(landmark.x*image.shape[1]), int(landmark.y*image.shape[0])
                            fingertip_positions.append((x, y))
            
            for x, y in fingertip_positions:
                cv.circle(image, (x, y), 5, (245,117,66), -1)  # Draw a circle at each fingertip
                cv.circle(image, (x, y), 5, (0, 0, 0), 1)      # Draw a black circumference at each fingertip

            cv.imshow('Fingertips', cv.flip(image, 1))  # Flip the image horizontally for a selfie-view display
            '''
            
            if cv.waitKey(5) & 0xFF == 27: # Press 'Esc' to exit the program
                break
    cv.destroyAllWindows()
    capture.release()

def run_main_hand():
    main_hand_tracking()
    
    
def hand_tracking_depth():
    '''
    Live depth estimation with MiDaS Neural Network
    '''
    # TODO:
    # - Heatmap with distances for the depth estimation
    # - Showing only the hand on the depth estimation feed
    # - Figuring out the hand rotation and how it affects the distance
    # - Getting more distance points
    
    # Defining parameters
    alpha = 0.2
    previous_depth = 0.0
    depth_scale = 1.0
    
    # Distance calculation - Need more data
    dist_program = [0.22281133589274, 0.13540361128668, 0.073747047642677] # Distance calculated from the program using dist_1 variable
    dist_real = [22.5, 45, 67.5] # Measured using a ruler from the webcam to the hand position
    A_coef, B_coef, C_coef = np.polyfit(dist_program, dist_real, 2) # Coefficients for the distance
       
    #model_type = "DPT_Hybrid"
    model_type = "MiDaS_small"
    dpt_model = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dpt_model.to(device)
    dpt_model.eval()
    
    dpt_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    #transform = dpt_transforms.dpt_transform # if using DPT_Hybrid model
    transform = dpt_transforms.small_transform # if using MiDaS_small
    
    cap = cv.VideoCapture(0)
    with mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
        while cap.isOpened():
            success, img = cap.read()
            start = time.time()
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            results = hands.process(img) # hand capture
            
            input_batch = transform(img).to(device)
            with torch.no_grad():
                prediction = dpt_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # ==================== Method 1 for depth estimation ====================
            
            # Recognizing hand landmarks and setting the distance (method 1)
            hand_landmarks_pos = [] # Setting up the x,y positions of the hand marks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_index = results.multi_hand_landmarks.index(hand_landmarks) # To get the hand index for check labels
                    hand_label = results.multi_handedness[hand_index].classification[0].label
                    for landmarks in hand_landmarks.landmark: # Filling the variable hand_landmarks_pos with x and y positions of each landmark
                        hand_landmarks_pos.append([landmarks.x, landmarks.y])
                    # Setting the distance on the palm
                    # mid_x, mid_y = hand_landmarks_pos[0] # Trying out to see if the wrist point is a good point to measure distance
                    x1, y1 = hand_landmarks_pos[5]
                    x2, y2 = hand_landmarks_pos[17]
                    mid_point = ((x1 + x2)/2, (y1 + y2)/2)
                    mid_x, mid_y = mid_point
                    
                    dist_1 = math.sqrt((y2 - y1)**2 + (x2 - x1)**2) # using 2 landmarks, does not take into account the palm rotation
                    angle_1 = math.atan2(y2-y1, x2-x1)
                    
                    dist_cm = A_coef*dist_1**2 + B_coef*dist_1 + C_coef 
                    # print(math.degrees(angle_1))
                    
                    mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, mp.solutions.drawing_utils.DrawingSpec(color=(52, 149, 235), thickness=2, circle_radius=2))
            
            # =========================== Method 1 - End ===========================
            
            # Predicting the depth and calculating the depth map
            depth_map = prediction.cpu().numpy()
            depth_map = cv.normalize(depth_map, None, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
            h, w = depth_map.shape
            
            # Frames per second counter
            end = time.time()
            fps = 1/(end - start)
            
            # ==================== Method 2 for depth estimation ====================
            
            # Distance estimation using the depth map (method 2)
            x_grid = np.arange(w)
            y_grid = np.arange(h)
            spline = RectBivariateSpline(y_grid, x_grid, depth_map)
            
            if results.multi_hand_landmarks:
                depth_mid_filt = spline(mid_y, mid_x) # set depth point to be calculated at the same point as method 1
                # EMA (exponential moving average) filter to reduce value fluctuation 
                depth_DPT = 1.0/(depth_mid_filt*depth_scale) # depth to disparity map
                filtered_depth = (alpha*depth_DPT + (1 - alpha)*previous_depth)[0][0]
                # filtered_depth = alpha*depth_mid_filt + (1 - alpha)*previous_depth
                depth_mid_filt = filtered_depth # distance calculated from camera to hand
            
            # =========================== Method 2 - End ===========================
            
            # Coloring the depth map
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            depth_map = (depth_map*255).astype(np.uint8)
            depth_map = cv.applyColorMap(depth_map, cv.COLORMAP_OCEAN)
            
            # Plotting the results
            img = cv.flip(img, 1)
            cv.putText(img, str(int(fps)), (10,65), cv.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
            if results.multi_hand_landmarks:
                cv.putText(img, "Depth (method 1): " + str(int(dist_cm)) + "cm", (10,90), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
                cv.putText(img, "Depth (method 2): " + str(int(depth_mid_filt*100)) + "cm", (10, 115), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
            cv.imshow("Live feed", img)
            cv.imshow("Depth map", cv.flip(depth_map,1))
            
            if cv.waitKey(5) & 0xFF == 27: # Exit program if 'Esc' is pressed
                break
        
    cap.release()
#==========================================================
# ===================== Initializer =====================
#==========================================================
'''def on_closing():

    #Destroy the screen opened with the button when the program closes with "Esc"

    root.destroy()

root = tk.Tk()
root.title("Select Program")
root.geometry("320x200")
tk.Frame(root).pack(fill="both", expand=True)

# Create buttons for selecting programs
main_finger_tracking_button = tk.Button(root, text="Finger tracking", command=run_main_fingers)
main_hand_tracking_button = tk.Button(root, text="Hand recognition", command=run_main_hand)

# Place the buttons in the center middle using grid
main_finger_tracking_button.pack(side="top", pady=15)
main_hand_tracking_button.pack(side="top", pady=50)

# Bind the window closure event to the on_closing function
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the tkinter main loop
root.mainloop()'''

if __name__ == '__main__':
    hand_tracking_depth()
    # color_track()
    # main_hand_tracking()
    # main_finger_tracking()