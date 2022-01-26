from cmath import nan
import os
from turtle import left
from typing_extensions import final
import warnings
from datetime import datetime
from os import listdir
from Line import Line

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

warnings.simplefilter('ignore', np.RankWarning)

def resize_frame(frame, output_shape):
    '''
    Resize the frame to the output_shape, filling the empty spaces with black pixels
    Parameters
    ----------
    frame: Frame that will be resized
    output_shape: Output video shape
    '''
    output_shape = (output_shape[0], output_shape[1], 3)
    output = np.zeros(output_shape, np.uint8)
    factor = min(output_shape[0]/frame.shape[0] , output_shape[1]/frame.shape[1])
    img_res = cv.resize(frame, dsize=(0,0), fx=factor, fy=factor)
    x0 = int((output_shape[1] - img_res.shape[1])/2)
    x1 = x0 + img_res.shape[1]
    y0 = int((output_shape[0] - img_res.shape[0])/2)
    y1 = y0 + img_res.shape[0]
    output[y0:y1, x0:x1, :] = img_res[:]

    return output

def generateImages(img, x_seg, colorRange, avgFinalLines, prevRoadLimits):
    '''
    Generate the Blur, Canny and segmented images
    Parameters
    ----------
    img: Source frame
    colorRange: Color range that will be used in the segmentation process
    '''
    img_blur = cv.blur(img, (31, 31))

    img_hsv = cv.cvtColor(img_blur, cv.COLOR_RGB2HSV)
    
    for (lower, upper) in colorRange:

        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mascara = cv.inRange(img_hsv, lower, upper)

    img_segmented = img.copy()
    img_segmented[mascara == 0] = 0
    img_segmented[mascara == 255] = 255


    # plt.imshow(img_segmented), plt.xticks([]), plt.yticks([]), plt.show()
    # cv.imwrite("3-segmentacao_verde.png", img_segmented)
    # img_and = cv.bitwise_and(img, img_segmented)
    # plt.imshow(img_and), plt.show()
    # cv.imwrite("4-segmented_parts.png", img_and)
    img_segmented, avgXSeg, detailedSeg, roadLimits = improvedSeg(img_segmented, x_seg, avgFinalLines, prevRoadLimits)
    img_canny = cv.Canny(img_segmented, 25, 35)
    
    return img_canny, img_segmented, img_blur, avgXSeg, detailedSeg, roadLimits

def heightedAverageLine(lines):
    '''
    Return the average line heighting the size of the lines. The bigger lines have a larger impact on the final average line
    Parameters
    ----------
    lines: Array Like
            Array containing the lines that will be weighted. Each one of the lines must in the format: [line, parameters, ang_error, y_error, abs(dist_error), score, lineSize]
    '''
    result = []
    for line in lines:
        n = int(line.size/10)
        for i in range(n):
            result.append(line.coords)
    avgLine = np.average(np.array(result), axis=0).astype(int)
    # avgLine[0] = avgLine[0].astype(int)
    return avgLine

def get_coordinates(height, lineParameters):
    '''
    Convert the line parameters to x,y coordenates. The output lines always will start from the image base and end at the center of the image
    Parameters
    ----------
    height: Height of the image
    lineParameters: Array Like
                    Parameters from the line equation of the line that will be converted. 
                    Line Equation: y = ax + b
                    Parameters: [a, b]
    '''
    # Line equation (y = ax + b)
    a, b = lineParameters
    y2 = int(height * 0.5)  # The line will end at the center of the image (0.5 * Height)
    if np.isnan(a):
        return np.array([int(b),height,int(b),y2])
    x = [abs(int((height - b) / a)), abs(int((y2 - b) / a))]
    if b > 0:
        x2 = max(x)
        x1 = min(x)
    else:
        x1 = max(x)
        x2 = min(x)
    return np.array([x1, height, x2, y2])

def draw_lines(img, lines, color=(0,255,255), average_lines=False):
    '''
    Draw the given lines to an image, coloring the space between the lines if the ´average_lines´ are True.
    Parameters
    ----------
    img : Image where the lines will be drawn.
    lines: array like
            array that contain the lines that will be drawn
    color : Color the will be used to drawn the lines
    average_lines : Boolean
               Define if the lines that will be drawn are average lines or not. If True, the space between the lines will be colored.
    '''
    imgWithLines = img.copy()
    
    if average_lines:
        lines = [lines]
    # if len(lines[0]) < 2:
    #     return imgLines
    if lines is None:
        return imgWithLines
    for line_pair in lines:
        if line_pair is not None:
            if average_lines and len(line_pair[0]) > 1 and len(line_pair[1]) > 1:
                points = np.array([(line_pair[0,0], line_pair[0,1]), (line_pair[0,2], line_pair[0,3]), 
                                (line_pair[1,2], line_pair[1,3]), (line_pair[1,0], line_pair[1,1])])
                try:
                    cv.fillPoly(imgWithLines, np.int32([points]), (255,0,0))
                except OverflowError:
                    return imgWithLines
            for line in line_pair:
                if len(line) == 4:
                    x1, y1, x2, y2 = line.reshape(4)
                    cv.line(imgWithLines, (x1, y1), (x2, y2), color, 10)
    return imgWithLines

def extrapolateLine(modelLine, avgPars, imgWidth, approx):
    (y1,y2) = (modelLine.projCoords[1], modelLine.projCoords[3])
    if modelLine.lineType == "Left":
        x1 = imgWidth + approx
        ang = avgPars[1][0]
    else:
        x1 = 0 - approx
        ang = avgPars[0][0]
    x2 = int(x1 + (y2-y1)/np.tan(ang))

    return (x1,y1,x2,y2)

def calc_average_lines(img, lines, avg_lines, avgXCenter, avgRoadSize, avgParameters, badLineCout, approx=40):
    '''
    Calculate the average lines from a array of lines.
    Parameters
    ----------
    img: frame where the lines will be drawn
    lines: Array Like
            Lines that will be used to calculate the average lines. Each line must be in the format [x1, y1, x2, y2]
    avgXCenter: Coordenate of the average x center of the road
    avgRoadSize: Average size of the road
    avgParameters: Array Like
                    Average line parameters. [[aLeft, bLeft], [aRight, bRight]]
    aprox: Value that will be used to approximate the coordenates of the final lines to the road center.
    '''
    (height, width, _)  = img.shape
        
    leftLines = []
    rightLines = []

    AvgLeftLine = []
    AvgRightLine = []


    if avgRoadSize == 0:
        avgRoadSize = int(0.8 * width)
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 != x2:
            a = (y2-y1)/(x2-x1)
            if abs(a) < 0.4:
                continue
        newLine = Line(line, avgXCenter,avgParameters, avgRoadSize, img)
        if newLine.lineType == "Left":
            leftLines.append(newLine)
        else:
            rightLines.append(newLine)
  
    n_detected_lines = len(leftLines) + len(rightLines)
    imgConsLines = img.copy()
    final_lines = []
    if len(leftLines) > 0:
        badLineCout[0] = 0
        sortedLeftLines = sorted(leftLines, key = lambda line:line.score, reverse=True)
        AvgLeftLine = heightedAverageLine(sortedLeftLines[:2])
        FinalLeftLine = Line(AvgLeftLine, avgXCenter, avgParameters, avgRoadSize, img)
        consideredLeftLines = sortedLeftLines[:2]
        for line in consideredLeftLines: 
            imgConsLines = line.draw(imgConsLines)
        final_lines.append(FinalLeftLine)
    

    if len(rightLines) > 0:
        badLineCout[1] = 0
        sortedRightLines = sorted(rightLines, key = lambda line:line.score, reverse=True)
        AvgRightLine = heightedAverageLine(sortedRightLines[:2])
        FinalRightLine = Line(AvgRightLine, avgXCenter, avgParameters, avgRoadSize, img)
        consideredRightLines = sortedRightLines[:2]
        for line in consideredRightLines: 
            imgConsLines = line.draw(imgConsLines)
        final_lines.append(FinalRightLine)

        if len(leftLines) == 0:
            badLineCout[0] += 1
            if badLineCout[0] > 2:
                artLeftLine = extrapolateLine(FinalRightLine, avgParameters, width, approx)
                FinalLeftLine = Line(artLeftLine, avgXCenter, avgParameters, avgRoadSize, img)
                final_lines.insert(0, FinalLeftLine)
    elif len(leftLines) > 0:
        badLineCout[1] += 1
        if badLineCout[1] > 2:
            maxRightLine = extrapolateLine(FinalLeftLine, avgParameters, width, approx)
            FinalRightLine = Line(maxRightLine, avgXCenter, avgParameters, avgRoadSize, img)
            final_lines.append(FinalRightLine)
    
    final_lines = np.array(final_lines)
    

    return final_lines, imgConsLines, n_detected_lines, badLineCout


def calcAvgRoadSize(height, left_lines, right_lines, n_lines=20):
    '''
    Calculate the average road size
    Parameters
    ----------
    height: Frame height
    left_lines: Array Like
                Array with all the last 100 left lines detected
    right_lines: Array Like
                Array with all the last 100 right lines detected
    n_lines: Number of lines to be considered when calculating the average road size
    '''
    dists = []
    arr_left_lines = np.array(left_lines)[:, -n_lines:]
    arr_right_lines = np.array(right_lines)[:, -n_lines:]

    for i in range(arr_left_lines.shape[1]):
        x1, y1, x2, y2 = arr_left_lines[:,i]
        x_left = x2 - ((height - y2)/(y1 - y2))*(x2 - x1)
        
        x1, y1, x2, y2 = arr_right_lines[:,i]
        x_right = x2 - ((height - y2)/(y1 - y2))*(x2 - x1)
        
        d = x_right - x_left
        dists.append(d)
    
    if len(dists) > 0:
        return np.average(np.array(dists))
    else:
        return 0

def calcAvgParameters(left_pars_lines, right_pars_lines, n_lines=30):
    '''
    Calculate the average parameters of the left and the right lines
    Parameters
    ----------
    left_lines: Array Like
                Array with all the last 100 left lines detected
    right_lines: Array Like
                Array with all the last 100 right lines detected
    n_lines: Number of lines to be considered when calculating the average parameters
    '''
    arr_pars_left_lines = np.array(left_pars_lines)[-n_lines:]
    arr_pars_right_lines = np.array(right_pars_lines)[-n_lines:]

    avgParsLeft = np.average(arr_pars_left_lines, axis=0)
    avgParsRigth = np.average(arr_pars_right_lines, axis=0)
    return [avgParsLeft, avgParsRigth]

def improvedSeg(img_seg, x_seg, avgLines, roadLimits, minWidth = 8):
    '''
    Improve the segmentation process, only considering the main road.
    Parameters
    ----------
    img_seg: Color segmented  image
    minWidth: Minimum width that will be considered when detecting a road limit
    '''
    imgNewSeg = img_seg.copy()
    height = img_seg.shape[0]
    leftRatios = []
    rightRatios = []
    # roadLimits = np.zeros((img_seg.shape[0], 2))
    left_limit = int(0.4*img_seg.shape[1])
    right_limit = int(0.6*img_seg.shape[1])
    img_seg_bin = cv.cvtColor(img_seg, cv.COLOR_BGR2GRAY)
    newImgSeg = np.copy(img_seg)
    detailedSeg = img_seg.copy()
    leftDif, rightDif = (0,0)
    limitLostLeft = False
    if x_seg == 0:
        x_seg = int(img_seg.shape[1]/2)

    x_left, x_right = (0, img_seg.shape[1])
    new_x_left, new_x_right = (0, img_seg.shape[1])
    best_columns = np.argpartition(np.average(img_seg_bin[:,left_limit:right_limit], axis=0), 4)[:4]
    minDif = img_seg.shape[1]

    for column in best_columns:
        dif = abs(left_limit + column - img_seg.shape[1]/2)
        if dif < minDif:
            new_x_seg = column + left_limit
            minDif = dif

    x_center = new_x_seg
    for y in range(img_seg_bin.shape[0]-1, 0, -1):
        detect = [False, False]
        line = img_seg_bin[y]
        point_count = 0
        if np.mean(line[int(new_x_seg-(minWidth)):new_x_seg+1]) < 255 and np.mean(line[new_x_seg:new_x_seg+(minWidth)]) < 255:
            for x in range(new_x_seg, img_seg.shape[1]):
                if line[x] == 255:
                    if point_count == minWidth:
                        new_x_right = x - minWidth
                        detect[1] = True
                        break
                    else:
                        point_count += 1
                else:
                    point_count = 0
            for x in range(new_x_seg, 0, -1):
                if line[x] == 255:
                    if point_count == minWidth:
                        new_x_left = x + minWidth
                        detect[0] = True
                        break
                    else:
                        point_count += 1
                else:
                    point_count = 0
        else:
            new_x_right = img_seg.shape[1]
            new_x_left = 0
            for i in range(1, img_seg.shape[1] - new_x_seg):
                if new_x_left == 0 and new_x_right == img_seg.shape[1]:
                    if (new_x_seg + i) <= img_seg.shape[1] and (np.mean(line[new_x_seg + i:new_x_seg + i+minWidth]) == 0):
                        # if np.mean(line[new_x_center + i:new_x_center + i+minWidth]) == 0:
                        # if point_count == minWidth:
                        new_x_left = new_x_seg + i
                        detect[0] = True
                    elif (new_x_seg - i) >= 0 and (np.mean(line[new_x_seg - i-minWidth:new_x_seg - i]) == 0):
                        if np.mean(line[new_x_seg - i-minWidth:new_x_seg - i]) == 0:
                            new_x_right = new_x_seg - i
                            detect[1] = True
                else:
                    if new_x_left > 0 and np.mean(line[new_x_seg + i:new_x_seg + i+minWidth]) == 255:
                        if np.mean(line[new_x_seg + i:new_x_seg + i+minWidth]) == 255:
                            new_x_right = new_x_seg + i
                            detect[1] = True
                            break
                    elif new_x_right < img_seg.shape[1] and np.mean(line[new_x_seg - i-minWidth:new_x_seg - i]) == 255:
                        if np.mean(line[new_x_seg - i-minWidth:new_x_seg - i]) == 255:
                            new_x_left = new_x_seg - i
                            detect[0] = True
                            break
            if not False in detect:
                new_x_seg = int(x_left + (x_right - x_left)/2)
            

        if (x_right < img_seg.shape[1] and new_x_left > x_right) or (x_left > 0 and new_x_right < x_left) or (new_x_left == 0 and new_x_right == img_seg.shape[1]):
            newImgSeg[y, :] = [255,255,255]
            continue
        
        imgNewSeg = cv.circle(imgNewSeg, (new_x_left, y),0,color=(255,0,0), thickness=5)
        imgNewSeg = cv.circle(imgNewSeg, (new_x_right, y),0,color=(255,0,0), thickness=5)

        cv.imshow('new Seg', imgNewSeg)
        if y != img_seg_bin.shape[0]-1:
            leftDif = new_x_left - x_left
            rightDif = new_x_right - x_right
            leftRatios.append(leftDif)
            rightRatios.append(rightDif)

        x_left = new_x_left
        x_right = new_x_right

        newImgSeg[y, :x_left] = [255,255,255]
        newImgSeg[y, x_right:] = [255,255,255]
        newImgSeg[y, x_left:x_right] = [0,0,0]

        detailedSeg = cv.circle(detailedSeg, (new_x_seg, y),0,color=(0,0,255), thickness=5)
        detailedSeg = cv.circle(detailedSeg, (x_left, y),0,color=(0,255,255), thickness=5)
        detailedSeg = cv.circle(detailedSeg, (x_right, y),0,color=(255,0,0), thickness=5)

        roadLimits[y, :] = [x_left, x_right]
        roadLimits = roadLimits.astype(int)
                

    # cv.imshow('Detailed Segmentation', detailedSeg)
    return newImgSeg, x_center, detailedSeg, roadLimits

def build_panel(FinalImages, n_detected_lines = 0, avgXCenter = 0, road_size = 0, video_path = None, angsDegree = [0,0]):
    '''
    Build a panel containing the most important images [Final image, Image with all the considered lines, Segmented image, Blured image]
    Parameters
    ----------
    FinalImages: Array Like
                Array containing all the images
    n_detected_lines: Number of not horizontal lines detected
    avgXCenter: Average x center of the road
    road_size: Calculated road size
    video_path: Vídeo that are being analysed
    angsDegree: Angle of the average line (in degree)
    '''
    
    [Final_img, imgConsideredLines, img_canny, img_segmented, img_blur, imgDetailedSeg] = FinalImages
    
    # plt.imshow(cv.cvtColor(img_blur, cv.COLOR_BGR2RGB)), plt.xticks([]), plt.yticks([]), plt.show()
    # plt.imshow(cv.cvtColor(imgDetailedSeg, cv.COLOR_BGR2RGB)), plt.xticks([]), plt.yticks([]), plt.show()
    # plt.imshow(cv.cvtColor(img_segmented, cv.COLOR_BGR2RGB)), plt.xticks([]), plt.yticks([]), plt.show()
    # plt.imshow(cv.cvtColor(img_canny, cv.COLOR_BGR2RGB)), plt.xticks([]), plt.yticks([]), plt.show()
    # plt.imshow(cv.cvtColor(imgConsideredLines, cv.COLOR_BGR2RGB)), plt.xticks([]), plt.yticks([]), plt.show()
    # plt.imshow(cv.cvtColor(Final_img, cv.COLOR_BGR2RGB)), plt.xticks([]), plt.yticks([]), plt.show()
    # cv.imwrite("2-img_blur.png", img_blur)
    # cv.imwrite("5-detailed_seg.png", imgDetailedSeg)
    # cv.imwrite("6-road_segmented.png", img_segmented)
    # cv.imwrite("7-img_canny.png", img_canny)
    # cv.imwrite("9-considered_lines.png", imgConsideredLines)
    # cv.imwrite("10-final_lines.png", Final_img)

    Final_img = cv.putText(img = Final_img,
                           text = f"Lines: {n_detected_lines:2d} - X center: {avgXCenter:.0f}", 
                           org = (20,20), fontFace = cv.FONT_HERSHEY_SIMPLEX,fontScale = 0.5, color = (255,255,255), thickness = 1, lineType = cv.LINE_AA)
    Final_img = cv.putText(img = Final_img,
                           text = f"Angs: {angsDegree[0]:.1f}/{angsDegree[1]:.1f}", 
                           org = (20,40), fontFace = cv.FONT_HERSHEY_SIMPLEX,fontScale = 0.5, color = (255,255,255), thickness = 1, lineType = cv.LINE_AA)
    Final_img = cv.putText(img = Final_img,
                           text = f"Road Size: {road_size:.0f}px", 
                           org = (20,60), fontFace = cv.FONT_HERSHEY_SIMPLEX,fontScale = 0.5, color = (255,255,255), thickness = 1, lineType = cv.LINE_AA)
    Final_img = cv.putText(img = Final_img,
                           text = f"Vid: {video_path}", 
                           org = (20,80), fontFace = cv.FONT_HERSHEY_SIMPLEX,fontScale = 0.5, color = (255,255,255), thickness = 1, lineType = cv.LINE_AA)
    
    if img_segmented is None:
        img_segmented = np.zeros((Final_img.shape[0], Final_img.shape[1], 3), np.uint8)
    if imgDetailedSeg is None:
        imgDetailedSeg = np.zeros((Final_img.shape[0], Final_img.shape[1]), np.uint8)
    if imgConsideredLines is None:
        imgConsideredLines = np.zeros((Final_img.shape[0], Final_img.shape[1], 3), np.uint8)

    #Horizontal video
    if Final_img.shape[1] > Final_img.shape[0]:
        img2 = cv.resize(img_segmented, dsize=(0,0), fx=1/3, fy=1/3)
        img1 = cv.resize(imgDetailedSeg, dsize=(0,0), fx=1/3, fy=1/3)
        img3 = cv.resize(imgConsideredLines, dsize=(0,0), fx=1/3, fy=1/3)
        if img1.shape[0]*3 >= Final_img.shape[0]:
            total_height = img1.shape[0]*3
        else:
            total_height = Final_img.shape[0]
        total_width = Final_img.shape[1] + img1.shape[1]
        height_segment = img1.shape[0]
        output_frame = np.zeros((total_height, total_width, 3), np.uint8)
        output_frame[:Final_img.shape[0], :Final_img.shape[1], :] = Final_img
        output_frame[:height_segment, Final_img.shape[1]:, :] = img1
        output_frame[height_segment:(2*height_segment), Final_img.shape[1]:, :] = img2
        output_frame[2*height_segment:2*height_segment + img3.shape[0], 
                        Final_img.shape[1]:Final_img.shape[1] + img3.shape[1], :] = img3
    else:
        img1 = imgConsideredLines
        img3 = imgDetailedSeg
        img2 = img_segmented
        total_height = img1.shape[0]

        total_width = 4 * Final_img.shape[1]
        height_segment = img2.shape[0]

        output_frame = np.zeros((total_height, total_width, 3), np.uint8)
        output_frame[:, :Final_img.shape[1], :] = Final_img
        output_frame[:, Final_img.shape[1]:2*Final_img.shape[1], :] = img1

        output_frame[:, 2*Final_img.shape[1]:3*Final_img.shape[1], :] = img2

        output_frame[:, 3*Final_img.shape[1]:, :] = img3

    return output_frame

def detectRoadLimits(vid, out1, output_shape1, resize_factor, video_path, n_avg_lines = 40, approx=40):
    '''
    Function that will detect the left and right limits of the road in a video
    Parameters
    ----------
    vid: Video that will be analysed
    out1: Video output 1
    output_shape1: Shape of the output video 1
    resize_factor: Factor that will be used when resizing the original frame
    video_path: Video that are being analysed
    n_avg_lines: Number of average lines that will be considered when calculating the final image lines
    '''
    avg_lines_coords = np.zeros((2,4))
    all_left_lines = []
    all_pars_left_lines = []
    all_right_lines = []
    all_pars_right_lines = []
    all_x_centers_points = []
    x_seg = 0
    avgFinalLines = []
    avgXCenter = 0
    road_sizes = []
    avgParameters = [(-1, 400), (1, -400)]
    colorRange = [([35, 0, 0], [100, 255, 255])]  # Defining the HSV color range that will be used in the color segmentation process
    badLineCout = [0,0]
    roadLimits = [0]
    while(vid.isOpened()):
        (sucess, frame) = vid.read()
        if not sucess:
            break
        frame = cv.resize(frame, (int(frame.shape[1]/resize_factor),int(frame.shape[0]/resize_factor)))
        width = frame.shape[1]
                
        if cv.waitKey(3) & 0xFF == 27 or frame is None:
            break
        
        if len(road_sizes) == 0:
            avgRoadSize = 0
        else:
            avgRoadSize = int(np.average(np.array(road_sizes)))
                   
        if avgXCenter == 0:
            avgXCenter = int(width/2)
        if len(roadLimits) == 1:
            roadLimits = np.zeros((frame.shape[0], 2))
        # Generate images
        img_canny, img_segmented, img_blur, x_seg, detailedSeg, roadLimits = generateImages(frame, x_seg, colorRange, avgFinalLines, roadLimits)

        lines = cv.HoughLinesP(img_canny, 
                                2, 
                                np.pi / 180, 
                                50, 
                                np.array([]), 
                                minLineLength = 15, 
                                maxLineGap =40)

        if lines is None:
            continue

        imgAllLines = draw_lines(frame, lines, color=(0,255,255))
        cv.imshow('all_lines', imgAllLines)

        # cv.imwrite("8-img_all_lines.png", imgAllLines)
        # cv.imwrite("1-img_original.png", frame)
        
        # plt.imshow(cv.cvtColor(imgAllLines, cv.COLOR_BGR2RGB)), plt.xticks([]), plt.yticks([]), plt.show()

        final_lines, imgConsideredLines, n_detected_lines, badLineCout = calc_average_lines(frame, lines, avg_lines_coords, avgXCenter, avgRoadSize, avgParameters, badLineCout)
        
        if n_detected_lines == 0:
            continue
        
        # imgConsideredLines = draw_lines(frame, considered_lines, color=(0,255,255))

        # Adding the new detected lines to the list of all lines
        final_left_line = [line for line in final_lines if line.lineType == "Left"]
        final_right_line = [line for line in final_lines if line.lineType == "Right"]
        if len(final_left_line) > 0:
            if len(all_left_lines) == n_avg_lines:
                all_left_lines.pop(0)
            all_left_lines.append(final_left_line[0])
        if len(final_right_line) > 0:
            if len(all_right_lines) == n_avg_lines:
                all_right_lines.pop(0)
            all_right_lines.append(final_right_line[0])

        # Calculating the average lines and adding the calculated road width and x center to the lists
        if len(all_left_lines) > 0 and len(all_right_lines) > 0:
            avg_coords_left_line = np.average([line.projCoords for line in all_left_lines], axis=0).astype(int)
            avg_coords_right_line = np.average([line.projCoords for line in all_right_lines], axis=0).astype(int)
            avg_coords_left_line[0] += approx
            avg_coords_left_line[2] += approx
            avg_coords_right_line[0] -= approx
            avg_coords_right_line[2] -= approx
            roadWidth = int(avg_coords_right_line[2] - avg_coords_left_line[2])
            x_center = int(avg_coords_left_line[2] + roadWidth/2)
            if len(road_sizes) > n_avg_lines:
                road_sizes.pop(0)
            road_sizes.append(roadWidth)
            if len(all_x_centers_points) > 10:
                all_x_centers_points.pop(0)
            all_x_centers_points.append(x_center)
            # all_x_centers_points.append(road_x_center)
            avgXCenter = int(np.mean(all_x_centers_points))
            if avgXCenter < 0:
                avgXCenter = 0
            elif avgXCenter > width:
                avgXCenter = width
            avgRoadSize = int(np.average(road_sizes))

            avg_left_Line = Line(avg_coords_left_line, x_center, avgParameters, avgRoadSize, frame)
            avg_right_Line = Line(avg_coords_right_line, x_center, avgParameters, avgRoadSize, frame)
            avgFinalLines = [avg_left_Line, avg_right_Line]
            
            
        imgAvgFinalLines = np.zeros_like(frame)
                      
        if len(avgFinalLines) == 2:
            points = np.array([(avgFinalLines[0].projCoords[:2]), (avgFinalLines[0].projCoords[2:]), 
                            (avgFinalLines[1].projCoords[2:]), (avgFinalLines[1].projCoords[:2])])
            
            try:
                cv.fillPoly(imgAvgFinalLines, np.int32([points]), (255,0,0))
            except OverflowError:
                pass
        if len(avgFinalLines) > 0:
            for line in avgFinalLines: 
                imgAvgFinalLines = line.draw(imgAvgFinalLines)
            # imgAvgFinalLines = draw_lines(frame, avg_lines, color=(0,255,255), average_lines=True)
        else:
            for line in final_lines: 
                imgAvgFinalLines = line.drawProjLine(imgAvgFinalLines)

        
        finalImg = cv.addWeighted(frame, 1, imgAvgFinalLines, 1, 1)

        # Draw x center point
        y_center = int(0.9 * frame.shape[0])
        finalImg = cv.circle(finalImg, 
              center=(avgXCenter,y_center), 
              radius=0,
              color=(0,0,255),
              thickness=8)
        
        FinalImages = [finalImg, imgConsideredLines, img_canny, img_segmented, img_blur, detailedSeg]

        if len(avgFinalLines) > 0:
            
            if not None in avgFinalLines[0].parameters:
                all_pars_left_lines.append(avgFinalLines[0].parameters)
            if not None in avgFinalLines[1].parameters:
                all_pars_right_lines.append(avgFinalLines[1].parameters)
            avgParameters = calcAvgParameters(all_pars_left_lines, all_pars_right_lines, n_lines=20) # Calculating the average line parameters
        else:
            avgParameters = [(1.8, 450), (1, -300)]

        (angDegreeLeft, angDegreeRight) = (0,0)
        if not np.isnan(avgParameters[0]).any():
            angDegreeLeft = avgParameters[0][0] * (180/np.pi)
        if not np.isnan(avgParameters[1]).any():
            angDegreeRight = avgParameters[1][0] * (180/np.pi)
        

        final_panel = build_panel(FinalImages, n_detected_lines, avgXCenter, avgRoadSize, video_path, [angDegreeLeft, angDegreeRight])
        cv.imshow("Final Panel", final_panel)

        out1.write(resize_frame(final_panel, output_shape1))

        
    cv.destroyAllWindows()
    vid.release()
    return

resizing_factor = 4 # Video reducing factor
fourcc = cv.VideoWriter_fourcc(*'mp4v')
folder_path = "C:/Users/zabfw3/Documents/Faculdade/TG/TG/Videos_Castanho/Milho"  # Folder where the source videos are located
folder = folder_path.split("/")[-1]
out1 = None
# out2 = Nonefolder_path
for video_path in listdir(folder_path):
    if not os.path.isfile(f"{folder_path}/{video_path}"):
        continue
    vid = cv.VideoCapture(f"{folder_path}/{video_path}")

    if out1 == None:
        frame = vid.read()[1]
        output_shape1 = (270,640)
        # output_shape1 = (int(frame.shape[0]/resizing_factor), int((4) * frame.shape[1]/resizing_factor))  # Defining the output video shape
        # output_shape2 = (int(frame.shape[0]/resizing_factor), int(frame.shape[1]/resizing_factor))

        # plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB)), plt.xticks([]), plt.yticks([]), plt.show()

        date = datetime.now()
        time_stamp = date.strftime('%d-%m-%y - %H-%M')
        path = "C:/Users/zabfw3/Documents/Faculdade/TG/generated_videos"
        if not os.path.exists(path):
            os.mkdir(path)
        out1 = cv.VideoWriter(f'{path}/Final_panel ({time_stamp}) ({folder}).mp4',fourcc, 30, frameSize=(output_shape1[1], output_shape1[0]))
        # out2= cv.VideoWriter(f'{path}/Detailed Segmentation ({time_stamp}).mp4',fourcc, 30, frameSize=(output_shape2[1], output_shape2[0]))

    print(f"New Video: {video_path}")
    detectRoadLimits(vid, out1, output_shape1, resizing_factor, video_path)
        

out1.release()
