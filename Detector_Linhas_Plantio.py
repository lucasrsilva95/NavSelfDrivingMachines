import os
import warnings
from datetime import datetime
from os import listdir
from Line import Line

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from imutils.video import FPS, FileVideoStream

warnings.simplefilter('ignore', np.RankWarning)

def generateImages(img, x_seg, colorRange, prevRoadLimits, pointThicknes):
    '''
    Generate the Blur, Canny and segmented images
    Parameters
    ----------
    img: Source frame
    colorRange: Color range that will be used in the segmentation process
    '''
    height, width, _ = img.shape
    kernel = int(0.13*width)
    if kernel % 2 == 0:
        kernel += 1


    img_blur = cv.GaussianBlur(img, (kernel, kernel), 0)

    img_hsv = cv.cvtColor(img_blur, cv.COLOR_RGB2HSV)
    
    for (lower, upper) in colorRange:

        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mascara = cv.inRange(img_hsv, lower, upper)

    img_segmented = img.copy()
    img_segmented[mascara == 0] = 0
    img_segmented[mascara == 255] = 255
    img_canny = cv.Canny(img_segmented, 25, 35)
    # cv.imshow('canny all', img_canny)
    lines = cv.HoughLinesP(img_canny, 
                                2, 
                                np.pi / 180, 
                                50, 
                                np.array([]), 
                                minLineLength = int(height/18),
                                maxLineGap = int(height/6))

    imgAllLines = draw_lines(img, lines, color=(0,255,255))
    # cv.imshow('img_seg_verde_all_lines', imgAllLines)
    # cv.imwrite("img_seg_verde_all_lines.png", imgAllLines)


    # cv.imwrite("img_segmentacao_verde.png", img_segmented)
    # img_and = cv.bitwise_and(img, img_segmented)
    # cv.imwrite("img_segmented_parts.png", img_and)

    img_segmented, avgXSeg, detailedSeg, roadLimits = improvedSeg(img_segmented, x_seg, prevRoadLimits, pointThicknes)
    img_canny = cv.Canny(img_segmented, 25, 35)
    
    return img_canny, img_segmented, img_blur, avgXSeg, detailedSeg, roadLimits

def weightedAverageLine(lines):
    '''
    Return the average line weighting the size of the lines. The bigger lines have a larger impact on the final average line
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
    return avgLine

def draw_lines(img, lines, color=(0,255,255), average_lines=False):
    '''
    Draw the given lines to an image, coloring the space between the lines if the `average_lines` are True.
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

def calc_average_lines(img, lines, avgXCenter, avgRoadSize, avgParameters, badLineCout, img_segmented, approx):
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
    demoImg = img.copy()
        
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
                demoImg = cv.line(demoImg, (x1,y1),(x2,y1), color=(0,0,255), thickness=10)
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

        for line in sortedLeftLines[:-3]:
            demoImg = line.draw(img=demoImg)
        for line in sortedLeftLines[-3:]:
            # demoImg = line.draw(img=demoImg, color=(0,0,255))
            demoImg = line.draw(img=demoImg)
        

        AvgLeftLine = weightedAverageLine(sortedLeftLines[:3])
        FinalLeftLine = Line(AvgLeftLine, avgXCenter, avgParameters, avgRoadSize, img)
        consideredLeftLines = sortedLeftLines[:3]
        for line in consideredLeftLines: 
            imgConsLines = line.draw(imgConsLines)
        final_lines.append(FinalLeftLine)
    

    if len(rightLines) > 0:
        badLineCout[1] = 0
        sortedRightLines = sorted(rightLines, key = lambda line:line.score, reverse=True)


        for line in sortedRightLines[:-3]:
            demoImg = line.draw(img=demoImg)
        for line in sortedRightLines[-3:]:
            demoImg = line.draw(img=demoImg)
        
        # cv.imshow('Demo Img', demoImg)
        # cv.imwrite('bad_horizontal_lines.png', demoImg)

        AvgRightLine = weightedAverageLine(sortedRightLines[:2])
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
    



    # final_lines = np.array(final_lines)
    # imgAvgLines = img.copy()

    # imgAvgLines = final_lines[0].drawProjLine(imgAvgLines, thickness=7)
    # imgAvgLines = final_lines[1].drawProjLine(imgAvgLines, thickness=7)

    # cv.imshow('Average Lines', imgAvgLines)
    # cv.imwrite('img_average_Lines.png', imgAvgLines)

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

def improvedSeg(img_seg, x_seg, roadLimits, pointThicknes, minWidth = 8):
    '''
    Improve the segmentation process, only considering the main road.
    Parameters
    ----------
    img_seg: Color segmented  image
    minWidth: Minimum width that will be considered when detecting a road limit
    '''
    imgNewSeg = img_seg.copy()
    height, width, _ = img_seg.shape
    minWidth = int(0.02*width)
    distsLeftRight = np.zeros(img_seg.shape[0])
    leftRatios = []
    rightRatios = []
    left_limit = int(0.4*width)
    right_limit = int(0.6*width)
    img_seg_bin = cv.cvtColor(img_seg, cv.COLOR_BGR2GRAY)
    newImgSeg = np.copy(img_seg)
    detailedSeg = img_seg.copy()
    leftDif, rightDif = (0,0)
    if x_seg == 0:
        x_seg = int(width/2)

    x_left, x_right = (0, width)
    new_x_left, new_x_right = (0, width)
    best_columns = np.argpartition(np.average(img_seg_bin[:,left_limit:right_limit], axis=0), 4)[:4]
    minDif = width

    for column in best_columns:
        dif = abs(left_limit + column - width/2)
        if dif < minDif:
            new_x_seg = column + left_limit
            minDif = dif

    x_center = new_x_seg
    for y in range(height-1, 0, -1):
        detect = [False, False]
        line = img_seg_bin[y]
        point_count = 0
        if np.mean(line[int(new_x_seg-(minWidth)):new_x_seg+1]) < 255 and np.mean(line[new_x_seg:new_x_seg+(minWidth)]) < 255:
            for x in range(new_x_seg, width):
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
            if new_x_seg - x_left < 20 or x_right - new_x_seg < 20:
                new_x_seg = int(x_left + (x_right - x_left)/2)

        # x_seg hit a border
        else:
            new_x_right = width
            new_x_left = 0
            for i in range(1, width - new_x_seg):
                if new_x_left == 0 and new_x_right == width:
                    if (new_x_seg + i) <= width and (np.mean(line[new_x_seg + i:new_x_seg + i+minWidth]) == 0):
                        
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
                    elif new_x_right < width and np.mean(line[new_x_seg - i-minWidth:new_x_seg - i]) == 255:
                        if np.mean(line[new_x_seg - i-minWidth:new_x_seg - i]) == 255:
                            new_x_left = new_x_seg - i
                            detect[0] = True
                            break
            if not False in detect:
                new_x_seg = int(x_left + (x_right - x_left)/2)
             
        if (x_right < width and new_x_left > x_right) or (x_left > 0 and new_x_right < x_left):
            newImgSeg[y, :] = [255,255,255]
            continue
        
        if y != height-1:
            leftDif = new_x_left - x_left
            rightDif = new_x_right - x_right
            leftRatios.append(leftDif)
            rightRatios.append(rightDif)

        x_left = new_x_left
        x_right = new_x_right

        newImgSeg[y, :x_left] = [255,255,255]
        newImgSeg[y, x_right:] = [255,255,255]
        newImgSeg[y, x_left:x_right] = [0,0,0]

        detailedSeg = cv.circle(detailedSeg, (new_x_seg, y),0,color=(0,0,255), thickness=pointThicknes)
        detailedSeg = cv.circle(detailedSeg, (x_left, y),0,color=(0,255,255), thickness=pointThicknes)
        detailedSeg = cv.circle(detailedSeg, (x_right, y),0,color=(255,0,0), thickness=pointThicknes)

        roadLimits[y, :] = [x_left, x_right]
        roadLimits = roadLimits.astype(int)
        if x_left > 0 and x_right < width:
            distsLeftRight[y] = x_right - x_left
 

    return newImgSeg, x_center, detailedSeg, roadLimits

def build_panel(FinalImages, gap):
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
    
    # cv.imwrite("img_blur.png", img_blur)
    # cv.imwrite("img_detailed_seg.png", imgDetailedSeg)
    # cv.imwrite("img_road_segmented.png", img_segmented)
    # cv.imwrite("img_canny.png", img_canny)
    # cv.imwrite("img_considered_lines.png", imgConsideredLines)
    # cv.imwrite("img_result.png", Final_img)

    
    if img_segmented is None:
        img_segmented = np.zeros((Final_img.shape[0], Final_img.shape[1], 3), np.uint8)
    if imgDetailedSeg is None:
        imgDetailedSeg = np.zeros((Final_img.shape[0], Final_img.shape[1]), np.uint8)
    if imgConsideredLines is None:
        imgConsideredLines = np.zeros((Final_img.shape[0], Final_img.shape[1], 3), np.uint8)

    #Horizontal video
    if Final_img.shape[1] > Final_img.shape[0]:

        height_segment = int(Final_img.shape[0]/4)
        width_segment = int(Final_img.shape[1]/4)

        img1 = cv.resize(img_blur, dsize=(width_segment,height_segment))
        img2 = cv.resize(imgDetailedSeg, dsize=(width_segment,height_segment))
        img3 = cv.resize(img_segmented, dsize=(width_segment,height_segment))
        img4 = cv.resize(imgConsideredLines, dsize=(width_segment,height_segment))
        # img4 = cv.resize(imgConsideredLines, dsize=(0,0), fx=1/4, fy=1/4)

        total_height = max(Final_img.shape[0], 4*height_segment)
        total_width = Final_img.shape[1] + width_segment
        output_frame = np.zeros((total_height, total_width, 3), np.uint8)

        output_frame[:Final_img.shape[0], :Final_img.shape[1], :] = Final_img
        output_frame[:height_segment, Final_img.shape[1]:, :] = img1
        output_frame[height_segment:(2*height_segment), Final_img.shape[1]:, :] = img2
        output_frame[(2*height_segment):(3*height_segment), Final_img.shape[1]:, :] = img3
        output_frame[(3*height_segment):(4*height_segment), Final_img.shape[1]:, :] = img4
    else:
        img1 = Final_img
        img2 = imgConsideredLines
        img3 = img_segmented
        img4_1 = cv.resize(img_blur, dsize=(0,0), fx=1/2, fy=1/2)
        img4_2 = cv.resize(imgDetailedSeg, dsize=(0,0), fx=1/2, fy=1/2)
        total_height = img1.shape[0]
        h_limits = [Final_img.shape[1], 2*Final_img.shape[1] + gap, 3*Final_img.shape[1] + 2*gap, 3*Final_img.shape[1]+img4_1.shape[1] + 3*gap]
        v_limits = [img4_1.shape[0], 2*img4_1.shape[0]]
        total_width = 3 * Final_img.shape[1] + img4_1.shape[1] + (3 * gap)

        output_frame = np.zeros((total_height, total_width, 3), np.uint8)
        output_frame[:] = (244,220,179)    # Color of the gap between images
        output_frame[:, :h_limits[0], :] = img1
        output_frame[:, h_limits[0]+gap:h_limits[1], :] = img2
        output_frame[:, h_limits[1]+gap:h_limits[2], :] = img3

        output_frame[:v_limits[0], h_limits[2]+gap:h_limits[3], :] = img4_1
        output_frame[v_limits[0]:v_limits[1], h_limits[2]+gap:h_limits[3], :] = img4_2

    return output_frame

def detectRoadLimits(vid, out, output_shape1, resize_factor, video_path, font_path, txtFile, proc_ratio = 30, n_avg_frames = 8):
    '''
    Function that will detect the left and right limits of the road in a video
    Parameters
    ----------
    vid: Video that will be analysed
    out: Video output 1
    output_shape1: Shape of the output video 1
    resize_factor: Factor that will be used when resizing the original frame
    video_path: Video that are being analysed
    proc_ratio: Number of frames that will be analysed per second
    n_avg_lines: Number of average lines that will be considered when calculating the final image lines
    '''
    
    all_left_lines = []
    all_pars_left_lines = []
    all_right_lines = []
    all_pars_right_lines = []
    x_seg = 0
    avgFinalLines = []
    avgXCenter = 0
    if min(output_shape1) < 400: #360p
        avgParameters = [(-1, 400), (1, -400)]   
        fontSize = 16; origin = (10,20) 
        lineThickness = 8
        pointThicknes = 6
    elif min(output_shape1) < 600:  #480p
        avgParameters = [(-1, 800), (1, -800)]    
        fontSize = 21; origin = (20,20)
        lineThickness = 10
        pointThicknes = 8
    else:                           #720p
        avgParameters = [(-1, 2000), (1, -2000)]    
        fontSize = 32; origin = (20,20)
        lineThickness = 12
        pointThicknes = 10

    road_sizes = []
    colorRange = [([35, 0, 0], [100, 255, 255])]  # Defining the HSV color range that will be used in the color segmentation process
    badLineCout = [0,0]
    roadLimits = [0]
    current_fps = FPS().start()
    avg_fps = FPS().start()

    procRate = 0
    all_proc_rates = []
    while(vid.isOpened()):
        (sucess, frame) = vid.read()
        if not sucess:
            break
        frame = cv.resize(frame, dsize=(0,0), fx=1/resize_factor, fy=1/resize_factor)
        height, width, _ = frame.shape
            
        approx = int(width/25)
        # approx = 0
        if cv.waitKey(3) & 0xFF == 27 or frame is None:
            break

        if current_fps._numFrames % int(30/proc_ratio) == 0:

            current_fps.stop()
            procRate = current_fps.fps()
            current_fps = FPS().start()
            # procRate = 30/(timeDelta.seconds + timeDelta.microseconds/1000000)
            # print(f"proc/delt Man: {procRate}/{timeDelta.seconds + timeDelta.microseconds/1e6} - proc/delt Auto: {pRate}/{dtime}")
            all_proc_rates.append(procRate)
            if len(road_sizes) == 0:
                avgRoadSize = 0
            else:
                avgRoadSize = int(np.average(np.array(road_sizes)))
                    
            if avgXCenter == 0:
                avgXCenter = int(width/2)
            if len(roadLimits) == 1:
                roadLimits = np.zeros((frame.shape[0], 2))
            # Generate images
            img_canny, img_segmented, img_blur, x_seg, detailedSeg, roadLimits = generateImages(frame, x_seg, colorRange, roadLimits, pointThicknes)

            lines = cv.HoughLinesP(img_canny, 
                                    2, 
                                    np.pi / 180, 
                                    50, 
                                    np.array([]), 
                                    minLineLength = int(height/18),
                                    maxLineGap = int(height/6))

            if lines is None:
                continue

            imgAllLines = draw_lines(frame, lines, color=(0,255,255))
            # cv.imshow('all_lines', imgAllLines)

            # cv.imwrite("img_all_lines.png", imgAllLines)
            # cv.imwrite("img_original.png", frame)
            

            final_lines, imgConsideredLines, n_detected_lines, badLineCout = calc_average_lines(frame, lines, avgXCenter, avgRoadSize, 
                                                                                        avgParameters, badLineCout, img_segmented, approx = approx)
            
            if n_detected_lines == 0:
                continue
            
            # imgConsideredLines = draw_lines(frame, considered_lines, color=(0,255,255))

            # Adding the new detected lines to the list of all lines
            final_left_line = [line for line in final_lines if line.lineType == "Left"]
            final_right_line = [line for line in final_lines if line.lineType == "Right"]
            if len(final_left_line) > 0:
                if len(all_left_lines) == n_avg_frames:
                    all_left_lines.pop(0)
                all_left_lines.append(final_left_line[0])
            if len(final_right_line) > 0:
                if len(all_right_lines) == n_avg_frames:
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

                if x_center < 0:
                    x_center = 0
                elif x_center > width:
                    x_center = width


                avgRoadSize = roadWidth
                avgXCenter = x_center

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
                    imgAvgFinalLines = line.draw(imgAvgFinalLines, thickness=lineThickness)
            else:
                for line in final_lines: 
                    imgAvgFinalLines = line.drawProjLine(imgAvgFinalLines, thickness=lineThickness)


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
        

        finalImg = cv.addWeighted(frame, 1, imgAvgFinalLines, 1, 1)

        # cv.imwrite('img_avg_lines.png', finalImg)

        # Draw x center point


        y_center = int(0.95 * frame.shape[0])
        finalImg = cv.circle(finalImg, 
              center=(avgXCenter,y_center), 
              radius=0,
              color=(0,0,255),
              thickness=lineThickness)
        
        FinalImages = [finalImg, imgConsideredLines, img_canny, img_segmented, img_blur, detailedSeg]

        final_panel = build_panel(FinalImages, gap = int(0.015*width))

        img_resize = cv.resize(final_panel, dsize=output_shape1)

        font = ImageFont.truetype(font_path, fontSize)
        img_pil = Image.fromarray(img_resize)
        draw = ImageDraw.Draw(img_pil)

        draw.text(xy = origin, 
                text = f"Lines: {n_detected_lines:2d} - X center: {avgXCenter:.0f}" +
                    f"\nAngs: {angDegreeLeft:.1f}°/{angDegreeRight:.1f}°" +
                    f"\nRoad Size: {avgRoadSize:.0f}px" +
                    f"\nProc: {procRate:.1f} fps" +
                    f"\nVideo: {video_path[:-4]}",
                    font = font, 
                    color = (255,255,255))

        
        Final_img = np.array(img_pil)
        
        cv.imshow("Final Image", Final_img)
        # cv.imwrite("Final_Image.png", Final_img)
        out.write(Final_img)

        # frame_cout += 1
        current_fps.update()
        avg_fps.update()

    avg_fps.stop()

    text_details = (f"\nElapsed time: {avg_fps.elapsed()//3600:02.0f}:{(avg_fps.elapsed()%3600)//60:02.0f}:{avg_fps.elapsed()%60:02.0f}" +
                    f"\nAverage processing speed: {np.average(all_proc_rates):.1f} fps\n\n")
    
    print(text_details)
    txtFile.write(text_details)
    
    cv.destroyAllWindows()
    vid.release()
    return

if __name__ == '__main__':
    # 720p: 1.5
    # 480p: 2.25
    # 360p: 3
    # 270p: 4
    resize_factor = 1.5 # Video reducing factor
    # proc_ratio = 6  #Number of frames to be analysed per second (max 30)
    font_path = "C:/Users/zabfw3/TG/NavSelfDrivingMachines/Fonts/arial.TTF"
    # global folder_path
    folder_path = "C:/Users/zabfw3/TG"
    videos_folder_path = f"{folder_path}/Source_videos"  # Folder where the source videos are located
    folder_name = videos_folder_path.split("/")[-1]
    if not os.path.exists(f"{folder_path}/generated_videos"):
        os.mkdir(f"{folder_path}/generated_videos")

    for proc_ratio in [1, 3]:
        date_start = datetime.now()
        time_stamp = date_start.strftime('%d-%m-%y - %H-%M')
        output_path = f"{folder_path}/generated_videos/Video ({time_stamp}) ({int(1080/resize_factor)}p) ({proc_ratio}fps)"
        os.mkdir(output_path)
        txtFile = open(f"{output_path}/Details final video.txt", "w+")
        fourcc = cv.VideoWriter_fourcc(*'mp4v')

        
        out1 = None
        total_frames = 0
        for video_path in listdir(videos_folder_path):
            if not os.path.isfile(f"{videos_folder_path}/{video_path}"):
                continue
            vid = cv.VideoCapture(f"{videos_folder_path}/{video_path}")
            total_frames += int(vid.get(cv.CAP_PROP_FRAME_COUNT))
            if out1 == None:
                frame = vid.read()[1]
                output_shape1 = (int(frame.shape[1]/resize_factor), int(frame.shape[0]//resize_factor))
                output_shape1 = (max(output_shape1), min(output_shape1))
            
                
                
                fileName = f'{output_path}/Final video ({int(1080/resize_factor)}p) ({proc_ratio}fps).mp4'
                out1 = cv.VideoWriter(fileName, fourcc, 30, frameSize=output_shape1)

            print(f"Video: {video_path}")
            txtFile.write(f"Video: {video_path}")
            detectRoadLimits(vid, out1, output_shape1, resize_factor, video_path, font_path, txtFile, proc_ratio, n_avg_frames=int(1.7*proc_ratio))
            
        out1.release()

        date_end = datetime.now()
        totalTime = date_end - date_start
        # newFileName = f"{output_path}/Final_video ({time_stamp}) ({int(1080/resize_factor)}p) ({proc_ratio}fps) - Tempo de execução {totalTime.seconds//60} min e {totalTime.seconds%60}s - Avg FPS {total_frames/(totalTime.seconds + totalTime.microseconds/1e6):.2f}fps.mp4"
        # os.rename(fileName, newFileName)

        text_final_detail = (f"\n====================== FINAL RESULT ======================" +
                            f"\nVideo quality: {int(1080/resize_factor)}p" +
                            f"\nProcessing rate: {proc_ratio} fps" +
                            f"\nExecution time: {totalTime.seconds//3600:02.0f}:{(totalTime.seconds%3600)//60:02.0f}:{totalTime.seconds%60:02.0f}" +
                            f"\nAverage processing speed: {total_frames/(totalTime.seconds + totalTime.microseconds/1e6):.1f} fps\n\n")

        print(text_final_detail)
        txtFile.write(text_final_detail)
        txtFile.close()

