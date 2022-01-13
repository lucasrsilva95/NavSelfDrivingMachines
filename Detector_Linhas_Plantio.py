import os
from typing_extensions import final
import warnings
from datetime import datetime
from os import listdir


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

warnings.simplefilter('ignore', np.RankWarning)

def calcAvgRoadSize(y_max, left_lines, right_lines, n_lines=20):

    dists = []
    arr_left_lines = np.array(left_lines)[:, -n_lines:]
    arr_right_lines = np.array(right_lines)[:, -n_lines:]

    for i in range(arr_left_lines.shape[1]):
        x1, y1, x2, y2 = arr_left_lines[:,i]
        x_left = x2 - ((y_max - y2)/(y1 - y2))*(x2 - x1)
        
        x1, y1, x2, y2 = arr_right_lines[:,i]
        x_right = x2 - ((y_max - y2)/(y1 - y2))*(x2 - x1)
        
        d = x_right - x_left
        dists.append(d)
    
    if len(dists) > 0:
        return np.average(np.array(dists))
    else:
        return 0

def calcAvgParameters(left_lines, right_lines, n_lines=30):
    parLeftLine = [[],[]]
    parRigthLine = [[],[]]
    arr_left_lines = np.array(left_lines)[:, -n_lines:]
    arr_right_lines = np.array(right_lines)[:, -n_lines:]

    for i in range(arr_left_lines.shape[1]):
        x1, y1, x2, y2 = arr_left_lines[:,i]
        pars = np.polyfit((x1, x2), (y1, y2), 1)
        parLeftLine[0].append(pars[0])
        parLeftLine[1].append(pars[1])
    for i in range(arr_right_lines.shape[1]):
        x1, y1, x2, y2 = arr_right_lines[:,i]
        pars = np.polyfit((x1, x2), (y1, y2), 1)
        parRigthLine[0].append(pars[0])
        parRigthLine[1].append(pars[1])
    avgParsLeft = [np.average(parLeftLine[0]), np.average(parLeftLine[1])]
    avgParsRigth = [np.average(parRigthLine[0]), np.average(parRigthLine[1])]
    return [avgParsLeft, avgParsRigth]

def distRoadCenter(line, x_center, y_max):
    x1, y1, x2, y2 = line.reshape(4)
    x = x2 - ((y_max - y2)/(y1 - y2))*(x2 - x1)
    d = x_center - x

    return d

def draw_lines(img, lines, color=(0,255,255), average_lines=False):
    # Obter uma imagem das linhas desenhadas em um fundo preto
    imgLinhas = np.zeros_like(img)

    if average_lines:
        lines = [lines]
    for line_pair in lines:
        if line_pair is not None:
            if average_lines and len(line_pair[0]) > 1 and len(line_pair[1]) > 1:
                points = np.array([(line_pair[0,0], line_pair[0,1]), (line_pair[0,2], line_pair[0,3]), 
                                (line_pair[1,2], line_pair[1,3]), (line_pair[1,0], line_pair[1,1])])
                try:
                    cv.fillPoly(imgLinhas, np.int32([points]), (255,0,0))
                except OverflowError:
                    return imgLinhas
            for line in line_pair:
                if len(line) == 4:
                    x1, y1, x2, y2 = line.reshape(4)
                    cv.line(imgLinhas, (x1, y1), (x2, y2), color, 10)
                # else:
                    # print("Erro")
    return imgLinhas

def heightedAverageLine(lines):
    result = []
    for line in lines:
        n = int(line[6]/10)
        for i in range(n):
            result.append(line)
    avgLine = np.average(np.array(result), axis=0)
    avgLine[0] = avgLine[0].astype(int)
    return avgLine

def drawCenterPoint(img, all_x_center_points, n_points=10):
    '''
    Draw the center point in a given image. the center point is the average of the last n_points. 

    Parameters
    ----------
    img : Image where the point will be drawn.
    all_x_center_points : array like
                          array that contains all the center points calculated so far.
    n_points : integer
               Number of points to be considered when calculating the average of the last points in ´all_x_center_points´.
    '''
    if len(all_x_center_points) == 0 or np.mean(all_x_center_points[-n_points:]) <= 0:
        avgXCenter = img.shape[1]/2
        return img, avgXCenter
    avgXCenter = int(np.mean(all_x_center_points[-n_points:]))
    y_center = int(0.9 * img.shape[0])
    img_with_center = cv.circle(img, 
              center=(avgXCenter,y_center), 
              radius=0,
              color=(0,0,255),
              thickness=8)
    return img_with_center, avgXCenter

def calc_average_lines(img, lines, avgXCenter, avgRoadSize, avgParameters, aprox=50, n_lines=3):
    '''
    Obter as linhas medias de um conjunto de linhas
    '''
    height = img.shape[0]
    width = img.shape[1]
    AvgLeftLine = []
    AvgRightLine = []
    lines_filter_right = []
    lines_filter_left = []
    if avgRoadSize == 0:
        avgRoadSize = 0.8 * width
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        try:
            parameters = np.polyfit((x1, x2), (y1, y2), 1)  # Retorna a inclinação e a coordenada y
        except OverflowError:
            pass
        # Parametros da equação da linha (y = ax + b)
        a = parameters[0]
        b = parameters[1]

        # if abs(a) < 0.4 or abs(a) > 8:
        if abs(a) < 0.4:
            # print(f"Bad line, a = {a}")
            continue
        if avgXCenter == 0 or avgXCenter > img.shape[1]:
            avgXCenter = img.shape[1]/2
        dist_center = distRoadCenter(line, avgXCenter, height)
        dist_error = avgRoadSize/2 - abs(dist_center)
        lineSize = np.sqrt((x2 - x1)**2 + (y1 - y2)**2)
        percDistBase = (height - y1)/height
        # if dist_center > 0 and x1 < largura/2:
        if x1 < avgXCenter and ((a < 0 and avgParameters[0][0] < 0) or  (a > 0 and avgParameters[0][0] > 0)):
            ang_error = abs(avgParameters[0][0] - a)
            y_error = abs(avgParameters[0][1] - b)
            score = (y_error/100) + ang_error*10 + abs(dist_error/40) - percDistBase*2 - lineSize/200
            lines_filter_left.append([line, parameters, ang_error, y_error, abs(dist_error), score, lineSize])
            
        elif x1 > avgXCenter and ((a < 0 and avgParameters[1][0] < 0) or  (a > 0 and avgParameters[1][0] > 0)):
            ang_error = abs(avgParameters[1][0] - a)
            y_error = abs(avgParameters[1][1] - b)
            score = (y_error/100) + ang_error*10 + abs(dist_error/40) - percDistBase*2 - lineSize/200
            lines_filter_right.append([line, parameters, ang_error, y_error, abs(dist_error), score, lineSize])
  
    finalLeftLine = []
    finalRigthLine = []
    consideredLeftLines = []
    consideredRightLines = []
    n_detected_lines = len(lines_filter_left) + len(lines_filter_right)
    if len(lines_filter_left) > 0:
        lines_filter_left.sort(key=lambda line_left: line_left[5], reverse=False)    # Filtering by angle error
        lines_filter_dist = lines_filter_left[:5]
        lines_filter_dist.sort(key = lambda line_dist: line_dist[4], reverse=False) # Filtering by y error
        lines_filter_angle = lines_filter_dist[:3]
        AvgLeftLine = heightedAverageLine(lines_filter_angle)

        consideredLeftLines = [line[0].reshape(4) for line in lines_filter_angle]

        # left_lines_par = [line[1].reshape(2) for line in lines_filter_angle]
        # avgLeftLinesPar = np.average(left_lines_par, axis=0)

        finalLeftLine = get_coordinates(img.shape[0], AvgLeftLine[1])
        finalLeftLine[0] += aprox
        finalLeftLine[2] += aprox

    if len(lines_filter_right) > 0:
        lines_filter_right.sort(key=lambda RightLine: RightLine[5], reverse=False)
        lines_filter_dist = lines_filter_right[:5]
        lines_filter_dist.sort(key = lambda line: line[4], reverse=False)
        lines_filter_angle = lines_filter_dist[:3]
        AvgRightLine = heightedAverageLine(lines_filter_angle)
        consideredRightLines = [line[0].reshape(4) for line in lines_filter_angle] 
        # rigth_lines_par = [line[1].reshape(2) for line in lines_filter_angle]
        # avgRigthLinesPar = AvgRightLine[1]

        finalRigthLine = get_coordinates(img.shape[0], AvgRightLine[1])
        finalRigthLine[0] -= aprox
        finalRigthLine[2] -= aprox



    return np.array([finalLeftLine, finalRigthLine]), np.array([consideredLeftLines, consideredRightLines]), n_detected_lines

def get_coordinates(y_max, parametros_linha):
    # Obter as coordenadas dos pontos da linha
    # Equação da Linha (y = ax + b)
    a, b = parametros_linha
    y2 = int(y_max * 0.5)  # Linha média vai até 3/5 da imagem
    x1 = int((y_max - b) / a)
    x2 = int((y2 - b) / a)
    return np.array([x1, y_max, x2, y2])

def improvedSeg(img_seg, x_center, minWidth = 18):
    img_seg_bin = cv.cvtColor(img_seg, cv.COLOR_BGR2GRAY)
    newImgSeg = np.copy(img_seg)
    # final_x_center = 0
    # final_x_center = int(x_center)
    img_x_center = int(img_seg.shape[1]/2)

    # if x_center > img_seg.shape[1]:
    #     new_x_center = img_seg.shape[1] - 10
    # elif x_center < 0:
    #     x_center = 0
    # else:
    #     new_x_center = int(x_center)
        
    new_x_center = int(img_x_center)
    for y in range(img_seg_bin.shape[0]-1, 0, -1):
        line = img_seg_bin[y]
        point_count = 0
        x_right = 0
        x_left = 0
        if line[new_x_center] == 0:
            for x in range(new_x_center, img_seg.shape[1]):
                if line[x] == 255:
                    if point_count == minWidth:
                        x_right = x - minWidth
                        newImgSeg[y, x_right:] = [255,255,255]
                        break
                    else:
                        point_count += 1
                else:
                    point_count = 0
            for x in range(new_x_center, 0, -1):
                if line[x] == 255:
                    if point_count == minWidth:
                        x_left = x + minWidth
                        newImgSeg[y, :x_left] = [255,255,255]
                        break
                    else:
                        point_count += 1
                else:
                    point_count = 0
            if img_x_center == 0:
                img_x_center = int(x_left + (x_right - x_left)/2)
        else:
            # newImgSeg[y, :] = [255,255,255]
            for i in range(1, img_seg.shape[1] - new_x_center):
                if x_left == 0 and line[new_x_center + i] == 0:
                    x_left = new_x_center + i
                elif x_left > 0 and line[new_x_center + i] == 255:
                    x_right = new_x_center + i
                    break
            if x_right == 0:
                x_right = img_seg.shape[1]
            new_x_center = int(x_left + (x_right - x_left)/2)
            for x in range(new_x_center, img_seg.shape[1]):
                if line[x] == 255:
                    if point_count == minWidth:
                        x_limit_right = x - minWidth
                        newImgSeg[y, x_limit_right:] = [255,255,255]
                        break
                    else:
                        point_count += 1
                else:
                    point_count = 0
            for x in range(new_x_center, 0, -1):
                if line[x] == 255:
                    if point_count == minWidth:
                        x_limit_left = x + minWidth
                        newImgSeg[y, :x_limit_left] = [255,255,255]
                        break
                    else:
                        point_count += 1
                else:
                    point_count = 0
                
    # if img_x_center == 0:
    #     img_x_center = new_x_center             
    return newImgSeg, img_x_center

def detectLines(img, limitesCores, avg_lines, avgXCenter, avgRoadSize, avgParameters):
        
    img_blur = cv.blur(img, (19, 19))

    cv.imshow('Blur', img_blur)
    img_hsv = cv.cvtColor(img_blur, cv.COLOR_RGB2HSV)
    # img_blur = cv.GaussianBlur(img_hsv, (185,265),0)
    

    #mascaraFinal = np.zeros(img_blur.shape[:2], dtype="uint8")   #Inicializando a mascara final

    for (lower, upper) in limitesCores:

        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mascara = cv.inRange(img_hsv, lower, upper)

        #mascaraFinal = cv.bitwise_or(mascaraFinal, mascara)

    # mascaraFinal = cv.bitwise_not(mascaraFinal)
    # res = cv.bitwise_and(img_blur, img_blur, mask=mascaraFinal)
    # img_filt = cv.cvtColor(res, cv.COLOR_HSV2RGB)
    img_segmented = img.copy()
    img_segmented[mascara == 0] = 0
    img_segmented[mascara == 255] = 255

    img_segmented, _ = improvedSeg(img_segmented, avgXCenter)
    img_canny = cv.Canny(img_segmented, 25, 35)
    lines = cv.HoughLinesP(img_canny, 
                                2, 
                                np.pi / 180, 
                                50, 
                                np.array([]), 
                                minLineLength = 20, 
                                maxLineGap = 70)

    if lines is None:
        return None, None, None, None, None, None, 0 

    final_lines, considered_lines, n_detected_lines = calc_average_lines(img, lines, avgXCenter, avgRoadSize, avgParameters)
    imgLines = draw_lines(img, considered_lines, color=(0,255,255))
    imgFinalLines = draw_lines(img, final_lines, color=(0,255,255), average_lines=True)
    
    imgWithLines = cv.addWeighted(img, 0.6, imgLines, 1, 1)
    if np.max(avg_lines) > 0:
        imgLinhasAvg = draw_lines(img, avg_lines, color=(0,255,255), average_lines=True)
        imgWithFinalLines = cv.addWeighted(img, 0.6, imgLinhasAvg, 1, 1)
    else:
        imgWithFinalLines = cv.addWeighted(img, 0.6, imgFinalLines, 1, 1)

    return imgWithFinalLines, imgWithLines, final_lines, imgFinalLines, img_segmented, img_canny, n_detected_lines

def resize_frame(frame, output_shape):
    '''
        Resize the frame to the output_shape, filling the empty spaces with black pixels
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

def build_panel(img_final, img_segmented=None, img_canny=None, img_with_all_lines=None, n_detected_lines = 0, road_size = 0):

    img_final = cv.putText(img = img_final, 
                           text = f"N: {n_detected_lines} - Road Size: {road_size:.1f}px", 
                           org = (20,30),
                           fontFace = cv.FONT_HERSHEY_SIMPLEX,
                           fontScale = 0.6,
                           color = (255,255,255),
                           thickness = 2,
                           lineType = cv.LINE_AA)
    

    if img_segmented is None:
        img_segmented = np.zeros((img_final.shape[0], img_final.shape[1], 3), np.uint8)
    if img_canny is None:
        img_canny = np.zeros((img_final.shape[0], img_final.shape[1]), np.uint8)
    if img_with_all_lines is None:
        img_with_all_lines = np.zeros((img_final.shape[0], img_final.shape[1], 3), np.uint8)

    #Horizontal video
    if img_final.shape[1] > img_final.shape[0]:
        img1 = cv.resize(img_segmented, dsize=(0,0), fx=1/3, fy=1/3)
        img2 = cv.resize(img_canny, dsize=(0,0), fx=1/3, fy=1/3)
        img3 = cv.resize(img_with_all_lines, dsize=(0,0), fx=1/3, fy=1/3)
        if img1.shape[0]*3 >= img_final.shape[0]:
            total_height = img1.shape[0]*3
        else:
            total_height = img_final.shape[0]
        total_width = img_final.shape[1] + img1.shape[1]
        height_segment = img1.shape[0]
        output_frame = np.zeros((total_height, total_width, 3), np.uint8)
        output_frame[:img_final.shape[0], :img_final.shape[1], :] = img_final
        output_frame[:height_segment, img_final.shape[1]:, :] = img1
        output_frame[height_segment:(2*height_segment), img_final.shape[1]:, 0] = img2
        output_frame[height_segment:(2*height_segment), img_final.shape[1]:, 1] = img2
        output_frame[height_segment:(2*height_segment), img_final.shape[1]:, 2] = img2
        output_frame[2*height_segment:2*height_segment + img3.shape[0], 
                        img_final.shape[1]:img_final.shape[1] + img3.shape[1], :] = img3
    else:
        # img2 = cv.resize(img_canny, dsize=(0,0), fx=1/2, fy=1/2)
        # img3 = cv.resize(img_with_all_lines, dsize=(0,0), fx=1/2, fy=1/2)
        img1 = img_with_all_lines
        img2 = img_canny
        img3 = img_segmented
        total_height = img1.shape[0]

        total_width = 4 * img_final.shape[1]
        height_segment = img2.shape[0]

        output_frame = np.zeros((total_height, total_width, 3), np.uint8)
        output_frame[:, :img_final.shape[1], :] = img_final
        output_frame[:, img_final.shape[1]:2*img_final.shape[1], :] = img1

        output_frame[:, 2*img_final.shape[1]:3*img_final.shape[1], 0] = img2
        output_frame[:, 2*img_final.shape[1]:3*img_final.shape[1], 1] = img2
        output_frame[:, 2*img_final.shape[1]:3*img_final.shape[1], 2] = img2

        output_frame[:, 3*img_final.shape[1]:, :] = img3

    return output_frame

#######  Análise Vídeo  #######
def final_func(vid, out1, output_shape1, resize_factor):

    n_avg_lines = 20
     
    avg_lines = np.zeros((2,4))
    all_left_lines = [[],[],[],[]]
    all_right_lines = [[],[],[],[]]
    all_x_centers_points = []
    avgXCenter = 0
    avg_road_sizes = []

    while(vid.isOpened()):
        (sucess, frame) = vid.read()
        if not sucess:
            break
        frame = cv.resize(frame, (int(frame.shape[1]/resize_factor),int(frame.shape[0]/resize_factor)))
        width = frame.shape[1]
        limitesCores = [([35, 0, 0], [100, 255, 255])]; blur = 65; largIni = -20; largFinal = width + 20; alturaCorte = -50  # Parametros Vid 2
        # limitesCores = [([60, 70, 50], [100, 240, 240])]; blur = 51; largIni = -20; largFinal = largura + 20; alturaCorte = -50  # Parametros Vid 2
        # limitesCores = [([60, 70, 50], [100, 240, 240])]; blur = 21; largIni = 80; largFinal = largura - 80; alturaCorte = -50  # Parametros Vid 4
        
        if cv.waitKey(3) & 0xFF == 27 or frame is None:
            break
        
        #try:
        # avgRoadSize = calcAvgRoadSize(frame.shape[0], all_left_lines, all_right_lines, n_lines=20)
        if len(avg_road_sizes) == 0:
            avgRoadSize = 0
        else:
            avgRoadSize = np.average(np.array(avg_road_sizes)[-5:])

        if len(all_left_lines[0]) > 0 and len(all_right_lines[0]) > 0:
            avgParameters = calcAvgParameters(all_left_lines, all_right_lines, n_lines=20)
        else:
            avgParameters = [(-1.8, 450), (1, -300)]
        if avgXCenter == 0:
            avgXCenter = int(width/2)
        img_parameters = cv.putText(img = np.zeros((200,500)), 
                           text = f"aLeft: {avgParameters[0][0]:.2f} / aRight: {avgParameters[1][0]:.2f}", 
                           org = (20,30), fontFace = cv.FONT_HERSHEY_SIMPLEX,fontScale = 0.6,color = (255,255,255),thickness = 1,lineType = cv.LINE_AA)
        img_parameters = cv.putText(img = img_parameters, 
                           text = f"x_Center: {avgXCenter:.0f} / Road_Size: {avgRoadSize:.2f}", 
                           org = (20,60), fontFace = cv.FONT_HERSHEY_SIMPLEX,fontScale = 0.6,color = (255,255,255),thickness = 1,lineType = cv.LINE_AA)
        cv.imshow("Parameters", img_parameters)
        
        imgComLinhas, img_with_all_lines, final_lines, imgFinalLines, img_segmented, img_canny, n_detected_lines = detectLines(frame, limitesCores, avg_lines, avgXCenter, avgRoadSize, avgParameters)
        
        if n_detected_lines == 0:
            continue

        finalImage, avgXCenter = drawCenterPoint(imgComLinhas, all_x_centers_points)
        
        final_panel = build_panel(finalImage, img_segmented, img_canny, img_with_all_lines, n_detected_lines, avgRoadSize)
        cv.imshow("Final Panel", final_panel)

        out1.write(resize_frame(final_panel, output_shape1))
        # out2.write(resize_frame(final_panel, output_shape2))

        for i, coord in enumerate(final_lines[0]):
            if len(all_left_lines[i]) > 100:
                all_left_lines[i].pop(0)
            all_left_lines[i].append(coord)
        for i, coord in enumerate(final_lines[1]):
            if len(all_right_lines[i]) > 100:
                all_right_lines[i].pop(0)
            all_right_lines[i].append(coord)

        if len(all_left_lines[0]) > 0 and len(all_right_lines[0]) > 0:
            for i in range(4):
                avg_lines[0,i] = np.mean(all_left_lines[i][-n_avg_lines:])
                avg_lines[1,i] = np.mean(all_right_lines[i][-n_avg_lines:])
            avg_lines = avg_lines.astype(int)
            roadWidth = int(avg_lines[1,0] - avg_lines[0,0])
            x_center = int(avg_lines[0,0] + roadWidth/2)

            avg_road_sizes.append(roadWidth)
            all_x_centers_points.append(x_center)

        # except:
        #     if imgLinhasAnterior is not None:
        #         imgComLinhas = cv.addWeighted(frame, 0.6, imgFinalLines, 1, 1)  # Adicionando as linhas à imagem original
                
        #         #cv.imshow("imgComLinhasMedias",imgComLinhas)
        #         out1.write(resize_frame(imgComLinhas, output_shape1))
        #         #out1.write(imgComLinhas)
        #         final_panel = build_panel(imgComLinhas)
        #         out2.write(resize_frame(final_panel, output_shape2))
        #     else:
        #         #cv.imshow("imgComLinhasMedias",frame)
        #         #out1.write(resize_frame(frame, output_shape1))
        #         final_panel = build_panel(frame)
        #         out2.write(resize_frame(final_panel, output_shape2))
                
        #         out1.write(frame)

            # pass
        # cv.waitKey(1)

    cv.destroyAllWindows()
    vid.release()
    return

resizing_factor = 3
fourcc = cv.VideoWriter_fourcc(*'mp4v')
folder_list = ["C:/Users/zabfw3/Documents/Faculdade/TG/TG/Videos_Castanho/Milho/VideosBons",
                "C:/Users/zabfw3/Documents/Faculdade/TG/TG/Videos_Castanho/Milho/Desafios"]
out1 = None
# out2 = None
for folder in folder_list:
    for video_path in listdir(folder):
        if not os.path.isfile(f"{folder}/{video_path}"):
            continue
        vid = cv.VideoCapture(f"{folder}/{video_path}")

        if out1 == None:
            frame = vid.read()[1]
            # output_shape1 = (int(frame.shape[0]/resizing_factor), int(frame.shape[1]/resizing_factor))
            output_shape1 = (int(frame.shape[0]/resizing_factor), int((4/3) * frame.shape[1]/resizing_factor))

            date = datetime.now()
            time_stamp = date.strftime('%d-%m-%y - %H-%M')
            path = "C:/Users/zabfw3/Documents/Faculdade/TG/generated_videos"
            if not os.path.exists(path):
                os.mkdir(path)
            # out2 = cv.VideoWriter(f'{path}/imgComLinhas ({time_stamp}).mp4',fourcc, 30, frameSize=(output_shape1[1], output_shape1[0]))
            out1 = cv.VideoWriter(f'{path}/Final_panel ({time_stamp}).mp4',fourcc, 30, frameSize=(output_shape1[1], output_shape1[0]))

        print(f"New Video: {video_path}")
        final_func(vid, out1,output_shape1, resizing_factor)
    

out1.release()
# out2.release()
