import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from os import listdir
import os
import warnings
warnings.simplefilter('ignore', np.RankWarning)

def takeSecond(list):
  return list[1]

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

                cv.fillPoly(imgLinhas, np.int32([points]), (255,0,0))
            for line in line_pair:
                if len(line) == 4:
                    x1, y1, x2, y2 = line.reshape(4)
                    cv.line(imgLinhas, (x1, y1), (x2, y2), color, 10)
                else:
                    print("Erro")
    return imgLinhas

def drawCenterPoint(img, all_x_center_points, n_points=5):
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
    if len(all_x_center_points) == 0:
        return img, None
    x_center = int(np.mean(all_x_center_points[-n_points:]))
    y_center = int(0.9 * img.shape[0])
    img_with_center = cv.circle(img, 
              center=(x_center,y_center), 
              radius=0,
              color=(0,0,255),
              thickness=8)
    return img_with_center, x_center

def calc_linhas_medias(img, linhas, aprox=40, n_lines=3):
    '''
    Obter as linhas medias de um conjunto de linhas
    '''
    altura = img.shape[0]
    largura = img.shape[1]
    esquerda = []
    direita = []
    linha_direita = []
    linha_esquerda = []
    considered_left_lines = []
    considered_rigth_lines = []

    for linha in linhas:
        x1, y1, x2, y2 = linha.reshape(4)
        try:
            parametros = np.polyfit((x1, x2), (y1, y2), 1)  # Retorna a inclinação e a coordenada y
        except:
            pass
        # Parametros da equação da linha (y = ax + b)
        a = parametros[0]
        b = parametros[1]
        #if (a < -5 or a > 6 or (a > -0.8 and a < 0.8)):
            #print(parametros)
        if a < -0.8 and a > -6 and x1 < largura / 2:  # Se a inclinação for menor do que 0, a inclinação está para a esquerda, se for maior a inclinação está para a direita
            esquerda.append((a, b))
        elif a > 0.8 and a < 6 and x1 > largura / 2:
            direita.append((a, b))
    
    esquerda.sort(key=takeSecond, reverse=True)
    direita.sort(key=takeSecond, reverse=False)

    # imgLinhas = draw_lines(img, np.array([esquerda, direita]), color=(0,255,255))
    
    # imgComLinhas = cv.addWeighted(img, 0.8, imgLinhas, 1, 1)

    if len(esquerda) > 0:
        considered_left_lines = [obter_coordenadas(img, line_esq) for line_esq in esquerda[:n_lines]]
        media_valores_esquerda = np.average(esquerda[:n_lines], axis=0)
        linha_esquerda = obter_coordenadas(img, media_valores_esquerda)
        linha_esquerda[0] += aprox
        linha_esquerda[2] += aprox
    if len(direita) > 0:
        considered_rigth_lines = [obter_coordenadas(img, line_dir) for line_dir in direita[:n_lines]]
        media_valores_direita = np.average(direita[:n_lines], axis=0)
        linha_direita = obter_coordenadas(img, media_valores_direita)
        linha_direita[0] -= aprox
        linha_direita[2] -= aprox

    
    
    return np.array([linha_esquerda, linha_direita]), np.array([considered_left_lines, considered_rigth_lines])

def obter_coordenadas(img, parametros_linha):
    # Obter as coordenadas dos pontos da linha
    # Equação da Linha (y = ax + b)
    a, b = parametros_linha
    y1 = img.shape[0]
    y2 = int(y1 * 0.5)  # Linha média vai até 3/5 da imagem
    x1 = int((y1 - b) / a)
    x2 = int((y2 - b) / a)
    return np.array([x1, y1, x2, y2])


def detectarLinhas(img, limitesCores, blur, largIni, largFinal, alturaCorte, avg_lines, all_x_centers_points):
    altura = img.shape[0]
    largura = img.shape[1]

    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    # img_blur = cv.medianBlur(img_hsv, 125)
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

    img_blur = cv.GaussianBlur(img_segmented, (45, 45), 0)
    img_canny = cv.Canny(img_blur.astype(np.uint8), 25, 35)

    lines = cv.HoughLinesP(img_canny, 
                                2, 
                                np.pi / 180, 
                                50, 
                                np.array([]), 
                                minLineLength = 40, 
                                maxLineGap = 60)

    
    final_lines, considered_lines = calc_linhas_medias(img, lines)
    imgLines = draw_lines(img, considered_lines, color=(0,255,255))
    imgFinalLines = draw_lines(img, final_lines, color=(0,255,255), average_lines=True)
    
    #imgComLinhas = cv.addWeighted(img, 0.8, imgLinhas, 1, 1)  # Adicionando as linhas à imagem original
    imgWithLines = cv.addWeighted(img, 0.6, imgLines, 1, 1)
    if np.max(avg_lines) > 0:
        imgLinhasAvg = draw_lines(img, avg_lines, color=(0,255,255), average_lines=True)
        # imgComLinhas = cv.addWeighted(imgComLinhas, 0.8, imgLinhasAvg, 1, 1)  # Adicionando as linhas à imagem original
        imgWithFinalLines = cv.addWeighted(img, 0.6, imgLinhasAvg, 1, 1)
    else:
        imgWithFinalLines = cv.addWeighted(img, 0.6, imgFinalLines, 1, 1)

    finalImage, x_center = drawCenterPoint(imgWithFinalLines, all_x_centers_points)
    return finalImage, imgWithLines, final_lines, imgFinalLines, img_segmented, img_canny, img_blur


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

def build_panel(img_final, img_segmented=None, img_canny=None, img_with_all_lines=None, type=1):

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
def final_func(vid, out1, out2, output_shape1, output_shape2, resize_factor):

    imgLinhasAnterior = None
    n_avg_lines = 10
     
    avg_lines = np.zeros((2,4))
    all_left_lines = [[],[],[],[]]
    all_right_lines = [[],[],[],[]]
    all_x_centers_points = []


    while(vid.isOpened()):
        (sucess, frame) = vid.read()
        if not sucess:
            break
        frame = cv.resize(frame, (int(frame.shape[1]/resize_factor),int(frame.shape[0]/resize_factor)))
        altura = frame.shape[0]
        largura = frame.shape[1]
        fps = None
        limitesCores = [([35, 0, 0], [100, 255, 255])]; blur = 65; largIni = -20; largFinal = largura + 20; alturaCorte = -50  # Parametros Vid 2
        # limitesCores = [([60, 70, 50], [100, 240, 240])]; blur = 51; largIni = -20; largFinal = largura + 20; alturaCorte = -50  # Parametros Vid 2
        # limitesCores = [([60, 70, 50], [100, 240, 240])]; blur = 21; largIni = 80; largFinal = largura - 80; alturaCorte = -50  # Parametros Vid 4

        if cv.waitKey(3) & 0xFF == 27 or frame is None:
            break
        try:
            imgComLinhas, img_with_all_lines, final_lines, imgFinalLines, img_segmented, img_canny, img_blur = detectarLinhas(frame, limitesCores, blur, largIni, largFinal, alturaCorte, avg_lines, all_x_centers_points)
            imgLinhasAnterior = imgFinalLines.copy()

            final_panel = build_panel(imgComLinhas, img_segmented, img_canny, img_with_all_lines)
            cv.imshow("Final Panel", final_panel)

            out1.write(resize_frame(imgComLinhas, output_shape1))
            #out1.write(imgComLinhas)
            out2.write(resize_frame(final_panel, output_shape2))

            for i, coord in enumerate(final_lines[0]):
                all_left_lines[i].append(coord)
            for i, coord in enumerate(final_lines[1]):
                all_right_lines[i].append(coord)

            for i in range(4):
                avg_lines[0,i] = np.mean(all_left_lines[i][-n_avg_lines:])
                avg_lines[1,i] = np.mean(all_right_lines[i][-n_avg_lines:])
            avg_lines = avg_lines.astype(int)
            x_center = int(avg_lines[0,0] + (avg_lines[1,0] - avg_lines[0,0])/2)
            all_x_centers_points.append(x_center)

        except:
            if imgLinhasAnterior is not None:
                imgComLinhas = cv.addWeighted(frame, 0.6, imgFinalLines, 1, 1)  # Adicionando as linhas à imagem original
                
                #cv.imshow("imgComLinhasMedias",imgComLinhas)
                out1.write(resize_frame(imgComLinhas, output_shape1))
                #out1.write(imgComLinhas)
                final_panel = build_panel(imgComLinhas)
                out2.write(resize_frame(final_panel, output_shape2))
            else:
                #cv.imshow("imgComLinhasMedias",frame)
                #out1.write(resize_frame(frame, output_shape1))
                final_panel = build_panel(frame)
                out2.write(resize_frame(final_panel, output_shape2))
                
                out1.write(frame)

            # pass
        # cv.waitKey(1)

    cv.destroyAllWindows()
    vid.release()
    return

resizing_factor = 3
fourcc = cv.VideoWriter_fourcc(*'mp4v')
folder = "C:/Users/zabfw3/Documents/Faculdade/TG/TG/Videos_Castanho/Milho/VideosBons"
out1 = None
out2 = None
for video_path in listdir(folder):
    if not os.path.isfile(folder + "/" + video_path):
        continue
    vid = cv.VideoCapture(folder + "/" + video_path)

    if out1 == None:
        frame = vid.read()[1]
        output_shape1 = (int(frame.shape[0]/resizing_factor), int(frame.shape[1]/resizing_factor))
        output_shape2 = (int(frame.shape[0]/resizing_factor), int((4/3) * frame.shape[1]/resizing_factor))

        date = datetime.now()
        time_stamp = date.strftime('%d-%m-%y - %H-%M')
        path = "C:/Users/zabfw3/Documents/Faculdade/TG/generated_videos"
        if not os.path.exists(path):
            os.mkdir(path)
        out1 = cv.VideoWriter(f'{path}/imgComLinhas ({time_stamp}).mp4',fourcc, 30, frameSize=(output_shape1[1], output_shape1[0]))
        out2 = cv.VideoWriter(f'{path}/Final_panel ({time_stamp}).mp4',fourcc, 30, frameSize=(output_shape2[1], output_shape2[0]))

    print(f"Vídeo: {video_path}")
    final_func(vid, out1, out2, output_shape1, output_shape2, resizing_factor)

out1.release()
out2.release()