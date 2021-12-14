import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def desenhar_linhas(img, linhas):
  # Obter uma imagem das linhas desenhadas em um fundo preto
  imgLinhas = np.zeros_like(img)

  if linhas is not None:
      for linha in linhas:
          if len(linha) == 4:
              x1, y1, x2, y2 = linha.reshape(4)
              cv.line(imgLinhas, (x1, y1), (x2, y2), (0, 255, 255), 10)
          else:
              print("Erro")
  return imgLinhas


def linhas_medias(img, linhas):
    # Obter as linhas medias de um conjunto de linhas
    altura = img.shape[0]
    largura = img.shape[1]
    esquerda = []
    direita = []
    linha_direita = []
    linha_esquerda = []
    for linha in linhas:
        x1, y1, x2, y2 = linha.reshape(4)
        try:
            parametros = np.polyfit((x1, x2), (y1, y2), 1)  # Retorna a inclinação e a coordenada y
        except:
            pass
        # Parametros da equação da linha (y = ax + b)
        a = parametros[0]
        b = parametros[1]
        if (a < -5 or a > 6 or (a > -0.8 and a < 0.8)):
            print(parametros)
        if a < -0.8 and a > -6 and x1 < largura / 2:  # Se a inclinação for menor do que 0, a inclinação está para a esquerda, se for maior a inclinação está para a direita
            esquerda.append((a, b))
        elif a > 0.8 and a < 6 and x1 > largura / 2:
            direita.append((a, b))
    if len(esquerda) > 0:
        media_valores_esquerda = np.average(esquerda, axis=0)
        linha_esquerda = obter_coordenadas(img, media_valores_esquerda)
    if len(direita) > 0:
        media_valores_direita = np.average(direita, axis=0)
        linha_direita = obter_coordenadas(img, media_valores_direita)
    return np.array([linha_esquerda, linha_direita])

def obter_coordenadas(img, parametros_linha):
    # Obter as coordenadas dos pontos da linha
    # Equação da Linha (y = ax + b)
    a, b = parametros_linha
    y1 = img.shape[0]
    y2 = int(y1 * 0.5)  # Linha média vai até 3/5 da imagem
    x1 = int((y1 - b) / a)
    x2 = int((y2 - b) / a)
    return np.array([x1, y1, x2, y2])



def detectarLinhas(img, limitesCores, blur, largIni, largFinal, alturaCorte):
    altura = img.shape[0]
    largura = img.shape[1]

    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    # img_blur = cv.medianBlur(img_hsv, 125)
    # img_blur = cv.GaussianBlur(img_hsv, (185,265),0)
    img_blur = cv.GaussianBlur(img_hsv, (165,321), 0)

    mascaraFinal = np.zeros(img_blur.shape[:2], dtype="uint8")   #Inicializando a mascara final

    for (lower, upper) in limitesCores:

        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mascara = cv.inRange(img_blur, lower, upper)

        mascaraFinal = cv.bitwise_or(mascaraFinal, mascara)

    res = cv.bitwise_and(img_blur, img_blur, mask=mascaraFinal)
    img_filt = cv.cvtColor(res, cv.COLOR_HSV2RGB)

    # img_blur = cv.medianBlur(img_filt, blur)
    # img_blur = cv.GaussianBlur(mascaraFinal, [5,5])
    # img_canny = cv.Canny(img_blur, 220, 220)
    img_canny = cv.Canny(img_filt, 130, 150)

    cv.imshow("img_filt", img_filt)
    out2.write(img_filt)


    triangulo1 = np.array([(largIni, altura), (largFinal, altura), (largura/2, alturaCorte)])
    # triangulo1 = np.array([(largIni, altura/2), (largFinal, altura/2), (largura/2, alturaCorte)])
    triangulo2 = np.array([(int(largura/3), altura), (int(largura*2/3), altura), (int(largura/2), altura/2)])
    retangulo = np.array([(0, altura), (largura, altura),(largura, 60), (0, 60)])
    mascara1 = np.zeros_like(img_canny)
    mascara2 = np.zeros_like(img_canny)
    mascara3 = np.zeros_like(img_canny)
    cv.fillPoly(mascara1, np.int32([triangulo1]), (255,255,255))
    img_mascarada1 = cv.bitwise_and(img_canny, mascara1)
    cv.fillPoly(mascara2, np.int32([triangulo2]), (255,255,255))
    mascara2 = cv.bitwise_not(mascara2)
    cv.fillPoly(mascara3, np.int32([retangulo]), (255,255,255))
    img_mascaradaFinal = cv.bitwise_and(cv.bitwise_and(img_mascarada1, mascara2), mascara3)
    # img_mascaradaFinal = cv.bitwise_and(img_mascarada1, mascara3)

    cv.imshow("canny final", img_mascaradaFinal)

    linhas = cv.HoughLinesP(img_mascaradaFinal, 2, np.pi / 180, 50, np.array([]), minLineLength=70, maxLineGap=70)

    imgLinhas = desenhar_linhas(img, linhas_medias(img, linhas))


    imgComLinhas = cv.addWeighted(img, 0.8, imgLinhas, 1, 1)  # Adicionando as linhas à imagem original

    return imgComLinhas, imgLinhas, img_filt, img_mascaradaFinal, img_canny, img_blur



#######  Análise Imagem  #######

# img_original = cv.imread('C:/Users/lucas/OpenCV_Projects/Imagens/Plantio/1.jpg')
# # img = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)
# # cv.imshow("Original",img_original)
# img_filt = filtrar_Cor(img_original)
# # img_canny = canny(img_filt)
# # plt.imshow(img_canny), plt.show()
# imgComFaixas = detectar_Faixas(img_filt)
# cv.imshow("canny", imgComFaixas)
# cv.waitKey(0)

# img_original = cv.imread('/content/drive/MyDrive/Colab_Notebooks/Imagens/Plantio/17.jpg')
# img_original = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)
# # img_original = cv.medianBlur(img_original,11)
# plt.imshow(img_original, cmap='gray') , plt.show()
# # histograma(img_original)


#######  Análise Vídeo  #######

vid = cv.VideoCapture("C:/Users/lucas/Google Drive (lucas.rodrigues@unesp.br)/Faculdade/TG/video_analisado.mp4")
imgLinhasAnterior = None
(grabbed, frame) = vid.read()
fatorResize = 3.5
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out1 = cv.VideoWriter('imgComLinhas.mp4',fourcc, 30, (int(frame.shape[1]/fatorResize),int(frame.shape[0]/fatorResize)))
out2 = cv.VideoWriter('imgFiltrada.mp4',fourcc, 30, (int(frame.shape[1]/fatorResize),int(frame.shape[0]/fatorResize)))

while(vid.isOpened()):
    sucess, frame = vid.read()
    frame = cv.resize(frame, (int(frame.shape[1]/fatorResize),int(frame.shape[0]/fatorResize)))
    altura = frame.shape[0]
    largura = frame.shape[1]
    fps = None
    limitesCores = [([60, 70, 50], [100, 240, 240])]; blur = 51; largIni = -20; largFinal = largura + 20; alturaCorte = -50  # Parametros Vid 2
    # limitesCores = [([60, 70, 50], [100, 240, 240])]; blur = 21; largIni = 80; largFinal = largura - 80; alturaCorte = -50  # Parametros Vid 4

    if cv.waitKey(3) & 0xFF == 27 or frame is None:
        break
    try:
        imgComLinhas, imgLinhas, img_filt, img_mascarada, img_canny, img_blur = detectarLinhas(frame, limitesCores, blur, largIni, largFinal, alturaCorte)
        imgLinhasAnterior = imgComLinhas.copy()

        # imgComLinhas = detectarLinhas(frame)
        cv.imshow("imgComLinhas",imgComLinhas)

        out1.write(imgComLinhas)
    except:
        if imgLinhasAnterior is not None:
            imgComLinhas = cv.addWeighted(frame, 0.8, imgLinhas, 1, 1)  # Adicionando as linhas à imagem original
            cv.imshow("imgComLinhas",imgComLinhas)
            out1.write(imgComLinhas)
        else:
            cv.imshow("imgComLinhas",frame)
            out1.write(frame)

        # pass
    # cv.waitKey(1)

cv.destroyAllWindows()
vid.release()
out1.release()
out2.release()