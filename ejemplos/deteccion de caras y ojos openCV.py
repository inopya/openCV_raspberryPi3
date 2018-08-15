# -*- coding: cp1252 -*-

#         _\|/_
#         (O-O)
# -----oOO-(_)-OOo----------------------------------------------------


#######################################################################
# ******************************************************************* #
# *                                                                 * #
# *                   Autor:  Eulogio López Cayuela                 * #
# *                                                                 * #
# *        ejemplos openCV ---  deteccion de rostros y ojos         * #
# *                                                                 * #
# *                  Versión 1.0   Fecha: 12/08/2018                * #
# *                                                                 * #
# ******************************************************************* #
#######################################################################




#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
#   IMPORTACION DE MODULOS
#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm

import cv2
import numpy as np


#definir los clasificadores haar a utilizar
detector_rostros = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
detector_ojos = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

#bandera para activar o desactivar la deteccion de ojos sobre los rostros detectados
FLAG_detectar_ojos = False

# inicializar la camara y tomar una muestra
video = cv2.VideoCapture(0)


#solo si la camara esta disponible para su uso se entra en el bucle de captura y deteccion
if video.isOpened()==True:

    #definimos la ventana en al que mostras la imagen de la camara y las detecciones si las hubiese
    cv2.namedWindow("Camara + detecciones", cv2.WINDOW_AUTOSIZE)

    while video.isOpened():
        #captura de un fotograma
        _, fotograma_original = video.read()
        #convertir a escala de gris
        fotograma_gris = cv2.cvtColor(fotograma_original, cv2.COLOR_BGR2GRAY)

        #buscar rostros
        rostros = detector_rostros.detectMultiScale(fotograma_gris, 1.3, 5)
        for (x,y,w,h) in rostros:
            #marcar los detecciones de rostros si los hay
            cv2.rectangle(fotograma_original,(x,y),(x+w,y+h),(255,0,0),2)
            #crear una subimagen con la porcion correspondiente al rostro 
            if FLAG_detectar_ojos == True:
                roi_gris = fotograma_gris[y:y+h, x:x+w]         #subimagen con al region de interes correspondiente al rostro converida a gris para la busqueda de ojos
                roi_color = fotograma_original[y:y+h, x:x+w]    #la misma porcion de imagen en color para marcar sobre ella las detecciones de los ojos
                #busqueda de ojos dentro de los rostros
                ojos = detector_ojos.detectMultiScale(roi_gris)
                for (ex,ey,ew,eh) in ojos:
                    #marcar las detecciones de ojos en los rostros
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        #mostrar la imagen capturada por la camara con sus posibles detecciones 
        cv2.imshow("Camara + detecciones", fotograma_original)

        #control de pulsacion de teclado
        pulsacion_teclado = cv2.waitKey(1) & 0xFF
        #la tecla ESC que permite salir del programa 
        if pulsacion_teclado == 27:
            #se cierran la ventana y se desconecta la camara
            video.release()
            cv2.destroyAllWindows()
            break

