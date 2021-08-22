# FILTRO PARA ENFATIZAR AS BORDAS DA IMAGEM
# COMBINANDO O IMPULSO DE DIRAC COM O FILTRO PASSA-ALTA
import numpy as np
import cv2 as cv


def edgeFilter(img):
    ker = np.array([
            [-1, -1, -1],
            [-1, 18, -1],
            [-1, -1, -1]])
    ker = (1.0/10.0) * ker
    output = cv.filter2D(img,-1,ker, delta=0)
    return output

img = cv.imread('images/Cars1.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('vanilla', img)
cv.imshow('filtered', filter(img))
cv.waitKey(0)
cv.destroyAllWindows()