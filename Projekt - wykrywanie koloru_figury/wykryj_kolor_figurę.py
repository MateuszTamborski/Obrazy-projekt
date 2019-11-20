import os
import cv2
import numpy as np

img_path = os.path.dirname(os.path.abspath(__file__)) + '\ZdjKart\\'

FIGURA_WIDTH = 70
FIGURA_HEIGHT = 125

KOLOR_WIDTH = 70
KOLOR_HEIGHT = 100

#przełącz z wyodrębniania figur na wyodrębnianie kolorów
figura_or_kolor = 1

def perspective_change(image, points, width, height):
    """Spłaszcza obraz karty do perspektywy rzędowej (obiekty bliskie znajdują się na dole, obiekty dalsze - na górze)
     o rozmiarze 200x300"""
    rectangle = np.zeros((4, 2), dtype="float32")

    s = np.sum(points, axis=2)

    topleft = points[np.argmin(s)]
    bottomright = points[np.argmax(s)]

    diff = np.diff(points, axis=-1)
    topright = points[np.argmin(diff)]
    bottomleft = points[np.argmax(diff)]

    # Tworzy macierz punktów w kolejności:
    # [górny-lewy, górny-prawy, dolny-prawy, dolny-lewy] przed transformacją

    if width <= 0.8 * height:  # jeśli karta jest w pionie
        rectangle[0] = topleft
        rectangle[1] = topright
        rectangle[2] = bottomright
        rectangle[3] = bottomleft

    if width >= 1.2 * height:  # jeśli karta jest w poziomie
        rectangle[0] = bottomleft
        rectangle[1] = topleft
        rectangle[2] = topright
        rectangle[3] = bottomright

    if width > 0.8 * height and width < 1.2 * height:  # jeśli karta jest ukośnie
        # jeśli najdalszy lewy punkt jest wyżej niż najdalszy prawy punkt,
        # karta przechylona w lewo
        if points[1][0][1] <= points[3][0][1]:
            # jeśli karta przechylona w lewo, approxPolyDP zwraca punkty w kolejności
            # górny-prawy, górny-lewy, dolny-lewy, dolny-prawy
            rectangle[0] = points[1][0]  # górny-lewy
            rectangle[1] = points[0][0]  # górny-prawy
            rectangle[2] = points[3][0]  # dolny-prawy
            rectangle[3] = points[2][0]  # dolny-lewy

        # jeśli najdalszy lewy punkt jest niżej niż najdalszy prawy punkt,
        # karta przechylona w prawo
        if points[1][0][1] > points[3][0][1]:
            # jeśli karta przechylona w prawo, approxPolyDP zwraca punkty w kolejności
            # górny-lewy, dolny-lewy, dolny-prawy, górny-prawy
            rectangle[0] = points[0][0]  # górny-lewy
            rectangle[1] = points[3][0]  # górny-prawy
            rectangle[2] = points[2][0]  # dolny-prawy
            rectangle[3] = points[1][0]  # dolny-lewy

    maxWidth = 200
    maxHeight = 300

    # Utwórz macierz docelową, oblicz macierz transformacji perspektywy i przekształć obraz karty
    targetArray = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], np.float32)
    M = cv2.getPerspectiveTransform(rectangle, targetArray)
    perspective = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    perspective = cv2.cvtColor(perspective, cv2.COLOR_BGR2GRAY)
    return perspective


for Name in ['As', 'Dwojka', 'Trojka', 'Czworka', 'Piatka', 'Szostka', 'Siodemka',
                 'Osemka', 'Dziewiatka', 'Dziesiatka', 'Walet', 'Dama', 'Krol', 'Pik', 'Karo', 'Trefl', 'Kier']:

    filename = Name + '.jpg'
    image = cv2.imread(img_path+filename)
    cv2.imshow("Card", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    returnvalue, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(contours) == 0:
        print('Nie znaleziono konturów!')
        quit()

    card = contours[0]

    perimeter = cv2.arcLength(card, True)
    approximation = cv2.approxPolyDP(card, 0.01 * perimeter, True)
    points = np.float32(approximation)

    x, y, w, h = cv2.boundingRect(card)

    perspective = perspective_change(image, points, w, h)

    corner = perspective[0:84, 0:32]
    corner_zoom = cv2.resize(corner, (0, 0), fx=4, fy=4)
    corner_blur = cv2.GaussianBlur(corner_zoom, (5, 5), 0)
    returnvalue, corner_thresh = cv2.threshold(corner_blur, 155, 255, cv2.THRESH_BINARY_INV)

    # wyodrębnij figurę lub kolor
    if figura_or_kolor <= 13:  # figury
        figura = corner_thresh[20:185, 0:128]
        figura_contours, hierarchy = cv2.findContours(figura, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        figura_contours = sorted(figura_contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(figura_contours[0])
        figura_roi = figura[y:y + h, x:x + w]
        figura_resized = cv2.resize(figura_roi, (FIGURA_WIDTH, FIGURA_HEIGHT), 0, 0)
        final_img = figura_resized

    if figura_or_kolor > 13:  # kolory
        kolor = corner_thresh[186:336, 0:128]
        kolor_contours, hierarchy = cv2.findContours(kolor, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        kolor_contours = sorted(kolor_contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(kolor_contours[0])
        kolor_roi = kolor[y:y + h, x:x + w]
        kolor_resized = cv2.resize(kolor_roi, (KOLOR_WIDTH, KOLOR_HEIGHT), 0, 0)
        final_img = kolor_resized

    cv2.imshow("Image", final_img)

    # zapis (wcisnij s)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('s'):
        cv2.imwrite(img_path + filename, final_img)

    figura_or_kolor = figura_or_kolor + 1
