import numpy as np
import cv2

#czcionka napisów na finalnych zdjęciach
czcionka = cv2.FONT_HERSHEY_COMPLEX

#threshold
BACKGROUND_THRESH = 60
CARD_THRESH = 30

#szerokość i wysokość rogu karty gdzie znajduje się figura i kolor
CORNER_HEIGHT = 84
CORNER_WIDTH = 32

#wymiary przykladowych zdj kolorów
KOLOR_WIDTH = 70
KOLOR_HEIGHT = 100

#wymiary przykladowych zdj figur
FIGURA_WIDTH = 70
FIGURA_HEIGHT = 125

#maks i min powierzchnia karty
CARD_MAX_AREA = 320000
CARD_MIN_AREA = 25000

#maks różnica figur i kolorów
FIGURA_DIFF_MAX = 2000
KOLOR_DIFF_MAX = 700

class Badana_karta:
    # Klasa przechowująca informacje o badanych kartach ze zdj
    def __init__(self):
        self.contour = []  # kontur karty
        self.width, self.height = 0, 0  # wymiary karty
        self.corners = []  # narożniki karty
        self.center = []  # punkt centralny karty
        self.perspective = []  # spłaszczony, szary i rozmyty obraz 200x300
        self.figura_img = []  # obraz figury karty
        self.kolor_img = []  # obraz koloru karty
        self.best_figura_match = "Nieznana"  # najlepiej dopasowana figura
        self.best_kolor_match = "Nieznana"  # najlepiej dopasowany kolor
        self.figura_diff = 0  # różnica pomiędzy zdj figury i najlepiej dopasowanym zdjęciem przykladowej figury
        self.kolor_diff = 0  # różnica pomiędzy zdj koloru i najlepiej dopasowanym zdjęciem przykładowego koloru

class Przykladowe_figury:
    # Klasa przechowująca informacje o zdjęciach przykładowych figur
    def __init__(self):
        self.img = []
        self.name = "Tymczasowe"

class Przykładowe_kolory:
    # Klasa przechowująca informacje o zdjęciach przykładowych kolorów
    def __init__(self):
        self.img = []
        self.name = "Tymczasowe"


def load_figury(filepath):
    """Funkcja ładująca zdjęcia figur. Są przechowywane jako lista obiektów klasy Train_ranks"""
    przykladowe_figury = []
    i = 0
    for Figura in ['As', 'Dwojka', 'Trojka', 'Czworka', 'Piatka', 'Szostka', 'Siodemka',
                 'Osemka', 'Dziewiatka', 'Dziesiatka', 'Walet', 'Dama', 'Krol']:
        przykladowe_figury.append(Przykladowe_figury())
        przykladowe_figury[i].name = Figura
        filename = Figura + '.jpg'
        przykladowe_figury[i].img = cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1
    return przykladowe_figury

def load_kolory(filepath):
    """Funkcja ładująca zdjęcia kolorów. Są przechowywane jako lista obiektów klasy Train_suits"""
    przykladowe_kolory = []
    i = 0
    for Kolor in ['Pik', 'Karo', 'Trefl', 'Kier']:
        przykladowe_kolory.append(Przykładowe_kolory())
        przykladowe_kolory[i].name = Kolor
        filename = Kolor + '.jpg'
        przykladowe_kolory[i].img = cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1
    return przykladowe_kolory

def preprocess_image(image):
    """Funkcja zwraca szare,rozmyte filtrem gaussowskim i thresholdowane zdj"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Użyta została metoda adaptacyjnego progu aby jak najbardziej uniezależnić wykrywanie kart od poziomu światła
    # Pobierany jest piksel tła z górnej, środkowej części zdjęcia i na jego podstawie określana jest intensywność światła.
    # Próg jest ustawiany o BKG_THRESH więcej niż pobrany piksel.
    img_width, img_height = np.shape(image)[:2]
    background = gray[int(img_height / 100)][int(img_width / 2)]
    thresh_level = background + BACKGROUND_THRESH
    returnvalue, thresh = cv2.threshold(blurred, thresh_level, 255, cv2.THRESH_BINARY)
    return thresh

def find_cards(thresh_image):
    """Funkcja znajduje wszystkie kontury o wymiarach kart na zdjęciu i zwraca liczbę kart i posortowaną
    od największego do najmniejszego listę konturów kart"""

    # znajdź kontury, posortuj wg rozmiarów konturów
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]), reverse=True)

    # jeśli brak konturów
    if len(contours) == 0:
        return [], []

    contours_sort = []
    hierarchy_sort = []
    contour_is_card = np.zeros(len(contours), dtype=bool)

    # Wypełnij puste listy posortowanymi konturami i hierarchiami. Indeksy listy konturów odpowiadają indeksom listy hierarchii.
    # Macierz hierarchii może być użyta aby sprawdzić czy kontury mają rodziców
    for i in index_sort:
        contours_sort.append(contours[i])
        hierarchy_sort.append(hierarchy[0][i])

    # Sprawdź które kontury to karty wg kryteriów:
    # 1) mniejsza powierzchnia niż maks rozmiary karty i większa powierzchnia niż min rozmiar karty
    # 2) brak rodziców,
    # 3) 4 narożniki
    for i in range(len(contours_sort)):
        area = cv2.contourArea(contours_sort[i])
        perimeter = cv2.arcLength(contours_sort[i], True)
        approximation = cv2.approxPolyDP(contours_sort[i], 0.01 * perimeter, True)
        if ((area < CARD_MAX_AREA) and (area > CARD_MIN_AREA)
                and (hierarchy_sort[i][3] == -1) and (len(approximation) == 4)):
            contour_is_card[i] = True
    return contours_sort, contour_is_card

def preprocess_card(contour, image):
    """Funkcja używa konturu aby znaleźć informacje o badanej karcie. Wyodrębnia obrazy koloru i figury z karty"""

    # inicjalizacja nowego obiektu klasy Badana_karta
    badana_Karta = Badana_karta()
    badana_Karta.contour = contour

    # Znajdz obwód karty aby aproksymować punkty narożników
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    points = np.float32(approximation)
    badana_Karta.corners = points

    # Znajdz wysokość i szerokość prostokąta ograniczającego kartę
    x, y, w, h = cv2.boundingRect(contour)
    badana_Karta.width, badana_Karta.height = w, h

    # Znajdz punkt centralny karty (średnia z x i y czterech narożników)
    mean = np.sum(points, axis=0) / len(points)
    central_x = int(mean[0][0])
    central_y = int(mean[0][1])
    badana_Karta.center = [central_x, central_y]

    # przekształć kartę w spłaszczony obraz 200x300 przy użyciu transormacji perspektywy
    badana_Karta.perspective = perspective_change(image, points, w, h)

    # 4-krotnie przybliż narożnik przekształconego obrazu karty
    Card_corner = badana_Karta.perspective[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    Card_corner_zoom = cv2.resize(Card_corner, (0, 0), fx=4, fy=4)

    # Pobierz biały pixel karty, w celu określenia wartości progowej
    white_level = Card_corner_zoom[15, int((CORNER_WIDTH * 4) / 2)]
    thresh_level = white_level - CARD_THRESH

    if thresh_level <= 0:
        thresh_level = 1
    returnvalue, card_thresh = cv2.threshold(Card_corner_zoom, thresh_level, 255, cv2.THRESH_BINARY_INV)

    # Podziel obraz na dwie połowy (górna pokaże figurę, dolna kolor)
    Card_figura = card_thresh[20:185, 0:128]
    Card_kolor = card_thresh[186:336, 0:128]

    # Znajdź kontur i prostokąt ograniczający figury, wyodrębnij i znajdź największy kontur
    Card_figura_contours, hierarchy = cv2.findContours(Card_figura, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Card_figura_contours = sorted(Card_figura_contours, key=cv2.contourArea, reverse=True)

    # Znajdz prostokąt ograniczający największy kontur aby zmienić rozmiar badanej figury na taki sam jak przykładowej figury
    if len(Card_figura_contours) != 0:
        x1, y1, w1, h1 = cv2.boundingRect(Card_figura_contours[0])
        Card_figura_roi = Card_figura[y1:y1 + h1, x1:x1 + w1]
        Card_figura_resized = cv2.resize(Card_figura_roi, (FIGURA_WIDTH, FIGURA_HEIGHT), 0, 0)
        badana_Karta.figura_img = Card_figura_resized

    # Podobnie dla koloru
    Card_kolor_contours, hierarchy = cv2.findContours(Card_kolor, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Card_kolor_contours = sorted(Card_kolor_contours, key=cv2.contourArea, reverse=True)

    if len(Card_kolor) != 0:
        x2, y2, w2, h2 = cv2.boundingRect(Card_kolor_contours[0])
        Card_kolor_roi = Card_kolor[y2:y2 + h2, x2:x2 + w2]
        Card_kolor_resized = cv2.resize(Card_kolor_roi, (KOLOR_WIDTH, KOLOR_HEIGHT), 0, 0)
        badana_Karta.kolor_img = Card_kolor_resized

    return badana_Karta

def match_card(badana_Karta, przykladowe_figury, przykladowe_kolory):
    """Funkcja znajduje najlepiej dopasowaną figurę i kolor dla badanej karty.
       Oblicza różnice pomiędzy figurą i kolorem badanej karty a przykładowymi figurami i kolorami -
       najlepszym dopasowaniem jest figura/kolor, która ma najmniejszą różnicę."""

    best_figura_match_diff = 10000
    best_kolor_match_diff = 10000
    best_figura_match_name = "Nieznana"
    best_kolor_match_name = "Nieznana"
    i = 0

    # Jeśli nie znaleziono konturu karty w funkcji preprocess_card
    # rozmiar obrazu jest równy 0 - pomijamy proces szukania różnic
    if (len(badana_Karta.figura_img) != 0) and (len(badana_Karta.kolor_img) != 0):
        # Szukaj najmniejszej różnicy pomiędzy figurą badanej karty a przykładowymi zdj figur
        for Figura in przykladowe_figury:
            diff_img = cv2.absdiff(badana_Karta.figura_img, Figura.img)
            figura_diff = int(np.sum(diff_img) / 255)

            if figura_diff < best_figura_match_diff:
                best_figura_match_diff = figura_diff
                best_figura_name = Figura.name

        # Tak samo dla koloru
        for Kolor in przykladowe_kolory:

            diff_img = cv2.absdiff(badana_Karta.kolor_img, Kolor.img)
            kolor_diff = int(np.sum(diff_img) / 255)

            if kolor_diff < best_kolor_match_diff:
                best_kolor_match_diff = kolor_diff
                best_kolor_name = Kolor.name

    # Połącz najlepsze dopasowanie figury i koloru żeby zidentyfikować badaną kartę.
    # Jeśli dopasowanie figury/koloru ma zbyt dużą wartość różnicy - pozostaje 'Nieznane'
    if (best_figura_match_diff < FIGURA_DIFF_MAX):
        best_figura_match_name = best_figura_name

    if (best_kolor_match_diff < KOLOR_DIFF_MAX):
        best_kolor_match_name = best_kolor_name

    return best_figura_match_name, best_kolor_match_name, best_figura_match_diff, best_kolor_match_diff


def draw_results(image, badana_Karta):
    """Rysuje nazwę karty i kontur"""

    x = badana_Karta.center[0]
    y = badana_Karta.center[1]

    figura_name = badana_Karta.best_figura_match
    kolor_name = badana_Karta.best_kolor_match

    # Podwójnie rysuje imię, aby litery miały czarny kontur
    cv2.putText(image, figura_name, (x - 65, y - 15), czcionka, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, figura_name, (x - 65, y - 15), czcionka, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(image, kolor_name, (x - 65, y + 20), czcionka, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, kolor_name, (x - 65, y + 20), czcionka, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return image

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
    target_array = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], np.float32)
    M = cv2.getPerspectiveTransform(rectangle, target_array)
    perspective = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    perspective = cv2.cvtColor(perspective, cv2.COLOR_BGR2GRAY)
    return perspective