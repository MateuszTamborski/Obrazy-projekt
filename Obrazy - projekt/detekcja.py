import cv2
import os
import projekt

IM_WIDTH = 1280
IM_HEIGHT = 720

#zaladowanie przykladów kolorów i figur
path = os.path.dirname(os.path.abspath(__file__))
przykladowe_figury = projekt.load_figury(path + '/ZdjKart/')
przykladowe_kolory = projekt.load_kolory(path + '/ZdjKart/')

images = ['img1', 'img2', 'img4', 'img5', 'img6', 'img7', 'img8', 'img10', 'zdj1', 'zdj2', 'zdj3', 'zdj4']

#dla każdego zdjęcia
for x in images:
    image = cv2.imread('example/' + x + '.jpg')  # wczytaj zdjęcie
    image = cv2.resize(image, (IM_WIDTH, IM_HEIGHT))  # zmien rozmiar zdjecia
    pre_proc = projekt.preprocess_image(image)   # poddaj zdjecie wstepnej obróbce
    contours_sort, contour_is_card = projekt.find_cards(pre_proc) # znajdz i posortuj kontury kart ze zdjecia
    #jeśli są kontury
    if len(contours_sort) != 0:
        cards = []
        k = 0
        #dla każdego wykrytego konturu
        for i in range(len(contours_sort)):
            if contour_is_card[i]:
                # Tworzy obiekt karty z konturu i dołącza do listy 'cards'.
                # Funkcja preprocess_card przyjmuje kontur karty i obraz następnie określa właściwości karty jak np. narożniki.
                # Tworzy spłaszczony obraz karty (200x300) i wydziela kolor i figurę karty ze zdj
                cards.append(projekt.preprocess_card(contours_sort[i], image))
                # Znajdź najlepsze dopasowanie koloru i figury dla karty.
                cards[k].best_figura_match, cards[k].best_kolor_match, cards[k].figura_diff, cards[k].kolor_diff = projekt.match_card(cards[k], przykladowe_figury, przykladowe_kolory)
                # Narysuj rezulatat(kolor i figurę) na zdjeciu.
                image = projekt.draw_results(image, cards[k])
                k = k + 1
        # Narysuj kontury kart na zdjeciu
        if len(cards) != 0:
            contours = []
            for i in range(len(cards)):
                contours.append(cards[i].contour)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    cv2.imwrite('final/final_' + x + '.jpg', image)  # zapis do pliku
    cv2.imshow('Image: ' + x + '.jpg', image)  # wyświetlenie rezulatatów w programie
    cv2.waitKey(0)  # oczekiwanie na wciśnięcie jakiegokolwiek przycisku
    cv2.destroyAllWindows()  # zamknięcie okna z wyświetlanym rezultatem
