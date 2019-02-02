# Paweł Borys s14291 
# =============================================================================
import cv2
import numpy as np
import math

#utworzenie obiektu capture (przechwytywanie obrazu), index 0/1/2 wybiera kamere
cap = cv2.VideoCapture(1)
while(cap.isOpened()):
    
    # odczyt obrazu
    # ret to boolean czy jest obraz czy nie, czy wywietla się ramka
    ret, img, = cap.read()
    ret, img2, = cap.read()
    # Mirror kamery
    img= cv2.flip(img, 1)
    
    
    # Do zmiennej grey wgrywamy konwersje RGB do sklai szarosci
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Do zmiennej blur wgrywamy zblurowaną (rozmycie gaussowskie) skalę szarosci, żeby wygładzić wszystkie ostre krawędzie( źródło, rozmiar blur, odchylenie)
    blurred = cv2.GaussianBlur(grey, (35,35), 0)

    # Przypisanie progu adaptacyjnego, tzn jeli wartoć pikseli jest większa niż wartoć progowa przypisuje się jej jedną wartosć
    # (źródło, wartosć progowa, co ma zostać podane 255- biała,styl progowania - tutaj binarnie więc białe albo czarne oraz binaryzacja otsu - usuwa szumy)
    ret, thresh1 = cv2.threshold(blurred, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # wyszukiwanie konturów - użycie trybu retr_tree, który buduje drzewo parent-child dla znalezionych konturów -raz łączenie wsyzstkich punktów w kontur 
    contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # Szukanie największego konturu w obrazie
    cnt = max(contours, key = cv2.contourArea)

    # Tworzenie ograniczającego prostokąta, Czerwony
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # inicjowanie otoczki wypukłej
    hull = cv2.convexHull(cnt)

    # rysowanie konturów - wykorzystanie numpy - (źródło.shape - opencv wyciąga kształt źródła ,używanie zakresu 0-255)
    drawing = np.zeros(img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    # rysowanie otoczki wypukłej - wygładza krzywizny pod kątem wad wypukłosci i koryguje je (true zwraca współrzędne punktów, a false indexy)
    hull = cv2.convexHull(cnt, returnPoints=False)

    # odnadywanie defektów połączeń/wypukłosci - "palców"
    defects = cv2.convexityDefects(cnt, hull)
    # zliczanie
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    # Zwraca tabicę dla defektów w której każdy wiersz zawiera wartoci:
    # pkt początkowy, pkt końcowy, najdalszy punkt, 
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

       # wzory na długoci boków trójkąta
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # zasada na cosinus
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # ignorowanie kątów wiekszych niz 90 stopni i oznaczanie pozostalych
        if angle <= 90:
            count_defects += 1
            cv2.circle(img, far, 1, [0,0,255], -1)
     

        # rysowanie lini łączącej pkt początkowy i końcowy
        cv2.line(img,start, end, [0,255,0], 2)
        

    # definiowanie akcji wg iloci defektów
    # (źródło, tekst, wielkosć czcionki, czcionka, centrowanie)
    if count_defects == 1:
        cv2.putText(img2,"Dwa", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 2:
        cv2.putText(img2,"Trzy", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 3:
        cv2.putText(img2,"Cztery", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 4:
        cv2.putText(img2,"Piec", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    else:
        cv2.putText(img2,"Dziala!", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    # Wyswietlanie okien
    all = np.hstack((drawing, img))
    #cv2.imshow('out', all)
    cv2.imshow('out', img2)
    #cv2.imshow('Pokaz', all)
    cv2.imshow('Prog adaptacyjny', thresh1)

    k = cv2.waitKey(10)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
