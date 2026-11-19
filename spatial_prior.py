"""Mapa położenia oparta na udziale trafień dla jednej klasy obiektów."""
import cv2
import numpy as np

GRID = (9, 16)  # wiersze, kolumny


def pola(srodki):
    """Środki ramek w formacie [x, y], znormalizowane do [0, 1]."""
    kolumny = np.clip((srodki[:, 0] * GRID[1]).astype(int), 0, GRID[1] - 1)
    wiersze = np.clip((srodki[:, 1] * GRID[0]).astype(int), 0, GRID[0] - 1)
    return wiersze, kolumny


def zbuduj_mape(srodki, trafienia):
    """Trafienie to ramka kandydata z IoU ≥ 0,5 względem obiektu danej klasy."""
    wszystkie = np.zeros(GRID)
    poprawne = np.zeros(GRID)
    wspolrzedne = pola(srodki)
    np.add.at(wszystkie, wspolrzedne, 1)
    np.add.at(poprawne, wspolrzedne, trafienia)
    wszystkie = cv2.GaussianBlur(wszystkie, (5, 5), 1)
    poprawne = cv2.GaussianBlur(poprawne, (5, 5), 1)
    srednia = (np.count_nonzero(trafienia) + 1) / (len(trafienia) + 2)
    # Przy małej liczbie kandydatów wracamy do średniej dla całego kadru.
    return (poprawne + 10 * srednia) / (wszystkie + 10)


def ocen_polozenie(mapa, srodki):
    return mapa[pola(srodki)]
