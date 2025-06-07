# Rekonstrukcja 3D fasady budynku na podstawie dwóch zdjęć stereo.

Zebrano zdjęcia fasady wydziału mechatroniki oddalone o 1000mm za pomocą jednego aparatu przy próbie jak najlepszego utrzymania kamery w jednej płaszczyźnie.


## 1. Kalibracja kamery `calibrate_camera`

Wykonano 10 zdjęć szachownicy 6x9 wydrukowanej z zasobów biblioteki OpenCV. Dla zapewnienia jej płaskości przytwierdzono ją do kawałka płaskiej deski. Jest to potrzebne, w przeciwnym wypadku uzyskane współczynniki dystorsji nie będą poprawne. Zmierzono odległość między punktami w rzeczywistości: 23.3mm.

W kodzie do kalibracji wykorzystywana jest funkcja `cv2.findChessboardCorners`, która analizuje podawane obrazy i odszukuje na nich szachownice. Na podstawie zbioru kilku obrazów uzyskujemy zbiór punktów `img_points`, gdzie punkty szachownicy były wykryte na obrazie relatywnie do siebie oraz `obj_points`, gdzie punkty szachownicy były wykryte w koordynatach 3D nieruchomych względem wzoru.

Powyższe zbiory punktów podawane są do `cv2.calibrateCamera`. Wynikiem kalibracji jest macierz wewnętrznych parametrów $K$ oraz wektor współczynników dystorsji.
$$ K = \begin{bmatrix} fs_x & 0 & o_x \\ 0 & fs_y & o_y \\ 0 & 0 & 1 \end{bmatrix} $$
gdzie:
- $f$ - ogniskowa
- $s_x, s_y$ - współczynniki odkrzywienia (skew)
- $o_x, o_y$ - offset od środka obrazu

```
Camera Matrix (K):
 [[782.55823012   0.         502.88276877]
 [  0.         781.41437917 379.54572024]
 [  0.           0.           1.        ]]
Distortion Coefficients:
 [[ 0.0825818  -0.16175887 -0.00030915 -0.00279349  0.1557988 ]]
```

## 2. Ładowanie obrazów stereo `load_image`

Zdjęcia uzyskane z telefonu są dosyć wysokiej rozdzielczości, dlatego postanowiłem je przeskalować w dół, aby stworzyć końcową chmurę punktów o przystępnej wielkości (każdy dodatkowy piksel obrazu to dodatkowy punkt w chmurze punktów).

Wykorzystano również `cv2.medianBlur` w celu złagodzenia szumów w obrazie co ma na celu poprawić warunki do dopasowania stereo wykorzystywanego w kolejnych krokach

## 3. Odzniekształcenie obrazu `undistort_image`

Wydobywamy nową macierz K oraz największy możliwy do wykorzystania obszar obrazu ROI po usunięciu zniekształceń funkcją `cv2.getOptimalNewCameraMatrix`. Usuwamy zniekształcenia `cv2.undistort`.

## 4. Dopasowanie punktów obrazów oraz stworzenia mapy dysparytetów

Wykorzystuję algorytm StereoSGBM:

| Parametr          | Wyjaśnienie                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| minDisparity      | Minimalna odległość tego samego piksela między dwoma obrazami, przyjmuję 0.                                                                                                                                                                                                                                                                                                    |
| numDisparities    | Musi być podzielny przez 16. Największa odległosć tego samego piksela odjąć minDisparity.                                                                                                                                                                                                                                                                                      |
| blockSize         | Rozmiar dopasowywanych bloków, nieparzysta wartość najczęściej w zakresie 3.. 11.                                                                                                                                                                                                                                                                                              |
| P1                | Parametr mierzący gładkość dysparytetów.                                                                                                                                                                                                                                                                                                                                       |
| P2                | Parametr mierzący gładkość dysparytetów.                                                                                                                                                                                                                                                                                                                                       |
| disp12MaxDiff     | Maksymalna różnica w pikselach dysparytety lewo-prawo.                                                                                                                                                                                                                                                                                                                         |
Następnie na podstawie obu obrazów przy pomocy `stereo_sgbm.compute` wyliczana jest mapa dysparytetów.

Dokonywana jest rektyfikacja stereo w celu wydobycia macierzy `Q` za pomocą `cv2.stereoRectify`

Przyjmowane są następujące wartości rotacji i translacji pomiędzy obrazami
$$ R = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} $$
$$T =  \begin{bmatrix} -b & 0 & 0\end{bmatrix} $$
gdzie
- $b$ - odległość między kamerą w obrazie lewym i prawym 


## 5. Generacja chmury punktów

Do stworzenia chmury punktów na podstawie mapy dysparytetów jest wykorzystywana biblioteka open3D. Dokonywana jest reprojekcja punktów `o3d.reprojectImageTo3D`, następnie na te punkty nakładana jest maska w celu usunięcia punktów niepoprawnie dobranych - bardzo odległych oraz bardzo bliskich kamerze.

Chmura punktow jest filtrowana i usuwane są statystycznie niepoprawne punkty `o3d.remove_statistical_outlier`. 

Na sam koniec chmura punktów jest wyświetlana oraz zapisywana.


#### Bibliografia
- https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- An Invitation to 3-D Vision From Images to Geometric Models -- Yi Ma, Stefano Soatto, Jana Kosecka, S. Shankar Sastry
- Heiko Hirschmuller. [Stereo processing by semiglobal matching and mutual information](http://www.openrs.org/photogrammetry/2015/SGM%202008%20PAMI%20-%20Stereo%20Processing%20by%20Semiglobal%20Matching%20and%20Mutual%20Informtion.pdf). _Pattern Analysis and Machine Intelligence, IEEE Transactions on_, 30(2):328–341, 2008.
