# ps_pycuda_speed_test
Ein Projekt für den Parallele Systeme Kurs der Hochschule Karlsruhe

## Inhaltsverzeichniss
1. [Allgemeine Informationen](#allgemeine-informationen)
2. [Verwendete Technologien](#verwendete-technologien)
3. [Installation](#installation)
4. [Mandelbrot Visualizer] (#Visualizer)
### Allgemeine Informationen
***
Das Projekt dient dem Vergleichen der Geschwindigkeit von Berechnungen mit Cuda und herkömmlichen CPU Berechnungen.
Hierfür werden zum einen komplexe Cuda Berechnungen mit einfacheren Verglichen, als auch die berechnung der Mandelbrotmenge mit CUDA und CPU.
### Verwendete Technologien
***
Lite der verwendeten Tools:
* Pycharm Professional 2021.2.2
* Cuda 3.2 Toolkit
* VS 2019
* NVIDIA GeForce GTX 1660 Ti
### Installation
***
im Installations Guide von PyCuda sollten alle Notwendigen Tolls installiert werden
https://wiki.tiker.net/PyCuda/Installation/

### Mandelbrot Visualizer
***
Dieses Programm dient der visualisierung der Mandelbrot Menge. Shortcuts:
* F12 Screenshot des momentanen Fensters
* W Zoom in
* S Zoom out
* Arrow keys Bewegen des momentanen Bildausschnittes
* ESC Quit

Bei zu starkem zoom wird das Bild sehr verpixelt, da die Zahlen für Python zu klein werden. mpmath ist leider zum momentanen Zeipunkt nicht mit numba kompatibel.
