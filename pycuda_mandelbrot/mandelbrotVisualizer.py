from time import time
import sys
import numpy as np
import pygame
import numba as nb
from numba import cuda
import math
import ctypes
#import seaborn
import mpmath as pm


@cuda.jit(device=True)
def calcMandel(c):
    Z = complex(0, 0)
    for i in range(0, MAX_ITER):
        Z = Z * Z + c
        if abs(Z) > UPPER_BOUND:
            return i + 1 - math.log(math.log2(abs(Z)))
    # for i in range(MAX_ITER):
    return 0


@cuda.jit
def render_gpu(image, input_matrix):

    i = cuda.grid(1)
    width = image.shape[0]
    height = image.shape[1]
    if i < width * height:
        x = np.int32(i / height)
        y = np.int32(i % height)

        v = calcMandel(input_matrix[x, y])

        #image[x, y, 0] = 1 / 4 * v % 255
        #image[x, y, 1] = 1 / 5 * v % 255
        #image[x, y, 2] = 1 / 7 * v % 255
        image[x, y, :] = 5 * v % 255
    return


def calcMandel_cpu(c):
    z = 0.0j
    for i in range(MAX_ITER):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= UPPER_BOUND:
            return i
    # for i in range(MAX_ITER):
    return 0


def render_cpu(image, input_Matrix):
    x = input_Matrix.shape[0]
    y = input_Matrix.shape[1]
    for i in range(x):
        for j in range(y):
            v = calcMandel_cpu(input_Matrix[i, j])
            image[i, j, :] = v
    return


def display_update_mandel():
    global surface
    x_min = start_point[0] - radius
    x_max = start_point[0] + radius
    y_min = start_point[1] - radius
    y_max = start_point[1] + radius
    display.fill((0, 0, 0))
    render_time = 0
    real_Teil = np.matrix(pm.linspace(x_min, x_max, HEIGHT), dtype=np.complex64)
    imaginaer_Teil = np.matrix(pm.linspace(y_max, y_min, WIDTH), dtype=np.complex64) * 1j
    input_Matrix = np.array(real_Teil + imaginaer_Teil.transpose(), dtype=np.complex64)
    if mode == 0:
        t1 = time()
        render_gpu[blocks, threadsperblock](image, input_Matrix)
        render_time = time() - t1
        print("gpu_calculation done " + str(render_time))
    if mode == 1:
        t1 = time()
        render_cpu(image, input_Matrix)
        render_time = time() - t1
        print("cpu_calculation done" + str(render_time))
    surface = pygame.surfarray.make_surface(image.astype("uint8"))
    display.blit(surface, (sidebar, 0))
    return truncate(render_time, 3)


#mandelbrot settings

WIDTH = 900
HEIGHT = WIDTH
MAX_ITER = 255
UPPER_BOUND = 4
start_point = [-1.3887052905883253, 0]
radius = 0.02579603060046148

#pygame settings
sidebar = 400
w, h = WIDTH + sidebar, HEIGHT
color = (255, 255, 190)
ctime = 0
mode = 0 #0 = gpu, 1 = cpu

pygame.init()
display = pygame.display.set_mode((w, h))
pygame.display.set_caption("Mandelbrot Visualization")
font_head = pygame.font.SysFont("Verdana", 35, bold=False)
font_main = pygame.font.SysFont("Verdana", 25)
running = True

image = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
a = np.arange(0, WIDTH)
b = np.arange(0, HEIGHT)
A, B = np.meshgrid(a, b)
Z = A + B
Z = 255 * Z / Z.max()

surface = pygame.surfarray.make_surface(image)
display.fill((0, 0, 0))

threadsperblock = 128
blocks = (image.shape[0] * image.shape[1] + (threadsperblock - 1)) // threadsperblock


def draw_sidebar():
    draw_text('main menu', font_head, color, display, 30, 30)
    draw_text('calculation time', font_main, color, display, 30, 100)
    draw_text(str(ctime) + " s", font_main, color, display, 30, 150)
    draw_text("coords [arrows]", font_main, color, display, 30, 220)
    draw_text(str(truncate(start_point[0], 8)), font_main, color, display, 30, 270)
    draw_text(str(truncate(start_point[1], 8)), font_main, color, display, 30, 320)
    draw_text("zoom [W][S]:", font_main, color, display, 30, 390)
    draw_text(str(int(1/radius)), font_main, color, display, 30, 440)
    draw_text("gpu <-> cpu mode", font_main, color, display, 30, 510)
    draw_text("too slow to work", font_main, color, display, 30, 560)
    #draw_text((("cpu", "gpu")[mode == 0]), font_main, color, display, 30, 560)
    draw_text("screenshot [F12]", font_main, color, display, 30, 630)


def draw_text(text, font, color, surface, x, y):
    textobj = font.render(text, 1, color)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)


def truncate(n, m):
    return int(n * 10**m) / 10**m


def update_screen():
    global ctime
    display.fill((0, 0, 0))
    ctime = display_update_mandel()
    draw_sidebar()


draw_sidebar()
update_screen()
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.key == pygame.K_s:
                radius *= 1.2
                update_screen()
            if event.key == pygame.K_w:
                radius *= 1/1.2
                update_screen()
            if event.key == pygame.K_DOWN:
                start_point[0] += 1/3 * radius
                update_screen()
            if event.key == pygame.K_UP:
                start_point[0] -= 1 / 3 * radius
                update_screen()
            if event.key == pygame.K_LEFT:
                start_point[1] += 1 / 3 * radius
                update_screen()
            if event.key == pygame.K_RIGHT:
                start_point[1] -= 1 / 3 * radius
                update_screen()
            if event.key == pygame.K_F12:
                abschnitt = str(start_point) + " " + str(radius)
                print(abschnitt)
                pygame.image.save(display, "screenshot" + abschnitt + ".jpg")
                print("screenshot saved")
    pygame.display.update()
