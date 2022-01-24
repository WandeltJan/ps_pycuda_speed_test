import pycuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
from PIL import Image, ImageDraw
from numba import jit
from time import time
import time as t

#variable Parameter
WIDTH = 4000
HEIGHT = WIDTH
MAX_ITER = 255
UPPER_BOUND = 2
start_point = [2.613577e-1, -2.018128e-3]
radius = 0.000003

x_min = start_point[0] - radius
x_max = start_point[0] + radius
y_min = start_point[1] - radius
y_max = start_point[1] + radius

mandel_kernel = ElementwiseKernel(
    "pycuda::complex<float> *input, float *output, int max_iters, float upper_bound",
    """
    output[i] = 1;

    pycuda::complex<float> c = input[i]; 
    pycuda::complex<float> z(0,0);

    for (int j = 0; j <= max_iters; j++)
        {

         z = z*z + c;

         if(abs(z) > upper_bound)
             {
              output[i] = j + 1 - log(log2(abs(z)));
              break;
             }

        }

    """,
    "mandel_ker")


mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }
  """)


def pycuda_create_fractal(min_x, max_x, min_y, max_y, image, max_iters, upper):
    # Pixel coordinaten zu komplexen Zahlen umwandeln
    real_Teil = np.matrix(np.linspace(x_min, x_max, WIDTH), dtype=np.complex64)
    imaginaer_Teil = np.matrix(np.linspace(y_max, y_min, HEIGHT), dtype=np.complex64) * 1j
    input_Matrix = np.array(real_Teil + imaginaer_Teil.transpose(), dtype=np.complex64)

    # Komplexe Zahlen in Gpu laden
    input_Matrix_GPU = gpuarray.to_gpu_async(input_Matrix)

    # synchronizieren (lock)
    pycuda.autoinit.context.synchronize()

    # output Matrix allocieren
    mandelbrot_graph_gpu = gpuarray.empty(shape=input_Matrix.shape, dtype=np.float32)

    # Mandelbrot Mengen berechnung durchführen für jede element der Inputmatrix
    t3 = time()
    mandel_kernel(input_Matrix_GPU, mandelbrot_graph_gpu, np.int32(max_iters), np.float32(upper))
    t4 = time()
    print("pure mandelbrot calc {}".format(t4 - t3))

    # synchronizieren (release)
    # pycuda.autoinit.context.synchronize()

    mandelbrot_graph = mandelbrot_graph_gpu.get_async()

    # pycuda.autoinit.context.synchronize()

    return mandelbrot_graph


def mandel_cpu(x, y, max_iters, upper):
  """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
  """
  c = complex(x, y)
  z = 0.0j
  for i in range(max_iters):
    z = z*z + c
    if (z.real*z.real + z.imag*z.imag) >= upper:
      return i
  return max_iters


def cpu_create_fractal(min_x, max_x, min_y, max_y, image, iters, upper):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters, upper)
            image[y, x] = color


@jit(nopython=True)
def mandel(x, y, max_iters, upper):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= upper:
            return i

    return 255

@jit(nopython=True)
def numba_create_fractal(min_x, max_x, min_y, max_y, image, iters, upper):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters, upper)
            image[y, x] = color

    return image


def manual_array_to_rgb(data):
    im = Image.new('HSV', (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            mandel_value = data[x][y]
            hue = int(255 * mandel_value / MAX_ITER)
            saturation = 255
            value = 255 if mandel_value < MAX_ITER else 0
            draw.point([x, y], (hue, saturation, value))

    im.convert("RGB")
    dst = Image.new("RGB", (WIDTH, HEIGHT))
    dst.paste(im, (0, 0))
    return dst


def pycuda_double_array_manual(size):
    #get cuda function
    func = mod.get_function("doublify")

    #create cpu array
    cpu_arr = np.random.randn(size, size)
    cpu_arr = cpu_arr.astype(np.float32)

    #load cpu array to gpu
    gpu_arr = gpuarray.to_gpu_async(cpu_arr)

    #call cuda function
    func(gpu_arr, block=(size, size, 1))

    #return data
    cpu_arr_doubled = gpu_arr.get_async()

    #compare the results
    #print(cpu_arr)
    #print(cpu_arr_doubled)
    #print("done")


def pycuda_double_array_automatic(size):
    gpu_arr = gpuarray.to_gpu(np.random.randn(size, size).astype(np.float32))
    array_doubled = (2*gpu_arr).get()
    #print(gpu_arr)
    #print(array_doubled)
    #print("done")


if __name__ == '__main__':
    #image_numba = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    image_pycuda = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    image_cpu = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    #start = time()
    #numba_data = numba_create_fractal(x_min, x_max, y_min, y_max, image_numba, MAX_ITER, UPPER_BOUND)
    #numba_end = time() - start
    #print("numba calculation {}s".format(numba_end))

    start = time()
    cpu_data = cpu_create_fractal(x_min, x_max, y_min, y_max, image_cpu, MAX_ITER, UPPER_BOUND)
    cpu_end = time() - start
    print("cpu calculation {}s".format(cpu_end))

    start = time()
    pycuda_data = pycuda_create_fractal(x_min, x_max, y_min, y_max, image_pycuda, MAX_ITER, UPPER_BOUND)
    endtime = time() - start
    print("pycuda calculation {}s".format(endtime))

    #visualise
    manual_array_to_rgb(pycuda_data).save(("mandeltotal_pycuda.png"))


    start = time()
    pycuda_double_array_manual(32)
    t.sleep(1)
    endtime = time() - start
    print("manual double calculation {}s".format(endtime-1))

    start = time()
    pycuda_double_array_automatic(64)
    t.sleep(1)
    endtime = time() - start
    print(" 64 double calculation {}s".format(endtime - 1))

    start = time()
    pycuda_double_array_automatic(32)
    t.sleep(1)
    endtime = time() - start
    print(" 32 double calculation {}s".format(endtime-1))

    start = time()
    pycuda_double_array_automatic(16)
    t.sleep(1)
    endtime = time() - start
    print(" 16 double calculation {}s".format(endtime-1))




