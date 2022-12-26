from __future__ import annotations

import sys
sys.path.insert(1,'C:\\Users\\gchan\\Desktop\\REANLEA\\2023\\common')

from reanlea_colors  import*
from func import*

from asyncore import poll2
from audioop import add
from cProfile import label
from distutils import text_file

import math
from math import pi


from manim import *


from numpy import array
import numpy as np
import random as rd
from dataclasses import dataclass
import fractions
from imp import create_dynamic
from multiprocessing import context
from multiprocessing.dummy import Value
from numbers import Number
from operator import is_not
from tkinter import CENTER, Y, Label, Scale
from imp import create_dynamic
from tracemalloc import start
from turtle import degrees, dot
from manim import*
from math import*
from manim_fonts import*
import numpy as np
import random
from sklearn.datasets import make_blobs

from manim.mobject.geometry.arc import Circle
from manim.mobject.geometry.polygram import Square, Triangle
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.types.vectorized_mobject import VMobject
from manim.utils.space_ops import angle_of_vector
from manim.mobject.geometry.tips import ArrowTriangleFilledTip
from manim.mobject.geometry.tips import ArrowTip
from manim.mobject.geometry.tips import ArrowTriangleTip
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import root_scalar
import requests
import io
from PIL import Image
from random import choice, seed
from random import random, seed
from enum import Enum
from scipy.stats import norm, gamma
from scipy.optimize import fsolve
import random


config.background_color= REANLEA_BACKGROUND_COLOR


###################################################################################################################


class Mandelbrot(Scene):
    def construct(self):


        # Set up the bounds of the Mandelbrot set
        xmin, xmax = -2, 1
        ymin, ymax = -1, 1

        # Set the number of pixels for the image
        pixels_x = 1000
        pixels_y = 1000

        # Create a blank image
        image = Image.new("RGB", (pixels_x, pixels_y))

        # Iterate over the pixels in the image
        for x in range(pixels_x):
            for y in range(pixels_y):
                # Map the pixel coordinates to the complex plane
                z = complex(xmin + (xmax - xmin) * x / pixels_x,
                            ymin + (ymax - ymin) * y / pixels_y)

                # Compute the value of the Mandelbrot function at this point
                c = z
                for i in range(255):
                    if abs(z) > 2:
                        break
                    z = z**2 + c

                # Color the pixel based on the number of iterations
                # required for the function to diverge
                image.putpixel((x, y), (i % 16 * 16,0, 0))

        # Create a image object from the image and add it to the scene
        mandelbrot_image = ImageMobject(image)
        mandelbrot_image.scale(2)
        self.add(mandelbrot_image)

        

        # manim -pqh open_ai.py Mandelbrot

        # manim -sqk open_ai.py Mandelbrot



###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2023\lab" 