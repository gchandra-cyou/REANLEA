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
from manim.opengl import*


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

config.disable_caching_warning=True
config.disable_caching=True
class MengerSponge(ThreeDScene):
    def construct(self):
        cube = Cube()
        self.play(Create(cube))
        self.wait(1)

        # Create the Menger sponge recursively
        self.create_menger(cube, depth=3)

    def create_menger(self, cube, depth):
        if depth == 0:
            return

        new_cubes = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i != 1 or j != 1 or k != 1:
                        new_cube = cube.copy()
                        new_cube.scale(1 / 3)
                        new_cube.shift(
                            cube.get_width() * (i - 1) / 3,
                            cube.get_height() * (j - 1) / 3,
                            cube.get_depth() * (k - 1) / 3,
                        )
                        new_cubes.append(new_cube)

        self.play(*[Create(c) for c in new_cubes])

        for c in new_cubes:
            self.create_menger(c, depth - 1)

    def camera_position(self, mob):
        self.set_camera_orientation(
            phi=75 * DEGREES,
            theta=-30 * DEGREES,
        )
        self.begin_3dillusion_mobject(mob)


        # manim -pqh open_ai.py MengerSponge

        # manim -sqh open_ai.py MengerSponge


class TesseractProjection(ThreeDScene):
    def construct(self):
        # Define vertices of the tesseract in 4D space
        vertices = [
            [-1, -1, -1, -1],
            [1, -1, -1, -1],
            # ... (other vertices)
            [1, 1, 1, 1],
            [-1, 1, 1, 1]
        ]

        # Define edges of the tesseract
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            # ... (other edges)
            (14, 15), (15, 12), (12, 13), (13, 14)
        ]

        # Project 4D vertices to 3D space (ignore 4th dimension)
        projected_vertices = [vertex[:3] for vertex in vertices]

        # Create dots for each projected vertex
        dots = VGroup(*[Dot(point=np.array(vertex), radius=0.08) for vertex in projected_vertices])

        # Create lines to represent edges in 3D space
        lines = VGroup(*[Line(np.array(vertices[i][:3]), np.array(vertices[j][:3])) for (i, j) in edges])

        # Set up camera orientation
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

        # Display dots and lines in the scene
        self.add(dots, lines)
        self.wait(3)

        # manim -pqh open_ai.py Tesseract

        # manim -sqh open_ai.py Tesseract


###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2023\lab" 