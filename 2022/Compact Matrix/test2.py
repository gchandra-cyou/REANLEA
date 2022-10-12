from __future__ import annotations
from asyncore import poll2
from audioop import add
from cProfile import label
from distutils import text_file

import math
from math import pi

import os,sys
from manim import *
from numpy import array
import numpy as np
import random as rd
import fractions
from imp import create_dynamic
from multiprocessing import context
from multiprocessing.dummy import Value
from numbers import Complex, Number
from operator import is_not
from tkinter import CENTER, N, Y, Label, Scale
from imp import create_dynamic
from tracemalloc import start
from turtle import degrees, dot
from manim import*
from math import*
from manim_fonts import*
import numpy as np
import random
from sklearn.datasets import make_blobs
from reanlea_colors import*
from func import*
from manim.mobject.geometry.arc import Circle
from manim.mobject.geometry.polygram import Square, Triangle
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.types.vectorized_mobject import VMobject
from manim.utils.space_ops import angle_of_vector
from manim.mobject.geometry.tips import ArrowTriangleFilledTip
from manim.mobject.geometry.tips import ArrowTip
from manim.mobject.geometry.tips import ArrowTriangleTip
from round_corner import*
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import root_scalar
from round_corner import angle_between_vectors_signed
import requests
import io
from PIL import Image


config.background_color= REANLEA_BACKGROUND_COLOR


###################################################################################################################

max_iter=80

def mandelbrot(c):
    z=0+0j
    n=0

    while abs(z)<=2 and n<max_iter:
        z=z*z+c
        n+=1
        return n

if __name__ == "__main__":
    for a in range(-10, 10, 5):
        for b in range(-10, 10, 5):
            c = complex(a / 10, b / 10)
            print(c, mandelbrot(c))


class MandelbrotTest(Scene):
    def construct(self):

        # Image size (pixels)
        WIDTH = 600
        HEIGHT = 400

# Plot window
        RE_START = -2
        RE_END = 1
        IM_START = -1
        IM_END = 1

        palette = []

        im = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
        

        for x in range(0, WIDTH):
            for y in range(0, HEIGHT):
        # Convert pixel coordinate to complex number
                c = R3_to_complex(
                    np.ndarray(RE_START + (x / WIDTH) * (RE_END - RE_START), IM_START + (y / HEIGHT) * (IM_END - IM_START),0)
                )
        # Compute the number of iterations
                m = mandelbrot(c)
        # The color depends on the number of iterations
                color = 255 - int(m * 255 / max_iter)
        # Plot the point
        self.play(
            Create(m)
        )


                
        

        # manim -pqh test2.py MandelbrotTest




class ComplexTransformation(LinearTransformationScene):
    def __init__(self):
                LinearTransformationScene.__init__(
                    self,
                    show_basis_vectors = False
                )


    def construct(self):
        function = lambda point: complex_to_R3(np.e**((2*np.pi)*R3_to_complex(point)))

        self.apply_nonlinear_transformation(function, run_time = 3)

        self.wait()

        
        # manim -pqh test2.py ComplexTransformation



# mandelbrot.py

from dataclasses import dataclass

@dataclass
class MandelbrotSet:
    max_iterations: int

    def __contains__(self, c: complex) -> bool:
        z = 0
        for _ in range(self.max_iterations):
            z = z ** 2 + c
            if abs(z) > 2:
                return False
        return True



class Ex(Scene):
    def construct(self):
        mandelbrot_set = MandelbrotSet(max_iterations=20)

        width, height = 512, 512
        scale = 0.0075
        BLACK_AND_WHITE = "1"

        image = Image.new(mode=BLACK_AND_WHITE, size=(width, height))
        for y in range(height):
            for x in range(width):
                c = scale * R3_to_complex(x - width / 2, height / 2 - y,0)
        image.putpixel((x, y), c not in mandelbrot_set)

        self.play(
            Create(image)
        )
        self.wait(2)


        # manim -pqh test2.py Ex
 

import matplotlib.pyplot as plt
import numpy as np

def get_iter(c:float, thresh:int, max_steps:int ) -> int:
    # Z_(n) = (Z_(n-1))^2 + c
    # Z_(0) = c
    z=c
    i=1
    while i<max_steps and (z*z)<thresh:
        z=z*z +c
        i+=1
    return i

class Ex2(Scene):
    def construct(self):
        n=100
        mx = 2.48 / (n-1)
        my = 2.26 / (n-1)
        mapper = lambda x: mx*x - 2
        img=np.full((n,n), 255)
        for x in range(n):
            for y in range(n):
                it = get_iter(*mapper(x), 4, 25)
                img[y][x] = 255 - it
        
        self.play(Write(img))

        # manim -pqh test2.py Ex2


def get_iter(c:complex, thresh:int =4, max_steps:int =25) -> int:
    # Z_(n) = (Z_(n-1))^2 + c
    # Z_(0) = c
    z=c
    i=1
    while i<max_steps and (z*z.conjugate()).real<thresh:
        z=z*z +c
        i+=1
    return i


def plotter(n, thresh, max_steps=25):
    mx = 2.48 / (n-1)
    my = 2.26 / (n-1)
    mapper = lambda x,y: (mx*x - 2, my*y - 1.13)
    img=np.full((n,n), 255)
    for x in range(n):
        for y in range(n):
            it = get_iter(complex(*mapper(x,y)), thresh=thresh, max_steps=max_steps)
            img[y][x] = 255 - it
    return img


class Ex3(Scene):
    def construct(self):
        
        im=plotter(n=1000, thresh=4, max_steps=50)

        self.add(im)

        # manim -pqh test2.py Ex3
















##########  https://slama.dev/manim/3d-and-the-other-graphs/  #########



class ComplexTransformation(LinearTransformationScene):
    def construct(self):
        square = Square().scale(2)
        function = lambda point: complex_to_R3(R3_to_complex(point)**2)

        self.add_transformable_mobject(square)

        self.apply_nonlinear_transformation(function)

        self.wait()


        # manim -pqh test2.py ComplexTransformation



from math import sin, cos


class ParametricGraphExample(Scene):
    def construct(self):
        axes = Axes(x_range=[-10, 10], y_range=[-5, 5])
        labels = axes.get_axis_labels(x_label="x", y_label="y")

        def f1(t):
            """Parametric function of a circle."""
            return (cos(t) * 3 - 4.5, sin(t) * 3)

        def f2(t):
            """Parametric function of <3."""
            return (
                0.2 * (16 * (sin(t)) ** 3) + 4.5,
                0.2 * (13 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(4 * t)),
            )

        # the t_range parameter determines the range of the parametric function parameter
        g1 = axes.plot_parametric_curve(f1, color=RED, t_range=[0, 2 * PI])
        g2 = axes.plot_parametric_curve(f2, color=BLUE, t_range=[-PI, PI])

        self.play(Write(axes), Write(labels))

        self.play(AnimationGroup(Write(g1), Write(g2), lag_ratio=0.5))

        self.play(Unwrite(axes), Unwrite(labels), Unwrite(g1), Unwrite(g2))


        # manim -pqh test2.py ParametricGraphExample


class Axes3DExample(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()

        x_label = axes.get_x_axis_label(Tex("x"))
        y_label = axes.get_y_axis_label(Tex("y")).shift(UP * 1.8)

        # 3D variant of the Dot() object
        dot = Dot3D()

        # zoom out so we see the axes
        self.set_camera_orientation(zoom=0.5)

        self.play(FadeIn(axes), FadeIn(dot), FadeIn(x_label), FadeIn(y_label))

        self.wait(0.5)

        # animate the move of the camera to properly see the axes
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES, zoom=1, run_time=1.5)

        # built-in updater which begins camera rotation
        self.begin_ambient_camera_rotation(rate=0.15)

        # one dot for each direction
        upDot = dot.copy().set_color(RED)
        rightDot = dot.copy().set_color(BLUE)
        outDot = dot.copy().set_color(GREEN)

        self.wait(1)

        self.play(
            upDot.animate.shift(UP),
            rightDot.animate.shift(RIGHT),
            outDot.animate.shift(OUT),
        )

        self.wait(2)


        # manim -pqh test2.py Axes3DExample



class Rotation3DExample(ThreeDScene):
    def construct(self):
        cube = Cube(side_length=3, fill_opacity=1).set_color_by_gradient(REANLEA_BLUE_LAVENDER,REANLEA_MAGENTA,REANLEA_BLUE)

        self.begin_ambient_camera_rotation(rate=0.3)

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        self.play(Write(cube), run_time=2)

        self.wait(3)

        self.play(Unwrite(cube), run_time=2)


        # manim -pqh test2.py Rotation3DExample