from __future__ import annotations
from asyncore import poll2
from audioop import add
from cProfile import label
from distutils import text_file
from faulthandler import disable

import math
from math import pi

import os,sys
from pickle import FRAME
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
from tkinter import CENTER, N, Y, Frame, Label, Scale, font
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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools as it



config.background_color= REANLEA_BACKGROUND_COLOR
config.max_files_cached=500


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


class Ex2(Scene):
    def construct(self):
        
        im=plotter(n=1000, thresh=4, max_steps=50)

        self.add(im)

        # manim -pqh test2.py Ex2



class Try(Scene):
    def construct(self):
        
        title=Text("Holomorphic Dynamics", font="Roboto").set_color(REANLEA_TXT_COL)
        underline=Underline(title).set_stroke(color=REANLEA_YELLOW_CREAM, opacity=0.5, width=5)

        self.play(
            Write(title),
            Write(underline)
        )

        self.wait(2)

        # manim -pqh test2.py Try





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






def mandelbrot(x, y, threshold):
    """Calculates whether the number c = x + i*y belongs to the 
    Mandelbrot set. In order to belong, the sequence z[i + 1] = z[i]**2 + c
    must not diverge after 'threshold' number of steps. The sequence diverges
    if the absolute value of z[i+1] is greater than 4.
    
    :param float x: the x component of the initial complex number
    :param float y: the y component of the initial complex number
    :param int threshold: the number of iterations to considered it converged
    """
    # initial conditions
    c = complex(x, y)
    z = complex(0, 0)
    
    for i in range(threshold):
        z = z**2 + c
        if abs(z) > 4.:  # it diverged
            return i
        
    return threshold - 1  # it didn't diverge


class MandelbrotSetEx(Scene):
    def construct(self):

        x_start, y_start = -2, -1.5  # an interesting region starts here
        width, height = 3, 3  # for 3 units up and right
        density_per_unit = 250  # how many pixles per unit

        # real and imaginary axis
        re = np.linspace(x_start, x_start + width, width * density_per_unit )
        im = np.linspace(y_start, y_start + height, height * density_per_unit)

        fig = plt.figure(figsize=(10, 10))  # instantiate a figure to draw
        ax = plt.axes()  # create an axes object


        def animate(i):

            ax.clear()  # clear axes object
            ax.set_xticks([], [])  # clear x-axis ticks
            ax.set_yticks([], [])  # clear y-axis ticks
    
            X = np.empty((len(re), len(im)))  # re-initialize the array-like image
            threshold = round(1.15**(i + 1))  # calculate the current threshold
    
            # iterations for the current threshold
            for i in range(len(re)):
                for j in range(len(im)):
                    X[i, j] = mandelbrot(re[i], im[j], threshold)
    
             # associate colors to the iterations with an iterpolation
            img = ax.imshow(X.T, interpolation="bicubic", cmap='magma')
            return [img]

        anim = animation.FuncAnimation(fig, animate, frames=45, interval=120, blit=True)
        anim.save('mandelbrot.gif',writer='imagemagick')
    


        # manim -pqh test2.py MandelbrotSetEx




class ComplexIdderation(Scene):
    def construct(self):
        
        # Mandelbrot function
        def z(n, c):
            if n == 0:
                return 0
            else:
                return z(n-1, c) ** 2 + c
        
        # Draw a complex plane
        plane = ComplexPlane(
            x_range=(-4, 2, 0.5),
            y_range=(-2, 2, 0.5),
            x_length=14,
            y_length=10,
        ).add_coordinates()
        self.add(plane)
        
        # Define some varibles
        numberOfPoints = 20
        c = .25+.25j
        dots = []
        lines = []
        
        # Define the starting point for the rest of the points to build off of
        start = ComplexValueTracker(c)
        
        # Function to generate dots based on what iderateion needs generated
        def generatePoint(n):
            dots.append(Dot().add_updater(lambda x: x.move_to(plane.n2p(z(n, start.get_value())))))
        
        # Function to generate a line based on what iderateion needs generated
        def generateLine(n):
            lines.append(Line().add_updater(lambda l: l.put_start_and_end_on(plane.n2p(z(n, start.get_value())), plane.n2p(z(n+1, start.get_value())))))
        
        # Generate all the point objects using the Mandelbrot function
        for i in range(1, numberOfPoints):
            generatePoint(i)
        
        # Generate the lines between the dot objects
        for i in range(1, numberOfPoints-1):
            generateLine(i)
        
        # Draw the dots
        for i in dots:
            self.add(i)
        
        #Draw the lines
        for i in lines:
            self.add(i)
            
        for i in range(1):
            self.play(start.animate.set_value(-0.5+0.5j), run_time=5)
            self.play(start.animate.set_value(-0.5-0.7j), run_time=5)
            self.play(start.animate.set_value(0.5+0.5j), run_time=5)



            # manim -pqh test2.py ComplexIdderation


class MandelbrotOrJuliaSet(ImageMobject):
    def __init__(
        self,
        M_or_J: str = 'M',
        constant: complex = complex(0, 0),
        dynamic_system_function = lambda z, c: z**2 + c,
        max_steps: int = 25,
        resolution: float = 1080,
        continum_effect: bool = True,
        x_range: list = [-2, 0.5],
        y_range: list = [-1.15, 1.15],
        **kwargs
    ):
        self.M_or_J = M_or_J
        self.max_steps = max_steps
        self.continum_effect = continum_effect

        if self.M_or_J not in ['M', 'J']:
            raise ValueError('\'M_or_J\'' + ' argument must be \'M\' or \'J\'.')

        if self.M_or_J == 'J':
            self.constant = constant
        
        self.dynamic_system_function = dynamic_system_function

        self.x_m, self.x_M = x_range[0], x_range[1]
        self.y_m, self.y_M = y_range[0], y_range[1]

        self.resolution = resolution, int(np.abs(self.x_M - self.x_m)/np.abs(self.y_M - self.y_m)*resolution)

        self.dx = np.abs(self.x_M - self.x_m)/self.resolution[1]
        self.dy = np.abs(self.y_M - self.y_m)/self.resolution[0]

        self.image = self.get_image()
        super().__init__(self.image, **kwargs)

        self.complex_plane = ComplexPlane(
            x_range=[self.x_m, self.x_M],
            y_range=[self.y_m, self.y_M],
            x_length=self.width,
            y_length=self.height
        ).move_to(self)



    def get_image(self) -> np.ndarray:
        """Gets the array to create the image."""
        array = np.zeros(self.resolution)
        for n in range(len(array)):
            for p in range(len(array[0])):
                array[n, p] = self.get_iterations_for_complex_number(self.mapper_array_to_plane(n, p), self.max_steps)
        return np.array(array, dtype='uint8')
    def mapper_array_to_plane(self, n: int, p: int) -> complex:
        """Returns the complex number corresponding to the `(n, p)` coefficient of `self.array`."""
        return complex(self.x_m + p*self.dx, self.y_M - n*self.dy)


    def mapper_plane_to_array(self, complex: complex) -> tuple[int, int]:
        """Returns the coordinates `(n, p)` in `self.array` corresponding to the complex number `complex`."""
        x, y = complex.real, complex.imag
        return int(np.floor((self.y_M - y)/self.dy)), int(np.floor((x - self.x_m)/self.dx))


    def get_iterations_for_complex_number(self, c: complex, max_steps: int) -> bool:
        """Gets the amount of white for each pixel of the image, depending on `M_or_J` argument value."""
        z = c
        i = 0

        if self.M_or_J == 'M':
            while i < max_steps and self.complex_module(z) <= 2:
                z = self.dynamic_system_function(z, c)
                i += 1

        if self.M_or_J == 'J':
            while i < max_steps and self.complex_module(z) <= 2:
                z = self.dynamic_system_function(z, self.constant)
                i += 1

        if self.continum_effect:
            if i == max_steps:
                return 0
            return i*(255/max_steps)

        return 255*(self.complex_module(z) <= 2)


    def complex_module(self, complex: complex) -> float:
        """Computes the complex module of `complex`."""
        return np.sqrt(complex.real**2 + complex.imag**2)


class ExMan(Scene):
    def construct(self):
        m=MandelbrotOrJuliaSet()

        self.play(
            FadeIn(m)
        )

        # manim -pqh test2.py ExMan



from manim.utils.unit import Pixels

class SubstitutionSystem(Scene):
    def construct(self):
        depth = 6
        base_s = Square(
            side_length=config["pixel_height"] * Pixels, fill_color=REANLEA_BACKGROUND_COLOR, fill_opacity=1, stroke_width=0
        )
        self.add(base_s)
        fractal = VGroup(base_s)
        old_fractal = VGroup()

        for _ in range(depth):
            old_fractal = fractal.copy()
            print(fractal)
            fractal = VGroup()
            color = random_bright_color()
            for s in old_fractal:
                s_g = VGroup(*[
                        s.copy().scale(0.5)
                        for _ in range(4)
                    ]).arrange_in_grid(cols=2, rows=2, buff=0)
                s_g[0].set_fill(color, 1)
                fractal.add(s_g)

            self.play(FadeTransform(old_fractal, fractal))



            # manim -pqh test2.py SubstitutionSystem




import random
import math

class GraphNetworks(Scene):
    def construct(self):
        FRAME_HEIGHT = 8.0
        FRAME_RATIO = 16.0/9.0
        FRAME_WIDTH = FRAME_HEIGHT * FRAME_RATIO
        '''background = Rectangle(width = FRAME_WIDTH, height = FRAME_HEIGHT, stroke_width = 0, fill_opacity = 1).set_color_by_gradient([RED, ORANGE])
        self.add(background)'''

        dotCount = 100
        dots = []
        directionX = []
        directionY = []
        speed = []
        
        alpha = ValueTracker(0)
        alphaLimit = 2
        def reset():
            for i in range(dotCount):
                directionX.append(random.uniform(-0.2, 0.2))
                directionY.append(random.uniform(-0.2, 0.2))
                speed.append(random.uniform(-2, 2))
                dots.append(Dot([(random.random()*FRAME_WIDTH/2.0)-FRAME_WIDTH/2.0, (random.random()*FRAME_HEIGHT/2.0)-FRAME_HEIGHT/2.0, 0]))
                   
        def generate_moving_dots():
            for i in range(dotCount):
                self.add(always_shift(dots[i], [directionX[i], directionY[i], 0], speed[i]))
        def lines_draw():
            lines = VGroup()
            for i in range(dotCount):
                for j in range(i+1, dotCount):
                    dist = math.sqrt(math.pow(dots[j].get_x() - dots[i].get_x(), 2) + math.pow(dots[j].get_y() - dots[i].get_y(), 2))
                    if dist >= alphaLimit:
                        alpha.set_value(0)
                    elif dist >= 0 and dist < alphaLimit:
                        alpha.set_value(-((dist / alphaLimit)) + 1)
                    else:
                        alpha.set_value(1)
                    lines.add(Line(start = [dots[i].get_x(), dots[i].get_y(), 0], end = [dots[j].get_x(), dots[j].get_y(), 0], stroke_width = 10, stroke_opacity = alpha.get_value()))

                if dots[i].get_x() < -(FRAME_WIDTH/2.0 + 1):
                    dots[i].set_x((FRAME_WIDTH/2.0 + 1))
                elif dots[i].get_x() > (FRAME_WIDTH/2.0 + 1):
                    dots[i].set_x(-(FRAME_WIDTH/2.0 + 1))

                if dots[i].get_y() < -(FRAME_HEIGHT/2.0 + 1):
                    dots[i].set_y((FRAME_HEIGHT/2.0 + 1))
                elif dots[i].get_y() > (FRAME_HEIGHT/2.0 + 1):
                    dots[i].set_y(-(FRAME_HEIGHT/2.0 + 1))
                    
            return lines
        
        reset()
        generate_moving_dots()
        self.add(always_redraw(lines_draw))

        self.wait(10)


        # manim -pqh test2.py GraphNetworks 




#config.disable_caching=True

from numba import jit
from numba import njit

class MandelbrotAndJuliaScene(Scene):
    def construct(self):
        jited_MandelbrotSet = njit()(MandelbrotSet)
        jited_JuliaSet = njit()(JuliatSet)
        mandelbrot_set = ImageMobject(jited_MandelbrotSet(
            max_steps=50,
            resolution=1080,
        )).set_z_index(4)
        mandelbrot_set_complex_plane = ComplexPlane(
            x_range = np.array([-2, 0.5]),
            y_range = np.array([-1.15, 1.15]),
            x_length = mandelbrot_set.height,
            y_length = mandelbrot_set.width 
        )
        old_height = mandelbrot_set.height
        mandelbrot_set.height = 3.5
        rectangle = BackgroundRectangle(mandelbrot_set, color=WHITE, stroke_width=2, stroke_opacity=1, fill_opacity=0, buff=0).set_z_index(5)
        Group(mandelbrot_set, rectangle).to_corner(UR, buff=0.2)

        mandelbrot_set_complex_plane.move_to(mandelbrot_set).scale(3.5/old_height)


        c = ComplexValueTracker(array_of_complex_constants[0][0])
        max_steps = ValueTracker(20)

        dot = always_redraw(lambda: Dot(mandelbrot_set_complex_plane.c2p(c.get_value().real, c.get_value().imag), radius=DEFAULT_SMALL_DOT_RADIUS, color=PURE_BLUE).set_z_index(6))
        label = always_redraw(lambda: MathTex('c', stroke_color=BLACK, stroke_width=0.75).scale(1.25).next_to(dot, UL, buff=0.1).set_z_index(6))

        eq = MathTex('z \\mapsto z^2 + c').to_corner(UL, buff=0.27).set_z_index(4)
        rectangle_eq = SurroundingRectangle(eq, color=WHITE, buff=0.25).set_z_index(5)

        constant = MathTex('c =').to_edge(buff=0.22).to_edge(DOWN, buff=0.65).set_z_index(10)
        for complex_constant in array_of_complex_constants:
            complex_constant[1].next_to(constant, RIGHT)
        constant_value = array_of_complex_constants[0][1].set_z_index(10)
        background = always_redraw(lambda: BackgroundRectangle(Group(constant, constant_value), color=BLACK))
        surrounding = always_redraw(lambda: SurroundingRectangle(Group(constant, constant_value), color=WHITE, buff=0.2)).set_z_index(11)


        julia_set = ImageMobject(jited_JuliaSet(
            c.get_value(),
            max_steps=int(max_steps.get_value()),
            resolution=1440,
            x_range=np.array([-1.75, 1.75]),
            y_range=np.array([-1.5, 1.5])
        )).set_z_index(3)
        julia_set.height = config.frame_height

        self.play(FadeIn(Group(mandelbrot_set, rectangle, eq, rectangle_eq, julia_set, dot, label, constant, constant_value, background, surrounding)))
        self.wait(1.75)

        def update_julia_set(julia_set):
            julia_set.become(ImageMobject(jited_JuliaSet(
                c.get_value(),
                max_steps=int(max_steps.get_value()),
                resolution=1440,
                x_range=np.array([-1.75, 1.75]),
                y_range=np.array([-1.5, 1.5])
            )).set_z_index(3))
            julia_set.height = config.frame_height

        julia_set.add_updater(update_julia_set)

        for k in range(1, len(array_of_complex_constants)):
            if k == 4:
                self.play(
                    c.animate.set_value(array_of_complex_constants[4][0]),
                    max_steps.animate.set_value(35),
                    Transform(array_of_complex_constants[0][1], array_of_complex_constants[4][1]),
                    run_time=3
                    )
                self.wait(1.75)
            if k == 6:
                self.play(
                    c.animate.set_value(array_of_complex_constants[6][0]),
                    max_steps.animate.set_value(500),
                    Transform(array_of_complex_constants[0][1], array_of_complex_constants[6][1]),
                    run_time=3
                    )
                self.wait(1.75)
            if k == 11:
                self.play(
                    c.animate.set_value(array_of_complex_constants[11][0]),
                    max_steps.animate.set_value(70),
                    Transform(array_of_complex_constants[0][1], array_of_complex_constants[11][1]),
                    run_time=3
                    )
                self.wait(1.75)
            else:
                self.play(
                    c.animate.set_value(array_of_complex_constants[k][0]),
                    max_steps.animate.set_value(20),
                    Transform(array_of_complex_constants[k-1][1], array_of_complex_constants[k][1]),
                    run_time=3
                    )
                self.wait(1.75)




array_of_complex_constants = [
    [complex(np.pi/17, np.pi/17), MathTex('-\\frac{\\pi}{17} + \\frac{\\pi}{17}i')],
    [complex(np.pi/17, 42/1975), MathTex("-\\frac{\\pi}{17} + \\frac{42}{1975}i")],
    [complex(-1.975, 0.42), MathTex("-1.975 + 0.42i")],
    [complex(np.pi/17, np.exp(-42)), MathTex("\\frac{\\pi}{17} + e^{-42}i")],
    [complex(0.3, 0.03), MathTex("0.3 + 0.03i")],
    [complex(0.42, 1), MathTex("0.42 + i")],
    [complex(-1/15, 2/3), MathTex("-\\frac{1}{15} + \\frac{2}{3}i")],
    [complex(-1.877, 0.115), MathTex("-\\frac{\\pi}{17} + \\frac{\\pi}{17}i")],
    [complex(0, 0), MathTex("0")],
    [complex(1/42, np.pi/17), MathTex("\\frac{1}{42} + \\frac{\\pi}{17}i")],
    [complex(-1.683, 0.6977), MathTex("-1.683 + 0.6977i")],
    [complex(0.07, 0.7), MathTex("0.07 + 0.7i")]
]




def MandelbrotSet(max_steps: int = 50, resolution: int = 1080, x_range: np.ndarray = np.array([-2, 0.5]), y_range: np.ndarray = np.array([-1.15, 1.15])) -> np.ndarray:
        x_m, x_M, y_m, y_M = x_range[0], x_range[1], y_range[0], y_range[1]
        resolution_tuple = resolution, int(np.abs(x_M - x_m)/np.abs(y_M - y_m)*resolution)
        dx, dy = np.abs(x_M - x_m)/resolution_tuple[1], np.abs(y_M - y_m)/resolution_tuple[0]
        array = np.zeros(resolution_tuple, dtype='uint8')
        for n in range(len(array)):
            for p in range(len(array[0])):
                c = complex(x_m + p*dx, y_M - n*dy)
                z = c
                i = 0
                while i < max_steps and np.abs(z) <= 2:
                    z = z**2 + c
                    i += 1
                if i == max_steps:
                    array[n, p] = 0
                else:
                    array[n, p] = i*(255/max_steps)
        return array



def JuliatSet(complex_constant: complex = complex(0, 0), max_steps: int = 50, resolution: int = 1080, x_range: np.ndarray = np.array([-2, 0.5]), y_range: np.ndarray = np.array([-1.15, 1.15])) -> np.ndarray:
    x_m, x_M, y_m, y_M = x_range[0], x_range[1], y_range[0], y_range[1]
    resolution_tuple = resolution, int(np.abs(x_M - x_m)/np.abs(y_M - y_m)*resolution)
    dx, dy = np.abs(x_M - x_m)/resolution_tuple[1], np.abs(y_M - y_m)/resolution_tuple[0]
    array = np.zeros(resolution_tuple, dtype='uint8')
    for n in range(len(array)):
        for p in range(len(array[0])):
            c = complex(x_m + p*dx, y_M - n*dy)
            z = c
            i = 0
            while i < max_steps and np.abs(z) <= 2:
                z = z**2 + complex_constant
                i += 1
            if i == max_steps:
                array[n, p] = 0
            else:
                array[n, p] = i*(255/max_steps)
    return array


    # manim -pqh test2.py MandelbrotAndJuliaScene



class Mandel(Scene):
    def construct(self):
        x=ValueTracker(3)
        jited_MandelbrotSet = njit()(MandelbrotSet)
        mandelbrot_set = ImageMobject(jited_MandelbrotSet(
            max_steps=x.get_value(),
            resolution=1080,
        )).set_z_index(4)
        mandelbrot_set_complex_plane = ComplexPlane(
            x_range = np.array([-2, 0.5]),
            y_range = np.array([-1.15, 1.15]),
            x_length = mandelbrot_set.height,
            y_length = mandelbrot_set.width 
        )
        old_height = mandelbrot_set.height
        mandelbrot_set.height = old_height
        

        mandelbrot_set_complex_plane.move_to(mandelbrot_set).scale(3.5/old_height).set_color(RED)
        

        def update_mandelbrot_set(mandelbrot_set):
            mandelbrot_set.become(ImageMobject(jited_MandelbrotSet(
                max_steps=int(x.get_value()),
                resolution=1080,
            )).set_z_index(3))
            mandelbrot_set.height = config.frame_height

        mandelbrot_set.add_updater(update_mandelbrot_set)


        
        self.play(
            FadeIn(mandelbrot_set),
        )
        self.play(
            x.animate.set_value(50),
            run_time=3
        )
        

        # manim -pql test2.py Mandel

        # manim -sqk test2.py Mandel






#arr = np.zeros((81, 81))




from colour import Color
class Col_Field(Scene):
    def construct(self):

        arr = np.zeros((81,81))

        def func(x, y):
            return x * np.cos(y) + y * np.exp(-(x**2))

        def interpolate_color_array(color1: Color, color2: Color, alpha: np.ndarray):
            arr = []
            for i, j in enumerate(alpha):
                arr += [color_to_int_rgba(interpolate_color(color1, color2, j))]
            return np.array(arr)

        for i, j in it.product(np.arange(-4, 4.1, .1), np.arange(-4, 4.1, .1)):
                arr[int((i + 4) *10), int((j + 4)*10 )] = func(i, j)
                arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
                arr = arr.T[::-1, :]
                field = np.zeros((*arr.shape, 4), "uint8")

        for i, j in enumerate(arr):
                field[i] = interpolate_color_array(RED, GREEN, j)



        scalar_field = ImageMobject(field)

        self.add(scalar_field)
        self.wait(2)

        # manim -pqh test2.py Col_Field





class testingSine(Scene):
    def construct(self):
        
        bit_sequence = [1,0,1,1,0,1,1,1]
        self.bit_sequence = bit_sequence
        self.current_bit = 2
        self.stream_started = False

        self.establish_clock()
        self.show_debug_clock()
        self.enter_bit_stream(bit_sequence)
        self.setup_dot()
        self.setup_modulator_box()
        self.animate_bit_stream()
        self.setup_curve()

        dot = self.dot
        sine_curve_line = self.sine_curve_line
        sq = self.sq
        label = self.label
        arrow = self.arrow
        channel = self.channel
        bits = self.bits
        clock = self.clock
        time_group = self.time_group
        
        brace = Brace(sq, UP)
        brace_label = VGroup()
        info = Text("current bit:", font='/usr/share/fonts/truetype/baekmuk/dotum.ttf')
        current_bit_tag = Integer(self.current_bit).scale(2).set_color(BLACK)
        current_bit_tag.add_updater(lambda x : x.set_value(self.current_bit))
        brace_label.add(info)
        brace_label.add(current_bit_tag)
        brace_label.scale(0.5)
        brace_label.arrange(RIGHT)
        brace_label.add_updater(lambda x : x.next_to(brace, UP))
        colour_group = VGroup(brace, brace_label).add_updater(self.colour_control)

        title = Text("FSK - Frequency Shift Keying", font='/usr/share/fonts/truetype/baekmuk/dotum.ttf').scale(0.8)
        title.to_corner(UP+LEFT)

        bits.add_updater(self.align_bit_stream_top)
        self.play(Write(title))
        self.wait(0.5)
        enter = VGroup()
        enter.add(arrow, channel, label)
        self.play(Write(enter), FadeIn(sq), Write(time_group))
        self.add(dot, sine_curve_line, colour_group)
        self.wait(2)

        for j in range(len(bits)):
            self.stream_started = True
            the_bit = bits[0]
            the_bit.generate_target()
            the_bit.target.set_opacity(0)
            the_bit.target.shift(RIGHT*2)
            self.play(MoveToTarget(the_bit), run_time = 1.4)
            bits.remove(the_bit)
            self.wait(1.6)

        self.wait(1)

    def establish_clock(self):
        self.clock = 0
        self.increment = 0

    def enter_bit_stream(self, bit_sequence):
        self.bit_stream = bit_sequence
        self.bits = VGroup()
        for i in range(len(bit_sequence)):
            self.bits.add(Text(str(bit_sequence[i])))
        self.current_index = 0
        

    def align_bit_stream_top(self, bits):
        top = bits.get_top()
        sq = self.sq
        bits.shift(UP*(sq.get_center()[1] - top[1]))

    def animate_bit_stream(self):
        self.bits.arrange(RIGHT)
        self.play(Write(self.bits))
        self.wait(1)
        self.bits.generate_target()
        self.bits.target.scale(0.5)
        self.bits.target.arrange(DOWN)
        self.bits.target.to_edge(LEFT)
        for i in range(len(self.bits)):
            if self.bit_sequence[i] == 0:
                self.bits.target[i].set_color(BLUE_D)
            else:
                self.bits.target[i].set_color(RED_D)
        top = self.bits.target.get_top()
        sq = self.sq
        self.bits.target.shift(UP*(sq.get_center()[1] - top[1]))
        self.play(MoveToTarget(self.bits), run_time = 3)

    def oscillate_dot(self, the_dot, dt):
            self.clock += dt
            y_shift = 0.7*np.sin(self.clock*(1.5 + 2.5*self.current_amp(self.clock))*np.pi)
            the_dot.move_to(self.dot_origin + [0,y_shift,0])

    def current_amp(self, t):
        if(t < 2):
            return 0
        else:
            bit_index = int((t-2)/3)
            if(bit_index in range(len(self.bit_sequence))):
                self.current_bit = self.bit_sequence[bit_index]
            else:
                return 0

            if(self.current_bit):
                return 1
            else:
                return 0

    def setup_dot(self):
        self.dot = Dot()
        self.dot.shift(UP*0+LEFT*4)
        self.dot_origin = self.dot.get_center()
        self.dot.add_updater(self.oscillate_dot)

    def setup_modulator_box(self):
        self.sq = Square()
        self.sq.move_to(self.dot.get_center())
        self.label = Text("Modulator",font='/usr/share/fonts/truetype/baekmuk/dotum.ttf').scale(0.5)
        self.label.move_to(self.sq.get_bottom()).shift(DOWN*0.35)
        self.arrow = Arrow(self.sq.get_right()-[0.25,0,0], self.sq.get_right() + [8,0,0])
        self.channel = Text("to channel", font='/usr/share/fonts/truetype/baekmuk/dotum.ttf')
        self.channel.scale(0.5).move_to(self.arrow.get_right() + [0,-1.1,0])
    
    def move_signal(self, path):
        path.move_to(self.dot_origin + [self.clock/4, 0, 0])

    def get_curve(self):
        
        last_line = self.curve[-1]
        x = self.dot.get_center()[0]
        y = self.dot.get_center()[1]
        new_line = Line(last_line.get_end(),np.array([x,y,0]), color=YELLOW_D)
        if(self.stream_started):
            self.curve.add(new_line)

        return self.curve

    def setup_curve(self):
        self.curve = VGroup()
        self.curve.add(Line(self.dot.get_center(),self.dot.get_center()))
        self.sine_curve_line = always_redraw(self.get_curve)
        self.sine_curve_line.add_updater(self.move_signal)

    def show_debug_clock(self):
        box = Square().to_corner(corner = UP + RIGHT)
        label = Text("time (s)", font='/usr/share/fonts/truetype/baekmuk/dotum.ttf')
        label.move_to(box.get_bottom() - [0,0.4,0]).scale(0.5)
        watch = DecimalNumber(0, show_ellipsis=False, num_decimal_places=2, include_sign=False)
        watch.move_to(box)
        watch.add_updater(lambda x : x.set_value(self.clock))
        self.time_group = VGroup(box, watch, label).scale(0.8)

    def colour_control(self, thing):
        if self.current_bit == 1:
            thing.set_color(RED_D)
        elif self.current_bit == 0:
            thing.set_color(BLUE_D)
    


        # manim -pqh test2.py testingSine




 
class Dragon(MovingCameraScene):
    def construct(self):
        dragon_curve = VMobject(stroke_color=GOLD)
        dragon_curve_points = [LEFT, RIGHT]
        dragon_curve.set_points_as_corners(dragon_curve_points)
        dragon_curve.corners = dragon_curve_points
        self.add(dragon_curve)
        dragon_curve.add_updater(
            lambda mobject: mobject.set_style(stroke_width=self.camera.frame.width / 10),
        )
        dragon_curve.update()
        self.wait()

        def rotate_half_points(points, alpha):
            static_part = points[:len(points)//2]
            about_point = points[len(points)//2]
            mat = rotation_matrix(-PI/2 * alpha, OUT)
            rotated_part = [
                np.dot((point - about_point), mat.T) + about_point
                for point in reversed(static_part)
            ]
            return static_part + [about_point] + rotated_part

        def rotate_half_curve(mobject, alpha):
            corners = mobject.corners
            new_corners = rotate_half_points(corners, alpha)
            mobject.set_points_as_corners(new_corners)
            return mobject

        for it in range(10):
            rotated_curve = VMobject().set_points_as_corners(rotate_half_points(dragon_curve.corners, 1))
            self.play(
                UpdateFromAlphaFunc(dragon_curve, rotate_half_curve),
                self.camera.auto_zoom(rotated_curve, margin=1)
            )
            current_corners = rotate_half_points(dragon_curve.corners, 1)
            current_corners = current_corners + current_corners[-1::-1]
            dragon_curve.set_points_as_corners(current_corners)
            dragon_curve.corners = current_corners

        self.wait()



        # manim -pqh test2.py Dragon



class VonKoch(VMobject):
    def __init__(self, number_of_iterations: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.lines = Line(5*RIGHT, 5*LEFT, **kwargs)
        self.add(self.lines)
        for _ in range(number_of_iterations):
            for line in self:
                self.add(*self.get_iteration(line, **kwargs))
                self.remove(line)


    def get_iteration(self, side: VMobject, **kwargs) -> tuple:
        start, end = side.get_start(), side.get_end()
        first_point, second_point = 2/3 * start + 1/3 * end, 1/3 * start + 2/3 * end
        new_line_1 = Line(start, first_point, **kwargs)
        new_line_2 = Line(first_point, second_point, **kwargs).rotate(-PI/3, about_point=first_point)
        new_line_3 = Line(first_point, second_point, **kwargs).rotate(PI/3, about_point=second_point)
        new_line_4 = Line(second_point, end, **kwargs)
        return VMobject().add(new_line_1, new_line_2, new_line_3, new_line_4)



class VonKochFractal(Scene):
    def construct(self):
        vonkoch = VonKoch(0, stroke_width=0.5).shift(DOWN)
        self.add(vonkoch)
        for k in range(8):
            self.play(Transform(vonkoch, VonKoch(k, stroke_width=0.5).shift(DOWN)))


    # manim -pqh test2.py VonKochFractal




def MandelbrotSet0(max_steps: int = 50, resolution: int = 1080, x_range: np.ndarray = np.array([-2, 0.5]), y_range: np.ndarray = np.array([-1.15, 1.15])) -> np.ndarray:
        x_m, x_M, y_m, y_M = x_range[0], x_range[1], y_range[0], y_range[1]
        resolution_tuple = resolution, int(np.abs(x_M - x_m)/np.abs(y_M - y_m)*resolution)
        dx, dy = np.abs(x_M - x_m)/resolution_tuple[1], np.abs(y_M - y_m)/resolution_tuple[0]
        array = np.zeros(resolution_tuple, dtype='uint8')
        for n in range(len(array)):
            for p in range(len(array[0])):
                c = complex(x_m + p*dx, y_M - n*dy)
                z = c
                i = 0
                while i < max_steps and np.abs(z) <= 2:
                    z = z**2 + c
                    i += 1
                if i == max_steps:
                    array[n, p] = 0
                else:
                    array[n, p] = 67*i*(255/max_steps)
        return array

config.background_color=REANLEA_BACKGROUND_COLOR_OXFORD_BLUE
class Mandel2(Scene):
    def construct(self):
        x=ValueTracker(10)
        mandelbrot_set = ImageMobject(MandelbrotSet0(
            max_steps=x.get_value(),
            resolution=1080,
        )).set_z_index(4)
        mandelbrot_set_complex_plane = ComplexPlane(
            x_range = np.array([-7.5, 7.5]),
            y_range = np.array([-4, 4]),
            x_length = mandelbrot_set.height/2,
            y_length = mandelbrot_set.width/2
        )
        old_height = mandelbrot_set.height
        mandelbrot_set.height = old_height
        

        mandelbrot_set_complex_plane.move_to(mandelbrot_set).scale(.5)

        
        self.play(
            FadeIn(mandelbrot_set),
        )
        self.wait(2)
        

        # manim -pql test2.py Mandel2

        # manim -sqk test2.py Mandel2



class ImageFromArray(Scene):
    def construct(self):

        def interpolate_color_array(color1: Color, color2: Color, alpha: np.ndarray):
            arr = []
            for i,j in enumerate(alpha):
                arr += [color_to_int_rgba(interpolate_color(color1, color2, j))]
            return np.array(arr)

        arr = np.zeros((10,10))
        field = np.zeros((*arr.shape,4), "uint8")
        for i, j in enumerate(arr):
            field[i] = interpolate_color_array(BLUE, GREEN, j)

        image = ImageMobject(field)
        
        
        image.height = 7
        self.play(
            FadeIn(image)
        )
        self.wait()

        # manim -sqk test2.py ImageFromArray




class EmojiImageMobject(ImageMobject):
    def __init__(self, emoji, **kwargs):
        emoji_code = "-".join(f"{ord(c):x}" for c in emoji)
        emoji_code = emoji_code.upper()  # <-  needed for openmojis
        url = f"https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/618x618/{emoji_code}.png"
        im = Image.open(requests.get(url, stream=True).raw)
        emoji_img = np.array(im.convert("RGBA"))
        ImageMobject.__init__(self, emoji_img, **kwargs)


class imoji(Scene):
    def construct(self):
        self.camera.background_color = YELLOW_A
        em = EmojiImageMobject("🐶").scale(1.1)
        self.add(em)


        # manim -pqh test2.py imoji




class test_x(Scene):
    def construct(self):

        theta_tracker = ValueTracker(.01)

        vect_1=Arrow(start=LEFT,end=RIGHT,max_tip_length_to_length_ratio=0.125, buff=0).set_color(PURE_RED).set_opacity(0.85)
        vect_1_lbl=MathTex("u").scale(1).next_to(vect_1,0.5*DOWN).set_color(PURE_RED)
        vect_1_moving=Arrow(start=LEFT,end=RIGHT,max_tip_length_to_length_ratio=0.125, buff=0).set_color(PURE_RED).set_opacity(0.85)
        vect_1_ref=vect_1_moving.copy()
        vect_1_moving.rotate(
            theta_tracker.get_value() * DEGREES, about_point=vect_1_moving.get_start()
        )

        ang=Angle(vect_1, vect_1_moving, radius=1, other_angle=False).set_stroke(color=PURE_GREEN, width=3).set_z_index(-1)
        ang_lbl = MathTex(r"\theta =").move_to(
            Angle(
                vect_1, vect_1_moving, radius=.85 + 3 * SMALL_BUFF, other_angle=False
            ).point_from_proportion(.5)                 # Gets the point at a proportion along the path of the VMobject.
        ).scale(.5)

        ang_theta=DecimalNumber(unit="^o").scale(.5)

        #projec_line=DashedLine(start=dot3.get_center(), end=np.array((dot3.get_center()[0],0,0)), stroke_width=1).set_color(REANLEA_AQUA_GREEN).set_z_index(-2)
        #d_line_1=DashedLine(start=vect_1_moving.get_end(), end=np.array((vect_1_moving.get_end()[0],0,0))).set_stroke(color=REANLEA_AQUA_GREEN, width=1)
        projec_line=always_redraw(
            lambda : DashedLine(start=vect_1_moving.get_end(), end=np.array((vect_1_moving.get_end()[0],0,0))).set_stroke(color=REANLEA_AQUA_GREEN, width=1)
        )


        vect_1_moving.add_updater(
            lambda x: x.become(vect_1_ref.copy()).rotate(
                theta_tracker.get_value() * DEGREES, about_point=vect_1_moving.get_start()
            )
        )
        

        ang.add_updater(
            lambda x: x.become(Angle(vect_1, vect_1_moving, radius=1, other_angle=False).set_stroke(color=PURE_GREEN, width=3).set_z_index(-1))
        )


        ang_lbl.add_updater(
            lambda x: x.move_to(
                Angle(
                    vect_1, vect_1_moving, radius=.85 + 13*SMALL_BUFF, other_angle=False
                ).point_from_proportion(0.5),
                aligned_edge=RIGHT
            )
        )
        #ang_theta.add_updater(lambda x: x.next_to(ang_lbl, RIGHT))

        ang_theta.add_updater( lambda x: x.set_value(theta_tracker.get_value()).next_to(ang_lbl, RIGHT))
        






        self.play(Create(vect_1))
        self.wait()
        self.play(
            Write(vect_1_moving),
        )
        

        

        





        '''bez_arr_1=bend_bezier_arrow().flip(DOWN).move_to(2.5*LEFT + 0.1*UP).flip(LEFT).rotate(45*DEGREES)

        with RegisterFont("Fuzzy Bubbles") as fonts:
            text_20=Text("unit vector", font=fonts[0]).scale(0.45)
            text_20.set_color_by_gradient(REANLEA_TXT_COL).shift(3*RIGHT)
        text_20.move_to(.75*LEFT+ 0.2*UP).rotate(20*DEGREES)'''

        sgn_pos_1=MathTex("+").scale(.75).set_color(PURE_GREEN).move_to(6.5*RIGHT)
        sgn_pos_2=Circle(radius=0.2, color=PURE_GREEN).move_to(sgn_pos_1.get_center()).set_stroke(width= 1)
        sgn_pos=VGroup(sgn_pos_1,sgn_pos_2)



        #self.add(bez_arr_1)
        #self.add(text_20)
        self.add(sgn_pos)
        self.wait()
        '''self.play(
            Write(vect_1),
            lag_ratio=0.5
        )'''
        self.play(Write(vect_1_lbl))
        self.wait(2)
        '''self.play(
            Unwrite(vect_1.reverse_direction()),
            Uncreate(bez_arr_1)
        )'''
        
        self.play(theta_tracker.animate.set_value(40))
        self.wait()
        #self.add(ang_lbl, ang_theta)
        self.play(
            Create(ang)
        )
        self.play(
            Write(ang_lbl),
            Write(ang_theta),
            lag_ratio=0.5
        )
        self.wait(1.25)
        bra_1=always_redraw(
            lambda : BraceBetweenPoints(
                point_1=vect_1.get_start(),
                point_2=np.array((vect_1_moving.get_end()[0],0,0)),
                direction=DOWN,
                color=REANLEA_SLATE_BLUE_LIGHTER
            ).set_stroke(width=.1)
        )
        self.play(
            Write(projec_line),
            Create(bra_1)
        )
        self.wait(2)
        self.play(
            theta_tracker.animate.increment_value(80),
            ang_lbl.animate.set_color(RED), 
            ang_theta.animate.set_color(RED),
            ang.animate.set_stroke(color=RED, width=3),
            run_time=2
        )
        
        self.wait(2)
        

        # manim -pqh test2.py test_x

        # manim -sqk test2.py test_x



class test_y(Scene):
    def construct(self):

        theta_tracker = ValueTracker(.01)
        start = .01
        

        vect_1=Arrow(start=LEFT,end=RIGHT,max_tip_length_to_length_ratio=0.125, buff=0).set_color(PURE_RED)
        vect_1_moving=Arrow(start=LEFT,end=RIGHT,max_tip_length_to_length_ratio=0.125, buff=0).set_color(PURE_RED).set_opacity(0.85)
        vect_1_ref=vect_1_moving.copy()
        vect_1_moving.rotate(
            theta_tracker.get_value() * DEGREES, about_point=vect_1_moving.get_start()
        )

        ang=Angle(vect_1, vect_1_moving, radius=1, other_angle=False).set_stroke(color=PURE_GREEN, width=3).set_z_index(-1)

        ang_theta=DecimalNumber(unit="^o").scale(.5).move_to(RIGHT+UP)

        



        vect_1_moving.add_updater(
            lambda x: x.become(vect_1_ref.copy()).rotate(
                theta_tracker.get_value() * DEGREES, about_point=vect_1_moving.get_start()
            )
        )
        

        ang.add_updater(
            lambda x: x.become(Angle(vect_1, vect_1_moving, radius=1, other_angle=False).set_stroke(color=PURE_GREEN, width=3).set_z_index(-1))
        )


        ang_theta.add_updater( lambda x: x.set_value(theta_tracker.get_value()).move_to(RIGHT+UP))
        

                              

        x_var = Variable(start, MathTex(r"\theta"), num_decimal_places=2)
        sqr_var = Variable(np.cos(start*DEGREES), '', num_decimal_places=2).move_to(DOWN+RIGHT)           #MathTex("u","\cdot",r"cos(\theta)")
        sqr_var_lbl_right=MathTex("\cdot","u").arrange(RIGHT, buff=0.2).move_to(2.4*RIGHT+DOWN).set_color(PURE_RED)
        sqr_var_lbl_left=MathTex("u","\cdot",r"cos(\theta)").arrange(RIGHT,buff=0.2).move_to(LEFT+DOWN)
        sqr_var_lbl_left[0:2].set_color(PURE_RED)
        sqr_var_lbl_left[2][0:3].set_color(PURE_GREEN)
        sqr_var_grp=VGroup(sqr_var_lbl_left,sqr_var, sqr_var_lbl_right).scale(0.65)
        #Group(x_var, sqr_var).arrange(DOWN)

        sqr_var.add_updater(lambda v: v.tracker.set_value(np.cos(x_var.tracker.get_value()*DEGREES)))  #very important !!! step
        


        self.play(
            Create(vect_1),
            Write(vect_1_moving),
            
        )
        self.play(
            theta_tracker.animate.set_value(40),
        )
        self.wait()
        self.play(
            Create(ang)
        )
        self.play(
            Write(ang_theta),
            Create(sqr_var_grp)
        )
        self.wait(2)
    
        self.play(
            theta_tracker.animate.increment_value(80),
            ang_theta.animate.set_color(RED),
            sqr_var_lbl_left[2][4].animate.set_color(RED),
            ang.animate.set_stroke(color=RED, width=3),
            x_var.tracker.animate.set_value(80), rate_func=linear,
            run_time=2
        )
        
        self.wait(2)
        

        # manim -pqh test2.py test_y

        # manim -sqk test2.py test_y


class VariableExample(Scene):
            def construct(self):
                start = .01                      

                x_var = Variable(start, MathTex(r"\theta"), num_decimal_places=2)
                sqr_var = Variable(np.cos(start*DEGREES), '', num_decimal_places=2)           #MathTex("u","\cdot",r"cos(\theta)")
                sqr_var_lbl_right=MathTex("\cdot","u").arrange(RIGHT, buff=0.2).move_to(2.4*RIGHT).set_color(PURE_RED)
                sqr_var_lbl_left=MathTex("u","\cdot",r"cos(\theta)").arrange(RIGHT,buff=0.2).move_to(LEFT)
                sqr_var_grp=Group(sqr_var_lbl_left,sqr_var, sqr_var_lbl_right)
                #Group(x_var, sqr_var).arrange(DOWN)

                sqr_var.add_updater(lambda v: v.tracker.set_value(np.cos(x_var.tracker.get_value()*DEGREES)))  #very important !!! step

                self.add(sqr_var_grp)
                self.play(x_var.tracker.animate.set_value(180), run_time=2, rate_func=linear)
                self.wait(0.1)


                # manim -pqh test2.py VariableExample

                # manim -sqk test2.py VariableExample
###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix"