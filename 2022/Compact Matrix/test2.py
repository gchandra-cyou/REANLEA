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

'''from numba import jit
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


'''



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


                txt_2_vect=MathTex(r"\vec{1}").move_to(2*LEFT+2*UP)

                self.play(Write(txt_2_vect))


                # manim -pqh test2.py VariableExample

                # manim -sqk test2.py VariableExample


class bezier_test(Scene):
    def construct(self):
        glowing_circle=get_glowing_surround_circle(circle=Circle(radius=0.2))
        stripe=get_stripe()
        surround_bezier=get_surround_bezier(text=Text("qwerty"))
        arrow_cubic_bezier_up=ArrowCubicBezierUp()
        arrow_quadric_bezier_down=ArrowQuadricBezierDown(text=Text("qwerty"))
        under_line_bez_arrow=under_line_bezier_arrow()
        bend_bez_arrow=bend_bezier_arrow().rotate(-30*DEGREES).scale(0.75).set_color(REANLEA_TXT_COL)
        bend_bezier_arrow_indi=bend_bezier_arrow_indicate().flip(RIGHT).scale(.75).rotate(-20*DEGREES).set_color(REANLEA_TXT_COL)
        
        grp=VGroup(glowing_circle,stripe,surround_bezier,arrow_cubic_bezier_up,arrow_quadric_bezier_down,under_line_bez_arrow,bend_bez_arrow, bend_bezier_arrow_indi)
        grp.arrange(RIGHT, buff=0.25).scale(0.5)

        self.play(
            Create(grp)
        )
        self.wait(2)

        # manim -pqh test2.py bezier_test

        # manim -sqk test2.py bezier_test


class test_mirror(Scene):
    def construct(self):
        mirror=get_mirror()

        self.play(
            Write(mirror)
        )

        self.wait(2)

        # manim -pqh test2.py test_mirror

        # manim -sqk test2.py test_mirror


class glow_rect(Scene):
    def construct(self):
        cir=Line().scale(5)
        x=line_highlight(buff_max=cir.get_length(), factor=.15, opacity_factor=.35).move_to(cir.get_center())
        
        self.play(Create(x))
        self.add(cir)
        
        
        self.wait(2)

        # manim -pqh test2.py glow_rect

        # manim -sqk test2.py glow_rect



class imoji2(Scene):
    def construct(self):
        zo=ValueTracker(0)
        d_d_arr_3=DashedDoubleArrow(
            start=LEFT, end=RIGHT,dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).shift(.3*UP).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA)

        d_d_arr_3_ref=d_d_arr_3.copy()

        d_d_arr_3.add_updater(
            lambda x: x.become(d_d_arr_3_ref.copy()).rotate(
                zo.get_value()*DEGREES , about_point=RIGHT+0.3*UP
            )
        )

        d_d_arr_4=DashedDoubleArrow(
            start=LEFT, end=RIGHT,dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA)



        self.wait()
        
        self.add(d_d_arr_4)

        self.play(
            d_d_arr_4.animate.shift(0.3*UP)
        )
        self.add(d_d_arr_3)
        self.play(
            d_d_arr_4.animate.set_opacity(0)
        )

        self.wait()
        self.play(
            zo.animate.set_value(-180)
        )
        self.wait(2)
        
        


        # manim -pqh test2.py imoji2



class lbl_test(Scene):
    def construct(self):

        fact=ValueTracker(0)

        vect=Arrow(start=LEFT,end=RIGHT)

        vect.add_updater(
            lambda z : z.become(
                Arrow(start=LEFT,end=np.array((1,0,0))*(1+fact.get_value()))
            )
        )
        value=DecimalNumber().set_color_by_gradient(REANLEA_MAGENTA).set_sheen(-0.1,LEFT).move_to(2*UP)

        value.add_updater(
            lambda x : x.set_value(1+fact.get_value())
        )

        self.play(
            Write(vect),
            Write(value)
        )
        self.wait()
        self.play(
            fact.animate.set_value(2)
        )

        self.wait(2)

        # manim -pqh test2.py lbl_test

        # manim -sqk test2.py lbl_test


class lbl_test_1(Scene):
    def construct(self):

        vect_4_lbl_eqn=MathTex(r"\vec{x}","=",r"x \cdot \vec{1}").scale(0.85).move_to(1.35*LEFT+ UP).set_color(PURE_RED)
        
        vect_4_lbl_eqn[0].set_color(PURE_GREEN)
        vect_4_lbl_eqn[2][0].set_color(PURE_GREEN)


        with RegisterFont("Cousine") as fonts:
            text_1 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Scaling Factor",
            )]).scale(0.24).set_color(REANLEA_GREY)

        text_1.move_to(1.7*UP+RIGHT)

        txt_blg_1=MathTex(r"\in", r"\mathbb{R}").set_color(REANLEA_TXT_COL).scale(0.7).move_to(1.35*UP+1.1*RIGHT)
        txt_blg_1[0].scale(0.65)
        txt_blg_1[1].set_color(REANLEA_BLUE_SKY)

        bez=bend_bezier_arrow_indicate().flip(RIGHT).move_to(1.4*UP+ 0.5*LEFT).scale(.75).rotate(-20*DEGREES).set_color(REANLEA_TXT_COL)

        grp=VGroup(vect_4_lbl_eqn, text_1, txt_blg_1, bez)

        self.play(
            Write(vect_4_lbl_eqn),
            Write(text_1),
            Write(txt_blg_1),
            Create(bez)
        )
        self.wait()

        self.play(
            grp.animate.shift(3*LEFT)
        )
        self.wait(2)



        vect_4_lbl_eqn=MathTex(r"\vec{x}","=",r"x \cdot \vec{1}").scale(0.85)#.move_to(line_1.n2p(-1)+ 2.9*UP).set_color(PURE_RED)
        vect_3_lbl=MathTex(r"\vec{2}").scale(.85).set_color(REANLEA_YELLOW_GREEN)#.move_to(line_1.n2p(-1)+ 0.9*UP)
        vect_3_lbl_eqn_dumy=MathTex(r"\vec{2}","=",r"2 \cdot \vec{1}").scale(.85).set_color(REANLEA_YELLOW_GREEN)#.move_to(line_1.n2p(-1)+ 2.9*UP)

        vect_4_lbl_eqn.shift(vect_3_lbl.get_center()+UP - vect_3_lbl_eqn_dumy[0].get_center())
        vect_4_lbl_eqn[0].set_color(PURE_GREEN)
        vect_4_lbl_eqn[2][0].set_color(PURE_GREEN)

        with RegisterFont("Cousine") as fonts:
            text_1 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Scaling Factor",
            )]).scale(0.24).set_color(REANLEA_GREY)

        text_1.move_to(1.7*UP+RIGHT)

        txt_blg_1=MathTex(r"\in", r"\mathbb{R}").set_color(REANLEA_TXT_COL).scale(0.7).move_to(1.35*UP+1.1*RIGHT)
        txt_blg_1[0].scale(0.65)
        txt_blg_1[1].set_color(REANLEA_BLUE_SKY)


        bez=bend_bezier_arrow_indicate().flip(RIGHT).move_to(1.4*UP+ 0.5*LEFT).scale(.75).rotate(-20*DEGREES).set_color(REANLEA_TXT_COL)


        # manim -pqh test2.py lbl_test_1

        # manim -sqk test2.py lbl_test_1



class ArrangeSumobjectsExample(Scene):
                def construct(self):
                    s= VGroup(*[Dot().shift(i*0.1*RIGHT*np.random.uniform(-1,1)+UP*np.random.uniform(-1,1)) for i in range(-15,15)])
                    s.shift(UP).set_color_by_gradient(REANLEA_BLUE, REANLEA_GREEN_LIGHTER)
                    s2= s.copy().set_color_by_gradient(REANLEA_ORANGE_DARKER,REANLEA_VIOLET,REANLEA_PURPLE)
                    s2.arrange_submobjects()
                    s2.shift(DOWN)
                    s3= VGroup(*[Dot().shift(i*0.1*RIGHT*np.random.uniform(-6,6)) for i in range(-15,15)])
                    s3.shift(2*DOWN).set_color_by_gradient(REANLEA_BLUE, PURE_GREEN, REANLEA_GREY_DARKER,REANLEA_VIOLET,REANLEA_AQUA_GREEN)
                    #self.add(s,s2)
                    g=VGroup(s,s2, s3)
                    self.play(Create(g, run_time=4))
                    self.wait()


                    # manim -pqh test2.py ArrangeSumobjectsExample



class tex_fill(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        with RegisterFont("Merienda One") as fonts:
            txt_4=Text("F i e l d", font=fonts[0]).scale(0.65).set_color(REANLEA_TXT_COL).move_to(5.5*LEFT+3.35*UP)

        stripe_1=get_stripe(factor=0.1, buff_max=1.75).move_to(5.35*LEFT+3*UP)

        fld_grp=VGroup(txt_4,stripe_1).move_to(3.25*UP)

        with RegisterFont("Cousine") as fonts:
            fld_dfn_tx_1 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "A Field",
                "is a set, together with two laws of composition :"
            )]).scale(0.35)

            fld_dfn_tx_2 = Text(" called addition : ", font=fonts[0]).scale(0.35)

            fld_dfn_tx_3 = Text(" called multiplication : ", font=fonts[0]).scale(0.35)

            fld_dfn_4 = Text(", which satisfies the following axioms : ", font=fonts[0]).scale(0.35).to_edge(edge=LEFT, buff=2).shift(0.5*UP)

            fld_dfn_tx_5 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "I. Addition makes",
                "into an abelian group",
                " Its Identity element is denoted by"
            )]).scale(0.35)

            fld_dfn_tx_6 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "II. Multiplication is Commutative, and it makes",
                "the set of all non-zero elements of",
                "into an abelian group.",
                "Its Identity element is denoted by"
            )]).scale(0.35)

            fld_dfn_tx_7 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "III. DISTRIBUTIVE LAW :",
                
            )]).scale(0.35).to_edge(edge=LEFT, buff=2.25).shift(2.05*DOWN)



        fld_dfn_mtx_1=MathTex(r"(F,+,\cdot)").scale(0.65)

        fld_dfn_1=VGroup(fld_dfn_tx_1[0],fld_dfn_mtx_1,fld_dfn_tx_1[1]).arrange(RIGHT,buff=0.2).to_edge(edge=LEFT).shift(2*UP)#.move_to(2*UP+2*LEFT)
        


        fld_dfn_mtx_2_1=MathTex(r"\diamondsuit", r"F \times F", r"\xrightarrow{+}","  F ",",").scale(0.65)
        fld_dfn_mtx_2_1[0].scale(0.65).shift(.2*LEFT)

        fld_dfn_mtx_2_2=MathTex("a",",","b",r"\rightsquigarrow","a","+","b").scale(0.65)

        fld_dfn_2=VGroup(fld_dfn_mtx_2_1, fld_dfn_tx_2,fld_dfn_mtx_2_2).arrange(RIGHT,buff=0.2).to_edge(edge=LEFT, buff=2).shift(1.5*UP)
        fld_dfn_2[1].shift(0.05*DOWN)
        fld_dfn_2[2].shift(0.05*DOWN)


        fld_dfn_mtx_3_1=MathTex(r"\diamondsuit", r"F \times F", r"\xrightarrow{\cdot}","  F ",",").scale(0.65)
        fld_dfn_mtx_3_1[0].scale(0.65).shift(.2*LEFT)

        fld_dfn_mtx_3_2=MathTex("a",",","b",r"\rightsquigarrow","a",r"\cdot","b").scale(0.65)

        fld_dfn_3=VGroup(fld_dfn_mtx_3_1, fld_dfn_tx_3,fld_dfn_mtx_3_2).arrange(RIGHT,buff=0.2).to_edge(edge=LEFT, buff=2).shift(UP)
        fld_dfn_3[1].shift(0.05*DOWN)
        fld_dfn_3[2].shift(0.05*DOWN)


        fld_dfn_mtx_5=MathTex("F",r"(F,+).","0.").scale(0.65)
        fld_dfn_5_1=VGroup(fld_dfn_tx_5[0],fld_dfn_mtx_5[0],fld_dfn_tx_5[1],fld_dfn_mtx_5[1]).arrange(RIGHT, buff=0.2).to_edge(edge=LEFT, buff=2.5)
        fld_dfn_5_1[1].shift(0.05*UP)
        fld_dfn_5_1[3].shift(0.05*UP)
        fld_dfn_5_2=VGroup(fld_dfn_tx_5[2],fld_dfn_mtx_5[2]).arrange(RIGHT, buff=0.2).to_edge(edge=LEFT, buff=2.9).shift(.35*DOWN)
        fld_dfn_5_2[1].shift(0.05*UP)
        fld_dfn_5=VGroup(fld_dfn_5_1,fld_dfn_5_2)


        fld_dfn_mtx_6=MathTex(r"(F^{\times},\cdot),","F","1.").scale(0.65)

        fld_dfn_6_1=VGroup(fld_dfn_tx_6[0],fld_dfn_mtx_6[0]).arrange(RIGHT, buff=0.2).to_edge(edge=LEFT, buff=2.375).shift(.85*DOWN)
        fld_dfn_6_1[1].shift(0.05*UP)
        fld_dfn_6_2=VGroup(fld_dfn_tx_6[1],fld_dfn_mtx_6[1],fld_dfn_tx_6[2]).arrange(RIGHT, buff=0.2).to_edge(edge=LEFT, buff=2.95).shift(1.2*DOWN)
        fld_dfn_6_2[1].shift(0.05*UP)
        fld_dfn_6_3=VGroup(fld_dfn_tx_6[3],fld_dfn_mtx_6[2]).arrange(RIGHT, buff=0.2).to_edge(edge=LEFT, buff=2.95).shift(1.55*DOWN)
        fld_dfn_6_3[1].shift(0.05*UP)

        fld_dfn_6=VGroup(fld_dfn_6_1, fld_dfn_6_2,fld_dfn_6_3)


        fld_dfn_mtx_7=MathTex(r"a \cdot (b +c)","=",r"a \cdot b", "+", r"a \cdot c", ",",r"\forall", "a",",","b",",","c","\in F").scale(0.65).to_edge(edge=LEFT, buff=2.95).shift(2.4*DOWN)
        fld_dfn_mtx_7[6:].shift(0.35*RIGHT)
        fld_dfn_mtx_7[7:].shift(0.1*RIGHT)

        fld_dfn_7=VGroup(fld_dfn_tx_7,fld_dfn_mtx_7)

        fld_dfn=VGroup(fld_dfn_1,fld_dfn_2,fld_dfn_3,fld_dfn_4,fld_dfn_5,fld_dfn_6,fld_dfn_7).shift(1.5*RIGHT)




        self.play(
            Write(txt_4)
        )
        self.play(
            Write(stripe_1)
        )
        self.play(
            Write(fld_dfn),
            run_time=6
        )

        self.wait(2)

        # manim -pqh test2.py tex_fill

        # manim -sqk test2.py tex_fill

        # manim -pqk -r 1080,1920 test2.py tex_fill




class ArrangeInGrid(Scene):
                def construct(self):
                    boxes = VGroup(*[
                        Rectangle(REANLEA_BLUE_ALT, 0.5, 0.5).add(Text(str(i+1)).scale(0.5).set_color(REANLEA_SAFRON))
                        for i in range(24)
                    ])
                    #self.add(boxes)

                    boxes.arrange_in_grid(
                        buff=(0.25,.5),
                        col_alignments="lccccr",
                        row_alignments="uccd",
                        col_widths=[1, *[None]*4, 1],
                        row_heights=[1, *[None]*2, 1],
                        flow_order="dr"
                    )
                    self.play(
                        Create(boxes, run_time=4)
                    )
                    self.wait()


            # manim -pqh test2.py ArrangeInGrid



class GetRowLabelsExample(Scene):
                def construct(self):
                    table = MathTable(
                        [
                            ["(a,1)", "(a,2)","(a,3)"],
                            ["(b,1)", "(b,2)","(b,3)"]
                        ],
                        #row_labels=[Text("a"), Text("b")],
                        #col_labels=[Text("1"), Text("2"), Text("3")]
                    )

                    table.get_vertical_lines().set_stroke(width=2, color=PURE_GREEN)
                    table.get_horizontal_lines().set_stroke(width=2, color=PURE_RED)
                    #table.get_entries_without_labels()[1][0][1].set_color(REANLEA_BLUE_SKY)
                    ent=table.get_entries_without_labels()

                    for k in range(len(ent)):
                        ent[k][0][1].set_color(REANLEA_BLUE_SKY)
                        ent[k][0][3].set_color(REANLEA_ORANGE)

                    table_2=Table(
                        [
                            ["1","2","3"]
                        ],
                        h_buff=2.25,
                    ).next_to(table, UP)
                    table_2.get_vertical_lines().set_opacity(0)

                    table_2_lbl=Text("B").next_to(table_2, UP+4*RIGHT).scale(.5)
                    table_2_lbl_ln=Line().set_stroke(width=2,color=REANLEA_GREY).scale(.5).rotate(PI/4).next_to(table_2_lbl, .5*DOWN+.5*LEFT)
                    t_2_lbl=VGroup(table_2_lbl,table_2_lbl_ln)

                    table_3=MathTable(
                        [
                            ["a"],
                            ["b"]
                        ],
                        v_buff=.85
                    ).next_to(table,LEFT)
                    table_3.get_horizontal_lines().set_opacity(0)


                    table_3_lbl=Text("A").next_to(table_3, 2*UP+LEFT).scale(.5)
                    table_3_lbl_ln=Line().set_stroke(width=2,color=REANLEA_GREY).scale(.5).rotate(3*PI/4).next_to(table_3_lbl, .5*DOWN+.5*RIGHT)
                    t_3_lbl=VGroup(table_3_lbl,table_3_lbl_ln)

                    '''lab = table.get_row_labels()
                    for item in lab:
                        item.set_color(random_bright_color())'''
                    
                    sr_table=SurroundingRectangle(table, color=REANLEA_WELDON_BLUE ,corner_radius=.15).set_fill(color=REANLEA_WELDON_BLUE, opacity=0.25)

                    sr_table_2=Ellipse(width=8, color=REANLEA_WELDON_BLUE).set_fill(color=REANLEA_WELDON_BLUE, opacity=0.25).move_to(table_2.get_center())
                    
                    sr_table_3=Ellipse(width=3, color=REANLEA_WELDON_BLUE).set_fill(color=REANLEA_WELDON_BLUE, opacity=0.25).move_to(table_3.get_center()).rotate(PI/2)


                    self.play(
                        Create(table, run_time=3)
                    )
                    self.play(
                        Create(sr_table)
                    )
                    self.play(
                        Write(table_2)
                    )
                    self.play(
                        Create(sr_table_2)
                    )
                    self.play(
                        Write(table_3)
                    )
                    self.play(
                        Create(sr_table_3)
                    )
                    self.play(Write(t_2_lbl))

                    self.play(Write(t_3_lbl))


                    self.wait()


            # manim -pqh test2.py GetRowLabelsExample

            # manim -sqk test2.py GetRowLabelsExample




class axEx(Scene):
    def construct(self):
        ax_1=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                "include_ticks": False,
            }, 
        ).set_color(REANLEA_WHITE).scale(.5).shift(3*LEFT).set_z_index(1)
        ax_2=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
            }, 
        ).set_color(REANLEA_WHITE).scale(.5).shift(3*LEFT).set_z_index(-1)

        ax_1_x_lbl=ax_1.get_x_axis_label(
            Tex("$x$-axis").scale(0.65),
            edge=DOWN,
            direction=DOWN,
            buff=0.3
        ).shift(RIGHT)
        ax_1_y_lbl=ax_1.get_y_axis_label(
            Tex("$y$-axis").scale(0.65).rotate(90 * DEGREES),       #label rotation
            edge=LEFT,
            direction=LEFT,
            buff=0.3,
        ).shift(UP)

        ax_1_lbl=VGroup(ax_1_x_lbl,ax_1_y_lbl)

        ax_1_coords=ax_1.copy().add_coordinates()
        
    


        dot_ax_1=Dot(ax_1.coords_to_point(0,0), color=PURE_GREEN).set_sheen(-0.4,DOWN).set_z_index(2)
        

        sgn_pos_1=MathTex("+").scale(.5).set_color(PURE_GREEN)
        sgn_pos_2=Circle(radius=0.15, color=PURE_GREEN).move_to(sgn_pos_1.get_center()).set_stroke(width= 1)
        sgn_pos=VGroup(sgn_pos_1,sgn_pos_2)

        sgn_neg_1=MathTex("-").scale(.5).set_color(PURE_RED).move_to(6.5*LEFT)
        sgn_neg_2=Circle(radius=0.15, color=PURE_RED).move_to(sgn_neg_1.get_center()).set_stroke(width= 1)
        sgn_neg=VGroup(sgn_neg_1,sgn_neg_2)

        r1=Polygon(ax_1.c2p(0,-1.5),ax_1.c2p(5.5,-1.5),ax_1.c2p(5.5,4.5),ax_1.c2p(0,4.5)).set_opacity(0)
        r1.set_fill(color=REANLEA_BLUE_LAVENDER, opacity=0.25)


        r2=Polygon(ax_1.c2p(0,-1.5),ax_1.c2p(0,4.5),ax_1.c2p(-1.5,4.5),ax_1.c2p(-1.5,-1.5)).set_opacity(0)
        r2.set_fill(color=REANLEA_CHARM, opacity=0.25)

        r3=Polygon(ax_1.c2p(-1.5,0),ax_1.c2p(5.5,0),ax_1.c2p(5.5,4.5),ax_1.c2p(-1.5,4.5)).set_opacity(0)
        r3.set_fill(color=REANLEA_BLUE_LAVENDER, opacity=0.25)

        r4=Polygon(ax_1.c2p(-1.5,0),ax_1.c2p(5.5,0),ax_1.c2p(5.5,-1.5),ax_1.c2p(-1.5,-1.5)).set_opacity(0)
        r4.set_fill(color=REANLEA_CHARM, opacity=0.25)

        

        self.play(
            Write(ax_1)
        )
        self.play(
            Write(ax_1_lbl)
        )
        self.play(
            Create(dot_ax_1)
        )
        self.play(
            Create(r1.reverse_direction())
        )
        self.play(
            Write(sgn_pos.move_to(ax_1.c2p(5,.5)))
        )
        self.play(
            FadeOut(sgn_pos),
            Uncreate(r1),
            lag_ratio=.5
        )
        self.play(
            Create(r2)
        )
        self.play(
            Write(sgn_neg.move_to(ax_1.c2p(-1,.5)))
        )
        self.play(
            FadeOut(sgn_neg),
            Uncreate(r2),
            lag_ratio=.5
        )
        self.wait(2)
        self.play(
            Create(r3)
        )
        self.play(
            Write(sgn_pos.move_to(ax_1.c2p(.5,4)))
        )
        self.play(
            FadeOut(sgn_pos),
            Uncreate(r3),
            lag_ratio=.5
        )
        self.play(
            Create(r4)
        )
        self.play(
            Write(sgn_neg.move_to(ax_1.c2p(.5,-1)))
        )
        self.play(
            FadeOut(sgn_neg),
            Uncreate(r4),
            lag_ratio=.5
        )
        self.play(
            Create(ax_2)
        )
        self.play(
            ax_1_x_lbl.animate.shift(.5*DOWN),
            ax_1_y_lbl.animate.shift(.5*LEFT),
            Write(ax_1_coords),
            FadeOut(ax_1)
        )
        
        

        


        self.wait(2)


        # manim -pqh test2.py axEx

        # manim -sqk test2.py axEx





class PointCloudDotExample(Scene):
    def construct(self):
        cloud_1 = PointCloudDot(color=RED)
        cloud_2 = PointCloudDot(stroke_width=4, radius=1)
        cloud_3 = PointCloudDot(density=15)

        group = Group(cloud_1, cloud_2, cloud_3).arrange()

        self.wait(2)
        self.play(
            group.animate.shift(RIGHT)
        )
        self.wait(2)


        # manim -pqh test2.py PointCloudDotExample



class ExamplePoint(Scene):
            def construct(self):
    
                for i in range(10):
                    point = Point(location=[0.63 * np.random.randint(-4, 4), 0.37 * np.random.randint(-4, 4), 0], color=REANLEA_BLUE_LAVENDER)
                    self.add(point)
                
                self.play(
                    Write(point)
                )


        # manim -pqh test2.py ExamplePoint




class Ex(Scene):
    def construct(self):


        #dots
        eps=.1
        line_1=Line(ORIGIN,2*RIGHT)

        dots=VGroup(
            *[
                Dot(point=i*RIGHT + j*UP,radius=0.0125)
                    for i in np.arange(eps,2+eps,eps) 
                    for j in np.arange(eps,2+eps,eps)
            ]
        )
        
        dots.set_color(REANLEA_BLUE_LAVENDER)

        line_2=Line(start=dots[0].get_center(),end=dots[-1].get_center())
        l_2_lngth=DecimalNumber(line_2.get_length()).shift(UP)
        


        #play
        self.wait()
        self.add(line_1)

        self.play(
            Create(dots)
        )

        self.wait(2)



        # manim -pqh test2.py Ex

        # manim -sqk test2.py Ex
        

class Ex2(Scene):
    def construct(self):
        ax=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
            }, 
        ).set_color(REANLEA_YELLOW_CREAM).scale(.5).shift(3*LEFT).set_z_index(-1)
        #a[0].move_to(ax.c2p(0,0))
        #a[1].move_to(ax.c2p(b,0))

        a=square_cloud(x_eps=.5, y_eps=.5,x_min=2, x_max=4, y_max=0, rad=DEFAULT_DOT_RADIUS, sheen_factor=0, col=PURE_RED).set_z_index(2)
        #a[0].set_opacity(0)
        a.shift(ax.c2p(0,0)[0]*RIGHT,ax.c2p(0,0)[1]*UP)

        b=square_cloud(x_eps=.5, y_eps=.5,x_max=0,y_min=1, y_max=3,  rad=DEFAULT_DOT_RADIUS, sheen_factor=0, col=PURE_GREEN).set_z_index(2)
        #b[0].set_opacity(0)   
        b.shift(ax.c2p(0,0)[0]*RIGHT,ax.c2p(0,0)[1]*UP)

        #m=a[2].get_center()[0]-a[1].get_center()[0]    


        c=square_cloud(x_eps=.5, y_eps=.5,x_min=2, x_max=4,y_min=1, y_max=3, rad=DEFAULT_DOT_RADIUS, sheen_factor=0)
        c.shift(ax.c2p(0,0)[0]*RIGHT,ax.c2p(0,0)[1]*UP)

        

        z=MathTex(ax.c2p(0,0)).scale(.5).to_edge(RIGHT).shift(2.5*DOWN)
        z0=MathTex(c[10].get_center()).scale(.5).to_edge(RIGHT).shift(3*DOWN)

        #c[0].shift(ax.c2p(0,0)[0]*RIGHT,ax.c2p(0,0)[1]*UP)

        '''for i in range(len(a)):
            a[i].move_to(ax.c2p(i*.5,0))
        
        for i in range(len(b)):
            b[i].move_to(ax.c2p(0,i*.5))'''

        

        

        self.play(
            Write(ax)
        )

        self.play(
            Write(c)
        )
        self.play(
            Write(z),
            Write(z0)
        )

        self.play(
            Create(a)
        )
        self.play(
            Create(b)
        )
        
        
        self.wait(2)

        # manim -pqh test2.py Ex2

        # manim -sqk test2.py Ex2



class Ex3(Scene):
    def construct(self):
        ax_2=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
            }, 
        ).set_color(REANLEA_YELLOW_CREAM)#.scale(.5).shift(3*LEFT).set_z_index(-1)

        dots_A_1=square_cloud(x_max=5, x_eps=1,col=REANLEA_GREEN_AUQA, rad=DEFAULT_DOT_RADIUS,  y_max=0).set_z_index(2)
        dots_A_1.shift(3*LEFT)
        

        dots_B_1=square_cloud(x_max=0,col=REANLEA_BLUE_SKY, rad=DEFAULT_DOT_RADIUS, y_eps=1, y_max=4).set_z_index(2)
        


        self.play(
            Create(ax_2)
        )
        self.play(
            Create(dots_A_1)
        )
        self.play(
            Create(dots_B_1)
        )
        
        self.wait(2)


        # manim -pqh test2.py Ex3

        # manim -sqk test2.py Ex3


class Ex4(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        ax_2=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                "include_ticks": False,
            }, 
        ).set_color(REANLEA_YELLOW_CREAM).scale(.5)#.shift(3*LEFT).set_z_index(1)

        s_fact=ax_2.c2p(0,0)[0]*RIGHT+ax_2.c2p(0,0)[1]*UP

        dot_a_1=Dot(ax_2.coords_to_point(2,0), color=REANLEA_GREEN_AUQA).set_sheen(-0.4,DOWN).set_z_index(2)
        dot_a_2=Dot(ax_2.coords_to_point(4,0), color=REANLEA_GREEN_AUQA).set_sheen(-0.4,DOWN).set_z_index(2)
        dots_A=VGroup(dot_a_1,dot_a_2)

        dot_b_1=Dot(ax_2.coords_to_point(0,1), color=REANLEA_BLUE_SKY).set_sheen(-0.4,DOWN).set_z_index(2)
        dot_b_2=Dot(ax_2.coords_to_point(0,2), color=REANLEA_BLUE_SKY).set_sheen(-0.4,DOWN).set_z_index(2)
        dot_b_3=Dot(ax_2.coords_to_point(0,3), color=REANLEA_BLUE_SKY).set_sheen(-0.4,DOWN).set_z_index(2)
        dots_B=VGroup(dot_b_1,dot_b_2,dot_b_3)


        dots_A_1=square_cloud(x_min=2,x_max=4,x_eps=1, y_max=0, col=REANLEA_GREEN_AUQA, rad=DEFAULT_DOT_RADIUS).shift(s_fact).set_z_index(2)
        dots_B_1=square_cloud(x_max=0,y_min=1,y_max=3, y_eps=1, col=REANLEA_BLUE_SKY,rad=DEFAULT_DOT_RADIUS).shift(s_fact).set_z_index(2)
        dots_C_1=square_cloud(x_min=2,x_max=4, x_eps=1, y_min=1,y_max=3, y_eps=1, rad=DEFAULT_DOT_RADIUS).shift(s_fact).set_z_index(2)

        dots_in_grp=VGroup(dots_A_1,dots_B_1,dots_C_1)

        def sq_cld(
            eps=1,
            **kwargs
        ):  
            n=.75*(1/eps)
            dots_A_1=square_cloud(x_min=2,x_max=4,x_eps=eps, y_max=0, col=REANLEA_GREEN_AUQA, rad=DEFAULT_DOT_RADIUS/n).shift(s_fact).set_z_index(2)
            dots_B_1=square_cloud(x_max=0,y_min=1,y_max=3, y_eps=eps, col=REANLEA_BLUE_SKY,rad=DEFAULT_DOT_RADIUS/n).shift(s_fact).set_z_index(2)
            dots_C_1=square_cloud(x_min=2,x_max=4, x_eps=eps, y_min=1,y_max=3, y_eps=eps, rad=DEFAULT_DOT_RADIUS/n).shift(s_fact).set_z_index(2)

            dots=VGroup(dots_A_1,dots_B_1,dots_C_1)

            return dots

        
        dots_2=sq_cld(eps=.5)
        dots_3=sq_cld(eps=.25)
        dots_4=sq_cld(eps=.125)
        dots_5=sq_cld(eps=.0625)

        x_grp=VGroup(ax_2,dots_5).save_state()

        line_x=Line(start=dots_A_1[0].get_center(), end=dots_A_1[-1].get_center()).set_stroke(width=3, color=REANLEA_GREEN_AUQA).set_z_index(4.5)
        line_y=Line(start=dots_B_1[0].get_center(), end=dots_B_1[-1].get_center()).set_stroke(width=3, color=REANLEA_BLUE_SKY).set_z_index(4.5)
        
        x_1=dots_A_1[0].get_center()[0]
        x_2=dots_A_1[-1].get_center()[0]

        y_1=dots_B_1[0].get_center()[1]
        y_2=dots_B_1[-1].get_center()[1]

        ind_sq=Polygon([x_1,y_1,0],[x_2,y_1,0],[x_2,y_2,0],[x_1,y_2,0]).set_opacity(0).set_fill(color=REANLEA_BLUE_LAVENDER, opacity=0.25)
        line=Line(start=[x_1,y_1,0], end=[x_2,y_2,0])

           
        z=MathTex(dots_A_1[0].get_center()).scale(.5)
        z0=MathTex(ax_2.c2p(0,0)).scale(.5).shift(.5*UP)
        za=MathTex(dots_A[0].get_center()).scale(.5).shift(UP)
        zb=MathTex(s_fact).scale(.5).shift(1.5*UP)


        dt_1=Dot(line_x.get_center(), color=REANLEA_MAGENTA, radius=.075).set_sheen(-.4,DOWN).set_z_index(5)
        dt_2=Dot(line_y.get_center(), color=REANLEA_YELLOW, radius=.075).set_sheen(-.4,DOWN).set_z_index(5)
        dt_3=Dot([dt_1.get_center()[0],dt_2.get_center()[1],0], color=REANLEA_YELLOW, radius=.075).set_sheen(-.4,DOWN).set_z_index(5)





        self.play(
            Create(ax_2)
        )
        self.play(
            Create(dots_A),
            Create(dots_B)
        )
        
        self.wait(2)

        self.play(
            ReplacementTransform(dots_A,dots_A_1),
            ReplacementTransform(dots_B,dots_B_1)
        )
        self.play(
            Write(dots_C_1)
        )
        self.wait()

        self.play(
            ReplacementTransform(dots_in_grp, dots_2)
        )
        self.play(
            ReplacementTransform(dots_2,dots_3)
        )
        self.play(
            ReplacementTransform(dots_3,dots_4)
        )
        self.play(
            ReplacementTransform(dots_4,dots_5)
        )
        self.wait()

        
        self.wait()

        self.play(
            Write(line_x),
            Write(line_y),
            TransformMatchingShapes(dots_5,ind_sq)
        )
        self.wait(2)

        self.play(
            Write(dt_1),
            Write(dt_2),
            Write(dt_3)
        )

        self.wait(2)

        # manim -pqh test2.py Ex4

        # manim -pql test2.py Ex4

        # manim -sqk test2.py Ex4

        # manim -sql test2.py Ex4


class Ex5(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        ax_2=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                "include_ticks": False,
            }, 
        ).set_color(REANLEA_YELLOW_CREAM).scale(.5)#.shift(3*LEFT).set_z_index(1)

        s_fact=ax_2.c2p(0,0)[0]*RIGHT+ax_2.c2p(0,0)[1]*UP

        dot_a_1=Dot(ax_2.coords_to_point(2,0), color=REANLEA_GREEN_AUQA).set_sheen(-0.4,DOWN).set_z_index(2)
        dot_a_2=Dot(ax_2.coords_to_point(4,0), color=REANLEA_GREEN_AUQA).set_sheen(-0.4,DOWN).set_z_index(2)
        dots_A=VGroup(dot_a_1,dot_a_2)

        dot_b_1=Dot(ax_2.coords_to_point(0,1), color=REANLEA_BLUE_SKY).set_sheen(-0.4,DOWN).set_z_index(2)
        dot_b_2=Dot(ax_2.coords_to_point(0,2), color=REANLEA_BLUE_SKY).set_sheen(-0.4,DOWN).set_z_index(2)
        dot_b_3=Dot(ax_2.coords_to_point(0,3), color=REANLEA_BLUE_SKY).set_sheen(-0.4,DOWN).set_z_index(2)
        dots_B=VGroup(dot_b_1,dot_b_2,dot_b_3)


        dots_A_1=square_cloud(x_min=1,x_max=4,x_eps=1, y_max=0, col=REANLEA_GREEN_AUQA, rad=DEFAULT_DOT_RADIUS).shift(s_fact).set_z_index(2)
        dots_B_1=square_cloud(x_max=0,y_min=1,y_max=3, y_eps=1, col=REANLEA_BLUE_SKY,rad=DEFAULT_DOT_RADIUS).shift(s_fact).set_z_index(2)
        dots_C_1=square_cloud(x_min=2,x_max=4, x_eps=1, y_min=1,y_max=3, y_eps=1, rad=DEFAULT_DOT_RADIUS).shift(s_fact).set_z_index(2)

        dots_in_grp=VGroup(dots_A_1,dots_B_1,dots_C_1)

        def sq_cld(
            eps=1,
            **kwargs
        ):  
            n=.75*(1/eps)
            dots_A_1=square_cloud(x_min=2,x_max=4,x_eps=eps, y_max=0, col=REANLEA_GREEN_AUQA, rad=DEFAULT_DOT_RADIUS/n).shift(s_fact).set_z_index(2)
            dots_B_1=square_cloud(x_max=0,y_min=1,y_max=3, y_eps=eps, col=REANLEA_BLUE_SKY,rad=DEFAULT_DOT_RADIUS/n).shift(s_fact).set_z_index(2)
            dots_C_1=square_cloud(x_min=2,x_max=4, x_eps=eps, y_min=1,y_max=3, y_eps=eps, rad=DEFAULT_DOT_RADIUS/n).shift(s_fact).set_z_index(2)

            dots=VGroup(dots_A_1,dots_B_1,dots_C_1)

            return dots

        
        dots_2=sq_cld(eps=.5)
        dots_3=sq_cld(eps=.25)
        dots_4=sq_cld(eps=.125)
        dots_5=sq_cld(eps=.0625)

        x_grp=VGroup(ax_2,dots_5).save_state()

        line_x=Line(start=dots_A_1[0].get_center(), end=dots_A_1[-1].get_center()).set_stroke(width=3, color=REANLEA_GREEN_AUQA).set_z_index(4.5)
        line_y=Line(start=dots_B_1[0].get_center(), end=dots_B_1[-1].get_center()).set_stroke(width=3, color=REANLEA_BLUE_SKY).set_z_index(4.5)
        
        x_1=dots_A_1[0].get_center()[0]
        x_2=dots_A_1[-1].get_center()[0]

        y_1=dots_B_1[0].get_center()[1]
        y_2=dots_B_1[-1].get_center()[1]

        ind_sq=Polygon([x_1,y_1,0],[x_2,y_1,0],[x_2,y_2,0],[x_1,y_2,0]).set_opacity(0).set_fill(color=REANLEA_BLUE_LAVENDER, opacity=0.25)
        line=Line(start=[x_1,y_1,0], end=[x_2,y_2,0])

           
        z=MathTex(dots_A_1[0].get_center()).scale(.5)
        z0=MathTex(ax_2.c2p(0,0)).scale(.5).shift(.5*UP)
        za=MathTex(dots_A[0].get_center()).scale(.5).shift(UP)
        zb=MathTex(s_fact).scale(.5).shift(1.5*UP)


        dt_1=Dot(line_x.get_center(), color=REANLEA_MAGENTA, radius=.075).set_sheen(-.4,DOWN).set_z_index(5)
        dt_2=Dot(line_y.get_center(), color=REANLEA_YELLOW, radius=.075).set_sheen(-.4,DOWN).set_z_index(5)
        #dt_3=Dot([dt_1.get_center()[0],dt_2.get_center()[1],0], color=REANLEA_YELLOW, radius=.075).set_sheen(-.4,DOWN).set_z_index(5)
        
        dt_3=always_redraw(
            lambda : Dot([dt_1.get_center()[0],dt_2.get_center()[1],0], color=REANLEA_YELLOW, radius=.075).set_sheen(-.4,DOWN).set_z_index(5)
        )

        d_line_x=always_redraw(
            lambda : DashedLine(start=dt_1.get_center(), end=dt_3.get_center()).set_stroke(color=REANLEA_AQUA_GREEN, width=1).set_z_index(7)
        )

        d_line_y=always_redraw(
            lambda : DashedLine(start=dt_2.get_center(), end=dt_3.get_center()).set_stroke(color=REANLEA_MAGENTA_LIGHTER, width=1).set_z_index(7)
        )
        dt_3_ref=dt_3.copy()





        self.play(
            Create(ax_2)
        )
        self.play(
            Write(line_x),
            Write(line_y),
            Write(ind_sq)
        )
        self.wait()

        self.play(
            Write(dt_1),
            Write(dt_2),
            
        )
        self.play(
            Write(d_line_x),
            Write(d_line_y)
        )
        self.play(
            Write(dt_3_ref)
        )
        self.add(dt_3)
        self.play(
            FadeOut(dt_3_ref)
        )
        

        self.play(
            dt_1.animate.shift(RIGHT),
            dt_2.animate.shift(.5*UP)
        )


        self.wait(2)

        # manim -pqh test2.py Ex5

        # manim -pql test2.py Ex5

        # manim -sqk test2.py Ex5

        # manim -sql test2.py Ex5


class Ex6(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        ax_2=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                "include_ticks": False,
            }, 
        ).set_color(REANLEA_YELLOW_CREAM).scale(.5).shift(3*LEFT).set_z_index(-1)
        s_fact=ax_2.c2p(0,0)[0]*RIGHT+ax_2.c2p(0,0)[1]*UP

        dots_A_1=square_cloud(x_min=1,x_max=4,x_eps=1, y_max=0, col=REANLEA_GREEN_AUQA, rad=DEFAULT_DOT_RADIUS).shift(s_fact).set_z_index(2)
        dots_B_1=square_cloud(x_max=0,y_min=1,y_max=3, y_eps=1, col=REANLEA_BLUE_SKY,rad=DEFAULT_DOT_RADIUS).shift(s_fact).set_z_index(2)
        dots_C_1=square_cloud(x_min=1,x_max=4, x_eps=1, y_min=1,y_max=3, y_eps=1, rad=DEFAULT_DOT_RADIUS).shift(s_fact).set_z_index(2)

        dots_in_grp=VGroup(dots_A_1,dots_B_1,dots_C_1)


        def sq_cld(
            eps=1,
            **kwargs
        ):  
            n=.75*(1/eps)
            dots_A_1=square_cloud(x_min=-2,x_max=5,x_eps=eps, y_max=0, col=REANLEA_GREEN_AUQA, rad=DEFAULT_DOT_RADIUS/n).shift(s_fact).set_z_index(2)
            dots_B_1=square_cloud(x_max=0,y_min=-2,y_max=4, y_eps=eps, col=REANLEA_BLUE_SKY,rad=DEFAULT_DOT_RADIUS/n).shift(s_fact).set_z_index(2)
            dots_C_1=square_cloud(x_min=-2,x_max=5, x_eps=eps, y_min=-2,y_max=4, y_eps=eps, rad=DEFAULT_DOT_RADIUS/n).shift(s_fact).set_z_index(2)

            dots=VGroup(dots_A_1,dots_B_1,dots_C_1)

            return dots

        dots_5=sq_cld(eps=.125)

        line_x=Line(start=dots_A_1[0].get_center(), end=dots_A_1[-1].get_center()).set_stroke(width=4.5, color=REANLEA_GREEN_AUQA).set_z_index(5)
        line_y=Line(start=dots_B_1[0].get_center(), end=dots_B_1[-1].get_center()).set_stroke(width=4.5, color=REANLEA_BLUE_SKY).set_z_index(5)

        line_x_lbl=Tex("A").scale(.5).set_color(REANLEA_GREEN_AUQA).next_to(line_x,DOWN)
        line_y_lbl=Tex("B").scale(.5).set_color(REANLEA_BLUE_SKY).next_to(line_y,LEFT)

        x_1=dots_A_1[0].get_center()[0]
        x_2=dots_A_1[-1].get_center()[0]

        y_1=dots_B_1[0].get_center()[1]
        y_2=dots_B_1[-1].get_center()[1]

        ind_sq=Polygon([x_1,y_1,0],[x_2,y_1,0],[x_2,y_2,0],[x_1,y_2,0]).set_opacity(0).set_fill(color=REANLEA_BLUE_LAVENDER, opacity=0.35)
        ind_sq_1=Polygon([ax_2.get_x_axis().get_start()[0],ax_2.get_y_axis().get_start()[1],0],
                         [ax_2.get_x_axis().get_end()[0],ax_2.get_y_axis().get_start()[1],0],
                         [ax_2.get_x_axis().get_end()[0],ax_2.get_y_axis().get_end()[1],0],
                         [ax_2.get_x_axis().get_start()[0],ax_2.get_y_axis().get_end()[1],0],).set_opacity(0).set_fill(color=REANLEA_BLUE_LAVENDER, opacity=0.35)
        

        ind_sq_lbl=MathTex(r"A \times B").scale(.5).set_color(REANLEA_BLUE_LAVENDER).next_to(ind_sq[-1],.65*UR)

        dt_1=Dot(line_x.get_center(), color=REANLEA_MAGENTA, radius=.075).set_sheen(-.4,DOWN).set_z_index(5)
        dt_2=Dot(line_y.get_center(), color=REANLEA_YELLOW, radius=.075).set_sheen(-.4,DOWN).set_z_index(5)
        #dt_3=Dot([dt_1.get_center()[0],dt_2.get_center()[1],0], color=REANLEA_YELLOW, radius=.075).set_sheen(-.4,DOWN).set_z_index(5)

               
        dt_3=always_redraw(
            lambda : Dot([dt_1.get_center()[0],dt_2.get_center()[1],0], color=REANLEA_PINK_LIGHTER, radius=.075).set_sheen(-.4,DOWN).set_z_index(5)
        )
        dt_3_ref=dt_3.copy()

        d_line_x=always_redraw(
            lambda : DashedLine(start=dt_1.get_center(), end=dt_3.get_center()).set_stroke(color=REANLEA_AQUA_GREEN, width=1).set_z_index(7)
        )

        d_line_y=always_redraw(
            lambda : DashedLine(start=dt_2.get_center(), end=dt_3.get_center()).set_stroke(color=REANLEA_MAGENTA_LIGHTER, width=1).set_z_index(7)
        )

        dt_1_lbl=always_redraw(
            lambda : Tex("x").scale(.3).set_color(REANLEA_MAGENTA_LIGHTER).next_to(dt_1,.25*UR)
        )
        dt_2_lbl=always_redraw(
            lambda : Tex("y").scale(.3).set_color(REANLEA_YELLOW).next_to(dt_2,.25*UR)
        )
        dt_3_lbl=always_redraw(
            lambda : MathTex(r"(x,y)").scale(.3).set_color(REANLEA_PINK).next_to(dt_3,.25*UR).set_z_index(3)
        )



        ax_1_x_lbl_r=ax_2.get_x_axis_label(
            MathTex(r"\mathbb{R}").scale(0.65)
        ).next_to(ax_2.get_x_axis().get_end(), DOWN).set_color_by_gradient(REANLEA_SLATE_BLUE,REANLEA_GREEN_AUQA)

        ax_1_y_lbl_r=ax_2.get_x_axis_label(
            MathTex(r"\mathbb{R}").scale(0.65)
        ).next_to(ax_2.get_y_axis().get_end(), LEFT).set_color_by_gradient(REANLEA_SLATE_BLUE,REANLEA_BLUE_SKY)
        


        self.play(
            Create(ax_2)
        )
        '''self.play(
            Write(dots_5)
        )
        self.play(
            Write(line_x),
            Write(line_y),
            Write(ind_sq)
        )
        self.wait()

        self.play(
            Write(line_x_lbl),
            Write(line_y_lbl)
        )
        self.wait(.5)
        self.play(
            Write(ind_sq_lbl)
        )

        self.wait()

        self.play(
            Write(dt_1),
            Write(dt_2)
        )
        self.play(
            Write(d_line_x),
            Write(d_line_y)
        )
        self.play(
            Write(dt_3_ref)
        )
        self.add(dt_3)
        self.play(
            FadeOut(dt_3_ref)
        )
        self.wait()

        self.play(
            Write(dt_1_lbl),
            Write(dt_2_lbl),
            Write(dt_3_lbl)
        )
        

        self.play(
            dt_1.animate.shift(RIGHT),
            dt_2.animate.shift(.5*UP)
        )
        self.wait()
        self.play(
            dt_1.animate.shift(1.25*LEFT),
            dt_2.animate.shift(.75*DOWN)
        )
        self.wait()
        self.play(
            dt_1.animate.shift(1.5*RIGHT),
            dt_2.animate.shift(.5*UP)
        )
        self.play(
            dt_2.animate.shift(.5*UP)
        )
        self.play(
            dt_1.animate.shift(1.5*LEFT),
            
        )
        self.play(
            dt_1.animate.shift(0.5*RIGHT),
            dt_2.animate.shift(DOWN)
        )
        self.wait(4)

        self.play(
            Write(ax_1_x_lbl_r),
            Write(ax_1_y_lbl_r)
        )
        
        self.play(
            ReplacementTransform(ind_sq,ind_sq_1)
        )'''
        
        

        self.wait(2)

        

        # manim -pqh test2.py Ex6

        # manim -pql test2.py Ex6

        # manim -sqk test2.py Ex6

        # manim -sql test2.py Ex6




class Ex7(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        ax_2=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                "include_ticks": False,
            }, 
        ).set_color(REANLEA_YELLOW_CREAM).scale(.5).shift(3*LEFT).set_z_index(-1).scale(.75)

        line_x_lbl=Tex("A").scale(.5).set_color(REANLEA_GREEN_AUQA).next_to(ax_2.get_x_axis().get_center(), DOWN)
        line_y_lbl=Tex("B").scale(.5).set_color(REANLEA_BLUE_SKY).next_to(ax_2.get_y_axis().get_center(), LEFT)

        lbl=VGroup(line_x_lbl,line_y_lbl)
        
        line_x_lbl_1=MathTex(r"\mathbb{R}").scale(.5).set_color(REANLEA_GREEN_AUQA).next_to(ax_2.get_x_axis().get_end(), UP)
        line_y_lbl_1=MathTex(r"\mathbb{R}").scale(.5).set_color(REANLEA_BLUE_SKY).next_to(ax_2.get_y_axis().get_end(), RIGHT)
        
        lbl_1=VGroup(line_x_lbl_1,line_y_lbl_1)

        ind_sq=Polygon([ax_2.get_x_axis().get_start()[0]/2,ax_2.get_y_axis().get_start()[1]/2,0],
                         [ax_2.get_x_axis().get_end()[0]/2,ax_2.get_y_axis().get_start()[1]/2,0],
                         [ax_2.get_x_axis().get_end()[0]/2,ax_2.get_y_axis().get_end()[1]/2,0],
                         [ax_2.get_x_axis().get_start()[0]/2,ax_2.get_y_axis().get_end()[1]/2,0],).set_opacity(0).set_fill(color=REANLEA_BLUE_LAVENDER, opacity=0.35).set_z_index(-2)


        ind_sq_1=Polygon([ax_2.get_x_axis().get_start()[0],ax_2.get_y_axis().get_start()[1],0],
                         [ax_2.get_x_axis().get_end()[0],ax_2.get_y_axis().get_start()[1],0],
                         [ax_2.get_x_axis().get_end()[0],ax_2.get_y_axis().get_end()[1],0],
                         [ax_2.get_x_axis().get_start()[0],ax_2.get_y_axis().get_end()[1],0],).set_opacity(0).set_fill(color=REANLEA_BLUE_LAVENDER, opacity=0.35).set_z_index(-2)


        x_grp=VGroup(ax_2,lbl_1)

        eqn_5=MathTex(r"\mathbb{R}",r"\times",r"\mathbb{R}","=",r"\{", r"(x_{1},x_{2})",r"\mid", r"x , y \in \mathbb{R}", r"\}")
        eqn_5.next_to(x_grp,DOWN).scale(.6).shift(.5*UP).set_color_by_gradient(REANLEA_WARM_BLUE,REANLEA_AQUA)

        eqn_6=MathTex(r"\mathbb{R}",r"\times",r"\mathbb{R}",r"\times","...",r"\times",r"\mathbb{R}","=",r"\{", r"(x_{1},x_{2}, ... , x_{n})",r"\mid", r"x_{i} \in \mathbb{R}","","for","","i=1,2,...,n", r"\}")
        eqn_6.next_to(x_grp,DOWN).scale(.6).shift(.5*UP+.5*RIGHT).set_color_by_gradient(REANLEA_AQUA)
        eqn_6[12].scale(.5)
        eqn_6[4].scale(.5)
        eqn_6[9][7:10].scale(.5)
        eqn_6[13][6:9].scale(.5)

        br_6=Brace(eqn_6[0:7], buff=.05).set_color(REANLEA_TXT_COL_DARKER)

        br_6_lbl=MathTex("n").scale(.55).next_to(br_6,DOWN).shift(.15*UP).set_color(REANLEA_AQUA)


        ax_3 = ThreeDAxes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False,
            axis_config={
                "font_size": 24,
                "include_ticks": False,
            }
        ).shift(3*LEFT).set_color(REANLEA_YELLOW_CREAM).scale(.5).scale(.75)
        '''ax_3.shift(
            (ax_2.get_center()[0]-ax_3.get_center()[1])*RIGHT,
            (ax_2.get_origin()[1]-ax_3.get_origin()[0])*UP,
        )'''
        
        

        x_label = ax_3.get_x_axis_label(MathTex(r"\mathbb{R}"))
        y_label = ax_3.get_y_axis_label(MathTex(r"\mathbb{R}")).shift(UP * 1.8)

        a=MathTex(ax_2.get_center())
        b=MathTex(ax_3.get_center()).shift(UP)




        

        self.play(
            Create(ax_2)
        )
        self.play(
            Write(lbl)
        )
        self.play(
            ReplacementTransform(lbl,lbl_1)
        )

        ''' self.play(
            Create(ind_sq)
        )
        self.play(
            ReplacementTransform(ind_sq,ind_sq_1)
        )'''

        self.wait()

        '''self.play(
            x_grp.animate.scale(.75)
        )'''

        self.wait()

        '''self.play(
            Write(eqn_5)
        )
        self.wait()
        self.play(
            TransformMatchingShapes(eqn_5,eqn_6)
        )
        self.play(
            Write(br_6)
        )
        self.play(Write(br_6_lbl))
        self.wait()

        self.play(
            FadeOut(ind_sq_1)
        )'''

        self.play(
            Write(a),
            Write(b)
        )

        self.play(
            Write(ax_3)
        )
        
        

        self.wait(2)

        

        # manim -pqh test2.py Ex7

        # manim -pql test2.py Ex7

        # manim -sqk test2.py Ex7

        # manim -sql test2.py Ex7



class Axes3DEx(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            tips=False,
            axis_config={
                "font_size": 24,
                "include_ticks": False,
            }
        )

        x_label = axes.get_x_axis_label(MathTex(r"\mathbb{R}"))
        y_label = axes.get_y_axis_label(MathTex(r"\mathbb{R}")).shift(UP * 1.8)
        z_label = axes.get_z_axis_label(MathTex(r"\mathbb{R}")).shift(UP * 1.8)


        # zoom out so we see the axes
        self.set_camera_orientation(zoom=0.5)

        self.play(FadeIn(axes), FadeIn(x_label), FadeIn(y_label))

        self.wait(0.5)

        # animate the move of the camera to properly see the axes
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES, zoom=1, run_time=1.5)

        # built-in updater which begins camera rotation
        self.begin_ambient_camera_rotation(rate=0.15)

        

        self.wait(3)



        # manim -pqh test2.py Axes3DEx

        # manim -sqk test2.py Axes3DEx


class Ex8(ThreeDScene):
    def construct(self):

        ax_2=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                "include_ticks": False,
            }, 
        ).scale(.5).set_z_index(-1)


        ax_3 = ThreeDAxes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            tips=False,
            axis_config={
                "font_size": 24,
                "include_ticks": False,
            }
        ).set_color(RED).scale(.5)
        

        a=MathTex(ax_2.get_center()).scale(.5)
        b=MathTex(ax_3.get_center()).shift(UP).scale(.5).set_color(RED)



        self.play(
            Create(ax_2)
        )

        self.play(
            Write(ax_3)
        )

        self.play(
            Write(a),
            Write(b)
        )

        
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES, zoom=1, run_time=1.5)

        self.begin_ambient_camera_rotation(rate=0.15)

        self.wait(5)

        

        # manim -pqh test2.py Ex8

        # manim -pql test2.py Ex8

        # manim -sqk test2.py Ex8

        # manim -sql test2.py Ex8



class Rotation3DExample(ThreeDScene):
    def construct(self):
        cube = Cube(side_length=3, fill_opacity=1).set_color_by_gradient(REANLEA_BLUE_LAVENDER,REANLEA_MAGENTA,REANLEA_BLUE)

        a=MathTex(r"\mathbb{R}^{3}").shift(2*UR)

        self.begin_ambient_camera_rotation(rate=0.3)

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        self.play(Write(cube), run_time=2)
        self.play(
            Write(a)
        )

        self.wait(3)

        self.play(Unwrite(cube), run_time=2)



'''config.frame_height = 16
config.frame_width = 16
config.pixel_width = 1920
config.pixel_height = 1920'''


class Ex9(ThreeDScene):
    def construct(self):

        ax_3 = ThreeDAxes(
            tips=False,
            axis_config={
                "font_size": 24,
                "include_ticks": False,
                #"stroke_width":4,
            }
        ).set_stroke(width=4, color=REANLEA_TXT_COL)

        x = MathTex(r"\mathbb{R}").next_to(ax_3, RIGHT).set_color(REANLEA_BLUE_LAVENDER).set_sheen(-.4, DOWN)
        y = MathTex(r"\mathbb{R}").next_to(ax_3, UP).set_color(REANLEA_BLUE_LAVENDER).set_sheen(-.4, DOWN)
        z = MathTex(r"\mathbb{R}").next_to(ax_3, OUT).set_color(REANLEA_BLUE_LAVENDER).set_sheen(-.4, DOWN)

        

        cube = Cube(side_length=3, fill_opacity=.45).set_color_by_gradient(REANLEA_BLUE_LAVENDER).scale(1.25)


        d_1=Dot3D(point=UP+RIGHT+OUT, color=REANLEA_BLUE_DARKER, resolution=[32,32])
        

        d_line_x=always_redraw(
            lambda : DashedLine(start=ax_3.c2p(0,0,0), end=[d_1.get_center()[0],0,0]).set_stroke(color=REANLEA_GREEN_DARKEST, width=1).set_z_index(7)
        )
        d_line_y=always_redraw(
            lambda : DashedLine(start=[d_1.get_center()[0],0,0], end=[d_1.get_center()[0],d_1.get_center()[1],0]).set_stroke(color=REANLEA_BLUE_ALT, width=1).set_z_index(7)
        )
        d_line_z=always_redraw(
            lambda : DashedLine(start=[d_1.get_center()[0],d_1.get_center()[1],0], end=[d_1.get_center()[0],d_1.get_center()[1],d_1.get_center()[2]]).set_stroke(color=REANLEA_BLUE, width=1).set_z_index(7)
        )

        d_line_grp=VGroup(d_line_x,d_line_y,d_line_z)

        grp=VGroup(ax_3,cube,d_1,d_line_x,d_line_y,d_line_z)
        


        self.play(
            Write(ax_3)
        )
        self.add_fixed_orientation_mobjects(x,y,z)

        

        self.play(
            FadeIn(d_1)
        )

        self.move_camera(phi=75* DEGREES, theta=30* DEGREES, zoom=1, run_time=1.5)

        self.play(
            Write(d_line_grp)
        )

        d_1_lbl=MathTex("(x,0,0)").next_to([d_1.get_center()[0],0,0], DOWN).scale(.3).shift(.1*LEFT).set_color(REANLEA_GREEN)
        #d_1_lbl.rotate(PI/2, about_point=[d_1.get_center()[0],0,0], axis=RIGHT)

        d_2_lbl=MathTex("(x,y,0)").next_to([d_1.get_center()[0],d_1.get_center()[1],0], UP).set_color(REANLEA_WARM_BLUE).scale(.3)
        #d_2_lbl.rotate(PI/2, about_point=[0,d_1.get_center()[1],0], axis=RIGHT)

        d_3_lbl=MathTex("(x,y,z)").next_to([d_1.get_center()[0],d_1.get_center()[1],d_1.get_center()[2]], OUT).scale(.3).set_color(REANLEA_BLUE_SKY)
        #d_3_lbl.rotate(PI/2, about_point=d_1.get_center(), axis=RIGHT).rotate(-PI, axis=OUT)

        d_lbl_grp=VGroup(d_1_lbl,d_2_lbl,d_3_lbl)

        self.add_fixed_orientation_mobjects(d_1_lbl,d_2_lbl,d_3_lbl)    
        

        self.begin_ambient_camera_rotation(rate=0.35)


        self.play(
            Write(cube),
            d_1_lbl.animate.set_stroke(color=REANLEA_GREEN_DARKEST),
            d_2_lbl.animate.set_color(REANLEA_BLUE_DARKER),
            d_3_lbl.animate.set_color(REANLEA_WARM_BLUE_DARKER),
            run_time=2
        )

        

        self.wait(10)

        

        

        

        # manim -pqh test2.py Ex9

        # manim -pql test2.py Ex9

        # manim -sqk test2.py Ex9

        # manim -sql test2.py Ex9



class Ex10(Scene):
    def construct(self):

        ax_2=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                "include_ticks": False,
            }, 
        ).set_color(REANLEA_YELLOW_CREAM).scale(.5).shift(3*LEFT).set_z_index(-1).scale(.75)


        dt_1=Dot(ax_2.c2p(2,0), color=REANLEA_MAGENTA, radius=.075).set_sheen(-.4,DOWN).set_z_index(6).save_state()
        dt_2=Dot(ax_2.c2p(0,2), color=REANLEA_YELLOW, radius=.075).set_sheen(-.4,DOWN).set_z_index(6).save_state()
        
               
        dt_3=always_redraw(
            lambda : Dot([dt_1.get_center()[0],dt_2.get_center()[1],0], color=REANLEA_PINK_LIGHTER, radius=.075).set_sheen(-.4,DOWN).set_z_index(6)
        )
        
        dt_1_lbl=always_redraw(
            lambda : MathTex("x").scale(.3).set_color(REANLEA_MAGENTA_LIGHTER).next_to(dt_1,.25*UR)
        )
        dt_2_lbl=always_redraw(
            lambda : MathTex("y").scale(.3).set_color(REANLEA_YELLOW).next_to(dt_2,.25*UR)
        )
        dt_3_lbl=always_redraw(
            lambda : MathTex(r"(x,y)").scale(.3).set_color(REANLEA_PINK).next_to(dt_3,.25*UR).set_z_index(3)
        )

        dt_lbl_grp=VGroup(dt_1_lbl,dt_2_lbl,dt_3_lbl)

        dt_1_lbl_0=MathTex("(x,0)").scale(.3).set_color(REANLEA_MAGENTA_LIGHTER).next_to(dt_1,.25*UR)
        dt_2_lbl_0=MathTex("(0,y)").scale(.3).set_color(REANLEA_YELLOW).next_to(dt_2,.25*UR)
        dt_lbl_grp_0=VGroup(dt_1_lbl_0,dt_2_lbl_0,dt_3_lbl)

        self.add(ax_2)
        self.play(
            Write(dt_1),
            Write(dt_2),
            Write(dt_3)
        )
        self.play(
            Write(dt_lbl_grp)
        )
        self.play(
            dt_1.animate.shift(RIGHT),
            dt_2.animate.shift(.5*UP)
        )
        self.wait()
        self.play(
            dt_1.animate.shift(1.25*LEFT),
            dt_2.animate.shift(.75*DOWN)
        )
        self.wait()
        self.play(
            dt_1.animate.shift(1.5*RIGHT),
            dt_2.animate.shift(.5*UP)
        )
        self.play(
            dt_2.animate.shift(.5*UP)
        )
        self.play(
            dt_1.animate.shift(1.5*LEFT),
            
        )
        self.play(
            dt_1.animate.shift(0.5*RIGHT),
            dt_2.animate.shift(DOWN)
        )
        self.wait()
        self.play(
            Restore(dt_1),
            Restore(dt_2)
        )
        self.play(
            ReplacementTransform(dt_lbl_grp,dt_lbl_grp_0)
        )


        self.wait(2)
        

        
        

        # manim -pqh test2.py Ex10

        # manim -pql test2.py Ex10

        # manim -sqk test2.py Ex10

        # manim -sql test2.py Ex10


class Ex11(Scene):
    def construct(self):

        ax_2=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                "include_ticks": False,
            }, 
        ).set_color(REANLEA_YELLOW_CREAM).scale(.5).shift(3*LEFT).set_z_index(-1)


        dt_1=Dot(ax_2.c2p(2,0), color=REANLEA_MAGENTA, radius=.075).set_sheen(-.4,DOWN).set_z_index(6).save_state()
        dt_2=Dot(ax_2.c2p(0,2), color=REANLEA_VIOLET, radius=.075).set_sheen(-.4,DOWN).set_z_index(6).save_state()
        
               
        dt_3=always_redraw(
            lambda : Dot([dt_1.get_center()[0],dt_2.get_center()[1],0], color=REANLEA_PINK_LIGHTER, radius=.075).set_sheen(-.4,DOWN).set_z_index(6)
        )

        dt_grp=VGroup(dt_1,dt_2,dt_3)
        
        dt_1_lbl=always_redraw(
            lambda : MathTex("x").scale(.3).set_color(REANLEA_MAGENTA_LIGHTER).next_to(dt_1,.25*UR)
        )
        dt_2_lbl=always_redraw(
            lambda : MathTex("y").scale(.3).set_color(REANLEA_YELLOW).next_to(dt_2,.25*UR)
        )
        dt_3_lbl=always_redraw(
            lambda : MathTex(r"(x,y)").scale(.3).set_color(REANLEA_PINK).next_to(dt_3,.25*UR).set_z_index(3)
        )

        dt_lbl_grp=VGroup(dt_1_lbl,dt_2_lbl,dt_3_lbl)

        dt_1_lbl_0=MathTex("(x,0)").scale(.3).set_color(REANLEA_MAGENTA_LIGHTER).next_to(dt_1,.25*UR)
        dt_2_lbl_0=MathTex("(0,y)").scale(.3).set_color(REANLEA_YELLOW).next_to(dt_2,.25*UR)
        dt_lbl_grp_0=VGroup(dt_1_lbl_0,dt_2_lbl_0,dt_3_lbl)




        self.add(ax_2)
        self.play(
            Write(dt_grp),
        )
        self.play(
            Write(dt_lbl_grp)
        )
        self.play(
            dt_1.animate.shift(RIGHT),
            dt_2.animate.shift(.5*UP)
        )
        self.wait()
        self.play(
            dt_1.animate.shift(1.25*LEFT),
            dt_2.animate.shift(.75*DOWN)
        )
        self.wait()
        self.play(
            dt_1.animate.shift(1.5*RIGHT),
            dt_2.animate.shift(.5*UP)
        )
        self.play(
            dt_2.animate.shift(.5*UP)
        )
        self.play(
            dt_1.animate.shift(1.5*LEFT),
            
        )
        self.play(
            dt_1.animate.shift(0.5*RIGHT),
            dt_2.animate.shift(DOWN)
        )
        self.wait()
        self.play(
            Restore(dt_1),
            Restore(dt_2)
        )
        self.play(
            ReplacementTransform(dt_lbl_grp,dt_lbl_grp_0)
        )

        grp_1=VGroup(ax_2,dt_grp,dt_lbl_grp_0)
        self.play(
            grp_1.animate.scale(.75).shift(.5*UP)
        )

        glow_dt_1=get_glowing_surround_circle(dt_1, color=REANLEA_GOLD, buff_max=.075, n=20)
        glow_dt_2=get_glowing_surround_circle(dt_2, color=REANLEA_GOLD)
        glow_dt_3=get_glowing_surround_circle(dt_3, color=PURE_GREEN)

        self.play(
            Create(glow_dt_1),
            Create(glow_dt_2)
        )

        arr_1=Arrow(start=ax_2.c2p(0,0), end=dt_1.get_center(), buff=0, tip_length=.125, stroke_width=4, color=REANLEA_SLATE_BLUE_LIGHTER).set_z_index(7)

        self.play(
            Create(glow_dt_1),
            Create(glow_dt_2)
        )
        self.play(
            Create(arr_1)
        )



        self.wait(4)
        
        

        
        

        # manim -pqh test2.py Ex11

        # manim -pql test2.py Ex11

        # manim -sqk test2.py Ex11

        # manim -sql test2.py Ex11


class Ex12(ThreeDScene):
    def construct(self):

        ax_3 = ThreeDAxes(
            tips=False,
            axis_config={
                "font_size": 24,
                "include_ticks": False,
                #"stroke_width":4,
            }
        ).set_stroke(width=4, color=REANLEA_TXT_COL)

        cube = Cube(side_length=3, fill_opacity=.45).set_color_by_gradient(REANLEA_BLUE_LAVENDER)


        d_1=Dot3D(point=UP+RIGHT+OUT, color=REANLEA_BLUE_DARKER, resolution=[32,32])
        

        d_line_x=always_redraw(
            lambda : DashedLine(start=ax_3.c2p(0,0,0), end=[d_1.get_center()[0],0,0]).set_stroke(color=REANLEA_GREEN_DARKEST, width=1).set_z_index(7)
        )

        d_line_x_lbl=MathTex("(x,0,0)").next_to([d_1.get_center()[0],0,0], UP).scale(.4).shift(.35*DOWN).set_color(REANLEA_GREEN).flip()
        d_line_x_lbl.rotate(PI/2, about_point=[d_1.get_center()[0],0,0], axis=RIGHT)#.rotate(-PI/2)

        self.add(ax_3)

        self.move_camera(phi=75* DEGREES, theta=30* DEGREES, zoom=1, run_time=1.5)

        self.play(
            #FadeIn(d_1),
            Write(d_line_x)
        )


        self.play(
            Write(d_line_x_lbl)
        )
        self.add_fixed_orientation_mobjects(d_line_x_lbl)
        self.wait()

        self.begin_ambient_camera_rotation(rate=0.35)

        self.wait(5)



        # manim -pqh test2.py Ex12

        # manim -pql test2.py Ex12

        # manim -sqk test2.py Ex12

        # manim -sql test2.py Ex12
        



class Test3DEx(ThreeDScene):
    def construct(self):
        ax = ThreeDAxes(
            tips=False,
            axis_config={
                "font_size": 24,
                "include_ticks": False,
                #"stroke_width":4,
            }
        ).set_stroke(width=4, color=REANLEA_TXT_COL)

        
        x = MathTex(r"\mathbb{R}").next_to(ax, RIGHT)
        y = MathTex(r"\mathbb{R}").next_to(ax, UP)
        z = MathTex(r"\mathbb{R}").next_to(ax, OUT)
        #z.rotate(angle=PI/2, axis=UP).rotate(PI/2, axis=RIGHT)
        

        d_1=Dot3D(point=UP+RIGHT+OUT, color=REANLEA_BLUE_DARKER, resolution=[32,32])

        d_line_x=always_redraw(
            lambda : DashedLine(start=ax.c2p(0,0,0), end=[d_1.get_center()[0],0,0]).set_stroke(color=REANLEA_GREEN_DARKEST, width=1).set_z_index(7)
        )

        d_line_x_lbl=MathTex("(x,0,0)").next_to([d_1.get_center()[0],0,0], DOWN).scale(.25).set_color(REANLEA_GREEN).shift(.15*LEFT)
        #d_line_x_lbl.rotate(PI/2, about_point=[d_1.get_center()[0],0,0], axis=RIGHT)
        
        self.play(Create(ax))
        
        self.add_fixed_orientation_mobjects(x, y, z)
        
        self.move_camera(phi=75* DEGREES, theta=30* DEGREES, zoom=1, run_time=1.5)
        self.play(
            Write(d_line_x)
        )
        self.add_fixed_orientation_mobjects(d_line_x_lbl)
        

        self.wait()

        self.begin_ambient_camera_rotation(rate=0.35)

        self.wait(5)


        # manim -pqh test2.py Test3DEx

        # manim -pql test2.py Test3DEx

        # manim -sqk test2.py Test3DEx



class Ex13(Scene):
    def construct(self):

        sep_ln=Line().scale(2).rotate(PI/2).set_stroke(width=5, color=[REANLEA_MAGENTA,REANLEA_WARM_BLUE]).shift(1.5*RIGHT)
        self.play(
            Write(sep_ln)
        )
        
        eqn_6=MathTex(r"\mathbb{R}",r"\times",r"\mathbb{R}",r"\times","...",r"\times",r"\mathbb{R}","=",r"\{", r"(x_{1},x_{2}, ... , x_{n})",r"\mid", r"x_{i} \in \mathbb{R}","","for","","i=1,2,...,n", r"\}")
        eqn_6[12].scale(.5)
        eqn_6[4].scale(.5)
        eqn_6[9][7:10].scale(.5)
        eqn_6[13][6:9].scale(.5)

        eqn_6.scale(.6).shift(1.75*UP+2.85*LEFT).set_color_by_gradient(REANLEA_AQUA)

        self.play(
            Write(eqn_6)
        )

        self.wait()

        eqn_7=MathTex(r"\mathbb{R}^{n}").next_to(eqn_6[7],LEFT).shift(.2*RIGHT)
        eqn_7.scale(.6).set_color_by_gradient(REANLEA_AQUA)

        self.play(
            TransformMatchingShapes(eqn_6[0:7],eqn_7)
        )
        
        
        self.wait()

        eqn_8=VGroup(eqn_7,eqn_6[7:])
        self.play(
            eqn_8.animate.shift(LEFT)
        )

        bend_bezier_arrow_indi=bend_bezier_arrow_indicate().scale(.55).set_color(REANLEA_TXT_COL)
        bend_bezier_arrow_indi.flip(RIGHT).rotate(90*DEGREES).next_to(eqn_8[0],DOWN).shift(.5*RIGHT)

        bend_bez_arrow=bend_bezier_arrow().rotate(-30*DEGREES).scale(0.55).set_color(REANLEA_TXT_COL).next_to(eqn_8[0],DOWN).shift(.4*RIGHT)

        self.play(
            Create(bend_bez_arrow)
        )

        with RegisterFont("Cedarville Cursive") as fonts:
            txt_2=Text("Vector Space", font=fonts[0]).scale(.5)#.to_edge(UP).shift(.5*DOWN)             
            # to_edge(UP) == move_to(3.35*UP)
            txt_2.set_color_by_gradient(REANLEA_BLUE_LAVENDER,REANLEA_SLATE_BLUE_LIGHTER)
            txt_2.next_to(bend_bez_arrow,RIGHT).shift(.1*LEFT+.05*DOWN).rotate(10*DEGREES)

        self.play(
            Create(txt_2)
        )

        with RegisterFont("Cousine") as fonts:

            vs_dfn_tx_1 = Text("vector addition : ", font=fonts[0]).scale(0.35).set_color(REANLEA_SLATE_BLUE_LIGHTEST)

            vs_dfn_tx_2 = Text("& scalar multiplication : ", font=fonts[0]).scale(0.35).shift(.55*RIGHT).set_color(REANLEA_SLATE_BLUE_LIGHTER)
        
        with RegisterFont("Caveat") as fonts:

            vs_dfn_tx_3 = Text("where,", font=fonts[0]).scale(0.5).shift(.1*LEFT).set_color(REANLEA_TXT_COL)

        vs_dfn_mtx_1=MathTex(r"(x_{1},x_{2}, ... , x_{n})","+",r"(y_{1},y_{2}, ... , y_{n})",r"=",r"(x_{1}+y_{1},x_{2}+y_{2}, ... , x_{n}+y_{n})").scale(0.45).next_to(vs_dfn_tx_1,DOWN).shift(3*RIGHT)
        vs_dfn_mtx_2=MathTex(r"\lambda",r"\cdot",r"(x_{1},x_{2}, ... , x_{n})",r"=",r"(\lambda \cdot x_{1},\lambda \cdot x_{2}, ... , \lambda \cdot x_{n})").scale(0.45).next_to(vs_dfn_tx_1,DOWN).shift(2.1*RIGHT)
        vs_dfn_mtx_3_1=MathTex(r"(x_{1},x_{2}, ... , x_{n})",r"\in \mathbb{R}^{n}").scale(0.45).next_to(vs_dfn_tx_3,DOWN).shift(RIGHT)
        vs_dfn_mtx_3_2=MathTex(r"\lambda",r"\in \mathbb{R}").scale(0.45).next_to(vs_dfn_tx_3,DOWN).shift(.35*DOWN+.35*RIGHT)
        vs_dfn_mtx_3=VGroup(vs_dfn_mtx_3_1,vs_dfn_mtx_3_2).shift(.15*UP+.35*RIGHT)

        vs_dfn_1=VGroup(vs_dfn_tx_1,vs_dfn_mtx_1).shift(5*LEFT)
        vs_dfn_2=VGroup(vs_dfn_tx_2,vs_dfn_mtx_2).shift(5*LEFT+DOWN)
        vs_dfn_3=VGroup(vs_dfn_tx_3,vs_dfn_mtx_3).shift(2*DOWN+3*LEFT)
        vs_dfn=VGroup(vs_dfn_1,vs_dfn_2,vs_dfn_3)

        self.play(
            Write(vs_dfn)
        )

        
        with RegisterFont("Caveat") as fonts:
            con_tex_1 = Text("what about the distance in the space", font=fonts[0]).scale(0.85).shift(.1*LEFT).set_color(REANLEA_TXT_COL)
        con_tex_2=MathTex(r"\mathbb{R}^{n}").next_to(con_tex_1,RIGHT).scale(.85).set_color(REANLEA_TXT_COL)
        con_tex=VGroup(con_tex_1,con_tex_2)

        self.play(
            FadeOut(vs_dfn)
        )
        self.play(
            Write(con_tex)
        )
        
        self.wait(2)





        # manim -pqh test2.py Ex13

        # manim -pql test2.py Ex13

        # manim -sqk test2.py Ex13

        # manim -sql test2.py Ex13



class ColoringVectorField(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_inde(-100)
        self.add(water_mark)

        func = lambda x: x - ORIGIN
        colors = [PURE_RED,PURE_GREEN,YELLOW_E]
        
        vf = ArrowVectorField(
            func, min_color_scheme_value=2, max_color_scheme_value=10, colors=colors
        )

        
        self.wait()
        self.play(
            Write(vf, run_time=1)
        )
        
        self.wait(2)


        # manim -pqh test2.py ColoringVectorField

        # manim -sqk test2.py ColoringVectorField


class CoordSysExample(Scene):
            def construct(self):

                # WATER MARK 

                water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
                self.add(water_mark)

                graphs = VGroup()
                for n in np.arange(1,15):                    
                    graphs += ImplicitFunction(
                        lambda x,y : np.abs(x)**n + np.abs(y)**n -1 ,
                    ).scale(3).set_stroke(width=1.25)
                
                graphs.set_color_by_gradient(REANLEA_BLUE,REANLEA_BLUE_SKY)

                self.wait()
                self.play(
                    Create(graphs, run_time=20)
                )
                self.wait(2)


                # manim -pqh test2.py CoordSysExample

                # manim -sqk test2.py CoordSysExample


class ColoringVectorFieldWithDot(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
       
        func = lambda x: x - ORIGIN
        colors = [REANLEA_BLUE_LAVENDER,REANLEA_SLATE_BLUE,REANLEA_AQUA,REANLEA_GREY]
        
        vf = ArrowVectorField(
            func, min_color_scheme_value=2, 
            max_color_scheme_value=10, 
            colors=colors
        )
       
        dots=VGroup()          
        for obj in vf:
            dots += Dot().move_to(obj.get_end()).set_color(obj.get_color()).scale(.75).set_sheen(-.4,DOWN)
        
        
        self.wait()

        self.play(
            Write(dots)
        )
        self.wait(2)

        self.play(
            Write(vf, run_time=2)
        )
        self.wait(2)

        self.play(
            FadeOut(dots)
        )
        self.wait(2)

        for i in range(280,285):
            self.play(Wiggle(vf[i], scale_value=2, rotation_angle=0.4))
        
        self.wait()

        self.play(
            Write(dots)
        )
        self.wait(2)

        self.play(
            Unwrite(vf)
        )

        self.wait(2)
        self.play(
            Unwrite(dots)
        )
        self.wait(2)
        

        # manim -pql test2.py ColoringVectorFieldWithDot

        # manim -pqh test2.py ColoringVectorFieldWithDot

        # manim -sqk test2.py ColoringVectorFieldWithDot


class coordsysobjectsEx(Scene):
    def construct(self):
        s= VGroup(*[Dot().shift(i*0.2*RIGHT*np.random.uniform(-1,1)+UP*np.random.uniform(-1,1)) for i in range(-15,15)])
        s.shift(UP).set_color_by_gradient(REANLEA_BLUE_LAVENDER, REANLEA_BLUE, PURE_GREEN)
        s2= s.copy().set_color_by_gradient(REANLEA_ORANGE_DARKER,REANLEA_VIOLET,REANLEA_PURPLE)
        s2.arrange_submobjects()
        s2.shift(DOWN)
        s3= VGroup(*[Dot().shift(i*0.1*RIGHT*np.random.uniform(-6,6)) for i in range(-15,15)])
        s3.shift(2*DOWN).set_color_by_gradient(REANLEA_BLUE, PURE_GREEN, REANLEA_GREY_DARKER,REANLEA_VIOLET,REANLEA_AQUA_GREEN)
        #self.add(s,s2)
        g=VGroup(s,s2, s3)
        self.play(Create(g, run_time=4))
        self.wait()
                


        # manim -pqh test2.py coordsysobjectsEx

class randompointsEx(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(2*RIGHT).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)


        s=VGroup(
            *[
                Dot().shift(i*0.2*RIGHT*np.random.uniform(-1,1)+i*0.2*UP*np.random.uniform(-1,1))
                for i in range(-15,15)
            ]
        )
        s.set_color_by_gradient(REANLEA_BLUE_DARKER,REANLEA_PURPLE,PURE_GREEN)

        dt=Dot().set_color(REANLEA_AQUA).shift(2*LEFT+UP)


        self.wait()
        self.play(
            Write(dt)
        )
        self.play(
            Write(s)
        )
        self.wait(2)

        ln=VGroup()
        for obj in s:
            ln += Line(start=dt.get_center(), end=obj.get_center()).set_stroke(width=2, color=[dt.get_color(), obj.get_color()])

        self.play(
            Write(ln)
        )
        self.wait(2)

        # manim -pqh test2.py randompointsEx

        # manim -sqk test2.py randompointsEx


class RedrawEx(Scene):
    def construct(self):
        dt_1=Dot().set_color(REANLEA_AQUA).shift(2*LEFT+UP)
        dt_2=Dot().set_color(REANLEA_PURPLE).shift(2*RIGHT+DOWN)
        ln_1=Line(start=dt_1.get_center(), end=dt_2.get_center()).set_stroke(width=2, color=[dt_2.get_color(),dt_1.get_color()])
        self.play(
            Write(dt_1)
        )
        self.play(
            Write(dt_2)
        )
        self.wait()
        self.play(
            Write(ln_1)
        )
        self.wait(3)

        dt_3=dt_1.copy().set_sheen(-.4,DOWN)
        self.play(
            Write(dt_3)
        )

        dt_3_bl=MathTex("x").scale(.5).set_color(REANLEA_YELLOW).next_to(dt_3,UP)
        self.play(
            Write(dt_3_bl)
        )
        dt_3_bl.add_updater(
            lambda z : z.become(
                MathTex("x").scale(.5).set_color(REANLEA_YELLOW).next_to(dt_3,UP)
            )
        )
        

        self.play(
            dt_3.animate.move_to(
                .5*dt_1.get_center()+.5*dt_2.get_center()
            )
        )
        self.play(
            dt_3.animate.move_to(
                .75*dt_1.get_center()+.25*dt_2.get_center()
            )
        )
        self.play(
            dt_3.animate.move_to(
                .25*dt_1.get_center()+.75*dt_2.get_center()
            )
        )
        self.wait(2)

        

        
        self.wait(2)


        # manim -pqh test2.py RedrawEx

class LatexArrowTest_1(Scene):
    def construct(self):

        eqn_10=MathTex(r"v' - \zeta v","=","0").set_color(REANLEA_GOLD).shift(2.75*DOWN)
        eqn_10[0][0].set_stroke(width=1.025)
        eqn_10[0][-1].set_stroke(width=1.025)

        self.play(
            Write(eqn_10)
        )
        self.wait()

        sr_eqn_10=SurroundingRectangle(eqn_10, color=REANLEA_WELDON_BLUE, buff=.2, corner_radius=.05)

        self.play(
            Create(sr_eqn_10)
        )


        arr=MathTex(r"\longrightarrow").rotate(-20*DEGREES).next_to(sr_eqn_10,RIGHT).set_stroke(width=2, color=[REANLEA_BLUE,REANLEA_BLUE_SKY]).shift(.35*LEFT+.25*DOWN)
        self.play(
            Write(arr)
        )

        

        with RegisterFont("Homemade Apple") as fonts:
            txt_1=Text("Linearly Dependent", font=fonts[0]).scale(.5)
            txt_1.set_color_by_gradient(REANLEA_BLUE_LAVENDER,REANLEA_TXT_COL_LIGHTER)
            txt_1.next_to(arr,RIGHT).shift(.1*LEFT+.3*DOWN)
        
        self.play(
            Write(txt_1)
        )

        self.wait(2)

        # manim -pqh test2.py LatexArrowTest_1

        # manim -sqk test2.py LatexArrowTest_1

        
class LatexArrowEx(Scene):
    def construct(self):
        
        arr_1=MathTex(r"\longrightarrow").rotate(-20*DEGREES).set_stroke(width=2, color=[REANLEA_BLUE,REANLEA_BLUE_SKY])
        self.play(
            Write(arr_1)
        )

        arr_2=MathTex(r"\rightsquigarrow").rotate(-20*DEGREES).shift(2*DOWN).set_stroke(width=3.5,color=[REANLEA_BLUE,REANLEA_BLUE_SKY])
        arr_2.scale(2)
        self.play(
            Write(arr_2)
        )

        self.wait(2)

        # manim -pqh test2.py LatexArrowEx

        # manim -sqk test2.py LatexArrowEx


class text_ex(Scene):
    def construct(self):
        
        with RegisterFont("Nanum Pen Script") as fonts:
            vsp_ruls = VGroup(*[Text(x, font=fonts[0]) for x in (
                "I. Scalar Multiplication",
                "II. Vector Addition",
            )]).scale(0.85).arrange_submobjects(DOWN).shift(4.5*LEFT)
            vsp_ruls[1].shift(.5*LEFT)

        self.play(
            Write(vsp_ruls)
        )
        self.wait(2)

        # manim -pqh test2.py text_ex



class emoji2(Scene):
    def construct(self):
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

        response = requests.get("https://cdn-0.emojis.wiki/emoji-pics/apple/person-surfing-apple.png", headers=headers)
        image_bytes = io.BytesIO(response.content)
        image = Image.open(image_bytes)
        
        emoji = ImageMobject(image)
        self.add(emoji.move_to(4*RIGHT+2*UP))
        self.play(emoji.animate.move_to(4*LEFT+2*DOWN))


        # manim -pqh test2.py emoji2


class emoji3(Scene):
    def construct(self):

        emoji = ImageMobject("🚀")
        self.add(emoji.move_to(4*RIGHT+2*UP))
        self.play(emoji.animate.move_to(4*LEFT+2*DOWN))


        # manim -pqh test2.py emoji3


class EmojiImageMobject(ImageMobject):
    def __init__(self, emoji, **kwargs):
        emoji_code = "-".join(f"{ord(c):x}" for c in emoji)
        emoji_code = emoji_code.upper()  # <-  needed for openmojis
        url = f"https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/618x618/{emoji_code}.png"
        im = Image.open(requests.get(url, stream=True).raw)
        emoji_img = np.array(im.convert("RGBA"))
        ImageMobject.__init__(self, emoji_img, **kwargs)


class emoji4(Scene):
    def construct(self):
        
        em = EmojiImageMobject("🚀").scale(.15).move_to(4*LEFT+2*DOWN)
        self.add(em)
        self.wait()
        self.play(
            em.animate.move_to(4*RIGHT+2*UP)
        )
        self.wait()
        self.play(
            em.animate.rotate(PI/2)
        )
        self.wait(2)


        # manim -pqh test2.py emoji4


class emoji5(Scene):
    def construct(self):
        
        em = EmojiImageMobject("🚀").scale(.15)
        self.play(
            FadeIn(em)
        )
        
        self.wait(2)


        # manim -pqh test2.py emoji5

        # manim -sqk test2.py emoji5

class arr_ex_1(Scene):
    def construct(self):
        arr_1=Arrow(start=LEFT,end=RIGHT)
        arr_2=Arrow(start=LEFT, end=3*RIGHT)

        self.wait()
        self.play(
            Create(arr_1, run_time=5)
        )
        self.wait()
        self.play(
            ReplacementTransform(arr_1,arr_2)
        )
        self.wait(2)

        # manim -pqh test2.py arr_ex_1



class case_tex_ex(Scene):
    def construct(self):
        
        case_tex_1= MathTex(r"\lambda v' + \gamma v = 0 \longrightarrow \begin{cases}"
                r" \text{for some} (\lambda , \gamma) \neq (0,0) & \leadsto \text{linearly dependent} \\"
                r"\text{only for} (\lambda , \gamma) = (0,0) & \leadsto \text{linearly independent}"
                r"\end{cases}"
        ).scale(.5)

        self.wait()
        self.play(
            Write(case_tex_1)
        )

        self.wait(2)

        # manim -pqh test2.py case_tex_ex

        # manim -sqk test2.py case_tex_ex



class des_tree_ex(Scene):
    def construct(self):
        
        ln_h_0=Line(start=LEFT, end=.25*LEFT).set_stroke(width=4, color=[REANLEA_AQUA_GREEN,REANLEA_GREY])
        ln_v_0=Line().rotate(PI/2).shift(.25*LEFT).set_stroke(width=4, color=[REANLEA_SLATE_BLUE,REANLEA_AQUA_GREEN,REANLEA_PURPLE])
        ln_h_1_0=Line(start=.27*LEFT,end=.75*RIGHT).shift(UP).set_stroke(width=4, color=[REANLEA_BLUE_SKY,REANLEA_SLATE_BLUE])
        ln_h_1_1=Line(start=.27*LEFT,end=.75*RIGHT).shift(DOWN).set_stroke(width=4, color=[REANLEA_BLUE_SKY,REANLEA_PURPLE])

        ln_grp=VGroup(ln_h_0,ln_v_0,ln_h_1_0,ln_h_1_1)


        self.play(
            Write(ln_grp)
        )


        # manim -pqh test2.py des_tree_ex

        # manim -sqk test2.py des_tree_ex


class des_tree_ex_1(Scene):
    def construct(self):
        
        ln=create_des_tree()


        self.play(
            Write(ln)
        )


        # manim -pqh test2.py des_tree_ex_1

        # manim -sqk test2.py des_tree_ex_1


class box_opa(Scene):
    def construct(self):

        with RegisterFont("Nanum Pen Script") as fonts:
            lin_dep_indep = VGroup(*[Text(x, font=fonts[0]) for x in (
                "for some",
                "only for",
            )]).scale(0.35).arrange_submobjects(DOWN).shift(4.5*LEFT+2.5*UP)
            lin_dep_indep[0].shift(.75*UP)
            lin_dep_indep[1].shift(.75*DOWN)

        with RegisterFont("Homemade Apple") as fonts:
            txt_3=Text("Linearly Dependent", font=fonts[0]).scale(.3)
            txt_3.set_color_by_gradient(REANLEA_TXT_COL)
            txt_3.shift(.75*RIGHT+.2*DOWN).next_to(lin_dep_indep[0],DOWN)

        self.play(
            Write(lin_dep_indep)
        )

        self.play(
            Write(txt_3)
        )

        grp=VGroup(txt_3,lin_dep_indep[0])

        sr_txt_3=SurroundingRectangle(grp, buff=.25, corner_radius=.05).set_stroke(width=0).set_fill(color=REANLEA_MAGENTA, opacity=.25).set_z_index(-1)

        self.play(
            Create(sr_txt_3)
        )

        # manim -pqh test2.py  box_opa

        # manim -sqk test2.py  box_opa


class txt_test_1(Scene):
    def construct(self):
        
        with RegisterFont("Reenie Beanie") as fonts:
            txt_3=Text("Basis", font=fonts[0]).scale(.6)

        self.play(
            Write(txt_3)
        )

        eqn_r2_1 = MathTex(
            r"\mathbb{R}^2&",r"= \{ (a,b)= a(1,0)+b(0,1) \mid a , b \in \mathbb{R} \} \\ &",r"= span \{ (1,0) , (0,1) \}",
        ).shift(3*DOWN).scale(.6).set_color(REANLEA_AQUA)

        self.play(
            AnimationGroup(
                *[Write(eq) for eq in eqn_r2_1],
                lag_ratio=2
            )
        )
            
        self.wait(2)

        # manim -pqh test2.py txt_test_1

        # manim -sqk test2.py txt_test_1


class bez_test_1(Scene):
    def construct(self):
        b1=bend_bezier_arrow().scale(.7).set_color(PURE_RED).flip().rotate(-PI/4)
        b2=underline_bez_curve()

        t1=MathTex(
            "dim",r"(\mathbb{R}^2)","=","2"
        ).scale(2).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_MAGENTA)
        t1[1:].shift(.1*RIGHT)

        self.add(b1,b2,t1)

        # manim -pqh test2.py bez_test_1

class dim2_ex(Scene):
    def construct(self):
        t1=MathTex(
            "dim",r"(\mathbb{R}^2)","=","2"
        ).scale(2).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_MAGENTA)
        t1[1:].shift(.1*RIGHT)
        b2=underline_bez_curve().next_to(t1,DOWN).scale(2)

        self.add(t1,b2)

        # manim -pqh test2.py dim2_ex
        



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
                image.putpixel((x, y), (i % 8 * 32, i % 16 * 16, i % 32 * 8))

        # Create a image object from the image and add it to the scene
        mandelbrot_image = ImageMobject(image)
        mandelbrot_image.scale(1.5)
        self.add(mandelbrot_image)

        '''self.wait()
        self.play(
            FadeIn(mandelbrot_image),
            run_time=4
        )
        self.wait(2)'''

        # manim -pqh test2.py Mandelbrot




class PythagoreanIdentity(Scene):
    def construct(self):
        
        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        self.wait()


        # SCENE

        ax_1=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(-2)
        
        

        


        dt_1=Dot().set_color(REANLEA_AQUA).move_to(ax_1.c2p(0,0))
        dt_2=Dot().set_color(REANLEA_PURPLE).move_to(ax_1.c2p(3,2))
        dt_3=Dot().set_color(REANLEA_PURPLE).move_to(ax_1.c2p(3,0))
        ln_1=Line(start=dt_1.get_center(), end=dt_2.get_center()).set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA])
        ln_2=Line(start=dt_1.get_center(), end=dt_3.get_center()).set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA])
        ln_3=Line(start=dt_3.get_center(), end=dt_2.get_center()).set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA])

        a_len=ax_1.c2p(3,0)[0]-ax_1.c2p(0,0)[0]
        b_len=ax_1.c2p(3,2)[1]-ax_1.c2p(3,0)[1]
        
        
        self.wait()
        self.play(
            Write(ln_1)
        )
        self.wait(3)

        self.play(
            Write(dt_1)
        )
        self.play(
            Write(dt_2)
        )
        self.play(
            Create(ln_2)
        )
        self.play(
            Write(dt_3)
        )
        self.play(
            Create(ln_3.set_z_index(-1))
        )

        self.wait(2)

        tr_angl=Polygon(dt_1.get_center(),dt_2.get_center(),dt_3.get_center()).set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA, REANLEA_SLATE_BLUE]).set_z_index(-1)

        self.play(
            FadeIn(tr_angl)
        )

        self.play(
            FadeOut(ln_1),
            FadeOut(ln_2),
            FadeOut(ln_3),
        )

        self.wait()

        self.play(
            tr_angl.animate.set_fill(opacity=1, color=REANLEA_BLUE)
        )
        '''self.play(
            tr_angl.animate.set_stroke(width=2, color=REANLEA_WHITE)
        )'''

        self.play(
            FadeOut(dt_1),
            FadeOut(dt_2),
            FadeOut(dt_3),
        )

        a_len_ln=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(3,0)).set_stroke(width=3, color=REANLEA_YELLOW).save_state()
        b_len_ln=DashedLine(start=ax_1.c2p(3,0),end=ax_1.c2p(3,2)).set_stroke(width=3, color=PURE_RED).save_state()
        c_len_ln=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(3,2)).set_stroke(width=3, color=REANLEA_BLUE_DARKER)

        

        self.play(
            Create(c_len_ln)
        )
        self.play(
            Create(a_len_ln)
        )
        self.play(
            Create(b_len_ln)
        )
        self.wait(2)

        a_len_ln_1=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(3,0)).set_stroke(width=3, color=REANLEA_YELLOW).move_to(3.74*RIGHT+2.5*DOWN)
        b_len_ln_1=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(2,0)).set_stroke(width=3, color=PURE_RED).move_to(3.31*RIGHT+2.75*DOWN)
        c_len_ln_1=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(3.61,0)).set_stroke(width=3, color=REANLEA_BLUE_SKY).move_to(4*RIGHT+2.25*DOWN)

        c_len_ln.set_stroke(width=3, color=REANLEA_BLUE_SKY).save_state()

        self.play(
            ReplacementTransform(c_len_ln,c_len_ln_1)
        )
        self.play(
            ReplacementTransform(a_len_ln,a_len_ln_1)
        )
        self.play(
            ReplacementTransform(b_len_ln,b_len_ln_1)
        )

        self.wait(2)
        
        
        tr_angl_0=tr_angl.copy().set_stroke(width=1).set_z_index(-6)
        tr_angl_1=tr_angl.copy().set_z_index(-2).save_state()
        

        self.add(tr_angl_1)

        self.play(
            tr_angl_1.animate.shift(3.5*RIGHT)
        )
        
        tr_angl_1_0=tr_angl_1.copy()
        self.add(tr_angl_1_0)

        tr_angl_1_ref=tr_angl_1.copy().save_state()

        tr_angl_2=tr_angl_1.copy().rotate(PI/2,about_point=ax_1.c2p(4,0))
        tr_angl_3=tr_angl_1.copy().rotate(PI,about_point=ax_1.c2p(4,0))
        tr_angl_4=tr_angl_1.copy().rotate(3*PI/2,about_point=ax_1.c2p(4,0))

        #self.add(tr_angl_3,tr_angl_4)
        


        

        rot_tracker=ValueTracker(0)

        tr_angl_1.add_updater(
            lambda x : x.become(tr_angl_1_ref.copy()).rotate(
                rot_tracker.get_value(), about_point=ax_1.c2p(4,0)
            )
        )
        self.play(
            rot_tracker.animate.set_value(PI/2)
        )
        
        self.play(
            FadeOut(tr_angl_1),
            FadeIn(tr_angl_2)
        )

        
        self.play(
            tr_angl_2.animate.shift((ax_1.c2p(4,0)-ax_1.c2p(3,0))[0]*LEFT)
        )
        self.play(
            tr_angl_2.animate.shift((ax_1.c2p(3,2)-ax_1.c2p(3,0))[1]*UP)
        )
        self.wait(2)

        rot_tracker.set_value(0)
        self.play(
            FadeIn(tr_angl_1)
        )
        
        self.wait()
        
        self.play(
            rot_tracker.animate.set_value(PI),
            run_time=1.5
        )
        self.play(
            FadeIn(tr_angl_3),
            FadeOut(tr_angl_1)
        )
        
        self.play(
            tr_angl_3.animate.shift((ax_1.c2p(1,5.0825)-ax_1.c2p(1,0))[1]*UP)
        )
        self.play(
            tr_angl_3.animate.shift((ax_1.c2p(4,0)-ax_1.c2p(1,0))[0]*LEFT)
        )
        
        self.wait(2)

        rot_tracker.set_value(0)
        self.play(
            FadeIn(tr_angl_1),
            FadeOut(tr_angl_1_0)
        )
        
        self.wait()
        
        self.play(
            rot_tracker.animate.set_value(-PI/2)
        )
        self.play(
            FadeIn(tr_angl_4),
            FadeOut(tr_angl_1)
        )

        self.play(
            tr_angl_4.animate.shift((ax_1.c2p(-2,3.0825)-ax_1.c2p(-2,0))[1]*UP)
        )
        self.play(
            tr_angl_4.animate.shift((ax_1.c2p(4,0)-ax_1.c2p(-2.0825,0))[0]*LEFT)
        )

        tr_angl_grp_1=VGroup(tr_angl,tr_angl_2,tr_angl_3,tr_angl_4)


        
        sq_2=Square(side_length=(ax_1.c2p(1,0)[0]-ax_1.c2p(0,0)[0])*5).set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA, REANLEA_SLATE_BLUE]).set_fill(color=REANLEA_BLUE_LAVENDER, opacity=1).set_z_index(-5)

        sq_2.move_to(ax_1.c2p(6.5,2.5))

        sq_2_ref=sq_2.copy()


        triangles = [tr_angl_2.copy() for i in range(0, 8)]
        time = 0.3

        triangles[0].next_to(sq_2.get_corner(UR), DL, buff=0)
        triangles[1].rotate(PI/2).next_to(sq_2.get_corner(UL), DR, buff=0)
        triangles[2].rotate(PI).next_to(sq_2.get_corner(DL), UR, buff=0)
        triangles[3].rotate(-PI/2).next_to(sq_2.get_corner(DR), UL, buff=0)


        sq_2_grp=VGroup(triangles[0],triangles[1],triangles[2],triangles[3])

        self.play(
            TransformMatchingShapes(tr_angl_grp_1,sq_2_grp)
        )

        self.play(
            FadeIn(sq_2)
        )

        '''self.play(
            Unwrite(sq_2),
            Unwrite(sq_2_grp)
        )'''

        
        sq_2_grp_ref=VGroup(sq_2_ref, triangles[4].become(triangles[0]), triangles[5].become(triangles[1]), triangles[6].become(triangles[2])
        , triangles[7].become(triangles[3]))

        equal = MathTex("=").scale(1.5).move_to(ax_1.c2p(2,2.5)).set_color_by_gradient(REANLEA_AQUA)

        self.play(
            sq_2_grp_ref.animate.move_to(ax_1.c2p(-2.5,2.5))
        )

        self.play(
            Write(equal)
        )

        sq_eras=Square(side_length=.125).set_fill(color=REANLEA_BACKGROUND_COLOR, opacity=1).set_stroke(color=REANLEA_BACKGROUND_COLOR).set_z_index(7)

        sq_eras_1=sq_eras.copy().move_to(ax_1.c2p(-5.1265,3))
        sq_eras_2=sq_eras.copy().move_to(ax_1.c2p(0,-.126))
        sq_eras_3=sq_eras.copy().move_to(ax_1.c2p(.126,0))
        sq_eras_4=sq_eras.copy().move_to(ax_1.c2p(-2,5.1265))
        sq_eras_grp=VGroup(sq_eras_1,sq_eras_2,sq_eras_3,sq_eras_4)
        self.add(sq_eras_grp)

        self.play(triangles[7].animate.move_to(Line(triangles[5].get_corner(DL), triangles[5].get_corner(UR)).get_center()))
        self.play(triangles[4].animate.next_to(sq_2_ref.get_corner(DR), UL, buff=0))
        self.play(triangles[6].animate.next_to(sq_2_ref.get_corner(DR), UL, buff=0))
        self.wait(0.5)

        c_square = Difference(sq_2, Union(triangles[0], triangles[1], triangles[2], triangles[3]), fill_color=REANLEA_BLUE_LAVENDER, fill_opacity=1)
        a_square = Square(side_length=a_len, fill_color=REANLEA_BLUE_LAVENDER, fill_opacity=1).next_to(sq_2_ref.get_corner(DL), UR, buff=0)
        b_square = Square(side_length=b_len, fill_color=REANLEA_BLUE_LAVENDER, fill_opacity=1).next_to(sq_2_ref.get_corner(UR), DL, buff=0)

        c_square.set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA, REANLEA_SLATE_BLUE]).set_z_index(-1)
        b_square.set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA, REANLEA_SLATE_BLUE]).set_z_index(-1)
        a_square.set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA, REANLEA_SLATE_BLUE]).set_z_index(-1)

        
        self.add(c_square)
        #sq_2_ref.set_fill(opacity=0)
        a_b_sq=VGroup(a_square,b_square)
        self.add(a_b_sq)

        triangles_grp=VGroup()
        for i in range(0,8):
            triangles_grp.add(triangles[i])

        self.play(
            Unwrite(triangles_grp),
            Unwrite(sq_2),
            Unwrite(sq_2_ref),
        )
        self.play(
            FadeOut(sq_eras_grp)
        )

        self.wait()


        self.play(
            Restore(a_len_ln),
            Restore(b_len_ln),
            Restore(c_len_ln)
        )

        self.play(
            c_len_ln.animate.set_stroke(width=3, color=REANLEA_BLUE_SKY)
        )

        self.wait(2)

        
        self.play(
            FadeOut(equal),
            c_square.animate.shift((ax_1.c2p(6,0)[0]-ax_1.c2p(0,0)[0])*LEFT)
        )
        self.play(
            b_square.animate.shift((ax_1.c2p(3,0)[0]-ax_1.c2p(-2,3)[0])*RIGHT+(ax_1.c2p(-2,3)[1]-ax_1.c2p(3,0)[1])*DOWN)
        )
        self.play(
            a_square.animate.shift((ax_1.c2p(0,0)[0]-ax_1.c2p(-5,3)[0])*RIGHT+(ax_1.c2p(-5,3)[1]-ax_1.c2p(0,0)[1])*DOWN)
        )
        self.wait(2)

        '''self.play(
            FadeIn(tr_angl_0)
        )
        self.play(
            FadeOut(a_len_ln),
            FadeOut(b_len_ln),
            FadeOut(c_len_ln)
        )'''
        



        

        self.wait(4)


        


        # manim -pqh test2.py PythagoreanIdentity

        # manim -sqk test2.py PythagoreanIdentity



class PythagoreanIdentity_1(Scene):
    def construct(self):
        
        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        self.wait()


        # SCENE

        triangle = Polygon(2 * LEFT, 2 * RIGHT, 2 * LEFT + 3 * UP, stroke_color=WHITE, fill_color=REANLEA_BLUE, fill_opacity=1
        , stroke_width=DEFAULT_STROKE_WIDTH/2).scale(0.5).move_to(ORIGIN)
        self.play(DrawBorderThenFill(triangle), run_time=0.8)
        a = MathTex("a").next_to(triangle, direction=LEFT, buff=0.1)
        b = MathTex("b").next_to(triangle, direction=DOWN, buff=0.08)
        c = MathTex("c").next_to(Line(triangle.get_corner(DR), triangle.get_corner(UL)).get_center(), direction=UR, buff=0.05)
        labels = VGroup(a, b, c)
        self.play(Write(labels))

        square = Square(side_length=0.2, stroke_width=DEFAULT_STROKE_WIDTH/2).next_to(triangle.get_corner(DL), UR, buff=0)
        self.play(FadeIn(square), run_time=0.6)
        self.play(FadeOut(square), run_time=0.4)

        square = Square(side_length=3+4, fill_color=REANLEA_BLUE_LAVENDER, fill_opacity=1).scale(0.5)
        self.play(FadeOut(triangle), Unwrite(labels))
        self.play(Write(square))
        self.play(square.animate.move_to(2.5*RIGHT))

        triangles = [triangle.copy() for i in range(0, 8)]
        time = 0.3
        self.play(Write(triangles[0].next_to(square.get_corner(DL), UR, buff=0)), run_time=time)
        self.play(Write(triangles[1].rotate(-PI/2).next_to(square.get_corner(UL), DR, buff=0)), run_time=time)
        self.play(Write(triangles[2].rotate(PI).next_to(square.get_corner(UR), DL, buff=0)), run_time=time)
        self.play(Write(triangles[3].rotate(PI/2).next_to(square.get_corner(DR), UL, buff=0)), run_time=time)

        c_1 = MathTex("c").next_to(Line(triangles[0].get_corner(DR), triangles[0].get_corner(UL)).get_center(), direction=UR, buff=0.05)
        c_2 = MathTex("c").next_to(Line(triangles[1].get_corner(DL), triangles[1].get_corner(UR)).get_center(), direction=DR, buff=0.05)
        labels = VGroup(c_1, c_2)
        self.play(Write(labels))
        c2 = MathTex(r"c^2").move_to(square.get_center())
        self.play(ReplacementTransform(labels, c2))
        self.add_foreground_mobject(c2)

        square2 = square.copy()
        dupelicate = VGroup(square2, triangles[4].become(triangles[0]), triangles[5].become(triangles[1]), triangles[6].become(triangles[2])
        , triangles[7].become(triangles[3]))
        self.play(dupelicate.animate.move_to(2.5*LEFT))
        equal = MathTex("=")
        self.play(Write(equal))

        self.play(triangles[7].animate.move_to(Line(triangles[5].get_corner(DL), triangles[5].get_corner(UR)).get_center()))
        self.play(triangles[4].animate.next_to(square2.get_corner(DR), UL, buff=0))
        self.play(triangles[6].animate.next_to(square2.get_corner(DR), UL, buff=0))
        self.wait(0.5)

        a_1 = MathTex("a").next_to(triangles[7], direction=DOWN, buff=0.1)
        a_2 = MathTex("a").next_to(triangles[4], direction=LEFT, buff=0.1)
        b_1 = MathTex("b").next_to(triangles[7], direction=RIGHT, buff=0.1)
        b_2 = MathTex("b").next_to(triangles[6], direction=UP, buff=0.1)
        labels = VGroup(a_1, a_2, b_1, b_2)
        self.play(Write(labels))
        self.wait(0.5)

        c_square = Difference(square, Union(triangles[0], triangles[1], triangles[2], triangles[3]), fill_color=REANLEA_BLUE_LAVENDER, fill_opacity=1)
        a_square = Square(side_length=3, fill_color=REANLEA_BLUE_LAVENDER, fill_opacity=1).scale(0.5).next_to(square2.get_corner(DL), UR, buff=0)
        b_square = Square(side_length=4, fill_color=REANLEA_BLUE_LAVENDER, fill_opacity=1).scale(0.5).next_to(square2.get_corner(UR), DL, buff=0)
        c_square.set_stroke(width=DEFAULT_STROKE_WIDTH / 2)
        a_square.set_stroke(width=DEFAULT_STROKE_WIDTH / 2)
        b_square.set_stroke(width=DEFAULT_STROKE_WIDTH / 2)

        a2 = MathTex(r"a^2").move_to(a_square.get_center())
        b2 = MathTex(r"b^2").move_to(b_square.get_center())
        new_labels = VGroup(a2, b2)
        self.play(ReplacementTransform(labels, new_labels))
        self.add_foreground_mobject(a2)
        self.add_foreground_mobject(b2)
        self.wait(0.5)

        square.set_fill(opacity=0)
        self.add(c_square)
        square2.set_fill(opacity=0)
        left_over = VGroup(a_square, b_square)
        self.add(left_over)
        
        triangles_group = VGroup(triangles[0])
        for i in range(1, 8):
            triangles_group.add(triangles[i])
        self.play(Unwrite(triangles_group), Unwrite(square), Unwrite(square2))
        self.play(c_square.animate.set_stroke(width=DEFAULT_STROKE_WIDTH), a_square.animate.set_stroke(width=DEFAULT_STROKE_WIDTH), 
        b_square.animate.set_stroke(width=DEFAULT_STROKE_WIDTH))

        identity = MathTex("a^2", "+", "b^2", "=", "c^2").move_to(3.5 * DOWN)
        self.play(GrowFromCenter(triangle),
        c_square.animate.move_to(ORIGIN).shift(0.75 * RIGHT + 1 * UP),
        a_square.animate.next_to(triangle, LEFT, buff=0),
        b_square.animate.next_to(triangle, DOWN, buff=0),
        ReplacementTransform(a2, identity[0]),
        ReplacementTransform(b2, identity[2]),
        ReplacementTransform(equal, identity[3]),
        ReplacementTransform(c2, identity[4]),
        GrowFromCenter(identity[1]))

        a = MathTex("a").next_to(triangle, direction=LEFT, buff=0.1)
        b = MathTex("b").next_to(triangle, direction=DOWN, buff=0.08)
        c = MathTex("c").next_to(Line(triangle.get_corner(DR), triangle.get_corner(UL)).get_center(), direction=UR, buff=0.05)
        labels = VGroup(a, b, c)
        self.play(Write(labels))
        square = Square(side_length=0.2, stroke_width=DEFAULT_STROKE_WIDTH/2).next_to(triangle.get_corner(DL), UR, buff=0)
        self.play(FadeIn(square), run_time=0.6)
        
        self.wait()


        # manim -pqh test2.py PythagoreanIdentity_1

        # manim -sqk test2.py PythagoreanIdentity_1


class PythagoreanIdentity_2(Scene):
    def construct(self):
        
        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        self.wait()


        # SCENE

        ax_1=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(-2)
        
        

        


        dt_1=Dot().set_color(REANLEA_AQUA).move_to(ax_1.c2p(0,0))
        dt_2=Dot().set_color(REANLEA_PURPLE).move_to(ax_1.c2p(3,2))
        dt_3=Dot().set_color(REANLEA_PURPLE).move_to(ax_1.c2p(3,0))
        ln_1=Line(start=dt_1.get_center(), end=dt_2.get_center()).set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA])
        ln_2=Line(start=dt_1.get_center(), end=dt_3.get_center()).set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA])
        ln_3=Line(start=dt_3.get_center(), end=dt_2.get_center()).set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA])

        a_len=ax_1.c2p(3,0)[0]-ax_1.c2p(0,0)[0]
        b_len=ax_1.c2p(3,2)[1]-ax_1.c2p(3,0)[1]
        
        
        self.wait()
        self.play(
            Write(ln_1)
        )
        self.wait()

        self.play(
            Create(ax_1)
        )
        self.wait(3)

        self.play(
            Write(dt_1)
        )
        self.play(
            Write(dt_2)
        )
        self.play(
            Create(ln_2)
        )
        self.play(
            Write(dt_3)
        )
        self.play(
            Create(ln_3.set_z_index(-1))
        )

        self.wait(2)

        tr_angl=Polygon(dt_1.get_center(),dt_2.get_center(),dt_3.get_center()).set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA, REANLEA_SLATE_BLUE]).set_z_index(-1)

        self.play(
            FadeIn(tr_angl)
        )

        self.play(
            FadeOut(ln_1),
            FadeOut(ln_2),
            FadeOut(ln_3),
        )

        self.wait()

        self.play(
            tr_angl.animate.set_fill(opacity=1, color=REANLEA_BLUE)
        )
        '''self.play(
            tr_angl.animate.set_stroke(width=2, color=REANLEA_WHITE)
        )'''

        self.play(
            FadeOut(dt_1),
            FadeOut(dt_2),
            FadeOut(dt_3),
            Unwrite(ax_1)
        )

        a_len_ln=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(3,0)).set_stroke(width=3, color=REANLEA_YELLOW).save_state()
        b_len_ln=DashedLine(start=ax_1.c2p(3,0),end=ax_1.c2p(3,2)).set_stroke(width=3, color=PURE_RED).save_state()
        c_len_ln=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(3,2)).set_stroke(width=3, color=REANLEA_BLUE_DARKER).save_state()

        a_ln_lab=MathTex("a").scale(.65).set_color(REANLEA_YELLOW).next_to(a_len_ln,DOWN)
        b_ln_lab=MathTex("b").scale(.65).set_color(PURE_RED).next_to(b_len_ln,RIGHT)
        c_ln_lab=MathTex("c").scale(.65).set_color(REANLEA_BLUE_SKY).move_to(ax_1.c2p(1.35,1.35))

        self.play(
            Create(c_len_ln)
        )
        self.play(
            Create(a_len_ln)
        )
        self.play(
            Create(b_len_ln)
        )
        self.wait(2)

        self.play(
            Write(a_ln_lab),
            Write(b_ln_lab),
            Write(c_ln_lab)
        )

        a_len_ln_1=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(3,0)).set_stroke(width=3, color=REANLEA_YELLOW).move_to(3.74*RIGHT+2.5*DOWN)
        b_len_ln_1=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(2,0)).set_stroke(width=3, color=PURE_RED).move_to(3.31*RIGHT+2.75*DOWN)
        c_len_ln_1=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(3.61,0)).set_stroke(width=3, color=REANLEA_BLUE_SKY).move_to(4*RIGHT+2.25*DOWN)

        a_ln_lab_1=MathTex("a").scale(.6).set_color(REANLEA_YELLOW).next_to(a_len_ln_1,LEFT)
        b_ln_lab_1=MathTex("b").scale(.6).set_color(PURE_RED).next_to(b_len_ln_1,LEFT)
        c_ln_lab_1=MathTex("c").scale(.6).set_color(REANLEA_BLUE_SKY).next_to(c_len_ln_1,LEFT)

        self.play(
            ReplacementTransform(c_len_ln,c_len_ln_1),
            ReplacementTransform(c_ln_lab,c_ln_lab_1)
        )
        self.play(
            ReplacementTransform(a_len_ln,a_len_ln_1),
            ReplacementTransform(a_ln_lab,a_ln_lab_1)
        )
        self.play(
            ReplacementTransform(b_len_ln,b_len_ln_1),
            ReplacementTransform(b_ln_lab,b_ln_lab_1)
        )

        self.wait(2)

        self.play(
            Restore(a_len_ln),
            Restore(b_len_ln),
            Restore(c_len_ln)
        )
        
        pythagoras_thm=MathTex(r"c^2","=",r"a^2","+",r"b^2").to_corner(UR, buff=1)

        sr_pythagoras_thm=SurroundingRectangle(pythagoras_thm, color=REANLEA_WELDON_BLUE, corner_radius=.125, buff=.25)

        self.play(
            Write(pythagoras_thm)
        )

        self.wait()

        self.play(
            Unwrite()
        )
        
        
        
        

        self.wait(4)


        


        # manim -pqh test2.py PythagoreanIdentity_2

        # manim -sqk test2.py PythagoreanIdentity_2


class SquareFill(Scene):
    def construct(self):
        
        sq=Square().set_fill(color=PURE_GREEN, opacity=1).set_stroke(color=PURE_GREEN)
        
        self.play(
            FadeIn(sq)
        )

        ax_1=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(2)

        ax_2=ax_1.copy()

        self.play(
            Write(ax_1),
            run_time=2
        )

        self.wait()

        self.play(
            Unwrite(ax_1)
        )
        self.wait(2)


        self.play(
            Create(ax_2),
            run_time=2
        )
        self.wait()

        self.play(
            Uncreate(ax_2)
        )

        self.wait(2)



        # manim -sqk test2.py SquareFill

        # manim -pqh test2.py SquareFill


class arr_tst_0(Scene):
    def construct(self):
        bez_arr_1=bend_bezier_arrow().flip(UP).rotate(-45*DEGREES)

        self.add(bez_arr_1)


        # manim -sqk test2.py arr_tst_0

        # manim -pqh test2.py arr_tst_0


class VMobject(VMobject):
    def pfp(self, alpha):
        return self.point_from_proportion(alpha)


class Dot_move_on_curve_set_point_smoothly(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        
        #c = Circle()
        c = VMobject(stroke_color=GREEN).set_points_smoothly([
                LEFT*2+UP*1.2, LEFT+UP*(-3), RIGHT*2+DOWN*1.7,
                DOWN*2+LEFT*2.5
            ])
        c1=VMobject().set_points_as_corners(points=[ORIGIN,2*UP,2*UP+3*RIGHT]) 
        a = Dot(color = YELLOW)

        self.add(a)

        self.play(UpdateFromAlphaFunc(a, lambda x, alpha: x.move_to(c1.pfp(alpha))), run_time = 3, rate_func= smooth)
        self.wait()


        # manim -pqh test2.py Dot_move_on_curve_set_point_smoothly



class Vx(VMobject):
    def pfp(self, alpha):
        return self.point_from_proportion(alpha)


class Dot_move_on_curve_set_point_smoothly_1(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        
        
        c1=Vx().set_points_as_corners(points=[ORIGIN,2*UP,2*UP+3*RIGHT]) 
        a = Dot(color = YELLOW)

        self.add(a)

        self.play(UpdateFromAlphaFunc(a, lambda x, alpha: x.move_to(c1.pfp(alpha))), run_time = 3, rate_func= smooth)
        self.wait()


        # manim -pqh test2.py Dot_move_on_curve_set_point_smoothly_1



class Scene4(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        


        # SCENE

        ax_1=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1) 

        rot_tracker=ValueTracker(0)

        ax_2=ax_1.copy()

        ax_2_ref=ax_2.copy()

        ax_2.add_updater(
            lambda x : x.become(ax_2_ref.copy()).rotate(
                rot_tracker.get_value(), about_point=ax_1.c2p(0,0)
            )
        )

        self.wait()

        self.play(
            Write(ax_1),
            run_time=2
        )

        dot_1=Dot(radius=0.15, color=REANLEA_AQUA_GREEN).move_to(ax_1.c2p(0,0)).set_sheen(-0.4,DOWN).set_z_index(3).save_state()

        self.play(
            Write(dot_1)
        )


        push_arr=Arrow(start=ax_1.c2p(-.8,0),end=ax_1.c2p(-.4,0),max_tip_length_to_length_ratio=.5, buff=0).set_color(REANLEA_YELLOW_GREEN).set_opacity(1).set_z_index(6)

        push_arr_lbl=Text("F").scale(.5).set_color(REANLEA_YELLOW_GREEN).next_to(push_arr,UP)
        push_arr_lbl.add_updater(
            lambda z : z.become(
                Text("F").scale(.5).set_color(REANLEA_YELLOW_GREEN).next_to(push_arr,UP)
            )
        )

        self.play(
            FadeIn(push_arr)
        )
        
        self.play(
            push_arr.animate.move_to(ax_1.c2p(-.35,0)),
            run_time=.35
        )
        self.play(
            dot_1.animate.move_to(ax_1.c2p(1,0))
        )  


        lbl_i=MathTex("1").scale(.55).set_color(REANLEA_AQUA).move_to(ax_1.c2p(1,-.5))  
        dt_1=Dot().set_color(REANLEA_AQUA).move_to(ax_1.c2p(1,0)).set_z_index(2)
        dt_2=Dot().set_color(REANLEA_PURPLE).move_to(ax_1.c2p(3,2)).set_z_index(2)

        self.play(
            FadeOut(push_arr),
            Write(lbl_i),
            Create(dt_1)
        )
        
        self.wait(2)

        self.play(
            Restore(dot_1)
        )

        push_arr_1=Arrow(start=ax_1.c2p(-.78,-.52),end=ax_1.c2p(-.45,-.3),max_tip_length_to_length_ratio=.5, buff=0).set_color(REANLEA_YELLOW_GREEN).set_opacity(1).set_z_index(6).save_state()
        
        self.play(
            FadeIn(push_arr_1)
        )
        
        self.play(
            push_arr_1.animate.move_to(ax_1.c2p(-.27,-.18)),
            run_time=.35
        )
        self.play(
            dot_1.animate.move_to(ax_1.c2p(3,2))
        )
        self.play(
            FadeOut(push_arr_1)
        )
        self.wait(2)

        self.add(dt_2)

        self.play(
            Restore(dot_1)
        )

        self.add(ax_2)        

        self.play(
            rot_tracker.animate.set_value(PI/4),
            run_time=4
        )

        

        push_arr_2=Arrow(start=ax_1.c2p(-.78,-.52),end=ax_1.c2p(-.45,-.3),max_tip_length_to_length_ratio=.5, buff=0).set_color(REANLEA_YELLOW_GREEN).set_opacity(1).set_z_index(6).save_state()
        
        self.play(
            FadeIn(push_arr_2)
        )
        
        self.play(
            push_arr_2.animate.move_to(ax_1.c2p(-.27,-.18)),
            run_time=.35
        )        



        self.wait(4)


        

        # manim -pqh test2.py Scene4

        # manim -sqk test2.py Scene4



class weier(Scene):
    def construct(self):
        n = 300
        a = ValueTracker(0.5)
        b = ValueTracker(0.6)
        xrng = ValueTracker(4)

        ax = Axes()
        func = VMobject()
        def axUpdater(mobj):
            xmin = -xrng.get_value()
            xmax = +xrng.get_value()
            newax =Axes(x_range=[xmin,xmax,10**int(np.log10(xmax)-1)],y_range=[-1,4])
            newax.add_coordinates()
            newfunc = newax.plot(
                lambda x: sum([a.get_value()**k*np.cos(b.get_value()**k*PI*x) for k in range(n)]),
                x_range=[xmin,xmax,xrng.get_value()/200],
                use_smoothing=False,
                ).set_color(RED).set_stroke(width=3)
            mobj.become(newax)
            func.become(newfunc)            
        ax.add_updater(axUpdater)

        self.add(ax,func)

        self.play(
            b.animate.set_value(7),
            run_time=2
        )        
        self.wait(2)
        self.play(
            xrng.animate.set_value(0.01),
            run_time=10
        ) 


        # manim -pqh test2.py weier



class AxesScale(Scene):
    def construct(self):

        xrng = ValueTracker(4)

        ax = Axes()

        def axUpdater(mobj):
            xmin = -xrng.get_value()
            xmax = +xrng.get_value()
            newax =Axes(x_range=[xmin,xmax,2**int(np.log10(xmax)-1)],y_range=[-1,4])
            newax.add_coordinates()
            mobj.become(newax)          
        ax.add_updater(axUpdater)

        self.add(ax)

               
        self.wait(2)
        self.play(
            xrng.animate.set_value(1),
            run_time=4
        ) 


        # manim -pqh test2.py AxesScale



class AxesScale_1(Scene):
    def construct(self):

        xrng_max = ValueTracker(5.5)
        xrng_min = ValueTracker(-1.5)

        ax=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)

        def axUpdater(mobj):
            xmin = xrng_min.get_value()
            xmax = xrng_max.get_value()
            newax =Axes(
                    x_range=[xmin,xmax,2**int(np.log10(xmax)-1)],
                    y_range=[-1.5,4.5],
                    y_length=(round(config.frame_width)-2)*6/7,
                    tips=False, 
                    axis_config={
                        "font_size": 24,
                        #"include_ticks": False,
                    }, 
                ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)
            newax.add_coordinates()
            mobj.become(newax)          
        ax.add_updater(axUpdater)

        self.add(ax)

               
        self.wait(2)
        self.play(
            xrng_max.animate.set_value(1),
            xrng_min.animate.set_value(1),
            run_time=4
        ) 


        # manim -pqh test2.py AxesScale_1



class AxesScale_2(Scene):
    def construct(self):

        xrng= ValueTracker(5.5)
        

        ax=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)

        def axUpdater(mobj):
            xmin = -xrng.get_value()
            xmax = xrng.get_value()
            newax =Axes(
                    x_range=[xmin,xmax,2**int(np.log10(xmax)-1)],
                    y_range=[-1.5,4.5],
                    y_length=(round(config.frame_width)-2)*6/7,
                    tips=False, 
                    axis_config={
                        "font_size": 24,
                    }, 
                ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)
            newax.add_coordinates()
            mobj.become(newax)          
        ax.add_updater(axUpdater)

        self.add(ax)

               
        self.wait(2)
        self.play(
            xrng.animate.set_value(1),
            run_time=4
        ) 


        # manim -pqh test2.py AxesScale_2



class LogScalingExample(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 10, 1],
            y_range=[-2, 6, 1],
            tips=False,
            axis_config={"include_numbers": True},
            y_axis_config={"scaling": LogBase(custom_labels=True)},
        )

        ax_1 = Axes(
            x_range=[0, 10, 1],
            y_range=[-2, 6, 1],
            tips=False,
            axis_config={"include_numbers": True},
            x_axis_config={"scaling": LinearBase(.5)},
            y_axis_config={"scaling": LogBase(custom_labels=True)},
        )

        # x_min must be > 0 because log is undefined at 0.
        graph = ax.plot(lambda x: x ** 2, x_range=[0.001, 10], use_smoothing=False)
        self.add(ax)

        self.play(
            Create(graph)
        )
        self.wait()

        self.play(
            Transform(ax,ax_1)
        )
        self.wait()


        # manim -pqh test2.py LogScalingExample



class scale_ax(Scene):
    def construct(self):

        xrng = ValueTracker(4)

        ax = Axes(x_range=[-4,4,2**int(np.log10(4)-1)])
        ax.add_coordinates()

        ax_1 = Axes(x_range=[-4,4,2**int(np.log10(1)-1)]).set_color(RED).set_z_index(-1).shift(UP)

        def axUpdater(mobj):
            xmin = -xrng.get_value()
            xmax = +xrng.get_value()
            newax =Axes(x_range=[xmin,xmax,2**int(np.log10(xmax)-1)])
            newax.add_coordinates()
            mobj.become(newax)          
        ax.add_updater(axUpdater)

        self.add(ax)
        self.add(ax_1)        
        self.wait()

        self.play(
            xrng.animate.set_value(2),
            run_time=4
        )
        self.wait(2)


        # manim -pqh test2.py scale_ax


class scAx(Scene):
    def construct(self):

        ax_1=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)

        ax_2_pre=Axes(
            x_range=[-5.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        )
        ax_2=Axes(
            x_range=[-5.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL).scale(.5).set_z_index(1)
        #.shift((ax_2_pre.c2p(0,0)[0]-ax_1.c2p(0,0)[0])*LEFT)

        self.wait()
        self.play(
            Create(ax_1)
        )
        self.wait()
        self.play(
            Create(ax_2),
            run_time=2
        )
        self.wait()

        # manim -pqh test2.py scAx

        # manim -sqk test2.py scAx



class weier(Scene):
    def construct(self):
        n = 300
        a = ValueTracker(0.5)
        b = ValueTracker(0.6)
        xrng = ValueTracker(4)

        ax = Axes()
        func = VMobject()
        def axUpdater(mobj):
            xmin = -xrng.get_value()
            xmax = +xrng.get_value()
            newax =Axes(x_range=[xmin,xmax,10**int(np.log10(xmax)-1)],y_range=[-1,4])
            newax.add_coordinates()
            newfunc = newax.plot(
                lambda x: sum([a.get_value()**k*np.cos(b.get_value()**k*PI*x) for k in range(n)]),
                x_range=[xmin,xmax,xrng.get_value()/200],
                use_smoothing=False,
                ).set_color(RED).set_stroke(width=3)
            mobj.become(newax)
            func.become(newfunc)            
        ax.add_updater(axUpdater)

        self.add(ax,func)

        self.play(
            b.animate.set_value(7),
            run_time=2
        )        
        self.wait(2)
        self.play(
            xrng.animate.set_value(0.01),
            run_time=10
        ) 


        # manim -pqh test2.py weier



class scale_ax_1(Scene):
    def construct(self):

        xrng_min = ValueTracker(1.5)
        xrng = ValueTracker(5.5)
        n=10
        

        ax = Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)
        ax.add_coordinates()
        func=VGroup()

        def axUpdater(mobj):
            xmin = -xrng_min.get_value()
            xmax = +xrng.get_value()
            newax=Axes(
                    x_range=[xmin,xmax,2**int(np.log10(xmax)-1)],
                    y_range=[-1.5,4.5],
                    y_length=(round(config.frame_width)-2)*6/7,
                    tips=False, 
                    axis_config={
                            "font_size": 24,
                        }, 
                ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)
            newax.add_coordinates()
            newfunc=newax.plot(
                lambda z : z**k for k in range(n)
                )
            mobj.become(newax)
            func.become(newfunc)            
        ax.add_updater(axUpdater)
                

        self.add(ax,func)     
        self.wait()

        self.play(
            xrng.animate.set_value(2.75),
            xrng_min.animate.set_value(.75),
            run_time=4
        )
        self.wait(2)


        # manim -pqh test2.py scale_ax_1



class weier_1(Scene):
    def construct(self):
    
        xrng = ValueTracker(5.5)
        xrng_min = ValueTracker(1.5)

        ax_ref = Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(3)
        self.play(
            Write(ax_ref)
        )
        func_ref=ax_ref.plot(
                lambda x: x,
                x_range=[0,2],
                use_smoothing=False,
                ).set_color(PURE_GREEN).set_stroke(width=5).set_z_index(3)
        
        self.play(
            Create(func_ref)
        )
        


        ax=VGroup()
        

        func = VMobject()
        def axUpdater(mobj):
            xmin = -xrng_min.get_value()
            xmax = +xrng.get_value()
            #newax =Axes(x_range=[xmin,xmax,10**int(np.log10(xmax)-1)],y_range=[-1,4])
            newax=Axes(
                    x_range=[xmin,xmax,2**int(np.log10(xmax)-1)],
                    y_range=[-1.5,4.5],
                    y_length=(round(config.frame_width)-2)*6/7,
                    tips=False, 
                    axis_config={
                            "font_size": 24,
                        }, 
                ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)
            newfunc = newax.plot(
                lambda x: x,
                x_range=[0,2],
                use_smoothing=False,
                ).set_color(PURE_GREEN).set_stroke(width=5)
            mobj.become(newax)
            func.become(newfunc)            
        ax.add_updater(axUpdater)

        self.add(ax,func)
        
        self.wait()

        self.play(
            FadeOut(ax_ref)
        )
        

               
        self.wait(2)
        self.play(
            xrng.animate.set_value(2.75),
            xrng_min.animate.set_value(.75),
            run_time=4
        ) 


        # manim -pqh test2.py weier_1


class flash_disp_ex(Scene):
    def construct(self):

        ax_1=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)

        dot_1=Dot(radius=0.15, color=REANLEA_AQUA_GREEN).move_to(ax_1.c2p(0,0)).set_sheen(-0.4,DOWN).set_z_index(3).save_state()

        dt_0=Dot().set_color(REANLEA_YELLOW).move_to(ax_1.c2p(0,0)).set_z_index(2)
        self.add(dt_0)

        x_proj_ln=DashedLine(start=dt_0.get_center(),end=dot_1.get_center()).set_stroke(width=3, color=REANLEA_YELLOW_CREAM).set_z_index(2)
        
        x_proj_ln.add_updater(
            lambda z : z.become(
                DashedLine(start=dt_0.get_center(),end=dot_1.get_center()).set_stroke(width=3, color=REANLEA_YELLOW_CREAM).set_z_index(2)
            )
        )

        dissipating_dt_1=Dot().move_to(ax_1.c2p(3,2)).set_opacity(opacity=0)
        dissipating_path_1 = TracedPath(dissipating_dt_1.get_center, dissipating_time=0.5, stroke_color=[REANLEA_AQUA,PURE_GREEN],stroke_opacity=[1, 0])
        self.add(dissipating_dt_1,dissipating_path_1)

        self.wait()

        self.add(ax_1,dt_0,dot_1,x_proj_ln,dissipating_dt_1)

        self.play(
            dot_1.animate.move_to(ax_1.c2p(3,0)),
            dissipating_dt_1.animate.move_to(ax_1.c2p(3,0)),
            Flash(point=Dot().move_to(ax_1.c2p(3,0)), color=REANLEA_GREEN_AUQA),
        )

        self.wait(2)



        # manim -pqh test2.py flash_disp_ex
        
###################################################################################################################


# Changing FONTS : import any font from Google
# some of my fav fonts: Cinzel,Kalam,Prata,Kaushan Script,Cormorant, Handlee,Monoton, Bad Script, Reenie Beanie, 
# Poiret One,Merienda,Julius Sans One,Merienda One,Cinzel Decorative, Montserrat, Cousine, Courier Prime , The Nautigal
# Marcellus SC,Contrail One,Thasadith,Spectral SC,Dongle,Cormorant SC,Comfortaa, Josefin Sans (LOVE), Fuzzy Bubbles

###################################################################################################################

# N.B.-
#  alt + z --> to controll (on/off) horizontal scrolling
      
###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix"






