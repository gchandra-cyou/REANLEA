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
from manim_physics import *
import pandas


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

config.background_color= WHITE
config.max_files_cached=500


###################################################################################################################

class post_1(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("water_mark_white.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(1).set_z_index(-100)
        self.add(water_mark)

        # MAIN SCENE

        np=NumberPlane(
            axis_config={
                "stroke_width":DEFAULT_STROKE_WIDTH*.4,
                "color": REANLEA_BLUE_DARKEST
            },
            background_line_style={
                "stroke_width": DEFAULT_STROKE_WIDTH*.25,
                "stroke_color": REANLEA_GREY_DARKER
            },
            tips=False,
        ).set_z_index(-5)

        np_1=NumberPlane(
            axis_config={
                "stroke_width":DEFAULT_STROKE_WIDTH*.15,
                "color": REANLEA_BLUE_DARKEST
            },
            background_line_style={
                "stroke_width": DEFAULT_STROKE_WIDTH*.085,
                "stroke_color": REANLEA_GREY_DARKER,
            },
            tips=False
        ).set_z_index(-6)
        np_1.add_coordinates()

        cir_1=Circle(color=REANLEA_BACKGROUND_COLOR_OXFORD_BLUE)
        cir_2=Circle().set_stroke(width=DEFAULT_STROKE_WIDTH,color=[REANLEA_WARM_BLUE,REANLEA_BLUE_DARKER]).scale(.5)

        
        self.play(
            Write(np)
        )
        self.play(
            Write(np_1)
        )
        self.play(
            Write(cir_1),
            Write(cir_2)
        )
        self.wait()

        matrix_1 = [[2.5,0], [0, 6]]
        matrix_2 = [[2/sqrt(5), 1/sqrt(5)], [-1/sqrt(5), 2/sqrt(5)]]

        cir_1.scale(.5)

        self.play(ApplyMatrix(matrix_1, np),ApplyMatrix(matrix_1, cir_1))   
        self.play(ApplyMatrix(matrix_2, np),ApplyMatrix(matrix_2, cir_1))   

        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("Circle & Ellipse are equivalent under an affine transformation." , font=fonts[0]).set_color_by_gradient("#A8A8A8")

        txt_1.scale(.35).shift(3.35*DOWN)

        self.add(txt_1)
        
        
        self.wait()

        

        # manim -pqh post.py post_1

        # manim -sqk post.py post_1

class post_2(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("water_mark_white.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(1).set_z_index(-100)
        self.add(water_mark)

        np=NumberPlane(
            axis_config={
                "stroke_width":DEFAULT_STROKE_WIDTH*.15,
                "color": REANLEA_GREY_DARKER
            },
            background_line_style={
                "stroke_width": DEFAULT_STROKE_WIDTH*.085,
                "stroke_color": REANLEA_GREY_DARKER,
            },
            tips=False
        ).set_z_index(-6)
        np.add_coordinates()

        self.add(np)

        ax=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(-5)
        ax_c=ax.copy()
        ax.shift((np.c2p(0,0)[0]-ax_c.c2p(0,0)[0])*RIGHT+(np.c2p(0,0)[1]-ax_c.c2p(0,0)[1])*UP*2)

        
        def graph(n=1,stroke_width=3):
            grph = VGroup(
                ax.plot(
                    lambda x: 0,
                    x_range=[-1,1,0.01]
                ).set_z_index(-1).set_stroke(width=2.5,color=REANLEA_GREY),
                ax.plot(
                    lambda x: 0,
                    x_range=[-1,0,0.01]
                ).set_stroke(width=stroke_width/n, color=REANLEA_BLUE_DARKEST),
                ax.plot(
                    lambda x: n*x,
                    x_range=[0,1/n,0.01]
                ).set_stroke(width=stroke_width/n,color=REANLEA_BLUE_DARKEST),
                ax.plot(
                    lambda x: 1,
                    x_range=[1/n,1,0.01]
                ).set_stroke(width=stroke_width/n,color=REANLEA_BLUE_DARKEST)
            )
            return grph
        
        x1=graph(n=1).scale(4)
        x2=graph(n=2).scale(4)
        
        self.play(
            Create(x1)
        )
        self.play(
            Create(x2)
        )

        x=VGroup(
            *[
                graph(n=i).scale(4)
                for i in range(3,40)
            ]
        )
        self.play(
            Create(x)
        )

        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("the space of continuous functions C[-1,1] with 1-norm is Cauchy but not Complete." , font=fonts[0]).set_color_by_gradient("#AEAEAE")

        txt_1.scale(.35).shift(3*DOWN)

        q1= MathTex(r"f_{n}(x) := \begin{cases}"
                r"0  &  \text{ if} \ \ \ -1 \leq x < 0 \\"
                r"nx &  \text{ if} \ \ \ 0 \leq x \leq \frac{1}{n} \\"
                r"1 &  \text{ if} \ \ \ \frac{1}{n} < x \leq 1 \\"
                r"\end{cases}"
        ).set_color("#000000").scale(.5).next_to(txt_1,UP).shift(.5*UP)

        self.add(txt_1,q1)


        self.wait(2)


        # manim -pqh post.py post_2

        # manim -sqk post.py post_2

###################################################################################################################

# Changing FONTS : import any font from Google
# some of my fav fonts: Cinzel,Kalam,Prata,Kaushan Script,Cormorant, Handlee,Monoton, Bad Script, Reenie Beanie, 
# Poiret One,Merienda,Julius Sans One,Merienda One,Cinzel Decorative, Montserrat, Cousine, Courier Prime , The Nautigal
# Marcellus SC,Contrail One,Thasadith,Spectral SC,Dongle,Cormorant SC,Comfortaa, Josefin Sans (LOVE), Fuzzy Bubbles

###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2023\lab" 