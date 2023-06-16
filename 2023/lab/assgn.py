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

        

        # manim -pqh assgn.py post_1

        # manim -sqk assgn.py post_1







###################################################################################################################

# Changing FONTS : import any font from Google
# some of my fav fonts: Cinzel,Kalam,Prata,Kaushan Script,Cormorant, Handlee,Monoton, Bad Script, Reenie Beanie, 
# Poiret One,Merienda,Julius Sans One,Merienda One,Cinzel Decorative, Montserrat, Cousine, Courier Prime , The Nautigal
# Marcellus SC,Contrail One,Thasadith,Spectral SC,Dongle,Cormorant SC,Comfortaa, Josefin Sans (LOVE), Fuzzy Bubbles

###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2023\lab" 