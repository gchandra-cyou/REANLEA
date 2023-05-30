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



config.background_color= "#FFFFFF"
config.max_files_cached=500


###################################################################################################################

class assgn1_1(Scene):
    def construct(self):
        with RegisterFont("Cousine") as fonts:
            txt_1 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Let",
                "Is the following a metric on this space?"
            )]).arrange_submobjects(RIGHT).scale(0.4).set_color("#000000")
        txt_2=MathTex(r"V= \mathbb{R}^{n}",".").scale(.5).next_to(txt_1[0]).set_color("#000000")
        txt_1[1].next_to(txt_2)
        q1= MathTex(r"d(x,y)",":=",r"\sum_{k=1}^{n}",r"\lvert x_{k}-y_{k} \rvert").set_color("#000000").scale(.75).next_to(txt_1,DOWN)
        
        self.add(q1,txt_1,txt_2)

        # manim -sqk assgn.py assgn1_1



class assgn1_2(Scene):
    def construct(self):
        with RegisterFont("Cousine") as fonts:
            txt_1 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Let",
                "Is the following a metric on this space?"
            )]).arrange_submobjects(RIGHT).scale(0.4).set_color("#000000")
        txt_2=MathTex(r"V= \mathbb{R}^{n}",".").scale(.5).next_to(txt_1[0]).set_color("#000000")
        txt_1[1].next_to(txt_2)
        q1= MathTex(r"d(x,y)",":=",r"\sum_{k=1}^{n}",r"\lvert x_{k}-y_{k} \rvert^{\frac{1}{2}}").set_color("#000000").scale(.75).next_to(txt_1,DOWN)
        
        self.add(q1,txt_1,txt_2)

        # manim -sqk assgn.py assgn1_2


class assgn1_3(Scene):
    def construct(self):
        with RegisterFont("Cousine") as fonts:
            txt_1 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Let",
                "Is the following a metric on this space?"
            )]).arrange_submobjects(RIGHT).scale(0.4).set_color("#000000")
        txt_2=MathTex(r"V= \mathbb{R}^{n}",".").scale(.5).next_to(txt_1[0]).set_color("#000000")
        txt_1[1].next_to(txt_2)
        q1= MathTex(r"d(x,y)",":=",r"max \bigl\{ \lvert x_{k}-y_{k} \rvert : 1 \leq k \leq n \bigr\}").set_color("#000000").scale(.75).next_to(txt_1,DOWN)
        
        self.add(q1,txt_1,txt_2)

        # manim -sqk assgn.py assgn1_3


class assgn1_4(Scene):
    def construct(self):
        with RegisterFont("Cousine") as fonts:
            txt_1 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Let",
                "be a non-empty set. Is the following a metric?"
            )]).arrange_submobjects(RIGHT).scale(0.4).set_color("#000000")
        txt_2=MathTex(r"X").scale(.5).next_to(txt_1[0]).set_color("#000000")
        txt_1[1].next_to(txt_2)
        q1= MathTex(r"d(x,y) := \begin{cases}"
                r"0  &  \text{ if} \ \ \ x=y \\"
                r"1 &  \text{ if} \ \ \ x \neq y"
                r"\end{cases}"
        ).set_color("#000000").scale(.75).next_to(txt_1,DOWN)
        
        self.add(q1,txt_1,txt_2)

        # manim -sqk assgn.py assgn1_4


class assgn1_5(Scene):
    def construct(self):
        with RegisterFont("Cousine") as fonts:
            txt_1 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Let",
                "be a non-empty set. Is the following a metric?"
            )]).arrange_submobjects(RIGHT).scale(0.4).set_color("#000000")
        txt_2=MathTex(r"X").scale(.5).next_to(txt_1[0]).set_color("#000000")
        txt_1[1].next_to(txt_2)
        q1= MathTex(r"d(x,y) := \begin{cases}"
                r"1  &  \text{ if} \ \ \ x=y \\"
                r"0 &  \text{ if} \ \ \ x \neq y"
                r"\end{cases}"
        ).set_color("#000000").scale(.75).next_to(txt_1,DOWN)
        
        self.add(q1,txt_1,txt_2)

        # manim -sqk assgn.py assgn1_5

###################################################################################################################

# Changing FONTS : import any font from Google
# some of my fav fonts: Cinzel,Kalam,Prata,Kaushan Script,Cormorant, Handlee,Monoton, Bad Script, Reenie Beanie, 
# Poiret One,Merienda,Julius Sans One,Merienda One,Cinzel Decorative, Montserrat, Cousine, Courier Prime , The Nautigal
# Marcellus SC,Contrail One,Thasadith,Spectral SC,Dongle,Cormorant SC,Comfortaa, Josefin Sans (LOVE), Fuzzy Bubbles

###################################################################################################################


###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix" 