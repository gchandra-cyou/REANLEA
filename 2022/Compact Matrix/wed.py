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



config.background_color= REANLEA_BACKGROUND_COLOR_OXFORD_BLUE
config.max_files_cached=500
config.frame_height = 16
config.frame_width = 9
config.pixel_width = 1080*3
config.pixel_height = 1920*3
config.frame_rate = 60

###################################################################################################################

class Ex(Scene):
    def construct(self):

        # IMAGE GROUP
        
        im=ImageMobject("wed.png").scale(.525).shift(3.5*UP)
        self.add(im)

        im_1=ImageMobject("ganesh.png").scale(.125).shift(3*RIGHT+ 0.5*DOWN)
        self.add(im_1)


        # NAME GROUP

        with RegisterFont("Great Vibes") as fonts:
            txt_1 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Susmita",
                "&",
                "Shatadru"
            )]).arrange_submobjects(1.25*DOWN)
            txt_1[0].set_color_by_gradient(GOLD, GOLD_E).scale(1.5)
            txt_1[1].set_color_by_gradient(GOLD).shift(.1*DOWN)
            txt_1[2].set_color_by_gradient(GOLD, GOLD_E).scale(1.5)

        with RegisterFont("Allura") as fonts:
            txt_d_p = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Sanatan Dutta & Lakshmi Dutta",
                "request the honour of your presence on the auspicious wedding ceremony",
                "of their daughter"
            )]).arrange_submobjects(.01*DOWN)

            txt_d_p[0].scale(.75).shift(1.75*UP).set_color_by_gradient(GOLD).set_sheen(-.2,DOWN)
            txt_d_p[1].scale(.5).shift(1.7*UP).set_color_by_gradient(REANLEA_PURPLE_LIGHTER).set_sheen(-.05,DOWN)
            txt_d_p[2].scale(.5).shift(3*LEFT +1.85*UP).set_color_by_gradient(REANLEA_PURPLE_LIGHTER).set_sheen(-.05,DOWN)

            txt_s_p = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Son of",
                "Swapan Kumar Niyogi & Kaberi Niyogi",
            )]).arrange_submobjects(.01*DOWN)

            txt_s_p[0].scale(.5).shift(1.95*DOWN).set_color_by_gradient(REANLEA_PURPLE_LIGHTER).set_sheen(-.05,DOWN)
            txt_s_p[1].scale(.65).shift(1.75*DOWN).set_color_by_gradient(GOLD).set_sheen(-.2,DOWN)


        
            


        #DATE GRODOWN

        with RegisterFont("MonteCarlo") as fonts:
            txt_2 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Sunday",
                "27",
                "at 8 pm,"
            )]).scale(.4).arrange_submobjects(1.5*RIGHT)
            txt_2.shift(2*DOWN).scale(2)
            txt_2.set_color_by_gradient(REANLEA_AQUA)

            txt_2[1].scale(3.5).set_color(GOLD).set_sheen(-.4, DOWN)
        
        strp_1=get_stripe(factor=.05, buff_max=2).next_to(txt_2[2], .25*UP).shift(.25*RIGHT)
        strp_2=get_stripe(factor=.05, buff_max=2).next_to(txt_2[2], .25*DOWN).shift(.25*RIGHT)
        strp_3=get_stripe(factor=.05, buff_max=2).next_to(txt_2[0], .25*UP).shift(.25*LEFT).flip(UP)
        strp_4=get_stripe(factor=.05, buff_max=2).next_to(txt_2[0], .25*DOWN).shift(.25*LEFT).flip(UP)
        strp=VGroup(strp_1,strp_2,strp_3,strp_4)

        txt_2_grp=VGroup(txt_2,strp)

        with RegisterFont("Kolker Brush") as fonts:
            txt_3 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "November",
                "2022"
            )]).scale(1).set_color(REANLEA_YELLOW_CREAM)

            txt_3[0].next_to(txt_2[1],UP)
            txt_3[1].next_to(txt_2[1],DOWN)

        


        # VENUE GROUP

        
        with RegisterFont("Kolker Brush") as fonts:
            txt_4=Text("Venue", font=fonts[0])
            txt_4.set_color_by_gradient(PURE_RED).shift(5.5*DOWN).scale(1.5)

        strp_5=get_stripe(factor=.04, buff_max=1.35, color=PURE_RED).next_to(txt_4,.25*DOWN).shift(.15*RIGHT)

        txt_4_grp=VGroup(txt_4,strp_5).scale(.5).to_edge(LEFT).shift(1.45*RIGHT+.55*DOWN).rotate(PI/4)

        with RegisterFont("Shalimar") as fonts:
            txt_5 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Divya Kalpataru Bhaban and Lakshman Bhaban",
                "Sreebas Angan Ghat",
                "Nabadwip, Nadia"
            )]).arrange_submobjects(DOWN).scale(.5).shift(7*DOWN)
            txt_5.set_color_by_gradient(REANLEA_YELLOW_CREAM).set_sheen(-.2,DOWN)
            txt_5.set_stroke(width=1)

        


        # GROUP REGION

        name_grp=VGroup(txt_1, txt_d_p,txt_s_p).shift(.25*DOWN)
        date_grp=VGroup(txt_2_grp,txt_3).shift(2.5*DOWN).scale(.9)
        venue_grp=VGroup(txt_4_grp,txt_5)


        # PLAY REGION
  
        self.wait()
        self.play(
            Write(name_grp)
        )
        self.play(
            Write(date_grp)
        )

        self.play(
            Write(venue_grp)
        )
        
        

        self.wait(2)

        # manim -pqh wed.py Ex

        # manim -sqk wed.py Ex



###################################################################################################################

# Playfair Display SC , Great Vibes , Merienda , Tangerine , Shalimar , Parisienne , Allura , Playball , Bad Script
# Cormorant SC, Montserrat, MonteCarlo , Kolker Brush

###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix"