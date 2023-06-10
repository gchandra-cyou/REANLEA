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


class dim2_ex(Scene):
    def construct(self):
        t1=MathTex(
            "dim",r"(\mathbb{R}^2)","=","2"
        ).scale(2).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_MAGENTA)
        t1[1:].shift(.1*RIGHT)
        b2=underline_bez_curve().next_to(t1,DOWN).scale(2)

        #self.add(t1,b2)

        self.wait()
        self.play(
            Write(t1),
            run_time=2
        )
        self.play(
            Create(b2)
        )
        self.wait(2)

        # manim -pqh test.py dim2_ex


class sq_cloud_ex(Scene):
    def construct(self):

        np=NumberPlane(
            axis_config={
                "stroke_width":DEFAULT_STROKE_WIDTH*1.15,
                "color": REANLEA_YELLOW_CREAM
            },
            background_line_style={
                "stroke_width": DEFAULT_STROKE_WIDTH*.1,
                "stroke_color": REANLEA_GREY_DARKER,
            },
            tips=False,
        ).set_z_index(-5)

        np_1=NumberPlane(
            axis_config={
                "stroke_width":DEFAULT_STROKE_WIDTH*.75,
                "color": REANLEA_YELLOW_CREAM
            },
            background_line_style={
                "stroke_width": 0,
                "stroke_color": REANLEA_GREY_DARKER,
            },
            tips=False
        ).set_z_index(-6)
        np_1.add_coordinates()

        arr_1=Arrow(start=np.c2p(0,0),end=np.c2p(2,1),tip_length=.125,stroke_width=4, buff=0).set_color_by_gradient(REANLEA_CYAN_LIGHT)

        s_fact=np.c2p(0,0)[0]*RIGHT+np.c2p(0,0)[1]*UP

        def sq_cld(
            eps=1,
            **kwargs
        ):  
            n=.75*(1/eps)
            dots_A_1=square_cloud(x_min=-7,x_max=7,x_eps=eps, y_max=0, col=REANLEA_GREEN_AUQA, rad=DEFAULT_DOT_RADIUS/n).shift(s_fact).set_z_index(2)
            dots_B_1=square_cloud(x_max=0,y_min=-4,y_max=4, y_eps=eps, col=REANLEA_BLUE_SKY,rad=DEFAULT_DOT_RADIUS/n).shift(s_fact).set_z_index(2)
            dots_C_1=square_cloud(x_min=-7,x_max=7, x_eps=eps, y_min=-4,y_max=4, y_eps=eps, rad=DEFAULT_DOT_RADIUS/n).shift(s_fact).set_z_index(2)

            dots=VGroup(dots_A_1,dots_B_1,dots_C_1)

            return dots

        
        dots_2=sq_cld(eps=.5)#.set_color_by_gradient(PURE_GREEN,PURE_RED)
        dots_3=sq_cld(eps=.25)#.set_color_by_gradient(PURE_GREEN,PURE_RED)
        dots_4=sq_cld(eps=.125)
        dots_5=sq_cld(eps=.0625)

        self.play(
            Write(np)
        )
        self.play(
            Write(np_1)
        )
        self.wait()

        self.play(
            Write(arr_1)
        )

        self.play(
            Write(dots_2)
        )

        self.play(
            ReplacementTransform(dots_2,dots_3)
        )

        matrix = [[1, .5], [.5, 1.5]]
        self.play(ApplyMatrix(matrix, dots_3), ApplyMatrix(matrix, np),ApplyMatrix(matrix, arr_1))   
        
        self.wait()

        ln_1=DashedLine(start=arr_1.get_end(),end=[arr_1.get_end()[0],0,0])

        self.play(
            Write(ln_1)
        )

        x=arr_1.get_end()[0]

        lbl_1=MathTex("(",x,r",0,0",")").scale(.5).next_to(ln_1.get_end(),UR)
        self.add(lbl_1)

        

        # manim -pqh test.py sq_cloud_ex

        # manim -sqk test.py sq_cloud_ex 


###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2023\lab" 