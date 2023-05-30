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


class CoordSysExample(Scene):
            def construct(self):

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


                # manim -pqh banner.py CoordSysExample

                # manim -sqk banner.py CoordSysExample



class CodeFromString(Scene):
    def construct(self):
        code = '''from reanlea import Scene, Square

class CoordSysExample(Scene):
            def construct(self):

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
'''
        rendered_code = Code(code=code, style=Code.styles_list[5],
            language="Python", font="Monospace").scale(.5)
        self.add(rendered_code)


        # manim -sqk banner.py CodeFromString


class Rotation3DExample(ThreeDScene):
    def construct(self):

        bg = ImageMobject("tran.png").scale(.15).set_z_index(-100)
        #self.add(bg)

        cube = Cube(side_length=3, fill_opacity=1).set_z_index(1).set_color_by_gradient(REANLEA_VIOLET).rotate(65*DEGREES, X_AXIS).rotate(60*DEGREES, Y_AXIS)


        self.add(cube)



        # manim -sqk banner.py Rotation3DExample

class dim(Scene):
     def construct(self):
        dim_r2=MathTex(
            "dim",r"(\mathbb{R}^2)","=","2"
        ).set_z_index(16).scale(2).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_MAGENTA)
        dim_r2[1:].shift(.1*RIGHT)
        self.play(
            Write(dim_r2),
            lag_ratio=.7
        )

        b2=underline_bez_curve().next_to(dim_r2,DOWN).scale(2).set_z_index(16)
        self.play(
            Create(b2)
        )

        self.wait(2)

        # manim -sqk banner.py dim
        

class ax_ex(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("water_mark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(1).set_z_index(-4)
        self.add(water_mark)

        ax_1=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(-3)

        cube = ImageMobject("cube.png").scale(.3).shift(.25*UP).set_z_index(5)
        self.add(cube)

        eu_sp = ImageMobject("eu_sp_1.png").scale(.5).shift(1.75*UP).set_z_index(-1)
        self.add(eu_sp)

        sx_1 = ImageMobject("sx_1.png").scale(.25).shift(2*DOWN+5*RIGHT).rotate(-30*DEGREES).set_opacity(.25).set_z_index(-2)
        self.add(sx_1)

        sx_2 = ImageMobject("sx_2.png").scale(.25).shift(4*LEFT).set_opacity(.25).set_z_index(-2)
        self.add(sx_2)

        sx_3 = ImageMobject("sx_3.png").scale(.25).shift(2.5*DOWN).set_opacity(.25).set_z_index(-2)
        self.add(sx_3)

        sx_4 = ImageMobject("sx_4.png").scale(.25).shift(4*RIGHT+1.5*UP).set_opacity(.25).set_z_index(-2)
        self.add(sx_4)

        sx_5 = ImageMobject("sx_5.png").scale(.25).shift(4*LEFT+2.5*DOWN).set_opacity(.25).set_z_index(-2)
        self.add(sx_5)

        bg_1 = ImageMobject("code.png").scale(.25).shift(4*LEFT+2.5*DOWN).set_opacity(.15).set_z_index(-5)
        self.add(bg_1)

        stripe1=get_stripe(factor=.05, buff_max=3,color=REANLEA_GOLD).shift(5.75*LEFT+2*UP)

        stripe2=get_stripe(factor=.05, buff_max=3,color=REANLEA_WELDON_BLUE).shift(2.75*LEFT+UP).rotate(PI)

        with RegisterFont("Montserrat") as fonts:
            txt_0 = Text("EUCLIDEAN SPACE" , font=fonts[0], color=GREEN)

        txt_0.next_to(stripe1,DOWN)
        

        self.add(ax_1)
    
    # manim -sqk banner.py ax_ex


class banner1(Scene):
     def construct(self):
          
        im_1 = ImageMobject("title_1.png").scale(.5)

        self.play(
            FadeIn(im_1)
        )
          
    # manim -sqk banner.py banner1

    # manim -pqk banner.py banner1

###################################################################################################################

# Changing FONTS : import any font from Google
# some of my fav fonts: Cinzel,Kalam,Prata,Kaushan Script,Cormorant, Handlee,Monoton, Bad Script, Reenie Beanie, 
# Poiret One,Merienda,Julius Sans One,Merienda One,Cinzel Decorative, Montserrat, Cousine, Courier Prime , The Nautigal
# Marcellus SC,Contrail One,Thasadith,Spectral SC,Dongle,Cormorant SC,Comfortaa, Josefin Sans (LOVE), Fuzzy Bubbles

###################################################################################################################


###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix" 