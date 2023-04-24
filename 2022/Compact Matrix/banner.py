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



config.background_color= WHITE
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
        rendered_code = Code(code=code,
                            language="Python", font="Monospace").scale(.5)
        self.add(rendered_code)


        # manim -sqk banner.py CodeFromString


class Rotation3DExample(ThreeDScene):
    def construct(self):

        bg = ImageMobject("tran.png").scale(.15).set_z_index(-100)
        #self.add(bg)

        cube = Cube(side_length=3, fill_opacity=1).set_z_index(1).set_color_by_gradient(REANLEA_BLUE_LAVENDER).rotate(65*DEGREES, X_AXIS).rotate(59*DEGREES, Y_AXIS)


        self.add(cube)



        # manim -sqk banner.py Rotation3DExample

class ax_ex(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("water_mark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.35).set_z_index(-100)
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
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5)

        bg = ImageMobject("cube.png").scale(.3).shift(.25*UP).set_z_index(5)
        self.add(bg)

        self.add(ax_1)
    
    # manim -sqk banner.py ax_ex
        
      
###################################################################################################################

# Changing FONTS : import any font from Google
# some of my fav fonts: Cinzel,Kalam,Prata,Kaushan Script,Cormorant, Handlee,Monoton, Bad Script, Reenie Beanie, 
# Poiret One,Merienda,Julius Sans One,Merienda One,Cinzel Decorative, Montserrat, Cousine, Courier Prime , The Nautigal
# Marcellus SC,Contrail One,Thasadith,Spectral SC,Dongle,Cormorant SC,Comfortaa, Josefin Sans (LOVE), Fuzzy Bubbles

###################################################################################################################


###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix" 