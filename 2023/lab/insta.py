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
config.frame_height = 16
config.frame_width = 9
config.pixel_width = 1080
config.pixel_height = 1920
config.frame_rate = 60

###################################################################################################################

class post_1(Scene):       
    def construct(self): 

        # WATER MARK 

        water_mark=ImageMobject("water_mark_white.png").scale(0.075).move_to(3*LEFT+6*UP).set_opacity(1).set_z_index(-100)
        self.add(water_mark)

        ax = Axes(
            x_range = [0.000,0.1,0.005],
            y_range = [-.01,.01,.005],
            x_axis_config = {
                "numbers_to_include": [0.0,.05,0.1],
                #"include_ticks" : False,
                "font_size" : 18
            },
            y_axis_config = {"numbers_to_include": [-.01,-.005,0,.005,.01],"font_size" : 18},
            tips = False
        ).set_color(REANLEA_GREY)

        dt=Dot(radius=.125/2,color=PURE_RED).move_to(ax.c2p(0.0,0)).set_z_index(5)


        #Get the graph

        graph = VGroup(
            ax.plot(
                lambda x: x**2,
                x_range=[0.00,0.1,0.0001],
                color=GREY,
            ).set_stroke(width=1),
            ax.plot(
                lambda x: np.sin(1/x)*x**2,
                x_range=[0.000001,0.1,0.00001],
                color=REANLEA_WARM_BLUE_DARKER,
            ).set_z_index(2),
            ax.plot(
                lambda x: - x**2,
                x_range=[0.00,0.1,0.0001],
                color=GREY,
            ).set_stroke(width=1)
        )


        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("is differentiable everywhere." , font=fonts[0]).set_color_by_gradient("#AEAEAE").scale(.35).shift(3.35*DOWN)

            txt_2 = Text("Is" , font=fonts[0]).set_color_by_gradient(REANLEA_INK_BLUE).scale(.4)


        q1= MathTex(r"f(x) := \begin{cases}"
                r"x^{2}sin(1/x)  &  \text{ if} \ \ \  x > 0 \\"
                r"0 &  \text{ if} \ \ \ x \leq 0 \\"
                r"\end{cases}"
        ).set_color("#000000").scale(.65).move_to(3.75*UP)
        q1[0][22].set_color(PURE_RED)

        q2= MathTex(r"\lim\limits_{x \to 0} f'(x)=f'(0)","?"
        ).set_color(REANLEA_INK_BLUE).scale(.5).next_to(txt_2).shift(.05*DOWN)

        q2[1].shift(.1*RIGHT)

        eq_2_grp=VGroup(q2,txt_2).shift(3*UP+LEFT)


        q3= MathTex(r"f'(0) =", r"\lim_{h\to 0} \frac{f(0+h)-f(0)}{h}"
        ).set_color("#000000").scale(.65).move_to(3.75*DOWN)
        q3_1= MathTex(r"\lim_{h\to 0} \frac{h^{2}sin(1/h)-0}{h}"
        ).set_color("#000000").scale(.65).next_to(q3[0]).shift(.125*LEFT+.05*UP)
        q3_2= MathTex(r"\lim_{h\to 0} hsin(1/h)"
        ).set_color("#000000").scale(.65).next_to(q3[0]).shift(.125*LEFT+.0525*DOWN)
        q3_2[0][6:].shift(.1*RIGHT)
        q3_2[0][7:].shift(.1*RIGHT)


        grph_grp=VGroup(ax,graph,dt).scale(.65)



        
        
        #Show animate
        self.play(Create(ax), run_time=2)
        self.play(
            Create(dt)
        )
        self.play(
            AnimationGroup(
                AnimationGroup(
                    Create(graph[0]),
                    Create(graph[1]),
                    Create(graph[2])
                ),
                run_time=5
            ),
            Write(q1)    
        )
        self.wait(2)

        self.play(
            Write(q3)
        )
        self.play(
            TransformMatchingShapes(q3[1],q3_1)
        )
        
        self.play(
            TransformMatchingShapes(q3_1,q3_2)
        )
        x_grp=VGroup(q3[0],q3_2)

        self.wait(2)

        self.play(
            AnimationGroup(
                x_grp.animate.move_to(4.5*UP),
                FadeOut(VGroup(ax,dt,graph,q1)),
                lag_ratio=.25
            )
        )
        self.wait(2)

        ax_1 = Axes(
            x_range = [0.0,0.2,0.05],
            y_range = [-.05,.05,.05],
            x_axis_config = {
                "numbers_to_include": [0.0,.05,0.1,0.15,0.2],
                #"include_ticks" : False,
                "font_size" : 18
            },
            y_axis_config = {"numbers_to_include": [-.1,-.05,0,.05,.1],"font_size" : 18},
            tips = False
        ).set_color(REANLEA_GREY).scale(.65)

        graph_1 = VGroup(
            ax_1.plot(
                lambda x: x,
                x_range=[0.00,0.2,0.0001],
                color=GREY,
            ).set_stroke(width=1),
            ax_1.plot(
                lambda x: np.sin(1/x)*x,
                x_range=[0.000001,0.2,0.00001],
                color=REANLEA_WARM_BLUE_DARKER,
            ).set_z_index(2),
            ax_1.plot(
                lambda x: - x,
                x_range=[0.00,0.2,0.0001],
                color=GREY,
            ).set_stroke(width=1)
        )

        dt_1=Dot(radius=.125/2,color=PURE_RED).move_to(ax_1.c2p(0.0,0)).set_z_index(5)

        self.play(
            Create(ax_1)
        )
        self.play(
            AnimationGroup(
                Create(graph_1[1].reverse_direction()),
                x_grp.animate.shift(2*LEFT),
            ),
            run_time=5
        )
        self.play(Create(dt_1))
        self.play(
            AnimationGroup(
                Create(graph_1[0]),
                Create(graph_1[2])
            )
        )

        graph_1_0_lbl=MathTex("y=h").scale(.75).set_color(GREY).rotate(PI/4).shift(2.25*LEFT+2.5*UP)

        graph_1_3_lbl=MathTex("y=-h").scale(.75).set_color(GREY).rotate(-PI/4).shift(2.25*LEFT+2.5*DOWN)

        q4=MathTex("=0").set_color("#000000").scale(.65).next_to(x_grp,DOWN).shift(.6*LEFT)


        self.play(
            AnimationGroup(
                Write(graph_1_0_lbl),
                Write(graph_1_3_lbl),
            )
        )
        self.wait()

        self.play(
            Write(q4)
        )
        self.wait()

        self.play(
            AnimationGroup(
                FadeOut(VGroup(ax_1,dt_1,graph_1,graph_1_0_lbl,graph_1_3_lbl)),
                q4.animate.next_to(x_grp,RIGHT).shift(.06*UP),
                q1.animate.move_to(ORIGIN)
            )
        )
        self.wait()

        q5= MathTex(r"f(x) := \begin{cases}"
                r"2xsin(1/x)-cos(1/x)  &  \text{ if} \ \ \  x > 0 \\"
                r"0 &  \text{ if} \ \ \ x \leq 0 \\"
                r"\end{cases}"
        ).set_color("#000000").scale(.65).move_to(3.5*DOWN)
        q5[0][31].set_color(PURE_RED)
        q5[0][9:].shift(.075*RIGHT)

        ax_2 = Axes(
            x_range = [0.0,0.1,0.005],
            y_range = [-1.1,1.1,.005],
            x_axis_config = {
                "numbers_to_include": [0.0,.05,0.1],
                #"include_ticks" : False,
                "font_size" : 18
            },
            y_axis_config = {"numbers_to_include": [-1,-.5,0,.5,1],"font_size" : 18},
            tips = False
        ).set_color(REANLEA_GREY)



        #Get the graph

        graph_2 = VGroup(
            ax_2.plot(
                lambda x: np.sin(1/x)*x*2 - np.cos(1/x),
                x_range=[0.001,0.1,0.00001],
                color=REANLEA_WARM_BLUE_DARKER,
            ).set_z_index(2)
        )

        ax_3 = Axes(
            x_range = [0.0,0.2,0.05],
            y_range = [-1.1,1.1,.5],
            x_axis_config = {
                "numbers_to_include": [0.0,.05,0.1,0.15,0.2],
                #"include_ticks" : False,
                "font_size" : 18
            },
            y_axis_config = {"numbers_to_include": [-1,-.5,0,.5,1],"font_size" : 18},
            tips = False
        ).set_color(REANLEA_GREY).scale(.65)

        dt_2=Dot(radius=.125/2,color=PURE_RED).move_to(ax_3.c2p(0.0,0)).set_z_index(15)

        graph_grp_2=VGroup(ax_2,graph_2,ax).scale(.65)


        self.play(
            AnimationGroup(
                AnimationGroup(
                    Write(ax_3),
                    Create(graph_2[0].reverse_direction()),
                    run_time=5
                ),
                Create(dt_2),
                lag_ratio=.9          
            ),
            ReplacementTransform(q1,q5)     
        )

        


        # manim -pqh insta.py post_1

        # manim -sqk insta.py post_1


###################################################################################################################

# Playfair Display SC , Great Vibes , Merienda , Tangerine , Shalimar , Parisienne , Allura , Playball , Bad Script
# Cormorant SC, Montserrat, MonteCarlo , Kolker Brush

###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2023\lab" 