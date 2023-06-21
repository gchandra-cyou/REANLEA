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

        #self.add(np)
        self.play(
            Write(np)
        )

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
            txt_1 = Text("(on) the space of continuous functions C[-1,1] with 1-norm is (Cauchy but) not Complete." , font=fonts[0]).set_color_by_gradient("#AEAEAE")

        txt_1.scale(.35).shift(3*DOWN)

        q1= MathTex(r"f_{n}(x) := \begin{cases}"
                r"0  &  \text{ if} \ \ \ -1 \leq x < 0 \\"
                r"nx &  \text{ if} \ \ \ 0 \leq x \leq \frac{1}{n} \\"
                r"1 &  \text{ if} \ \ \ \frac{1}{n} < x \leq 1 \\"
                r"\end{cases}"
        ).set_color("#000000").scale(.5).next_to(txt_1,UP).shift(.5*UP)

        #self.add(txt_1,q1)
        self.play(
            Write(q1)
        )
        self.play(
            Write(txt_1)
        )


        self.wait(2)


        # manim -pqh post.py post_2

        # manim -sqk post.py post_2


class post_3(MovingCameraScene):       
    def construct(self): 

        # WATER MARK 

        water_mark=ImageMobject("water_mark_white.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(1).set_z_index(-100)
        self.add(water_mark)

        ax = Axes(
            x_range = [0.01,1.5,0.01/4],
            y_range = [-2,2,1.0],
            x_axis_config = {
                "numbers_to_include": [0,1,2,3],
                "include_ticks" : False
            },
            y_axis_config = {"numbers_to_include": [-2,-1,0,1,2]},
            tips = False
        ).set_color(REANLEA_GREY)

        dt=Dot(radius=.125/4,color=PURE_RED).move_to(ax.c2p(0.01,0)).set_z_index(2)
        
        #Defining graph function
        def func2(x):
            if x == 0:
                return 0
            else:
                return np.sin(1/x)
            

        #Get the graph
        graph2 = ax.plot(func2)
        graph2.set_stroke(width = 1.5, color=REANLEA_WARM_BLUE_DARKER).reverse_direction()
        
        #Show animate
        self.play(Create(ax), run_time=2)
        self.play(Create(graph2),run_time=5)
        self.play(Write(dt))

        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("only discontinuous at 0" , font=fonts[0]).set_color_by_gradient("#AEAEAE").scale(.35).shift(2.5*DOWN)
            txt_2 = Text("as x approaches 0, the frequency of oscillation increases without bound." , font=fonts[0]).set_color_by_gradient("#AEAEAE").next_to(txt_1,DOWN).scale(.35)
            txt_1[4:17].set_color(PURE_RED)


        q1= MathTex(r"f(x) := sin(1/x)"
        ).set_color("#000000").scale(.5).next_to(txt_1,UP)

        self.play(
            Write(q1)
        )
        self.wait()
        self.play(
            Write(txt_1)
        )
        self.play(
            Write(txt_2)
        )
        self.wait()



        self.wait(2)

        # manim -pqh post.py post_3

        # manim -sqk post.py post_3


class post_4(Scene):       
    def construct(self): 

        # WATER MARK 

        water_mark=ImageMobject("water_mark_white.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(1).set_z_index(-100)
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

        dt=Dot(radius=.125/4,color=PURE_RED).move_to(ax.c2p(0.0,0)).set_z_index(5)


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
                r"0 &  \text{ if} \ \ \ x=0 \\"
                r"\end{cases}"
        ).set_color("#000000").scale(.5).next_to(txt_1,UP)

        q2= MathTex(r"\lim\limits_{x \to 0} f'(x)=f'(0)","?"
        ).set_color(REANLEA_INK_BLUE).scale(.5).next_to(txt_2).shift(.05*DOWN)

        q2[1].shift(.1*RIGHT)

        eq_2_grp=VGroup(q2,txt_2).shift(3*UP+LEFT)



        
        
        #Show animate
        self.play(Create(ax), run_time=2)
        self.play(
            Create(dt)
        )
        self.play(Create(graph),run_time=5)
        self.add(q1,txt_1,eq_2_grp)
        self.wait(2)


        # manim -pqh post.py post_4

        # manim -sqk post.py post_4


class post_5(Scene):       
    def construct(self): 

        # WATER MARK 

        water_mark=ImageMobject("water_mark_white.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(1).set_z_index(-100)
        self.add(water_mark)

        # OBJECT ZONE

        ax = Axes(
            x_range = [0.00001,1.00001,0.2],
            y_range = [0,.4,.1],
            x_axis_config = {
                "numbers_to_include": [0.0,.2,0.4,0.6,0.8,1.0],
                #"include_ticks" : False,
                "font_size" : 18
            },
            y_axis_config = {"numbers_to_include": [.1,.2,.3,.4],"font_size" : 18},
            tips = False
        ).set_color("#AEAEAE")

        #Get the graph

        def func(x):
            if x > 0:
                return np.exp(-1/x)
            else:
                0

        graph= ax.plot(
                func,
                color=REANLEA_WARM_BLUE_DARKER,
            )
        
        grph_grp=VGroup(ax,graph).scale(.65).shift(.55*UP)

        dt=Dot(radius=.125/4,color=PURE_RED).move_to(ax.c2p(0,0)).set_z_index(5)

        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("smooth function, which is not analytic." , font=fonts[0]).set_color_by_gradient(GREY).scale(.35).shift(3.35*DOWN)


        q1= MathTex(r"e(x) := \begin{cases}"
                r"e^{-\frac{1}{x}} &  \text{ if} \ \ \  x > 0 \\"
                r"0 &  \text{ if} \ \ \ x \leq 0 \\"
                r"\end{cases}"
        ).set_color("#000000").scale(.5).next_to(txt_1,UP)
        
        
        #Show animate
        self.play(Create(ax), run_time=2)
        self.play(
            Create(dt)
        )
        self.play(Create(graph),run_time=5)
        self.add(q1,txt_1)
        self.wait(2)


        # manim -pqh post.py post_5

        # manim -sqk post.py post_5


class post_6(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("water_mark_white.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(1).set_z_index(-100)
        self.add(water_mark)

        # OBJECT ZONE

        ax=Axes(
            x_range=[0,1.5,1],
            y_range=[0,10,4],
            x_length=5,
            tips=False, 
            x_axis_config={
                "numbers_to_include": [0.0,1],
                "font_size": 18,
            },
            y_axis_config={
                "numbers_to_include": [0.0,4,8],
                "font_size": 18,
            }
        ).set_color("#AEAEAE").scale(.5).set_z_index(-5)

        self.add(ax)

        
        def graph(n=1,stroke_width=5):
            grph = VGroup(
                ax.plot(
                    lambda x: n*n*x,
                    x_range=[0,1/n,0.001]
                ).set_stroke(width=stroke_width/n,color=REANLEA_WARM_BLUE_DARKER),
                ax.plot(
                    lambda x: 2*n -n*n*x,
                    x_range=[1/n,2/n,0.001]
                ).set_stroke(width=stroke_width/n,color=REANLEA_WARM_BLUE_DARKER),
                ax.plot(
                    lambda x: 0,
                    x_range=[2/n,1,0.001]
                ).set_stroke(width=stroke_width/n,color=PURE_RED)
            )
            return grph
        
        x1=ax.plot(
                    lambda x: x,
                    x_range=[0,1,0.001]
                ).set_stroke(width=2.5,color=REANLEA_WARM_BLUE_DARKER)
        
        x2=graph(n=2)

        self.play(
            Create(x1)
        )
        self.play(
            Create(x2)
        )

        x=VGroup(
            *[
                graph(n=i)
                for i in range(3,15)
            ]
        )

        self.play(
            Create(x)
        )

        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("sequence of functions converges pointwise to zero function, but not uniformly." , font=fonts[0]).set_color_by_gradient(GREY)

        txt_1.scale(.35).shift(3*DOWN)

        q1= MathTex(r"f_{n}(x) := \begin{cases}"
                r"n^{2}x  &  \text{ if} \ \ \ 0 \leq x \leq \frac{1}{n} \\"
                r"2n-n^{2}x &  \text{ if} \ \ \ \frac{1}{n} \leq x \leq \frac{2}{n} \\"
                r"0 &  \text{ if} \ \ \ \frac{2}{n} < x \leq 1 \\"
                r"\end{cases}"
        ).set_color(REANLEA_WARM_BLUE_DARKER).scale(.5).shift(1.5*UP+3*RIGHT)
        q1[0][41].set_color(PURE_RED)

        #self.add(txt_1,q1)
        self.play(
            Write(q1)
        )
        self.play(
            Write(txt_1)
        )


        self.wait(2)


        # manim -pqh post.py post_6

        # manim -sqk post.py post_6


class post_7(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("water_mark_white.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(1).set_z_index(-100)
        self.add(water_mark)

        # OBJECT ZONE

        ax=Axes(
            x_range=[-2,2],
            y_range=[-2,2],
            x_length=5,
            tips=False, 
            x_axis_config={
                "numbers_to_include": [0.0,1],
                "font_size": 18,
            },
            y_axis_config={
                "numbers_to_include": [0.0,4,8],
                "font_size": 18,
            }
        ).shift(DOWN)


        
        def graph(n=1,stroke_width=2):
            grph = VGroup(
                ax.plot(
                    lambda x: sqrt((x**2)+1/n),
                    x_range=[-2,2,0.001]
                ).set_stroke(width=stroke_width/n,color=REANLEA_WARM_BLUE_DARKER)
            )
            return grph
        
        grph_1= VGroup(
                ax.plot(
                    lambda x: x,
                    x_range=[0,2,0.001]
                ).set_stroke(width=2,color=[REANLEA_VIOLET_DARKER]),
                ax.plot(
                    lambda x: -x,
                    x_range=[-2,0,0.001]
                ).set_stroke(width=2,color=[REANLEA_VIOLET_DARKER])
            )
        
        
        x1=graph(n=1)
        x2=graph(n=2)

        self.play(
            Create(x1)
        )
        self.play(
            Create(x2)
        )

        x=VGroup(
            *[
                graph(n=i)
                for i in range(3,15)
            ]
        )

        self.play(
            Create(x)
        )
        self.play(
            Create(grph_1)
        )


        lbl_x1=MathTex(r"y^{2}=x^{2}+\frac{1}{n}").scale(.35).set_color(REANLEA_WARM_BLUE_DARKER).next_to(x1,UP).shift(1.85*DOWN)

        lbl_grph1_0=MathTex(r"y=\lvert x \rvert").scale(.35).set_color(REANLEA_VIOLET_DARKER).next_to(grph_1[0],RIGHT).rotate(45*DEGREES).shift(1.5*LEFT)

        lbl_grph1_1=MathTex(r"y=\lvert x \rvert").scale(.35).set_color(REANLEA_VIOLET_DARKER).next_to(grph_1[1],LEFT).rotate(-45*DEGREES).shift(1.5*RIGHT)

        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("The uniform limit of differentiable functions need not be differentiable." , font=fonts[0]).set_color_by_gradient(GREY)

        txt_1.scale(.35).shift(2*DOWN)


        self.add(lbl_x1,lbl_grph1_0,lbl_grph1_1,txt_1)



        self.wait(2)

        # manim -pqh post.py post_7

        # manim -sqk post.py post_7


class post_8(Scene):       
    def construct(self): 

        # WATER MARK 

        water_mark=ImageMobject("water_mark_white.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(1).set_z_index(-100)
        self.add(water_mark)

        ax = Axes(
            x_range = [0.000,0.1,0.005],
            y_range = [-.01,.01,.005],
            x_axis_config = {
                "numbers_to_include": [0.0,.05,0.1],
                #"include_ticks" : False,
                "font_size" : 18,
            },
            y_axis_config = {"numbers_to_include": [-.01,-.005,0,.005,.01],"font_size" : 18},
            tips = False
        ).set_color(REANLEA_GREY)

        dt=Dot(radius=.125/4,color=PURE_RED).move_to(ax.c2p(0.0,0)).set_z_index(5)


        #Get the graph

        graph = VGroup(
            ax.plot(
                lambda x: x**2,
                x_range=[0.00,0.1,0.0001],
                color=GREY,
            ).set_stroke(width=1),
            ax.plot(
                lambda x: np.sin(1/(x**2))*x**2,
                x_range=[0.0001,0.0995,0.0000075],
                color=REANLEA_WARM_BLUE_DARKER,
            ).set_z_index(2),
            ax.plot(
                lambda x: - x**2,
                x_range=[0.00,0.1,0.0001],
                color=GREY,
            ).set_stroke(width=1)
        )

        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("is differentiable with unbounded derivative around zero." , font=fonts[0]).set_color_by_gradient("#AEAEAE").scale(.35).shift(3.35*DOWN)

            txt_2 = Text("Though" , font=fonts[0]).set_color_by_gradient(REANLEA_INK_BLUE).scale(.4)


        q1= MathTex(r"f(x) := \begin{cases}"
                r"x^{2}sin(1/x^{2})  &  \text{ if} \ \ \  x \neq 0 \\"
                r"0 &  \text{ if} \ \ \ x=0 \\"
                r"\end{cases}"
        ).set_color("#000000").scale(.5).next_to(txt_1,UP)

        q2= MathTex(r"f'(0)=0",".").set_color(REANLEA_INK_BLUE).scale(.5).next_to(txt_2)

        q2[1].shift(.1*RIGHT)

        eq_2_grp=VGroup(q2,txt_2).shift(3*UP+LEFT)



        
        
        #Show animate
        self.play(Create(ax), run_time=2)
        self.play(
            Create(dt)
        )
        self.play(Create(graph),run_time=5)
        self.add(q1,txt_1,eq_2_grp)
        self.wait(2)


        # manim -pqh post.py post_8

        # manim -sqk post.py post_8


class post_9(ZoomedScene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("water_mark_white.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(1).set_z_index(-100)
        self.add(water_mark)

        # OBJECT ZONE

        n = 300
        a = ValueTracker(0.5)
        b = ValueTracker(0.6)
        xrng = ValueTracker(4.5)

        ax=Axes(
            x_range=[-4.5,4.5],
            y_range=[-2.5,4.5],
            y_length=(round(config.frame_width)-2)*7/9,
            tips=False, 
            axis_config={
                "font_size": 18,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_GREY).scale(1).set_z_index(-5)
       
        func = VMobject()
        def axUpdater(mobj):
            xmin = -xrng.get_value()
            xmax = +xrng.get_value()
            newax =Axes(
                x_range=[xmin,xmax,10**int(np.log10(xmax)-1)],
                y_range=[-2.5,2.5],
                y_length=(round(config.frame_width)-2)*5/9,
                tips=False, 
                axis_config={
                    "font_size": 18,
                    #"include_ticks": False,
                },
            ).set_color(REANLEA_GREY).scale(1).set_z_index(-5)
            newax.add_coordinates()

            newfunc = newax.plot(
                lambda x: sum([a.get_value()**k*np.cos(b.get_value()**k*PI*x) for k in range(n)]),
                x_range=[xmin,xmax,xrng.get_value()/200],
                use_smoothing=False,
                ).set_color(REANLEA_WARM_BLUE_DARKER).set_stroke(width=3)
            mobj.become(newax)
            func.become(newfunc)            
        ax.add_updater(axUpdater)

        self.add(ax,func)

        

        self.play(
            b.animate.set_value(7),
            run_time=2
        )  

        grp=VGroup(ax,func)
        self.play(
            grp.animate.scale(.65).shift(.5*UP)
        ) 


        self.wait(2)

       # manim -pqh post.py post_9

       # manim -sqk post.py post_9


class substack_banner(Scene):       
    def construct(self): 

        # WATER MARK 

        water_mark=ImageMobject("water_mark_white.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(1).set_z_index(-100)
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

        dt=Dot(radius=.125/4,color=PURE_RED).move_to(ax.c2p(0.0,0)).set_z_index(5)


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
            txt_1 = Text("Geometric foundation of Mathematics and Physics with Animated visuals." , font=fonts[0]).set_color_by_gradient("#AEAEAE").scale(.35).shift(3.35*DOWN)

            txt_2 = Text("Is" , font=fonts[0]).set_color_by_gradient(REANLEA_INK_BLUE).scale(.4)


        q1= MathTex(r"f(x) := \begin{cases}"
                r"x^{2}sin(1/x)  &  \text{ if} \ \ \  x > 0 \\"
                r"0 &  \text{ if} \ \ \ x=0 \\"
                r"\end{cases}"
        ).set_color("#000000").scale(.5).next_to(txt_1,UP)

        q2= MathTex(r"\lim\limits_{x \to 0} f'(x)=f'(0)","?"
        ).set_color(REANLEA_INK_BLUE).scale(.5).next_to(txt_2).shift(.05*DOWN)

        q2[1].shift(.1*RIGHT)

        eq_2_grp=VGroup(q2,txt_2).shift(3*UP+LEFT)



        
        
        #Show animate
        self.play(Create(ax), run_time=2)
        self.play(
            Create(dt)
        )
        self.play(Create(graph),run_time=5)
        self.add(txt_1)
        self.wait(2)


        # manim -pqh post.py substack_banner

        # manim -sqk post.py substack_banner


###################################################################################################################

# Changing FONTS : import any font from Google
# some of my fav fonts: Cinzel,Kalam,Prata,Kaushan Script,Cormorant, Handlee,Monoton, Bad Script, Reenie Beanie, 
# Poiret One,Merienda,Julius Sans One,Merienda One,Cinzel Decorative, Montserrat, Cousine, Courier Prime , The Nautigal
# Marcellus SC,Contrail One,Thasadith,Spectral SC,Dongle,Cormorant SC,Comfortaa, Josefin Sans (LOVE), Fuzzy Bubbles

###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2023\lab" 