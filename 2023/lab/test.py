from __future__ import annotations

import sys
sys.path.insert(1,'C:\\Users\\gchan\\Desktop\\REANLEA\\2023\\common')
sys.path.insert(1,'C:\\Users\\gchan\\Desktop\\REANLEA\\2023')

# from "common" we're importing "reanlea_colors" & "func" here.
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
import random

import matplotlib.colors as mc
import colorsys




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

        matrix = [[1/sqrt(2), 1/sqrt(2)], [-1/sqrt(2), 1/sqrt(2)]]
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


class defFun(Scene):
    def construct(self):
        def f(x):
            if x < 0.5:
                return 0
            else:
                return 2*(x-0.5)

        ax = Axes(
            x_range = [0, 1, 1],
            y_range = [0, 1, 1],
            tips=False
        )

        plt = ax.plot(f, discontinuities = [0.5]).set_stroke(width=15, color=[PURE_GREEN,REANLEA_WARM_BLUE])


        self.play(
            Create(plt)
        )

        self.wait(3)


        # manim -pqh test.py defFun


class constrct_fnx(Scene):
    def construct(self):

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

        self.add(ax)

        graph = VGroup(
            ax.plot(
                lambda x: 0,
                x_range=[-1,0,0.01]
            ).set_stroke(width=1,color=REANLEA_GREY),
            ax.plot(
                lambda x: 2*x,
                x_range=[0,.5,0.01]
            ).set_stroke(width=1,color=BLUE),
            ax.plot(
                lambda x: 1,
                x_range=[.5,1,0.01]
            ).set_stroke(width=1,color=GREEN)
        )

        self.play(
            Create(graph)
        )

        self.wait(2)


        # manim -pqh test.py constrct_fnx

        # manim -sqk test.py constrct_fnx



class constrct_fnxz(Scene):
    def construct(self):

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

        
        def graph(n=1,stroke_width=3):
            grph = VGroup(
                ax.plot(
                    lambda x: 0,
                    x_range=[-1,1,0.01]
                ).set_z_index(-1).set_stroke(width=2.5,color=REANLEA_GREY),
                ax.plot(
                    lambda x: 0,
                    x_range=[-1,0,0.01]
                ).set_stroke(width=stroke_width/n, color=REANLEA_WARM_BLUE),
                ax.plot(
                    lambda x: n*x,
                    x_range=[0,1/n,0.01]
                ).set_stroke(width=stroke_width/n,color=REANLEA_WARM_BLUE),
                ax.plot(
                    lambda x: 1,
                    x_range=[1/n,1,0.01]
                ).set_stroke(width=stroke_width/n,color=REANLEA_WARM_BLUE)
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
                for i in range(3,50)
            ]
        )
        self.play(
            Create(x)
        )

        self.wait(2)


        # manim -pqh test.py constrct_fnxz

        # manim -sqk test.py constrct_fnxz
        

class constrct_xsqsin1byx(Scene):
    def construct(self):

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

        graph_1=ax.plot(
            lambda x : np.sin(1/x)*x**2,
            discontinuities=[0],
        ).scale(5)

        self.add(graph_1)
        
        self.wait(2)

        # manim -sqk test.py constrct_xsqsin1byx


class Grafik(MovingCameraScene):       
    def construct(self): 

        ax = Axes(
            x_range = [0.001,3,0.01],
            y_range = [-2,2,1.0],
            x_axis_config = {
                "numbers_to_include": [0,1,2,3],
                "include_ticks" : False
            },
            y_axis_config = {"numbers_to_include": [-2,-1,0,1,2]},
            tips = False
        )
        
        #Defining graph function
        def func2(x):
            if x == 0:
                return 0
            else:
                return np.sin(1/x)
            

        #Get the graph
        graph2 = ax.plot(func2)
        graph2.set_stroke(width = 1.5)
        
        
        #Set up its label
        axes_labels2 = ax.get_axis_labels()
        
        #Show animate
        self.play(Create(ax), run_time=2)
        self.play(Create(graph2),run_time=5)
        self.wait(2)

        # manim -pqh test.py Grafik


class Grafik_1(MovingCameraScene):       
    def construct(self): 

        ax = Axes(
            x_range = [0.0001,1.5,0.01],
            y_range = [-2,2,1.0],
            x_axis_config = {
                "numbers_to_include": [0,1,2,3],
                "include_ticks" : False
            },
            y_axis_config = {"numbers_to_include": [-2,-1,0,1,2]},
            tips = False
        ).set_color(GREY)

        dt=Dot(radius=.125/4,color=PURE_RED).move_to(ax.c2p(0,0)).set_z_index(2)
        
        #Defining graph function
        def func2(x):
            if x == 0:
                return 0
            else:
                return np.sin(1/x)
            

        #Get the graph
        graph2 = VGroup(
            ax.plot(func2, x_range=[0.0005, 0.1, 0.00001]),
            ax.plot(func2, x_range=[0.1, 1.5, 0.01])
        )

        graph2.set_stroke(width = 1.5, color=BLUE)
        
        #Show animate
        self.play(Create(ax), run_time=2)
        self.play(Write(dt))
        self.play(Create(graph2),run_time=5)
        self.wait(2)


        # manim -pqh test.py Grafik_1


class Grafik_2(MovingCameraScene):       
    def construct(self): 

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
        ).set_color(GREY)


        #Get the graph

        graph = VGroup(
            ax.plot(
                lambda x: x**2,
                x_range=[0.00,0.1,0.0001],
                color=RED,
            ),
            ax.plot(
                lambda x: np.sin(1/x)*x**2,
                x_range=[0.000001,0.1,0.00001],
                color=BLUE,
            ).set_z_index(2),
            ax.plot(
                lambda x: - x**2,
                x_range=[0.00,0.1,0.0001],
                color=RED,
            )
        )

        
        #Show animate
        self.play(Create(ax), run_time=2)
        self.play(Create(graph),run_time=5)
        self.wait(2)


        # manim -pqh test.py Grafik_2

        # manim -sqk test.py Grafik_2


class Grafik_3(MovingCameraScene):       
    def construct(self): 

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
        ).set_color(GREY)


        #Get the graph

        def func(x):
            if x > 0:
                return np.exp(-1/x)
            else:
                0

        graph= ax.plot(
                func,
                color=RED,
            )
        
        #Show animate
        self.play(Create(ax), run_time=2)
        self.play(Create(graph),run_time=5)
        self.wait(2)


        # manim -pqh test.py Grafik_3

        # manim -sqk test.py Grafik_3


class post_6_tst(Scene):
    def construct(self):

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
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(-5)

        self.add(ax)

        
        def graph(n=1,stroke_width=5):
            grph = VGroup(
                ax.plot(
                    lambda x: n*n*x,
                    x_range=[0,1/n,0.001]
                ).set_stroke(width=stroke_width/n,color=REANLEA_WARM_BLUE),
                ax.plot(
                    lambda x: 2*n -n*n*x,
                    x_range=[1/n,2/n,0.001]
                ).set_stroke(width=stroke_width/n,color=REANLEA_WARM_BLUE),
                ax.plot(
                    lambda x: 0,
                    x_range=[2/n,1,0.001]
                ).set_stroke(width=stroke_width/n,color=REANLEA_WARM_BLUE)
            )
            return grph
        
        x2=graph(n=2)

        
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
            txt_1 = Text("(on) the space of continuous functions C[-1,1] with 1-norm is (Cauchy but) not Complete." , font=fonts[0]).set_color_by_gradient("#AEAEAE")

        txt_1.scale(.35).shift(3*DOWN)

        q1= MathTex(r"f_{n}(x) := \begin{cases}"
                r"0  &  \text{ if} \ \ \ -1 \leq x < 0 \\"
                r"nx &  \text{ if} \ \ \ 0 \leq x \leq \frac{1}{n} \\"
                r"1 &  \text{ if} \ \ \ \frac{1}{n} < x \leq 1 \\"
                r"\end{cases}"
        ).set_color("#000000").scale(.5).next_to(txt_1,UP).shift(.5*UP)

        #self.add(txt_1,q1)
        '''self.play(
            Write(q1)
        )
        self.play(
            Write(txt_1)
        )'''


        self.wait(2)


        # manim -pqh test.py post_6_tst

        # manim -sqk test.py post_6_tst

class post_7_tst(Scene):
    def construct(self):

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
        )


        
        def graph(n=1,stroke_width=2):
            grph = VGroup(
                ax.plot(
                    lambda x: sqrt((x**2)+1/n),
                    x_range=[-2,2,0.001]
                ).set_stroke(width=stroke_width,color=REANLEA_WARM_BLUE)
            )
            return grph
        
        grph_1= VGroup(
                ax.plot(
                    lambda x: x,
                    x_range=[0,2,0.001]
                ).set_stroke(width=2,color=PURE_RED),
                ax.plot(
                    lambda x: -x,
                    x_range=[-2,0,0.001]
                ).set_stroke(width=2,color=PURE_RED)
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

        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("sequence of functions converges pointwise to zero function, but not uniformly." , font=fonts[0]).set_color_by_gradient(GREY)

        txt_1.scale(.35).shift(3*DOWN)

        q1= MathTex(r"f_{n}(x) := \begin{cases}"
                r"n^{2}x  &  \text{ if} \ \ \ 0 \leq x \leq \frac{1}{n} \\"
                r"2n-n^{2}x &  \text{ if} \ \ \ \frac{1}{n} \leq x \leq \frac{2}{n} \\"
                r"0 &  \text{ if} \ \ \ \frac{2}{n} < x \leq 1 \\"
                r"\end{cases}"
        ).set_color(REANLEA_WARM_BLUE_DARKER).scale(.5).shift(1.5*UP+3*RIGHT)

        
        '''self.play(
            Write(q1)
        )
        self.play(
            Write(txt_1)
        )'''


        self.wait(2)


        # manim -sqk test.py post_7_tst


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

        '''Ref : https://www.whitman.edu/documents/Academics/Mathematics/2019/Vesneske-Gordon.pdf'''


        # manim -pqh test.py weier

        # manim -sqk test.py weier


class post_9_tst(ZoomedScene):
    def construct(self):

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
        ).set_color(REANLEA_TXT_COL_DARKER).scale(1).set_z_index(-5)
       
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
            ).set_color(REANLEA_TXT_COL_DARKER).scale(1).set_z_index(-5)
            newax.add_coordinates()

            newfunc = newax.plot(
                lambda x: sum([a.get_value()**k*np.cos(b.get_value()**k*PI*x) for k in range(n)]),
                x_range=[xmin,xmax,xrng.get_value()/200],
                use_smoothing=False,
                ).set_color(PURE_RED).set_stroke(width=3)
            mobj.become(newax)
            func.become(newfunc)            
        ax.add_updater(axUpdater)

        self.add(ax,func)

        self.play(
            b.animate.set_value(7),
            run_time=2
        )   


        self.wait(2)

       # manim -pqh test.py post_9_tst

       # manim -sqk test.py post_9_tst


class MovingZoomedSceneAround(ZoomedScene):
    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self,
            zoom_factor=0.3,
            zoomed_display_height=1,
            zoomed_display_width=6,
            image_frame_stroke_width=20,
            zoomed_camera_config={
                "default_frame_stroke_width": 3,
                "background_opacity": 1,
                },
            **kwargs
        )

    def construct(self):
        dot = Dot().shift(LEFT * 3 + UP)
        image=ImageMobject("ganesh.png")
        image.height=7
        

        self.add(image,dot)
        zoomed_camera = self.zoomed_camera
        zoomed_display = self.zoomed_display
        frame = zoomed_camera.frame
        zoomed_display_frame = zoomed_display.display_frame

        frame.move_to(dot)
        frame.set_color(PURPLE)
        zoomed_display_frame.set_color(RED)
        zoomed_display.shift(DOWN)

        zd_rect = BackgroundRectangle(zoomed_display, color=WHITE, fill_opacity=0, buff=MED_SMALL_BUFF)
        self.add_foreground_mobject(zd_rect)

        unfold_camera = UpdateFromFunc(zd_rect, lambda rect: rect.replace(zoomed_display))


        self.play(Create(frame))
        self.activate_zooming()

        self.play(self.get_zoomed_display_pop_out_animation(), unfold_camera)
        # Scale in        x   y  z
        scale_factor = [0.5, 1.5, 0]
        self.play(
            frame.animate.scale(scale_factor),
            zoomed_display.animate.scale(scale_factor)
        )
        self.wait()
        self.play(ScaleInPlace(zoomed_display, 2))
        self.wait()
        self.play(frame.animate.shift(2.5 * DOWN))
        self.wait()
        self.play(self.get_zoomed_display_pop_out_animation(), unfold_camera, rate_func=lambda t: smooth(1 - t))
        self.play(Uncreate(zoomed_display_frame), FadeOut(frame))
        self.wait()



        # manim -pqh test.py MovingZoomedSceneAround

class UseZoomedScene(ZoomedScene):
        def construct(self):
            dot = Dot().set_color(GREEN)

            self.add(dot)
            self.wait(1)
            self.activate_zooming(animate=False)
            self.wait(1)
            


            # manim -pqh test.py UseZoomedScene

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


        # manim -pqh test.py post_4

        # manim -sqk test.py post_4


class pitch_deck(Scene):
    def construct(self):

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

        #self.add(np)

        ax=Axes(
            x_range=[-.5,4.5],
            y_range=[-.5,4.5],
            y_length=(round(config.frame_width)-2),
            tips=False, 
            axis_config={
                "font_size": 18,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.65).set_z_index(-5).shift(LEFT)

        ln=Line(start=ax.c2p(0,0),end=ax.c2p(4,4)).set_stroke(width=1.5).set_color(REANLEA_TXT_COL_DARKER)

        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("serious learners." , font=fonts[0]).set_color_by_gradient("#AEAEAE").scale(.35).next_to(ax.c2p(4.5,0))
            num_1=Text("3%" , font=fonts[0]).set_color_by_gradient("#AEAEAE").scale(.35).next_to(ax.c2p(4,0),DOWN)

            txt_2 = Text("youtube viewers" , font=fonts[0]).set_color_by_gradient("#AEAEAE").scale(.35).next_to(ax.c2p(0,4.5),UP)
            num_2=Text("0.1%" , font=fonts[0]).set_color_by_gradient("#AEAEAE").scale(.35).next_to(ax.c2p(0,4),LEFT)

        self.add(ax,ln,txt_1,txt_2,num_1,num_2)


        # manim -pqh test.py pitch_deck

        # manim -sqk test.py pitch_deck

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


class delx(Scene):
    def construct(self):
        
        dt_y=PointCloudDot(color=REANLEA_BACKGROUND_COLOR_OXFORD_BLUE)

        self.add(dt_y)

        # manim -sqk test.py delx


class Sierpinski(Scene):

   config.disable_caching_warning=True
  
   def subdivide(self, square, n):

   # Subdivide a square of size n into 4 equal parts of size n/2.
   # Return the three subsquares in the upper left, lower left, and lower right corners as a VGroup.

        ULsq = Square(
            side_length=n/2, 
            color=BLACK, 
            fill_color=YELLOW, 
            fill_opacity=1,
            stroke_width=0.25
        ).align_to(square,LEFT+UP)
        LLsq = ULsq.copy().shift(DOWN*n/2)
        LRsq = LLsq.copy().shift(RIGHT*n/2)
        sqs = VGroup(ULsq,LLsq,LRsq)
        return sqs
 
    
   def construct(self):

        size = 6  # size of initial square
        orig_size = size
        iterations = 6  # numeber of iterations in construction

        title = Text("Sierpinski Triangle").to_edge(UP)
        self.play(Create(title))
        self.wait()
        S = Square(
            side_length=size, 
            color=BLACK, 
            fill_color=YELLOW, 
            fill_opacity=1,
            stroke_width=0.5
            ).to_edge(LEFT,buff=.75).shift(DOWN*0.3)
        text1 = Text("Start with a square", font_size=24).move_to([2,2,0])
        text2 = Text("Divide into 4 equal subsquares", font_size=24).align_to(text1,LEFT).shift(UP)
        text3 = Text("Remove the upper right square",font_size=24).align_to(text1,LEFT)
        text4 = Text("Repeat with each remaining subsquare", font_size=24).align_to(text1,LEFT).shift(DOWN*3) 
        textct = Text("Iteration 1",color=YELLOW).align_to(text1,LEFT).shift(DOWN*2)   

        # First iteration with instructions                   
        self.add(textct)
        self.wait(1)       
        self.add(text1)
        self.play(FadeIn(S))
        self.wait(1)
        self.add(text2)
        self.wait(0.2)
        vertLine = Line(S.get_left(), S.get_right(), color=BLACK,stroke_width=1)
        horizLine = Line(S.get_top(), S.get_bottom(), color=BLACK,stroke_width=1)
        self.play(Create(vertLine), Create(horizLine), run_time=2)                 
        B=[0]
        B[0] = self.subdivide(S,size)
        self.wait(1)
        self.add(text3)
        self.wait(0.5)         
        self.add(*B[0])
        self.play(FadeOut(S), run_time=1.5)
        self.wait(1)
 

        # temporarily split off the three subsquares to illustrate construction on each subsquare
        # and draw lines to split each into 4 additional subsquares
        self.remove(textct)
        textct = Text("Iteration "+str(2), color=YELLOW).align_to(text1,LEFT).shift(DOWN*2)
        self.add(textct)
        self.wait(1)
        self.add(text4)   
        self.wait(1) 
        self.play(B[0][0].animate.shift(UP*0.5),B[0][2].animate.shift(RIGHT*0.5))
        self.wait(1)
        for k in range(3):
           vertLine = Line(B[0][k].get_left(), B[0][k].get_right(), color=BLACK,stroke_width=1)
           horizLine = Line(B[0][k].get_top(), B[0][k].get_bottom(), color=BLACK,stroke_width=1)
           self.play(Create(vertLine),Create(horizLine), run_time=0.33)
           self.wait(0.5)

        # Remaining iterations
        for m in range(0,iterations-1):
           size=size/2
           C = [0]*(3**(m+1))
           if (m > 0): self.wait(1.5)
           for k in range(3**m):
              C[3*k]=self.subdivide(B[k][0],size)
              C[3*k+1]=self.subdivide(B[k][1],size)
              C[3*k+2]=self.subdivide(B[k][2],size)
              self.add(*C[3*k],*C[3*k+1],*C[3*k+2])             
              self.remove(*B[k]) 
           if (m == 0): # recombine the squares of iteration 1 back into place
              self.wait(1)
              self.play(C[0].animate.shift(DOWN*0.5),C[2].animate.shift(LEFT*0.5))
           self.remove(textct)
           textct = Text("Iteration "+str(m+2), color=YELLOW).align_to(text1,LEFT).shift(DOWN*2)
           self.add(textct)            
           if (m < iterations-2): B = C.copy()
        self.wait(2)

        # Demonstrate self-similarity
        self.remove(text1,text2,text3,text4,textct)
        self.wait(1)

        VGTL = VGroup()  # top left corner
        VGBL = VGroup()  # bottom left corner
        VGBR = VGroup()  # bottom right corner
        m = 3**(iterations-2)
        for k in range(m):
           VGTL += C[k] 
        for k in range(m,2*m):
           VGBL += C[k]
        for k in range(2*m,3*m):
           VGBR += C[k]

        # Method 1 - show each corner is self-similar

        # set colors
        text4 = Text("Three self-similar pieces", color=YELLOW).move_to([2,2,0])
        self.add(text4)

        self.play(VGTL.animate.set_color(PURE_RED)) 
        self.play(VGBL.animate.set_color(PURE_GREEN)) 
        self.play(VGBR.animate.set_color(ORANGE)) 
        VGTL.save_state()  # save corners in current colors for method 2
        VGBL.save_state()
        VGBR.save_state()
        # shift the three corners apart to illustrate self-similarity
        self.play(VGTL.animate.shift(UP*0.5),VGBR.animate.shift(RIGHT*0.5))
        self.wait(1.5)
        # shift back and restore colors
        self.play(VGTL.animate.shift(DOWN*0.5),VGBR.animate.shift(LEFT*0.5))
        self.play(
            VGTL.animate.set_color(YELLOW).set_fillcapacity(1), 
            VGBL.animate.set_color(YELLOW).set_fillcapacity(1), 
            VGBR.animate.set_color(YELLOW).set_fillcapacity(1)
            )    
        self.wait(1.5)  

        # Method 2 - iterated function system

        # Combine all corners into one mobject and make a copy
        VGall = VGroup(*VGTL,*VGBL,*VGBR)
        VGallcp = VGall.copy().set_color(GRAY)       
        VGall.save_state()
        
        self.remove(text4)
        text5 = Text("Iterated Function System", color=YELLOW).move_to([2,2,0])
        self.add(text5)
        self.wait(1)

        # first scaling and translation to upper left corner
        text5 = MarkupText(f'<span fgcolor="{PURE_RED}" weight="{BOLD}">1.</span> Scale by 1/2, translate up', 
                           color=YELLOW, font_size=30
                           ).move_to([2.75,1,0])
        self.add(text5)
        self.add(VGallcp,VGall)
        self.play(Transform(VGall,VGBL),run_time=2)  
        self.play(VGall.animate.set_color(PURE_RED))
        self.play(VGall.animate.shift(UP*orig_size/2))
        self.wait(2)

        # second scaling to lower left corner (no translation)
        text6 = MarkupText(f'<span fgcolor="{PURE_GREEN}" weight="{BOLD}">2.</span> Scale by 1/2', 
                           color=YELLOW, font_size=30
                           ).align_to(text5, LEFT)        
        VGall.restore()      
        self.add(text6)
        self.play(Transform(VGall,VGBL),run_time=2)
        self.play(VGall.animate.set_color(PURE_GREEN))        
        self.wait(2)

        # third scaling and translation to lower right corner
        text7 = MarkupText(f'<span fgcolor="{ORANGE}" weight="{BOLD}">3.</span> Scale by 1/2, translate right', 
                           color=YELLOW, font_size=30
                           ).align_to(text5, LEFT).shift(DOWN)       
        VGall.restore()
        self.add(text7)
        self.play(Transform(VGall,VGBL),run_time=2)
        self.play(VGall.animate.set_color(ORANGE))
        self.play(VGall.animate.shift(RIGHT*orig_size/2))
        self.wait(2)

        # restore the three (self-similar) corners to their individual colors
        VGTL.restore()
        VGBL.restore()
        VGBR.restore()
        self.wait(4)


        # manim -sqk test.py Sierpinski

        # manim -pqh test.py Sierpinski


class Sierpinski_02(Scene):

   config.disable_caching_warning=True
  
   def subdivide(self, square, n):
        ULsq = Square(
            side_length=n/2, 
            color=BLACK, 
            fill_color=YELLOW, 
            fill_opacity=1,
            stroke_width=0
        ).align_to(square,LEFT+UP)
        LLsq = ULsq.copy().shift(DOWN*n/2)
        LRsq = LLsq.copy().shift(RIGHT*n/2)
        URsq = ULsq.copy().set_fill(PURE_GREEN).shift(RIGHT*n/2)
        sqs = VGroup(ULsq,LLsq,LRsq,URsq)
        return sqs
 
    
   def construct(self):

        size = 6  
        iterations = 7

        S = Square(
            side_length=size, 
            color=BLACK, 
            fill_color=YELLOW, 
            fill_opacity=1,
            stroke_width=0.5
            )                      
        
        self.play(FadeIn(S))
        self.wait(1)
                         
        B=[0]
        B[0] = self.subdivide(S,size)
        
        # Remaining iterations
        for m in range(0,iterations-1):
           size=size/2
           C = [0]*(3**(m+1))
           if (m > 0): self.wait(1.5)
           for k in range(3**m):
              C[3*k]=self.subdivide(B[k][0],size)
              C[3*k+1]=self.subdivide(B[k][1],size)
              C[3*k+2]=self.subdivide(B[k][2],size)
              #self.add(*C[3*k],*C[3*k+1],*C[3*k+2])     
              #self.remove(*B[k]) 
              #self.play(Write(*B[k][-1]))
              self.add(*B[k][-1])
           if (m == 0): # recombine the squares of iteration 1 back into place
              self.wait(.5)
                           
           if (m < iterations-2): B = C.copy()

        self.wait(2)


        # manim -sqk test.py Sierpinski_02

        # manim -pqh test.py Sierpinski_02




class Sierpinski_03(Scene):

   config.disable_caching_warning=True
  
   def subdivide(self, square, n):
        mid_sq = Square(
            side_length=n/3, 
            fill_color="#E1CD00", 
            fill_opacity=1,
            stroke_width=0
        ).move_to(square.get_center())
        sq_U = mid_sq.copy().shift(UP*n/3)
        sq_D = mid_sq.copy().shift(DOWN*n/3)
        sq_R = mid_sq.copy().shift(RIGHT*n/3)
        sq_L = mid_sq.copy().shift(LEFT*n/3)
        sq_UR = sq_U.copy().shift(RIGHT*n/3)
        sq_DR = sq_R.copy().shift(DOWN*n/3)
        sq_DL = sq_D.copy().shift(LEFT*n/3)
        sq_UL = sq_U.copy().shift(LEFT*n/3)
        mid_sq.set_fill("E1CD00")
        
        sqs = VGroup(sq_U,sq_D,sq_R,sq_L,sq_UR,sq_DL,sq_DR,sq_UL,mid_sq)
        return sqs
 
    
   def construct(self):

        size = 6  
        iterations = 6

        S = Square(
            side_length=size,  
            fill_color="#00673A", 
            fill_opacity=1,
            stroke_width=0.5
            )                      
        
        self.play(FadeIn(S))
        self.wait(1)
                         
        B=[0]
        B[0] = self.subdivide(S,size)
        
        # Remaining iterations
        for m in range(0,iterations-1):
           size=size/3
           C = [0]*(8**(m+1))
           if (m > 0): self.wait(1.5)
           for k in range(8**m):
              C[8*k]=self.subdivide(B[k][0],size)
              C[8*k+1]=self.subdivide(B[k][1],size)
              C[8*k+2]=self.subdivide(B[k][2],size)
              C[8*k+3]=self.subdivide(B[k][3],size)
              C[8*k+4]=self.subdivide(B[k][4],size)
              C[8*k+5]=self.subdivide(B[k][5],size)
              C[8*k+6]=self.subdivide(B[k][6],size)
              C[8*k+7]=self.subdivide(B[k][7],size)
              
              #self.add(*C[3*k],*C[3*k+1],*C[3*k+2])     
              #self.remove(*B[k]) 
              self.add(*B[k][-1])
              #self.play(Write(*B[k][-1]))
              
           if (m == 0): # recombine the squares of iteration 1 back into place
              self.wait(.5)
                           
           if (m < iterations-2): B = C.copy()
        self.wait(2)


        # manim -sqk test.py Sierpinski_03

        # manim -pqh test.py Sierpinski_03

class SierpinskiCarpet(Scene):
    def construct(self):
        # Set the total number of times the process will be repeated
        total = 7

        # Calculate the size of the image
        size = 3**total

        # Create an empty image
        square = np.empty([size, size, 3], dtype=np.uint8)
        color = np.array([222, 198, 0], dtype=np.uint8)

        # Fill it with black
        square.fill(0)

        for i in range(0, total + 1):
            stepdown = 3**(total - i)
            for x in range(0, 3**i):
                if x % 3 == 1:
                    for y in range(0, 3**i):
                        if y % 3 == 1:
                            square[y * stepdown:(y + 1) * stepdown, x * stepdown:(x + 1) * stepdown] = color

            # Convert the NumPy array to an image and display it
            img = Image.fromarray(square)
            self.add(ImageMobject(img).scale(.2))
            #self.play(Write(img))
            self.wait(0.5)  # Adjust the animation speed as needed


            # manim -pqh test.py SierpinskiCarpet


class SierpinskiCarpet_1(Scene):
    def construct(self):
        # Set the total number of times the process will be repeated
        total = 3

        # Calculate the size of the image
        size = 3**total

        # Create an empty image
        square = Square(side_length=3).set_color(WHITE)
        col = REANLEA_GOLD

        self.add(square)


        for i in range(0, total + 1):
            stepdown = 3**(total - i)
            for x in range(0, 3**i):
                if x % 3 == 1:
                    for y in range(0, 3**i):
                        if y % 3 == 1:
                            len = square.get_side_length()/(3**i)
                            sq=Square(side_length=len).set_color(color=col).set_fill(col,opacity=1)

                            # Calculate the position of the square within the Sierpinski Carpet
                            x_position = (x-1) * stepdown  # Adjusted for x % 3 == 1
                            y_position = (y-1) * stepdown  # Adjusted for y % 3 == 1
                            position = np.array([x_position, y_position, 0])

                            # Set the position of the square
                            sq.move_to(position)

                            # Add the square to the scene
                            
            self.add(sq)
            
            self.wait(0.5)  


            # manim -pqh test.py SierpinskiCarpet_1

            # manim -sqh test.py SierpinskiCarpet_1


#------------------------- https://github.com/Rousan99/Azazayav ------------------------


class SierpinskiTriangleTest(Scene):
	def construct(self):
		tt = Title("Sierpinski Triangle")
		tt.set_color(GREEN)
		self.play(Write(tt))
		a = Triangle()
		a.scale(4)
		self.play(Write(a))
		self.play(FadeOut(tt))
		p1 ,p2,p3 =(a.get_points()[0],a.get_points()[3],a.get_points()[7])#0 for 1st , 3 for 2nd and 7 for 3rd vertex
		tit = [p1,p2,p3]
		points=[self.generate_point_in_polygon(p1,p2,p3) for i in range(1)]
		r = Dot(color=RED)
		r.scale(0.20)
		r.move_to(np.array(points[0]))
		self.play(Write(r))
		self.wait()
		y = np.array(points[0])
		X = GREEN
		for i in range(1000):
			ran = random.choice(tit)
			if np.array_equal(ran,tit[0]):
				X = RED
			elif np.array_equal(ran,tit[1]):
				X = BLUE
			else:
				X = YELLOW
			q = Dot(color=X)
			q.scale(0.15)
			new = np.array([(y[0]+ran[0])/2,(y[1]+ran[1])/2,0])
			q.move_to(new)
			self.play(Write(q),run_time=0.002)
			y = new




	def generate_point_in_polygon(self,p1,p2,p3,**kwargs):
		s ,t = sorted([random.random(),random.random()])
		return (s*p1[0] + (t-s)*p2[0] + (1-t)*p3[0] , s*p1[1] + (t-s)*p2[1] + (1-t)*p3[1] , 0)
    

    # manim -pqh test.py SierpinskiTriangleTest


class SierpinskiTriangle2(Scene):
        
        def sub_triangle(self,triangle):
            vertices = triangle.get_vertices()
            a=vertices[0]
            b=vertices[1]
            c=vertices[2]
            tri_0=Polygon((a+b)/2,(b+c)/2,(c+a)/2).set_fill(color="#00673A",opacity=1).set_stroke(color="#00673A",width=0)
            tri_1=Polygon((a+b)/2,a,(c+a)/2)
            tri_2=Polygon((a+b)/2,(b+c)/2,b)
            tri_3=Polygon(c,(b+c)/2,(c+a)/2)

            tris=VGroup(tri_1,tri_2,tri_3,tri_0)

            return tris
        
        def construct(self):
            iterations=8

            Tri=Triangle().scale(4).set_stroke(width=0).set_fill(color="#cca300",opacity=1)

            self.play(FadeIn(Tri))
            self.wait()
            

            B=[0]
            B[0] = self.sub_triangle(Tri)

            self.play(Write(B[0][-1]))

            for m in range(0,iterations-1):
                grp=VGroup()
                C = [0]*(3**(m+1))
                if (m > 0): self.wait(1.5)
                for k in range(3**m):
                    C[3*k]=self.sub_triangle(B[k][0])
                    C[3*k+1]=self.sub_triangle(B[k][1])
                    C[3*k+2]=self.sub_triangle(B[k][2])
                     
                    grp += VGroup(*B[k-1][-1])

                self.play(Write(grp))  
             
                   
                if (m == 0): # recombine the squares of iteration 1 back into place
                 self.wait(.5)          
                           
                if (m < iterations-2): B = C.copy()
            
          
            self.wait(2)



        # manim -pqh test.py SierpinskiTriangle2



class CW_KochCurve(Scene):
    def construct(self):
        def KochCurve(
            n, length=12, stroke_width=8, color=("#0A68EF", "#4AF1F2", "#0A68EF")
        ):

            l = length / (3 ** n)

            LineGrp = Line().set_length(l)

            def NextLevel(LineX):
                return VGroup(
                    *[LineX.copy().rotate(i) for i in [0, PI / 3, -PI / 3, 0]]
                ).arrange(RIGHT, buff=0, aligned_edge=DOWN)

            for _ in range(n):
                LineGrp = NextLevel(LineGrp)

            KC = (
                VMobject(stroke_width=stroke_width)
                .set_points(LineGrp.get_all_points())
                .set_color(color)
            )
            return KC

        
        kc = KochCurve(0, stroke_width=12).to_edge(DOWN, buff=2.5)

        self.add(kc)
        self.wait()

        for i in range(1, 6):
            self.play(
                kc.animate.become(
                    KochCurve(i, stroke_width=12 - (2 * i)).to_edge(DOWN, buff=2.5)
                ),
            )
            self.wait()



            #  manim -pqh test.py CW_KochCurve


config.background_color=WHITE
class Cantor_Set(Scene):
        def subdivide(self, line):
            len=line.get_length()/3

            ln_0=Line().set_stroke(color=REANLEA_WARM_BLUE_DARKER,width=8).set_length(len).move_to(line.get_center())

            ln_1=ln_0.copy().shift(LEFT*len)
            ln_2=ln_0.copy().shift(RIGHT*len)

            ln_0.set_stroke(width=9, color=WHITE)

            lns=VGroup(ln_1,ln_2,ln_0)
            return lns
        
        def construct(self):

            water_mark=ImageMobject("water_mark_white.png").scale(0.075).move_to(5*LEFT+3*UP).set_opacity(1).set_z_index(-100)
            self.add(water_mark)

            #anim zone 

            length=12
            iterations=9

            level = Variable(0, Tex("iterations:"), var_type=Integer).set_color(REANLEA_BACKGROUND_COLOR_OXFORD_BLUE)
            txt = (
                VGroup(Tex("Cantor Set", font_size=60).set_color(REANLEA_BACKGROUND_COLOR_OXFORD_BLUE), level).arrange(DOWN, aligned_edge=LEFT).to_corner(UR)
            )
            self.play(Write(txt))

            line=Line(color=WHITE).set_stroke(color=REANLEA_WARM_BLUE_DARKER,width=8).set_length(length)

            self.play(Create(line))
            self.wait()

            B=[0]
            B[0]=self.subdivide(line)
            

            for m in range(0,iterations-1):
                grp=VGroup()
            
                C = [0]*(2**(m+1))
                if (m > 0): self.wait(1.5)
                for k in range(2**m):
                    C[2*k]=self.subdivide(B[k][0])
                    C[2*k+1]=self.subdivide(B[k][1])
                                        
                    grp += VGroup(*B[k-1])
                
                grp.move_to(.2*m*DOWN)
                                         
                self.play(FadeIn(grp),level.tracker.animate.set_value(m+1))  
                
                   
                if (m == 0):
                 grp+=VGroup(B[0])

                self.play(grp.animate.shift(.2*DOWN))
                self.wait(.15)          
                           
                if (m < iterations-2): B = C.copy()
            
            self.wait(2)
            



        #  manim -pqh test.py Cantor_Set

        #  manim -sqk test.py Cantor_Set

###################################################################################################################




###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2023\lab"