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
###################################################################################################################




###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2023\lab" 