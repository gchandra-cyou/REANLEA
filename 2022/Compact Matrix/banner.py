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


class esp_ex(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("background_4.png").scale(.5).set_z_index(-10)
        self.add(water_mark)

        

        graphs = VGroup()
        for n in np.arange(1,15):    
            if n==1:
                graphs += ImplicitFunction(lambda x,y : np.abs(x)**n + np.abs(y)**n -1).scale(3).set_stroke(width=2)
            else:
                graphs += ImplicitFunction(lambda x,y : np.abs(x)**n + np.abs(y)**n -1).scale(3).set_stroke(width=7/n)
                
        graphs.scale(.70).set_color_by_gradient(REANLEA_BLUE,REANLEA_BLUE_SKY).move_to(ORIGIN)
        #.next_to(eqn_1,DOWN).shift(.5*DOWN)
        graphs[0].set_color_by_gradient(REANLEA_RED,REANLEA_BLUE_SKY)

        eqn_1=MathTex(r"d_{k}(x,y)",r"= \lVert x-y \rVert",r"= \Biggl\lbrack \sum_{i=1}^{n} \lvert x_{i}-y_{i} \rvert ^{k} \Biggr\rbrack ^{1/k} ").scale(.75).set_color_by_gradient(REANLEA_YELLOW, REANLEA_BLUE_LAVENDER).next_to(graphs,UP).set_stroke(width=1.25).shift(.25*RIGHT)

        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("Geometry of Euclidean Space." , font=fonts[0]).set_color_by_gradient(REANLEA_BLUE_LAVENDER)

        self.add(eqn_1)
        graphs.shift(.35*DOWN)
        txt_1.scale(.65).next_to(graphs,DOWN).shift(.35*DOWN)

        dot1= Dot(radius=.25).scale(.85).set_color(REANLEA_PURPLE_LIGHTER).set_sheen(-0.4,DOWN).move_to(graphs.get_center())
        glowing_circle_1=get_glowing_surround_circle(dot1, color=REANLEA_YELLOW)

        self.play(
            Create(graphs[0:]), 
            run_rime=18
        )

        self.add(txt_1,dot1,glowing_circle_1)


    
    # manim -sqk banner.py esp_ex

class esp_ex_1(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("background_4.png").scale(.5).set_z_index(-10)
        self.add(water_mark)

        
        # MOBJECTS

        dot1=Dot(radius=0.25, color=REANLEA_GREEN).move_to(3*RIGHT).set_sheen(-0.6,DOWN)
        dot2=Dot(radius=0.25, color=REANLEA_VIOLET_LIGHTER).move_to(3*LEFT).set_sheen(-0.4,DOWN)
        dot3=Dot(radius=0.125, color=REANLEA_YELLOW).move_to(LEFT+2.5*UP).set_sheen(-0.6,DOWN)
        dot3_1=Dot(radius=0.125, color=REANLEA_YELLOW).move_to(LEFT+2.5*UP).set_sheen(-0.6,DOWN)

        dt_grp=VGroup(dot1,dot2,dot3,dot3_1)

        line1=Line(start=dot2.get_center(), end=dot1.get_center()).set_color(REANLEA_YELLOW_DARKER).set_stroke(width=10).set_z_index(-2)
        line2=Line(start=dot2.get_center(), end=dot3.get_center()).set_color(REANLEA_GREEN_DARKER).set_stroke(width=5).set_z_index(-3)
        line3=Line(start=dot3.get_center(), end=dot1.get_center()).set_color(REANLEA_VIOLET_DARKER).set_stroke(width=5).set_z_index(-1)
        
        line2_1=Line(start=dot2.get_center(), end=dot3.get_center()).set_color(REANLEA_GREEN_DARKER).set_stroke(width=5).set_z_index(-3)
        line3_1=Line(start=dot3.get_center(), end=dot1.get_center()).set_color(REANLEA_VIOLET_DARKER).set_stroke(width=5).set_z_index(-1)

        line1_p1=Line(start=dot2.get_center(), end=np.array((dot3.get_center()[0],0,0)))
        line1_p1.set_color(REANLEA_YELLOW).set_opacity(0.65).set_stroke(width=5).set_z_index(-1)

        ln_grp=VGroup(line1)

        projec_line=DashedLine(start=dot3.get_center(), end=np.array((dot3.get_center()[0],0,0)), stroke_width=1).set_color(REANLEA_AQUA_GREEN).set_z_index(-2)
        
        angle_12=Angle(line1,line2, radius=.5, other_angle=False).set_color(REANLEA_GREEN).set_z_index(-3)
        angle_13=Angle(line3,line1, radius=.65, quadrant=(-1,-1),other_angle=False).set_color(REANLEA_VIOLET).set_z_index(-3)

        circ=DashedVMobject(Circle(radius=line1_p1.get_length()), dashed_ratio=0.5, num_dashes=100).move_to(dot2.get_center()).set_stroke(width=0.65)
        circ.set_color_by_gradient(REANLEA_WHITE,REANLEA_WARM_BLUE,REANLEA_YELLOW_CREAM)

        cir_grp=VGroup(projec_line,circ)

        brace_line2=Brace(Line(start=dot2.get_center(), end=np.array((dot3.get_center()[0],0,0)))).set_color(REANLEA_GREEN).set_opacity(0.8).set_z_index(-1)
        brace_line3=Brace(Line(start=np.array((dot3.get_center()[0],0,0)), end=dot1.get_center())).set_color(REANLEA_VIOLET).set_opacity(0.8).set_z_index(-1)

        brc_grp=VGroup(brace_line2,brace_line3)

        dot1_lbl2=MathTex("y").scale(0.6).next_to(dot1, RIGHT)
        dot2_lbl2=MathTex("x").scale(0.6).next_to(dot2, LEFT)
        dot3_lbl=MathTex("z").scale(0.6).next_to(dot3, UP)

        lbl_grp=VGroup(dot1_lbl2,dot2_lbl2,dot3_lbl)


        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("Distance is not a difference between two points." , font=fonts[0]).set_color_by_gradient(REANLEA_BLUE_LAVENDER)
            txt_1.scale(.5).shift(2.75*DOWN)

        eq14=MathTex(r"d|",r"_{\mathbb{X} \times \mathbb{X}}","(x,y)").scale(1.35).set_color_by_gradient(REANLEA_AQUA_GREEN,REANLEA_WARM_BLUE)
        eq14[1].next_to(eq14[0].get_center(),0.01*RIGHT+0.1*DOWN)
        eq14[2].next_to(eq14[0],3.5*RIGHT)
        #eq14.move_to(2*DOWN)
        eq14[1].scale(0.5)

        eq15=MathTex(r"\in \mathbb{R}^{+} \cup \{0\}").scale(1.3).next_to(eq14,RIGHT).set_color_by_tex("",color=(REANLEA_CYAN_LIGHT,REANLEA_WARM_BLUE))
        
        eq145=VGroup(eq14,eq15).scale(.7).shift(2*UP+2*RIGHT)

        ind_ln_0=Line().scale(.85).set_stroke(width=1).rotate(-135*DEGREES).next_to(eq145,DOWN).shift(2*LEFT)

        self.add(dt_grp,txt_1,ln_grp,cir_grp,brc_grp,lbl_grp,eq145,ind_ln_0)



    
    # manim -sqk banner.py esp_ex_1


class yt_banner(Scene):
    def construct(self):
        config.background_color="#000327" # secondary color : "#000658"

        water_mark=ImageMobject("yt_banner_1.png").scale(.3725).set_z_index(-10)
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
                ).set_stroke(width=stroke_width/n,color=REANLEA_GOLD)
            )
            return grph
        
        grph_1= VGroup(
                ax.plot(
                    lambda x: x,
                    x_range=[0,2,0.001]
                ).set_stroke(width=2,color=[REANLEA_WARM_BLUE]),
                ax.plot(
                    lambda x: -x,
                    x_range=[-2,0,0.001]
                ).set_stroke(width=2,color=[REANLEA_WARM_BLUE])
            )
        
        
        x1=graph(n=1)
        x2=graph(n=2)


        x=VGroup(
            *[
                graph(n=i)
                for i in range(3,15)
            ]
        )



        lbl_x1=MathTex(r"y^{2}=x^{2}+\frac{1}{n}").scale(.35).set_color(REANLEA_WHITE).next_to(x1,UP).shift(1.85*DOWN)

        lbl_grph1_0=MathTex(r"y=\lvert x \rvert").scale(.35).set_color(REANLEA_WHITE).next_to(grph_1[0],RIGHT).rotate(45*DEGREES).shift(1.5*LEFT)

        lbl_grph1_1=MathTex(r"y=\lvert x \rvert").scale(.35).set_color(REANLEA_WHITE).next_to(grph_1[1],LEFT).rotate(-45*DEGREES).shift(1.5*RIGHT)

        x_grp=VGroup(x1,x2,x,grph_1,lbl_x1,lbl_grph1_0,lbl_grph1_1).scale(.4).move_to(ORIGIN).shift(.2*UP)

        self.add(x_grp)

        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("Geometric foundation of Mathematics and Physics with Animated visuals." , font=fonts[0]).set_color_by_gradient(REANLEA_TXT_COL_LIGHTER)

        txt_1.scale(.2).shift(.85*DOWN)


        self.add(lbl_x1,lbl_grph1_0,lbl_grph1_1,txt_1)

        # manim -sqk banner.py -r 5120,2630 yt_banner

        # manim -sqk banner.py yt_banner


class yt_banner_1(Scene):
    def construct(self):
        config.background_color="#000327" # secondary color : "#000658"

        water_mark=ImageMobject("yt_banner_1.png").scale(.3725).set_z_index(-10)
        self.add(water_mark)

        # OBJECT ZONE

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
        ).set_color(REANLEA_GREY).set_stroke(width=.35)

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
                color=REANLEA_GOLD,
            ).set_stroke(width=1.25).set_z_index(2),
            ax.plot(
                lambda x: - x**2,
                x_range=[0.00,0.1,0.0001],
                color=GREY,
            ).set_stroke(width=1)
        )
        
        
        #Show animate

        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("Geometric foundation of Mathematics and Physics with Animated visuals." , font=fonts[0]).set_color_by_gradient(REANLEA_TXT_COL_LIGHTER)
        txt_1.scale(.2).shift(.85*DOWN)

        grp_x=VGroup(graph,dt,ax).scale(.25).shift(.25*UP)


        self.add(grp_x,txt_1)

        # manim -sqk banner.py -r 5120,2630 yt_banner_1

        # manim -sqk banner.py yt_banner_1


class yt_banner_1_1(Scene):
    def construct(self):
        config.background_color="#000327" # secondary color : "#000658"

        water_mark=ImageMobject("yt_banner_1_1.png").scale(.3725).set_z_index(-10)
        self.add(water_mark)

        # OBJECT ZONE

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
        ).set_color(REANLEA_GREY).set_stroke(width=.35)

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
                color=REANLEA_GOLD,
            ).set_stroke(width=1.25).set_z_index(2),
            ax.plot(
                lambda x: - x**2,
                x_range=[0.00,0.1,0.0001],
                color=GREY,
            ).set_stroke(width=1)
        )
        
        
        #Show animate

        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("Geometric foundation of Mathematics and Physics with Animated visuals." , font=fonts[0]).set_color_by_gradient(REANLEA_TXT_COL_LIGHTER)
        txt_1.scale(.2).shift(.85*DOWN)

        grp_x=VGroup(graph,dt,ax).scale(.25).shift(.25*UP)


        self.add(grp_x,txt_1)

        # manim -sqk banner.py -r 5120,2630 yt_banner_1

        # manim -sqk banner.py yt_banner_1_1


class yt_banner_2(Scene):
    def construct(self):
        config.background_color="#000327" # secondary color : "#000658"

        water_mark=ImageMobject("yt_banner_2.png").scale(.6).set_z_index(-10)
        self.add(water_mark)

        # OBJECT ZONE

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
        ).set_color(REANLEA_GREY).set_stroke(width=.35)

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
                color=REANLEA_GOLD,
            ).set_stroke(width=1.25).set_z_index(2),
            ax.plot(
                lambda x: - x**2,
                x_range=[0.00,0.1,0.0001],
                color=GREY,
            ).set_stroke(width=1)
        )
        
        
        #Show animate

        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("Geometric foundation of Mathematics and Physics with Animated visuals." , font=fonts[0]).set_color_by_gradient(REANLEA_TXT_COL_LIGHTER)
        txt_1.scale(.2).shift(1.4*DOWN)

        grp_x=VGroup(graph,dt,ax).scale(.3).shift(.25*UP)


        self.add(grp_x,txt_1)

        # manim -sqk banner.py -r 5120,2630 yt_banner_2

        # manim -sqk banner.py yt_banner_2

###################################################################################################################

# Changing FONTS : import any font from Google
# some of my fav fonts: Cinzel,Kalam,Prata,Kaushan Script,Cormorant, Handlee,Monoton, Bad Script, Reenie Beanie, 
# Poiret One,Merienda,Julius Sans One,Merienda One,Cinzel Decorative, Montserrat, Cousine, Courier Prime , The Nautigal
# Marcellus SC,Contrail One,Thasadith,Spectral SC,Dongle,Cormorant SC,Comfortaa, Josefin Sans (LOVE), Fuzzy Bubbles

###################################################################################################################


###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix" 