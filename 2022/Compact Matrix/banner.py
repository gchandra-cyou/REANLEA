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

        self.add(dot1,glowing_circle_1)


    
    # manim -sqk banner.py esp_ex


def get_rays(
        factor=1,scale_about_point=ORIGIN,rotate_about_point=ORIGIN, buff_min=0, buff_max=360, color=REANLEA_TXT_COL_DARKER, n=10
):
    line=DashedLine(ORIGIN,RIGHT, stroke_width=1).set_color(color).scale(factor,about_point=scale_about_point)

    rays=VGroup(
        *[
            line.copy().rotate(k*DEGREES, about_point=rotate_about_point)
            for k in np.linspace(buff_min,buff_max,n)
        ]
    )

    return rays

class esp_ex_1(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("background_4.png").scale(.5).set_z_index(-100)
        self.add(water_mark)

        
        # MOBJECTS

        dot1=Dot(radius=0.25, color=REANLEA_GREEN).move_to(3*RIGHT).set_sheen(-0.6,DOWN)
        dot2=Dot(radius=0.25, color=REANLEA_VIOLET_LIGHTER).move_to(3*LEFT).set_sheen(-0.4,DOWN)
        dot3=Dot(radius=0.15, color=REANLEA_YELLOW).move_to(LEFT+2.5*UP).set_sheen(-0.6,DOWN)
        dot3_1=Dot(radius=0.15, color=REANLEA_YELLOW).move_to(LEFT+2.5*UP).set_sheen(-0.6,DOWN)

        dt_grp=VGroup(dot1,dot2,dot3,dot3_1)

        glowing_circle_1=get_glowing_surround_circle(dot1, color=REANLEA_YELLOW).set_z_index(-20)
        glowing_circle_2=get_glowing_surround_circle(dot2, color=REANLEA_YELLOW).set_z_index(-20)
        glowing_circle_3=get_glowing_surround_circle(dot3, color=REANLEA_PURPLE_LIGHTER).set_z_index(-20)

        gl_cir_grp=VGroup(glowing_circle_1,glowing_circle_2,glowing_circle_3)

        line1=Line(start=dot2.get_center(), end=dot1.get_center()).set_color(REANLEA_YELLOW_DARKER).set_stroke(width=10).set_z_index(-2)
        line2=Line(start=dot2.get_center(), end=dot3.get_center()).set_color(REANLEA_GREEN_DARKER).set_stroke(width=7).set_z_index(-3)
        line3=Line(start=dot3.get_center(), end=dot1.get_center()).set_color(REANLEA_VIOLET_DARKER).set_stroke(width=7).set_z_index(-1)
        
        line2_1=Line(start=dot2.get_center(), end=dot3.get_center()).set_color(REANLEA_GREEN_DARKER).set_stroke(width=5).set_z_index(-3)
        line3_1=Line(start=dot3.get_center(), end=dot1.get_center()).set_color(REANLEA_VIOLET_DARKER).set_stroke(width=5).set_z_index(-1)

        line1_p1=Line(start=dot2.get_center(), end=np.array((dot3.get_center()[0],0,0)))
        line1_p1.set_color(REANLEA_YELLOW).set_opacity(0.65).set_stroke(width=5).set_z_index(-1)

        ln_grp=VGroup(line1,line2,line3)

        projec_line=DashedLine(start=dot3.get_center(), end=np.array((dot3.get_center()[0],0,0)), stroke_width=1).set_color(REANLEA_AQUA_GREEN).set_z_index(-2)
        
        angle_12=Angle(line1,line2, radius=.5, other_angle=False).set_color(REANLEA_GREEN).set_z_index(-3)
        angle_13=Angle(line3,line1, radius=.65, quadrant=(-1,-1),other_angle=False).set_color(REANLEA_VIOLET).set_z_index(-3)

        circ=DashedVMobject(Circle(radius=line1_p1.get_length()), dashed_ratio=0.5, num_dashes=100).move_to(dot2.get_center()).set_stroke(width=1)
        circ.set_color_by_gradient(REANLEA_CYAN_LIGHT).set_z_index(10)

        circ_ref=Circle(radius=line1_p1.get_length()).move_to(dot2.get_center())

        cir_grp=VGroup(projec_line,circ)

        rays=get_rays(n=100,color=PURE_GREEN,factor=2).move_to(dot2.get_center())

        brace_line2=Brace(Line(start=dot2.get_center(), end=np.array((dot3.get_center()[0],0,0)))).set_color(REANLEA_GREEN).set_opacity(0.8).set_z_index(-1)
        brace_line3=Brace(Line(start=np.array((dot3.get_center()[0],0,0)), end=dot1.get_center())).set_color(REANLEA_VIOLET).set_opacity(0.8).set_z_index(-1)

        brc_grp=VGroup(brace_line2,brace_line3)

        dot1_lbl2=MathTex("y").scale(0.6).next_to(dot1, RIGHT)
        dot2_lbl2=MathTex("x").scale(0.6).next_to(dot2, LEFT)
        dot3_lbl=MathTex("z").scale(0.6).next_to(dot3, UP)

        lbl_grp=VGroup(dot1_lbl2,dot2_lbl2,dot3_lbl)


        with RegisterFont("Cousine") as fonts:
            txt_1 = Text("Geometry of Euclidean Spaces." , font=fonts[0]).set_color_by_gradient(REANLEA_BLUE_LAVENDER)
            txt_1.scale(.5).shift(3*DOWN)

            txt_2 = Text("Topology via Algebra." , font=fonts[0]).set_color_by_gradient(REANLEA_BLUE_LAVENDER)
            txt_2.scale(.7).shift(2.75*DOWN)

        eq14=MathTex(r"d|",r"_{\mathbb{X} \times \mathbb{X}}","(x,y)").scale(1.35).set_color_by_gradient(REANLEA_YELLOW_CREAM,REANLEA_YELLOW)
        eq14[1].next_to(eq14[0].get_center(),0.01*RIGHT+0.1*DOWN)
        eq14[2].next_to(eq14[0],3.5*RIGHT)
        #eq14.move_to(2*DOWN)
        eq14[1].scale(0.5)

        eq15=MathTex(r"\in \mathbb{R}^{+} \cup \{0\}").scale(1.3).next_to(eq14,RIGHT).set_color_by_tex("",color=(REANLEA_CYAN_LIGHT,REANLEA_YELLOW))
        
        eq145=VGroup(eq14,eq15).scale(.7).shift(2*UP+2*RIGHT)

        ind_ln_0=Line().scale(.85).set_stroke(width=1).rotate(-135*DEGREES).next_to(eq145,DOWN).shift(2*LEFT)

        

        self.add(dt_grp,gl_cir_grp,ln_grp,cir_grp,brc_grp,lbl_grp,eq145,ind_ln_0,rays)



    
    # manim -sqk banner.py esp_ex_1


class esp_ex_2(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("background_4.png").scale(.5).set_z_index(-100)
        self.add(water_mark)

        
        # MOBJECTS

        line_1= NumberLine(
            x_range=[-8, 8, 1],
            length=32,
            include_ticks=False,
        ).set_color(REANLEA_BLUE_LAVENDER).set_stroke(width=4).move_to(DOWN)

        zero_tick = VGroup(
            Line(0.3 * UP, 0.3 * DOWN, stroke_width=2.0, color=REANLEA_VIOLET_LIGHTER),
            MathTex("0"),
        )
        zero_tick[0].move_to(line_1.n2p(-1))
        zero_tick[1].next_to(zero_tick[0], DOWN)
        zero_tick.set_z_index(3)

        one_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_GREEN),
            MathTex("1").scale(.5),
        )
        one_tick[0].move_to(line_1.n2p(0))
        one_tick[1].next_to(one_tick[0], DOWN)
        one_tick.set_z_index(3)

        minus_one_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_YELLOW),
            MathTex("-1").scale(.5),
        )
        minus_one_tick[0].move_to(line_1.n2p(-2))
        minus_one_tick[1].next_to(minus_one_tick[0], DOWN)
        minus_one_tick.set_z_index(3)


        two_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_GREEN),
            MathTex("2").scale(.5),
        )
        two_tick[0].move_to(line_1.n2p(1))
        two_tick[1].next_to(two_tick[0], DOWN)
        two_tick.set_z_index(3)

        three_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_GREEN),
            MathTex("3").scale(.5),
        )
        three_tick[0].move_to(line_1.n2p(2))
        three_tick[1].next_to(three_tick[0], DOWN)
        three_tick.set_z_index(3)

        four_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_GREEN),
            MathTex("4").scale(.5),
        )
        four_tick[0].move_to(line_1.n2p(3))
        four_tick[1].next_to(four_tick[0], DOWN)
        four_tick.set_z_index(3)

        five_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_YELLOW),
            MathTex("-2").scale(.5),
        )
        five_tick[0].move_to(line_1.n2p(-3))
        five_tick[1].next_to(five_tick[0], DOWN)
        five_tick.set_z_index(3)

        tick_grp=VGroup(zero_tick,one_tick,minus_one_tick, two_tick,three_tick,four_tick,five_tick)

        sgn_pos_1=MathTex("+").scale(.75).set_color(PURE_GREEN).move_to(6.5*RIGHT)
        sgn_pos_2=Circle(radius=0.2, color=PURE_GREEN).move_to(sgn_pos_1.get_center()).set_stroke(width= 1)
        sgn_pos=VGroup(sgn_pos_1,sgn_pos_2)

        sgn_neg_1=MathTex("-").scale(.75).set_color(REANLEA_YELLOW).move_to(6.5*LEFT)
        sgn_neg_2=Circle(radius=0.2, color=REANLEA_YELLOW).move_to(sgn_neg_1.get_center()).set_stroke(width= 1)
        sgn_neg=VGroup(sgn_neg_1,sgn_neg_2)

        sgn_grp=VGroup(sgn_pos,sgn_neg)

        so_on_txt_symbol=Text("...").move_to(0.9*DOWN+6.9*RIGHT).scale(0.5).set_color(REANLEA_GREEN)

        mirror_1=get_mirror().move_to(line_1.n2p(-1)).shift(.12*LEFT)

        vect_1=Arrow(start=line_1.n2p(-1),end=line_1.n2p(0),max_tip_length_to_length_ratio=0.125, buff=0).set_color(PURE_GREEN).set_opacity(1)
        vect_1.set_z_index(4)
        vect_1_lbl=MathTex("u").scale(.85).next_to(vect_1,0.5*DOWN).set_color(PURE_GREEN)

        vect_1_moving=Arrow(start=line_1.n2p(-1),end=line_1.n2p(0),max_tip_length_to_length_ratio=0.125, buff=0).set_color(REANLEA_YELLOW).rotate(137.2*DEGREES,about_point=line_1.n2p(-1)).set_z_index(3)

        ang=Angle(vect_1, vect_1_moving, radius=0.5, other_angle=False).set_stroke(color=REANLEA_SAFRON_LIGHTER, width=3).set_z_index(-1)

        ang_lbl = MathTex(r"\theta =").move_to(
            Angle(
                vect_1, vect_1_moving, radius=.85 + 3 * SMALL_BUFF, other_angle=False
            ).point_from_proportion(.5)                         # Gets the point at a proportion along the path of the VMobject.
        ).scale(.5).set_color(REANLEA_YELLOW)

        ang_theta=DecimalNumber(137.29,unit="^o").scale(.5).set_color(REANLEA_YELLOW).next_to(ang_lbl).shift(.175*LEFT)

        projec_line=always_redraw(
            lambda : DashedLine(start=vect_1_moving.get_end(), end=np.array((vect_1_moving.get_end()[0],line_1.n2p(0)[1],0))).set_stroke(color=REANLEA_AQUA_GREEN, width=1)
        )

        bra_1=always_redraw(
            lambda : BraceBetweenPoints(
                point_1=vect_1.get_start(),
                point_2=np.array((vect_1_moving.get_end()[0],0,0)),
                direction=DOWN,
                color=REANLEA_YELLOW
            ).set_stroke(width=0.1).set_z_index(5)
        )

        vect_grp=VGroup(vect_1,vect_1_lbl,vect_1_moving,ang,ang_lbl,ang_theta,projec_line,bra_1)

        ang_theta_cos_lbl_left=MathTex("u","\cdot",r"cos(\theta)").arrange(RIGHT,buff=0.2).move_to(UP +RIGHT)
        ang_theta_cos_lbl_right=MathTex("\cdot","u").arrange(RIGHT, buff=0.2).set_color(PURE_GREEN).move_to(UP +4.55*RIGHT)
        ang_theta_cos=MathTex("=-0.735").arrange(RIGHT, buff=0.2).set_color(REANLEA_YELLOW).next_to(ang_theta_cos_lbl_right,LEFT).shift(.1*RIGHT)
        ang_theta_cos_lbl_left[0:2].set_color(PURE_GREEN)
        ang_theta_cos_lbl_left[2][0:3].set_color(REANLEA_WARM_BLUE)
        ang_theta_cos_lbl_left[2][4].set_color(REANLEA_SAFRON_LIGHTER)
        ang_theta_cos_grp=VGroup(ang_theta_cos_lbl_left,ang_theta_cos,ang_theta_cos_lbl_right).scale(.65)
        sur_ang_theta_cos_grp=SurroundingRectangle(ang_theta_cos_grp, color=REANLEA_TXT_COL,corner_radius=0.125, buff=0.2)

        ang_lbl=VGroup(ang_theta_cos_grp,sur_ang_theta_cos_grp)


        with RegisterFont("Cousine") as fonts:
            text_1 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Scaling Factor",
            )]).scale(0.24).set_color(REANLEA_GREY)

        text_1.move_to(1.8*UP+5.5*RIGHT)

        txt_blg_1=MathTex(r"\in", r"\mathbb{R}").set_color(REANLEA_TXT_COL).scale(0.7).next_to(text_1,DOWN)
        txt_blg_1[0].scale(0.65)
        txt_blg_1[1].set_color(REANLEA_CYAN_LIGHT)

        bez=bend_bezier_arrow_indicate().flip(RIGHT).move_to(1.5*UP+ 4*RIGHT).scale(.75).rotate(-20*DEGREES).set_color(REANLEA_TXT_COL)

        txt_grp=VGroup(text_1,bez,txt_blg_1)

        ang_lbl_grp=VGroup(txt_grp,ang_lbl).scale(.9).shift(DOWN)

        fld_exp_1=MathTex(r"(\mathbb{R},+,\cdot)").set_color_by_gradient(REANLEA_GOLD,REANLEA_PINK)

        vsp_exp_1=MathTex(r"\mathbb{R}").set_color_by_gradient(REANLEA_GREEN)

        with RegisterFont("Pacifico") as fonts:
            fld_exp_2=Text(" is a Field", font=fonts[0]).scale(0.65).set_color_by_gradient(REANLEA_PINK,REANLEA_MAGENTA).move_to(5.5*LEFT+3.35*UP)
            
        with RegisterFont("Cousine") as fonts:
            vsp_exp_2=Text(" - Vector Space", font=fonts[0]).scale(0.65).set_color_by_gradient(REANLEA_GREEN,REANLEA_AQUA)


        fld_exp=VGroup(fld_exp_1,fld_exp_2).arrange(RIGHT, buff=0.2).shift(1.75*UP+.5*RIGHT).scale(1)

        vsp_exp=VGroup(vsp_exp_1,vsp_exp_2).arrange(RIGHT, buff=0.2).shift(3.25*UP+.45*RIGHT).scale(.75)

        l_1=Line().rotate(PI/2).set_stroke(width=5, color=(PURE_GREEN,REANLEA_BLUE_SKY)).scale(0.35).next_to(fld_exp,UP).shift(1.45*LEFT)

        dot_1=Dot(radius=0.125, color=PURE_RED).move_to(line_1.n2p(-1)).set_sheen(-0.4,DOWN).set_opacity(1).set_z_index(5)
        dot_2=Dot(radius=0.2, color=REANLEA_VIOLET_LIGHTER).move_to(line_1.n2p(0)).set_sheen(-0.4,DOWN).set_z_index(3)
        glowing_circle_2=get_glowing_surround_circle(dot_2, color=REANLEA_YELLOW).set_z_index(-20)

        rays=get_rays(n=100,color=PURE_GREEN,factor=.65).move_to(dot_2.get_center()).set_z_index(4)

        dot_grp=VGroup(dot_1,dot_2,glowing_circle_2,rays)

        d_line_1=DashedLine(line_1.n2p(1.77), end=line_1.n2p(1.77)+.3*UP, stroke_width=1).set_color(PURE_RED)

        d_d_arr_1=DashedDoubleArrow(
            start=line_1.n2p(-1.35)+.75*DOWN, end=line_1.n2p(0.77)+.5*UP, dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).shift(0.3*UP).set_color_by_gradient(REANLEA_YELLOW_GREEN)

        ind_ln_0=Line().scale(.85).set_stroke(width=1).rotate(105*DEGREES).move_to(line_1.n2p(-1.35)+UP).shift(2.5*LEFT)

        with RegisterFont("Cousine") as fonts:
            txt_2 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "each of its points",
                "represents a vector"
            )]).scale(0.2).set_color(REANLEA_WHITE).arrange_submobjects(.25*DOWN).move_to(1.75*UP+3.75*RIGHT)

        sr_bez_1=get_surround_bezier(txt_2).set_color(REANLEA_GREY_DARKER)

        txt_2_grp=VGroup(txt_2,sr_bez_1).next_to(ind_ln_0,UP)


        self.add(line_1,tick_grp,sgn_grp,so_on_txt_symbol,mirror_1,vect_grp,ang_lbl_grp,fld_exp,l_1,vsp_exp,dot_grp,d_d_arr_1,ind_ln_0,txt_2_grp)


    
    # manim -sqk banner.py esp_ex_2


class esp_ex_2_1(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("background_4.png").scale(.5).set_z_index(-100)
        self.add(water_mark)

        
        # MOBJECTS

        line_1= NumberLine(
            x_range=[-8, 8, 1],
            length=32,
            include_ticks=False,
        ).set_color(REANLEA_BLUE_LAVENDER).set_stroke(width=4).move_to(DOWN)

        zero_tick = VGroup(
            Line(0.3 * UP, 0.3 * DOWN, stroke_width=2.0, color=REANLEA_VIOLET_LIGHTER),
            MathTex("0"),
        )
        zero_tick[0].move_to(line_1.n2p(-1))
        zero_tick[1].next_to(zero_tick[0], DOWN)
        zero_tick.set_z_index(3)

        one_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_GREEN),
            MathTex("1").scale(.5),
        )
        one_tick[0].move_to(line_1.n2p(0))
        one_tick[1].next_to(one_tick[0], DOWN)
        one_tick.set_z_index(3)

        minus_one_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_YELLOW),
            MathTex("-1").scale(.5),
        )
        minus_one_tick[0].move_to(line_1.n2p(-2))
        minus_one_tick[1].next_to(minus_one_tick[0], DOWN)
        minus_one_tick.set_z_index(3)


        two_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_GREEN),
            MathTex("2").scale(.5),
        )
        two_tick[0].move_to(line_1.n2p(1))
        two_tick[1].next_to(two_tick[0], DOWN)
        two_tick.set_z_index(3)

        three_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_GREEN),
            MathTex("3").scale(.5),
        )
        three_tick[0].move_to(line_1.n2p(2))
        three_tick[1].next_to(three_tick[0], DOWN)
        three_tick.set_z_index(3)

        four_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_GREEN),
            MathTex("4").scale(.5),
        )
        four_tick[0].move_to(line_1.n2p(3))
        four_tick[1].next_to(four_tick[0], DOWN)
        four_tick.set_z_index(3)

        five_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_YELLOW),
            MathTex("-2").scale(.5),
        )
        five_tick[0].move_to(line_1.n2p(-3))
        five_tick[1].next_to(five_tick[0], DOWN)
        five_tick.set_z_index(3)

        tick_grp=VGroup(zero_tick,one_tick,minus_one_tick, two_tick,three_tick,four_tick,five_tick)

        sgn_pos_1=MathTex("+").scale(.75).set_color(PURE_GREEN).move_to(6.5*RIGHT)
        sgn_pos_2=Circle(radius=0.2, color=PURE_GREEN).move_to(sgn_pos_1.get_center()).set_stroke(width= 1)
        sgn_pos=VGroup(sgn_pos_1,sgn_pos_2)

        sgn_neg_1=MathTex("-").scale(.75).set_color(REANLEA_YELLOW).move_to(6.5*LEFT)
        sgn_neg_2=Circle(radius=0.2, color=REANLEA_YELLOW).move_to(sgn_neg_1.get_center()).set_stroke(width= 1)
        sgn_neg=VGroup(sgn_neg_1,sgn_neg_2)

        sgn_grp=VGroup(sgn_pos,sgn_neg)

        so_on_txt_symbol=Text("...").move_to(0.9*DOWN+6.9*RIGHT).scale(0.5).set_color(REANLEA_GREEN)

        mirror_1=get_mirror().move_to(line_1.n2p(-1)).shift(.12*LEFT)

        vect_1=Arrow(start=line_1.n2p(-1),end=line_1.n2p(0),max_tip_length_to_length_ratio=0.125, buff=0).set_color(PURE_GREEN).set_opacity(1)
        vect_1.set_z_index(4)
        vect_1_lbl=MathTex("u").scale(.85).next_to(vect_1,0.5*DOWN).set_color(PURE_GREEN)

        vect_1_moving=Arrow(start=line_1.n2p(-1),end=line_1.n2p(0),max_tip_length_to_length_ratio=0.125, buff=0).set_color(REANLEA_YELLOW).rotate(137.2*DEGREES,about_point=line_1.n2p(-1)).set_z_index(3)

        ang=Angle(vect_1, vect_1_moving, radius=0.5, other_angle=False).set_stroke(color=REANLEA_SAFRON_LIGHTER, width=3).set_z_index(-1)

        ang_lbl = MathTex(r"\theta =").move_to(
            Angle(
                vect_1, vect_1_moving, radius=.85 + 3 * SMALL_BUFF, other_angle=False
            ).point_from_proportion(.5)                         # Gets the point at a proportion along the path of the VMobject.
        ).scale(.5).set_color(REANLEA_YELLOW)

        ang_theta=DecimalNumber(137.29,unit="^o").scale(.5).set_color(REANLEA_YELLOW).next_to(ang_lbl).shift(.175*LEFT)

        projec_line=always_redraw(
            lambda : DashedLine(start=vect_1_moving.get_end(), end=np.array((vect_1_moving.get_end()[0],line_1.n2p(0)[1],0))).set_stroke(color=REANLEA_AQUA_GREEN, width=1)
        )

        bra_1=always_redraw(
            lambda : BraceBetweenPoints(
                point_1=vect_1.get_start(),
                point_2=np.array((vect_1_moving.get_end()[0],0,0)),
                direction=DOWN,
                color=REANLEA_YELLOW
            ).set_stroke(width=0.1).set_z_index(5)
        )

        vect_grp=VGroup(vect_1,vect_1_lbl,vect_1_moving,ang,ang_lbl,ang_theta,projec_line,bra_1)

        ang_theta_cos_lbl_left=MathTex("u","\cdot",r"cos(\theta)").arrange(RIGHT,buff=0.2).move_to(UP +RIGHT)
        ang_theta_cos_lbl_right=MathTex("\cdot","u").arrange(RIGHT, buff=0.2).set_color(PURE_GREEN).move_to(UP +4.55*RIGHT)
        ang_theta_cos=MathTex("=-0.735").arrange(RIGHT, buff=0.2).set_color(REANLEA_YELLOW).next_to(ang_theta_cos_lbl_right,LEFT).shift(.1*RIGHT)
        ang_theta_cos_lbl_left[0:2].set_color(PURE_GREEN)
        ang_theta_cos_lbl_left[2][0:3].set_color(REANLEA_WARM_BLUE)
        ang_theta_cos_lbl_left[2][4].set_color(REANLEA_SAFRON_LIGHTER)
        ang_theta_cos_grp=VGroup(ang_theta_cos_lbl_left,ang_theta_cos,ang_theta_cos_lbl_right).scale(.65)
        sur_ang_theta_cos_grp=SurroundingRectangle(ang_theta_cos_grp, color=REANLEA_TXT_COL,corner_radius=0.125, buff=0.2)

        ang_lbl=VGroup(ang_theta_cos_grp,sur_ang_theta_cos_grp)


        dot_1=Dot(radius=0.125, color=PURE_RED).move_to(line_1.n2p(-1)).set_sheen(-0.4,DOWN).set_opacity(1).set_z_index(5)
        dot_2=Dot(radius=0.2, color=REANLEA_VIOLET_LIGHTER).move_to(line_1.n2p(0)).set_sheen(-0.4,DOWN).set_z_index(3)
        glowing_circle_2=get_glowing_surround_circle(dot_2, color=REANLEA_YELLOW).set_z_index(-20)

        rays=get_rays(n=100,color=PURE_GREEN,factor=.75).move_to(dot_2.get_center()).set_z_index(4)

        dot_grp=VGroup(dot_1,dot_2,glowing_circle_2,rays)

        line_2= NumberLine(
            x_range=[-2, 2, 1],
            length=8,
            include_ticks=False,
        ).set_color(REANLEA_BLUE_LAVENDER).set_stroke(width=4)

        line_2_grp=VGroup(line_2).move_to(.75*UP+.5*RIGHT)

        grp_x=VGroup(mirror_1,vect_grp,dot_grp).move_to(.75*UP+.5*RIGHT).scale(1.5)
        mirror_1.scale(.67).shift(.05*RIGHT)

        self.add(grp_x,line_2_grp)

        # line_1,tick_grp,sgn_grp,so_on_txt_symbol,


    
    # manim -sqk banner.py esp_ex_2_1



class esp_ex_3_0(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("background_01.png").scale(.5).set_z_index(-100)
        self.add(water_mark)

        
        # MOBJECTS

        # Group-II

        grid = NumberPlane(axis_config={"include_tip":True},
            background_line_style={
                "stroke_color": BLUE,
                "stroke_width": 0.8,
                "stroke_opacity": 0.2
            }
        )


        r1 = lambda theta: 2 + 0.2 * np.sin(4*theta) + 0.01*theta*theta*(theta-2*np.pi)*(theta-2*np.pi)
        grph_1 = grid.plot_polar_graph(r1, [0, 2 * PI])
        grph_1.set_stroke(width=7, color=[REANLEA_GREEN_AUQA,REANLEA_SLATE_BLUE]).scale(.75).shift(4*LEFT).scale(.5).set_stroke(width=5),
        grph_1_lbl=MathTex("A").shift(grph_1.get_center()).set_color(REANLEA_GREEN_AUQA).shift(2*UP)

        r2 = lambda theta: 2 + 0.2 * np.cos(4*theta) + 0.01*theta*theta*(theta-2*np.pi)*(theta-2*np.pi)
        grph_2 = grid.plot_polar_graph(r2, [0, 2 * PI])
        grph_2.set_stroke(width=7, color=[REANLEA_SLATE_BLUE, REANLEA_BLUE_SKY]).scale(.75).shift(4.5*RIGHT).scale(.5).set_stroke(width=5),
        grph_2_lbl=MathTex("B").shift(grph_2.get_center()).set_color(REANLEA_BLUE_SKY).shift(2*UP)



        dot_1=Dot(radius=0.125, color=REANLEA_GREEN_AUQA).move_to(grph_1.get_center()).set_sheen(-0.6,DOWN)
        dot_1_lbl=MathTex("x").set_color(REANLEA_GREEN_AUQA).move_to(grph_1.get_center()+.5*DOWN).scale(.6)

        dot_2=Dot(radius=0.125, color=REANLEA_BLUE_SKY).move_to(grph_2.get_center()).set_sheen(-0.6,DOWN)
        dot_2_lbl=MathTex("y").set_color(REANLEA_BLUE_SKY).move_to(grph_2.get_center()+.5*DOWN).scale(.6)

        eqn_1=MathTex("A",r"\times","B","=",r"\{", r"(x,y)",r"\mid", r"x \in A",",", r"y \in B", r"\}").shift(2*DOWN)
        eqn_1[0].set_color(REANLEA_GREEN_AUQA)
        eqn_1[1].set_color(PURE_RED)
        eqn_1[2].set_color(REANLEA_BLUE_SKY)
        eqn_1[5:10].scale(.9)
        eqn_1[5][1].set_color(REANLEA_GREEN_AUQA)
        eqn_1[5][3].set_color(REANLEA_BLUE_SKY)
        eqn_1[7].set_color(REANLEA_GREEN_AUQA)
        eqn_1[7][1].set_color(PURE_RED).scale(.65)
        eqn_1[9].set_color(REANLEA_BLUE_SKY)
        eqn_1[9][1].set_color(PURE_RED).scale(.65)
        eqn_1[8:].shift(0.15*RIGHT)
        eqn_1[9:].shift(0.1*RIGHT)

        eqn_1[3:5].set_color(REANLEA_PURPLE_LIGHTER)
        eqn_1[5][0].set_color(REANLEA_PURPLE_LIGHTER)
        eqn_1[5][4].set_color(REANLEA_PURPLE_LIGHTER)
        eqn_1[6].set_color(REANLEA_PURPLE_LIGHTER)
        eqn_1[8].set_color(REANLEA_PURPLE_LIGHTER)
        eqn_1[10].set_color(REANLEA_PURPLE_LIGHTER)

        set_a_grp=VGroup(grph_1,grph_1_lbl,dot_1,dot_1_lbl).shift(2*RIGHT)
        set_b_grp=VGroup(grph_2,grph_2_lbl,dot_2,dot_2_lbl).shift(2*LEFT)
        cp_grp_1=VGroup(set_a_grp,set_b_grp,eqn_1).scale(.65).move_to(4.5*RIGHT)

    
        self.add(cp_grp_1)

         #Group-III

        tbl_AB=MathTable(
            [
                ["(a,1)", "(a,2)","(a,3)"],
                ["(b,1)", "(b,2)","(b,3)"]
            ]
        )
        tbl_AB.get_vertical_lines().set_stroke(width=2, color=REANLEA_BLUE_SKY)
        tbl_AB.get_horizontal_lines().set_stroke(width=2, color=REANLEA_SLATE_BLUE)
        ent_tbl_AB=tbl_AB.get_entries_without_labels()

        for k in range(len(ent_tbl_AB)):
            ent_tbl_AB[k][0][1].set_color(REANLEA_GREEN_AUQA)
            ent_tbl_AB[k][0][3].set_color(REANLEA_BLUE_SKY)

        tbl_AB_lbl=MathTex(r"A \times B").next_to(tbl_AB, 2.5*DOWN + RIGHT).scale(.75).set_color(PURE_RED)
        tbl_AB_lbl[0][0].set_color(REANLEA_GREEN_AUQA)
        tbl_AB_lbl[0][2].set_color(REANLEA_BLUE_SKY)
        tbl_AB_lbl_ln=Line().set_stroke(width=2,color=REANLEA_GREY).scale(.5).rotate(3*PI/4).next_to(tbl_AB_lbl, .5*UP+.5*LEFT).set_z_index(2)
        sr_tbl_AB=SurroundingRectangle(tbl_AB, color=REANLEA_WELDON_BLUE ,corner_radius=.25).set_fill(color=REANLEA_WELDON_BLUE, opacity=0.25)
        t_AB=VGroup(tbl_AB,tbl_AB_lbl,tbl_AB_lbl_ln,sr_tbl_AB)


        tbl_A=MathTable(
            [
              ["a"],
              ["b"]  
            ],
            v_buff=0.85
        ).next_to(tbl_AB, LEFT).set_color(REANLEA_GREEN_AUQA)
        tbl_A.get_horizontal_lines().set_opacity(0)

        tbl_A_lbl=MathTex("A").next_to(tbl_A, 2*UP+LEFT).scale(.75).set_color(REANLEA_GREEN_AUQA)
        tbl_A_lbl_ln=Line().set_stroke(width=2,color=REANLEA_GREY).scale(.5).rotate(3*PI/4).next_to(tbl_A_lbl, .5*DOWN+.5*RIGHT).set_z_index(2)
        sr_tbl_A=Ellipse(width=3, color=REANLEA_WELDON_BLUE).set_fill(color=REANLEA_GREEN_AUQA, opacity=0.15).move_to(tbl_A.get_center()).rotate(PI/2)

        t_A=VGroup(tbl_A,tbl_A_lbl,tbl_A_lbl_ln,sr_tbl_A)




        tbl_B=MathTable(
            [
                ["1","2","3"]
            ],
            h_buff=2.25
        ).next_to(tbl_AB,UP).set_color(REANLEA_BLUE_SKY)
        tbl_B.get_vertical_lines().set_opacity(0)

        tbl_B_lbl=Text("B").next_to(tbl_B, UP+4*RIGHT).scale(.75).set_color(REANLEA_BLUE_SKY)
        tbl_B_lbl_ln=Line().set_stroke(width=2,color=REANLEA_GREY).scale(.5).rotate(PI/4).next_to(tbl_B_lbl, .5*DOWN+.5*LEFT).set_z_index(2)
        sr_tbl_B=Ellipse(width=8, color=REANLEA_WELDON_BLUE).set_fill(color=REANLEA_BLUE_SKY, opacity=0.25).move_to(tbl_B.get_center())
                    
        t_B=VGroup(tbl_B,tbl_B_lbl,tbl_B_lbl_ln,sr_tbl_B)



        eqn_2_ref=MathTex("A","=",r"\{","2,4", r"\}").set_color(REANLEA_GREEN_AUQA).scale(.7).to_edge(LEFT, buff=.5).shift(2.5*UP)

        tbl_AB_ref=MathTable(
            [
                ["(2,1)", "(2,2)","(2,3)"],
                ["(4,1)", "(4,2)","(4,3)"]
            ]
        )
        tbl_AB_ref.get_vertical_lines().set_stroke(width=2, color=REANLEA_BLUE_SKY)
        tbl_AB_ref.get_horizontal_lines().set_stroke(width=2, color=REANLEA_SLATE_BLUE)
        ent_tbl_AB_ref=tbl_AB_ref.get_entries_without_labels()

        for k in range(len(ent_tbl_AB_ref)):
            ent_tbl_AB_ref[k][0][1].set_color(REANLEA_GREEN_AUQA)
            ent_tbl_AB_ref[k][0][3].set_color(REANLEA_BLUE_SKY)

        tbl_AB_lbl_ref=MathTex(r"A \times B").next_to(tbl_AB_ref, 2.5*DOWN + RIGHT).scale(.75).set_color(PURE_RED)
        tbl_AB_lbl_ref[0][0].set_color(REANLEA_GREEN_AUQA)
        tbl_AB_lbl_ref[0][2].set_color(REANLEA_BLUE_SKY)
        tbl_AB_lbl_ln_ref=Line().set_stroke(width=2,color=REANLEA_GREY).scale(.5).rotate(3*PI/4).next_to(tbl_AB_lbl_ref, .5*UP+.5*LEFT).set_z_index(2)
        sr_tbl_AB_ref=SurroundingRectangle(tbl_AB, color=REANLEA_WELDON_BLUE ,corner_radius=.25).set_fill(color=REANLEA_WELDON_BLUE, opacity=0.25)
        t_AB_ref=VGroup(tbl_AB_ref,tbl_AB_lbl_ref,tbl_AB_lbl_ln_ref,sr_tbl_AB_ref)


        tbl_A_ref=MathTable(
            [
              ["2"],
              ["4"]  
            ],
            v_buff=0.85
        ).next_to(tbl_AB_ref, LEFT).set_color(REANLEA_GREEN_AUQA)
        tbl_A_ref.get_horizontal_lines().set_opacity(0)

        tbl_A_lbl_ref=MathTex("A").next_to(tbl_A, 2*UP+LEFT).scale(.75).set_color(REANLEA_GREEN_AUQA)
        tbl_A_lbl_ln_ref=Line().set_stroke(width=2,color=REANLEA_GREY).scale(.5).rotate(3*PI/4).next_to(tbl_A_lbl, .5*DOWN+.5*RIGHT).set_z_index(2)
        sr_tbl_A_ref=Ellipse(width=3, color=REANLEA_WELDON_BLUE).set_fill(color=REANLEA_GREEN_AUQA, opacity=0.15).move_to(tbl_A.get_center()).rotate(PI/2)

        t_A_ref=VGroup(tbl_A_ref,tbl_A_lbl_ref,tbl_A_lbl_ln_ref,sr_tbl_A_ref)


        tbl_B_ref=MathTable(
            [
                ["1","2","3"]
            ],
            h_buff=2.25
        ).next_to(tbl_AB_ref,UP).set_color(REANLEA_BLUE_SKY)
        tbl_B_ref.get_vertical_lines().set_opacity(0)

        tbl_B_lbl_ref=Text("B").next_to(tbl_B_ref, UP+4*RIGHT).scale(.75).set_color(REANLEA_BLUE_SKY)
        tbl_B_lbl_ln_ref=Line().set_stroke(width=2,color=REANLEA_GREY).scale(.5).rotate(PI/4).next_to(tbl_B_lbl_ref, .5*DOWN+.5*LEFT).set_z_index(2)
        sr_tbl_B_ref=Ellipse(width=8, color=REANLEA_WELDON_BLUE).set_fill(color=REANLEA_BLUE_SKY, opacity=0.25).move_to(tbl_B.get_center())
                    
        t_B_ref=VGroup(tbl_B_ref,tbl_B_lbl_ref,tbl_B_lbl_ln_ref,sr_tbl_B_ref)

        tbl_grp=VGroup(t_AB,t_A,t_B).scale(0.75).shift(2*LEFT+DOWN)

        tbl_grp_ref=VGroup(t_AB_ref,t_A_ref,t_B_ref).scale(0.75).shift(2*LEFT+DOWN)

        eqn_3=MathTex("B","=",r"\{","1,2,3", r"\}").set_color(REANLEA_BLUE_SKY).scale(.7).to_edge(LEFT, buff=.5).shift(2*UP)

        cp_grp_2_ref=VGroup(eqn_2_ref,eqn_3,tbl_grp_ref)

        self.add(cp_grp_2_ref)
        

        # Group-IV

        sep_ln=Line().scale(2).rotate(PI/2).set_stroke(width=5, color=[REANLEA_MAGENTA,REANLEA_WARM_BLUE]).shift(1.5*RIGHT)

        self.add(sep_ln)

        self.play(
            cp_grp_1.animate.shift(1.5*UP),
            cp_grp_2_ref.animate.scale(.6).next_to(cp_grp_1, DOWN).shift(1.25*UP)
        )


        eqn_8=MathTex(r"\mathbb{R}",r"\times",r"\mathbb{R}",r"\times",r"\mathbb{R}","=",r"\{", r"(x,y,z)",r"\mid", r"x, y, z \in \mathbb{R}", r"\}")
        eqn_8.scale(.6).set_color_by_gradient(REANLEA_WARM_BLUE,REANLEA_AQUA).shift(0.5*UP).save_state()

        self.add(eqn_8)
        eqn_8.move_to(2.57049457*DOWN+2.62319193*LEFT)

        #STRIPE HEAD

        with RegisterFont("Montserrat") as fonts:
            txt_1=Text("C A R T E S I A N    P R O D U C T", font=fonts[0])#.to_edge(UP).shift(.5*DOWN)             # to_edge(UP) == move_to(3.35*UP)
            txt_1.set_color_by_gradient(REANLEA_BLUE_LAVENDER,REANLEA_SLATE_BLUE_LIGHTER).to_edge(UP).scale(.5)


        strp_1=get_stripe(factor=0.05, buff_max=5.2).move_to(3*UP+.2*RIGHT)

        self.add(strp_1)


        #BOX

        rect=Rectangle(height=4.5, width=8).scale(.8).shift(2.75*LEFT).set_stroke(opacity=1, color=[REANLEA_VIOLET,REANLEA_AQUA])

        self.play(
            Create(rect)
        )



    
    # manim -sqk banner.py esp_ex_3_0

config.background_color="#00003c"
class esp_ex_3_1(ThreeDScene):
    def construct(self):     

        #self.set_camera_orientation(phi=95 * DEGREES, theta=30 * DEGREES)

        # WATER MARK 

        '''water_mark=ImageMobject("esp_ex_3_0.png").scale(.5).set_z_index(-100)
        self.add(water_mark)'''
     
        # MOBJECTS

        ax_3 = ThreeDAxes(
            tips=False,
            axis_config={
                "font_size": 24,
                "include_ticks": False,
                #"stroke_width":4,
            }
        ).set_stroke(width=4, color=REANLEA_TXT_COL)

        x = MathTex(r"\mathbb{R}").next_to(ax_3, RIGHT).set_color(REANLEA_BLUE_LAVENDER).set_sheen(-.4, DOWN)
        y = MathTex(r"\mathbb{R}").next_to(ax_3, UP).set_color(REANLEA_BLUE_LAVENDER).set_sheen(-.4, DOWN)
        z = MathTex(r"\mathbb{R}").next_to(ax_3, OUT).set_color(REANLEA_BLUE_LAVENDER).set_sheen(-.4, DOWN)

        

        cube = Cube(side_length=3, fill_opacity=.45).set_color_by_gradient(REANLEA_BLUE_LAVENDER).scale(1.25)


        d_1=Dot3D(point=UP+RIGHT+OUT, color=REANLEA_BLUE_DARKER, resolution=[32,32])
        

        d_line_x=always_redraw(
            lambda : DashedLine(start=ax_3.c2p(0,0,0), end=[d_1.get_center()[0],0,0]).set_stroke(color=REANLEA_GREEN_DARKEST, width=1).set_z_index(7)
        )
        d_line_y=always_redraw(
            lambda : DashedLine(start=[d_1.get_center()[0],0,0], end=[d_1.get_center()[0],d_1.get_center()[1],0]).set_stroke(color=REANLEA_BLUE_ALT, width=1).set_z_index(7)
        )
        d_line_z=always_redraw(
            lambda : DashedLine(start=[d_1.get_center()[0],d_1.get_center()[1],0], end=[d_1.get_center()[0],d_1.get_center()[1],d_1.get_center()[2]]).set_stroke(color=REANLEA_BLUE, width=1).set_z_index(7)
        )

        d_line_grp=VGroup(d_line_x,d_line_y,d_line_z)

        grp=VGroup(ax_3,cube,d_1,d_line_x,d_line_y,d_line_z)

        ax_lbl=VGroup(x,y,z)
        


        self.play(
            Write(ax_3)
        )
        self.play(Write(ax_lbl))
        self.add_fixed_orientation_mobjects(x,y,z)

        

        self.play(
            FadeIn(d_1)
        )

        self.move_camera(phi=75* DEGREES, theta=30* DEGREES, zoom=1, run_time=1.5)


        self.play(
            Write(d_line_grp)
        )

        d_1_lbl=MathTex("(x,0,0)").next_to([d_1.get_center()[0],0,0], DOWN).scale(.3).shift(.1*LEFT).set_color(REANLEA_GREEN_DARKER)
        #d_1_lbl.rotate(PI/2, about_point=[d_1.get_center()[0],0,0], axis=RIGHT)

        d_2_lbl=MathTex("(x,y,0)").next_to([d_1.get_center()[0],d_1.get_center()[1],0], UP).set_color(REANLEA_WARM_BLUE).scale(.3)
        #d_2_lbl.rotate(PI/2, about_point=[0,d_1.get_center()[1],0], axis=RIGHT)

        d_3_lbl=MathTex("(x,y,z)").next_to([d_1.get_center()[0],d_1.get_center()[1],d_1.get_center()[2]], OUT).scale(.3).set_color(REANLEA_BLUE_SKY)
        #d_3_lbl.rotate(PI/2, about_point=d_1.get_center(), axis=RIGHT).rotate(-PI, axis=OUT)

        d_lbl_grp=VGroup(d_1_lbl,d_2_lbl,d_3_lbl)

        self.play(Write(d_lbl_grp))

        self.add_fixed_orientation_mobjects(d_1_lbl,d_2_lbl,d_3_lbl)    
        

        self.begin_ambient_camera_rotation(rate=0.35)

        self.play(
            Write(cube),
            d_1_lbl.animate.set_stroke(color=REANLEA_TXT_COL_DARKER),
            d_2_lbl.animate.set_color(REANLEA_BLUE_DARKER),
            d_3_lbl.animate.set_color(REANLEA_WARM_BLUE_DARKER),
            run_time=2
        )

        self.wait(3.5)


        # manim -sqk banner.py esp_ex_3_1

#config.background_color=WHITE
class esp_ex_4_0(Scene):
    def construct(self):
        

        ax_1=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                "include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(-2)

        ## vector field ##

        func = lambda x: x - ax_1.c2p(0,0)
        colors = [REANLEA_BLUE_LAVENDER,REANLEA_AQUA,PURE_GREEN]
        
        vf = ArrowVectorField(
            func, min_color_scheme_value=2, 
            max_color_scheme_value=10, 
            colors=colors
        ).set_z_index(-102)
       
        dots=VGroup()          
        for obj in vf:
            dots += Dot().move_to(obj.get_end()).set_color(obj.get_color()).scale(.75).set_sheen(-.4,DOWN)
        dots.set_z_index(-102)
        
        self.wait(10)

        self.play(
            Write(dots)
        )
        self.wait(10)

        self.play(
            Write(vf, run_time=2)
        )
        self.wait(10)

        self.play(
            FadeOut(dots)
        )
        self.wait(10)

        self.play(
            Write(dots)
        )

        self.wait(10)
        
        r_tot=Rectangle(width=16, height=9, color=REANLEA_BACKGROUND_COLOR).set_opacity(.25).set_z_index(-101)
        self.play(
            Create(r_tot),
            run_time=2
        )
        self.wait()

        def func(t):
            return [t,np.exp(1-t ** 2),0]
        
        f = ParametricFunction(func, t_range=np.array([-3, 3]), fill_opacity=0).set_stroke(width=25, color=[REANLEA_AQUA,REANLEA_BLUE_SKY,REANLEA_AQUA_GREEN]).scale(1.5).shift(1.5*DOWN)   # "#04ff9c","#1c00fb","#04ff9c"
        self.play(Write(f))

        # manim -sqk banner.py esp_ex_4_0
        

###################################################################################################################

class yt_banner(Scene):
    def construct(self):
        config.background_color="#000327" # secondary color : "#000658"

        '''water_mark=ImageMobject("yt_banner_1.png").scale(.3725).set_z_index(-10)
        self.add(water_mark)'''

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
            txt_1 = Text("Animated Science" , font=fonts[0]).set_color_by_gradient(REANLEA_TXT_COL_LIGHTER)

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
            txt_1 = Text("Animated Visuals of Mathematics and Physics with Geometric foundation." , font=fonts[0]).set_color_by_gradient(REANLEA_TXT_COL_LIGHTER)
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

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix" 