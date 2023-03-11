############################################# by GOBINDA CHANDRA ###################################################

                                    # VISIT    : https://reanlea.com/ 
                                    # YouTube  : https://www.youtube.com/Reanlea/ 
                                    # Twitter  : https://twitter.com/Reanlea_ 

####################################################################################################################

from __future__ import annotations
from ast import Return
from cProfile import label
from calendar import c
from difflib import restore


import fractions
from imp import create_dynamic
import math
from multiprocessing import context
from multiprocessing import dummy
from multiprocessing.dummy import Value
from numbers import Number
import sre_compile
from tkinter import Y, Label, font
from imp import create_dynamic
from tracemalloc import start
from turtle import degrees, width
from manim import*
from math import*
from manim_fonts import*
import numpy as np
import random
from sklearn.datasets import make_blobs
from reanlea_colors import*
from func import*
from PIL import Image
#from func import EmojiImageMobject

config.max_files_cached=500

config.background_color= REANLEA_BACKGROUND_COLOR


###################################################################################################################


class Scene1(Scene):
    def construct(self):
        
        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        self.wait()


        ## CUT - I  ##

        dt_1=Dot().set_color(REANLEA_AQUA).shift(2*LEFT+UP)
        dt_2=Dot().set_color(REANLEA_PURPLE).shift(2*RIGHT+DOWN)
        ln_1=Line(start=dt_1.get_center(), end=dt_2.get_center()).set_stroke(width=2, color=[dt_2.get_color(),dt_1.get_color()])
        self.play(
            Write(dt_1)
        )
        self.play(
            Write(dt_2)
        )
        self.wait()
        self.play(
            Write(ln_1)
        )
        self.wait(3)

        dts=VGroup(
            *[
                Dot().shift(i*0.2*RIGHT*np.random.uniform(-1,1)+i*0.2*UP*np.random.uniform(-1,1))
                for i in range(-15,15)
            ]
        )
        dts.set_color_by_gradient(REANLEA_SLATE_BLUE,REANLEA_MAGENTA,PURE_GREEN)

        self.play(
            Create(dts)
        )
        self.wait()

        lns=VGroup()
        for obj in dts:
            lns += Line(start=dt_1.get_center(), end=obj.get_center()).set_stroke(width=2, color=[ obj.get_color(),dt_1.get_color()])

        self.play(
            Write(lns),
            run_time=2
        )
        self.wait(4)

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

        self.play(
            Create(ax_1)
        )
        self.wait()

        self.play(
            Indicate(dt_2,scale_factor=1.75, color=PURE_RED)
        )
        dt_2_lbl=MathTex("(x,y)").scale(.5).set_color(REANLEA_PURPLE_LIGHTER).next_to(dt_2, RIGHT)
        self.play(
            Write(dt_2_lbl)
        )
        self.wait(2)

        self.play(
            Indicate(ln_1, color=PURE_RED)
        )

        dt_1_dummy=dt_1.copy().set_sheen(-.4,DOWN)
        self.play(
            dt_1_dummy.animate.move_to(dt_2.get_center()).set_color(dt_2.get_color()),
            run_time=2.5
        )
        self.wait(2)

        d_ln_x_1=DashedLine(start=ax_1.c2p(0,0), end=[dt_1_dummy.get_center()[0], ax_1.c2p(0,0)[1],0]).set_stroke(width=2, color=YELLOW)
        d_ln_x_1.add_updater(
            lambda z : z.become(
                DashedLine(start=ax_1.c2p(0,0), end=[dt_1_dummy.get_center()[0], ax_1.c2p(0,0)[1],0]).set_stroke(width=2, color=YELLOW)
            )
        )
        self.play(
            Write(d_ln_x_1)
        )


        d_ln_x_1_lbl=MathTex("x").scale(.35).set_color(REANLEA_YELLOW).next_to(d_ln_x_1,DOWN).set_z_index(5)
        d_ln_x_1_lbl.add_updater(
            lambda z : z.become(
                MathTex("x").scale(.35).set_color(REANLEA_YELLOW).next_to(d_ln_x_1,DOWN).set_z_index(5)
            )
        )
        self.play(
            TransformMatchingShapes(dt_2_lbl.copy()[0][1],d_ln_x_1_lbl)
        )



        d_ln_y_1=DashedLine(start=[dt_1_dummy.get_center()[0], ax_1.c2p(0,0)[1],0] , end= dt_2.get_center()).set_stroke(width=2, color=YELLOW)

        d_ln_y_1.add_updater(
            lambda z : z.become(
                DashedLine(start=[dt_1_dummy.get_center()[0], ax_1.c2p(0,0)[1],0] , end= dt_1_dummy.get_center()).set_stroke(width=2, color=YELLOW)
            )
        )
        self.play(
            Write(d_ln_y_1)
        )

        d_ln_y_1_lbl=MathTex("y").scale(.35).set_color(REANLEA_YELLOW).next_to(d_ln_y_1,LEFT).set_z_index(5)
        d_ln_y_1_lbl.add_updater(
            lambda z : z.become(
                MathTex("y").scale(.35).set_color(REANLEA_YELLOW).next_to(d_ln_y_1,LEFT).set_z_index(5)
            )
        )
        self.play(
            TransformMatchingShapes(dt_2_lbl.copy()[0][3],d_ln_y_1_lbl)
        )

        self.play(
            dt_1_dummy.animate.move_to(
                .5*dt_1.get_center()+.5*dt_2.get_center()
            )
        )
        self.play(
            dt_1_dummy.animate.move_to(
                .75*dt_1.get_center()+.25*dt_2.get_center()
            )
        )
        self.play(
            dt_1_dummy.animate.move_to(
                .25*dt_1.get_center()+.75*dt_2.get_center()
            )
        )
        self.wait(2)

        d_ln_lbl_grp=VGroup(d_ln_x_1_lbl,d_ln_y_1_lbl)

        eqn_1_0=MathTex("x",r"\longrightarrow","y").set_color(REANLEA_TXT_COL_LIGHTER).move_to(5*RIGHT+2.5*UP)
        eqn_1_0[1].scale(1.25)
        eqn_1_0[0].shift(.1*LEFT)
        eqn_1_0[2].shift(.1*RIGHT)

        eqn_1_1=MathTex(r"\rho").scale(.75).next_to(eqn_1_0[1],UP).shift(.25*DOWN)

        eqn_1=VGroup(eqn_1_0,eqn_1_1)


        self.play(
            TransformMatchingShapes(d_ln_lbl_grp.copy(),eqn_1)
        )
        self.wait()
        
        self.play(
            Indicate(ln_1, color=PURE_RED, scale_factor=1),
            run_time=3
        )
        self.wait()
        
        eqn_2=MathTex(r"\rho","(x)","=","y").move_to(5*RIGHT+2.5*UP)
        eqn_2.set_color(REANLEA_SLATE_BLUE_LIGHTEST)
        eqn_2[0].scale(1.25).shift(.15*LEFT)
        
 
        self.play(
            TransformMatchingShapes(eqn_1,eqn_2)
        )
        self.wait(2)

        self.play(
            Indicate(dt_1, color=PURE_RED),
            Indicate(dt_2, color=PURE_RED)
        )

        dt_1_lbl=MathTex(r"(x_{1},y_{1})").scale(.5).set_color(REANLEA_AQUA).next_to(dt_1, LEFT)
        dt_2_lbl_2=MathTex(r"(x_{2},y_{2})").scale(.5).set_color(REANLEA_PURPLE_LIGHTER).next_to(dt_2, RIGHT)

        self.play(
            TransformMatchingShapes(dt_2_lbl,dt_2_lbl_2),
            Write(dt_1_lbl)
        )
        self.play(
            Indicate(ln_1, color=PURE_RED, scale_factor=1)
        )

        dt_1_2_lbl_grp=VGroup(dt_1_lbl,dt_2_lbl_2,d_ln_lbl_grp)
        

        eqn_3=MathTex(
            r"\frac{y-y_{1}}{x-x_{1}}","=",r"\frac{y_{1}-y_{2}}{x_{1}-x_{2}}"
        ).scale(.75).move_to(5*RIGHT+2*UP)

        eqn_3[0][0].set_color(REANLEA_YELLOW)
        eqn_3[0][5].set_color(REANLEA_YELLOW)
        eqn_3[0][2:4].set_color(REANLEA_AQUA)
        eqn_3[0][7:9].set_color(REANLEA_AQUA)
        eqn_3[2][0:2].set_color(REANLEA_AQUA)
        eqn_3[2][6:8].set_color(REANLEA_AQUA)
        eqn_3[2][3:5].set_color(REANLEA_PURPLE_LIGHTER)
        eqn_3[2][9:11].set_color(REANLEA_PURPLE_LIGHTER)

        

        self.play(
            eqn_2.animate.scale(.5).shift(.5*UP), 
        )
        sr_eq_2=SurroundingRectangle(eqn_2, color=REANLEA_WELDON_BLUE, corner_radius=.125, buff=.15)
        self.play(
            Write(sr_eq_2)
        )

        self.play(
            TransformMatchingShapes(dt_1_2_lbl_grp.copy(), eqn_3)
        )
        self.wait()

        eqn_4=MathTex(
            "y","=",r"(\frac{y_{1}-y_{2}}{x_{1}-x_{2}})",r"(x-x_{1})","+","y_{1}"
        ).scale(.75).move_to(4.5*RIGHT+2*UP)
         
        eqn_4[0].set_color(REANLEA_YELLOW)
        eqn_4[3][1].set_color(REANLEA_YELLOW)
        eqn_4[2][1:3].set_color(REANLEA_AQUA)
        eqn_4[2][7:9].set_color(REANLEA_AQUA)
        eqn_4[2][4:6].set_color(REANLEA_PURPLE_LIGHTER)
        eqn_4[2][10:12].set_color(REANLEA_PURPLE_LIGHTER)
        eqn_4[3][3:5].set_color(REANLEA_AQUA)
        eqn_4[5].set_color(REANLEA_AQUA)


        self.play(
            TransformMatchingShapes(eqn_3,eqn_4)
        )
        self.wait()
        self.play(
            Indicate(
                lns, color=PURE_RED, scale_factor=1
            ),
            run_time=2
        )
        self.wait()
        self.play(
            Unwrite(dts),
            Unwrite(lns),
            FadeOut(eqn_2),
            FadeOut(sr_eq_2)
        )




        self.wait(4)





        # manim -pqh anim2.py Scene1

        # manim -pql anim2.py Scene1

        # manim -sqk anim2.py Scene1

        # manim -sql anim2.py Scene1




###################################################################################################################


class Scene2(Scene):
    def construct(self):
        
        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)


        ## PREVIOUS LAST SCENE  ##

        dt_1=Dot().set_color(REANLEA_AQUA).shift(2*LEFT+UP)
        dt_2=Dot().set_color(REANLEA_PURPLE).shift(2*RIGHT+DOWN)
        ln_1=Line(start=dt_1.get_center(), end=dt_2.get_center()).set_stroke(width=2, color=[dt_2.get_color(),dt_1.get_color()])
        ln_1.add_updater(
            lambda z : z.become(
                Line(start=dt_1.get_center(), end=dt_2.get_center()).set_stroke(width=2, color=[dt_2.get_color(),dt_1.get_color()])
            )
        )

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

        dt_1_lbl=MathTex(r"(x_{1},y_{1})").scale(.5).set_color(REANLEA_AQUA).next_to(dt_1, LEFT)
        dt_1_lbl.add_updater(
            lambda z : z.become(
                MathTex(r"(x_{1},y_{1})").scale(.5).set_color(REANLEA_AQUA).next_to(dt_1, LEFT).set_z_index(5)
            )
        )
        dt_2_lbl=MathTex(r"(x_{2},y_{2})").scale(.5).set_color(REANLEA_PURPLE_LIGHTER).next_to(dt_2, RIGHT)
        dt_2_lbl.add_updater(
            lambda z : z.become(
                MathTex(r"(x_{2},y_{2})").scale(.5).set_color(REANLEA_PURPLE_LIGHTER).next_to(dt_2, RIGHT).set_z_index(5)
            )
        )

        eqn_1=MathTex(
            "y","=",r"(\frac{y_{1}-y_{2}}{x_{1}-x_{2}})",r"(x-x_{1})","+","y_{1}"
        ).scale(.75).move_to(4.5*RIGHT+2*UP)
        
        eqn_1[0].set_color(REANLEA_YELLOW)
        eqn_1[3][1].set_color(REANLEA_YELLOW)
        eqn_1[2][1:3].set_color(REANLEA_AQUA)
        eqn_1[2][7:9].set_color(REANLEA_AQUA)
        eqn_1[2][4:6].set_color(REANLEA_PURPLE_LIGHTER)
        eqn_1[2][10:12].set_color(REANLEA_PURPLE_LIGHTER)
        eqn_1[3][3:5].set_color(REANLEA_AQUA)
        eqn_1[5].set_color(REANLEA_AQUA)

        grp_1_1=VGroup(dt_1,dt_1_lbl,dt_2,dt_2_lbl,ln_1,ax_1,eqn_1)
        
        dt_1_dummy=dt_1.copy().set_sheen(-.4,DOWN)

        dt_1_dummy.move_to(.25*dt_1.get_center()+.75*dt_2.get_center()).set_color(dt_2.get_color())

        dt_1_dummy.add_updater(
            lambda z : z.become(
                dt_1.copy().set_sheen(-.4,DOWN).move_to(.25*dt_1.get_center()+.75*dt_2.get_center()).set_color(dt_2.get_color())
            )
        )
        
        
        d_ln_x_1=DashedLine(start=ax_1.c2p(0,0), end=[dt_1_dummy.get_center()[0], ax_1.c2p(0,0)[1],0]).set_stroke(width=2, color=YELLOW)
        d_ln_x_1.add_updater(
            lambda z : z.become(
                DashedLine(start=ax_1.c2p(0,0), end=[dt_1_dummy.get_center()[0], ax_1.c2p(0,0)[1],0]).set_stroke(width=2, color=YELLOW)
            )
        )

        d_ln_x_1_lbl=MathTex("x").scale(.35).set_color(REANLEA_YELLOW).next_to(d_ln_x_1,DOWN).set_z_index(5)
        d_ln_x_1_lbl.add_updater(
            lambda z : z.become(
                MathTex("x").scale(.35).set_color(REANLEA_YELLOW).next_to(d_ln_x_1,DOWN).set_z_index(5)
            )
        )

        d_ln_y_1=DashedLine(start=[dt_1_dummy.get_center()[0], ax_1.c2p(0,0)[1],0] , end= dt_1_dummy.get_center()).set_stroke(width=2, color=YELLOW)

        d_ln_y_1.add_updater(
            lambda z : z.become(
                DashedLine(start=[dt_1_dummy.get_center()[0], ax_1.c2p(0,0)[1],0] , end= dt_1_dummy.get_center()).set_stroke(width=2, color=YELLOW)
            )
        )

        d_ln_y_1_lbl=MathTex("y").scale(.35).set_color(REANLEA_YELLOW).next_to(d_ln_y_1,LEFT).set_z_index(5)
        d_ln_y_1_lbl.add_updater(
            lambda z : z.become(
                MathTex("y").scale(.35).set_color(REANLEA_YELLOW).next_to(d_ln_y_1,LEFT).set_z_index(5)
            )
        )

        grp_1_2=VGroup(d_ln_x_1,d_ln_x_1_lbl,d_ln_y_1,d_ln_y_1_lbl)

        grp_1=VGroup(grp_1_1,dt_1_dummy,grp_1_2)

        self.add(grp_1)


        ## MAIN SCENE ##

        self.wait()

        self.play(
            Indicate(ln_1, color=PURE_RED, scale_factor=1)
        )

        bez_1=bend_bezier_arrow().scale(.55).set_color(REANLEA_TXT_COL).flip().rotate(-PI/2-PI/6).shift(3.325*UP+1.15*LEFT)

        self.play(
            Create(bez_1)
        )

        eqn_2=MathTex(
            "L",":=",r"\{","(x,y)",r"\in \mathbb{R}^{2}", r"\mid",  "y","=",r"(\frac{y_{1}-y_{2}}{x_{1}-x_{2}})",r"(x-x_{1})","+","y_{1}",r"\}"
        ).scale(.55).move_to(3.5*RIGHT+1.5*UP)
        
        eqn_2[3][1].set_color(REANLEA_YELLOW)
        eqn_2[3][3].set_color(REANLEA_YELLOW)
        eqn_2[6].set_color(REANLEA_YELLOW)
        eqn_2[9][1].set_color(REANLEA_YELLOW)
        eqn_2[8][1:3].set_color(REANLEA_AQUA)
        eqn_2[8][7:9].set_color(REANLEA_AQUA)
        eqn_2[8][4:6].set_color(REANLEA_PURPLE_LIGHTER)
        eqn_2[8][10:12].set_color(REANLEA_PURPLE_LIGHTER)
        eqn_2[9][3:5].set_color(REANLEA_AQUA)
        eqn_2[11].set_color(REANLEA_AQUA)

        ln_1_dummy=Line(start=dt_1.get_center(), end=dt_2.get_center()).scale(6).set_stroke(width=2, color=PURE_RED).set_z_index(-2)

        self.play(
            ReplacementTransform(eqn_1,eqn_2),
            FadeIn(ln_1_dummy),
        )
        self.wait(1.5)
        self.play(
            FadeOut(ln_1_dummy),
            run_time=.5
        )
        
        self.wait()

        self.play(
            Indicate(eqn_2[6:12], color=PURE_GREEN)
        )
        self.wait()

        self.play(
            Indicate(dt_1, color=PURE_RED),
            Indicate(dt_2, color=PURE_RED)
        )
        self.play(
            FadeOut(bez_1),
            eqn_2.animate.move_to(3*UP)
        )
        self.wait(2)

        self.play(
            dt_1.animate.move_to(ax_1.c2p(0,0)),
            dt_2.animate.move_to(ax_1.c2p(4,2)),
        )
        self.wait()

        dt_1_lbl_1=MathTex("(0,0)").scale(.5).next_to(ax_1.c2p(0,0)).shift(.35*DOWN).set_color(dt_1.get_color())
        dt_2_lbl_1=MathTex("(a,b)").scale(.5).move_to(dt_2_lbl.get_center()).set_color(dt_2_lbl.get_color())


        eqn_3=MathTex(
            "L",":=",r"\{","(x,y)",r"\in \mathbb{R}^{2}", r"\mid",  "y","=",r"(\frac{0-b}{0-a})",r"(x-0)","+","0",r"\}"
        ).scale(.55).move_to(3*UP)

        eqn_3[3][1].set_color(REANLEA_YELLOW)
        eqn_3[3][3].set_color(REANLEA_YELLOW)
        eqn_3[6].set_color(REANLEA_YELLOW)
        eqn_3[9][1].set_color(REANLEA_YELLOW)
        eqn_3[8][1].set_color(REANLEA_AQUA)
        eqn_3[8][5].set_color(REANLEA_AQUA)
        eqn_3[8][3].set_color(REANLEA_PURPLE_LIGHTER)
        eqn_3[8][7].set_color(REANLEA_PURPLE_LIGHTER)
        eqn_3[9][3].set_color(REANLEA_AQUA)
        eqn_3[11].set_color(REANLEA_AQUA)
        
        self.play(
            TransformMatchingShapes(dt_1_lbl,dt_1_lbl_1),
            TransformMatchingShapes(dt_2_lbl,dt_2_lbl_1),
            TransformMatchingShapes(eqn_2,eqn_3)
        )

        eqn_4=MathTex(
            "L",":=",r"\{","(x,y)",r"\in \mathbb{R}^{2}", r"\mid",  "y","=",r"(\frac{b}{a})",r"x",r"\}"
        ).scale(.55).move_to(3*UP)

        eqn_4[3][1].set_color(REANLEA_YELLOW)
        eqn_4[3][3].set_color(REANLEA_YELLOW)
        eqn_4[6].set_color(REANLEA_YELLOW)
        eqn_4[9].set_color(REANLEA_YELLOW)
        eqn_4[8][1].set_color(REANLEA_PURPLE_LIGHTER)
        eqn_4[8][3].set_color(REANLEA_PURPLE_LIGHTER)
        
        self.play(
            TransformMatchingShapes(eqn_3, eqn_4)
        )
        '''d_lns_1=ax_1.get_lines_to_point(ax_1.c2p(4,2)).set_color(REANLEA_VIOLET)

        self.play(
            Write(d_lns_1)
        )'''

        dt_1_dummy_lbl_1=MathTex(r"(x,(\frac{b}{a})x)").scale(.45).set_color(REANLEA_GOLD).next_to(dt_1_dummy, UP)
        dt_1_dummy_lbl_2=MathTex(r"(x,\mu x)").scale(.45).set_color(REANLEA_GOLD).next_to(dt_1_dummy, UP)

        eqn_4_1=MathTex(
            "L",":=",r"\{","(x,y)",r"\in \mathbb{R}^{2}", r"\mid",  "y","=",r"\mu",r"x",r"\}"
        ).scale(.55).move_to(3*UP)

        eqn_4_1[3][1].set_color(REANLEA_YELLOW)
        eqn_4_1[3][3].set_color(REANLEA_YELLOW)
        eqn_4_1[6].set_color(REANLEA_YELLOW)
        eqn_4_1[9].set_color(REANLEA_YELLOW)
        eqn_4_1[8].set_color(REANLEA_PURPLE_LIGHTER)

        eqn_4_2=MathTex(r"\mu","=",r"\frac{b}{a}").scale(.55).move_to(3*UP+3*RIGHT)
        eqn_4_2[0].set_color(REANLEA_GOLD)
        eqn_4_2[2].set_color(REANLEA_GOLD)
        

        self.play(
            Write(dt_1_dummy_lbl_1),
            
        )
        
        self.wait()
        self.play(
            TransformMatchingShapes(dt_1_dummy_lbl_1,dt_1_dummy_lbl_2),
            TransformMatchingShapes(eqn_4,eqn_4_1)
        )
        self.play(
            Write(eqn_4_2)
        )
        self.wait()
        
        d_ln_lbl_grp=VGroup(d_ln_x_1,d_ln_x_1_lbl,d_ln_y_1,d_ln_y_1_lbl)
        self.play(
            Uncreate(d_ln_lbl_grp)
        )

        dt_1_dummy_1=dt_1.copy().set_sheen(-.4,DOWN)

        dt_1_dummy_1.move_to(.25*dt_1.get_center()+.75*dt_2.get_center()).set_color(dt_2.get_color())

        dt_1_dummy_lbl_2.add_updater(
            lambda z : z.become(
                MathTex(r"(x,\mu x)").scale(.45).set_color(REANLEA_GOLD).next_to(dt_1_dummy_1, UP)
            )
        )

        self.add(dt_1_dummy_1)
        self.play(
            FadeOut(dt_1_dummy)
        )
        dt_1_dummy_1.set_z_index(2)

        self.play(
            dt_1_dummy_1.animate.move_to(
                .75*ax_1.c2p(0,0)+.25*ax_1.c2p(4,2)
            )
        )
        self.play(
            dt_1_dummy_1.animate.move_to(
                .05*ax_1.c2p(0,0)+.95*ax_1.c2p(4,2)
            )
        )
        self.play(
            dt_1_dummy_1.animate.move_to(
                .85*ax_1.c2p(0,0)+.15*ax_1.c2p(4,2)
            )
        )
        self.play(
            dt_1_dummy_1.animate.move_to(
                .5*ax_1.c2p(0,0)+.5*ax_1.c2p(4,2)
            )
        )
        self.play(
            dt_1_dummy_1.animate.move_to(
                .65*ax_1.c2p(0,0)+.35*ax_1.c2p(4,2)
            )
        )
        dt_3=dt_1.copy().set_color(PURE_GREEN).move_to(.65*ax_1.c2p(0,0)+.35*ax_1.c2p(4,2))
        dt_3.set_sheen(-.4,DOWN)

        self.add(dt_3)
        self.play(
            dt_1_dummy_1.animate.move_to(
                .25*ax_1.c2p(0,0)+.75*ax_1.c2p(4,2)
            )
        )

        dt_3_lbl=MathTex(r"(x',\mu x')").scale(.45).set_color(REANLEA_GOLD).next_to(dt_3,UP)
        self.play(
            Write(dt_3_lbl)
        )

        d_ln_2_x=DashedLine(start=ax_1.c2p(0,0), end=[dt_3.get_center()[0], ax_1.c2p(0,0)[1],0]).set_stroke(width=2, color=PURE_GREEN)

        d_ln_2_y=DashedLine(start=[dt_3.get_center()[0], ax_1.c2p(0,0)[1],0], end=dt_3.get_center()).set_stroke(width=2, color=PURE_GREEN)

        d_ln_2=VGroup(d_ln_2_x,d_ln_2_y)

        self.play(
            Write(d_ln_2),
            lag_ratio=1
        )

        dt_1_dummy_1.set_z_index(0)

        d_ln_3_x=DashedLine(start=ax_1.c2p(0,0), end=[dt_1_dummy_1.get_center()[0], ax_1.c2p(0,0)[1],0]).set_stroke(width=1.75, color=REANLEA_PINK_LIGHTER)

        d_ln_3_y=DashedLine(start=[dt_1_dummy_1.get_center()[0], ax_1.c2p(0,0)[1],0], end=dt_1_dummy_1.get_center()).set_stroke(width=1.75, color=REANLEA_PINK_LIGHTER)

        d_ln_3=VGroup(d_ln_3_x,d_ln_3_y)

        self.play(
            Write(d_ln_3),
            lag_ratio=1
        )

        d_ln_2_x_lbl=MathTex("x'").set_color(PURE_GREEN).scale(.45).move_to(ax_1.c2p(.7,-.25))
        d_ln_3_x_lbl=MathTex("x").set_color(REANLEA_PINK_LIGHTER).scale(.45).move_to(ax_1.c2p(2.2,-.28))

        d_ln_2_3_x_lbl=VGroup(d_ln_2_x_lbl,d_ln_3_x_lbl)

        self.play(
            dt_1_lbl_1.animate.shift(LEFT).scale(.8),
            Write(d_ln_2_x_lbl)
        )
        self.play(
            Write(d_ln_3_x_lbl)
        )
        self.wait()

        d_ln_2_x_dummy=d_ln_2_x.copy().set_stroke(width=3).move_to(4*RIGHT+2.25*DOWN)
        d_ln_3_x_dummy=d_ln_3_x.copy().set_stroke(width=3).move_to(4.685*RIGHT+2.6*DOWN)
        d_ln_2_3_x_dummy_grp=VGroup(d_ln_2_x_dummy,d_ln_3_x_dummy)
        d_ln_2_3_x_grp=VGroup(d_ln_2_x,d_ln_3_x)

        d_ln_2_x_dummy_lbl=MathTex(r"\zeta").scale(.65).set_color(PURE_GREEN).next_to(d_ln_2_x_dummy).shift(1.35*RIGHT)
        d_ln_3_x_dummy_lbl=MathTex(r"1").scale(.65).set_color(REANLEA_PINK_LIGHTER).next_to(d_ln_3_x_dummy)
        d_ln_2_3_x_dummy_lbl_grp=VGroup(d_ln_2_x_dummy_lbl,d_ln_3_x_dummy_lbl)

        



        eqn_5=MathTex("x'" ,"=",r"\zeta", "x").scale(1.05).set_color(REANLEA_BLUE_LAVENDER).move_to(4.75*LEFT+2.5*UP)

        self.play(
            TransformMatchingShapes(d_ln_2_3_x_lbl.copy(), eqn_5),
            TransformMatchingShapes(d_ln_2_3_x_grp.copy(), d_ln_2_3_x_dummy_grp)
        )
        self.play(            
            Write(d_ln_2_3_x_dummy_lbl_grp)
        )
        self.wait()

        eqn_6=MathTex(r"\Rightarrow ", r"\mu x'&",r"=\mu \zeta x\\ &",r"= \zeta \mu x",).scale(1.05).set_color(REANLEA_BLUE_LAVENDER).next_to(eqn_5,DOWN).shift(.35*LEFT)
        eqn_6[0].shift(.1*LEFT)

        self.play(
            AnimationGroup(
                *[Write(eq) for eq in eqn_6],
                lag_ratio=2
            )
        )

        eqn_5_6_grp=VGroup(eqn_5,eqn_6)
        sr_eqn_5_6_grp=SurroundingRectangle(eqn_5_6_grp, color=REANLEA_WELDON_BLUE, buff=.25, corner_radius=.125).set_opacity(.35).shift(.1*DOWN).set_z_index(-5)
        self.play(
            Write(sr_eqn_5_6_grp)
        )

        arr_1=Arrow(stroke_width=6,tip_length=0.1).rotate(-PI/2).next_to(sr_eqn_5_6_grp,DOWN).scale(.5).set_stroke(width=6, color=[REANLEA_BLUE,REANLEA_BLUE_SKY]).shift(.35*UP)
        self.play(
            Create(arr_1)
        )

        eqn_7=MathTex(r"(x',\mu x')&",r"=(\zeta x, \zeta \mu x)\\ &",r"= \zeta (x, \mu x)",).scale(.75).set_color(REANLEA_YELLOW_CREAM).next_to(arr_1,DOWN).shift(.25*DOWN)

        self.play(
            AnimationGroup(
                *[Write(eq) for eq in eqn_7],
                lag_ratio=2
            )
        )

        sr_eqn_7=SurroundingRectangle(eqn_7, color=REANLEA_BLUE_DARKER, buff=.15, corner_radius=.125).set_opacity(.25).set_z_index(-1)

        self.play(
            Write(sr_eqn_7)
        )

        ln_2=Line(start=dt_1.get_center(), end=dt_2.get_center()).set_stroke(width=2, color=PURE_RED).scale(6).set_z_index(-2)

        self.play(
            Write(ln_2),
            dt_1_lbl_1.animate.shift(.2*DOWN)
        )
        self.wait()

        dt_3_lbl_1=MathTex(r"\zeta (x, \mu x)").scale(.45).set_color(REANLEA_GOLD).next_to(dt_3,UP)
    
        self.play(
            TransformMatchingShapes(eqn_7[2][1:].copy(), dt_3_lbl_1),
            TransformMatchingShapes(dt_3_lbl,dt_3_lbl_1),
            lag_ratio=.1
        )
        

        dt_4=dt_1.copy().set_color(REANLEA_WARM_BLUE).set_sheen(-.4,DOWN).move_to(dt_1_dummy_1.get_center()).set_z_index(-1)
        gl_dt_4=get_glowing_surround_circle(dt_4, color=REANLEA_BLUE_LAVENDER)
        dt_4_grp=VGroup(dt_4,gl_dt_4)
        self.play(
            Create(dt_4_grp)
        )
        self.wait(2)
        self.play(
            dt_4_grp.animate.move_to(
                -.75*ax_1.c2p(0,0)+1.75*ax_1.c2p(4,2)
            )
        )
        dt_4_lbl=MathTex(r"\xi (x, \mu x)").scale(.45).set_color(REANLEA_GOLD).next_to(dt_4,DOWN)
        dt_4_lbl.add_updater(
            lambda z : z.become(
                MathTex(r"\xi (x, \mu x)").scale(.45).set_color(REANLEA_GOLD).next_to(dt_4,DOWN)
            )
        )
        self.wait()
        self.play(
            Write(dt_4_lbl)
        )
        self.wait(2)

        bez_2=bend_bezier_arrow().scale(.35).set_color(REANLEA_TXT_COL).rotate(-40*DEGREES).next_to(eqn_4_1[6], DOWN).shift(.2*RIGHT+.2*UP)

        self.play(
            Create(bez_2)
        )

        eqn_8=MathTex("f(x)","=",r"\mu x").scale(.65).set_color(REANLEA_WHITE).next_to(bez_2,RIGHT).shift(.3*DOWN + .15*LEFT)

        self.play(
            Write(eqn_8)
        )
        self.wait()

        eqn_8_1=MathTex("f(x')","=",r"\mu x'").scale(.75).shift(5*RIGHT).set_z_index(1)

        self.play(
            ReplacementTransform(eqn_8.copy(),eqn_8_1)
        )

        sr_eqn_8_1=SurroundingRectangle(eqn_8_1, color=REANLEA_MAGENTA, buff=.65, corner_radius=.125).set_opacity(.25)

        self.play(
            Write(sr_eqn_8_1)
        )
        self.wait()

        eqn_8_2=MathTex(r"f(\zeta x)","=",r"\zeta \mu x").scale(.75).shift(5*RIGHT).set_z_index(1)

        self.play(
            TransformMatchingShapes(eqn_8_1,eqn_8_2)
        )
        self.wait()

        eqn_8_3=MathTex(r"f(\zeta x)","=",r"\zeta f(x)").scale(.75).shift(5*RIGHT).set_z_index(1)

        self.play(
            TransformMatchingShapes(eqn_8_2,eqn_8_3)
        )


        dt_3_lbl_2=MathTex(r"v'").scale(.675).set_color(dt_3.get_color()).next_to(dt_3,UP)
        dt_3_lbl_2[0][0].set_stroke(width=1.025)
        

        dt_1_dummy_lbl_3=MathTex("v").scale(.675).set_color(dt_1_dummy_1.get_color()).next_to(dt_1_dummy_1,UP).set_stroke(width=1.025)

        self.play(
            ReplacementTransform(dt_3_lbl_1,dt_3_lbl_2),
            ReplacementTransform(dt_1_dummy_lbl_2,dt_1_dummy_lbl_3)
        )
        self.wait(2)

        eqn_9_pre_grp=VGroup(dt_3_lbl_2, dt_1_dummy_lbl_3)

        eqn_9=MathTex(r"v' = \zeta v").set_color(REANLEA_GOLD).shift(2.75*DOWN)
        eqn_9[0][0].set_stroke(width=1.025)
        eqn_9[0][-1].set_stroke(width=1.025)
        

        self.play(
            ReplacementTransform(eqn_9_pre_grp.copy(),eqn_9)
        )

        eqn_10=MathTex(r"v' - \zeta v","=","0").set_color(REANLEA_GOLD).shift(2.75*DOWN)
        eqn_10[0][0].set_stroke(width=1.025)
        eqn_10[0][-1].set_stroke(width=1.025)

        self.play(
            TransformMatchingShapes(eqn_9,eqn_10)
        )

        sr_eqn_10=SurroundingRectangle(eqn_10, color=REANLEA_WELDON_BLUE, buff=.2, corner_radius=.05)

        self.play(
            Write(sr_eqn_10)
        )
        self.wait(2)

        '''arr_2=Arrow(tip_length=.1).rotate(-20*DEGREES).next_to(sr_eqn_10,RIGHT).scale(.5).set_stroke(width=6, color=[REANLEA_BLUE,REANLEA_BLUE_SKY]).shift(.75*LEFT+.25*DOWN)
        self.play(
            Create(arr_2)
        )'''
        arr_2=MathTex(r"\longrightarrow").rotate(-20*DEGREES).next_to(sr_eqn_10,RIGHT).set_stroke(width=2, color=[REANLEA_BLUE,REANLEA_BLUE_SKY]).shift(.35*LEFT+.25*DOWN)
        self.play(
            Write(arr_2)
        )

        with RegisterFont("Homemade Apple") as fonts:
            txt_1=Text("Linearly Dependent", font=fonts[0]).scale(.5)
            txt_1.set_color_by_gradient(REANLEA_BLUE_LAVENDER,REANLEA_TXT_COL_LIGHTER)
            txt_1.next_to(arr_2,RIGHT).shift(.1*LEFT+.3*DOWN)
        
        self.play(
            Write(txt_1)
        )

        arr_3=MathTex(r"\longrightarrow").rotate(ln_2.get_angle()+ PI/2).set_stroke(width=2, color=[REANLEA_BLUE,REANLEA_BLUE_SKY]).shift(.05*UP)
        self.play(
            Write(arr_3)
        )
        
        eqn_11_pre_grp=VGroup(dt_1_dummy_lbl_3, eqn_4_1)

        eqn_11=MathTex(
            r"L := \{ \lambda v \mid \lambda \in \mathbb{R} \}"
        ).next_to(arr_3,UP).scale(.55).set_color_by_gradient(REANLEA_GREEN_AUQA).rotate(ln_2.get_angle()).shift(.35*LEFT+.25*DOWN) # .shift(.45*LEFT+1.05*DOWN) .shift(3*DOWN+3*LEFT)
        eqn_11[0][5].set_stroke(width=1.25).scale(1.35).shift(.035*UP)
        eqn_11[0][5:].shift(.05*RIGHT)


        self.play(
            ReplacementTransform(eqn_11_pre_grp.copy(),eqn_11)
        )
        eqn_11_1=eqn_11.copy()
        self.play(
            Write(eqn_11_1)
        )
        self.wait()

        
        self.wait()

        uncrt_grp=VGroup(
            eqn_8_3,sr_eqn_8_1,eqn_10,sr_eqn_10,arr_2,txt_1,arr_3,eqn_11,eqn_11_1,bez_2,eqn_8,eqn_5_6_grp,sr_eqn_5_6_grp,arr_1,eqn_7,sr_eqn_7,d_ln_2_3_x_dummy_grp,d_ln_2_3_x_dummy_lbl_grp,eqn_4_1,eqn_4_2,dt_4_lbl
        )
        self.play(
            FadeOut(uncrt_grp)
        )
        self.play(
            dt_4_grp.animate.move_to(1.5*UP)
        )

        dt_4_lbl_1=MathTex("u").scale(.675).set_color(REANLEA_GOLD).next_to(dt_4_grp,UP).set_stroke(width=1.025)

        self.play(
            Write(dt_4_lbl_1)
        )
        self.wait()


        arr_4=Arrow(start=ax_1.c2p(0,0),end=dt_4_grp.get_center(),tip_length=.125,stroke_width=4, buff=0).set_color(REANLEA_ORANGE).set_z_index(1)

        

        self.play(
            Create(arr_4)
        )

        self.wait(2)
 
        with RegisterFont("Nanum Pen Script") as fonts:
            vsp_ruls = VGroup(*[Text(x, font=fonts[0]) for x in (
                "I. Scalar Multiplication",
                "II. Vector Addition",
            )]).scale(0.65).arrange_submobjects(DOWN).shift(4.5*LEFT+2.25*UP)
            vsp_ruls[1].shift(.4*LEFT)

        self.play(
            Write(vsp_ruls[0])
        )
        self.wait()
        self.play(
            Write(vsp_ruls[1])
        )

        self.wait(2)
        self.play(
            FadeOut(vsp_ruls)
        )
        self.wait(2)

        self.play(
            Indicate(ln_2, color=PURE_GREEN),
            run_time=2.5
        )
        self.wait(2)

        ## emoji ##

        '''em = EmojiImageMobject("🚀").scale(.15).move_to(ax_1.c2p(0,0)).rotate(-20*DEGREES)
        self.play(
            FadeIn(em)
        )
        arr_5_1=Arrow(start=ax_1.c2p(0,0),end=dt_1_dummy_1.get_center(), buff=0.0, tip_length=.025).set_stroke(width=3, color=[REANLEA_YELLOW_GREEN])
        self.wait()
        self.play(
            em.animate.move_to(
                dt_1_dummy_1.get_center()+.1*UP
            )
        )
        self.wait()

        ln_3_dummy=Line(start=dt_1_dummy_1.get_center(), end=dt_4.get_center())
        self.play(
            em.animate.rotate(ln_3_dummy.get_angle()-ln_2.get_angle()).shift(.1*LEFT+.1*DOWN)
        )
        self.wait(2)

        self.play(
            em.animate.move_to(
                dt_4.get_center()
            ).shift(.05*LEFT+.05*DOWN)
        )'''

        ## emoji_end ##

        arr_5_1=Arrow(start=ax_1.c2p(0,0),end=dt_1_dummy_1.get_center(),tip_length=.125,stroke_width=4, buff=0).set_color(REANLEA_TXT_COL)

        arr_5_1_1=arr_5_1.copy().set_color(REANLEA_YELLOW_CREAM).set_z_index(2)

        arr_5_1_lbl=MathTex("v").scale(.675).set_color(REANLEA_TXT_COL).move_to(ax_1.c2p(1.85,.65)).set_stroke(width=1.025).set_z_index(5)

        arr_5_2=Arrow(start=dt_1_dummy_1.get_center(), end=dt_4.get_center(),tip_length=.125,stroke_width=4, buff=0).set_color(dt_4.get_color())

        arr_5_2_lbl=MathTex("w").scale(.675).next_to(arr_5_2, RIGHT).set_color(dt_4.get_color()).set_stroke(width=1.025).shift(.6*LEFT+.1*UP)        # .move_to(ax_1.c2p(2.65,2.65))

        em = EmojiImageMobject("🚀").scale(.15).move_to(ax_1.c2p(0,0)).rotate(-20*DEGREES).set_z_index(5)
        self.play(
            FadeIn(em)
        )
        

        self.play(
            Create(arr_5_1, run_time=2),
            Write(arr_5_1_lbl),
            em.animate.move_to(
                dt_1_dummy_1.get_center()+.1*UP
            )
        )
        
        ln_3_dummy=Line(start=dt_1_dummy_1.get_center(), end=dt_4.get_center())
        self.play(
            em.animate.rotate(ln_3_dummy.get_angle()-ln_2.get_angle()).shift(.1*LEFT+.1*DOWN)
        )
        
        self.play(
            Create(arr_5_2, run_time=2),
            Write(arr_5_2_lbl),
            em.animate.move_to(
                dt_4.get_center()
            ).shift(.05*LEFT+.05*DOWN)
        )

        self.wait(3)

        self.play(
            FadeOut(em)
        )

        d_arr_5_1=DashedArrow(start=ax_1.c2p(0,0),end=dt_1_dummy_1.get_center(),
        dash_length=2.0,stroke_width=2.5, tip_length=0.1, buff=0).set_color(REANLEA_TXT_COL).set_z_index(-1)

        d_arr_5_1_lbl=MathTex("v").scale(.675).set_color(REANLEA_TXT_COL).move_to(ax_1.c2p(1.85,.65)).set_stroke(width=1.025).set_z_index(5)

        d_arr_5_2=DashedArrow(start=dt_1_dummy_1.get_center(), end=dt_4.get_center(),dash_length=2.0,stroke_width=2.5, tip_length=0.1, buff=0).set_color(dt_4.get_color()).set_z_index(-1)

        d_arr_5_2_lbl=MathTex("w").scale(.675).next_to(arr_5_2, RIGHT).set_color(dt_4.get_color()).set_stroke(width=1.025).shift(.6*LEFT+.1*UP)

        self.add(d_arr_5_1,d_arr_5_2,d_arr_5_1_lbl, d_arr_5_2_lbl)

        x_a=ax_1.c2p(0,0)
        x_b=dt_1_dummy_1.get_center()
        x_c=dt_4.get_center()

        d_arr_5_2.set_z_index(0)
        d_arr_5_1.set_z_index(0)

        self.wait(2)

        self.play(
            d_arr_5_2.animate.shift((x_b[0]-x_a[0])*LEFT+(x_b[1]-x_a[1])*DOWN),
            d_arr_5_2_lbl.animate.shift((x_b[0]-x_a[0])*LEFT+(x_b[1]-x_a[1])*DOWN).shift(.4*LEFT+.25*DOWN),
            run_time=2
        )

        self.wait()

        self.play(
            d_arr_5_1.animate.shift((x_c[1]-x_b[1])*UP+(x_b[0]-x_c[0])*LEFT),
            d_arr_5_1_lbl.animate.shift((x_c[1]-x_b[1])*UP+(x_b[0]-x_c[0])*LEFT).shift(.25*UP+.5*LEFT),
            run_time=2
        )

        self.wait(2)

        self.play(
            Indicate(arr_5_1),
            Indicate(arr_5_2),
            run_time=2
        )
        self.wait(2)

        arr_5_2_lbl.add_updater(
            lambda z : z.become(
                MathTex("w").scale(.675).next_to(arr_5_2, RIGHT).set_color(dt_4.get_color()).set_stroke(width=1.025).shift(.6*LEFT+.1*UP) 
            )
        )

        dt_5=dt_1.copy().set_color(REANLEA_PINK).set_sheen(-.4,DOWN).move_to(ax_1.c2p(4,5))

        dt_5_lbl=MathTex(r"u_{1}").scale(.675).set_color(REANLEA_PINK).next_to(dt_5,UP).set_stroke(width=1.025)

        self.play(
            Create(dt_5)
        )
        self.wait()

        self.play(
            Write(dt_5_lbl)
        )
        self.wait(2)

        arr_6=Arrow(start=ax_1.c2p(0,0),end=dt_5.get_center(),tip_length=.125,stroke_width=4, buff=0).set_color(REANLEA_PINK_DARKER).set_z_index(1)

        self.play(
            Create(arr_6),
            run_time=1.5
        )
        self.wait(2)

        dt_6_dummy=dt_1.copy().set_color(REANLEA_PINK).set_sheen(-.4,DOWN).move_to(ax_1.c2p(5.325,2.6625))

        arr_6_1=Arrow(start=ax_1.c2p(0,0),end=dt_6_dummy.get_center(),tip_length=.125,stroke_width=4, buff=0).set_color(REANLEA_TXT_COL).set_z_index(1)

        arr_6_1_lbl=MathTex(r"v_{1}").scale(.675).set_color(REANLEA_TXT_COL).move_to(ax_1.c2p(3.55,1.35)).set_stroke(width=1.025).set_z_index(5)

        arr_6_2=Arrow(start=dt_6_dummy.get_center(),end=dt_5.get_center(),tip_length=.125,stroke_width=4, buff=0).set_color(dt_4.get_color()).set_z_index(1)

        arr_6_2_lbl=MathTex(r"w_{1}").scale(.675).next_to(arr_6_2, RIGHT).set_color(dt_4.get_color()).set_stroke(width=1.025).shift(.6*LEFT+.1*UP)

        arr_6_lbl_grp=VGroup(arr_6_1_lbl,arr_6_2_lbl)
        arr_6_grp=VGroup(arr_6_1,arr_6_2)

        x_d=dt_6_dummy.get_center()

        arr_5_2_dummy=arr_5_2.copy().shift((x_d[0]-x_b[0])*RIGHT+(x_d[1]-x_b[1])*UP)

        self.play(
            ReplacementTransform(arr_5_1,arr_6_1),
            ReplacementTransform(arr_5_2,arr_5_2_dummy),
            TransformMatchingShapes(arr_5_1_lbl,arr_6_1_lbl),
            run_time=2
        )

        self.play(
            ReplacementTransform(arr_5_2_dummy, arr_6_2),
            TransformMatchingShapes(arr_5_2_lbl,arr_6_2_lbl)
        )

        self.wait(2)

        dt_5_lbl_1=MathTex("=",r"v_{1}","+",r"w_{1}").scale(.675).set_color(REANLEA_PINK).next_to(dt_5_lbl,RIGHT).set_stroke(width=1.025).shift(.1*LEFT)
        dt_5_lbl_1[1].set_color(REANLEA_TXT_COL)
        dt_5_lbl_1[3].set_color(REANLEA_WARM_BLUE)

        self.play(
            TransformMatchingShapes(arr_6_lbl_grp.copy(),dt_5_lbl_1)
        )
        self.wait(2)

        dt_5_lbl_2=MathTex("=",r"\lambda_{1} v","+",r"\lambda_{2} w").scale(.675).set_color(REANLEA_PINK).next_to(dt_5_lbl,RIGHT).set_stroke(width=1.025).shift(.1*LEFT)
        dt_5_lbl_2[1][-1].set_color(REANLEA_TXT_COL)
        dt_5_lbl_2[3][-1].set_color(REANLEA_WARM_BLUE)

        self.play(
            ReplacementTransform(dt_5_lbl_1,dt_5_lbl_2)
        )
        self.wait()

        dt_5_lbl_3=MathTex("=",r"(\lambda_{1}",",",r"\lambda_{2})").scale(.675).set_color(REANLEA_PINK).next_to(dt_5_lbl,RIGHT).set_stroke(width=1.025).shift(.1*LEFT)

        self.play(
            ReplacementTransform(dt_5_lbl_2,dt_5_lbl_3)
        )

        self.wait(2)


        ## vector field ##

        func = lambda x: x - ax_1.c2p(0,0)
        colors = [REANLEA_BLUE_LAVENDER,REANLEA_SLATE_BLUE,REANLEA_AQUA,REANLEA_GREY]
        
        vf = ArrowVectorField(
            func, min_color_scheme_value=2, 
            max_color_scheme_value=10, 
            colors=colors
        ).set_z_index(-102)
       
        dots=VGroup()          
        for obj in vf:
            dots += Dot().move_to(obj.get_end()).set_color(obj.get_color()).scale(.75).set_sheen(-.4,DOWN)
        dots.set_z_index(-102)
        
        self.wait(2)

        self.play(
            Write(dots)
        )
        self.wait(2)

        self.play(
            Write(vf, run_time=2)
        )
        self.wait(2)

        self.play(
            FadeOut(dots)
        )
        self.wait(2)

        self.play(
            Write(dots)
        )

        self.wait(2)
        
        r_tot=Rectangle(width=16, height=9, color=REANLEA_BLUE_DARKEST).set_opacity(.65).set_z_index(-101)
        self.play(
            Create(r_tot),
            FadeOut(arr_6_lbl_grp),
            run_time=2
        )
        self.wait(2)

        ## vector field end ##


        eqn_12=MathTex(r"v' - \zeta v","=","0").set_color(REANLEA_BLUE_LAVENDER).shift(2.75*DOWN)
        eqn_12[0][0:2].set_stroke(width=1.025).set_color(REANLEA_GREEN)
        eqn_12[0][-1].set_stroke(width=1.025).set_color(REANLEA_PURPLE)

        self.play(
            Create(arr_5_1_1),
            Write(eqn_12)
        )
        self.wait()

        ln_3=Line().rotate(PI/2).next_to(eqn_12,RIGHT).shift(RIGHT).scale(.25).set_stroke(width=4, color=[REANLEA_BLUE_SKY,REANLEA_AQUA_GREEN])
        eqn_12_1=MathTex(r"\zeta \neq 0").set_color(REANLEA_BLUE_LAVENDER).next_to(eqn_12,RIGHT).shift(2.5*RIGHT)

        eqn_12_0_3=eqn_12[0][3]

        self.play(
            Indicate(eqn_12[0][3]),
            Write(ln_3),
            ReplacementTransform(eqn_12_0_3.copy(), eqn_12_1)
        )
        self.wait(2)

        eqn_13=MathTex(r"\lambda v' + \gamma v","=","0").set_color(REANLEA_BLUE_LAVENDER).shift(2.75*DOWN)
        eqn_13[0][1:3].set_stroke(width=1.025).set_color(REANLEA_GREEN)
        eqn_13[0][-1].set_stroke(width=1.025).set_color(REANLEA_PURPLE)

        eqn_13_1=MathTex(r"(\lambda, \gamma) \neq (0,0)").set_color(REANLEA_BLUE_LAVENDER).next_to(eqn_12,RIGHT).shift(2*RIGHT)

        self.play(
            ReplacementTransform(eqn_12,eqn_13),
            ReplacementTransform(eqn_12_1,eqn_13_1)
        )
        self.wait()

        self.play(
            Indicate(eqn_13[0][0]),
            Indicate(eqn_13[0][4])        
        )

        self.wait(2)

        eqn_13_grp=VGroup(eqn_13,eqn_13_1,ln_3)

        eqn_14=eqn_13.copy().scale(.625).move_to(6*LEFT+2.75*UP)

        des_tree=create_des_tree().scale(.75).next_to(eqn_14)        

        self.play(
            ReplacementTransform(eqn_13_grp,eqn_14)
        )

        self.play(
            Write(des_tree)
        )


        with RegisterFont("Nanum Pen Script") as fonts:
            lin_dep_indep = VGroup(*[Text(x, font=fonts[0]) for x in (
                "for some",
                "only for",
            )]).scale(0.35).arrange_submobjects(DOWN).shift(4.5*LEFT+2.5*UP)
            lin_dep_indep[0].next_to(des_tree).shift(.75*UP)
            lin_dep_indep[1].next_to(des_tree).shift(.75*DOWN)


        self.play(
            Write(lin_dep_indep[0])
        )

        eqn_15_1=MathTex(r"(\lambda, \gamma) \neq (0,0)").set_color(REANLEA_BLUE_SKY).set_stroke(width=1.1, color=REANLEA_BLUE_SKY).scale(.5).next_to(lin_dep_indep[0])

        self.play(
            Write(eqn_15_1)
        )


        with RegisterFont("Homemade Apple") as fonts:
            txt_2=Text("Linearly Dependent", font=fonts[0]).scale(.3)
            txt_2.set_color_by_gradient(REANLEA_TXT_COL)
            txt_2.next_to(lin_dep_indep[0],DOWN).shift(.75*RIGHT+.1*UP)

        self.play(
            Write(txt_2)
        )

        lin_grp_0=VGroup(txt_2,lin_dep_indep[0],eqn_15_1)

        sr_lin_grp_0=SurroundingRectangle(lin_grp_0, buff=.125, corner_radius=.05).set_stroke(width=0).set_fill(color=REANLEA_MAGENTA, opacity=.25).set_z_index(-1)

        self.play(
            Create(sr_lin_grp_0)
        )


        with RegisterFont("Homemade Apple") as fonts:
            txt_3=Text("Linearly Independent", font=fonts[0]).scale(.25)
            txt_3.set_color_by_gradient(REANLEA_TXT_COL)
            txt_3.next_to(lin_dep_indep[1],UP).shift(.75*RIGHT+.2*DOWN)

        eqn_15_2=MathTex(r"(\lambda, \gamma) = (0,0)").set_color(REANLEA_BLUE_SKY).set_stroke(width=1.1, color=REANLEA_BLUE_SKY).scale(.5).next_to(lin_dep_indep[1])

        
        lin_grp_1=VGroup(txt_3,lin_dep_indep[1],eqn_15_2)

        sr_lin_grp_1=SurroundingRectangle(lin_grp_1, buff=.125, corner_radius=.05).set_stroke(width=0).set_fill(color=REANLEA_MAGENTA, opacity=.25).set_z_index(-1)

        self.play(
            Create(sr_lin_grp_1)
        )

        self.play(
            Write(txt_3)
        )

        self.play(
            Write(lin_dep_indep[1])
        )

        self.play(
            Write(eqn_15_2)
        )

        self.wait(2)

        arr_7_1=arr_5_1_1.copy()

        self.play(
            arr_7_1.animate.shift(DOWN+RIGHT)
        )

        arr_7_2=arr_7_1.copy()

        self.play(
            arr_7_2.animate.shift(.5*DOWN+RIGHT)
        )

        arr_7_3=arr_7_2.copy()

        self.play(
            arr_7_3.animate.shift(.25*DOWN+1.25*RIGHT)
        )

        arr_7_4=arr_7_3.copy()

        self.play(
            arr_7_4.animate.shift(.125*UP+2.25*RIGHT)
        )

        self.wait(2)

        self.play(
            arr_7_3.animate.rotate(PI).set_color(REANLEA_YELLOW_DARKER)
        )

        self.wait(3)


        self.play(
            arr_7_1.animate.move_to(arr_5_1_1.get_center()),
            arr_7_2.animate.move_to(arr_5_1_1.get_center()),
            arr_7_3.animate.move_to(arr_5_1_1.get_center()).shift((dt_1_dummy_1.get_center()[0]-ax_1.c2p(0,0)[0])*LEFT+(dt_1_dummy_1.get_center()[1]-ax_1.c2p(0,0)[1])*DOWN),
            arr_7_4.animate.move_to(arr_5_1_1.get_center())
        )

        self.wait(2)

        self.play(
            FadeOut(arr_5_1_1),
            FadeOut(arr_7_2),
            FadeOut(arr_7_4),
        )

        with RegisterFont("Courier Prime") as fonts:
            txt_4 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Pair of Vectors",
                "parallel to each",
                "other are linearly",
                "D E P E N D E N T"
            )]).arrange_submobjects(DOWN).scale(0.4).set_color(REANLEA_GREY)
            txt_4.move_to(5*RIGHT + 2.25*DOWN)

            txt_5 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Pair of Vectors not",
                "parallel to each",
                "other are linearly",
                "I N D E P E N D E N T"
            )]).arrange_submobjects(DOWN).scale(0.4).set_color(REANLEA_GREY)
            txt_5.move_to(5*RIGHT + 2.25*DOWN)

            txt_6_1 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "there are atmost two ",
                "linearly independent ",
                "vectors on , based on",
                "them we can generate",
                "all the vectors of"
            )]).arrange_submobjects(DOWN).scale(0.4).set_color(REANLEA_GREY)
            txt_6_1.move_to(4.85*RIGHT + 2.25*DOWN)
            txt_6_1[2][9:].shift(.35*RIGHT)
        
        txt_6_2=MathTex(r"\mathbb{R}^2").scale(0.6).set_color(REANLEA_GREY).next_to(txt_6_1[2][9]).shift(.65*LEFT+.1*UP)

        txt_6_3=MathTex(r"\mathbb{R}^2").scale(0.6).set_color(REANLEA_GREY).next_to(txt_6_1[4]).shift(.1*LEFT+.025*UP)

        txt_6=VGroup(txt_6_1,txt_6_2,txt_6_3)

        

        


        self.wait()
        self.play(
            Write(txt_4)
        )
        self.wait()
        self.play(
            dt_1.animate.set_z_index(4),
            dt_2.animate.set_z_index(4),
            dt_1_dummy_1.animate.set_z_index(2),
            arr_7_1.animate.set_z_index(3),
            dt_3.animate.set_z_index(4),
            d_ln_2_y.animate.set_z_index(5),
            d_ln_3_y.animate.set_z_index(5),
            ReplacementTransform(txt_4,txt_5)
        )
        self.wait()
        self.play(
            FadeOut(txt_5)
        )
        self.wait(3)

        self.play(
            FadeIn(txt_6)
        )
        self.wait(2)

        arr_8_1=Arrow(start=ax_1.c2p(0,0),end=ax_1.c2p(1,0),tip_length=.125,stroke_width=4, buff=0).set_color(REANLEA_CYAN_LIGHT)

        arr_8_2=Arrow(start=ax_1.c2p(0,0),end=ax_1.c2p(0,1),tip_length=.125,stroke_width=4, buff=0).set_color(REANLEA_CYAN_LIGHT)

        arr_ax=VGroup(arr_8_1,arr_8_2).set_z_index(2)

        self.play(
            Create(arr_ax)
        )

        self.wait(2)

        arr_9=MathTex(r"\longrightarrow").move_to(5*RIGHT+DOWN).rotate(PI/3).set_stroke(width=2, color=[REANLEA_WARM_BLUE,REANLEA_PURPLE])

        self.play(
            Write(arr_9)
        )

        with RegisterFont("Reenie Beanie") as fonts:
            txt_7=Text("Basis", font=fonts[0]).scale(.75).set_color(REANLEA_CYAN_LIGHT).next_to(arr_9).shift(.65*UP+.65*LEFT)

        self.play(
            Write(txt_7)
        )

        self.wait(2)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_8=Text("Standard Basis", font=fonts[0]).scale(.75).set_color(REANLEA_CYAN_LIGHT).next_to(arr_9).shift(.65*UP+1.65*LEFT)


        self.play(
            ReplacementTransform(arr_ax.copy(),txt_8),
            TransformMatchingShapes(txt_7,txt_8),
        )

        undrln_txt_8=underline_bez_curve().scale(.7).next_to(txt_8,DOWN).shift(.185*UP).set_color(REANLEA_YELLOW_GREEN)

        self.play(
            Create(undrln_txt_8)
        )


        d_ln_4_x=DashedLine(start=ax_1.c2p(0,0), end=[dt_2.get_center()[0], ax_1.c2p(0,0)[1],0]).set_stroke(width=2, color=REANLEA_CYAN_LIGHT)

        d_ln_4_y=DashedLine(start=[dt_2.get_center()[0], ax_1.c2p(0,0)[1],0], end=dt_2.get_center()).set_stroke(width=2, color=REANLEA_CYAN_LIGHT)

        d_ln_4=VGroup(d_ln_4_x,d_ln_4_y).set_z_index(5)

        self.play(
            Create(d_ln_4),
            lag_ratio=1
        )

        self.wait(1.25)

        d_ln_4_x_lbl=MathTex("a").set_color(REANLEA_BLUE_SKY).scale(.45).move_to(ax_1.c2p(3.5,-.28))
        d_ln_4_y_lbl=MathTex("b").set_color(REANLEA_BLUE_SKY).scale(.45).move_to(ax_1.c2p(4.3,1))

        d_ln_4_lbl=VGroup(d_ln_4_x_lbl,d_ln_4_y_lbl)

        self.play(
            Write(d_ln_4_lbl),
            lag_ratio=1
        )

        self.wait(2)

        arr_10=bend_bezier_arrow().scale(.55).set_color(REANLEA_SAFRON_DARKER).flip().rotate(-PI/4).next_to(txt_8,LEFT).shift(.225*RIGHT +.1*UP)

        mtxt_1=MathTex("=","a","(1,0)","+","b","(0,1)").scale(.5).next_to(dt_2_lbl_1).set_color(REANLEA_CYAN_LIGHT)
        mtxt_1[1:].shift(.15*RIGHT)
        mtxt_1[1].set_color(REANLEA_BLUE_SKY)
        mtxt_1[4].set_color(REANLEA_BLUE_SKY)


        self.play(
            Create(arr_10, run_time=.75),
            Write(mtxt_1)
        )

        eqn_r2_1 = MathTex(
            r"\mathbb{R}^2&",r" := \{ (a,b)= a(1,0)+b(0,1) \mid a , b \in \mathbb{R} \} \\ &",r"= span \{ (1,0) , (0,1) \}",
        ).shift(3.15*DOWN).scale(.55).set_color(REANLEA_TXT_COL_LIGHTER)

        eqn_r2_1[1][2].set_color(REANLEA_TXT_COL).scale(1.2)
        eqn_r2_1[1][28].set_color(REANLEA_TXT_COL).scale(1.2)
        eqn_r2_1[1][4].set_color(REANLEA_BLUE_SKY)
        eqn_r2_1[1][23].set_color(REANLEA_BLUE_SKY)
        eqn_r2_1[1][25].set_color(REANLEA_BLUE_SKY)
        eqn_r2_1[1][6].set_color(REANLEA_BLUE_SKY)
        eqn_r2_1[1][9].set_color(REANLEA_BLUE_SKY)
        eqn_r2_1[1][16].set_color(REANLEA_BLUE_SKY)
        eqn_r2_1[1][10:15].set_color(REANLEA_CYAN_LIGHT)
        eqn_r2_1[1][17:22].set_color(REANLEA_CYAN_LIGHT)
        eqn_r2_1[1][22].set_color(PURE_GREEN).scale(1.5)

        eqn_r2_1[2][1:5].set_color(REANLEA_GREEN_AUQA).scale(.8)
        eqn_r2_1[2][6:11].set_color(REANLEA_CYAN_LIGHT)
        eqn_r2_1[2][12:17].set_color(REANLEA_CYAN_LIGHT)
        eqn_r2_1[2][5].set_color(REANLEA_TXT_COL).scale(1.2)
        eqn_r2_1[2][17].set_color(REANLEA_TXT_COL).scale(1.2)

        self.play(
            AnimationGroup(
                *[Write(eq) for eq in eqn_r2_1],
                lag_ratio=2
            )
        )

        self.wait(2)

        with RegisterFont("Courier Prime") as fonts:
            txt_9 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "if a vector space is",
                "generated by a basis",
                "having finite number of",
                "elements, then all the",
                "basis of that vector space",
                "contains the same number",
                "of vectors."
            )]).arrange_submobjects(DOWN).scale(0.35).set_color(REANLEA_GREY)
            txt_9.move_to(5*RIGHT + 2.25*DOWN)

        self.play(
            arr_9.animate.rotate(PI),
            ReplacementTransform(txt_6, txt_9)
        )

        self.wait(2)

        dim_r2=MathTex(
            "dim",r"(\mathbb{R}^2)","=","2"
        ).set_z_index(16).scale(2).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_MAGENTA)
        dim_r2[1:].shift(.1*RIGHT)

        r_tot_1=Rectangle(width=16, height=9, color=REANLEA_BACKGROUND_COLOR).set_opacity(.75).set_z_index(15)
        self.play(
            water_mark.animate.set_z_index(16),
        )
        self.play(
            FadeIn(r_tot_1),
            run_time=2
        )
        self.play(
            Write(dim_r2),
            lag_ratio=.7
        )

        b2=underline_bez_curve().next_to(dim_r2,DOWN).scale(2).set_z_index(16)
        self.play(
            Create(b2)
        )

        self.wait(2)


        '''self.play(
            *[FadeOut(mobj) for mobj in self.mobjects],
            run_time=3
        ) '''

        self.play(
            r_tot_1.animate.set_opacity(1)
        )

        

        self.wait(4)
        


        

        #self.wait(4)





        # manim -pqh anim2.py Scene2

        # manim -pql anim2.py Scene2

        # manim -sqk anim2.py Scene2

        # manim -sql anim2.py Scene2

        # manim -o myscene --format=gif -n 178,186 anim2.py Scene2



class Scene2_1(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        self.wait()

        ## PREVIOUS LAST SCENE  ##


        dim_r2=MathTex(
            "dim",r"(\mathbb{R}^2)","=","2"
        ).set_z_index(16).scale(2).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_MAGENTA)
        dim_r2[1:].shift(.1*RIGHT)
        
        b2=underline_bez_curve().next_to(dim_r2,DOWN).scale(2).set_z_index(16)

        grp_prv_scn=VGroup(dim_r2,b2)

        self.add(grp_prv_scn)



        # manim -pqh anim2.py Scene2_1

        # manim -pql anim2.py Scene2_1

        # manim -sqk anim2.py Scene2_1

        # manim -sql anim2.py Scene2_1

        
###################################################################################################################


class Scene3(Scene):
    def construct(self):
        
        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        self.wait()


        # SCENE

        ax_1=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(2)

        ax_2=ax_1.copy().set_z_index(-2)

        


        dt_1=Dot().set_color(REANLEA_AQUA).move_to(ax_1.c2p(0,0))
        dt_2=Dot().set_color(REANLEA_PURPLE).move_to(ax_1.c2p(3,2))
        dt_3=Dot().set_color(REANLEA_SLATE_BLUE).move_to(ax_1.c2p(3,0))
        ln_1=Line(start=dt_1.get_center(), end=dt_2.get_center()).set_stroke(width=5, color=[REANLEA_PURPLE,REANLEA_AQUA])
        ln_2=Line(start=dt_1.get_center(), end=dt_3.get_center()).set_stroke(width=5, color=[REANLEA_SLATE_BLUE,REANLEA_AQUA])
        ln_3=Line(start=dt_3.get_center(), end=dt_2.get_center()).set_stroke(width=5, color=[REANLEA_SLATE_BLUE,REANLEA_VIOLET])

        a_len=ax_1.c2p(3,0)[0]-ax_1.c2p(0,0)[0]
        b_len=ax_1.c2p(3,2)[1]-ax_1.c2p(3,0)[1]
        
        
        self.wait()
        self.play(
            Create(ln_1),
        )
        
        self.wait(3)

        self.play(
            Write(dt_1)
        )
        self.play(
            Write(dt_2)
        )

        self.wait(.5)

        self.play(
            Write(ax_2),
            run_time=2
        )

        self.wait(2)


        r_tot_1=Rectangle(width=16, height=9, color=REANLEA_BACKGROUND_COLOR).set_opacity(.75).set_z_index(2)
        self.play(
            water_mark.animate.set_z_index(3),
        )
        self.play(
            FadeIn(r_tot_1),
            run_time=2
        )

        with RegisterFont("Cousine") as fonts:
            txt_1=Text("Pythagoras Theorem", font=fonts[0])
            txt_1.set_color_by_gradient(REANLEA_TXT_COL_LIGHTER).set_z_index(4)

        self.play(
            Create(txt_1)
        )
        self.wait()

        self.play(
            txt_1.animate.scale(.5).shift(3*UP)
        )

        undr_bez=underline_bez_curve().scale(1.25).next_to(txt_1, DOWN).shift(.2*UP).set_z_index(4)

        self.play(
            Write(undr_bez)
        )

        hed_txt_bez=VGroup(txt_1,undr_bez)
    

        sym_1=Text("\" ").scale(3).set_color_by_gradient(REANLEA_SLATE_BLUE).set_z_index(4).move_to(5*LEFT+.75*UP).rotate(PI)
        self.play(
            Create(sym_1)
        )

        with RegisterFont("Cousine") as fonts:
            txt_2 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "In a right-angled triangle, the square of the hypotenuse ",
                "side is equal to the sum of squares of the other two sides."
            )]).arrange_submobjects(DOWN).scale(0.4).set_color(REANLEA_GREY).set_z_index(4)
            txt_2.move_to(ORIGIN)
        
        self.play(
            AddTextWordByWord(txt_2),
            run_time=6
        )
        self.wait(3)

        self.play(
            FadeOut(r_tot_1),
            FadeOut(txt_2),
            FadeOut(sym_1),
            hed_txt_bez.animate.scale(.6).move_to(5*LEFT+3*DOWN),
            water_mark.animate.set_z_index(-100)
        )
        self.wait(2)

        self.play(
            Create(ln_3.reverse_direction().set_z_index(-1))
        )
        self.play(
            Write(dt_3)
        )
        self.play(
            Create(ln_2)
        )

        self.wait(2)

        tr_angl=Polygon(dt_1.get_center(),dt_2.get_center(),dt_3.get_center()).set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA, REANLEA_SLATE_BLUE]).set_z_index(-1)

        self.play(
            FadeIn(tr_angl)
        )

        self.play(
            FadeOut(ln_1),
            FadeOut(ln_2),
            FadeOut(ln_3)
        )
        

        self.wait()

        self.play(
            tr_angl.animate.set_fill(opacity=1, color=REANLEA_BLUE)
        )
        '''self.play(
            tr_angl.animate.set_stroke(width=2, color=REANLEA_WHITE)
        )'''

        self.play(
            FadeOut(dt_1),
            FadeOut(dt_2),
            FadeOut(dt_3)
        )

        a_len_ln=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(3,0)).set_stroke(width=3, color=PURE_GREEN).save_state()
        b_len_ln=DashedLine(start=ax_1.c2p(3,0),end=ax_1.c2p(3,2)).set_stroke(width=3, color=PURE_RED).save_state()
        c_len_ln=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(3,2)).set_stroke(width=3, color=REANLEA_BLUE_DARKER)

        a_ln_lab=MathTex("a").scale(.65).set_color(PURE_GREEN).next_to(a_len_ln,DOWN)
        b_ln_lab=MathTex("b").scale(.65).set_color(PURE_RED).next_to(b_len_ln,RIGHT)
        c_ln_lab=MathTex("c").scale(.65).set_color(REANLEA_BLUE_SKY).move_to(ax_1.c2p(1.35,1.35))
        

        

        self.play(
            Create(c_len_ln),
            Write(c_ln_lab),
            lag_ratio=.95
        )
        self.play(
            Create(a_len_ln),
            Write(a_ln_lab),
            lag_ratio=.95
        )
        self.play(
            Create(b_len_ln),
            Write(b_ln_lab),
            lag_ratio=.95
        )

        self.wait()
        self.play(
            Unwrite(ax_2)
        )
        self.wait(2)

        a_len_ln_1=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(3,0)).set_stroke(width=3, color=PURE_GREEN).move_to(4.24*RIGHT+3.05*DOWN)
        b_len_ln_1=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(2,0)).set_stroke(width=3, color=PURE_RED).move_to(3.81*RIGHT+3.35*DOWN)
        c_len_ln_1=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(3.61,0)).set_stroke(width=3, color=REANLEA_BLUE_SKY).move_to(4.5*RIGHT+2.75*DOWN)

        a_ln_lab_1=MathTex("a").scale(.6).set_color(PURE_GREEN).next_to(a_len_ln_1,LEFT)
        b_ln_lab_1=MathTex("b").scale(.6).set_color(PURE_RED).next_to(b_len_ln_1,LEFT)
        c_ln_lab_1=MathTex("c").scale(.6).set_color(REANLEA_BLUE_SKY).next_to(c_len_ln_1,LEFT)

        c_len_ln.set_stroke(width=3, color=REANLEA_BLUE_SKY).save_state()

        self.play(
            ReplacementTransform(c_len_ln,c_len_ln_1),
            ReplacementTransform(c_ln_lab,c_ln_lab_1)
        )
        self.play(
            ReplacementTransform(a_len_ln,a_len_ln_1),
            ReplacementTransform(a_ln_lab,a_ln_lab_1)
        )
        self.play(
            ReplacementTransform(b_len_ln,b_len_ln_1),
            ReplacementTransform(b_ln_lab,b_ln_lab_1)
        )

        self.wait(2)
        
        
        tr_angl_0=tr_angl.copy().set_stroke(width=1).set_z_index(-6)
        tr_angl_1=tr_angl.copy().set_z_index(-2).save_state()
        

        self.add(tr_angl_1)

        self.play(
            tr_angl_1.animate.shift(3.5*RIGHT)
        )
        
        tr_angl_1_0=tr_angl_1.copy()
        self.add(tr_angl_1_0)

        tr_angl_1_ref=tr_angl_1.copy().save_state()

        tr_angl_2=tr_angl_1.copy().rotate(PI/2,about_point=ax_1.c2p(4,0))
        tr_angl_3=tr_angl_1.copy().rotate(PI,about_point=ax_1.c2p(4,0))
        tr_angl_4=tr_angl_1.copy().rotate(3*PI/2,about_point=ax_1.c2p(4,0))

        #self.add(tr_angl_3,tr_angl_4)
        


        

        rot_tracker=ValueTracker(0)

        tr_angl_1.add_updater(
            lambda x : x.become(tr_angl_1_ref.copy()).rotate(
                rot_tracker.get_value(), about_point=ax_1.c2p(4,0)
            )
        )
        self.play(
            rot_tracker.animate.set_value(PI/2)
        )
        
        self.play(
            FadeOut(tr_angl_1),
            FadeIn(tr_angl_2)
        )

        
        self.play(
            tr_angl_2.animate.shift((ax_1.c2p(4,0)-ax_1.c2p(3,0))[0]*LEFT)
        )
        self.play(
            tr_angl_2.animate.shift((ax_1.c2p(3,2)-ax_1.c2p(3,0))[1]*UP)
        )
        

        rot_tracker.set_value(0)
        self.play(
            FadeIn(tr_angl_1)
        )
        
        
        
        self.play(
            rot_tracker.animate.set_value(PI),
            run_time=1.5
        )
        self.play(
            FadeIn(tr_angl_3),
            FadeOut(tr_angl_1)
        )
        
        self.play(
            tr_angl_3.animate.shift((ax_1.c2p(1,5.0825)-ax_1.c2p(1,0))[1]*UP)
        )
        self.play(
            tr_angl_3.animate.shift((ax_1.c2p(4,0)-ax_1.c2p(1,0))[0]*LEFT)
        )
        
        

        rot_tracker.set_value(0)
        self.play(
            FadeIn(tr_angl_1),
            FadeOut(tr_angl_1_0)
        )
        
        
        
        self.play(
            rot_tracker.animate.set_value(-PI/2)
        )
        self.play(
            FadeIn(tr_angl_4),
            FadeOut(tr_angl_1)
        )

        self.play(
            tr_angl_4.animate.shift((ax_1.c2p(-2,3.0825)-ax_1.c2p(-2,0))[1]*UP)
        )
        self.play(
            tr_angl_4.animate.shift((ax_1.c2p(4,0)-ax_1.c2p(-2.0825,0))[0]*LEFT)
        )

        tr_angl_grp_1=VGroup(tr_angl,tr_angl_2,tr_angl_3,tr_angl_4)

        self.wait(1.5)
        
        sq_2=Square(side_length=(ax_1.c2p(1,0)[0]-ax_1.c2p(0,0)[0])*5).set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA, REANLEA_SLATE_BLUE]).set_fill(color=REANLEA_BLUE_LAVENDER, opacity=1).set_z_index(-5)

        sq_2.move_to(ax_1.c2p(6.5,2.5))

        sq_2_ref=sq_2.copy()


        triangles = [tr_angl_2.copy() for i in range(0, 8)]
        time = 0.3

        triangles[0].next_to(sq_2.get_corner(UR), DL, buff=0)
        triangles[1].rotate(PI/2).next_to(sq_2.get_corner(UL), DR, buff=0)
        triangles[2].rotate(PI).next_to(sq_2.get_corner(DL), UR, buff=0)
        triangles[3].rotate(-PI/2).next_to(sq_2.get_corner(DR), UL, buff=0)


        sq_2_grp=VGroup(triangles[0],triangles[1],triangles[2],triangles[3])

        self.play(
            TransformMatchingShapes(tr_angl_grp_1,sq_2_grp)
        )

        self.play(
            FadeIn(sq_2)
        )

        '''self.play(
            Unwrite(sq_2),
            Unwrite(sq_2_grp)
        )'''

        
        sq_2_grp_ref=VGroup(sq_2_ref, triangles[4].become(triangles[0]), triangles[5].become(triangles[1]), triangles[6].become(triangles[2])
        , triangles[7].become(triangles[3]))

        equal = MathTex("=").scale(1.5).move_to(ax_1.c2p(2,2.5)).set_color_by_gradient(REANLEA_AQUA)

        self.play(
            sq_2_grp_ref.animate.move_to(ax_1.c2p(-2.5,2.5))
        )

        self.play(
            Write(equal)
        )

        sq_eras=Square(side_length=.125).set_fill(color=REANLEA_BACKGROUND_COLOR, opacity=1).set_stroke(color=REANLEA_BACKGROUND_COLOR).set_z_index(7)

        sq_eras_1=sq_eras.copy().move_to(ax_1.c2p(-5.1265,3))
        sq_eras_2=sq_eras.copy().move_to(ax_1.c2p(0,-.126))
        sq_eras_3=sq_eras.copy().move_to(ax_1.c2p(.126,0))
        sq_eras_4=sq_eras.copy().move_to(ax_1.c2p(-2,5.1265))
        sq_eras_grp=VGroup(sq_eras_1,sq_eras_2,sq_eras_3,sq_eras_4)
        self.add(sq_eras_grp)

        self.play(triangles[7].animate.move_to(Line(triangles[5].get_corner(DL), triangles[5].get_corner(UR)).get_center()))
        self.play(triangles[4].animate.next_to(sq_2_ref.get_corner(DR), UL, buff=0))
        self.play(triangles[6].animate.next_to(sq_2_ref.get_corner(DR), UL, buff=0))
        self.wait(0.5)

        c_square = Difference(sq_2, Union(triangles[0], triangles[1], triangles[2], triangles[3]), fill_color=REANLEA_BLUE_LAVENDER, fill_opacity=1)
        a_square = Square(side_length=a_len, fill_color=REANLEA_BLUE_LAVENDER, fill_opacity=1).next_to(sq_2_ref.get_corner(DL), UR, buff=0)
        b_square = Square(side_length=b_len, fill_color=REANLEA_BLUE_LAVENDER, fill_opacity=1).next_to(sq_2_ref.get_corner(UR), DL, buff=0)

        c_square.set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA, REANLEA_SLATE_BLUE]).set_z_index(-1)
        b_square.set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA, REANLEA_SLATE_BLUE]).set_z_index(-1)
        a_square.set_stroke(width=5, color=[REANLEA_VIOLET,REANLEA_AQUA, REANLEA_SLATE_BLUE]).set_z_index(-1)

        a_sq_lbl_0_1=MathTex("a").scale(.6).set_color(REANLEA_GREEN_DARKEST).move_to(ax_1.c2p(-2.3,1.5))
        a_sq_lbl_0_2=MathTex("a").scale(.6).set_color(REANLEA_GREEN_DARKEST).move_to(ax_1.c2p(-3.5,2.7))
        a_sq_lbl_0=VGroup(a_sq_lbl_0_1,a_sq_lbl_0_2)
        a_sq_lbl_1=MathTex(r"a^2").scale(.6).set_color(REANLEA_GREEN_DARKEST).move_to(a_square
        .get_center())

        b_sq_lbl_0_1=MathTex("b").scale(.6).set_color(PURE_RED).move_to(ax_1.c2p(-1,3.3))
        b_sq_lbl_0_2=MathTex("b").scale(.6).set_color(PURE_RED).move_to(ax_1.c2p(-1.7,4))
        b_sq_lbl_0=VGroup(b_sq_lbl_0_1,b_sq_lbl_0_2)
        b_sq_lbl_1=MathTex(r"b^2").scale(.6).set_color(PURE_RED).move_to(b_square.get_center())


        c_sq_lbl_0_1=MathTex("c").scale(.6).set_color(REANLEA_BLUE_DARKER).move_to(ax_1.c2p(5.25,1.8))
        c_sq_lbl_0_2=MathTex("c").scale(.6).set_color(REANLEA_BLUE_DARKER).move_to(ax_1.c2p(7.4,1.3))
        c_sq_lbl_0=VGroup(c_sq_lbl_0_1,c_sq_lbl_0_2)
        c_sq_lbl_1=MathTex(r"c^2").scale(.6).set_color(REANLEA_BLUE_DARKER).move_to(c_square.get_center())


        self.play(
            ReplacementTransform(a_ln_lab_1.copy(), a_sq_lbl_0)
        )

        self.play(
            ReplacementTransform(b_ln_lab_1.copy(), b_sq_lbl_0)
        )

        self.play(
            ReplacementTransform(c_ln_lab_1.copy(), c_sq_lbl_0)
        )



        
        #self.add(c_square)
        #sq_2_ref.set_fill(opacity=0)
        a_b_c_sq=VGroup(a_square,b_square,c_square)
        #self.add(a_b_c_sq)

        self.play(
            FadeIn(a_b_c_sq),
            ReplacementTransform(a_sq_lbl_0,a_sq_lbl_1),
            ReplacementTransform(b_sq_lbl_0,b_sq_lbl_1),
            ReplacementTransform(c_sq_lbl_0,c_sq_lbl_1),
        )

        c_sq_lbl_1.add_updater(
            lambda z : z.become(
                MathTex(r"c^2").scale(.6).set_color(REANLEA_BLUE_DARKER).move_to(c_square.get_center())
            )
        )

        b_sq_lbl_1.add_updater(
            lambda z : z.become(
                MathTex(r"b^2").scale(.6).set_color(PURE_RED).move_to(b_square.get_center())
            )
        )

        a_sq_lbl_1.add_updater(
            lambda z : z.become(
                MathTex(r"a^2").scale(.6).set_color(REANLEA_GREEN_DARKEST).move_to(a_square.get_center())
            )
        )

        triangles_grp=VGroup()
        for i in range(0,8):
            triangles_grp.add(triangles[i])

        self.play(
            Unwrite(triangles_grp),
            Unwrite(sq_2),
            Unwrite(sq_2_ref),
        )
        self.play(
            FadeOut(sq_eras_grp)
        )

        self.wait()

        a_sq_lbl_2=MathTex(r"a^2").scale(.6).set_color(REANLEA_GREEN_DARKEST).move_to(a_square
        .get_center())
        b_sq_lbl_2=MathTex(r"b^2").scale(.6).set_color(PURE_RED).move_to(b_square.get_center())
        c_sq_lbl_2=MathTex(r"c^2").scale(.6).set_color(REANLEA_BLUE_DARKER).move_to(c_square.get_center())

        sqr_lbl_grp_0=VGroup(a_sq_lbl_1,b_sq_lbl_1,c_sq_lbl_1)
        sqr_lbl_grp=VGroup(a_sq_lbl_2,b_sq_lbl_2,c_sq_lbl_2)

        pythagoras_thm=MathTex(r"c^2","=",r"a^2","+",r"b^2").shift(2.25*DOWN)
        pythagoras_thm[0].set_color(REANLEA_BLUE_SKY)
        pythagoras_thm[2].set_color(PURE_GREEN)
        pythagoras_thm[4].set_color(PURE_RED)

        self.play(
            ReplacementTransform(sqr_lbl_grp,pythagoras_thm)
        )

        self.wait()


        self.play(
            Restore(a_len_ln),
            Restore(b_len_ln),
            Restore(c_len_ln)
        )

        self.play(
            c_len_ln.animate.set_stroke(width=3, color=REANLEA_BLUE_SKY)
        )


        self.wait(2)

        
        self.play(
            FadeOut(equal),
            c_square.animate.shift((ax_1.c2p(6,0)[0]-ax_1.c2p(0,0)[0])*LEFT),
            pythagoras_thm.animate.to_corner(UR, buff=1)
        )
        self.play(
            b_square.animate.shift((ax_1.c2p(3,0)[0]-ax_1.c2p(-2,3)[0])*RIGHT+(ax_1.c2p(-2,3)[1]-ax_1.c2p(3,0)[1])*DOWN)
        )
        self.play(
            a_square.animate.shift((ax_1.c2p(0,0)[0]-ax_1.c2p(-5,3)[0])*RIGHT+(ax_1.c2p(-5,3)[1]-ax_1.c2p(0,0)[1])*DOWN)
        )
        self.wait(2)

        self.play(
            FadeIn(tr_angl_0)
        )

        self.wait(2)

        pythagoras_thm_1=MathTex(r"c","=",r"\sqrt{a^2 + b^2}").to_corner(UR, buff=1)
        pythagoras_thm_1[0].set_color(REANLEA_BLUE_SKY)
        pythagoras_thm_1[2][2:4].set_color(PURE_GREEN)
        pythagoras_thm_1[2][5:7].set_color(PURE_RED)
        

        self.play(
            TransformMatchingShapes(pythagoras_thm,pythagoras_thm_1),
            Indicate(c_len_ln),
            Indicate(c_len_ln_1),
            Indicate(c_ln_lab_1)
        )

        self.wait()

        dt_1_ref=Dot().set_color(REANLEA_AQUA).move_to(ax_1.c2p(0,0)).set_z_index(-1)
        dt_2_ref=Dot().set_color(REANLEA_PURPLE).move_to(ax_1.c2p(3,2)).set_z_index(-1)
        
        ln_1_ref=Line(start=dt_1_ref.get_center(), end=dt_2_ref.get_center()).set_stroke(width=5, color=[REANLEA_PURPLE,REANLEA_AQUA]).set_z_index(-3)

        self.add(ln_1_ref)

        self.play(
            Unwrite(a_b_c_sq),
            FadeIn(dt_1_ref),
            FadeIn(dt_2_ref),
            FadeOut(a_sq_lbl_1),
            Uncreate(b_sq_lbl_1),
            FadeOut(c_sq_lbl_1),
        )
        self.wait(2)

        a_ln_lab_ref=MathTex("a").scale(.65).set_color(PURE_GREEN).next_to(a_len_ln,DOWN)
        b_ln_lab_ref=MathTex("b").scale(.65).set_color(PURE_RED).next_to(b_len_ln,RIGHT)
        c_ln_lab_ref=MathTex("c").scale(.65).set_color(REANLEA_BLUE_SKY).move_to(ax_1.c2p(1.35,1.35))

        

        self.play(
            ReplacementTransform(c_ln_lab_1, c_ln_lab_ref),
            ReplacementTransform(b_ln_lab_1, b_ln_lab_ref),
            ReplacementTransform(a_ln_lab_1, a_ln_lab_ref),
            Uncreate(c_len_ln_1),
            Uncreate(b_len_ln_1),
            Uncreate(a_len_ln_1)
        )
        self.wait()

        pythagoras_thm_1_ref=pythagoras_thm_1.copy().move_to(ax_1.c2p(1.35,1.35)).rotate(ln_1_ref.get_angle()).scale(.5)

        pythagoras_thm_grp=VGroup(c_ln_lab_ref,pythagoras_thm_1)

        self.play(
            ReplacementTransform(pythagoras_thm_grp, pythagoras_thm_1_ref)
        )
        self.wait()

        ax_3=ax_1.copy().set_z_index(-4)

        self.play(
            Write(ax_3),
            run_time=2
        )

        self.play(
            FadeOut(tr_angl_0),
            FadeOut(c_len_ln),
            FadeOut(hed_txt_bez)
        )

        dt_2_lbl_0=MathTex("v").set_color(REANLEA_SLATE_BLUE_LIGHTEST).scale(.85).next_to(dt_2_ref,UR, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2)

        self.play(
            Write(dt_2_lbl_0)
        )

        dt_2_lbl_1=MathTex("=","(","a",",","b",")").scale(.65).next_to(dt_2_lbl_0,RIGHT, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2)
        dt_2_lbl_1[2].set_color(PURE_GREEN)
        dt_2_lbl_1[4].set_color(PURE_RED)

        self.play(
            Create(dt_2_lbl_1)
        )
        


        
        '''self.play(
            FadeOut(a_len_ln),
            FadeOut(b_len_ln),
            FadeOut(c_len_ln)
        )'''

        self.wait(4)



        # manim -pqh anim2.py Scene3

        # manim -sqk anim2.py Scene3


class VMoveMob(VMobject):
    def pfp(self, alpha):
        return self.point_from_proportion(alpha)

class Scene3_1(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        


        # SCENE

        ax_1=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(2)

        ax_2=ax_1.copy().set_z_index(-5)

        self.add(ax_2)

        dt_1=Dot().set_color(REANLEA_AQUA).move_to(ax_1.c2p(0,0))
        dt_2=Dot().set_color(REANLEA_PURPLE).move_to(ax_1.c2p(3,2))
        
        ln_1=Line(start=dt_1.get_center(), end=dt_2.get_center()).set_stroke(width=5, color=[REANLEA_PURPLE,REANLEA_AQUA]).set_z_index(-1)

        self.add(dt_1,dt_2,ln_1)

        dt_1_lbl_0=MathTex("o").set_color(REANLEA_SLATE_BLUE_LIGHTEST).scale(.85).next_to(dt_1,DL, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2)

        dt_2_lbl_0=MathTex("v").set_color(REANLEA_SLATE_BLUE_LIGHTEST).scale(.85).next_to(dt_2,UR, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2)

        dt_2_lbl_1=MathTex("=","(","a",",","b",")").scale(.65).next_to(dt_2_lbl_0,RIGHT, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2)
        dt_2_lbl_1[2].set_color(PURE_GREEN)
        dt_2_lbl_1[4].set_color(PURE_RED)

        self.add(dt_2_lbl_0,dt_2_lbl_1)

        pythagoras_thm_1=MathTex(r"c","=",r"\sqrt{a^2 + b^2}").move_to(ax_1.c2p(1.35,1.35)).rotate(ln_1.get_angle()).scale(.5)
        pythagoras_thm_1[0].set_color(REANLEA_BLUE_SKY)
        pythagoras_thm_1[2][2:4].set_color(PURE_GREEN)
        pythagoras_thm_1[2][5:7].set_color(PURE_RED)


        self.add(pythagoras_thm_1)

        a_len_ln=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(3,0)).set_stroke(width=3, color=PURE_GREEN).save_state()
        b_len_ln=DashedLine(start=ax_1.c2p(3,0),end=ax_1.c2p(3,2)).set_stroke(width=3, color=PURE_RED).save_state()
        c_len_ln=DashedLine(start=ax_1.c2p(0,0),end=ax_1.c2p(3,2)).set_stroke(width=3, color=PURE_RED).save_state()
        

        a_ln_lab=MathTex("a").scale(.65).set_color(PURE_GREEN).next_to(a_len_ln,DOWN)
        b_ln_lab=MathTex("b").scale(.65).set_color(PURE_RED).next_to(b_len_ln,RIGHT)
        

        self.add(a_len_ln,b_len_ln,a_ln_lab,b_ln_lab)

        ### MAIN SCENE

        self.play(
            Write(dt_1_lbl_0)
        )
        self.wait()

        pythagoras_thm_2=MathTex(r"d(v,o)","=",r"\sqrt{a^2 + b^2}").move_to(ax_1.c2p(1.35,1.35)).rotate(ln_1.get_angle()).scale(.5)
        pythagoras_thm_2[0].set_color(REANLEA_BLUE_SKY)
        pythagoras_thm_2[2][2:4].set_color(PURE_GREEN)
        pythagoras_thm_2[2][5:7].set_color(PURE_RED)

        self.play(
            ReplacementTransform(pythagoras_thm_1,pythagoras_thm_2)
        )

        self.wait()

        arr_1=Arrow(start=ax_1.c2p(0,0),end=ax_1.c2p(3,2),tip_length=.125,stroke_width=4, buff=0).set_color_by_gradient(REANLEA_CYAN_LIGHT)

        self.play(
            Write(arr_1)
        )
        self.play(
            Indicate(dt_2,color=PURE_RED),
            Flash(dt_2,color=PURE_GREEN),
            run_time=1.75            
        )
        self.wait()

        pythagoras_thm_3=MathTex(r"\lVert v \rVert","=",r"\sqrt{a^2 + b^2}").move_to(ax_1.c2p(1.35,1.35)).rotate(ln_1.get_angle()).scale(.5)
        pythagoras_thm_3[0].set_color(REANLEA_CYAN_LIGHT)
        pythagoras_thm_3[2][2:4].set_color(PURE_GREEN)
        pythagoras_thm_3[2][5:7].set_color(PURE_RED)

        self.play(
            ReplacementTransform(pythagoras_thm_2,pythagoras_thm_3)
        )
        self.wait(2)

        pythagoras_thm_3_0=pythagoras_thm_3.copy()

        norm_v_0=MathTex(r"\lVert v \rVert").move_to(ax_1.c2p(1.35,1.35)).rotate(ln_1.get_angle()).scale(.5).set_color(REANLEA_CYAN_LIGHT)
        
        norm_v_1=MathTex(r"\lVert v \rVert","=",r"\sqrt{a^2 + b^2}").scale(.8).to_corner(UR)
        norm_v_1[0].set_color(REANLEA_CYAN_LIGHT)
        norm_v_1[2][2:4].set_color(PURE_GREEN)
        norm_v_1[2][5:7].set_color(PURE_RED)

        self.play(
            ReplacementTransform(pythagoras_thm_3,norm_v_0),
            ReplacementTransform(pythagoras_thm_3_0,norm_v_1)
        )
        self.wait(2)

        dot_1=Dot(radius=0.15, color=REANLEA_AQUA_GREEN).move_to(ax_1.c2p(0,0)).set_sheen(-0.4,DOWN).set_z_index(2).save_state()
        self.play(
            Write(dot_1)
        )

        ln_1_length=ln_1.get_length()

        push_arr=Arrow(start=ax_1.c2p(-.8,0),end=ax_1.c2p(-.4,0),max_tip_length_to_length_ratio=.5, buff=0).set_color(REANLEA_YELLOW_GREEN).set_opacity(1).set_z_index(6)

        push_arr_lbl=Text("F").scale(.5).set_color(REANLEA_YELLOW_GREEN).next_to(push_arr,UP)
        push_arr_lbl.add_updater(
            lambda z : z.become(
                Text("F").scale(.5).set_color(REANLEA_YELLOW_GREEN).next_to(push_arr,UP)
            )
        )

        self.play(
            FadeIn(push_arr)
        )
        self.play(
            Write(push_arr_lbl)
        )
        self.play(
            push_arr.animate.move_to(ax_1.c2p(-.35,0)),
            run_time=.35
        )
        self.play(
            dot_1.animate.shift(ln_1_length*RIGHT)
        )
        
        self.play(
            FadeOut(push_arr_lbl),
            run_time=.35
        )
        self.play(
            FadeOut(push_arr)
        )
        

        
        self.wait(2)

        circ=DashedVMobject(Circle(radius=ln_1_length), dashed_ratio=0.5, num_dashes=100).move_to(dt_1.get_center()).set_stroke(width=0.65)
        circ.set_color_by_gradient(REANLEA_WHITE,REANLEA_WARM_BLUE,REANLEA_YELLOW_CREAM)
        

        self.play(
            Write(circ)
        )
        

        dot_1_lbl_1=MathTex("(",r"\lVert v \rVert",",","0",")").scale(.45).next_to(dot_1,DR, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2)
        self.wait(1.25)
        
        self.play(
            Write(dot_1_lbl_1)
        )
        self.wait(2)

        self.play(
            Restore(dot_1),
            FadeOut(dot_1_lbl_1)
        )
        self.wait(2)

        push_arr_1=Arrow(start=ax_1.c2p(0,-.8),end=ax_1.c2p(0,-.4),max_tip_length_to_length_ratio=.5, buff=0).set_color(REANLEA_YELLOW_GREEN).set_opacity(1).set_z_index(6)

        push_arr_1_lbl=Text("F").scale(.5).set_color(REANLEA_YELLOW_GREEN).next_to(push_arr_1,RIGHT)
        push_arr_1_lbl.add_updater(
            lambda z : z.become(
                Text("F").scale(.5).set_color(REANLEA_YELLOW_GREEN).next_to(push_arr_1,RIGHT)
            )
        )

        self.play(
            FadeIn(push_arr_1)
        )
        self.play(
            Write(push_arr_1_lbl)
        )
        self.play(
            push_arr_1.animate.move_to(ax_1.c2p(0,-.35)),
            run_time=.35
        )
        self.play(
            dot_1.animate.shift(ln_1_length*UP)
        )
        
        self.play(
            FadeOut(push_arr_1_lbl),
            run_time=.35
        )
        self.play(
            FadeOut(push_arr_1)
        )
        



        dot_1_lbl_2=MathTex("(","0",",",r"\lVert v \rVert",")").scale(.45).next_to(dot_1,UR, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2)
        self.wait(1.25)
        
        self.play(
            Write(dot_1_lbl_2)
        )
        
        self.wait(2)

        self.play(
            Restore(dot_1),
            FadeOut(dot_1_lbl_2)
        )

        self.wait()




        push_arr_2=Arrow(start=ax_1.c2p(-.78,-.52),end=ax_1.c2p(-.45,-.3),max_tip_length_to_length_ratio=.5, buff=0).set_color(REANLEA_YELLOW_GREEN).set_opacity(1).set_z_index(6)

        push_arr_2_lbl=Text("F").scale(.5).set_color(REANLEA_YELLOW_GREEN).next_to(push_arr_2,UP)
        push_arr_2_lbl.add_updater(
            lambda z : z.become(
                Text("F").scale(.5).set_color(REANLEA_YELLOW_GREEN).next_to(push_arr_2,UP)
            )
        )

        self.play(
            FadeIn(push_arr_2)
        )
        self.play(
            Write(push_arr_2_lbl)
        )
        self.play(
            push_arr_2.animate.move_to(ax_1.c2p(-.27,-.18)),
            run_time=.35
        )
        self.play(
            dot_1.animate.move_to(dt_2.get_center())
        )
        
        self.play(
            FadeOut(push_arr_2_lbl),
            run_time=.35
        )
        self.play(
            FadeOut(push_arr_2)           
        )

        self.wait(2)

        sr_a_len_ln=SurroundingRectangle(a_len_ln, stroke_width=1, color=REANLEA_YELLOW_CREAM)

        self.play(
            Circumscribe(a_len_ln,stroke_width=1.5, color=REANLEA_GOLD),
            Create(sr_a_len_ln),
            run_time=2
        )

        bez_arr_1=bend_bezier_arrow().flip(UP).rotate(-45*DEGREES).scale(.65).next_to(a_len_ln,DOWN,buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2).shift(1.2*RIGHT+.075*UP)

        with RegisterFont("Fuzzy Bubbles") as fonts:
            txt_1=Text("projection along x-axis", font=fonts[0]).scale(0.25)
            txt_1.set_color_by_gradient(REANLEA_TXT_COL).move_to(bez_arr_1,DR).shift(2*RIGHT+.2*UP).set_stroke(width=1.05)
        

        self.play(
            Create(bez_arr_1)
        )
        self.play(
            Write(txt_1)
        )
        self.wait()

        angl_1=Angle(a_len_ln,c_len_ln).set_color(REANLEA_YELLOW_GREEN).set_stroke(width=1.5)
        self.play(
            Create(angl_1)
        )
        angl_1_lbl=MathTex(r"\theta").scale(.4).set_color(REANLEA_YELLOW_GREEN).next_to(angl_1,UR, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2).shift(.25*DOWN)
        self.play(
            Write(angl_1_lbl)
        )


        x_proj_lbl=MathTex("=",r"\lVert v \rVert",r"\cdot",r"cos\theta").scale(.65).next_to(a_ln_lab,RIGHT, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2)
        x_proj_lbl[1].set_color(REANLEA_CYAN_LIGHT)
        x_proj_lbl[3][3].set_color(REANLEA_YELLOW_GREEN)
        self.play(
            ReplacementTransform(angl_1_lbl.copy(),x_proj_lbl)
        )
        self.wait()

        x_proj_lbl_1=MathTex("a","=",r"\lVert v \rVert",r"\cdot",r"cos\theta").scale(.8).next_to(norm_v_1,DOWN, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER)
        x_proj_lbl_1[0].set_color(PURE_GREEN)
        x_proj_lbl_1[2].set_color(REANLEA_CYAN_LIGHT)
        x_proj_lbl_1[4][3].set_color(REANLEA_YELLOW_GREEN)
        self.play(
            ReplacementTransform(x_proj_lbl,x_proj_lbl_1)
        )


        self.wait()

        self.play(
            Uncreate(sr_a_len_ln.reverse_direction()),
            FadeOut(bez_arr_1),
            FadeOut(txt_1)
        )

        self.wait(2)



        sr_b_len_ln=SurroundingRectangle(b_len_ln, stroke_width=1, color=REANLEA_YELLOW_CREAM)

        self.play(
            Circumscribe(b_len_ln,stroke_width=1.5, color=REANLEA_GOLD),
            Create(sr_b_len_ln),
            run_time=2
        )
        self.wait()
        y_proj_lbl=MathTex("=",r"\lVert v \rVert",r"\cdot",r"sin\theta").scale(.65).next_to(b_ln_lab,RIGHT, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2)
        y_proj_lbl[1].set_color(REANLEA_CYAN_LIGHT)
        y_proj_lbl[3][3].set_color(REANLEA_YELLOW_GREEN)

        self.play(
            ReplacementTransform(angl_1_lbl.copy(),y_proj_lbl)
        )
        self.wait()

        y_proj_lbl_1=MathTex("b","=",r"\lVert v \rVert",r"\cdot",r"sin\theta").scale(.8).next_to(x_proj_lbl_1,DOWN, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER)
        y_proj_lbl_1[0].set_color(PURE_RED)
        y_proj_lbl_1[2].set_color(REANLEA_CYAN_LIGHT)
        y_proj_lbl_1[4][3].set_color(REANLEA_YELLOW_GREEN)
        self.play(
            ReplacementTransform(y_proj_lbl,y_proj_lbl_1)
        )

        self.play(
            Uncreate(sr_b_len_ln.reverse_direction())
        )
        self.wait(2)

        self.play(
            Indicate(dt_1,color=PURE_RED),
            Flash(dt_1,color=PURE_GREEN),
            run_time=1.75            
        )
        self.wait()

        self.play(
            Indicate(dt_2,color=PURE_RED),
            Flash(dt_2,color=PURE_GREEN),
            run_time=1.75            
        )
        self.wait()

        self.play(
            Circumscribe(a_len_ln,stroke_width=1.5, color=REANLEA_GOLD),
            run_time=2
        )
        self.wait()
        self.play(
            Circumscribe(b_len_ln,stroke_width=1.5, color=REANLEA_GOLD),
            run_time=2
        )
        self.wait(2)

        arr_2=Arrow(start=ax_1.c2p(0,0),end=ax_1.c2p(3,0),tip_length=.125,stroke_width=4, buff=0).set_color_by_gradient(REANLEA_GREEN)
        arr_3=Arrow(start=ax_1.c2p(3,0),end=ax_1.c2p(3,2),tip_length=.125,stroke_width=4, buff=0).set_color_by_gradient(REANLEA_CHARM)

        self.play(
            Write(arr_2)
        )
        self.play(
            Write(arr_3)
        )
        self.wait(2)

        
        self.play(
            arr_3.animate.shift(
                (ax_1.c2p(3,0)[0]-ax_1.c2p(0,0)[0])*LEFT
            )
        )
        self.wait(2)

        force_0=MathTex(r"F").next_to(norm_v_0,DR, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/3).rotate(ln_1.get_angle()).scale(.5).set_color(REANLEA_CYAN_LIGHT)

        self.play(
            Write(force_0)
        )

        force_1=MathTex(r"F",r"\cdot",r"cos\theta").next_to(a_len_ln,UP, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/3).scale(.5).set_color(PURE_GREEN)
        force_1[0].set_color(REANLEA_CYAN_LIGHT)
        force_1[2][3].set_color(REANLEA_YELLOW_GREEN)

        self.play(
            Write(force_1)
        )

        force_2=MathTex(r"F",r"\cdot",r"sin\theta").next_to(arr_3,LEFT, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/3).rotate(PI/2).scale(.5).set_color(REANLEA_CHARM).shift(.5*RIGHT)
        force_2[0].set_color(REANLEA_CYAN_LIGHT)
        force_2[2][3].set_color(REANLEA_YELLOW_GREEN)
        self.play(
            Write(force_2)
        )
        self.wait(2)

        force_final=MathTex("F","=",r"F",r"\cdot",r"cos\theta","+",r"F",r"\cdot",r"sin\theta").to_corner(UL, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/3).scale(.75).set_color(REANLEA_TXT_COL).shift(1.5*DOWN)
        force_final[0].set_color(REANLEA_CYAN_LIGHT)
        force_final[2].set_color(REANLEA_CYAN_LIGHT)
        force_final[4][3].set_color(REANLEA_YELLOW_GREEN)
        force_final[6].set_color(REANLEA_CYAN_LIGHT)
        force_final[8][3].set_color(REANLEA_YELLOW_GREEN)

        self.play(
            ReplacementTransform(
                VGroup(force_1,force_2,force_0).copy(),force_final
            )
        )

        self.play(
            Restore(dot_1)
        )
        self.wait()

        push_arr_3=Arrow(start=ax_1.c2p(-.78,-.52),end=ax_1.c2p(-.45,-.3),max_tip_length_to_length_ratio=.5, buff=0).set_color(REANLEA_YELLOW_GREEN).set_opacity(1).set_z_index(6).save_state()

        push_arr_3_lbl=Text("F").scale(.5).set_color(REANLEA_YELLOW_GREEN).next_to(push_arr_3,UP)
        push_arr_3_lbl.add_updater(
            lambda z : z.become(
                Text("F").scale(.5).set_color(REANLEA_YELLOW_GREEN).next_to(push_arr_3,UP)
            )
        )

        self.play(
            FadeIn(push_arr_3),
            FadeIn(push_arr_3_lbl)
        )

        self.play(
            push_arr_3.animate.move_to(ax_1.c2p(-.27,-.18)),
            run_time=.35
        )
        self.play(
            dot_1.animate.move_to(ax_1.c2p(3,0)),
            Indicate(force_1),
            Circumscribe(force_final[2:5])
        )
        
        self.play(
            FadeOut(push_arr_3_lbl),
            run_time=.35
        )
        self.play(
            FadeOut(push_arr_3)           
        )
        self.wait()

        self.play(
            Restore(dot_1)
        )
        self.play(
            Restore(push_arr_3),
            Write(push_arr_3_lbl)
        )

        self.play(
            push_arr_3.animate.move_to(ax_1.c2p(-.27,-.18)),
            run_time=.35
        )
        self.play(
            dot_1.animate.move_to(ax_1.c2p(0,2)),
            Indicate(force_2),
            Circumscribe(force_final[6:])
        )
        
        self.play(
            FadeOut(push_arr_3_lbl),
            run_time=.35
        )
        self.play(
            FadeOut(push_arr_3)           
        )
        self.wait()


        self.play(
            Restore(dot_1)
        )

        dot_1_0=dot_1.copy().save_state()
        dot_1_1=dot_1.copy().save_state()
        self.add(dot_1_0,dot_1_1)

        self.play(
            Restore(push_arr_3)
        )
        self.play(
            Write(push_arr_3_lbl)
        )

        self.play(
            push_arr_3.animate.move_to(ax_1.c2p(-.27,-.18)),
            run_time=.35
        )
        self.play(
            dot_1.animate.move_to(ax_1.c2p(3,2)),
            dot_1_0.animate.move_to(ax_1.c2p(3,0)),
            dot_1_1.animate.move_to(ax_1.c2p(0,2)),
            Indicate(force_1),
            Indicate(force_2),
            Circumscribe(force_final[2:5]),
            Circumscribe(force_final[6:]),
            run_time=1.75
        )
        
        self.play(
            FadeOut(push_arr_3_lbl),
            run_time=.35
        )
        self.play(
            FadeOut(push_arr_3)           
        )
        self.wait(2)

        self.play(
            Restore(dot_1),
            Restore(dot_1_0),
            Restore(dot_1_1),
            run_time=1.25
        )
        self.wait(2)

        c1=VMoveMob().set_points_as_corners(points=[ax_1.c2p(0,0),ax_1.c2p(3,0),ax_1.c2p(3,2)]) 
        c2=VMoveMob().set_points_as_corners(points=[ax_1.c2p(0,0),ax_1.c2p(0,2),ax_1.c2p(3,2)]) 

        self.play(
            dot_1.animate.move_to(ax_1.c2p(3,2)),
            UpdateFromAlphaFunc(dot_1_0, lambda x, alpha: x.move_to(c1.pfp(alpha))),
            UpdateFromAlphaFunc(dot_1_1, lambda x, alpha: x.move_to(c2.pfp(alpha))),
            run_time = 3, rate_func= smooth
        )



        

        self.wait(4)



        # manim -pqh anim2.py Scene3_1

        # manim -sqk anim2.py Scene3_1


###################################################################################################################


class Scene4(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        


        # SCENE

        ax_1=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1) 

        rot_tracker=ValueTracker(0)

        ax_2=ax_1.copy()

        ax_2_ref=ax_2.copy()

        ax_2.add_updater(
            lambda x : x.become(ax_2_ref.copy()).rotate(
                rot_tracker.get_value(), about_point=ax_1.c2p(0,0)
            )
        )

        self.wait()

        self.play(
            Write(ax_1),
            run_time=2
        )

        dot_1=Dot(radius=0.15, color=REANLEA_AQUA_GREEN).move_to(ax_1.c2p(0,0)).set_sheen(-0.4,DOWN).set_z_index(4).save_state()

        dot_1_1=dot_1.copy().save_state()

        self.play(
            Write(dot_1)
        )

        dt_0=Dot().set_color(REANLEA_YELLOW).move_to(ax_1.c2p(0,0)).set_z_index(3)
        self.add(dt_0)


        push_arr=Arrow(start=ax_1.c2p(-.8,0),end=ax_1.c2p(-.4,0),max_tip_length_to_length_ratio=.5, buff=0).set_color(REANLEA_YELLOW_GREEN).set_opacity(1).set_z_index(6)

        push_arr_lbl=Text("F").scale(.5).set_color(REANLEA_YELLOW_GREEN).next_to(push_arr,UP)
        push_arr_lbl.add_updater(
            lambda z : z.become(
                Text("F").scale(.5).set_color(REANLEA_YELLOW_GREEN).next_to(push_arr,UP)
            )
        )

        self.play(
            FadeIn(push_arr)
        )
        
        self.play(
            push_arr.animate.move_to(ax_1.c2p(-.35,0)),
            run_time=.35
        )
        self.play(
            dot_1.animate.move_to(ax_1.c2p(1,0))
        )  


        lbl_i=MathTex("i").scale(.45).set_color(REANLEA_AQUA).move_to(ax_1.c2p(1.2,.2))  
        dt_1=Dot().set_color(REANLEA_AQUA).move_to(ax_1.c2p(1,0)).set_z_index(3)
        dt_2=Dot().set_color(REANLEA_PINK).move_to(ax_1.c2p(3,2)).set_z_index(1)

        dt_2_lbl=MathTex("v").scale(.7).set_color(REANLEA_PINK_LIGHTER).next_to(dt_2,RIGHT)

        self.play(
            FadeOut(push_arr),
            Create(dt_1)
        )
        
        self.wait(2)

        self.play(
            Write(lbl_i),
            Restore(dot_1)
        )

        push_arr_1=Arrow(start=ax_1.c2p(-.78,-.52),end=ax_1.c2p(-.45,-.3),max_tip_length_to_length_ratio=.5, buff=0).set_color(REANLEA_YELLOW_GREEN).set_opacity(1).set_z_index(6).save_state()
        
        self.play(
            FadeIn(push_arr_1)
        )
        
        self.play(
            push_arr_1.animate.move_to(ax_1.c2p(-.27,-.18)),
            run_time=.35
        )
        self.play(
            dot_1.animate.move_to(ax_1.c2p(3,2))
        )
        self.play(
            FadeOut(push_arr_1)
        )
        self.wait(2)

        self.add(dt_2)
        self.play(
            Write(dt_2_lbl)
        )

        self.play(
            Restore(dot_1)
        )

        dissipating_dt_1=Dot().move_to(ax_1.c2p(3,2)).set_opacity(opacity=0)
        dissipating_path_1 = TracedPath(dissipating_dt_1.get_center, dissipating_time=0.5, stroke_color=[REANLEA_AQUA,PURE_GREEN],stroke_opacity=[1, 0])
        self.add(dissipating_dt_1,dissipating_path_1)


        push_arr_2=Arrow(start=ax_1.c2p(-.78,-.52),end=ax_1.c2p(-.45,-.3),max_tip_length_to_length_ratio=.5, buff=0).set_color(REANLEA_YELLOW_GREEN).set_opacity(1).set_z_index(6).save_state()

        x_proj_ln=DashedLine(start=dt_0.get_center(),end=dot_1.get_center()).set_stroke(width=3, color=[REANLEA_YELLOW_CREAM]).set_z_index(2)
        self.add(x_proj_ln)

        x_proj_ln.add_updater(
            lambda z : z.become(
                DashedLine(start=dt_0.get_center(),end=dot_1.get_center()).set_stroke(width=3, color=[REANLEA_YELLOW_CREAM]).set_z_index(2)
            )
        )
        
        self.play(
            FadeIn(push_arr_2)
        )  

        ln_x=Line(ax_1.c2p(0,0),ax_1.c2p(3,0))
        ln_y=Line(ax_1.c2p(3,2),ax_1.c2p(3,0))

        self.play(
            push_arr_2.animate.move_to(ax_1.c2p(-.27,-.18)),
            run_time=.35
        )

        self.play(
            AnimationGroup(
                MoveAlongPath(dissipating_dt_1,ln_y),
                Flash(point=Dot().move_to(ax_1.c2p(3,0)), color=REANLEA_GREEN_AUQA),
                #MoveAlongPath(dot_1,ln_x),
                lag_ratio=0.5
            ),
            dot_1.animate(run_time=1.25).move_to(ax_1.c2p(3,0))
        )

        self.play(
            FadeOut(push_arr_2)
        )

        '''self.play(
            dot_1.animate.move_to(ax_1.c2p(3,0)),
            dissipating_dt_1.animate.move_to(ax_1.c2p(3,0)),
            Flash(point=Dot().move_to(ax_1.c2p(3,0)), color=REANLEA_GREEN_AUQA),
        )''' 
    
        self.wait(2)

        arr_1=Arrow(start=ax_1.c2p(0,0),end=ax_1.c2p(3,2),tip_length=.125,stroke_width=4, buff=0).set_color_by_gradient(REANLEA_CYAN_LIGHT).set_z_index(1)

        self.play(
            Write(arr_1)
        )
        self.wait(2)


        ln_1=Line(start=ax_1.c2p(0,0),end=ax_1.c2p(3,2)).set_stroke(width=4, color=[REANLEA_PINK,REANLEA_YELLOW])

        ln_2=Line(start=ax_1.c2p(0,0),end=ax_1.c2p(3,0)).set_stroke(width=4, color=[REANLEA_PINK,REANLEA_YELLOW])

        ln_1_lbl=MathTex(r"\lVert v \rVert").scale(.5).set_color(REANLEA_CYAN_LIGHT).move_to(ax_1.c2p(1.35,1.45)).rotate(ln_1.get_angle())

        self.play(
            Unwrite(arr_1),
            Write(ln_1)
        )

        self.play(
            Create(ln_1_lbl)
        )
        self.wait(2)

        angl_1=Angle(ln_2,ln_1).set_color(REANLEA_YELLOW_GREEN).set_stroke(width=3.5).set_z_index(-1)
        self.play(
            Create(angl_1)
        )
        angl_1_lbl=MathTex(r"\theta").scale(.4).set_color(REANLEA_YELLOW_GREEN).next_to(angl_1,UR, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2).shift(.25*DOWN)
        self.play(
            Write(angl_1_lbl)
        )
        self.wait(2)

        ln_2_lbl_0=MathTex(r"\lVert v \rVert",r"\cdot",r"cos\theta").scale(.5).set_color(REANLEA_AQUA).move_to(ax_1.c2p(1.75,-.375))

        self.play(
            Write(ln_2_lbl_0)
        )
        self.wait(1.5)

        ln_2_lbl_1=MathTex(r"\lVert i \rVert",r"\cdot").scale(.5).set_color(REANLEA_AQUA).next_to(ln_2_lbl_0,LEFT, buff=.125)

        ln_2_lbl=VGroup(ln_2_lbl_0,ln_2_lbl_1)

        self.play(
            TransformMatchingShapes(lbl_i.copy(),ln_2_lbl_1)
        )


        dot_2=Dot(radius=0.1, color=REANLEA_AQUA_GREEN).move_to(ax_1.c2p(3,0)).set_sheen(-0.4,DOWN).set_z_index(3)

        self.add(dot_2)

        self.play(
            Restore(dot_1)
        )

        self.add(dot_1_1)
        self.play(
            FadeOut(dot_1)
        )
        self.wait(2)

        push_arr_3=Arrow(start=ax_1.c2p(-.8,0),end=ax_1.c2p(-.4,0),max_tip_length_to_length_ratio=.5, buff=0).set_color(REANLEA_YELLOW_GREEN).set_opacity(1).set_z_index(6)

        self.play(
            FadeIn(push_arr_3)
        )
        
        self.play(
            push_arr_3.animate.move_to(ax_1.c2p(-.35,0)),
            run_time=.35
        )

        dt_3=Dot().set_color(REANLEA_GOLDENROD).move_to(ax_1.c2p(2,0)).set_z_index(3)

        dt_3_lbl=MathTex("2 \cdot i").scale(.45).set_color(REANLEA_GOLDENROD).move_to(ax_1.c2p(2.35,.2)) 

        self.play(
            dot_1_1.animate.move_to(ax_1.c2p(2,0))
        )
        self.add(dt_3)

        self.play(
            FadeOut(push_arr_3)
        )
        self.wait() 

        
        ax_3=Axes(
            x_range=[-1.5,6.5],
            y_range=[-1.5,4.5],
            x_length=(round(config.frame_width)-2)*8/7,
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)

        ax_3_ref=ax_3.copy()
        ax_3.shift((ax_1.c2p(0,0)[0]-ax_3_ref.c2p(0,0)[0])*RIGHT)

        self.play(
            Restore(dot_1_1),
            Write(dt_3_lbl)
        )
        self.add(dot_1)
        self.play(
            FadeOut(dot_1_1)
        )

        self.wait(2)


        push_arr_4=Arrow(start=ax_1.c2p(-.78,-.52),end=ax_1.c2p(-.45,-.3),max_tip_length_to_length_ratio=.5, buff=0).set_color(REANLEA_YELLOW_GREEN).set_opacity(1).set_z_index(6).save_state()

        self.play(
            push_arr_4.animate.move_to(ax_1.c2p(-.27,-.18)),
            run_time=.35
        )

        self.play(
            dot_1.animate.move_to(ax_1.c2p(6,0)),
            Create(ax_3, run_time=2),
            lag_ratio=.08
        )
        self.play(
            FadeOut(push_arr_4)
        )
        self.wait()
        
        self.play(
            ln_2_lbl.animate.move_to(ax_1.c2p(3.2,-.375)).set_color(REANLEA_GOLD)
        )

        ln_3_lbl_0=MathTex(r"2 \cdot ").scale(.5).set_color(REANLEA_GOLD).next_to(ln_2_lbl, LEFT, buff=.1)

        ln_3_lbl=MathTex(r"\lVert 2 \cdot i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").scale(.5).set_color(REANLEA_GOLD).move_to(ax_1.c2p(3,-.375))

        self.play(
            Write(ln_3_lbl_0)
        )
        self.wait()

        ln_2_lbl_ref=VGroup(ln_3_lbl_0,ln_2_lbl)

        self.play(
            TransformMatchingShapes(ln_2_lbl_ref,ln_3_lbl)
        )
        self.wait(2)

        dot_3=Dot(radius=0.1, color=REANLEA_BLUE_LAVENDER).move_to(ax_1.c2p(6,0)).set_sheen(-0.4,DOWN).set_z_index(3)
        self.add(dot_3)

        fade_out_grp=VGroup(lbl_i,dt_3_lbl,ln_3_lbl,ln_1_lbl)
        lbl_i_1=MathTex("i").scale(.45).set_color(REANLEA_AQUA).move_to(ax_1.c2p(1.2,.2))  
        self.add(lbl_i_1)
        self.play(
            dot_1.animate.move_to(ax_1.c2p(0,0)).set_opacity(0),
            FadeOut(fade_out_grp),
            run_time=1.25
        )

        d_d_line_1=DashedDoubleArrow(
            start=ax_1.c2p(0,-.9), end=ax_1.c2p(3,-.9), dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).set_color_by_gradient(REANLEA_YELLOW_LIGHTER,REANLEA_GREEN_AUQA)

        d_d_line_1_lbl=MathTex(r"\lVert i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").scale(.5).set_color(REANLEA_AQUA_GREEN).next_to(d_d_line_1,RIGHT)

        d_d_line_2=DashedDoubleArrow(
            start=ax_1.c2p(0,-1.35), end=ax_1.c2p(6,-1.35), dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.01, buff=10
        ).set_color_by_gradient(REANLEA_YELLOW_LIGHTER,REANLEA_BLUE_LAVENDER)

        d_d_line_2_lbl=MathTex(r"\lVert 2 \cdot i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").scale(.5).set_color(REANLEA_BLUE_LAVENDER).next_to(d_d_line_2,RIGHT)

        self.play(
            Create(d_d_line_1)
        )
        self.play(
            Create(d_d_line_1_lbl)
        )

        self.play(
            Create(d_d_line_2)
        )
        self.play(
            Create(d_d_line_2_lbl)
        )
        
        self.wait(2)

        self.wait(2)

        self.play(
            ax_3.animate.add_coordinates()
        )


        self.wait(4)


        

        # manim -pqh anim2.py Scene4

        # manim -sqk anim2.py Scene4
        




class Scene4_1(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        


        # PREVIOUS SCENE

        ax_1=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1) 


        ax_2=Axes(
            x_range=[-1.5,6.5],
            y_range=[-1.5,4.5],
            x_length=(round(config.frame_width)-2)*8/7,
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)
        ax_2.add_coordinates()
        
        ax_1_ref=ax_1.copy()
        ax_2_ref=ax_2.copy()
        ax_2.shift((ax_1.c2p(0,0)[0]-ax_2_ref.c2p(0,0)[0])*RIGHT)
        ax_3=ax_2.copy()

        dt_0=Dot().set_color(REANLEA_YELLOW).move_to(ax_1.c2p(0,0)).set_z_index(5)
        dt_1=Dot().set_color(REANLEA_AQUA).move_to(ax_1.c2p(1,0)).set_z_index(3)
        dt_1_ref=Dot().set_color(REANLEA_AQUA).move_to(ax_1.c2p(1,0)).set_z_index(3)
        dt_2=Dot().set_color(REANLEA_GOLDENROD).move_to(ax_1.c2p(2,0)).set_z_index(3)
        dt_3=Dot().set_color(REANLEA_PINK).move_to(ax_1.c2p(3,2)).set_z_index(3)

        dt_3_lbl=MathTex("v").scale(.7).set_color(REANLEA_PINK_LIGHTER).next_to(dt_3,RIGHT)

        ln_0010=Line(start=ax_1.c2p(0,0),end=ax_1.c2p(1,0)).set_stroke(width=4, color=[REANLEA_AQUA,REANLEA_YELLOW_LIGHTER]).set_z_index(2)

        ln_0032=Line(start=ax_1.c2p(0,0),end=ax_1.c2p(3,2)).set_stroke(width=4, color=[REANLEA_PINK,REANLEA_YELLOW])

        dot_0=Dot(radius=0.1, color=REANLEA_AQUA_GREEN).move_to(ax_1.c2p(3,0)).set_sheen(-0.4,DOWN).set_z_index(3)

        dot_1=Dot(radius=0.1, color=REANLEA_BLUE_LAVENDER).move_to(ax_1.c2p(6,0)).set_sheen(-0.4,DOWN).set_z_index(3)

        angl_1=Angle(ln_0010,ln_0032).set_color(REANLEA_YELLOW_GREEN).set_stroke(width=3.5).set_z_index(-1)
        
        angl_1_lbl=MathTex(r"\theta").scale(.4).set_color(REANLEA_YELLOW_GREEN).next_to(angl_1,UR, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2).shift(.25*DOWN).set_z_index(2)


        d_d_line_1=DashedDoubleArrow(
            start=ax_1.c2p(0,-.9), end=ax_1.c2p(3,-.9), dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).set_color_by_gradient(REANLEA_YELLOW_LIGHTER,REANLEA_GREEN_AUQA)

        d_d_line_1_lbl=MathTex(r"\lVert i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").scale(.5).set_color(REANLEA_AQUA_GREEN).next_to(d_d_line_1,RIGHT)

        d_d_line_2=DashedDoubleArrow(
            start=ax_1.c2p(0,-1.35), end=ax_1.c2p(6,-1.35), dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.01, buff=10
        ).set_color_by_gradient(REANLEA_YELLOW_LIGHTER,REANLEA_BLUE_LAVENDER)

        d_d_line_2_lbl=MathTex(r"\lVert 2 \cdot i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").scale(.5).set_color(REANLEA_BLUE_LAVENDER).next_to(d_d_line_2,RIGHT)

        lbl_i=MathTex("i").scale(.45).set_color(REANLEA_AQUA).move_to(ax_1.c2p(1.2,.2))  

        
        
        
        self.add(ax_2, dt_0,dt_1,dt_1_ref,dt_2,dt_3,dt_3_lbl,ln_0032,dot_0,dot_1,angl_1,angl_1_lbl,d_d_line_1,d_d_line_1_lbl,d_d_line_2,d_d_line_2_lbl,lbl_i)



        # MAIN SCENE

        self.wait(3)

        self.play(
            Create(ln_0010)
        )

        self.wait(2)



        xrng = ValueTracker(6.5)
        xrng_min = ValueTracker(1.5)


        ax_2_x=VGroup()
        dt_1_x=VMobject()
        dt_3_x=VMobject()
        ln_0032_x = VMobject()


        def axUpdater(mobj):
            xmin = -xrng_min.get_value()
            xmax = +xrng.get_value()
            #newax =Axes(x_range=[xmin,xmax,10**int(np.log10(xmax)-1)],y_range=[-1,4])
            newax=Axes(
                    x_range=[xmin,xmax,2**int(np.log10(xmax)-1)],
                    y_range=[-1.5,4.5],
                    x_length=(round(config.frame_width)-2)*8/7,
                    y_length=(round(config.frame_width)-2)*6/7,
                    tips=False, 
                    axis_config={
                            "font_size": 24,
                        }, 
                ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)
            newax.add_coordinates()
            newax_ref=newax.copy()
            newax.shift((ax_1.c2p(0,0)[0]-newax_ref.c2p(0,0)[0])*RIGHT)

            newfunc = Line(start=newax.c2p(0,0),end=newax.c2p(3,2)).set_stroke(width=4, color=[REANLEA_PINK,REANLEA_YELLOW])
            
            new_dt_1=Dot().set_color(REANLEA_AQUA).move_to(newax.c2p(1,0)).set_z_index(3)

            new_dt_3=Dot().set_color(REANLEA_PINK).move_to(newax.c2p(3,2)).set_z_index(3)

            mobj.become(newax)
            ln_0032_x.become(newfunc)  
            dt_1_x.become(new_dt_1).set_z_index(3)
            dt_3_x.become(new_dt_3)  

        ax_2_x.add_updater(axUpdater)

        self.add(ax_2_x,ln_0032_x,dt_1_x,dt_3_x)
        
        self.play(
            FadeOut(ax_2),
            FadeOut(dt_1)
        )
        
        self.play(
            AnimationGroup(
                AnimationGroup(
                    xrng.animate.set_value(3.25),
                    xrng_min.animate.set_value(.75),
                ),
                AnimationGroup(
                    Flash(point=Dot().move_to(ax_1.c2p(2,0)), color=REANLEA_BLUE_LAVENDER),
                    dt_2.animate.set_color(REANLEA_CYAN_LIGHT)
                ),
                lag_ratio=.5
            ),
            run_time=2
        )

        self.wait(2)

        ln_dis_1=Line(ax_1.c2p(6,2),ax_1.c2p(6,0))
        ln_dis_2=Line(ax_1.c2p(3,2),ax_1.c2p(3,0))


        dissipating_dt_1=Dot().move_to(ax_1.c2p(3,2)).set_opacity(opacity=0)
        dissipating_path_1 = TracedPath(dissipating_dt_1.get_center, dissipating_time=0.5, stroke_color=[REANLEA_BLUE_LAVENDER],stroke_opacity=[1, 0])
        self.add(dissipating_dt_1,dissipating_path_1)

        dissipating_dt_2=Dot().move_to(ax_1.c2p(6,2)).set_opacity(opacity=0)
        dissipating_path_2 = TracedPath(dissipating_dt_2.get_center, dissipating_time=0.5, stroke_color=[REANLEA_AQUA],stroke_opacity=[1, 0])
        self.add(dissipating_dt_2,dissipating_path_2)


        self.play(
            AnimationGroup(
                MoveAlongPath(dissipating_dt_1,ln_dis_1),
                Flash(point=Dot().move_to(ax_1.c2p(3,0)), color=REANLEA_GREEN_AUQA),
                lag_ratio=0.5
            ),
            AnimationGroup(
                MoveAlongPath(dissipating_dt_2,ln_dis_2),
                Flash(point=Dot().move_to(ax_1.c2p(6,0)), color=REANLEA_BLUE_LAVENDER),
                lag_ratio=0.5
            )
        )

        self.wait(2)

        

        self.play(
            Indicate(dt_3),
            Indicate(dt_1_ref),
            Wiggle(angl_1),
            Indicate(angl_1_lbl),            
        )

        self.play(
            Circumscribe(VGroup(d_d_line_1,d_d_line_1_lbl))
        )
        self.wait(2)

        txt_v_0=MathTex("v",",").set_color(REANLEA_PINK_LIGHTER).move_to(5.25*LEFT+2*UP)
        txt_v_0[1].set_color(REANLEA_TXT_COL)
        txt_i_0=MathTex("i").set_color(REANLEA_AQUA).move_to(4.9*LEFT+2.1*UP)
        txt_com_0=MathTex(",").set_color(REANLEA_TXT_COL).next_to(txt_i_0,RIGHT, buff=.1).shift(.18*DOWN)
        txt_th_0=MathTex(r"\theta").set_color(REANLEA_YELLOW).move_to(4.45*LEFT+2.1*UP)

        txt_v_1=txt_v_0.copy()
        txt_i_1=txt_i_0.copy()

        txt_lbl_grp_0=VGroup(txt_v_0,txt_i_0,txt_com_0,txt_th_0).scale(1.5).move_to(4.9*LEFT+2.1*UP)
        txt_lbl_grp_1=VGroup(txt_v_1,txt_i_1).scale(1.5).move_to(4.9*LEFT+2.1*UP)
    
        self.play(           
            AnimationGroup(
                Indicate(dt_3),
            ReplacementTransform(dt_3.copy(),txt_v_0[0]),
            lag_ratio=.1
            )
        )
        self.play(           
            AnimationGroup(
                Indicate(dt_1_ref),
            ReplacementTransform(dt_1_ref.copy(),txt_i_0[0]),
            lag_ratio=.1
            )
        )
        self.play(           
            AnimationGroup(
                Indicate(angl_1_lbl),
            ReplacementTransform(angl_1_lbl.copy(),txt_th_0),
            lag_ratio=.1
            )
        )
        self.play(
            FadeIn(txt_v_0[1]),
            FadeIn(txt_com_0)
        )
        self.wait()
        

        arr_1=Arrow(max_tip_length_to_length_ratio=.1, color=REANLEA_BLUE).rotate(-90*DEGREES).next_to(txt_i_0,DOWN).set_stroke(width=2, color=[REANLEA_BLUE,REANLEA_BLUE_SKY]).shift(.35*DOWN).scale(.85)

        self.play(
            Indicate(txt_lbl_grp_0),
            Create(arr_1),
            run_time=2
        )
        self.wait()

        txt_scl_1=MathTex(r"\lVert i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").set_color(REANLEA_AQUA_GREEN).next_to(arr_1,DOWN, buff=.1).shift(.35*DOWN)

        self.play(
            ReplacementTransform(d_d_line_1_lbl.copy(),txt_scl_1)
        )
        self.wait(2)


        self.play(
            TransformMatchingShapes(txt_lbl_grp_0,txt_lbl_grp_1)
        )
        self.wait(2)

        indct_ln_1=Line().set_stroke(width=2,color=REANLEA_GREY).scale(.25).rotate(PI/6).next_to(txt_lbl_grp_1, RIGHT).shift(.2*UP).set_z_index(2)

        self.play(
            Write(indct_ln_1)
        )

        with RegisterFont("Reenie Beanie") as fonts:
            txt_1=Text("Vectors", font=fonts[0]).scale(.75).set_color(REANLEA_CYAN_LIGHT).next_to(indct_ln_1).shift(.35*UP)

        
        self.play(
            Create(txt_1)
        )

        indct_ln_2=Line().set_stroke(width=2,color=REANLEA_GREY).scale(.25).rotate(-PI/6).next_to(txt_scl_1, DOWN).shift(.2*RIGHT).set_z_index(2)

        self.play(
            Create(indct_ln_2)
        )

        with RegisterFont("Reenie Beanie") as fonts:
            txt_2=Text("Scalars", font=fonts[0]).scale(.75).set_color(REANLEA_CYAN_LIGHT).next_to(indct_ln_2).shift(.25*DOWN)

        
        self.play(
            Create(txt_2)
        )
        self.wait(2)

        innr_prdct_sym=MathTex(r"\langle ,\rangle").set_color(REANLEA_BLUE_LAVENDER).next_to(arr_1,RIGHT)

        self.play(
            Create(innr_prdct_sym)
        )

        self.wait(2)

        innr_prdct_grp_0=VGroup(txt_lbl_grp_1,arr_1,txt_scl_1,innr_prdct_sym,indct_ln_1,indct_ln_2,txt_1,txt_2)

        self.play(
            innr_prdct_grp_0.animate.scale(.5).shift(.5*LEFT+.5*UP),
        )

        sr_innr_prdct_grp_0=SurroundingRectangle(innr_prdct_grp_0, color=REANLEA_PURPLE_LIGHTER, buff=.25, corner_radius=.125).set_opacity(.25)

        self.play(
            Write(sr_innr_prdct_grp_0)
        )

        innr_prdct_dfn_0=MathTex(r"\langle , \rangle",":",r"V \times V","\longrightarrow",r"\mathbb{R").scale(.65).set_color_by_gradient(REANLEA_CYAN_LIGHT).move_to(3*UP+3*RIGHT)
        innr_prdct_dfn_0_0=innr_prdct_dfn_0[0].copy().set_z_index(-1)

        with RegisterFont("Courier Prime") as fonts:
            innr_prdct_dfn_1=Text("by", font=fonts[0]).scale(.35).set_color(REANLEA_CYAN_LIGHT).next_to(innr_prdct_dfn_0, RIGHT).shift(.1*RIGHT+.02*DOWN)
        
        innr_prdct_dfn_2=MathTex(r"\langle u,v \rangle","=",r"\lVert u \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").scale(.7).set_color_by_gradient(REANLEA_CYAN_LIGHT).move_to(2.5*UP+3.25*RIGHT)

        innr_prdct_dfn_3=MathTex(r"\langle i,v \rangle","=",r"\lVert i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").scale(.7).set_color_by_gradient(REANLEA_CYAN_LIGHT).move_to(2.5*UP+3.25*RIGHT)

        innr_prdct_dfn_3_ref=innr_prdct_dfn_3[0].copy().set_color(REANLEA_AQUA)

        
        innr_prdct_dfn_grp_1=VGroup(innr_prdct_dfn_0,innr_prdct_dfn_1)

        innr_prdct_dfn_grp_2=VGroup(innr_prdct_dfn_grp_1,innr_prdct_dfn_2,innr_prdct_dfn_3).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_AQUA)

        self.play(
            ReplacementTransform(innr_prdct_sym.copy(),innr_prdct_dfn_0_0),
            Write(innr_prdct_dfn_grp_1)
        )

        self.play(
            Write(innr_prdct_dfn_2)
        )

        self.play(
            FadeOut(innr_prdct_dfn_0_0)
        )
        self.wait(2)

        self.play(
            ReplacementTransform(innr_prdct_dfn_2,innr_prdct_dfn_3)
        )
        self.wait(2)

        d_d_line_1_ref=d_d_line_1.copy().set_z_index(4).save_state()

        d_d_line_1_ref_1=d_d_line_1.copy().shift((ax_1.c2p(0,0)[1]-ax_1.c2p(0,-.9)[1])*UP)

        self.play(
            d_d_line_1_ref.animate.shift(
                (
                    ax_1.c2p(0,0)[1]-ax_1.c2p(0,-.9)[1]
                )*UP
            )
        )

        self.play(
            Indicate(innr_prdct_dfn_2[2], color=PURE_RED)
        )
        self.play(
            Indicate(innr_prdct_dfn_2[4], color=PURE_RED)
        )
        self.play(
            Indicate(innr_prdct_dfn_2[6][3], color=PURE_RED)
        )

        self.wait(2)

        rot_tracker=ValueTracker(0)
        ln_grp=VGroup(ln_0010,ln_0032,dt_1,dt_3).copy()

        ln_grp_x=ln_grp.copy()
        
        ln_grp_0=VGroup(d_d_line_1_ref).copy()
        ln_grp += ln_grp_0

        ln_grp_ref=ln_grp.copy()

        ln_grp.add_updater(
            lambda x : x.become(ln_grp_ref.copy()).rotate(
                rot_tracker.get_value(), about_point=ax_1.c2p(0,0)
            )
        )


        self.add(ln_grp)

        self.play(
            FadeOut(d_d_line_1_ref)
        )
        self.play(
            rot_tracker.animate.set_value(PI/2),
            run_time=2
        )
        self.wait()

        self.play(
            rot_tracker.animate.set_value(PI*2),
            run_time=8
        )

        self.wait()

        self.add(ln_grp_x, d_d_line_1_ref_1)
        
        
        self.play(
            FadeOut(ln_grp)
        )

        # sided rectangle appear

        rect_overlap=Rectangle(width=10.25, height=9, color=REANLEA_BACKGROUND_COLOR).to_edge(RIGHT, buff=0).set_opacity(.825).set_z_index(10)

        #self.add(rect_overlap)
        self.play(
            Create(rect_overlap)
        )

        ax_4=Axes(
            x_range=[-3.5,3.5],
            y_range=[-1.5,3.5],
            y_length=(round(config.frame_width)-2)*5/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                "include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).move_to(rect_overlap.get_center()).set_z_index(11)

        self.play(
            Write(ax_4)
        )

        self.wait(2)

        sgn_pos_1=MathTex("+").scale(.75).set_color(PURE_GREEN)#.move_to(6.5*RIGHT)
        sgn_pos_2=Circle(radius=0.2, color=PURE_GREEN).move_to(sgn_pos_1.get_center()).set_stroke(width= 1)
        sgn_pos=VGroup(sgn_pos_1,sgn_pos_2).move_to(ax_4.c2p(3.25,1.25)).scale(.6).set_z_index(11)

        sgn_neg_1=MathTex("-").scale(.75).set_color(REANLEA_YELLOW)#.move_to(6.5*LEFT)
        sgn_neg_2=Circle(radius=0.2, color=REANLEA_YELLOW).move_to(sgn_neg_1.get_center()).set_stroke(width= 1)
        sgn_neg=VGroup(sgn_neg_1,sgn_neg_2).move_to(ax_4.c2p(-3.25,1.25)).scale(.6).set_z_index(11)

        sgn_grp=VGroup(sgn_pos,sgn_neg)

        tracker = ValueTracker(0.5)

        graph_ref = always_redraw(lambda: ax_4.plot(
            lambda t,tracker=tracker: np.cos(tracker.get_value() * t),
            
        ).set_stroke(width=7, color=[REANLEA_AQUA,REANLEA_BLUE,REANLEA_AQUA])).set_z_index(11)

        graph_ref_lbl_0=MathTex(r"cos(\theta)").set_color_by_gradient(REANLEA_BLUE,REANLEA_AQUA).scale(.75).move_to(ax_4.c2p(1.75,1.65)).set_z_index(11)

        graph_ref_lbl_1=MathTex("=",r"cos(-\theta)").set_color_by_gradient(REANLEA_BLUE,REANLEA_AQUA).scale(.75).next_to(graph_ref_lbl_0,RIGHT, buff=.1).set_z_index(11)

        graph_ref_lbl=VGroup(graph_ref_lbl_0,graph_ref_lbl_1)


        self.play(
            Create(graph_ref),
            Write(graph_ref_lbl_0)
        )

        self.play(
            Write(sgn_grp)
        )

        self.play(tracker.animate(run_time=6).set_value(5))

        graph=ax_4.plot(
            lambda x: np.cos(5*x)
        ).set_stroke(width=7, color=[REANLEA_AQUA,REANLEA_BLUE,REANLEA_AQUA]).set_z_index(11)


        self.add(graph)
        self.play(
            FadeOut(graph_ref)
        )

        self.wait(2)

        xt_l=ValueTracker(-3.5)
        xt_r=ValueTracker(3.5)

        initial_point_l= [ax_4.coords_to_point(xt_l.get_value(), np.cos(xt_l.get_value()*5))]
        dot_l = Dot(point=initial_point_l, color=graph.get_color()).set_z_index(11)

        initial_point_r= [ax_4.coords_to_point(xt_r.get_value(), np.cos(xt_r.get_value()*5))]
        dot_r = Dot(point=initial_point_r, color=graph.get_color()).set_z_index(11)


        self.play(
            FadeIn(dot_l),
            FadeIn(dot_r)
        )
        self.wait()

        self.play(
            Write(graph_ref_lbl_1)
        )
        self.wait()

        self.play(
            graph_ref_lbl.animate.shift(UP)
        )

        d_line=DashedLine(
            start=dot_l.get_center(), end=dot_r.get_center(),stroke_width=2
        ).set_color_by_gradient(dot_l.get_color()).set_z_index(11)

        dot_c=Dot(point=(dot_l.get_center()+dot_r.get_center())/2, color=PURE_RED).set_z_index(14)

        self.play(
            AnimationGroup(
                Create(d_line),
                Flash(point=(dot_l.get_center()+dot_r.get_center())/2, color=REANLEA_GOLD),
                lag_ratio=.4
            ),
            AnimationGroup(                
                FadeIn(dot_c)
            ),
            lag_ratio=.05,
            run_time=2.5
        )
        
        value=DecimalNumber(np.cos(xt_r.get_value()*5)).set_color_by_gradient(PURE_RED).set_sheen(-0.1,LEFT).scale(.75).move_to(ax_4.c2p(5,np.cos(xt_r.get_value()*5))).set_z_index(11)

        self.play(
            Write(value)
        )

        self.wait(2)

        dot_l.add_updater(lambda x: x.move_to(ax_4.c2p(xt_l.get_value(), np.cos(xt_l.get_value()*5))).set_z_index(11))

        dot_r.add_updater(lambda x: x.move_to(ax_4.c2p(xt_r.get_value(), np.cos(xt_r.get_value()*5))).set_z_index(11))

        d_line.add_updater(
            lambda z : z.become(
                DashedLine(
                    start=dot_l.get_center(), end=dot_r.get_center(),stroke_width=2
                ).set_color_by_gradient(dot_l.get_color()).set_z_index(11)
            )
        )

        dot_c.add_updater(lambda x: x.move_to((dot_l.get_center()+dot_r.get_center())/2).set_z_index(12))

        value.add_updater(
            lambda x : x.set_value(np.cos(xt_r.get_value()*5)).move_to(ax_4.c2p(5,np.cos(xt_r.get_value()*5))).set_color(PURE_RED).set_z_index(11))

        self.play(
            xt_l.animate.set_value(0),
            xt_r.animate.set_value(0),
            run_time=10
        )
        self.wait(2)

        graph_l=ax_4.plot(
            lambda x: np.cos(5*x) , x_range=[-3.5,0]
        ).set_stroke(width=7, color=[REANLEA_BLUE,REANLEA_AQUA]).set_z_index(11)

        graph_r=ax_4.plot(
            lambda x: np.cos(5*x) , x_range=[0,3.5]
        ).set_stroke(width=7, color=[REANLEA_AQUA,REANLEA_BLUE]).set_z_index(11)
        

        self.add(graph_l,graph_r)

        self.play(
            AnimationGroup(
                FadeOut(graph),
                graph_l.animate.flip(about_point=ax_4.c2p(0,.5)),
                lag_ratio=.5
            )
        )

        self.wait(2)

        innr_prdct_dfn_4=MathTex("=",r"\lVert i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos(-\theta)").scale(.7).set_color_by_gradient(REANLEA_AQUA).move_to(2*UP+3.92*RIGHT)
        innr_prdct_dfn_4_ref=innr_prdct_dfn_4.copy()

        uncrt_grp_0=VGroup(sgn_grp,graph_l,graph_r,ax_4,dot_l,dot_r,dot_c,d_line,value,graph_ref_lbl)

        graph_ref_lbl_1_copy=graph_ref_lbl_1.copy()[1]

        self.play(
            AnimationGroup(
                FadeOut(uncrt_grp_0),
                Uncreate(rect_overlap)
            ),
            AnimationGroup(
              ReplacementTransform(graph_ref_lbl_1_copy,innr_prdct_dfn_4_ref[5]),
              Write(innr_prdct_dfn_4)
            )
        )

        self.play(
            FadeOut(innr_prdct_dfn_4_ref[5])
        )

        self.wait(4)


        # manim -sqk anim2.py Scene4_1

        self.play(
            ln_grp_x.animate.flip(RIGHT, about_point=ax_1.c2p(0,0))
        )

        self.wait(2)

        ln_dis_3=Line(ax_1.c2p(3,-2),ax_1.c2p(3,2))
        ln_dis_4=Line(ax_1.c2p(3,-2),ax_1.c2p(3,0))


        dissipating_dt_3=Dot().move_to(ax_1.c2p(3,-2)).set_opacity(opacity=0)
        dissipating_path_3 = TracedPath(dissipating_dt_3.get_center, dissipating_time=0.5, stroke_color=[REANLEA_PINK_LIGHTER],stroke_opacity=[1, 0])
        self.add(dissipating_dt_3,dissipating_path_3)

        dissipating_dt_4=Dot().move_to(ax_1.c2p(3,-2)).set_opacity(opacity=0)
        dissipating_path_4 = TracedPath(dissipating_dt_4.get_center, dissipating_time=0.5, stroke_color=[REANLEA_AQUA],stroke_opacity=[1, 0])
        self.add(dissipating_dt_4,dissipating_path_4)


        self.play(
            AnimationGroup(
                MoveAlongPath(dissipating_dt_3,ln_dis_3),
                Flash(point=Dot().move_to(ax_1.c2p(3,2)), color=REANLEA_PINK_LIGHTER),
                lag_ratio=0.5
            ),
            AnimationGroup(
                MoveAlongPath(dissipating_dt_4,ln_dis_4),
                Flash(point=Dot().move_to(ax_1.c2p(3,0)), color=REANLEA_GREEN_AUQA),
                lag_ratio=0.5
            )
        )
        self.wait(2)

        innr_prdct_dfn_4_1_ref=innr_prdct_dfn_4[1].copy().move_to(innr_prdct_dfn_4[3].get_center())

        innr_prdct_dfn_4_3_ref=innr_prdct_dfn_4[3].copy().move_to(innr_prdct_dfn_4[1].get_center())

        self.play(
            Transform(innr_prdct_dfn_4[1], innr_prdct_dfn_4_1_ref),
            Transform(innr_prdct_dfn_4[3],innr_prdct_dfn_4_3_ref, path_arc=-180*DEGREES)
        )

        self.wait(1.5)

        innr_prdct_dfn_5=MathTex("=",r"\langle v,i \rangle").scale(.7).set_color_by_gradient(REANLEA_AQUA).move_to(1.5*UP+2.93*RIGHT)

        self.play(
            Write(innr_prdct_dfn_5)
        )


        

        dt_1_neg=Dot().set_color(REANLEA_AQUA).move_to(ax_1.c2p(1,0)).set_z_index(3)

        ln_0010_neg=Line(start=ax_1.c2p(0,0),end=ax_1.c2p(1,0)).set_stroke(width=4, color=[REANLEA_AQUA,REANLEA_YELLOW_LIGHTER]).set_z_index(2)

        dt_3_neg=Dot().set_color(REANLEA_PINK).move_to(ax_1.c2p(3,-2)).set_z_index(3)

        ln_0032_neg=Line(start=ax_1.c2p(0,0),end=ax_1.c2p(3,-2)).set_stroke(width=4, color=[REANLEA_PINK,REANLEA_YELLOW])


        rot_tracker_neg=ValueTracker(0)

        ln_neg_grp=VGroup(dt_1_neg,dt_3_neg,ln_0010_neg,ln_0032_neg)

        ln_neg_grp_ref=ln_neg_grp.copy()

        ln_neg_grp.add_updater(
            lambda x : x.become(ln_neg_grp_ref.copy()).rotate(
                rot_tracker_neg.get_value(), about_point=ax_1.c2p(0,0)
            )
        )

        self.add(ln_neg_grp)
        self.play(
            FadeOut(ln_grp_x)
        )
        self.wait(2)

        
        self.play(
            AnimationGroup(
                AnimationGroup(
                    xrng.animate.set_value(6.5),
                    xrng_min.animate.set_value(1.5),
                ),
                AnimationGroup(
                    dt_2.animate.set_color(REANLEA_GOLDENROD)
                ),
            ),
            Unwrite(d_d_line_1_ref_1.reverse_direction()),
            run_time=2
        )

        self.wait(2)

        self.play(
            rot_tracker_neg.animate.set_value(ln_0032.get_angle()),
            run_time=2
        )
        self.wait()

        ln_0032_neg_ref=ln_0032_neg.copy()
        self.add(ln_0032_neg_ref)

        self.play(
            FadeOut(ln_0032_neg)
        )
        self.play(
            Uncreate(ln_0032_neg_ref)
        )
        self.wait(2)



        ax_2_y=VGroup()
        dt_1_y=VMobject()
        dt_1_neg_y=VMobject()
        ln_0010_neg_y = VMobject()

        xrng_1 = ValueTracker(6.5)
        xrng_min_1 = ValueTracker(1.5)

        def axUpdater_1(mobj):
            xmin = -xrng_min_1.get_value()
            xmax = +xrng_1.get_value()
            #newax =Axes(x_range=[xmin,xmax,10**int(np.log10(xmax)-1)],y_range=[-1,4])
            newax=Axes(
                    x_range=[xmin,xmax,2**int(np.log10(xmax)-1)],
                    y_range=[-1.5,4.5],
                    x_length=(round(config.frame_width)-2)*8/7,
                    y_length=(round(config.frame_width)-2)*6/7,
                    tips=False, 
                    axis_config={
                            "font_size": 24,
                        }, 
                ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)
            newax.add_coordinates()
            newax_ref=newax.copy()
            newax.shift((ax_1.c2p(0,0)[0]-newax_ref.c2p(0,0)[0])*RIGHT)

            newfunc = Line(start=newax.c2p(0,0),end=newax.c2p(.8321,.5547)).set_stroke(width=4, color=[REANLEA_AQUA,REANLEA_YELLOW_LIGHTER])
            
            new_dt_1=Dot().set_color(REANLEA_AQUA).move_to(newax.c2p(1,0)).set_z_index(3)

            new_dt_1_neg=Dot().set_color(REANLEA_AQUA).move_to(newax.c2p(.8321,.5547)).set_z_index(3)

            mobj.become(newax)
            ln_0010_neg_y.become(newfunc)  
            dt_1_y.become(new_dt_1).set_z_index(3)
            dt_1_neg_y.become(new_dt_1_neg)  

        ax_2_y.add_updater(axUpdater_1)

        self.add(ln_0010_neg_y,dt_1_y,dt_1_neg_y)

        self.play(
            FadeIn(ax_2_y),
            FadeOut(ax_2_x)
        )
        
        
        
        self.play(
            AnimationGroup(
                AnimationGroup(
                    xrng_1.animate.set_value(6.5/np.sqrt(13)),
                    xrng_min_1.animate.set_value(1.5/np.sqrt(13)),
                ),
                AnimationGroup(
                    Flash(point=Dot().move_to(ax_1.c2p(np.sqrt(13),0)), color=REANLEA_BLUE_LAVENDER),
                    dt_3_neg.animate.set_color(REANLEA_CYAN_LIGHT)
                ),
                lag_ratio=.5
            ),
            run_time=2
        )

        self.wait(2)

        

        ln_dis_5=Line(ax_1.c2p(3,2),ax_1.c2p(3,0))
        ln_dis_6=Line(ax_1.c2p(3,.5547),ax_1.c2p(3,0))


        dissipating_dt_5=Dot().move_to(ax_1.c2p(3,2)).set_opacity(opacity=0)
        dissipating_path_5 = TracedPath(dissipating_dt_5.get_center, dissipating_time=0.5, stroke_color=[REANLEA_PINK_LIGHTER],stroke_opacity=[1, 0])
        self.add(dissipating_dt_5,dissipating_path_5)

        dissipating_dt_6=Dot().move_to(ax_1.c2p(3,.5547)).set_opacity(opacity=0)
        dissipating_path_6 = TracedPath(dissipating_dt_6.get_center, dissipating_time=0.5, stroke_color=[REANLEA_AQUA],stroke_opacity=[1, 0])
        self.add(dissipating_dt_6,dissipating_path_6)


        self.play(
            AnimationGroup(
                MoveAlongPath(dissipating_dt_5,ln_dis_5),
                Flash(point=Dot().move_to(ax_1.c2p(3,0)), color=REANLEA_PINK_LIGHTER),
                lag_ratio=0.5
            ),
            AnimationGroup(
                MoveAlongPath(dissipating_dt_6,ln_dis_6),
                Flash(point=Dot().move_to(ax_1.c2p(3,0)), color=REANLEA_GREEN_AUQA),
                lag_ratio=0.5
            )
        )
        self.wait()

        d_line_1=DashedLine(
            start=ax_1.c2p(3,2), end=ax_1.c2p(3,0),stroke_width=2
        ).set_color_by_gradient(REANLEA_AQUA,REANLEA_BLUE_SKY).set_z_index(5)

        self.play(
            Write(d_line_1)
        )

        d_d_line_1_ref_2=d_d_line_1.copy().set_z_index(5)

        self.add(d_d_line_1_ref_2)

        self.play(
            d_d_line_1_ref_2.animate.shift((ax_1.c2p(0,0)[1]-ax_1.c2p(0,-.9)[1])*UP).set_color(REANLEA_RED)         
        )

        self.play(
            innr_prdct_dfn_3_ref.animate.shift(1*DOWN)
        )

        self.wait()

        innr_prdct_dfn_5_ref=innr_prdct_dfn_5.copy()

        innr_prdct_dfn_3_ref_5_ref_grp=VGroup(innr_prdct_dfn_3_ref,innr_prdct_dfn_5_ref)

        self.play(
            innr_prdct_dfn_3_ref_5_ref_grp.animate.shift(1.25*DOWN+1.5*RIGHT).scale(1.25).set_color_by_gradient(REANLEA_AQUA,REANLEA_PURPLE_LIGHTER)
        )

        sr_innr_prdct_dfn_3_ref_5_ref_grp=SurroundingRectangle(innr_prdct_dfn_3_ref_5_ref_grp, buff=.25,corner_radius=.125, color=REANLEA_WELDON_BLUE)

        self.play(
            Create(sr_innr_prdct_dfn_3_ref_5_ref_grp)
        )


        indct_ln_3=Line().set_stroke(width=2,color=REANLEA_GREY).scale(.3).rotate(-PI/4).next_to(sr_innr_prdct_dfn_3_ref_5_ref_grp,DOWN).shift(.4*UP).set_z_index(2)

        self.play(
            Write(indct_ln_3)
        )

        with RegisterFont("Reenie Beanie") as fonts:
            txt_3=Text("Symmetric", font=fonts[0]).scale(.75).set_color(REANLEA_CYAN_LIGHT).next_to(indct_ln_3).shift(.45*DOWN+.75*LEFT)

        
        self.play(
            Create(txt_3)
        )

        self.wait(2)

        bulet_1=Dot(point=[-5.9,-1,0], radius=DEFAULT_DOT_RADIUS/1.25, color=REANLEA_WHITE).set_sheen(-.4,DOWN)
        txt_4=txt_3.copy()

        self.play(
            Write(bulet_1)
        )

        self.play(
            txt_4.animate.next_to(bulet_1, RIGHT)
        )

        self.wait(4)

        self.play(
            AnimationGroup(
                Unwrite(txt_3),
                Uncreate(indct_ln_3)
            )
        )
        self.wait()

        ln_0010_neg_ref=ln_0010_neg.copy()
        self.add(ln_0010_neg_ref)

        self.play(
            FadeOut(ln_0010_neg)
        )
        
        dt_1_neg_ref=dt_1_neg.copy()
        self.add(dt_1_neg_ref)
        self.play(
            FadeOut(dt_1_neg)
        )

        self.play(
            AnimationGroup(
                Uncreate(sr_innr_prdct_dfn_3_ref_5_ref_grp),
                Unwrite(innr_prdct_dfn_3_ref_5_ref_grp),
                FadeOut(innr_prdct_dfn_4),
                FadeOut(innr_prdct_dfn_5)
            ),
            AnimationGroup(
                AnimationGroup(
                    xrng_1.animate.set_value(6.5),
                    xrng_min_1.animate.set_value(1.5),
                )
            ),
            AnimationGroup(
                FadeOut(d_d_line_1_ref_2),
                dt_3_neg.animate.move_to(ax_1.c2p(1,0)).set_opacity(0)
            )
        )

        self.play(
            AnimationGroup(
                Uncreate(ln_0010_neg_ref),
                FadeOut(dt_1_neg_ref)
            )
        )

        ln_0010_neg_y_ref=ln_0010_neg_y.copy()
        dt_1_neg_y_ref=dt_1_neg_y.copy()

        self.add(ln_0010_neg_y_ref,dt_1_neg_y_ref)

        self.play(
            FadeOut(ln_0010_neg_y),
            FadeOut(dt_1_neg_y)
        )

        self.play(
            FadeOut(dt_1_neg_y_ref),
            Uncreate(ln_0010_neg_y_ref)
        )
        



        
        
        

        self.wait(4)


        

    # manim -pqh anim2.py Scene4_1

    # manim -pql anim2.py Scene4_1

    # manim -sqk anim2.py Scene4_1
        



class Scene4_2(Scene):
    def construct(self):


        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        


        # PREVIOUS SCENE

        ax_1_prev=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)
 
        ax_1=Axes(
            x_range=[-1.5,6.5],
            y_range=[-1.5,4.5],
            x_length =(round(config.frame_width)-2)*8/7,
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)
        ax_1.add_coordinates()


        ax_1_ref=ax_1.copy()
        ax_1.shift((ax_1_prev.c2p(0,0)[0]-ax_1_ref.c2p(0,0)[0])*RIGHT)

        dt_0=Dot().set_color(REANLEA_YELLOW).move_to(ax_1.c2p(0,0)).set_z_index(5)
        dt_1=Dot().set_color(REANLEA_AQUA).move_to(ax_1.c2p(1,0)).set_z_index(3)
        dt_2=Dot().set_color(REANLEA_GOLDENROD).move_to(ax_1.c2p(2,0)).set_z_index(3)
        dt_3=Dot().set_color(REANLEA_PINK).move_to(ax_1.c2p(3,2)).set_z_index(3)

        dt_3_lbl=MathTex("v").scale(.7).set_color(REANLEA_PINK_LIGHTER).next_to(dt_3,RIGHT)

        ln_0010=Line(start=ax_1.c2p(0,0),end=ax_1.c2p(1,0)).set_stroke(width=4, color=[REANLEA_AQUA,REANLEA_YELLOW_LIGHTER]).set_z_index(2)

        ln_0032=Line(start=ax_1.c2p(0,0),end=ax_1.c2p(3,2)).set_stroke(width=4, color=[REANLEA_PINK,REANLEA_YELLOW])

        dot_0=Dot(radius=0.1, color=REANLEA_AQUA_GREEN).move_to(ax_1.c2p(3,0)).set_sheen(-0.4,DOWN).set_z_index(3)

        dot_1=Dot(radius=0.1, color=REANLEA_BLUE_LAVENDER).move_to(ax_1.c2p(6,0)).set_sheen(-0.4,DOWN).set_z_index(3)

        angl_1=Angle(ln_0010,ln_0032).set_color(REANLEA_YELLOW_GREEN).set_stroke(width=3.5).set_z_index(-1)
        
        angl_1_lbl=MathTex(r"\theta").scale(.4).set_color(REANLEA_YELLOW_GREEN).next_to(angl_1,UR, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2).shift(.25*DOWN).set_z_index(2)

        d_d_line_1=DashedDoubleArrow(
            start=ax_1.c2p(0,-.9), end=ax_1.c2p(3,-.9), dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).set_color_by_gradient(REANLEA_YELLOW_LIGHTER,REANLEA_GREEN_AUQA)

        d_d_line_1_lbl=MathTex(r"\lVert i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").scale(.5).set_color(REANLEA_AQUA_GREEN).next_to(d_d_line_1,RIGHT)

        d_d_line_2=DashedDoubleArrow(
            start=ax_1.c2p(0,-1.35), end=ax_1.c2p(6,-1.35), dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.01, buff=10
        ).set_color_by_gradient(REANLEA_YELLOW_LIGHTER,REANLEA_BLUE_LAVENDER)

        d_d_line_2_lbl=MathTex(r"\lVert 2 \cdot i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").scale(.5).set_color(REANLEA_BLUE_LAVENDER).next_to(d_d_line_2,RIGHT)


        innr_prdct_dfn_0=MathTex(r"\langle , \rangle",":",r"V \times V","\longrightarrow",r"\mathbb{R").scale(.65).set_color_by_gradient(REANLEA_CYAN_LIGHT).move_to(3*UP+3*RIGHT)

        with RegisterFont("Courier Prime") as fonts:
            innr_prdct_dfn_1=Text("by", font=fonts[0]).scale(.35).set_color(REANLEA_CYAN_LIGHT).next_to(innr_prdct_dfn_0, RIGHT).shift(.1*RIGHT+.02*DOWN)
        
        innr_prdct_dfn_2=MathTex(r"\langle i,v \rangle","=",r"\lVert i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").scale(.7).set_color_by_gradient(REANLEA_CYAN_LIGHT).move_to(2.5*UP+3.25*RIGHT)

        innr_prdct_dfn=VGroup(innr_prdct_dfn_0,innr_prdct_dfn_1,innr_prdct_dfn_2).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_AQUA)

        innr_prdct_dfn_2.set_color(REANLEA_AQUA)

        d_line_1=DashedLine(
            start=ax_1.c2p(3,2), end=ax_1.c2p(3,0),stroke_width=2
        ).set_color_by_gradient(REANLEA_AQUA,REANLEA_BLUE_SKY).set_z_index(5)

        bulet_1=Dot(point=[-5.9,-1,0], radius=DEFAULT_DOT_RADIUS/1.25, color=REANLEA_WHITE).set_sheen(-.4,DOWN)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_sym_1=Text("Symmetric", font=fonts[0]).scale(.75).set_color(REANLEA_CYAN_LIGHT).next_to(bulet_1,RIGHT)

        lbl_i=MathTex("i").scale(.45).set_color(REANLEA_AQUA).move_to(ax_1.c2p(1.2,.2))  


        self.add(ax_1,dt_0,dt_1,dt_2,dt_3,dt_3_lbl,ln_0010,ln_0032,dot_0,dot_1,angl_1,angl_1_lbl,d_d_line_1,d_d_line_1_lbl,d_d_line_2,d_d_line_2_lbl,innr_prdct_dfn,d_line_1, txt_sym_1,bulet_1, lbl_i)



        txt_v_0=MathTex("v",",").set_color(REANLEA_PINK_LIGHTER).move_to(5.25*LEFT+2*UP)
        txt_v_0[1].set_color(REANLEA_TXT_COL)
        txt_i_0=MathTex("i").set_color(REANLEA_AQUA).move_to(4.9*LEFT+2.1*UP)
        txt_com_0=MathTex(",").set_color(REANLEA_TXT_COL).next_to(txt_i_0,RIGHT, buff=.1).shift(.18*DOWN)
        txt_th_0=MathTex(r"\theta").set_color(REANLEA_YELLOW).move_to(4.45*LEFT+2.1*UP)

        txt_v_1=txt_v_0.copy()
        txt_i_1=txt_i_0.copy()

        txt_lbl_grp_0=VGroup(txt_v_0,txt_i_0,txt_com_0,txt_th_0).scale(1.5).move_to(4.9*LEFT+2.1*UP)
        txt_lbl_grp_1=VGroup(txt_v_1,txt_i_1).scale(1.5).move_to(4.9*LEFT+2.1*UP)
        

        arr_1=Arrow(max_tip_length_to_length_ratio=.1, color=REANLEA_BLUE).rotate(-90*DEGREES).next_to(txt_i_0,DOWN).set_stroke(width=2, color=[REANLEA_BLUE,REANLEA_BLUE_SKY]).shift(.35*DOWN).scale(.85)

        self.add(arr_1)

        txt_scl_1=MathTex(r"\lVert i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").set_color(REANLEA_AQUA_GREEN).next_to(arr_1,DOWN, buff=.1).shift(.35*DOWN)

        self.add(txt_scl_1)

        '''self.play(
            TransformMatchingShapes(txt_lbl_grp_0,txt_lbl_grp_1)
        )'''

        self.add(txt_lbl_grp_1)

        indct_ln_1=Line().set_stroke(width=2,color=REANLEA_GREY).scale(.25).rotate(PI/6).next_to(txt_lbl_grp_1, RIGHT).shift(.2*UP).set_z_index(2)

        self.add(indct_ln_1)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_1=Text("Vectors", font=fonts[0]).scale(.75).set_color(REANLEA_CYAN_LIGHT).next_to(indct_ln_1).shift(.35*UP)
        
        self.add(txt_1)

        indct_ln_2=Line().set_stroke(width=2,color=REANLEA_GREY).scale(.25).rotate(-PI/6).next_to(txt_scl_1, DOWN).shift(.2*RIGHT).set_z_index(2)

        self.add(indct_ln_2)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_2=Text("Scalars", font=fonts[0]).scale(.75).set_color(REANLEA_CYAN_LIGHT).next_to(indct_ln_2).shift(.25*DOWN)

        self.add(txt_2)
        
        innr_prdct_sym=MathTex(r"\langle ,\rangle").set_color(REANLEA_BLUE_LAVENDER).next_to(arr_1,RIGHT)

        self.add(innr_prdct_sym)

        innr_prdct_grp_0=VGroup(txt_lbl_grp_1,arr_1,txt_scl_1,innr_prdct_sym,indct_ln_1,indct_ln_2,txt_1,txt_2).scale(.5).shift(.5*LEFT+.5*UP)

        sr_innr_prdct_grp_0=SurroundingRectangle(innr_prdct_grp_0, color=REANLEA_PURPLE_LIGHTER, buff=.25, corner_radius=.125).set_opacity(.25)

        self.add(sr_innr_prdct_grp_0)


        # MAIN SCENE

        innr_prdct_add_0=innr_prdct_dfn_2[0].copy()
        self.play(
            Indicate(innr_prdct_dfn_2)
        )
        self.play(
            innr_prdct_add_0.animate.shift(DOWN)
        )
        self.wait()

        vects_grp_old=VGroup(dt_0,dt_1,dt_3,ln_0010,ln_0032)

        dt_0_x=dt_0.copy().set_z_index(2)
        dt_1_x=dt_1.copy()
        dt_3_x=dt_3.copy()
        ln_0010_x=ln_0010.copy()
        ln_0032_x=ln_0032.copy()
        

        vects_grp_0=VGroup(dt_0_x,dt_1_x,dt_3_x,ln_0010_x,ln_0032_x)

        innr_prdct_plus_0=MathTex("+").scale(.5).set_color(REANLEA_AQUA).next_to(innr_prdct_add_0,RIGHT)

        innr_prdct_add_1=innr_prdct_add_0.copy()

        self.play(
            AnimationGroup(
                innr_prdct_add_1.animate.next_to(innr_prdct_plus_0,RIGHT),
                Write(innr_prdct_plus_0),
                lag_ratio=.35
            ),
            AnimationGroup(
                vects_grp_0.animate.shift((ax_1.c2p(3,0)[0]-ax_1.c2p(0,0)[0])*RIGHT),
            )
        )
        self.wait(2)


        innr_prdct_add_grp_0=VGroup(innr_prdct_add_0,innr_prdct_add_1,innr_prdct_plus_0)

        d_line_1_x=DashedLine(
            start=ax_1.c2p(6,2), end=ax_1.c2p(6,0),stroke_width=2
        ).set_color_by_gradient(REANLEA_AQUA,REANLEA_BLUE_SKY).set_z_index(5)

        self.play(
            innr_prdct_add_grp_0.animate.shift(LEFT),
            Create(d_line_1_x)
        )





        innr_prdct_plus_1=MathTex("=").scale(.5).set_color(REANLEA_AQUA).next_to(innr_prdct_add_1,RIGHT)

        innr_prdct_add_2=MathTex(r"2",r"\cdot",r"\langle i,v \rangle").scale(.7).set_color_by_gradient(REANLEA_AQUA).next_to(innr_prdct_plus_1,RIGHT)


        self.play(
            AnimationGroup(
                Write(innr_prdct_add_2),
                Write(innr_prdct_plus_1),
                lag_ratio=.15
            )
        )
        self.wait(2)




        innr_prdct_add_3 = MathTex(r"2",r"\cdot",r"\lVert i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").scale(.7).set_color(REANLEA_AQUA).next_to(innr_prdct_plus_1,RIGHT)

        self.play(
            TransformMatchingShapes(innr_prdct_add_2, innr_prdct_add_3)
        )

        self.wait(2)


        innr_prdct_add_4 = MathTex(r"\lVert 2 \cdot i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").scale(.7).set_color(REANLEA_AQUA).next_to(innr_prdct_plus_1,RIGHT)

        d_d_line_2_x=d_d_line_2.copy().set_z_index(6).save_state()
        self.add(d_d_line_2_x)

        self.play(
            ReplacementTransform(innr_prdct_add_3, innr_prdct_add_4),
            d_d_line_2_x.animate.shift(
                (ax_1.c2p(0,0)[1]-ax_1.c2p(0,-1.35)[1])*UP
            ).set_color(PURE_RED)
        )

        self.wait(2)


        ###  SCALING EFFECT  ###

        pau_1 = ValueTracker(6.5)
        nau_1 = ValueTracker(1.5)

        pau_2 = ValueTracker(6.5)
        nau_2 = ValueTracker(1.5)

        pau_3 = ValueTracker(6.5)
        nau_3 = ValueTracker(1.5)

        pau_4 = ValueTracker(6.5)
        nau_4 = ValueTracker(1.5)


        axs_1=VGroup()
        dt_1_00=VMobject()
        dt_1_10=VMobject()
        dt_1_32=VMobject()
        line_1_0010 = VMobject()
        line_1_0032 = VMobject()
        d_line_1_ver=VMobject()

        axs_2=VGroup()
        dt_2_00=VMobject()
        dt_2_10=VMobject()
        dt_2_32=VMobject()
        line_2_0010 = VMobject()
        line_2_0032 = VMobject()
        d_line_2_ver=VMobject()

        axs_3=VGroup()
        dt_3_10=VMobject()
        dt_3_32=VMobject()
        line_3_0010 = VMobject()
        line_3_0032 = VMobject()

        axs_4=VGroup()


        def axUpdater(mobj):
            xmin_1 = -nau_1.get_value()
            xmax_1 = +pau_1.get_value()

            xmin_2 = -nau_2.get_value()
            xmax_2 = +pau_2.get_value()

            xmin_3 = -nau_3.get_value()
            xmax_3 = +pau_3.get_value()


            axs_1_prev=Axes(
                x_range=[-1.5,5.5],
                y_range=[-1.5,4.5],
                y_length=(round(config.frame_width)-2)*6/7,
                tips=False, 
                axis_config={
                    "font_size": 24,
                    #"include_ticks": False,
                }, 
            ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)


            new_axs_1=Axes(
                    x_range=[xmin_1,xmax_1,2**int(np.log10(xmax_1)-1)],
                    y_range=[-1.5,4.5],
                    x_length =(round(config.frame_width)-2)*8/7,
                    y_length=(round(config.frame_width)-2)*6/7,
                    tips=False,
                    axis_config={
                            "font_size": 24,
                        }, 
                ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_opacity(0).set_z_index(1)

            new_axs_1_ref=new_axs_1.copy()
            new_axs_1.shift((axs_1_prev.c2p(0,0)[0]-new_axs_1_ref.c2p(0,0)[0])*RIGHT)

            new_dt_1_00=Dot().set_color(REANLEA_YELLOW).move_to(new_axs_1.c2p(0,0)).set_z_index(5)
            new_dt_1_10=Dot().set_color(REANLEA_AQUA).move_to(new_axs_1.c2p(1,0)).set_z_index(6)
            new_dt_1_32=Dot().set_color(REANLEA_PINK).move_to(new_axs_1.c2p(3,2)).set_z_index(3)
            new_line_1_0010=Line(start=new_axs_1.c2p(0,0),end=new_axs_1.c2p(1,0)).set_stroke(width=4, color=[REANLEA_AQUA,REANLEA_YELLOW_LIGHTER]).set_z_index(2)
            new_line_1_0032=Line(start=new_axs_1.c2p(0,0),end=new_axs_1.c2p(3,2)).set_stroke(width=4, color=[REANLEA_PINK,REANLEA_YELLOW])
            new_d_line_1_ver=DashedLine(start=new_axs_1.c2p(3,2), end=new_axs_1.c2p(3,0),stroke_width=2).set_color_by_gradient(REANLEA_AQUA,REANLEA_BLUE_SKY).set_z_index(5)


            new_axs_2=Axes(
                    x_range=[xmin_2,xmax_2,2**int(np.log10(xmax_2)-1)],
                    y_range=[-1.5,4.5],
                    x_length =(round(config.frame_width)-2)*8/7,
                    y_length=(round(config.frame_width)-2)*6/7,
                    tips=False,
                    axis_config={
                            "font_size": 24,
                        }, 
                ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_opacity(0).set_z_index(1).shift((axs_1_prev.c2p(0,0)[0]-new_axs_1_ref.c2p(0,0)[0])*RIGHT).shift((new_axs_1.c2p(3,0)[0]-new_axs_1.c2p(0,0)[0])*RIGHT)

            new_dt_2_00=Dot().set_color(REANLEA_YELLOW).move_to(new_axs_2.c2p(0,0)).set_z_index(5)
            new_dt_2_10=Dot().set_color(REANLEA_AQUA).move_to(new_axs_2.c2p(1,0)).set_z_index(5)
            new_dt_2_32=Dot().set_color(REANLEA_PINK).move_to(new_axs_2.c2p(3,2)).set_z_index(3)
            new_line_2_0010=Line(start=new_axs_2.c2p(0,0),end=new_axs_2.c2p(1,0)).set_stroke(width=4, color=[REANLEA_AQUA,REANLEA_YELLOW_LIGHTER]).set_z_index(2)
            new_line_2_0032=Line(start=new_axs_2.c2p(0,0),end=new_axs_2.c2p(3,2)).set_stroke(width=4, color=[REANLEA_PINK,REANLEA_YELLOW])
            new_d_line_2_ver=DashedLine(start=new_axs_2.c2p(3,2), end=new_axs_2.c2p(3,0),stroke_width=2).set_color_by_gradient(REANLEA_AQUA,REANLEA_BLUE_SKY).set_z_index(5)



            new_axs_3=Axes(
                    x_range=[xmin_3,xmax_3,2**int(np.log10(xmax_3)-1)],
                    y_range=[-1.5,4.5],
                    x_length =(round(config.frame_width)-2)*8/7,
                    y_length=(round(config.frame_width)-2)*6/7,
                    tips=False,
                    axis_config={
                            "font_size": 24,
                        }, 
                ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_opacity(0).set_z_index(1).shift((axs_1_prev.c2p(0,0)[0]-new_axs_1_ref.c2p(0,0)[0])*RIGHT)

            

            new_dt_3_10=Dot().set_color(REANLEA_BLUE_SKY).move_to(new_axs_3.c2p(1,0)).set_z_index(5)
            new_dt_3_32=Dot().set_color(REANLEA_PURPLE).move_to(new_axs_3.c2p(3,2)).set_z_index(3)
            new_line_3_0010=Line(start=new_axs_3.c2p(0,0),end=new_axs_3.c2p(1,0)).set_stroke(width=4, color=[REANLEA_BLUE_SKY,REANLEA_YELLOW_LIGHTER]).set_z_index(2)
            new_line_3_0032=Line(start=new_axs_3.c2p(0,0),end=new_axs_3.c2p(3,2)).set_stroke(width=4, color=[REANLEA_PURPLE,REANLEA_YELLOW])

            
            mobj.become(new_axs_1)
            dt_1_00.become(new_dt_1_00)
            dt_1_10.become(new_dt_1_10).set_z_index(3)
            dt_1_32.become(new_dt_1_32)
            line_1_0010.become(new_line_1_0010)
            line_1_0032.become(new_line_1_0032)
            d_line_1_ver.become(new_d_line_1_ver).set_z_index(6)

            axs_2.become(new_axs_2)
            dt_2_00.become(new_dt_2_00)
            dt_2_10.become(new_dt_2_10)
            dt_2_32.become(new_dt_2_32)
            line_2_0010.become(new_line_2_0010)
            line_2_0032.become(new_line_2_0032)
            d_line_2_ver.become(new_d_line_2_ver).set_z_index(6)

            axs_3.become(new_axs_3).set_z_index(-1)
            dt_3_10.become(new_dt_3_10)
            dt_3_32.become(new_dt_3_32)
            line_3_0010.become(new_line_3_0010)
            line_3_0032.become(new_line_3_0032)
        
        axs_1.add_updater(axUpdater)

        def axUpdater_1(mobj):

            xmin_4 = -nau_4.get_value()
            xmax_4 = +pau_4.get_value()

            axs_4_prev=Axes(
                x_range=[-1.5,5.5],
                y_range=[-1.5,4.5],
                y_length=(round(config.frame_width)-2)*6/7,
                tips=False, 
                axis_config={
                    "font_size": 24,
                    #"include_ticks": False,
                }, 
            ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(1)


            new_axs_4=Axes(
                    x_range=[xmin_4,xmax_4,2**int(np.log10(xmax_4)-1)],
                    y_range=[-1.5,4.5],
                    x_length =(round(config.frame_width)-2)*8/7,
                    y_length=(round(config.frame_width)-2)*6/7,
                    tips=False,
                    axis_config={
                            "font_size": 24,
                        }, 
                ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_opacity(1).set_z_index(-1)

            new_axs_4_ref=new_axs_4.copy()
            new_axs_4.shift((axs_4_prev.c2p(0,0)[0]-new_axs_4_ref.c2p(0,0)[0])*RIGHT)

            new_axs_4.add_coordinates()

            mobj.become(new_axs_4).set_z_index(-1)

        axs_4.add_updater(axUpdater_1)


        self.add(axs_1, dt_1_10,dt_1_32,line_1_0010 ,line_1_0032,d_line_1_ver,dt_1_00,
        axs_2,dt_2_10,dt_2_32,line_2_0010 ,line_2_0032,d_line_2_ver,dt_2_00,
        axs_3,dt_3_10,dt_3_32,line_3_0010 ,line_3_0032,
        axs_4)

        self.play(
            FadeOut(ax_1),
            vects_grp_old.animate.set_opacity(.25),
            vects_grp_0.animate.set_opacity(0),
            FadeOut(d_line_1),
            FadeOut(d_line_1_x)
        )



        innr_prdct_add_5=MathTex(r"\langle 2 \cdot i,v \rangle").scale(.7).set_color_by_gradient(REANLEA_AQUA).next_to(innr_prdct_plus_1,RIGHT)


        self.play(
            AnimationGroup(
                AnimationGroup(
                    pau_3.animate.set_value(6.5/2),
                    nau_3.animate.set_value(1.5/2),

                    pau_4.animate.set_value(6.5/2),
                    nau_4.animate.set_value(1.5/2)
                ),
                AnimationGroup(
                    Flash(point=Dot().move_to(ax_1.c2p(2,0)), color=REANLEA_BLUE_LAVENDER),
                    dt_2.animate.set_color(REANLEA_CYAN_LIGHT)
                ),
                lag_ratio=.5
            ),            
            AnimationGroup(
                Restore(d_d_line_2_x),
                TransformMatchingShapes(innr_prdct_add_4,innr_prdct_add_5)
            ),
            run_time=2
        ) 
        self.wait(2)



        innr_prdct_add_6=MathTex(r"2 \cdot",r"\langle i,v \rangle").scale(.7).set_color_by_gradient(REANLEA_AQUA).next_to(innr_prdct_plus_1,LEFT)

        self.play(
            TransformMatchingShapes(innr_prdct_add_grp_0,innr_prdct_add_6)
        )

        self.wait(2)

        innr_prdct_add_grp_1=VGroup(innr_prdct_add_5,innr_prdct_add_6,innr_prdct_plus_1)

        self.play(
            Circumscribe(innr_prdct_add_grp_1, color=REANLEA_YELLOW_CREAM, run_time=1.75)
        )
        self.wait(2)

        lam_1=MathTex(r"\lambda").scale(.7).set_color(REANLEA_AQUA).move_to(innr_prdct_add_5[0][1].get_center())

        lam_2=MathTex(r"\lambda").scale(.7).set_color(REANLEA_AQUA).move_to(innr_prdct_add_6[0][0].get_center())

        uncrt_grp_0=VGroup(d_d_line_1,d_d_line_1_lbl,d_d_line_2,d_d_line_2_lbl,d_d_line_2_x)



        self.play(
            Transform(innr_prdct_add_5[0][1], lam_1),
            Transform(innr_prdct_add_6[0][0], lam_2),
            FadeOut(uncrt_grp_0)
        )

        lam_expli_0=MathTex(r"\lambda = \lambda_{1} + \lambda_{2}").scale(.5).set_color(REANLEA_YELLOW_GREEN).rotate(PI/4).shift(1.75*UP+.5*LEFT)

        self.play(
            Write(lam_expli_0)
        )

        sr_bez_0=get_surround_bezier(lam_expli_0).rotate(PI/4)

        innr_prdct_add_7=MathTex(r"( \lambda_{1} + \lambda_{2})",r"\cdot",r"\langle i,v \rangle","=",r"\langle ( \lambda_{1} + \lambda_{2}) \cdot i,v \rangle").scale(.7).set_color(REANLEA_AQUA).move_to(innr_prdct_add_grp_1.get_center()).shift(.2*DOWN)
        
        lam_val_trac_0=ValueTracker(2)
        lam_txt_0=MathTex(r"\lambda","=").scale(.7).set_color(REANLEA_BLUE_SKY).scale(1.25)
        lam_val_0=DecimalNumber().set_color_by_gradient(REANLEA_PURPLE).set_sheen(-0.1,LEFT)
        lam_val_0.add_updater(
            lambda x : x.set_value(lam_val_trac_0.get_value())
        )
        lam_val_grp_0=VGroup(lam_txt_0, lam_val_0).arrange(RIGHT, buff=0.3).move_to(2*DOWN+5.5*RIGHT).scale(.7)


        lam_val_trac_1=ValueTracker(1)
        lam_txt_1=MathTex(r"\lambda_{1}","=").scale(.7).set_color(REANLEA_AQUA).scale(1.25)
        lam_val_1=DecimalNumber().set_color_by_gradient(REANLEA_PINK).set_sheen(-0.1,LEFT)
        lam_val_1.add_updater(
            lambda x : x.set_value(lam_val_trac_1.get_value())
        )
        lam_val_grp_1=VGroup(lam_txt_1, lam_val_1).arrange(RIGHT, buff=0.3).move_to(2.5*DOWN+5.5*RIGHT).scale(.7)


        lam_val_trac_2=ValueTracker(1)
        lam_txt_2=MathTex(r"\lambda_{2}","=").scale(.7).set_color(REANLEA_AQUA).scale(1.25)
        lam_val_2=DecimalNumber().set_color_by_gradient(REANLEA_PINK).set_sheen(-0.1,LEFT)
        lam_val_2.add_updater(
            lambda x : x.set_value(lam_val_trac_2.get_value())
        )
        lam_val_grp_2=VGroup(lam_txt_2, lam_val_2).arrange(RIGHT, buff=0.3).move_to(3*DOWN+5.5*RIGHT).scale(.7)

        lam_val_grp=VGroup(lam_val_grp_0,lam_val_grp_1,lam_val_grp_2)

        self.play(
            Create(sr_bez_0),
            innr_prdct_add_grp_1.animate.shift(.3*UP).scale(.7).set_color(REANLEA_GREY),
            TransformMatchingShapes(innr_prdct_add_grp_1.copy(),innr_prdct_add_7),
            Write(lam_val_grp)
        )

        self.wait(2)

        innr_prdct_add_8=MathTex(r"\lambda_{1} \cdot \langle i,v \rangle",r"+",r"\lambda_{2} \cdot \langle i,v \rangle","=",r"\langle ( \lambda_{1} + \lambda_{2}) \cdot i,v \rangle").scale(.7).set_color(REANLEA_AQUA).move_to(innr_prdct_add_7.get_center())

        self.play(
            TransformMatchingShapes(innr_prdct_add_7,innr_prdct_add_8)
        )

        self.play(
            pau_1.animate.set_value(6.5/1.5),
            nau_1.animate.set_value(1.5/1.5),

            pau_2.animate.set_value(6.5/1),
            nau_2.animate.set_value(1.5/1),

            pau_3.animate.set_value(6.5/2.5),
            nau_3.animate.set_value(1.5/2.5),

            pau_4.animate.set_value(6.5/2.5),
            nau_4.animate.set_value(1.5/2.5),


            lam_val_trac_0.animate.set_value(2.5),
            lam_val_trac_1.animate.set_value(1.5),
            lam_val_trac_2.animate.set_value(1),

            run_time=2
        ) 
        self.wait()


        self.play(
            pau_1.animate.set_value(6.5/1),
            nau_1.animate.set_value(1.5/1),

            pau_2.animate.set_value(6.5/1.75),
            nau_2.animate.set_value(1.5/1.75),

            pau_3.animate.set_value(6.5/2.75),
            nau_3.animate.set_value(1.5/2.75),

            pau_4.animate.set_value(6.5/2.75),
            nau_4.animate.set_value(1.5/2.75),


            lam_val_trac_0.animate.set_value(2.75),
            lam_val_trac_1.animate.set_value(1),
            lam_val_trac_2.animate.set_value(1.75),

            run_time=2
        ) 
        self.wait()


        self.play(
            AnimationGroup(
                AnimationGroup(
                    pau_1.animate.set_value(6.5/1),
                    nau_1.animate.set_value(1.5/1),

                    pau_2.animate.set_value(6.5/1),
                    nau_2.animate.set_value(1.5/1),

                    pau_3.animate.set_value(6.5/2),
                    nau_3.animate.set_value(1.5/2),

                    pau_4.animate.set_value(6.5/2),
                    nau_4.animate.set_value(1.5/2),


                    lam_val_trac_0.animate.set_value(2),
                    lam_val_trac_1.animate.set_value(1),
                    lam_val_trac_2.animate.set_value(1),
                ),
                AnimationGroup(
                    Flash(point=Dot().move_to(ax_1.c2p(2,0)), color=REANLEA_BLUE_LAVENDER),
                    dt_2.animate.set_color(REANLEA_GOLDENROD)
                ),
                lag_ratio=.5
            ),
            run_time=2
        ) 
        self.wait(2)

        fr_all_txt_0= MathTex(r"\forall \lambda_{1} , \lambda_{2} \in \mathbb{R}").scale(.6).move_to(5.5*RIGHT+.75*UP).set_color(REANLEA_TXT_COL)
        fr_all_txt_0[0][1:].shift(.1*RIGHT)

        self.play(
            Write(fr_all_txt_0)
        )

        self.wait(4)

        innr_prdct_add_9=MathTex(r"\langle \lambda_{1} \cdot i,v \rangle",r"+",r"\langle \lambda_{2} \cdot i,v \rangle","=",r"\langle \lambda_{1} \cdot i + \lambda_{2} \cdot i,v \rangle").scale(.7).set_color(REANLEA_AQUA).move_to(innr_prdct_add_7.get_center())

        self.play(
            ReplacementTransform(innr_prdct_add_8,innr_prdct_add_9)
        )

        self.wait(2)

        dt_rep_0=Dot().set_color(PURE_RED).set_sheen(-.25,DOWN).move_to(ax_1.c2p(0,0)).set_z_index(5)

        self.play(
            FadeIn(dt_rep_0)
        )

        self.play(
            dt_rep_0.animate.move_to(ax_1.c2p(5.5,0))
        )
        self.play(
            dt_rep_0.animate.move_to(ax_1.c2p(-1.25,0))
        )
        self.play(
            dt_rep_0.animate.move_to(ax_1.c2p(3,0))
        )
        self.play(
            dt_rep_0.animate.move_to(ax_1.c2p(-1,0))
        )
        self.play(
            dt_rep_0.animate.move_to(ax_1.c2p(6.25,0))
        )

        dt_rep_lbl=MathTex(r"u","=",r"\mu \cdot i").scale(.85).set_color(PURE_RED).move_to(2.5*DOWN+.75*RIGHT)
        dt_rep_lbl[2][2].set_color(REANLEA_AQUA)

        self.play(
            ReplacementTransform(dt_rep_0.copy(),dt_rep_lbl),
            dt_rep_0.animate.move_to(ax_1.c2p(4.55,0))   
        )

        with RegisterFont("Cousine") as fonts:
            txt_3=Text("for some", font=fonts[0]).scale(.25).set_color(REANLEA_TXT_COL).next_to(dt_rep_lbl,RIGHT)

        self.play(
            Write(txt_3)
        )

        dt_rep_lbl_1=MathTex(r"\mu \in \mathbb{R}").scale(.5).set_color(PURE_RED).next_to(txt_3,RIGHT)

        self.play(
            Create(dt_rep_lbl_1)
        )

        self.wait(2)

        innr_prdct_add_10=MathTex(r"\langle u_{1} ,v \rangle",r"+",r"\langle u_{2} ,v \rangle","=",r"\langle u_{1} + u_{2} ,v \rangle").scale(.7).set_color(REANLEA_AQUA).move_to(innr_prdct_add_7.get_center())

        self.play(
            ReplacementTransform(innr_prdct_add_9,innr_prdct_add_10),
            FadeOut(fr_all_txt_0),
            FadeOut(lam_expli_0),
            FadeOut(sr_bez_0)
        )

        self.wait(2)

        innr_prdct_add_11=MathTex(r"\langle \lambda_{1} \cdot u_{1},v \rangle",r"+",r"\langle \lambda_{2} \cdot u_{2},v \rangle","=",r"\langle \lambda_{1} \cdot u_{1} + \lambda_{2} \cdot u_{2},v \rangle").scale(.7).set_color(REANLEA_AQUA).move_to(innr_prdct_add_7.get_center())

        self.play(
            ReplacementTransform(innr_prdct_add_10,innr_prdct_add_11)
        )
        self.play(
            Create(fr_all_txt_0)
        )
        self.wait()

        innr_prdct_add_12=MathTex(r"\lambda_{1} \cdot \langle u_{1},v \rangle",r"+",r"\lambda_{2} \cdot \langle u_{2},v \rangle","=",r"\langle \lambda_{1} \cdot u_{1} + \lambda_{2} \cdot u_{2},v \rangle").scale(.7).set_color(REANLEA_AQUA).move_to(innr_prdct_add_7.get_center())

        self.play(
            ReplacementTransform(innr_prdct_add_11,innr_prdct_add_12)
        )

        

        bend_bez_arrow_1=bend_bezier_arrow().scale(0.5).set_color(REANLEA_TXT_COL).flip().rotate(-80*DEGREES).move_to(1.8*UP+1.1*LEFT)

        self.add(bend_bez_arrow_1)

        with RegisterFont("Cousine") as fonts:
            txt_4 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Linear on",
                "first component"
            )]).arrange_submobjects(DOWN).scale(0.3).set_color(REANLEA_GREY).move_to(2.8*UP+.8*LEFT).rotate(-15*DEGREES)

        self.add(txt_4)

        self.play(
            Create(txt_4),
            innr_prdct_add_12[0][7].animate.set_color(REANLEA_SAFRON),
            innr_prdct_add_12[2][7].animate.set_color(REANLEA_SAFRON),
            innr_prdct_add_12[4][13].animate.set_color(REANLEA_SAFRON),
        )
        self.wait(2)

        innr_prdct_add_12_ref_0_4t6=innr_prdct_add_12[0][4:6].copy().shift(.35*RIGHT)
        innr_prdct_add_12_ref_0_7=innr_prdct_add_12[0][7].copy().move_to(innr_prdct_add_12[0][4].get_center())
        innr_prdct_add_12_ref_0_6=innr_prdct_add_12[0][6].copy().shift(.15*LEFT)

        innr_prdct_add_12_ref_2_4t6=innr_prdct_add_12[2][4:6].copy().shift(.35*RIGHT)
        innr_prdct_add_12_ref_2_7=innr_prdct_add_12[2][7].copy().move_to(innr_prdct_add_12[2][4].get_center())
        innr_prdct_add_12_ref_2_6=innr_prdct_add_12[2][6].copy().shift(.15*LEFT)

        innr_prdct_add_12_ref_4_1t12=innr_prdct_add_12[4][1:12].copy().shift(.35*RIGHT)
        innr_prdct_add_12_ref_4_13=innr_prdct_add_12[4][13].copy().shift(2.5*LEFT)
        innr_prdct_add_12_ref_4_12=innr_prdct_add_12[4][12].copy().shift(2.15*LEFT)



        self.play(
            AnimationGroup(
                Transform(innr_prdct_add_12[0][4:6],innr_prdct_add_12_ref_0_4t6),
                Transform(innr_prdct_add_12[0][7],innr_prdct_add_12_ref_0_7, path_arc=-180*DEGREES),
                Transform(innr_prdct_add_12[0][6],innr_prdct_add_12_ref_0_6)
            ),
            AnimationGroup(
                Transform(innr_prdct_add_12[2][4:6],innr_prdct_add_12_ref_2_4t6),
                Transform(innr_prdct_add_12[2][7],innr_prdct_add_12_ref_2_7, path_arc=-180*DEGREES),
                Transform(innr_prdct_add_12[2][6],innr_prdct_add_12_ref_2_6)
            ),
            AnimationGroup(
                Transform(innr_prdct_add_12[4][1:12],innr_prdct_add_12_ref_4_1t12),
                Transform(innr_prdct_add_12[4][13],innr_prdct_add_12_ref_4_13, path_arc=-180*DEGREES),
                Transform(innr_prdct_add_12[4][12],innr_prdct_add_12_ref_4_12)
            ),
            Indicate(txt_sym_1, color=REANLEA_MAGENTA_MID_LIGHTER)
        )
        self.wait(2)

        bulet_2=Dot(point=[-5.9,-1.75,0], radius=DEFAULT_DOT_RADIUS/1.25, color=REANLEA_WHITE).set_sheen(-.4,DOWN)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_bil_1=Text("Bilinear", font=fonts[0]).scale(.75).set_color(REANLEA_CYAN_LIGHT).next_to(bulet_2,RIGHT)


        self.play(
            Write(bulet_2)
        )
        self.play(
            ReplacementTransform(txt_4[0][0:6].copy(), txt_bil_1)
        )

        self.wait(2)

        

        uncrt_grp=VGroup(axs_1, dt_1_10,dt_1_32,line_1_0010 ,line_1_0032,d_line_1_ver,dt_1_00,
        axs_2,dt_2_10,dt_2_32,line_2_0010 ,line_2_0032,d_line_2_ver,dt_2_00,
        axs_3,dt_3_10,dt_3_32,line_3_0010 ,line_3_0032,
        axs_4)

        bez_grp=VGroup(bend_bez_arrow_1,txt_4)
        innr_prdct_add_def_grp=VGroup(innr_prdct_add_12,fr_all_txt_0)
        dt_rep_grp=VGroup(dt_rep_0,dt_rep_lbl,dt_rep_lbl_1,txt_3)
        dot_grp=VGroup(dot_1,dot_0,dt_2)
        lbl_grp=VGroup(dt_3_lbl,angl_1,angl_1_lbl,lbl_i)

        innr_prdct_dfn_3=MathTex(r"\langle i,v \rangle","=",r"\lVert i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").set_color_by_gradient(REANLEA_CYAN_LIGHT).scale(1.25).shift(RIGHT)

        self.play(
            AnimationGroup(
                FadeOut(uncrt_grp),
                FadeOut(bez_grp),
                FadeOut(innr_prdct_add_def_grp),
                FadeOut(lam_val_grp),
                FadeOut(dt_rep_grp),
                FadeOut(innr_prdct_add_grp_1),
                FadeOut(dot_grp),
                FadeOut(lbl_grp)
            ),
            vects_grp_old.animate.set_opacity(0),
            ReplacementTransform(innr_prdct_dfn_2.copy(),innr_prdct_dfn_3)
        )

        




        

        



        


        self.wait(4)




    # manim -pqh anim2.py Scene4_2

    # manim -pql anim2.py Scene4_2

    # manim -sqk anim2.py Scene4_2  




class Scene4_3(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        water_mark_1=water_mark.copy()
        


        # PREVIOUS SCENE


        innr_prdct_dfn_0=MathTex(r"\langle , \rangle",":",r"V \times V","\longrightarrow",r"\mathbb{R").scale(.65).set_color_by_gradient(REANLEA_CYAN_LIGHT).move_to(3*UP+3*RIGHT)

        with RegisterFont("Courier Prime") as fonts:
            innr_prdct_dfn_1=Text("by", font=fonts[0]).scale(.35).set_color(REANLEA_CYAN_LIGHT).next_to(innr_prdct_dfn_0, RIGHT).shift(.1*RIGHT+.02*DOWN)
        
        innr_prdct_dfn_2=MathTex(r"\langle i,v \rangle","=",r"\lVert i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").scale(.7).set_color_by_gradient(REANLEA_CYAN_LIGHT).move_to(2.5*UP+3.25*RIGHT)

        innr_prdct_dfn=VGroup(innr_prdct_dfn_0,innr_prdct_dfn_1,innr_prdct_dfn_2).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_AQUA)

        innr_prdct_dfn_2.set_color(REANLEA_AQUA)


        bulet_1=Dot(point=[-5.9,-1,0], radius=DEFAULT_DOT_RADIUS/1.25, color=REANLEA_WHITE).set_sheen(-.4,DOWN)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_sym_1=Text("Symmetric", font=fonts[0]).scale(.75).set_color(REANLEA_CYAN_LIGHT).next_to(bulet_1,RIGHT)
 


        self.add(innr_prdct_dfn, txt_sym_1,bulet_1)



        txt_v_0=MathTex("v",",").set_color(REANLEA_PINK_LIGHTER).move_to(5.25*LEFT+2*UP)
        txt_v_0[1].set_color(REANLEA_TXT_COL)
        txt_i_0=MathTex("i").set_color(REANLEA_AQUA).move_to(4.9*LEFT+2.1*UP)
        txt_com_0=MathTex(",").set_color(REANLEA_TXT_COL).next_to(txt_i_0,RIGHT, buff=.1).shift(.18*DOWN)
        txt_th_0=MathTex(r"\theta").set_color(REANLEA_YELLOW).move_to(4.45*LEFT+2.1*UP)

        txt_v_1=txt_v_0.copy()
        txt_i_1=txt_i_0.copy()

        txt_lbl_grp_0=VGroup(txt_v_0,txt_i_0,txt_com_0,txt_th_0).scale(1.5).move_to(4.9*LEFT+2.1*UP)
        txt_lbl_grp_1=VGroup(txt_v_1,txt_i_1).scale(1.5).move_to(4.9*LEFT+2.1*UP)
        

        arr_1=Arrow(max_tip_length_to_length_ratio=.1, color=REANLEA_BLUE).rotate(-90*DEGREES).next_to(txt_i_0,DOWN).set_stroke(width=2, color=[REANLEA_BLUE,REANLEA_BLUE_SKY]).shift(.35*DOWN).scale(.85)

        self.add(arr_1)

        txt_scl_1=MathTex(r"\lVert i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").set_color(REANLEA_AQUA_GREEN).next_to(arr_1,DOWN, buff=.1).shift(.35*DOWN)

        self.add(txt_scl_1)

        self.add(txt_lbl_grp_1)

        indct_ln_1=Line().set_stroke(width=2,color=REANLEA_GREY).scale(.25).rotate(PI/6).next_to(txt_lbl_grp_1, RIGHT).shift(.2*UP).set_z_index(2)

        self.add(indct_ln_1)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_1=Text("Vectors", font=fonts[0]).scale(.75).set_color(REANLEA_CYAN_LIGHT).next_to(indct_ln_1).shift(.35*UP)
        
        self.add(txt_1)

        indct_ln_2=Line().set_stroke(width=2,color=REANLEA_GREY).scale(.25).rotate(-PI/6).next_to(txt_scl_1, DOWN).shift(.2*RIGHT).set_z_index(2)

        self.add(indct_ln_2)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_2=Text("Scalars", font=fonts[0]).scale(.75).set_color(REANLEA_CYAN_LIGHT).next_to(indct_ln_2).shift(.25*DOWN)

        self.add(txt_2)
        
        innr_prdct_sym=MathTex(r"\langle ,\rangle").set_color(REANLEA_BLUE_LAVENDER).next_to(arr_1,RIGHT)

        self.add(innr_prdct_sym)

        innr_prdct_grp_0=VGroup(txt_lbl_grp_1,arr_1,txt_scl_1,innr_prdct_sym,indct_ln_1,indct_ln_2,txt_1,txt_2).scale(.5).shift(.5*LEFT+.5*UP)

        sr_innr_prdct_grp_0=SurroundingRectangle(innr_prdct_grp_0, color=REANLEA_PURPLE_LIGHTER, buff=.25, corner_radius=.125).set_opacity(.25)

        innr_prdct_dfn_3=MathTex(r"\langle i,v \rangle","=",r"\lVert i \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos\theta").set_color_by_gradient(REANLEA_CYAN_LIGHT).scale(1.25).shift(RIGHT)

        bulet_2=Dot(point=[-5.9,-1.75,0], radius=DEFAULT_DOT_RADIUS/1.25, color=REANLEA_WHITE).set_sheen(-.4,DOWN)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_bil_1=Text("Bilinear", font=fonts[0]).scale(.75).set_color(REANLEA_CYAN_LIGHT).next_to(bulet_2,RIGHT)

        self.add(sr_innr_prdct_grp_0, innr_prdct_dfn_3, bulet_2, txt_bil_1)


        ### MAIN SCENE

        self.wait(2)

        innr_prdct_dfn_4=MathTex(r"\langle v,v \rangle","=",r"\lVert v \rVert",r"\cdot",r"\lVert v \rVert",r"\cdot",r"cos0").set_color_by_gradient(REANLEA_CYAN_LIGHT).scale(1.25).shift(RIGHT)

        self.play(
            ReplacementTransform(innr_prdct_dfn_3,innr_prdct_dfn_4)
        )
        self.wait()

        cos_0_val=MathTex("1").set_color_by_gradient(REANLEA_CYAN_LIGHT).scale(1.25).move_to(innr_prdct_dfn_4[-1].get_center()).shift(.3*LEFT)

        self.play(
            Transform(innr_prdct_dfn_4[-1],cos_0_val)
        )
        self.wait()

        v_sq_0=MathTex(r"\lVert v \rVert ^{2}").set_color_by_gradient(REANLEA_CYAN_LIGHT).scale(1.25).move_to(innr_prdct_dfn_4[2].get_center()).shift(.2*RIGHT+.05*UP)

        self.play(
            Transform(innr_prdct_dfn_4[2:], v_sq_0)
        )

        self.play(
            innr_prdct_dfn_4.animate.shift(RIGHT)
        )

        self.wait(2)

        innr_prdct_dfn_4_geq=MathTex(r"\geq 0").set_color_by_gradient(REANLEA_CYAN_LIGHT).scale(1.25).next_to(innr_prdct_dfn_4,RIGHT)

        self.play(
            Write(innr_prdct_dfn_4_geq)
        )
        self.wait(2)

        with RegisterFont("Cousine") as fonts:
            txt_4=Text("and the equality holds \"if and only if\" ", font=fonts[0]).scale(.35).set_color(REANLEA_TXT_COL).next_to(innr_prdct_dfn_4,DOWN).shift(RIGHT)

        self.play(
            Write(txt_4)
        )

        innr_prdct_dfn_5=MathTex(r"\lVert v \rVert = 0").set_color_by_gradient(REANLEA_CYAN_LIGHT).next_to(txt_4,DOWN).shift(LEFT)

        self.play(
            Create(innr_prdct_dfn_5)
        )
        self.wait(2)

        arr_2=MathTex(r"\longrightarrow").rotate(-30*DEGREES).next_to(innr_prdct_dfn_5,RIGHT).set_stroke(width=2, color=[REANLEA_PURPLE,REANLEA_BLUE_SKY]).shift(.15*LEFT+.45*DOWN)

        self.play(
            Write(arr_2)
        )

        innr_prdct_dfn_6=MathTex(r"v").set_color_by_gradient(REANLEA_TXT_COL)

        with RegisterFont("Cousine") as fonts:
            txt_5=Text("is the ZERO vector.", font=fonts[0]).scale(.35).set_color(REANLEA_TXT_COL)#.next_to(innr_prdct_dfn_6,RIGHT)

        innr_prdct_dfn_6_grp=VGroup(innr_prdct_dfn_6,txt_5).arrange_submobjects(RIGHT, buff=.2).next_to(arr_2,DOWN).shift(1.95*RIGHT+.35*UP)

        self.play(
            Write(innr_prdct_dfn_6_grp)
        )
        self.wait(2)

        with RegisterFont("Fuzzy Bubbles") as fonts:
            txt_pos_def_0=Text("Positive Definite", font=fonts[0]).scale(.75).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_BLUE_LAVENDER).shift(1.25*UP+RIGHT)

        strp_1=get_stripe(factor=0.05, buff_max=4.2, color=PURE_GREEN).next_to(txt_pos_def_0,DOWN).shift(.15*UP+.175*RIGHT)

        self.play(
            Write(txt_pos_def_0)
        )
        self.play(
            Create(strp_1)
        )
        self.wait()

        bulet_3=Dot(point=[-5.9,-2.5,0], radius=DEFAULT_DOT_RADIUS/1.25, color=REANLEA_WHITE).set_sheen(-.4,DOWN)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_pos_def_1=Text("Positive Definite", font=fonts[0]).scale(.75).set_color(REANLEA_CYAN_LIGHT).next_to(bulet_3,RIGHT)

        self.play(
            FadeIn(bulet_3)
        )
        self.play(
            ReplacementTransform(txt_pos_def_0.copy(),txt_pos_def_1)
        )

        self.wait(2)

        with RegisterFont("Fuzzy Bubbles") as fonts:
            txt_x_0=Text("Inner Product", font=fonts[0]).scale(.75).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_BLUE_LAVENDER).shift(2*UP)

        self.play(            
            AnimationGroup(
                *[FadeOut(mobj) for mobj in self.mobjects],
            ),
            FadeIn(water_mark_1),
            Write(txt_x_0),
            run_time=3
        )
        self.wait(2)

        ln_0=Line().set_stroke(width=7.5, color=[REANLEA_BLUE_LAVENDER,REANLEA_BLUE_SKY,REANLEA_PURPLE]).rotate(PI/2).next_to(txt_x_0,DOWN)

        self.play(
            Create(ln_0.reverse_direction())
        )

        with RegisterFont("Fuzzy Bubbles") as fonts:
            txt_x_1=Text("Magnitude / Norm", font=fonts[0]).scale(.75).set_color_by_gradient(REANLEA_PURPLE,REANLEA_BLUE_LAVENDER).next_to(ln_0,DOWN)
        
        self.play(
            Write(txt_x_1)
        )
        self.wait(2)

        ln_txt_grp_0=VGroup(txt_x_0,ln_0,txt_x_1)

        self.play(
            ln_txt_grp_0.animate.shift(3.25*LEFT)
        )

        ln_1=Line(start=1.5*LEFT,end=1.5*RIGHT).set_stroke(width=7.5, color=[REANLEA_GOLD,REANLEA_PURPLE,REANLEA_BLUE_LAVENDER]).next_to(txt_x_1,RIGHT)

        self.play(
            Create(ln_1)
        )

        with RegisterFont("Fuzzy Bubbles") as fonts:
            txt_x_2 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Pythagoras\'",
                "Theorem"
            )]).arrange_submobjects(DOWN).scale(.75).set_color_by_gradient(REANLEA_GOLD,REANLEA_TXT_COL).next_to(ln_1,RIGHT)
        
        self.play(
            Write(txt_x_2)
        )
        self.wait(2)

        ln_2=Line(start=2.3*RIGHT+.6*DOWN,end=2.65*LEFT+1.55*UP).set_stroke(width=7.5, color=[REANLEA_TXT_COL,REANLEA_GOLD,REANLEA_BLUE_SKY])

        self.play(
            Create(ln_2)
        )
        self.wait()

        arr_circ_1=MathTex(r"\circlearrowleft").set_color(REANLEA_BLUE_SKY).scale(3.5).set_stroke(width=7.5, color=[REANLEA_BLUE_LAVENDER,REANLEA_BLUE_SKY,REANLEA_PURPLE,REANLEA_GOLD]).shift(1.55*LEFT+.1*UP)

        self.play(
            Write(arr_circ_1)
        )

        uncrt_grp=VGroup(ln_0,ln_1,ln_2,txt_x_0,txt_x_1,txt_x_2,arr_circ_1)

        self.play(
            FadeOut(uncrt_grp),
            run_time=3
        )

        








        self.wait(4)






        # manim -pqh anim2.py Scene4_3

        # manim -pql anim2.py Scene4_3

        # manim -sqk anim2.py Scene4_3
        


###################################################################################################################

class Scene5(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        self.wait()


        # MAIN SCENE

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

        self.play(
            Write(ax_1)
        )
        dt_0=Dot(point=ax_1.c2p(0,0), color=REANLEA_GREY, radius=DEFAULT_SMALL_DOT_RADIUS*1.25).set_z_index(1).set_sheen(.4,UP)
        self.play(
            FadeIn(dt_0)
        )

        dt_1=Dot(point=ax_1.c2p(-1,2), color=REANLEA_AQUA).set_sheen(-.4,DOWN)

        self.play(
            Write(dt_1)
        )

        dt_1_lbl=MathTex("v").scale(.75).set_color(REANLEA_AQUA).next_to(dt_1,UL)

        arr_1=Arrow(start=ax_1.c2p(0,0),end=ax_1.c2p(-1,2),tip_length=.125,stroke_width=4, buff=0).set_color_by_gradient(REANLEA_BLUE_SKY)
        arr_1_copy_1=arr_1.copy().set_opacity(.5)

        self.play(
            Write(arr_1),
            Write(dt_1_lbl)
        )
        self.add(arr_1_copy_1)
        self.wait(2)

        nrm_def_0=MathTex(r"\langle v,v \rangle","=",r"\lVert v \rVert ^{2}").set_color(REANLEA_TXT_COL).shift(3*UP+4*RIGHT)
        nrm_def_0_copy_1=nrm_def_0.copy()

        self.play(
            ReplacementTransform(dt_1_lbl.copy(),nrm_def_0_copy_1),
            Write(nrm_def_0)
        )
        self.play(FadeOut(nrm_def_0_copy_1))
        self.wait(2)

        nrm_def_1=MathTex(r"\Rightarrow",r"\lVert v \rVert","=",r"\sqrt{\langle v,v \rangle}").set_color(REANLEA_TXT_COL).next_to(nrm_def_0,DOWN).shift(.175*RIGHT)
        nrm_def_1_ref=nrm_def_1.copy().set_z_index(11)

        self.play(
            Write(nrm_def_1),
            nrm_def_0.animate.scale(.75)
        )
        self.wait(2)

        dt_2=Dot(point=ax_1.c2p(2,3), color=REANLEA_GOLD).set_sheen(-.4,DOWN).set_z_index(-1)
        arr_2=Arrow(start=ax_1.c2p(0,0),end=ax_1.c2p(2,3),tip_length=.125,stroke_width=4, buff=0).set_color_by_gradient(REANLEA_YELLOW)
        dt_2_lbl=MathTex("X").scale(.5).set_color(REANLEA_YELLOW).move_to(ax_1.c2p(.75,1.75))
        

        self.play(
            Write(dt_2)
        )
        self.play(
            Write(arr_2),
            Write(dt_2_lbl)
        )
        self.wait()

        dt_3=Dot(point=ax_1.c2p(3,1), color=REANLEA_GREEN).set_sheen(-.4,DOWN)

        arr_3=Arrow(start=ax_1.c2p(0,0),end=ax_1.c2p(3,1),tip_length=.125,stroke_width=4, buff=0).set_color_by_gradient(REANLEA_GREEN)

        dt_3_lbl=MathTex("Y").scale(.5).set_color(REANLEA_GREEN).move_to(ax_1.c2p(1.75,.25))

        self.play(
            Write(dt_3)
        )
        self.wait()

        self.play(
            Write(arr_3),
            Write(dt_3_lbl)
        )
        self.play(
            arr_1.animate.shift(
                (ax_1.c2p(3,1)[0]-ax_1.c2p(0,0)[0])*RIGHT+
                (ax_1.c2p(3,1)[1]-ax_1.c2p(0,0)[1])*UP
            )
        )
        self.wait(2)

        v_equal_1=MathTex("v","=","X-Y").scale(.75).move_to(ax_1.c2p(4,2))
        v_equal_1[0].set_color(REANLEA_AQUA).scale(1.25)
        v_equal_1[2][0].set_color(REANLEA_YELLOW)
        v_equal_1[2][2].set_color(REANLEA_GREEN)

        self.play(
            Write(v_equal_1)
        )
        self.wait(2)

        dt_2_lbl_1=MathTex(r"(x_{1},x_{2})").scale(.5).set_color(REANLEA_YELLOW).move_to(ax_1.c2p(2.15,3.5))

        dt_3_lbl_1=MathTex(r"(y_{1},y_{2})").scale(.5).set_color(REANLEA_GREEN).move_to(ax_1.c2p(3.45,.6))

        self.play(
            Write(dt_2_lbl_1),
            Write(dt_3_lbl_1)
        )

        dt_2_3_lbl_1_grp=VGroup(dt_2_lbl_1,dt_3_lbl_1)

        v_equal_2=MathTex("=",r"(x_{1}-y_{1},x_{2}-y_{2})").scale(.75).next_to(v_equal_1,RIGHT)
        v_equal_2[1][1:3].set_color(REANLEA_YELLOW)
        v_equal_2[1][7:9].set_color(REANLEA_YELLOW)
        v_equal_2[1][4:6].set_color(REANLEA_GREEN)
        v_equal_2[1][10:12].set_color(REANLEA_GREEN)

        self.play(
            ReplacementTransform(dt_2_3_lbl_1_grp.copy(),v_equal_2)
        )
        self.wait(2)

        v_equal_grp=VGroup(v_equal_1,v_equal_2)
        #v_equal_grp_1=VGroup(v_equal_1,v_equal_2).move_to(3*UP+2.75*LEFT).set_z_index(11)
        v_equal_grp_1=v_equal_grp.copy().set_z_index(11)
        

        rect_overlap=Rectangle(width=16, height=9, color=REANLEA_BACKGROUND_COLOR).to_edge(RIGHT, buff=0).set_opacity(.825).set_z_index(10)
        self.add(v_equal_grp_1)
        nrm_def_0_copy_2=nrm_def_0.copy().set_z_index(11)


        self.play(
            AnimationGroup(
                Create(rect_overlap),
                v_equal_grp_1.animate.move_to(3*UP+2.75*LEFT),
                FadeIn(nrm_def_0_copy_2)
            ),
            run_time=2.5
        )

        v_eq_nrm_grp=VGroup(v_equal_grp_1[0],nrm_def_0_copy_2)
        v_eq_nrm_grp_2=VGroup(v_equal_grp_1[0][0],v_equal_grp_1[1],nrm_def_0_copy_2)

        sep_ln_1=Line(start=2.65*UP+5.5*LEFT, end=2.65*UP+5*RIGHT).set_stroke(width=5, color=[REANLEA_AQUA, REANLEA_PURPLE, REANLEA_BLUE_LAVENDER,REANLEA_BLUE_SKY]).set_z_index(11)

        self.play(
            Create(sep_ln_1)
        )

        eqn_1=MathTex(r"\lVert X-Y \rVert ^{2}&",r"= \langle X-Y , X-Y \rangle \\ &",r"= \langle X , X \rangle - \langle X , Y \rangle - \langle Y , X \rangle + \langle Y , Y \rangle \\ &",r"= \lVert X \rVert ^{2} + \lVert Y \rVert ^{2} - \langle X , Y \rangle - \langle X , Y \rangle \\ &",r"= \lVert X \rVert ^{2} + \lVert Y \rVert ^{2} - 2 \langle X , Y \rangle ").scale(.75).set_color_by_gradient(REANLEA_TXT_COL_LIGHTER).set_z_index(11)

        eqn_1_ref=eqn_1[0:2].copy()
        

        self.play(
            ReplacementTransform(v_eq_nrm_grp.copy(), eqn_1_ref),
            run_time=2
        )

        self.play(
            AnimationGroup(
                *[Write(eq) for eq in eqn_1],
                lag_ratio=2
            )
        )

        self.play(
            FadeOut(eqn_1_ref)
        )
        self.wait(2)

        self.play(
            eqn_1.animate.scale(.75).shift(3.75*LEFT+.5*UP)
        )

        sep_ln_2 = Line(start=.5*LEFT+2.5*UP, end=.5*LEFT+1.5*DOWN).set_stroke(width=3, color=[REANLEA_WARM_BLUE,REANLEA_BLUE_LAVENDER,REANLEA_VIOLET,REANLEA_AQUA]).set_z_index(11)

        self.play(
            Create(sep_ln_2)
        )

        eqn_2=MathTex(r"\lVert X-Y \rVert ^{2}&",r"= \lVert (x_{1}-y_{1},x_{2}-y_{2}) \rVert^{2} \\ &", r"=(x_{1}-y_{1})^2 + (x_{2}-y_{2})^2 \\ &",r"=(x_{1}^2 - 2x_{1}y_{1}+ y_{1}^2)+(x_{2}^2 - 2x_{2}y_{2}+ y_{2}^2) \\ &",r"=(x_{1}^2+x_{2}^2)+(y_{1}^2+y_{2}^2) -2(x_{1}y_{1}+x_{2}y_{2}) \\ &",r"= \lVert X \rVert ^{2} + \lVert Y \rVert ^{2} -2(x_{1}y_{1}+x_{2}y_{2})").scale(.6).set_color_by_gradient(REANLEA_TXT_COL_LIGHTER).set_z_index(11).shift(3.25*RIGHT+.45*UP)

        eqn_2_ref_0=eqn_2[0:2].copy()
        eqn_2_ref_1=eqn_2[2:].copy()

        self.play(
            ReplacementTransform(v_eq_nrm_grp_2.copy(),eqn_2_ref_0)
        )


        with RegisterFont("Cousine") as fonts:
            txt_x_1=Text("Pythagoras\' Theorem", font=fonts[0]).scale(.5).move_to(3*RIGHT).set_z_index(11)

        self.play(
            Write(txt_x_1)
        )
        self.wait()

        self.play(
            AnimationGroup(
                FadeOut(txt_x_1)
            ),
            AnimationGroup(
                *[Write(eq) for eq in eqn_2_ref_1],
                lag_ratio=2
            ),
            lag_ratio=.6
        )
        self.add(eqn_2)

        self.play(
            FadeOut(eqn_2_ref_0),
            FadeOut(eqn_2_ref_1)
        )
        self.wait(2)

        eqn_1_x_0=eqn_1[-1].copy()
        eqn_2_x_0=eqn_2[-1].copy()

        eqn_1_2_x_0_grp=VGroup(eqn_1_x_0,eqn_2_x_0)

        eqn_3=MathTex(r"\lVert X \rVert ^{2} + \lVert Y \rVert ^{2} - 2 \langle X , Y \rangle ","=",r"\lVert X \rVert ^{2} + \lVert Y \rVert ^{2} -2(x_{1}y_{1}+x_{2}y_{2})").set_z_index(11).shift(2.75*DOWN).scale(.75).set_color_by_gradient(REANLEA_WARM_BLUE,REANLEA_VIOLET_LIGHTER,REANLEA_WARM_BLUE)

        self.play(
            ReplacementTransform(eqn_1_2_x_0_grp,eqn_3)
        )
        self.wait(2)

        eqn_4=MathTex(r"\langle X , Y \rangle ","=",r"(x_{1}y_{1}+x_{2}y_{2})").set_z_index(11).shift(2.75*DOWN).scale(.75).set_color_by_gradient(REANLEA_WARM_BLUE,REANLEA_VIOLET_LIGHTER,REANLEA_WARM_BLUE)

        self.play(
            ReplacementTransform(eqn_3,eqn_4)
        )

        sr_eqn_4=SurroundingRectangle(eqn_4, buff=.25, corner_radius=.125).set_stroke(width=3, color=[REANLEA_WARM_BLUE,REANLEA_VIOLET])

        self.play(
            Write(sr_eqn_4)
        )

        eqn_4_grp=VGroup(eqn_4,sr_eqn_4)

        uncrt_grp_0=VGroup(eqn_1,eqn_2,sep_ln_1,sep_ln_2,v_equal_grp_1,nrm_def_0_copy_2,rect_overlap)

        eqn_5=MathTex(r"\lVert X-Y \rVert ^{2}&",r"= \lVert X \rVert ^{2} + \lVert Y \rVert ^{2} -2",r" \langle X , Y \rangle ").scale(.57).set_color_by_gradient(REANLEA_TXT_COL_LIGHTER).shift(4.25*LEFT+2.5*UP)

        eqn_5_ref=MathTex(r" \lVert X \rVert \cdot \lVert Y \rVert \cdot cos\theta ").shift(2.475*UP+1.7*LEFT).scale(.57)

        eqn_5_ref_copy=MathTex(r" \lVert X \rVert \cdot \lVert Y \rVert \cdot cos\theta ").shift(2.475*UP+1.7*LEFT).scale(.57)

        eqn_1_1=eqn_1.copy()

        self.play(
            FadeOut(uncrt_grp_0),
            ReplacementTransform(eqn_1_1.copy(),eqn_5),
            eqn_4_grp.animate.shift(4.5*RIGHT)
        )

        self.wait(2)

        angl_1=Angle(arr_3,arr_2, radius=.5).set_color(REANLEA_YELLOW_GREEN).set_stroke(width=3.5).set_z_index(-1)
        
        angl_1_lbl=MathTex(r"\theta").scale(.65).set_color(REANLEA_YELLOW_GREEN).next_to(angl_1,UR, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2).set_z_index(2).shift(.125*DOWN)

        self.play(
            Write(angl_1)
        )
        self.play(
            Write(angl_1_lbl)
        )

        self.wait(2)

        eqn_6=MathTex(r"\langle X,Y \rangle","=",r" \lVert X \rVert \cdot \lVert Y \rVert \cdot cos\theta ").shift(2.75*DOWN+4.5*LEFT).scale(.6)

        dt_angl_lbl_grp=VGroup(dt_2_lbl,dt_3_lbl,angl_1_lbl)

        self.play(
            ReplacementTransform(dt_angl_lbl_grp.copy(),eqn_6)
        )
        self.wait(2)

        eqn_6_ref=eqn_6[-1].copy()

        self.play(
            AnimationGroup(
                ReplacementTransform(eqn_6_ref,eqn_5_ref_copy),
                Transform(eqn_5[-1],eqn_5_ref),
                lag_ratio=.25
            )
        )
        self.wait()
        self.play(
            FadeOut(eqn_5_ref_copy)
        )
        
        dt_2_3_grp_1=VGroup(dt_2_lbl,dt_3_lbl)

        trans_grp_1=VGroup(dt_2_3_grp_1[0],dt_2_3_grp_1[1],v_equal_grp)
        trans_grp_2=VGroup(dt_2_3_grp_1[0].copy(),dt_2_3_grp_1[1].copy(),v_equal_grp.copy())

        dt_2_lbl_alt=MathTex("a").set_color(REANLEA_YELLOW).move_to(ax_1.c2p(.75,1.75)).scale(.8)
        dt_3_lbl_alt=MathTex("b").set_color(REANLEA_GREEN).move_to(ax_1.c2p(1.75,.25)).scale(.8)
        dt_x_lbl_alt=MathTex("c").move_to(v_equal_1[0].get_center()).set_color(REANLEA_AQUA).scale(.8)

        dt_lbl_grp=VGroup(dt_2_lbl_alt,dt_3_lbl_alt,dt_x_lbl_alt)

        eqn_7=MathTex(r"c^{2}","=",r"a^{2} + b^{2} -2","a","b",r"\cos\theta").scale(.67).set_color_by_gradient(REANLEA_TXT_COL_LIGHTER).shift(4.25*LEFT+2.5*UP)
        eqn_7[0][0].set_color(REANLEA_AQUA)
        eqn_7[2][0].set_color(REANLEA_YELLOW)
        eqn_7[2][3].set_color(REANLEA_GREEN)
        eqn_7[3].set_color(REANLEA_YELLOW)
        eqn_7[4].set_color(REANLEA_GREEN)
        eqn_7[5][-1].set_color(REANLEA_YELLOW_GREEN)
        eqn_7[5:].shift(.05*RIGHT)

        self.play(
            ReplacementTransform(trans_grp_1,dt_lbl_grp),
            ReplacementTransform(eqn_5,eqn_7)
        )
        self.wait(2)

        bez_arr_1=bend_bezier_arrow().flip(UP).rotate(-45*DEGREES).scale(.5).next_to(eqn_7,DOWN).set_z_index(11)

        with RegisterFont("Homemade Apple") as fonts:
            txt_2=Text("Law of Cosines", font=fonts[0]).scale(.5)
            txt_2.set_color_by_gradient(REANLEA_BLUE_LAVENDER,REANLEA_TXT_COL_LIGHTER)
            txt_2.next_to(bez_arr_1,DOWN).shift(2*RIGHT+.75*UP).set_z_index(11)

        self.add(nrm_def_1_ref)        

        undr_bez=underline_bez_curve().scale(1.25).next_to(txt_2, DOWN).shift(.2*UP).set_z_index(11)

        rect_overlap_1=Rectangle(width=16, height=4, color=REANLEA_BACKGROUND_COLOR).to_edge(RIGHT, buff=0).set_opacity(.825).set_z_index(10)

        self.play(
            Write(rect_overlap_1),
            Create(bez_arr_1)
        )
        self.play(
            Write(txt_2)
        )

        self.play(            
            Write(undr_bez)
        )
        self.wait(2)

        with RegisterFont("Cousine") as fonts:
            txt_3 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "The square of any one side of a triangle",
                "is equal to the difference between the",
                "sum of squares of the other two sides",
                "and double the product of other sides",
                "and cosine angle included between them."
            )]).scale(0.4).set_color(REANLEA_GREY).arrange_submobjects(.3*DOWN).shift(.45*DOWN+.5*RIGHT).set_z_index(11)

        self.play(
            AnimationGroup(
                *[Write(xx) for xx in txt_3],
                lag_ratio=1.25
            )
        )

        self.wait(2)

        remain_grp_1=VGroup()

        eqn_4_grp_copy=eqn_4_grp.copy().set_z_index(12)
        eqn_6_copy=eqn_6.copy().set_z_index(12)
        nrm_def_0_copy_3=nrm_def_0.copy()
        water_mark_1=water_mark.copy()


        self.play(            
            AnimationGroup(
                *[FadeOut(mobj) for mobj in self.mobjects],
            ),
            FadeIn(water_mark_1),
            FadeIn(nrm_def_0_copy_3),
            eqn_4_grp_copy.animate.move_to(ORIGIN).shift(1.25*UP),
            eqn_6_copy.animate.move_to(2.25*UP+4.5*LEFT),
            run_time=3
        )
        self.wait(2)


        


        self.wait(4)


        # manim -pqh anim2.py Scene5

        # manim -pql anim2.py Scene5

        # manim -sqk anim2.py Scene5

        # manim -sqh anim2.py Scene5
        





class Scene5_1(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        self.wait()


        # PREVIOUS SCENE

        nrm_def_0=MathTex(r"\langle v,v \rangle","=",r"\lVert v \rVert ^{2}").set_color(REANLEA_TXT_COL).shift(3*UP+4*RIGHT).scale(.75)

        eqn_0=MathTex(r"\langle X , Y \rangle ","=",r"(x_{1}y_{1}+x_{2}y_{2})").shift(2.75*DOWN).scale(.75).set_color_by_gradient(REANLEA_WARM_BLUE,REANLEA_VIOLET_LIGHTER,REANLEA_WARM_BLUE).move_to(1.25*UP)

        sr_eqn_0=SurroundingRectangle(eqn_0, buff=.25, corner_radius=.125).set_stroke(width=3, color=[REANLEA_WARM_BLUE,REANLEA_VIOLET])

        eqn_0_grp=VGroup(eqn_0,sr_eqn_0)

        eqn_1=MathTex(r"\langle X,Y \rangle","=",r" \lVert X \rVert \cdot \lVert Y \rVert \cdot cos\theta ").move_to(2.25*UP+4.5*LEFT).scale(.6)

        self.add(nrm_def_0,eqn_0_grp,eqn_1)


        # MAIN SCENE

        eqn_2_0=MathTex(r"\langle X , X \rangle ","=",r"x_{1}x_{1}+x_{2}x_{2}").scale(.75).set_color_by_gradient(REANLEA_WARM_BLUE,REANLEA_VIOLET_LIGHTER,REANLEA_WARM_BLUE)

        self.play(
            ReplacementTransform(eqn_0.copy(),eqn_2_0)
        )

        eqn_2_1=MathTex(r"\langle X , X \rangle &",r"= x_{1}^{2}+x_{2}^{2} \\ &",r"= \lVert X \rVert ^{2}").scale(.75).set_color_by_gradient(REANLEA_WARM_BLUE,REANLEA_VIOLET_LIGHTER,REANLEA_WARM_BLUE)

        self.play(
            ReplacementTransform(eqn_2_0,eqn_2_1[0:2])
        )
        self.wait(2)

        self.play(
            Write(eqn_2_1[2])
        )

        eqn_3_0=MathTex(r"\Rightarrow \lVert X \rVert",r"= \sqrt{ x_{1}^{2}+x_{2}^{2} }").scale(.75).set_color_by_gradient(REANLEA_WARM_BLUE,REANLEA_VIOLET_LIGHTER,REANLEA_WARM_BLUE).next_to(eqn_2_1,DOWN).shift(.15*RIGHT)

        eqn_3_1=MathTex(r"= \Biggl\lbrack \sum_{i=1}^{2} x_{i}^{2} \Biggr\rbrack ^{1/2} ").scale(.75).set_color_by_gradient(REANLEA_WARM_BLUE,REANLEA_VIOLET_LIGHTER,REANLEA_WARM_BLUE).next_to(eqn_3_0)

        self.play(
            TransformMatchingShapes(eqn_2_1.copy(),eqn_3_0)
        )
        self.wait(2)

        self.play(
            Write(eqn_3_1)
        )
        self.wait(2)



        
        indct_ln_0=Line().scale(.35).rotate(-115*DEGREES).next_to(eqn_3_1,DOWN).shift(.25*LEFT).set_stroke(width=3, color=[REANLEA_WARM_BLUE,REANLEA_PINK])

        self.play(
            Create(indct_ln_0)
        )

        with RegisterFont("Cousine") as fonts:
            txt_0=Text("2 dimensional space", font=fonts[0]).scale(.35)
            txt_0.set_color_by_gradient(REANLEA_BLUE_LAVENDER,REANLEA_TXT_COL_LIGHTER)
            txt_0.next_to(indct_ln_0,DOWN)

        self.play(
            Write(txt_0)
        )
        self.wait(2)



        eqn_2_3_grp=VGroup(eqn_2_1,eqn_3_0,eqn_3_1, indct_ln_0, txt_0)

        self.play(
            eqn_2_3_grp.animate.shift(4*LEFT)
        )

        sep_ln_0=Line(start=1.5*UP, end=1.5*DOWN).set_stroke(width=3, color=[REANLEA_AQUA,REANLEA_PINK,REANLEA_AQUA]).shift(1.5*RIGHT+DOWN)

        self.play(
            Create(sep_ln_0)
        )

        eqn_4=MathTex(r"\lVert X \rVert",r"= \Biggl\lbrack \sum_{i=1}^{n} x_{i}^{2} \Biggr\rbrack ^{1/2} ").scale(.75).set_color_by_gradient(REANLEA_WARM_BLUE,REANLEA_VIOLET_LIGHTER,REANLEA_WARM_BLUE).next_to(sep_ln_0).shift(1.5*RIGHT)

        self.play(
            TransformMatchingShapes(VGroup(eqn_3_0[0].copy(),eqn_3_1.copy()),eqn_4)
        )

        self.wait(2)

        indct_ln_1=Line().scale(.35).rotate(-115*DEGREES).next_to(eqn_4,DOWN).shift(.25*LEFT).set_stroke(width=3, color=[REANLEA_WARM_BLUE,REANLEA_ORANGE])

        self.play(
            Create(indct_ln_1)
        )

        with RegisterFont("Cousine") as fonts:
            txt_1=Text("n dimensional space", font=fonts[0]).scale(.35)
            txt_1.set_color_by_gradient(REANLEA_BLUE_LAVENDER,REANLEA_TXT_COL_LIGHTER)
            txt_1.next_to(indct_ln_1,DOWN)

        self.play(
            Write(txt_1)
        )
        self.wait(2)

        rect_overlap_0=Rectangle(width=16, height=9, color=REANLEA_BACKGROUND_COLOR).to_edge(RIGHT, buff=0).set_opacity(.825).set_z_index(10)

        water_mark_1=water_mark.copy().set_z_index(11)

        self.play(
            FadeIn(water_mark_1),
            FadeIn(rect_overlap_0)
        )

        with RegisterFont("Cousine") as fonts:
            txt_2_0_0 = Text("Given", font=fonts[0]).scale(.35).set_z_index(11)
            txt_2_0_1=MathTex(r"x>0").scale(.6).next_to(txt_2_0_0,RIGHT).set_z_index(11)
            txt_2_0_2 = Text("and", font=fonts[0]).scale(.35).next_to(txt_2_0_1,RIGHT).set_z_index(11)
            txt_2_0_3=MathTex(r"n \in \mathbb{N}").scale(.6).next_to(txt_2_0_2,RIGHT).set_z_index(11)
            txt_2_0_4 = Text(", then there is a", font=fonts[0]).scale(.35).next_to(txt_2_0_3,RIGHT).set_z_index(11).shift(.15*LEFT)

            txt_2_0_4[0][0].shift(.075*DOWN)

            txt_2_0_grp=VGroup(txt_2_0_0,txt_2_0_1,txt_2_0_2,txt_2_0_3,txt_2_0_4).move_to(ORIGIN)

        with RegisterFont("Cousine") as fonts:
            txt_2_1_0 = Text("unique", font=fonts[0]).scale(.35).set_z_index(11)
            txt_2_1_1 = MathTex(r"y>0").scale(.6).next_to(txt_2_1_0,RIGHT).set_z_index(11)
            txt_2_1_2 = Text("such that", font=fonts[0]).scale(.35).next_to(txt_2_1_1,RIGHT).set_z_index(11)
            txt_2_1_3 = MathTex(r"y^{n}=x").scale(.6).next_to(txt_2_1_2,RIGHT).set_z_index(11)
            txt_2_1_4 = Text(".", font=fonts[0]).scale(.35).next_to(txt_2_1_3,RIGHT).set_z_index(11).shift(.15*LEFT+.075*DOWN)
            

            txt_2_1_grp=VGroup(txt_2_1_0,txt_2_1_1,txt_2_1_2,txt_2_1_3,txt_2_1_4).next_to(txt_2_0_grp,DOWN)

        with RegisterFont("Cousine") as fonts:
            txt_2_2_0 = Text("That is,", font=fonts[0]).scale(.35).set_z_index(11)
            txt_2_2_1 = MathTex(r"y=\sqrt[\leftroot{-1}\uproot{2}n]{x}").scale(.6).next_to(txt_2_2_0,RIGHT).set_z_index(11)
            txt_2_2_2 = Text("exists and is unique.", font=fonts[0]).scale(.35).next_to(txt_2_2_1,RIGHT).set_z_index(11)

            txt_2_2_grp=VGroup(txt_2_2_0,txt_2_2_1,txt_2_2_2).next_to(txt_2_1_grp,DOWN)

        with RegisterFont("Cousine") as fonts:
            txt_2_3_0 = Text("[ Hint: Consider,", font=fonts[0]).scale(.35).set_z_index(11)
            txt_2_3_1 = MathTex(r"y = l.u.b \: \{s \in \mathbb{R} : s^{n} \leq x\}").scale(.6).next_to(txt_2_3_0,RIGHT).set_z_index(11)
            txt_2_3_2 = Text("]", font=fonts[0]).scale(.35).set_z_index(11).next_to(txt_2_3_1,RIGHT).shift(.15*LEFT)

            txt_2_3_grp=VGroup(txt_2_3_0,txt_2_3_1,txt_2_3_2).next_to(txt_2_2_grp,DOWN).scale(.75)


        txt_2_grp=VGroup(txt_2_0_grp,txt_2_1_grp,txt_2_2_grp,txt_2_3_grp).move_to(ORIGIN)
        

        
        

        self.play(
            Write(txt_2_0_grp)
        )
        self.wait()

        self.play(
            Write(txt_2_1_grp)
        )
        self.wait()

        self.play(
            Write(txt_2_2_grp)
        )
        self.wait()

        self.play(
            Write(txt_2_3_grp)
        )
        self.wait(4)

        with RegisterFont("Fuzzy Bubbles") as fonts:
            txt_3 = Text("Cauchy-Schwarz Inequality", font=fonts[0]).set_z_index(11)

        self.play(
            ReplacementTransform(txt_2_grp,txt_3)
        )

        self.wait(2)

        water_mark_2=water_mark.copy()

        self.play(            
            AnimationGroup(
                *[FadeOut(mobj) for mobj in self.mobjects],
            ),
            FadeIn(water_mark_2),
            run_time=3
        )


        self.wait(4)




        # manim -pqh anim2.py Scene5_1

        # manim -pql anim2.py Scene5_1

        # manim -sqk anim2.py Scene5_1

        # manim -sqh anim2.py Scene5_1



###################################################################################################################

class Scene6(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        self.wait()


        # MAIN SCENE

        with RegisterFont("Homemade Apple") as fonts:
            txt_cs_0 = Text("Cauchy  Schwarz Inequality" , font=fonts[0])
            

        self.play(
            Write(txt_cs_0)
        )

        undr_bez_0=underline_bez_curve().next_to(txt_cs_0,DOWN).scale(2)

        self.play(
            Write(undr_bez_0)
        )

        self.wait(2)

        txt_cs_grp_0=VGroup(txt_cs_0,undr_bez_0)

        self.play(
            txt_cs_grp_0.animate.shift(1.5*UP).scale(.5)
        )

        with RegisterFont("Courier Prime") as fonts:
            txt_cs_def_0 = Text("For all" , font=fonts[0]).scale(.35)
        
        txt_cs_def_1=MathTex(r"x",",","y",r"\in" r"\mathbb{R}^{n}", ",").scale(.5).next_to(txt_cs_def_0,RIGHT).shift(.05*LEFT)
        txt_cs_def_1[1:].shift(.05*RIGHT)
        txt_cs_def_1[3:].shift(.05*RIGHT)
        txt_cs_def_1[4:].shift(.05*RIGHT)
        txt_cs_def_1[5:].shift(.05*RIGHT)

        txt_cs_def_2=MathTex(r"\langle x,y \rangle",r"\leq",r"\lVert x \rVert ",r"\lVert y \rVert ", ".").scale(.5).next_to(txt_cs_def_1,RIGHT).shift(.05*LEFT)
        txt_cs_def_2[1:].shift(.05*RIGHT)
        txt_cs_def_2[2:].shift(.05*RIGHT)
        txt_cs_def_2[3:].shift(.05*RIGHT)
        

        txt_cs_def=VGroup(txt_cs_def_0,txt_cs_def_1,txt_cs_def_2).scale(2).move_to(ORIGIN)

        self.play(
            Create(txt_cs_def),
            run_time=3
        )

        self.wait(3)

        txt_cs_grp_1=VGroup(txt_cs_grp_0,txt_cs_def)

        self.play(
            txt_cs_grp_1.animate.scale(.5).move_to(3*RIGHT+3*UP)
        )

        self.wait(2)



        with RegisterFont("Fuzzy Bubbles") as fonts:
            txt_0 = VGroup(*[Text(x, font=fonts[0]) for x in (
               "Inner",
               "Product" 
            )]).arrange_submobjects(DOWN).shift(5.75*LEFT).scale(.5)

        self.play(
            Create(txt_0),
            run_time=2
        )


        s0=AnnularSector(inner_radius=2,angle=2*PI).set_stroke(width=10, color=[REANLEA_AQUA,REANLEA_PURPLE]).shift(6*LEFT)

        x_trac_1=ValueTracker(0)
        x_trac_2_0=ValueTracker(0)
        x_trac_2_1=ValueTracker(0)
        x_trac_3=ValueTracker(0)

        s1=AnnularSector(inner_radius=2,angle=2*PI/7).set_stroke(width=10, color=[REANLEA_AQUA,REANLEA_PURPLE]).set_z_index(2).shift(6*LEFT).rotate(7*PI/10, about_point=6*LEFT)

        s1_ref=s1.copy()

        s1.add_updater(
            lambda x : x.become(s1_ref.copy()).rotate(
            x_trac_1.get_value(), about_point=6*LEFT
            )
        )

        s2_0=AnnularSector(inner_radius=2,angle=3*PI/14).set_stroke(width=10, color=[REANLEA_AQUA,REANLEA_PURPLE]).set_z_index(2).shift(6*LEFT).rotate(7*PI/10, about_point=6*LEFT)

        s2_0_ref=s2_0.copy()

        s2_0.add_updater(
            lambda x : x.become(s2_0_ref.copy()).rotate(
            x_trac_2_0.get_value(), about_point=6*LEFT
            )
        ) 

        s2_1=AnnularSector(inner_radius=2,angle=3*PI/14).set_stroke(width=10, color=[REANLEA_AQUA,REANLEA_PURPLE]).set_z_index(2).shift(6*LEFT).rotate(7*PI/10, about_point=6*LEFT)

        s2_1_ref=s2_1.copy()

        s2_1.add_updater(
            lambda x : x.become(s2_1_ref.copy()).rotate(
            x_trac_2_1.get_value(), about_point=6*LEFT
            )
        ) 

        s3=AnnularSector(inner_radius=2,angle=2*PI/7).set_stroke(width=10, color=[REANLEA_AQUA,REANLEA_PURPLE]).set_z_index(2).shift(6*LEFT).rotate(7*PI/10, about_point=6*LEFT)

        s3_ref=s3.copy()

        s3.add_updater(
            lambda x : x.become(s3_ref.copy()).rotate(
            x_trac_3.get_value(), about_point=6*LEFT
            )
        )

        self.wait()
        self.add(s1,s2_0,s2_1,s3)

        self.play(
            AnimationGroup(
                x_trac_1.animate.set_value((3*PI/10)+(5*PI/12)),
                x_trac_2_0.animate.set_value((3*PI/10)+(5*PI/12)+(2*PI/7)+(PI/18)),
                x_trac_2_1.animate.set_value((3*PI/10)+(5*PI/12)+(2*PI/7)+(PI/18)+(3*PI/14)+(PI/18)),
                x_trac_3.animate.set_value((3*PI/10)+(5*PI/12)+(2*PI/7)+((3*PI/7)+(PI/18))+(2*PI/18))
            ),
            run_time=1.35
        )

        grid=NumberPlane()

        dt_1=Dot(grid.polar_to_point(2, PI+(5*PI/12)+(2*PI/7)+(PI/18)-5*DEGREES)).shift(6*LEFT).set_color(PURE_GREEN).set_sheen(-.4,DOWN)

        dt_2=Dot(grid.polar_to_point(2, PI+(5*PI/12)+(2*PI/7)+(PI/18)+(3*PI/14)+(PI/18)-5*DEGREES)).shift(6*LEFT).set_color(REANLEA_WHITE).set_sheen(-.4,DOWN)

        dt_3=Dot(grid.polar_to_point(2, PI+(5*PI/12)+(2*PI/7)+((3*PI/7)+(PI/18))+(2*PI/18)-5*DEGREES)).shift(6*LEFT).set_color(REANLEA_YELLOW ).set_sheen(-.4,DOWN)

        self.play(
            AnimationGroup(
                Write(dt_1),
                Write(dt_2),
                Write(dt_3)
            ),
            AnimationGroup(
                Flash(dt_1, color=PURE_GREEN),
                Flash(dt_2, color=REANLEA_WHITE),
                Flash(dt_3, color=REANLEA_YELLOW)
            ),
            lag_ratio=.75
        )

        with RegisterFont("Fuzzy Bubbles") as fonts:

            txt_1=Text("Bilinear", font=fonts[0]).scale(.45).set_color_by_gradient(PURE_GREEN).next_to(dt_1,RIGHT)

            txt_2=Text("Symmetric", font=fonts[0]).scale(.45).set_color_by_gradient(REANLEA_WHITE).next_to(dt_2,RIGHT)

            txt_3=Text("Positive Definite", font=fonts[0]).scale(.45).set_color_by_gradient(REANLEA_YELLOW).next_to(dt_3,RIGHT)

        self.play(
            AnimationGroup(
                Write(txt_1),
                Write(txt_2),
                Write(txt_3)
            )
        )

        self.wait(2)

        innr_prdct_grp=VGroup(txt_0,txt_1,txt_2,txt_3,s1,s2_0,s2_1,s3,dt_1,dt_2,dt_3)

        ax_1=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).shift(2.25*RIGHT+.5*DOWN)

        self.play(
            Write(ax_1)
        )

        dot_0=Dot(ax_1.c2p(0,0)).set_color(REANLEA_BLUE_LAVENDER).set_sheen(-.4,DOWN).scale(.5).set_z_index(5)

        dot_i=Dot(ax_1.c2p(1,0)).set_color(REANLEA_BLUE_SKY).set_sheen(-.4,DOWN).set_z_index(5)

        self.play(
            Write(dot_0)
        )

        dot_1=Dot(ax_1.c2p(1,1)).set_color(REANLEA_BLUE_SKY).set_sheen(-.4,DOWN).scale(.5).set_z_index(5)

        dot_2=Dot(ax_1.c2p(2,1)).set_color(REANLEA_YELLOW).set_sheen(-.4,DOWN).scale(.5).set_z_index(5)

        arr_1=Arrow(start=ax_1.c2p(0,0),end=ax_1.c2p(1,1),tip_length=.125,stroke_width=4, buff=0).set_color_by_gradient(REANLEA_BLUE_SKY)

        arr_1_lbl=MathTex("x").scale(.6).move_to(ax_1.c2p(.3,.7)).set_color(REANLEA_BLUE_SKY)

        arr_2_x=Arrow(start=ax_1.c2p(0,0),end=ax_1.c2p(2,1),tip_length=.125,stroke_width=4, buff=0).set_color_by_gradient(REANLEA_YELLOW)

        arr_2_x_lbl=MathTex("y").scale(.6).move_to(ax_1.c2p(1.35,.3)).set_color(REANLEA_YELLOW)

        arr_2_ref=arr_2_x.copy().set_opacity(.5)

        self.play(
            Write(dot_1)
        )
        self.wait()
        self.play(
            Write(arr_1),
            ReplacementTransform(txt_cs_def_2[0][1].copy(), arr_1_lbl)
        )
        self.wait(2)

        self.play(
            Write(dot_2)
        )
        self.wait()
        self.play(
            Write(arr_2_x),
            ReplacementTransform(txt_cs_def_2[0][2].copy(), arr_2_x_lbl)
        )
        self.add(arr_2_ref)
        
        self.wait(2)

        self.play(
            arr_2_x.animate.shift(
                (ax_1.c2p(1,1)[1]-ax_1.c2p(0,0)[1])*UP+
                (ax_1.c2p(1,1)[0]-ax_1.c2p(0,0)[0])*RIGHT
            ),
            dot_2.animate.set_opacity(.5)
        )

        self.wait(2)

        dot_3=Dot(ax_1.c2p(3,2)).set_color(REANLEA_GREEN).set_sheen(-.4,DOWN).scale(.5).set_z_index(5)

        self.play(
            Write(dot_3)
        )

        arr_2=always_redraw(
            lambda : Arrow(start=ax_1.c2p(1,1),end=dot_3.get_center(),tip_length=.125,stroke_width=4, buff=0).set_color_by_gradient(REANLEA_YELLOW)
        )
        self.add(arr_2)
        self.play(
            FadeOut(arr_2_x)
        )

        arr_3=always_redraw(
            lambda : Arrow(start=ax_1.c2p(0,0),end=dot_3.get_center(),tip_length=.125,stroke_width=4, buff=0).set_color_by_gradient(REANLEA_GREEN)
        )

        arr_3_lbl=MathTex("w","=","x","+","y").scale(.6).move_to(ax_1.c2p(4,2))
        arr_3_lbl[0].set_color(REANLEA_GREEN)
        arr_3_lbl[2].set_color(REANLEA_BLUE_SKY)
        arr_3_lbl[4].set_color(REANLEA_YELLOW)


        self.wait()
        self.play(
            Write(arr_3),
            ReplacementTransform(VGroup(arr_1_lbl.copy(),arr_2_x_lbl.copy()),arr_3_lbl)
        )

        self.wait(2)

        lbl_1_1=MathTex("w","=","x","+","y").scale(1.25)
        lbl_1_1[0].set_color(REANLEA_GREEN)
        lbl_1_1[2].set_color(REANLEA_BLUE_SKY)
        lbl_1_1[3].scale(.65)
        lbl_1_1[4].set_color(REANLEA_YELLOW).shift(0.875*RIGHT)

        value_1=DecimalNumber(1).set_color(REANLEA_BLUE_LAVENDER).next_to(lbl_1_1[3])
        value_1.add_updater(
            lambda z : z.set_value(
                (np.sqrt(
                        np.square(dot_3.get_center()[0]-dot_1.get_center()[0])+
                        np.square(dot_3.get_center()[1]-dot_1.get_center()[1])
                )/
                np.sqrt(
                        np.square(dot_i.get_center()[0]-dot_0.get_center()[0])+
                        np.square(dot_i.get_center()[1]-dot_0.get_center()[1])
                ))/np.sqrt(5)
            )
        )

        lbl_1=VGroup(lbl_1_1,value_1).move_to(ax_1.c2p(3,3.5)).scale(.75)

        self.play(
            TransformMatchingShapes(arr_3_lbl,lbl_1)
        )

        self.play(
            dot_3.animate.move_to(ax_1.c2p(4,2.5))
        )
        self.wait()

        self.play(
            dot_3.animate.move_to(ax_1.c2p(6,3.5))
        )
        self.wait()

        self.play(
            dot_3.animate.move_to(ax_1.c2p(5,3))
        )
        self.wait()

        self.play(
            dot_3.animate.move_to(ax_1.c2p(7,4))
        )
        self.wait()

        self.play(
            dot_3.animate.move_to(ax_1.c2p(3,2))
        )

        self.play(
            dot_3.animate.move_to(ax_1.c2p(3.5,2.25))
        )
        
        lbl_2= MathTex("w","=","x","+","t","y").scale(.6).move_to(ax_1.c2p(4.5,2.25))
        lbl_2[0].set_color(REANLEA_GREEN)
        lbl_2[2].set_color(REANLEA_BLUE_SKY)
        lbl_2[3].scale(.65)
        lbl_2[4].set_color(REANLEA_BLUE_LAVENDER)
        lbl_2[5].set_color(REANLEA_YELLOW).shift(.05*RIGHT)

        self.play(
            TransformMatchingShapes(lbl_1,lbl_2)
        )
        self.wait(2)

        eqn_0_0=MathTex(r"\langle w,w \rangle").scale(.7).move_to(ax_1.c2p(1,3.5))
        eqn_0_1=MathTex("=",r"\lVert w \rVert ^{2}").scale(.7).next_to(eqn_0_0)
        eqn_0_1_ref=MathTex(r"\geq",r"0").scale(.7).next_to(eqn_0_1)
        eqn_0_2=MathTex("=",r"\lVert x + ty \rVert ^{2}").scale(.7).next_to(eqn_0_0)
        eqn_0_3=MathTex("=",r"\langle x + ty,x + ty \rangle").scale(.7).next_to(eqn_0_0).shift(.025*DOWN)
        eqn_0_4=MathTex("=",r"f(t)").scale(.7).next_to(eqn_0_3)

        self.play(
            TransformMatchingShapes(lbl_2[0].copy(),eqn_0_0)
        )
        self.wait()
        self.play(
            Write(eqn_0_1)
        )
        self.wait()

        ln_0=Line().scale(.25).set_stroke(width=1).rotate(35*DEGREES).next_to(eqn_0_1,UP).shift(.35*RIGHT+.25*DOWN)

        with RegisterFont("Cousine") as fonts:
            ln_0_lbl=Text(r"Scalar (Real Valued)", font=fonts[0]).scale(.25).set_color_by_gradient(REANLEA_BLUE_LAVENDER).next_to(ln_0.get_end()).shift(.1875*LEFT)

        self.play(
            Create(ln_0)
        )
        self.play(
            Write(ln_0_lbl)
        ) 
        self.wait(2) 
        self.play(
            Write(eqn_0_1_ref)
        )  
        self.wait()   
        self.play(
            Indicate(txt_3),
            FocusOn(txt_3)
        ) 
        self.wait(2)

        self.play(
            TransformMatchingShapes(eqn_0_1,eqn_0_2),
            Indicate(lbl_2[2:]),
            FocusOn(lbl_2[2:]),
            eqn_0_1_ref.animate.rotate(-30*DEGREES).scale(.5).next_to(eqn_0_2).shift(.3*DOWN+.35*LEFT)
        )
        self.wait()

        self.play(
            TransformMatchingShapes(eqn_0_2,eqn_0_3),
            eqn_0_1_ref.animate.next_to(eqn_0_3).shift(.25*DOWN+.2*LEFT)
        )
        self.wait(2)

        self.play(
            Write(eqn_0_4)
        )

        eqn_0=VGroup(eqn_0_0,eqn_0_3,eqn_0_4)

        rect_overlap=Rectangle(width=8, height=6, color=REANLEA_BACKGROUND_COLOR).to_edge(RIGHT, buff=0).set_opacity(.825).set_z_index(10).shift(.75*DOWN)

        eqn_1 = MathTex(r"f(t) &",r"= \langle x + ty,x + ty \rangle \\&",r"= \langle x,x \rangle +  2\langle x,y \rangle t + \langle y,y \rangle t^{2} \\&",r" = c+bt+at^{2}").scale(.7).move_to(2.5*RIGHT+1.5*UP).set_z_index(11)
        eqn_1[2][13:].shift(.1*RIGHT)
        eqn_1[2][20:].shift(.1*RIGHT)
        eqn_1[3].shift(1.25*DOWN)

        eqn_1_sym_1=MathTex(r"\geq 0").scale(.5).set_z_index(11).next_to(eqn_1[3],RIGHT).shift(.0675*DOWN)

        self.play(
            TransformMatchingShapes(eqn_0.copy(),eqn_1[0:2])
        )
        self.play(
            Write(rect_overlap)
        )
        self.wait()

        self.play(
            Write(eqn_1[2]),
            Indicate(txt_1),
            Indicate(txt_2),
            FocusOn(txt_1),
            FocusOn(txt_2)
        )
        self.wait(2)
        
        sr_eqn_1_2_0=SurroundingRectangle(eqn_1[2][1:6], buff=SMALL_BUFF*.85, color=REANLEA_WELDON_BLUE).set_stroke(width=1).set_z_index(11)
        sr_eqn_1_2_1=SurroundingRectangle(eqn_1[2][7:13], buff=SMALL_BUFF*.85, color=REANLEA_WELDON_BLUE).set_stroke(width=1).set_z_index(11)
        sr_eqn_1_2_2=SurroundingRectangle(eqn_1[2][15:20], buff=SMALL_BUFF*.85, color=REANLEA_WELDON_BLUE).set_stroke(width=1).set_z_index(11)

        

        self.play(
            Create(sr_eqn_1_2_0),
            Create(sr_eqn_1_2_1),
            Create(sr_eqn_1_2_2)
        )
        self.wait()

        indct_ln_0=Line(start=ax_1.c2p(2.825,3.55), end=ax_1.c2p(2.825,2.75)).set_stroke(width=1.5, color=[REANLEA_BLUE_SKY,REANLEA_MAGENTA]).set_z_index(11)

        indct_ln_1=Line(start=ax_1.c2p(1.25,3.55), end=ax_1.c2p(2.825,2.75)).set_stroke(width=1.5, color=[REANLEA_BLUE_SKY,REANLEA_MAGENTA]).set_z_index(11)

        indct_ln_2=Line(start=ax_1.c2p(4.4,3.55), end=ax_1.c2p(2.825,2.75)).set_stroke(width=1.5, color=[REANLEA_MAGENTA,REANLEA_BLUE_SKY]).set_z_index(11)

        indct_ln_grp=VGroup(indct_ln_0,indct_ln_1,indct_ln_2)

        self.play(
            Write(indct_ln_grp)
        )
        self.wait()

        with RegisterFont("Cousine") as fonts:
            indct_ln_grp_lbl=Text(r"Scalar ", font=fonts[0]).scale(.45).set_color_by_gradient(REANLEA_BLUE_SKY).move_to(ax_1.c2p(2.825,2.5)).set_z_index(11)

        self.play(
            Write(indct_ln_grp_lbl)
        )
        self.wait(2)

        eqn_2=MathTex(r"&",r"c=\langle x,x \rangle \\&" , r"b= 2\langle x,y \rangle \\&", r"a=\langle y,y \rangle \\&").scale(.55).set_color_by_gradient(REANLEA_MAGENTA,REANLEA_BLUE_SKY).move_to(6*RIGHT).set_z_index(11)

        self.play(
            TransformMatchingShapes(VGroup(eqn_1[2][1:6],eqn_1[2][7:13], eqn_1[2][15:20]).copy(),eqn_2)
        )
        sep_ln_0=Line().set_stroke(width=1,color=[REANLEA_BLUE_LAVENDER,REANLEA_CYAN_LIGHT]).scale(.75).rotate(PI/2).next_to(eqn_2,LEFT).set_z_index(11)

        self.play(
            Create(sep_ln_0.reverse_direction())
        )

        self.play(
            Write(eqn_1[3])
        )
        self.wait(2)

        self.play(
            Write(eqn_1_sym_1)
        )
        self.wait()

        ln_1=Line().scale(.25).set_stroke(width=1).rotate(-35*DEGREES).next_to(eqn_1,DOWN).shift(.5*LEFT+.15*UP).set_z_index(11)

        with RegisterFont("Cousine") as fonts:
            ln_1_lbl=Text(r"quadratic polynomial", font=fonts[0]).scale(.25).set_color_by_gradient(REANLEA_BLUE_LAVENDER).next_to(ln_1.get_end()).shift(.175*LEFT).set_z_index(11)

        self.play(
            AnimationGroup(
            Create(ln_1),
            Write(ln_1_lbl),
            Lag_ratio=1
            )
        )
        self.wait(2)

        eqn_3=MathTex(r"f(t) > 0").scale(.7).set_color_by_gradient(REANLEA_PINK,REANLEA_SLATE_BLUE_LIGHTER).move_to(1.75*DOWN).set_z_index(11)

        self.play(
            Write(eqn_3)
        )


        rect_overlap_1=Rectangle(width=8, height=6, color=REANLEA_BACKGROUND_COLOR).to_edge(LEFT, buff=0).set_opacity(.825).set_z_index(10).shift(.75*DOWN)

        ax_2_1=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.2).shift(5*LEFT+2*UP).set_z_index(11)

        ax_2_1_x=ax_2_1.copy()

        ax_2_2=ax_2_1.copy().next_to(ax_2_1_x,DOWN)
        ax_2_3=ax_2_1.copy().next_to(ax_2_2,DOWN)

        
        self.play(
            FadeIn(rect_overlap_1),
            Write(ax_2_1)
        )

        graph_1=ax_2_1.plot(
            lambda x: x**2-6*x+8 , x_range=[0.551,5.449]
        ).set_stroke(width=7, color=[REANLEA_BLUE,REANLEA_WARM_BLUE]).scale(.5).set_z_index(11)

        self.play(
            Create(graph_1)
        )

        eqn_3_1=MathTex(r"\Rightarrow",r"f(t) \neq 0",",",r"\forall t").scale(.7).set_color_by_gradient(REANLEA_SLATE_BLUE_LIGHTER, REANLEA_BLUE_SKY).next_to(eqn_3).set_z_index(11)
        eqn_3_1[2].scale(.7).shift(.025*DOWN)
        eqn_3_1[3].shift(.05*RIGHT)
        eqn_3_1[3][1:].shift(.075*RIGHT)

        self.play(
            Write(eqn_3_1)
        )
        self.wait(2)

        eqn_3_grp=VGroup(eqn_3,eqn_3_1)

        self.play(
            eqn_3_grp.animate.shift(LEFT)
        )

        sep_ln_1=Line().set_stroke(width=1,color=[REANLEA_BLUE_LAVENDER,REANLEA_CYAN_LIGHT]).scale(.75).rotate(PI/2).next_to(eqn_3_grp,RIGHT).shift(.5*RIGHT+.5*DOWN).set_z_index(11)

        self.play(
            Create(sep_ln_1.reverse_direction())
        )

        eqn_4_1=MathTex(r"f(t) = 0").scale(.7).set_color_by_gradient(REANLEA_YELLOW_GREEN, REANLEA_GOLD).move_to(1.75*DOWN+5*RIGHT).set_z_index(11)

        eqn_4_2=MathTex(r"c+bt+at^{2} = 0").scale(.7).set_color_by_gradient(REANLEA_YELLOW_GREEN, REANLEA_GOLD).move_to(1.75*DOWN+5*RIGHT).set_z_index(11)

        eqn_4_3=MathTex(r"\Rightarrow",r"t=\frac{-b \pm \sqrt{b^{2}-4ac}}{2a}").scale(.7).set_color_by_gradient(REANLEA_YELLOW_GREEN, REANLEA_GOLD).next_to(eqn_4_2,DOWN).set_z_index(11)

        eqn_4_4=MathTex(r"\notin \mathbb{R}").scale(.55).next_to(eqn_4_3,DOWN).shift(1.25*RIGHT+.25*UP).set_z_index(11)

        self.play(
            Write(eqn_4_1)
        )
        self.wait(2)

        self.play(
            ReplacementTransform(eqn_4_1,eqn_4_2)
        )
        self.wait()

        self.play(
            Create(eqn_4_3)
        )

        graph_1_lbl_1 = MathTex(r"f > 0").scale(.5).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_SLATE_BLUE).next_to(graph_1,RIGHT).shift(.35*UP+.45*RIGHT).set_z_index(11)

        self.play(
            Write(graph_1_lbl_1)
        )

        with RegisterFont("Cousine") as fonts:
            graph_1_lbl_2=Text(r"no real root", font=fonts[0]).scale(.25).set_color_by_gradient(REANLEA_BLUE_LAVENDER).next_to(graph_1_lbl_1,DOWN).shift(.15*UP).set_z_index(11)
        
        self.play(
            Write(graph_1_lbl_2)
        )

        eqn_5_1=MathTex(r"b^{2}-4ac",r"< 0").scale(.7).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_SLATE_BLUE).next_to(eqn_3_grp,DOWN).shift(.5*DOWN).set_z_index(11)
        eqn_5_1[1].scale(.7)

        eqn_5_1_ref=eqn_5_1.copy()

        graph_1_lbl_3=MathTex(r"b^{2}-4ac",r"< 0").scale(.5).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_PURPLE).next_to(graph_1_lbl_2,DOWN).set_z_index(11)

        self.play(
            TransformMatchingShapes(eqn_4_3[1].copy(),eqn_5_1[0])
        )
        self.play(
            Write(eqn_5_1[1])
        )

        eqn_5_2=MathTex(r"\sqrt{b^{2}-4ac}",r"\in \mathbb{C} \setminus \mathbb{R}").scale(.7).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_SLATE_BLUE).next_to(eqn_3_grp,DOWN).shift(.5*DOWN).set_z_index(11)
        eqn_5_2[1].scale(.7).set_color_by_gradient(PURE_RED,REANLEA_GOLD)

        self.play(
            TransformMatchingShapes(eqn_5_1,eqn_5_2),
            Write(eqn_4_4)
        )
        self.wait()

        self.play(
            ReplacementTransform(eqn_5_1_ref,graph_1_lbl_3)
        )

        self.wait(2)

        eqn_6_1=MathTex(r"b^{2}-4ac",r"= 0").scale(.7).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_SLATE_BLUE).shift(1.75*DOWN+RIGHT).set_z_index(11)

        self.play(
            FadeOut(eqn_3_grp),
            FadeOut(eqn_4_4),
            TransformMatchingShapes(eqn_5_2,eqn_6_1)
        )
        self.wait(2)

        indct_arr_1=MathTex(r"\rightarrow").set_stroke(width=3, color=[REANLEA_WARM_BLUE,REANLEA_PINK]).rotate(-PI/2).scale(.75).next_to(eqn_6_1,DOWN).set_z_index(11)

        self.play(
            Create(indct_arr_1)
        )

        eqn_6_2=MathTex(r"t=\frac{-b \pm \sqrt{b^{2}-4ac}}{2a}").scale(.5).set_color_by_gradient(REANLEA_PINK,REANLEA_YELLOW_CREAM, REANLEA_SLATE_BLUE_LIGHTER,REANLEA_AQUA_GREEN).next_to(indct_arr_1,DOWN).set_z_index(11)

        self.play(
            TransformMatchingShapes(eqn_4_3.copy(),eqn_6_2)
        )

        eqn_6_2_1=MathTex("0").scale(.5).set_color_by_gradient(REANLEA_SLATE_BLUE_LIGHTER).move_to(eqn_6_2[0][5:13].get_center()+.5*LEFT).set_z_index(11)

        self.play(
            Indicate(indct_arr_1),
            Transform(eqn_6_2[0][5:13],eqn_6_2_1)
        )

        eqn_6_3=MathTex(r"t=\frac{-b \pm 0}{2a}").scale(.5).set_color_by_gradient(REANLEA_PINK,REANLEA_YELLOW_CREAM, REANLEA_SLATE_BLUE_LIGHTER,REANLEA_AQUA_GREEN).next_to(indct_arr_1,DOWN).set_z_index(11)

        self.play(
            TransformMatchingShapes(eqn_6_2,eqn_6_3)
        )
        self.wait(2)

        indct_ln_3=Line().scale(.175).set_stroke(width=1).rotate(-35*DEGREES).next_to(eqn_6_3,DOWN).shift(.2*UP+.5*RIGHT).set_z_index(11)

        with RegisterFont("Cousine") as fonts:
            indct_ln_3_lbl=Text(r"double root", font=fonts[0]).scale(.25).set_color_by_gradient(REANLEA_BLUE_LAVENDER).next_to(indct_ln_3.get_end()).shift(.175*LEFT).set_z_index(11)

        self.play(
            Create(indct_ln_3)
        )
        self.play(
            Write(indct_ln_3_lbl)
        )

        self.wait(2)

        self.play(
            Write(ax_2_2)
        )

        graph_2=ax_2_2.plot(
            lambda x: x**2-6*x+8 , x_range=[0.551,5.449]
        ).set_stroke(width=7, color=[REANLEA_BLUE,REANLEA_WARM_BLUE]).scale(.5).set_z_index(11)
        graph_2.shift(
            (ax_2_2.c2p(3,.5)-ax_2_2.c2p(3,0))*DOWN
        )

        dt_1=Dot(ax_2_2.c2p(3,0)).scale(.675).set_color(PURE_GREEN).set_z_index(11)

        self.play(
            AnimationGroup(
                Create(graph_2)
            ),
            AnimationGroup(
                Create(dt_1),
                Flash(dt_1.get_center(), color=PURE_GREEN),
                lag_ratio=.15
            ),
            lag_ratio=.05
        )


        graph_2_lbl_1 = MathTex(r"f \geq 0").scale(.5).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_SLATE_BLUE).next_to(graph_2,RIGHT).shift(.35*UP+.45*RIGHT).shift(
            (ax_2_2.c2p(3,.5)-ax_2_2.c2p(3,0))*UP
        ).set_z_index(11)

        with RegisterFont("Cousine") as fonts:
            graph_2_lbl_2=Text(r"one double root", font=fonts[0]).scale(.25).set_color_by_gradient(REANLEA_BLUE_LAVENDER).next_to(graph_2_lbl_1,DOWN).shift(.15*UP).set_z_index(11)

        self.play(
            Write(graph_2_lbl_1)
        )
        self.wait(2)

        self.play(
            Write(graph_2_lbl_2)
        )

        self.wait(2)

        graph_2_lbl_3=MathTex(r"b^{2}-4ac",r"= 0").scale(.5).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_PURPLE).next_to(graph_2_lbl_2,DOWN).set_z_index(11)

        self.play(
            TransformMatchingShapes(eqn_6_1.copy(),graph_2_lbl_3)
        )
        self.wait(2)

        eqn_7_1=MathTex(r"b^{2}-4ac",r"> 0").scale(.7).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_SLATE_BLUE).shift(1.75*DOWN+RIGHT).set_z_index(11)

        eqn_7_2=MathTex(r"t=\frac{-b \pm \sqrt{b^{2}-4ac}}{2a}").scale(.5).set_color_by_gradient(REANLEA_PINK,REANLEA_YELLOW_CREAM, REANLEA_SLATE_BLUE_LIGHTER,REANLEA_AQUA_GREEN).next_to(indct_arr_1,DOWN).set_z_index(11)

        with RegisterFont("Cousine") as fonts:
            indct_ln_3_lbl_2=Text(r"real roots", font=fonts[0]).scale(.25).set_color_by_gradient(REANLEA_BLUE_LAVENDER).next_to(indct_ln_3.get_end()).shift(.175*LEFT).set_z_index(11)
        
        graph_3=ax_2_3.plot(
            lambda x: x**2-6*x+8 , x_range=[0.551,5.449]
        ).set_stroke(width=7, color=[REANLEA_BLUE,REANLEA_WARM_BLUE]).scale(.5).set_z_index(12)
        graph_3.shift(
            (ax_2_2.c2p(3,.5)-ax_2_2.c2p(3,-.5))*DOWN
        )

        dt_2=Dot(ax_2_3.c2p(2.5,0)).scale(.675).set_color(PURE_GREEN).set_z_index(12)
        dt_3=Dot(ax_2_3.c2p(3.5,0)).scale(.675).set_color(PURE_GREEN).set_z_index(12)

        self.play(
            AnimationGroup(
                TransformMatchingShapes(eqn_6_1,eqn_7_1),
                TransformMatchingShapes(eqn_6_3,eqn_7_2),
                TransformMatchingShapes(indct_ln_3_lbl,indct_ln_3_lbl_2),
            ),
            AnimationGroup(
                Write(ax_2_3),
                AnimationGroup(
                    AnimationGroup(
                        Create(graph_3),
                        AnimationGroup(
                            Create(dt_2),
                            Flash(dt_2.get_center(), color=REANLEA_BLUE_LAVENDER),
                            lag_ratio=.15
                        ),             
                        lag_ratio=.05
                    ),
                    AnimationGroup(
                        Create(dt_3),
                        Flash(dt_3.get_center(), color=REANLEA_BLUE_LAVENDER),
                        lag_ratio=.15
                    ),
                    lag_ratio=.2
                ),
                lag_ratio=1
            )
        )
        self.wait(2)

        r1=Polygon(ax_2_3.c2p(-1.5,0),ax_2_3.c2p(5.5,0),ax_2_3.c2p(5.5,-1.5),ax_2_3.c2p(-1.5,-1.5)).set_opacity(0).set_z_index(11)
        r1.set_fill(color=REANLEA_CHARM, opacity=0.15).set_z_index(11)

        self.play(
            Create(r1)
        )

        sgn_neg_1=MathTex("-").scale(.75).set_stroke(width=2).set_color(PURE_RED)
        sgn_neg_2=Circle(radius=0.2, color=PURE_RED).move_to(sgn_neg_1.get_center()).set_stroke(width= 1)
        sgn_neg=VGroup(sgn_neg_1,sgn_neg_2).scale(.35).move_to(ax_2_3.c2p(-.75,-.75)).set_z_index(12)

        self.play(
            Write(sgn_neg)
        )
        self.wait(2)

        dt_4=Dot(ax_2_3.c2p(3,-.5)).scale(.675).set_color(PURE_RED).set_z_index(12)

        d_ln_1=DashedLine(start=ax_2_3.c2p(3,-.5), end=ax_2_3.c2p(0,-.5), stroke_width=1).set_color(PURE_RED).set_z_index(12)

        self.play(
            Write(dt_4)
        )
        self.play(
            Create(d_ln_1)
        )
        self.wait(2)

        dt_4_lbl_1=MathTex(r"f(t) < 0").scale(.25).set_color_by_gradient(REANLEA_GOLDENROD,REANLEA_YELLOW_CREAM).move_to(ax_2_3.c2p(3.5,-1)).set_z_index(12)

        self.play(
            Write(dt_4_lbl_1)
        )

        self.wait(2)

        graph_3_lbl_1_0 = MathTex(r"f ").scale(.5).next_to(graph_3,RIGHT).shift(.35*UP+.25*RIGHT).shift(
            (ax_2_2.c2p(3,.5)-ax_2_2.c2p(3,-.5))*UP
        ).set_z_index(11)
        with RegisterFont("Cousine") as fonts:
            graph_3_lbl_1_1 = Text(r"is both + & -", font=fonts[0]).scale(.25).set_color_by_gradient(REANLEA_BLUE_LAVENDER).next_to(graph_3_lbl_1_0,RIGHT).set_z_index(11)

        graph_3_lbl_1=VGroup(graph_3_lbl_1_0,graph_3_lbl_1_1).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_SLATE_BLUE)

        with RegisterFont("Cousine") as fonts:
            graph_3_lbl_2=Text(r"two real roots", font=fonts[0]).scale(.25).set_color_by_gradient(REANLEA_BLUE_LAVENDER).next_to(graph_3_lbl_1,DOWN).shift(.15*UP).set_z_index(11)

        self.play(
            Write(graph_3_lbl_1)
        )
        self.wait(2)

        self.play(
            Write(graph_3_lbl_2)
        )

        self.wait(2)

        graph_3_lbl_3=MathTex(r"b^{2}-4ac",r"> 0").scale(.5).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_PURPLE).next_to(graph_3_lbl_2,DOWN).set_z_index(11)

        self.play(
            TransformMatchingShapes(eqn_7_1.copy(),graph_3_lbl_3)
        )
        self.wait(2)


        rect_overlap_2=Rectangle(width=8, height=2.5, color=REANLEA_BACKGROUND_COLOR).to_edge(RIGHT, buff=0).set_opacity(.825).set_z_index(20).shift(2.5*DOWN)

        self.play(
            Create(rect_overlap_2)
        )

        eqn_1_grp_1=VGroup(eqn_1,eqn_1_sym_1)

        eqn_8_1= MathTex(r"f \geq 0").scale(.75).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_SLATE_BLUE).shift(1.75*DOWN).set_z_index(21)

        self.play(
            Indicate(eqn_1_grp_1),
            Write(eqn_8_1)
        )

        indct_arr_2=MathTex(r"\rightarrow").set_stroke(width=3, color=[REANLEA_WARM_BLUE,REANLEA_PINK]).rotate(-PI/2).scale(.75).next_to(eqn_8_1,DOWN).set_z_index(21)

        self.play(
            Create(indct_arr_2)
        )

        graph_1_lbl_3.set_z_index(21)
        graph_2_lbl_3.set_z_index(21)

        graph_1_lbl_3_ref=MathTex(r"b^{2}-4ac",r"< 0").scale(.5).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_PURPLE).next_to(indct_arr_2,DOWN).set_z_index(21)

        graph_2_lbl_3_ref=MathTex(r"b^{2}-4ac",r"= 0").scale(.5).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_PURPLE).next_to(indct_arr_2,DOWN).shift(.5*DOWN).set_z_index(21)

        self.play(
            TransformMatchingShapes(graph_1_lbl_3.copy(),graph_1_lbl_3_ref)
        )
        self.wait()

        self.play(
            TransformMatchingShapes(graph_2_lbl_3.copy(),graph_2_lbl_3_ref)
        )
        self.wait()

        graph_1_2_lbl_3_ref_grp=VGroup(graph_1_lbl_3_ref,graph_2_lbl_3_ref)

        eqn_8_2=MathTex(r"b^{2}-4ac",r"\leq 0").scale(.5).set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_PURPLE).next_to(indct_arr_2,DOWN).set_z_index(21)

        self.play(
            TransformMatchingShapes(graph_1_2_lbl_3_ref_grp,eqn_8_2)
        )
        self.wait(2)

        indct_ln_4=Line().scale(.65).set_stroke(width=1).rotate(50*DEGREES).next_to(eqn_8_2,RIGHT).shift(.6*UP+.15*LEFT).set_z_index(21)

        self.play(
            Create(indct_ln_4)
        )

        eqn_2.set_z_index(21)

        eqn_8_3=MathTex(
            r"(2 \langle x,y \rangle)^{2}",r"-",r"4 \langle x,x \rangle \langle y,y \rangle","\leq 0"
        ).scale(.5).set_color_by_gradient(REANLEA_PINK,REANLEA_BLUE_SKY).next_to(indct_ln_4.get_end(),RIGHT).shift(.1*UP+.1*LEFT).set_z_index(21)

        self.play(
            ReplacementTransform(VGroup(eqn_8_2,eqn_2).copy(),eqn_8_3)
        )
        self.wait(2)

        eqn_8_4=MathTex(
            r"4 \langle x,y \rangle^{2}",r"-",r"4 \langle x,x \rangle \langle y,y \rangle","\leq 0"
        ).scale(.5).set_color_by_gradient(REANLEA_PINK,REANLEA_BLUE_SKY).next_to(indct_ln_4.get_end(),RIGHT).shift(.1*UP+.1*LEFT).set_z_index(21)

        self.play(
            TransformMatchingShapes(eqn_8_3,eqn_8_4)
        )
        self.wait(2)

        eqn_8_5=MathTex(
            r"4 \langle x,y \rangle^{2}",r"\leq",r"4 \langle x,x \rangle \langle y,y \rangle"
        ).scale(.5).set_color_by_gradient(REANLEA_PINK,REANLEA_BLUE_SKY).next_to(indct_ln_4.get_end(),RIGHT).shift(.1*UP+.1*LEFT).set_z_index(21)

        self.play(
            TransformMatchingShapes(eqn_8_4,eqn_8_5)
        )
        self.wait(2)

        eqn_8_6=MathTex(
            r"\langle x,y \rangle^{2}",r"\leq",r"\langle x,x \rangle \langle y,y \rangle"
        ).scale(.5).set_color_by_gradient(REANLEA_PINK,REANLEA_BLUE_SKY).next_to(indct_ln_4.get_end(),RIGHT).shift(.1*UP+.1*LEFT).set_z_index(21)

        self.play(
            TransformMatchingShapes(eqn_8_5,eqn_8_6)
        )
        self.wait(2)

        eqn_8_7=MathTex(
            r"\langle x,y \rangle^{2}",r"\leq",r"\lVert x \rVert^{2} \lVert y \rVert^{2}"
        ).scale(.5).set_color_by_gradient(REANLEA_PINK,REANLEA_BLUE_SKY).next_to(indct_ln_4.get_end(),RIGHT).shift(.1*UP+.1*LEFT).set_z_index(21)

        self.play(
            TransformMatchingShapes(eqn_8_6,eqn_8_7)
        )
        self.wait(2)

        eqn_8_8=MathTex(
            r"\langle x,y \rangle^{2}",r"\leq",r"(\lVert x \rVert \ \lVert y \rVert)^{2}"
        ).scale(.5).set_color_by_gradient(REANLEA_PINK,REANLEA_BLUE_SKY).next_to(indct_ln_4.get_end(),RIGHT).shift(.1*UP+.1*LEFT).set_z_index(21)

        self.play(
            TransformMatchingShapes(eqn_8_7,eqn_8_8)
        )
        self.wait(2)

        brace_1=Brace(eqn_8_8[2][1:8])

        self.play(
            Write(brace_1)
        )


        










        self.wait(4)

        

        




        # manim -pqh anim2.py Scene6

        # manim -pql anim2.py Scene6

        # manim -sqh anim2.py Scene6

        # manim -sqk anim2.py Scene6
        

        
###################################################################################################################

# Changing FONTS : import any font from Google
# some of my fav fonts: Cinzel,Kalam,Prata,Kaushan Script,Cormorant, Handlee,Monoton, Bad Script, Reenie Beanie, 
# Poiret One,Merienda,Julius Sans One,Merienda One,Cinzel Decorative, Montserrat, Cousine, Courier Prime , The Nautigal
# Marcellus SC,Contrail One,Thasadith,Spectral SC,Dongle,Cormorant SC,Comfortaa, Josefin Sans (LOVE), Fuzzy Bubbles

###################################################################################################################


###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix" 