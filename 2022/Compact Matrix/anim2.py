############################################# by GOBINDA CHANDRA ###################################################

                                    # VISIT    : https://reanlea.com/ 
                                    # YouTube  : https://www.youtube.com/Reanlea/ 
                                    # Twitter  : https://twitter.com/Reanlea_ 
                                    # Facebook : https://www.facebook.com/reanlea.ed/ 
                                    # Telegram : https://t.me/reanlea/ 

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
        

        self.play(
            Write(dt_1_dummy_lbl_1)
        )
        self.wait()
        self.play(
            TransformMatchingShapes(dt_1_dummy_lbl_1,dt_1_dummy_lbl_2)
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
        dt_1_dummy_2=dt_1.copy().set_color(PURE_GREEN).move_to(.65*ax_1.c2p(0,0)+.35*ax_1.c2p(4,2))
        dt_1_dummy_2.set_sheen(-.4,DOWN)

        self.add(dt_1_dummy_2)
        self.play(
            dt_1_dummy_1.animate.move_to(
                .25*ax_1.c2p(0,0)+.75*ax_1.c2p(4,2)
            )
        )
        

        self.wait(2)





        # manim -pqh anim2.py Scene2

        # manim -pql anim2.py Scene2

        # manim -sqk anim2.py Scene2

        # manim -sql anim2.py Scene2




###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix" 