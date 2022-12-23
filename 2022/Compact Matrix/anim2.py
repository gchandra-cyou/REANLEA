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
            Write(arr_10)
        )
        self.play(
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
            TransformMatchingShapes(txt_6, txt_9)
        )
        


        

        self.wait(4)





        # manim -pqh anim2.py Scene2

        # manim -pql anim2.py Scene2

        # manim -sqk anim2.py Scene2

        # manim -sql anim2.py Scene2




###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix" 