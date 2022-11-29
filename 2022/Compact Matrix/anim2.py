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
        



        self.wait(4)





        # manim -pqh anim2.py Scene1

        # manim -pql anim2.py Scene1

        # manim -sqk anim2.py Scene1

        # manim -sql anim2.py Scene1




###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix" 