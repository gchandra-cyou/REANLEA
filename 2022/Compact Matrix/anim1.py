############################################# by GOBINDA CHANDRA ###################################################

                                    # VISIT : https://reanlea.com/ 
                                    # YouTube : https://www.youtube.com/Reanlea/ 
                                    # Twitter : https://twitter.com/Reanlea_ 
                                    # Facebook : https://www.facebook.com/reanlea.ed/ 
                                    # Telegram : https://t.me/reanlea/ 

#####################################################################################################################

from __future__ import annotations
from ast import Return
from cProfile import label
from difflib import restore


import fractions
from imp import create_dynamic
from multiprocessing import context
from multiprocessing import dummy
from multiprocessing.dummy import Value
from numbers import Number
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

config.max_files_cached=500

config.background_color= REANLEA_BACKGROUND_COLOR


###################################################################################################################

class Scene1(Scene):
    def construct(self):
        
        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        # TRACKER

        x=ValueTracker(-2)

        # OBJECT REGION

        line_1= NumberLine(
            x_range=[-8, 8, 1],
            length=32,
            include_ticks=False,
        ).set_color(REANLEA_BLUE_LAVENDER).set_stroke(width=4).move_to(DOWN)

        line_1_center=line_1.n2p(-2)

        zero_tick = VGroup(
            Line(0.3 * UP, 0.3 * DOWN, stroke_width=2.0, color=REANLEA_VIOLET_LIGHTER),
            MathTex("0"),
        )
        zero_tick[0].move_to(line_1.n2p(-2))
        zero_tick[1].next_to(zero_tick[0], DOWN)
        zero_tick.set_z_index(3)

        one_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_GREEN),
            MathTex("1").scale(.5),
        )
        one_tick[0].move_to(line_1.n2p(-1))
        one_tick[1].next_to(one_tick[0], DOWN)
        one_tick.set_z_index(3)

        minus_one_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_YELLOW),
            MathTex("-1").scale(.5),
        )
        minus_one_tick[0].move_to(line_1.n2p(-3))
        minus_one_tick[1].next_to(minus_one_tick[0], DOWN)
        minus_one_tick.set_z_index(3)


        two_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_GREEN),
            MathTex("2").scale(.5),
        )
        two_tick[0].move_to(line_1.n2p(0))
        two_tick[1].next_to(two_tick[0], DOWN)
        two_tick.set_z_index(3)

        three_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_GREEN),
            MathTex("3").scale(.5),
        )
        three_tick[0].move_to(line_1.n2p(1))
        three_tick[1].next_to(three_tick[0], DOWN)
        three_tick.set_z_index(3)

        four_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_GREEN),
            MathTex("4").scale(.5),
        )
        four_tick[0].move_to(line_1.n2p(2))
        four_tick[1].next_to(four_tick[0], DOWN)
        four_tick.set_z_index(3)

        five_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_GREEN),
            MathTex("5").scale(.5),
        )
        five_tick[0].move_to(line_1.n2p(3))
        five_tick[1].next_to(five_tick[0], DOWN)
        five_tick.set_z_index(3)


        dot_1=Dot(radius=0.2, color=REANLEA_VIOLET_LIGHTER).move_to(line_1.n2p(-2)).set_sheen(-0.4,DOWN).set_z_index(1)
        dot_2=dot_1.copy().set_opacity(0.4)






        line_2=Line().move_to(UP).set_color(PURE_RED).set_z_index(4)
        line_2.save_state()

        brace_line_2=Brace(line_2, stroke_width=.01).set_color(PURE_GREEN).set_opacity(0.5)
        brace_line_2_label=brace_line_2.get_tex("1").scale(0.65).set_color(REANLEA_GREEN).shift(.25*UP)


        line_3=line_2.copy().move_to(0.5*UP+1.5*LEFT)
        line_4=line_2.copy().move_to(0.5*UP+1.5*RIGHT)

        d_line_1=DashedDoubleArrow(
            start=line_1.n2p(-2), end=line_1.n2p(0), dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA).shift(0.5*UP)

        d_line_1_label=MathTex("1+1").set_color_by_gradient(REANLEA_TXT_COL_LIGHTER).scale(0.5).next_to(d_line_1,0.5*UP)

        line_3_4=Line(start=line_1.n2p(-2), end=line_1.n2p(0))

        brace_line_3_4=Brace(line_3_4, stroke_width=.01).set_color(REANLEA_GREY).set_opacity(0.5).shift(0.75*DOWN)
        brace_line_3_4_label=MathTex("2",r"\times","1").scale(0.65).set_color(REANLEA_TXT_COL_LIGHTER).next_to(brace_line_3_4,0.5*DOWN)

        # TEXT REGION 

        so_on_txt_symbol=Text("...").move_to(0.9*DOWN+6.9*RIGHT).scale(0.5).set_color(REANLEA_GREEN)

        # EQUATION REGION

        eq_1_1=MathTex("1","+","1","=","?").scale(1.3).set_color(REANLEA_PURPLE_LIGHTER)#.set_color_by_gradient(REANLEA_GREEN_AUQA,REANLEA_PURPLE)
        eq_1_2=MathTex("1","+","1","=","2").scale(1.3).set_color(REANLEA_PURPLE_LIGHTER)#.set_color_by_gradient(REANLEA_GREEN_AUQA,REANLEA_PURPLE)
        eq_1_3=MathTex("=","2",r"\times","1").scale(1.3).next_to(eq_1_2).set_color(REANLEA_PURPLE_LIGHTER)#.set_color_by_gradient(REANLEA_PURPLE,REANLEA_PINK_DARKER)

        # UPDATER REGION

        dot_1.add_updater(lambda z : z.move_to(line_1.n2p(x.get_value())))

        # GROUP REGION

        eq_1_grp=VGroup(eq_1_1,eq_1_2,eq_1_3)











        # PLAY REGION
        self.play(
            Write(eq_1_1)
        )
        self.wait(2)
        self.play(
            ReplacementTransform(eq_1_1,eq_1_2),     
        )
        self.wait(2)
        self.play(Write(eq_1_3))
        self.wait(2)
        
        self.play(
            eq_1_grp.animate.move_to(2.5*UP).scale(0.7),
            Create(line_1)
        )
        self.wait(2)
        self.play(
            Write(zero_tick)
        )
        self.wait()
        self.play(Create(line_2))
        self.play(Write(brace_line_2))
        self.play(
            ReplacementTransform(eq_1_2[0].copy(),brace_line_2_label)
        )
        self.wait()
        self.play(
            Uncreate(brace_line_2),
            Uncreate(brace_line_2_label)
        )
        self.play(
            line_2.animate.move_to(line_1_center+RIGHT)
        )
        self.wait()
        self.play(
            FocusOn(focus_point=line_1.n2p(-1), opacity=.27)
        )
        self.play(
            Create(one_tick)
        )
        self.wait(2)



        self.play(
            line_2.animate.move_to(line_1_center+LEFT)
        )
        self.play(
            FocusOn(focus_point=line_1.n2p(-3), opacity=.27)
        )
        self.play(
            Create(minus_one_tick)
        )
        self.wait(2)
        self.play(Restore(line_2))
        self.wait(2)
        self.play(
            Write(dot_1),
            Write(dot_2)
        )
        self.wait(2)
        self.play(
            x.animate.set_value(-1),
            run_time=2
        )
        self.wait(4)

        self.play(
            x.animate.set_value(0),
            run_time=2
        )
        self.wait()
        self.play(Create(two_tick))
        self.wait(2)

        self.play(Write(d_line_1))
        self.play(Create(d_line_1_label))
        self.wait(2)


        self.play(
            ReplacementTransform(line_2.copy(),line_3),
            ReplacementTransform(line_2.copy(),line_4),
            FadeOut(line_2)
        )
        self.play(
            line_3.animate.set_color(REANLEA_PINK_LIGHTER),
            line_4.animate.set_color(REANLEA_SLATE_BLUE)
        )
        self.wait(2)

        self.play(
            line_3.animate.move_to(line_1_center+RIGHT),
            line_4.animate.move_to(line_1_center+3*RIGHT),
        )
        self.wait(2)
        self.play(
            Write(brace_line_3_4),
            Write(brace_line_3_4_label)
        )
        self.wait(2)
        self.play(Create(three_tick))
        self.play(Create(four_tick))
        self.play(Create(five_tick))
        self.play(Create(so_on_txt_symbol))
        self.wait(4)

        


        # manim -pqh anim1.py Scene1

        # manim -pql anim1.py Scene1

        # manim -sqk anim1.py Scene1


















###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix"