############################################# by GOBINDA CHANDRA ###################################################

                                    # VISIT : https://reanlea.com/ 
                                    # YouTube : https://www.youtube.com/Reanlea/ 
                                    # Twitter : https://twitter.com/Reanlea_ 
                                    # Facebook : https://www.facebook.com/reanlea.ed/ 
                                    # Telegram : https://t.me/reanlea/ 

#####################################################################################################################

from __future__ import annotations
from cProfile import label


import fractions
from imp import create_dynamic
from multiprocessing import context
from multiprocessing.dummy import Value
from numbers import Number
from tkinter import Y, Label, font
from imp import create_dynamic
from turtle import degrees
from manim import*
from math import*
from manim_fonts import*
import numpy as np
import random
from sklearn.datasets import make_blobs
from reanlea_colors import*
from func import*

config.background_color= REANLEA_BACKGROUND_COLOR


###################################################################################################################

class Scene1(Scene):
    def construct(self):

        zoom_exp = 1

        scene = VGroup()

        #object region

        dumy_line = Line(8*LEFT, 8*RIGHT, stroke_width=2.0).shift(DOWN)
        line= NumberLine(
            x_range=[-80, 80, 1],
            length=400,
            include_ticks=False,
        )
        line.move_to(line.n2p(-2)).shift(DOWN)
        scene.add(line)

        center=line.n2p(0)

        context=Square(200,fill_opacity=0.0, stroke_opacity=0.0).move_to(center)
        scene.add(context)

        zero_tick = VGroup(
            Line(0.3 * UP, 0.3 * DOWN, stroke_width=2.0, color=REANLEA_VIOLET_LIGHTER),
            MathTex("0"),
        )
        zero_tick[0].move_to(line.n2p(0))
        zero_tick[1].next_to(zero_tick[0], DOWN)

        one_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_GREEN),
            MathTex("1").scale(.5),
        )
        one_tick[0].move_to(line.n2p(1))
        one_tick[1].next_to(one_tick[0], DOWN)


        def set_zoom_exp(new_zoom_exp):
            nonlocal zoom_exp
            scale_factor=2**(zoom_exp - new_zoom_exp)
            zoom_exp=new_zoom_exp
            return scene.animate.scale(scale_factor)

        
        dot1= VGroup(
            Dot(radius=.15).move_to(line.n2p(1.25)).set_color(REANLEA_AQUA_GREEN),
            MathTex("x_1")
        )
        dot1[1].next_to(dot1[0],UP)

        dot2= VGroup(
            Dot(radius=.15).move_to(line.n2p(2.5)).set_color(REANLEA_AQUA),
            MathTex("x_2")
        )
        dot2[1].next_to(dot2[0],UP)
        #dot2= Dot(radius=.15).move_to(line.n2p(2.5)).set_color(REANLEA_CHARM)

        

        '''for i in np.arange(3,10):
            dots += Dot(radius=.15).move_to(line.n2p(1.25*i))
            dots.set_color_by_gradient(REANLEA_CHARM,REANLEA_AQUA, REANLEA_GREEN_JADE)'''

        '''for i in np.arange(3,10):
            lab=MathTex(f"x_{i}")
            dots += Dot(radius=.15).move_to(line.n2p(1.25*i))
            #dots.set_color_by_gradient(REANLEA_CHARM,REANLEA_AQUA, REANLEA_GREEN_JADE)

        dots.add(*[i.next_to(v) for i,v in zip(lab,dots)])

        scene.add(dot1,dot2,dots)'''

        dots = VGroup()

        for i in np.arange(3,10):
            dots += Dot(radius=.15).move_to(line.n2p(1.25*i))
            dots.set_color_by_gradient(REANLEA_AQUA,REANLEA_BLUE, REANLEA_YELLOW)

        labs=VGroup()

        for i in np.arange(3,10):
            labs +=MathTex(f"x_{i}")

        g=VGroup()
        g.add(*[i.next_to(v, direction=UP) for i,v in zip(labs,dots)])

        dot3=Tex("...").next_to(dots[-1], direction= 10*RIGHT +UP).scale(2).set_color(REANLEA_GREEN_LIGHTER)

        scene.add(dot1,dot2,dots,labs,dot3,one_tick)

        dash_arrow=DashedArrow(start=line.n2p(-.1), end=line.n2p(1.1),dash_length=2.0, max_tip_length_to_length_ratio=0.08, color=RED).shift(.5*UP)




        #text region
        with RegisterFont("Kalam") as fonts:
            text_1 = VGroup(*[Text(x, font=fonts[1], weight=BOLD) for x in (
                "Imagine you've a Dot!",
                "Somewhere along the real line."
            )]).arrange_submobjects(DOWN).to_edge(UP).shift(0.5*DOWN).scale(0.6).set_color(REANLEA_TXT_COL)


            text_2 = VGroup(*[Text(x, font=fonts[1], weight=BOLD) for x in (
                "Two Different Point Represents",
                "Two Dfferent Static Positions..."
            )]).arrange_submobjects(DOWN).to_edge(UP).shift(0.5*DOWN).scale(0.6).set_color(REANLEA_TXT_COL)


            text_3 = VGroup(*[Text(x, font=fonts[1], weight=BOLD) for x in (
                "I. The Point can't be ADDED to a similar point.",
                "II. nor it can be SCALAR MULTIPLIED"
            )]).arrange_submobjects(DOWN).to_edge(UP).shift(0.5*DOWN).scale(0.5).set_color(REANLEA_TXT_COL)

            text_4 = VGroup(*[Text(x, font=fonts[1], weight=BOLD) for x in (
                "Magnitude & Direction",
                "are both Added"
            )]).arrange_submobjects(DOWN).to_edge(UP).shift(0.5*DOWN).scale(0.5).set_color(REANLEA_TXT_COL)


        with RegisterFont("Montserrat") as fonts:
            text_5=Text("WHAT   ABOUT   D I S T A N C E ?", font=fonts[0]).scale(0.6)


            grp1=VGroup(text_1,text_2)

            grp2= VGroup(line,zero_tick,one_tick,dot1,dot2,dot3,dots,labs,dash_arrow,text_1,text_2,text_3,text_4)

            

            



        ####play region
        
        
        self.play(
            DrawBorderThenFill(dumy_line)
        )
        self.play(Create(text_1))
        self.add(line)
        self.play(
            Create(dot1[0])
        )
        self.play(
            Flash(
                dot1[0],
                color=RED, flash_radius=0.15+SMALL_BUFF, time_width=0.3
            )
        )

        self.play(FadeIn(zero_tick))
        self.wait()
        self.play(Create(one_tick))
        self.wait(2)
        self.play(Write(dot1[1]))
        self.wait(2)

        self.play(Transform(text_1,text_2))
        self.play(Create(dot2))
        self.wait(2)

        self.play(Create(dots))
        self.play(FadeOut(grp1))
        self.play(set_zoom_exp(2.5), run_time=3)
        self.play(Write(text_3))


        self.play(FadeOut(dumy_line))
        self.wait(2)
        self.play(FadeOut(text_3))

        self.play(set_zoom_exp(1), run_time=1.5)


        
        self.play(Create(dash_arrow))
        self.wait(3)
        self.play(Write(text_4))
        self.wait(3)

        self.play(
            *[FadeOut(mobj) for mobj in self.mobjects],
            run_time=2
        )
        self.play(Write(text_5))
        self.wait(1.75)
        self.play(FadeOut(text_5))
        self.wait(5)
        



         # manim -pqh anim.py Scene1



###################################################################################################################


class Scene2(Scene):
    def construct(self):

        zoom_exp = 1

        scene = VGroup()

        #object region

        dumy_line = Line(8*LEFT, 8*RIGHT, stroke_width=2.0).shift(DOWN)
        line= NumberLine(
            x_range=[-8, 8, 1],
            length=32,
            include_ticks=False,
        )
        line.move_to(line.n2p(-2)).shift(DOWN)
        scene.add(line)

        center=line.n2p(-2)
        p1=np.array((-0.75, -0.5, 0.0))
        p2=np.array((4.25, -0.5, 0.0))

        zero_tick = VGroup(
            Line(0.2 * UP, 0.2 * DOWN, stroke_width=2.0, color=REANLEA_VIOLET_LIGHTER),
            MathTex("0"),
        )
        zero_tick[0].move_to(line.n2p(0))
        zero_tick[1].next_to(zero_tick[0], DOWN)

        dot1= VGroup(
            Dot(radius=.15).move_to(line.n2p(1.75)).set_color(REANLEA_RED_LIGHTER).scale(0.6),
            MathTex("x").scale(0.4)
        )
        dot1[1].next_to(dot1[0],DOWN)
        dot2= VGroup(
            Dot(radius=.15).move_to(line.n2p(4)).set_color(REANLEA_YELLOW_GREEN).scale(0.6),
            MathTex("y").scale(0.4)
        )
        dot2[1].next_to(dot2[0],DOWN)


        grp=VGroup(dot1,dot2)

        d_line=DashedDoubleArrow(start=p1, end=p2, dash_length=2.0,stroke_width=2, max_tip_length_to_length_ratio=0.025, color=REANLEA_MAGENTA_LIGHTER)
        
        



        ####play region

        self.play(
            DrawBorderThenFill(dumy_line)
        )
        self.add(line)
        self.play(Create(zero_tick))
        self.play(Create(grp))
        self.wait(3)
        self.play(Create(d_line))
        self.wait(2)



        # manim -pqh anim.py Scene2