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

config.max_files_cached=200

config.background_color= REANLEA_BACKGROUND_COLOR


###################################################################################################################

# PRE-INTRO

class Scene1(Scene):
    def construct(self):

        # WATER-MARK
        with RegisterFont("Montserrat") as fonts:
            water_mark=Text("R E A N L E A ", font=fonts[0]).scale(0.3).to_edge(UP).shift(.5*DOWN + 5*LEFT).set_opacity(.15)            # to_edge(UP) == move_to(3.35*UP)
            water_mark.set_color_by_gradient(REANLEA_GREY)
        water_mark.save_state()

        # HEADING
        with RegisterFont("Cousine") as fonts:
            text_1 = Text("You MAY KNOW almost EVERYTHING if you want.", font=fonts[0]).set_color_by_gradient(REANLEA_GREY).scale(.4)
            text_2 = Text("But without investing ENOUGH TIME you CAN'T LEARN anything ...", font=fonts[0]).set_color_by_gradient(REANLEA_GREY).scale(.4)
            
        with RegisterFont("Courier Prime") as fonts:
            text_3 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "You MAY KNOW almost EVERYTHING if you want.",
                "But without investing ENOUGH TIME you CAN'T LEARN anything... "
            )]).arrange_submobjects(DOWN).to_edge(UP).shift(0.5*DOWN).scale(0.4).set_color(REANLEA_GREY)
            text_3.move_to(ORIGIN)
            
        grp=VGroup(text_1,text_2).arrange(DOWN)



        self.add(water_mark)
        self.play(
            AddTextWordByWord(text_3),
        )


        self.wait(3)

        self.play(
            FadeOut(text_3),
            run_time=1.5
        )
        self.wait()



        # manim -pqh anim.py Scene1


###################################################################################################################


class Scene2(Scene):
    def construct(self):

        zoom_exp = 1

        scene = VGroup()

        # WATER MARK 

        with RegisterFont("Montserrat") as fonts:
            water_mark=Text("R E A N L E A ", font=fonts[0]).scale(0.3).to_edge(UP).shift(.5*DOWN + 5*LEFT).set_opacity(.15)            # to_edge(UP) == move_to(3.35*UP)
            water_mark.set_color_by_gradient(REANLEA_GREY)
        water_mark.save_state()

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

            grp2= VGroup(line,zero_tick,one_tick,dot1,dot2,dot3,dots,labs,dash_arrow,text_4)

            

            



        ####play region

        self.add(water_mark)
        
        
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

        '''self.play(
            *[FadeOut(mobj) for mobj in self.mobjects],
            run_time=2
        )'''
        self.play(FadeOut(grp2))
        #self.add(water_mark)
        self.play(Write(text_5))
        self.wait(1.75)
        self.play(FadeOut(text_5))
        self.wait(5)
        


         # manim -pqh anim.py Scene2


###################################################################################################################


class Scene3(Scene):
    def construct(self):
         
        # WATER-MARK
        with RegisterFont("Montserrat") as fonts:
            water_mark=Text("R E A N L E A ", font=fonts[0]).scale(0.3).to_edge(UP).shift(.5*DOWN + 5*LEFT).set_opacity(.15)            # to_edge(UP) == move_to(3.35*UP)
            water_mark.set_color_by_gradient(REANLEA_GREY)
        water_mark.save_state()

        # HEADING
        with RegisterFont("Montserrat") as fonts:
            text_1=Text("D I S T A N C E ", font=fonts[0]).scale(2)#.to_edge(UP).shift(.5*DOWN)             # to_edge(UP) == move_to(3.35*UP)
            text_1.set_color_by_gradient(REANLEA_GREY_DARKER,REANLEA_TXT_COL_DARKER)

        text_1.save_state()

        s1=AnnularSector(inner_radius=2, outer_radius=2.75, angle=2*PI, color=REANLEA_GREY_DARKER).set_opacity(0.3).move_to(5.5*LEFT)
        s2=AnnularSector(inner_radius=.2, outer_radius=.4, angle=2*PI, color=REANLEA_GREY).set_opacity(0.3).move_to(UP + 5*RIGHT)
        s3=AnnularSector(inner_radius=1, outer_radius=1.5, color=REANLEA_SLATE_BLUE).set_opacity(0.6).move_to(3.5*DOWN + 6.5*RIGHT).rotate(PI/2)
        ann=VGroup(s1,s2,s3)

        # PLAY REGION
        self.add(s1,s2,s3, water_mark)
        self.play(
            Wiggle(s1),
            Wiggle(s2),
            Create(text_1),
        )
        self.wait()
        self.play(
            text_1.animate.scale(0.3).to_edge(UP).shift(0.5*DOWN).set_color_by_gradient(REANLEA_GREY_DARKER,REANLEA_TXT_COL_DARKER),
            FadeOut(ann)
        )
        self.wait(3)



        # manim -pqk anim.py Scene3


###################################################################################################################


class Scene4(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()

        # WATER MARK 

        with RegisterFont("Montserrat") as fonts:
            water_mark=Text("R E A N L E A ", font=fonts[0]).scale(0.3).to_edge(UP).shift(.5*DOWN + 5*LEFT).set_opacity(.15)            # to_edge(UP) == move_to(3.35*UP)
            water_mark.set_color_by_gradient(REANLEA_GREY)
        water_mark.save_state()

        

        scene = VGroup()

        # Tracker 

        zoom_exp = 1

        # object region

        dumy_line = Line(8*LEFT, 8*RIGHT, stroke_width=2.0).shift(DOWN)
        line= NumberLine(
            x_range=[-8, 8, 1],
            length=32,
            include_ticks=False,
            stroke_width=2.0
        )
        line.move_to(line.n2p(-2)).shift(DOWN)
        scene.add(line)

        center=line.n2p(-2)


        dot1= VGroup(
            Dot(radius=.25).move_to(line.n2p(1.75)).scale(0.6).set_color(REANLEA_VIOLET_LIGHTER).set_sheen(-0.4,DOWN),
            MathTex("x").scale(0.6)
        )
        dot1[1].next_to(dot1[0],DOWN)
        dot2= VGroup(
            Dot(radius=.25).move_to(line.n2p(4)).scale(0.6).set_color(REANLEA_GREEN).set_sheen(-0.6,DOWN),
            MathTex("y").scale(0.6)
        )
        dot2[1].next_to(dot2[0],DOWN)

        


        grp=VGroup(dot1,dot2)

        '''p1=np.array((-0.75, -0.5, 0.0))
        p2=np.array((4.25, -0.5, 0.0))'''

        p1=np.array((-4,-1,0))
        p2=np.array((-4,0,0))
        p3=np.array((-4,1,0))

        p4=np.array((-.5,-1,0))
        p5=np.array((-.5,-.5,0))
        p6=np.array((-.5,0,0))

        p7=np.array((4,-1,0))
        p8=np.array((4,-.5,0))
        p9=np.array((4,1,0))

        


        zero_tick = VGroup(
            Line(0.2 * UP, 0.2 * DOWN, stroke_width=4.0, color=REANLEA_VIOLET_LIGHTER),
            MathTex("0"),
        )
        zero_tick[0].move_to(line.n2p(0))
        zero_tick[1].next_to(zero_tick[0], DOWN)

        


        ### dashed line 

        # horizontal dashed line
        d_line=DashedDoubleArrow(
            start=p5, end=p8, dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA)

        d_line_label= MathTex("d(x,y)").next_to(d_line, .1*UP).scale(0.45).set_color(REANLEA_GREY)

        d_line1=DashedDoubleArrow(
            start=p2, end=p6, dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA)

        d_line1_label= MathTex("d(0,x)").next_to(d_line1, .1*UP).scale(0.45).set_color(REANLEA_VIOLET_LIGHTER)

        d_line2=DashedDoubleArrow(
            start=p3, end=p9, dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.01, buff=10
        ).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA)

        d_line2_label= MathTex("d(0,y)").next_to(d_line2, .1*UP).scale(0.45).set_color(REANLEA_GREEN_LIGHTER)

        d_line_label_grp=VGroup(d_line_label,d_line1_label,d_line2_label)


        # Vertical dashed line
        v_line1=DashedLine(
            start=p4, end=p5, stroke_width=1
        ).set_color(RED_D)

        v_line2=DashedLine(
            start=p7, end=p8, stroke_width=1
        ).set_color(RED_D)

        v_line3=DashedLine(
            start=p1, end=p2, stroke_width=1
        ).set_color(RED_D)

        v_line4=DashedLine(
            start=p5, end=p6, stroke_width=1
        ).set_color(RED_D)

        v_line5=DashedLine(
            start=p8, end=p9, stroke_width=1
        ).set_color(RED_D)

        v_line6=DashedLine(
            start=p2, end=p3, stroke_width=1
        ).set_color(RED_D)

        


        grp2=VGroup(v_line1,v_line2)
        grp3=VGroup(v_line3,v_line6,v_line4,v_line5)
        grp4=VGroup(d_line1,d_line2)
        grp5=VGroup(d_line1_label,d_line2_label)

        
        



        #### text region


        with RegisterFont("Montserrat") as fonts:
            text_1=Text("D I S T A N C E ", font=fonts[0]).scale(.6).to_edge(UP).shift(0.5*DOWN)
            text_1.set_color_by_gradient(REANLEA_GREY_DARKER,REANLEA_TXT_COL_DARKER)

        text_1.save_state()



        #glowing circle 

        #1

        glow_circ_grp_1_1 = VGroup(dot1,dot2)
        glow_circ_grp_1_2 = VGroup(v_line3,v_line4,v_line5,v_line6,d_line1,d_line2,d_line1_label,d_line2_label,zero_tick)
        glow_circ_grp_1_2.save_state()

        glowing_circles_1=VGroup()                   # VGroup( doesn't have append method) 

        for dot in glow_circ_grp_1_1:
            glowing_circle=get_glowing_surround_circle(dot[0], color=REANLEA_YELLOW)
            glowing_circle.save_state()
            glowing_circles_1 += glowing_circle
        
        glowing_circles_1.save_state()

        #2

        glow_circ_grp_2_1 = VGroup(dot1,zero_tick)
        glow_circ_grp_2_2 = VGroup(v_line1,v_line2,v_line5,v_line6,d_line,d_line2,d_line_label,d_line2_label,dot2)
        glow_circ_grp_2_2.save_state()


        glowing_circles_2=VGroup()                   # VGroup( doesn't have append method) 

        for dot in glow_circ_grp_2_1:
            glowing_circle=get_glowing_surround_circle(dot[0], color=REANLEA_YELLOW)
            glowing_circle.save_state()
            glowing_circles_2 += glowing_circle
        
        glowing_circles_2.save_state()

        #3

        glow_circ_grp_3_1 = VGroup(dot2,zero_tick)
        glow_circ_grp_3_2 = VGroup(v_line1,v_line4,d_line1,d_line,d_line1_label,d_line_label,dot1)
        glow_circ_grp_3_2.save_state()


        glowing_circles_3=VGroup()                   # VGroup( doesn't have append method) 

        for dot in glow_circ_grp_3_1:
            glowing_circle=get_glowing_surround_circle(dot[0], color=REANLEA_YELLOW)
            glowing_circle.save_state()
            glowing_circles_3 += glowing_circle
        
        glowing_circles_3.save_state()

        # VALUE UPDATER
        '''value=DecimalNumber().set_color_by_gradient(REANLEA_GREEN_LIGHTER).set_sheen(-0.4,DR).scale(1.35)

        value.add_updater(
            lambda x : x.set_value((dot2[0].get_center()[0]-dot1[0].get_center()[0])/2.25)
        )

        v_line2.add_updater(
            lambda x : x.move_to(
                dot2[0].get_center()+ 0.25*UP
            )
        )

        dot2[0].add_updater(lambda z : z.set_x(x.get_value()))

        d_line.add_updater(
            lambda z: z.become(
                DashedDoubleArrow(
                    start=dot1[0].get_center()+2*UP, end=dot2[0].get_center()+2*UP, dash_length=2.0,stroke_width=2, 
                    max_tip_length_to_length_ratio=0.015, buff=10
                ).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA)
            )
        )
        '''




        # equation 

        eq1 = MathTex("d(0,x)", "+", "d(x,y)", "=", "d(0,y)").move_to(3*DOWN)
        #eq2 = MathTex("\Rightarrow", "d(x,y)", "=", "d(0,y)", "-", "d(0,x)").move_to(3*DOWN)
        eq2 = MathTex("d(x,y)", "=", "d(0,y)", "-", "d(0,x)").move_to(3*DOWN)
        r_arr= MathTex("\Rightarrow").next_to(eq2, LEFT)
        eq3 = MathTex("d(x,y)", "=", "y", "-", "x").move_to(2.5*DOWN).set_color(REANLEA_BLUE_LAVENDER)
        eq4 = MathTex("d(x,y)","=", "|y-x|").set_color(REANLEA_GREEN_LIGHTER).set_sheen(-0.4,DR).move_to(UP).scale(1.35)
        #eq4_1= MathTex("d(x,y)", "=").set_color(REANLEA_GREEN_LIGHTER).set_sheen(-0.1,DR).move_to(UP).scale(1.35)
        eq5=MathTex(">", "0").set_color(REANLEA_GREEN_LIGHTER).set_sheen(-0.4,DR).move_to(UP).scale(1.35)
        eq5_grp=VGroup(eq4,eq5).arrange(RIGHT, buff=0.3).move_to(UP)
        
        #eq_grp=VGroup(d_line1_label.copy(),d_line2_label.copy(),d_line_label.copy())

        # decriptive text 

        # DESIGN

        


        

        #### play region

        self.add(water_mark)
        self.play(Write(text_1))
        
        self.wait()

        self.play(
            DrawBorderThenFill(dumy_line)
        )
        self.add(line)
        self.play(Uncreate(dumy_line))
        self.play(Create(zero_tick))
        self.play(Create(grp))
        self.wait()
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(DOWN + 1.5*RIGHT),
            text_1.animate.scale(0.5).move_to(0.425*UP + 1.5 *RIGHT),
            water_mark.animate.scale(0.5).move_to(0.465*UP + LEFT),
        )
        
        self.wait()
        self.play(Create(grp2))
        self.wait(2)
        self.play(Write(d_line))
        self.wait(2)
        self.play(Write(d_line_label))
        self.wait(2)
        self.play(Restore(self.camera.frame), Restore(text_1), Restore(water_mark))

        self.wait(2)
        self.play(Write(grp3), run_time=2.5)
        self.wait()
        self.play(
            Write(grp4),
            Write(d_line1_label),
            Write(d_line2_label),
            run_time=2
        )
        self.wait(2)


        self.play(
            FadeIn(*glowing_circles_1),
            glow_circ_grp_1_2.animate.set_opacity(0.4)
        )
        self.wait()
        self.play(ReplacementTransform(d_line_label.copy(), eq1[2]))
        self.wait()
        self.play(
            FadeOut(*glowing_circles_1),
            Restore(glow_circ_grp_1_2)
        )


        self.play(
            FadeIn(*glowing_circles_2),
            glow_circ_grp_2_2.animate.set_opacity(0.4)
        )
        self.play(ReplacementTransform(d_line1_label.copy(), eq1[0]))
        self.wait()
        self.play(
            FadeOut(*glowing_circles_2),
            Restore(glow_circ_grp_2_2)
        )
       

        self.play(
            FadeIn(*glowing_circles_3),
            glow_circ_grp_3_2.animate.set_opacity(0.4)
        )
        self.play(ReplacementTransform(d_line2_label.copy(), eq1[4]))
        self.wait()
        self.play(
            FadeOut(*glowing_circles_3),
            Restore(glow_circ_grp_3_2)
        )

        self.wait()

        
        # equation animation segment

        eq1_sub_grp=VGroup(eq1[0], eq1[2], eq1[4])

        self.play(TransformMatchingShapes(eq1_sub_grp, eq1))
        #self.play(FadeOut(eq1_sub_grp))
        self.wait()
        self.play(
            eq1.animate.scale(0.75).move_to(.25*LEFT + 2.15*DOWN).set_fill(color=REANLEA_GREY_DARKER, opacity=0.75),
            ReplacementTransform(eq1.copy(),eq2),
            FadeIn(r_arr),
        )
        self.wait()
        self.play(
            FadeOut(eq1),
            FadeOut(r_arr),
            #ReplacementTransform(eq2,eq3)
            eq2.animate.move_to(2.5*DOWN).scale(1.1).set_fill(color=REANLEA_BLUE_LAVENDER)
        )
        self.play(
            Circumscribe(eq2, color=REANLEA_CHARM, run_time=1.5)
        )
        self.play(
            ReplacementTransform(eq2, eq3)
        )
        self.wait(2)

        self.play(
            FadeOut(grp3),
            FadeOut(grp4),
            FadeOut(grp5),
            run_time=3
        )
        self.wait()

        scene1=VGroup(line, zero_tick, grp, grp2, d_line, d_line_label)


        self.play(
            Transform(eq3,eq4),
            scene1.animate.move_to(1.5*DOWN+4*LEFT),
            TransformMatchingShapes(d_line_label,eq4)
            #eq4.animate.scale(0.7).move_to(1*UP).set_color(REANLEA_GREEN_LIGHTER).set_sheen(-0.1,DR).scale(1.5)
        )
        
        self.wait(3)
        self.play(Write(eq5), run_time=2)
        self.wait()

        scene2=VGroup(line,zero_tick,grp,grp2,d_line)
        scene2_1=VGroup(eq5, eq4, eq3)

        sur_text1=get_surround_bezier(text_1)

        self.play(
            scene2.animate.set_opacity(.25),
            Create(sur_text1),
            scene2_1.animate.set_opacity(0.1)
        )
        self.wait(3)
        
        with RegisterFont("Caveat") as fonts:
            text_2=Text("Let's have a little deeper look ... ", font=fonts[0]).scale(.55)
            text_2.set_color_by_gradient(REANLEA_GREY).set_opacity(0.6).shift(3*RIGHT)


        indicBez=ArrowCubicBezierUp().scale(1.4)

           
        grp6=VGroup(indicBez,text_2)
        

        self.play(
            Create(indicBez),
            lag_ratio=0.2
        )
        self.play(
            AddTextLetterByLetter(text_2)
        )
        self.wait(3)

        self.play(
            grp6.animate.rotate(30*DEGREES).scale(0.65).move_to(2*UP+3*RIGHT)
        )
        self.wait(3)

        with RegisterFont("Cousine") as fonts:
            text_3 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Q1. What's the distance between a point from itself ?",
                " and what can you say if the distance between two points be zero ?"
            )]).arrange_submobjects(DOWN).to_edge(UP).shift(0.5*DOWN).scale(0.4).set_color(REANLEA_GREY)
            text_3[1].shift(1.4*RIGHT)
            text_3.move_to(ORIGIN)


        self.play(
            AddTextWordByWord(text_3)
        )
        self.wait(3)



        #### last scene of Scene2

        line_last =NumberLine(
            x_range=[-3,3],
            include_ticks=False,
            include_tip=False
        ).set_color(REANLEA_PINK_DARKER).set_opacity(0.6)

        dot1_last=Dot(radius=0.25, color=REANLEA_GREEN).move_to(3*RIGHT).set_sheen(-0.6,DOWN)
        dot2_last=Dot(radius=0.25, color=REANLEA_VIOLET_LIGHTER).move_to(3*LEFT).set_sheen(-0.4,DOWN)

        dot1_last_lbl=MathTex("y").scale(0.6).next_to(dot1_last, DOWN)
        dot2_last_lbl=MathTex("x").scale(0.6).next_to(dot2_last, DOWN)


        p1_last= dot1.get_center()+UP
        p2_last= dot2.get_center()+UP

        d_line_last=DashedDoubleArrow(
            start=dot2_last.get_center()+UP, end=dot1_last.get_center()+UP, dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA)

        v_line1_last=DashedLine(
            start=dot1_last.get_center(), end=dot1_last.get_center()+UP, stroke_width=1
        ).set_color(RED_D)

        v_line2_last=DashedLine(
            start=dot2_last.get_center(), end=dot2_last.get_center()+UP, stroke_width=1
        ).set_color(RED_D)

        grp_last=VGroup(line_last,dot1_last,dot2_last, d_line_last, v_line1_last, v_line2_last, dot1_last_lbl,dot2_last_lbl)


        grp7=VGroup(grp6,text_1,text_3, sur_text1)

        self.play(
            ReplacementTransform(scene2,grp_last, run_time=1.5),
            scene2_1.animate.set_opacity(0),
            FadeOut(grp7)
        )
        self.wait(3)



        # manim -pqh anim.py Scene4

        # manim -sqk anim.py Scene4


###################################################################################################################


class Scene5(Scene):
    def construct(self):

        # WATER-MARK
        with RegisterFont("Montserrat") as fonts:
            water_mark=Text("R E A N L E A ", font=fonts[0]).scale(0.3).to_edge(UP).shift(.5*DOWN + 5*LEFT).set_opacity(.15)            # to_edge(UP) == move_to(3.35*UP)
            water_mark.set_color_by_gradient(REANLEA_GREY)
        water_mark.save_state()

        # Tracker 
        x=ValueTracker(-3)
        theta_tracker=ValueTracker(0)
        scale_tracker=ValueTracker(1)

        # MOBJECTS
        line=NumberLine(
            x_range=[-3,3],
            include_ticks=False,
            include_tip=False
        ).set_color(REANLEA_YELLOW_DARKER).set_opacity(0.6)
        
        dot1=Dot(radius=0.25, color=REANLEA_GREEN).move_to(3*RIGHT).set_sheen(-0.6,DOWN)
        dot2=Dot(radius=0.25, color=REANLEA_VIOLET_LIGHTER).move_to(3*LEFT).set_sheen(-0.4,DOWN)
        dot3=dot2.copy().set_opacity(0.4)
        


        # DOT LABELS

        dot1_lbl=MathTex("y").scale(0.6).next_to(dot1, DOWN)
        dot3_lbl=MathTex("x").scale(0.6).next_to(dot3, DOWN)

        # POINTS
        p1= dot1.get_center()+UP
        p2= dot2.get_center()+UP
        p3=dot3.get_center()



        #dashed lines
        d_line=DashedDoubleArrow(
            start=p2, end=p1, dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA)


        v_line1=DashedLine(
            start=dot1.get_center(), end=dot1.get_center()+UP, stroke_width=1
        ).set_color(RED_D)

        v_line2=DashedLine(
            start=dot2.get_center(), end=dot2.get_center()+UP, stroke_width=1
        ).set_color(RED_D)


        # GROUPS
        grp1=VGroup(line,dot1,dot2, d_line, v_line1, v_line2)
        grp1_2=VGroup(v_line1, v_line2)
        


        #value updater
        value=DecimalNumber().set_color_by_gradient(REANLEA_MAGENTA).set_sheen(-0.1,LEFT)

        value.add_updater(
            lambda x : x.set_value((1-(dot2.get_center()[0]/3))/2)
        )
        

        v_line2.add_updater(
            lambda x : x.move_to(
                dot2.get_center()+ 0.5*UP
            )
        )

        

        dot2.add_updater(lambda z : z.set_x(x.get_value()))

        d_line.add_updater(
            lambda z: z.become(
                DashedDoubleArrow(
                    start=dot2.get_center()+UP, end=dot1.get_center()+UP, dash_length=2.0,stroke_width=2, 
                    max_tip_length_to_length_ratio=0.015, buff=10
                ).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA)
            )
        )

        # Arrow Zone
        grp3=VGroup()
        p1 = ParametricFunction(
            lambda t: bezier_updated(t,
                np.array([
                    [1.91,.29,0],
                    [.2,1.1, 0],
                    [1.9, 2.53, 0],
                ]),
                np.array([1,1,1])),
            t_range=[0, 1],
            color=REANLEA_CHARM,
        )
        
        p1.move_to(ORIGIN).rotate(50*DEGREES)

        p=CurvesAsSubmobjects(p1)
        p.set_color_by_gradient(REANLEA_YELLOW_CREAM,REANLEA_CHARM).set_stroke(width=3)

        grp3 += p


        ar= Arrow(max_stroke_width_to_length_ratio=0,max_tip_length_to_length_ratio=0.15).move_to(p1.get_end()+.55*DOWN).rotate(PI/2)
        ar.set_color(REANLEA_CHARM)

        
        grp3 += ar
        grp3.move_to(1.35*DOWN+RIGHT)



        # TEXT

        tex=MathTex("d(x,y)=")#.set_color(REANLEA_GREEN_LIGHTER).set_sheen(-0.4,DR)
        grp2=VGroup(tex,value).arrange(RIGHT, buff=0.3).move_to(2*UP).set_color_by_gradient(REANLEA_GREEN,REANLEA_MAGENTA)

        with RegisterFont("Caveat") as fonts:
            text_1=Text("Continuous Motion ... ", font=fonts[0]).scale(.55)
            text_1.set_color_by_gradient(REANLEA_YELLOW_CREAM,REANLEA_AQUA).set_opacity(0.6).shift(3*RIGHT)
        
        text_1.next_to(grp3.get_boundary_point(DOWN),4*RIGHT+UP)
        

        text_2_1=MathTex(r"\rightarrow", "0")
        text_2_2=Tex("as")
        text_2_3=MathTex("x",r"\rightarrow", "y")

        text_2=VGroup(text_2_1,text_2_2,text_2_3).arrange(RIGHT, buff=0.4).next_to(value, 1.5*RIGHT)
        text_2.set_color_by_gradient(REANLEA_MAGENTA,REANLEA_AQUA)

        with RegisterFont("Cousine") as fonts:
            text_3=Text("What if we add one more point ... ", font=fonts[0]).scale(.55)
            text_3.set_color_by_gradient(REANLEA_GREY).set_opacity(0.6).shift(3*RIGHT)
        text_3.move_to(2*DOWN)


        # play region

        self.add(water_mark, grp1, dot1_lbl,dot3_lbl)
        self.wait()
        self.play(Write(grp2))
        self.add(dot3)
        self.wait()
        '''self.play(
            MoveAlongPath(dot2, line1, rate_func=rate_functions.ease_in_out_sine),
            run_time=3
        )'''
        self.play(
            x.animate.set_value(dot1.get_center()[0]),
            run_time=3
        )
        self.wait(2)
        
        self.play(
            x.animate.set_value(dot3.get_center()[0] + dot1.get_center()[0]/4),
        )
        self.play(
            x.animate.set_value(dot3.get_center()[0]/2 + dot1.get_center()[0]),
        )
        self.play(
            x.animate.set_value(dot3.get_center()[0] + dot1.get_center()[0]/3),
            Create(grp3)
        )
        self.play(
            x.animate.set_value(dot3.get_center()[0]/4 + dot1.get_center()[0]),
           AddTextLetterByLetter(text_1)
        )
        self.play(
            x.animate.set_value(dot3.get_center()[0] + dot1.get_center()[0]),
        )
        self.play(
            x.animate.set_value(dot3.get_center()[0]/6 + dot1.get_center()[0]),
        )
        self.play(
            x.animate.set_value(dot3.get_center()[0]),
            
        )
        self.play(
            Write(text_2),
            x.animate.set_value(dot3.get_center()[0]/10 + dot1.get_center()[0]),
            run_time=3
        )
        self.wait()
        
        self.play(
            x.animate.set_value(dot1.get_center()[0]),
            run_time=3
        )


        self.wait(2)

        self.play(
            Uncreate(grp3),
            Uncreate(text_1)
        )
        self.play(
            AddTextLetterByLetter(text_3)
        )
        self.wait()

        self.play(FadeOut(d_line))
        self.play(
            FadeOut(grp2),
            FadeOut(text_2),
            FadeOut(text_3),
            FadeOut(grp1_2),
            x.animate.set_value(dot3.get_center()[0]),
            line.animate.set_opacity(1).set_stroke(width=10),
        )
        self.play(FadeOut(dot3))
        
        
        self.wait(3)





        # manim -pqh anim.py Scene5

        # manim -pql anim.py Scene5

        # manim -sqk anim.py Scene5


###################################################################################################################


class Scene6(Scene):
    def construct(self):

        # WATER-MARK
        with RegisterFont("Montserrat") as fonts:
            water_mark=Text("R E A N L E A ", font=fonts[0]).scale(0.3).to_edge(UP).shift(.5*DOWN + 5*LEFT).set_opacity(.15)            # to_edge(UP) == move_to(3.35*UP)
            water_mark.set_color_by_gradient(REANLEA_GREY)
        water_mark.save_state()


        # Tracker 
        theta_tracker=ValueTracker(0)
        scale_tracker=ValueTracker(1)
        


        # MOBJECTS
        dot1=Dot(radius=0.25, color=REANLEA_GREEN).move_to(3*RIGHT).set_sheen(-0.6,DOWN)
        dot2=Dot(radius=0.25, color=REANLEA_VIOLET_LIGHTER).move_to(3*LEFT).set_sheen(-0.4,DOWN)
        dot3=Dot(radius=0.125, color=REANLEA_YELLOW).move_to(LEFT+2.5*UP).set_sheen(-0.6,DOWN)


        line1=Line(start=dot2.get_center(), end=dot1.get_center()).set_color(REANLEA_YELLOW_DARKER).set_stroke(width=10).set_z_index(-2)
        line2=Line(start=dot2.get_center(), end=dot3.get_center()).set_color(REANLEA_GREEN_DARKER).set_stroke(width=5).set_z_index(-3)
        line3=Line(start=dot3.get_center(), end=dot1.get_center()).set_color(REANLEA_VIOLET_DARKER).set_stroke(width=5).set_z_index(-1)
        

        line1_p1_tracker=ValueTracker(line1.get_angle())

        line1_p1=Line(start=dot2.get_center(), end=np.array((dot3.get_center()[0],0,0)))
        line1_p1.set_color(REANLEA_YELLOW).set_opacity(0.65).set_stroke(width=5).set_z_index(-1)
        line1_p1_ref=line1_p1.copy()
        line1_p1.rotate(
            line1_p1_tracker.get_value(), about_point=dot2.get_center()
        )
        line1_p1.save_state()

        


        projec_line=DashedLine(start=dot3.get_center(), end=np.array((dot3.get_center()[0],0,0)), stroke_width=1).set_color(REANLEA_AQUA_GREEN).set_z_index(-2)
        
        angle_12=Angle(line1,line2, radius=.5, other_angle=False).set_color(REANLEA_GREEN).set_z_index(-3)
        angle_13=Angle(line3,line1, radius=.65, quadrant=(-1,-1),other_angle=False).set_color(REANLEA_VIOLET).set_z_index(-3)

        circ=DashedVMobject(Circle(radius=line1_p1.get_length()), dashed_ratio=0.5, num_dashes=100).move_to(dot2.get_center()).set_stroke(width=0.65)
        circ.set_color_by_gradient(REANLEA_WHITE,REANLEA_WARM_BLUE,REANLEA_YELLOW_CREAM)



        line2_ref=line2.copy()
        line2=always_redraw(lambda : Line(start=dot2.get_center(), end=dot3.get_center()).set_color(REANLEA_GREEN_DARKER).set_stroke(width=5))

        line2.add_updater(
            lambda z : z.become(line2_ref.copy().rotate(
                theta_tracker.get_value()*DEGREES, about_point=dot2.get_center()
            ).scale(
                scale_tracker.get_value(), about_point=dot2.get_center()
            ))
        )

        line3=always_redraw(lambda : Line(start=dot3.get_center(), end=dot1.get_center()).set_color(REANLEA_VIOLET_DARKER).set_stroke(width=5))


        # DOT & Line LABELS
        dot1_lbl=MathTex("y").scale(0.6).next_to(dot1, DOWN)
        dot2_lbl=MathTex("x").scale(0.6).next_to(dot2, DOWN)
        dot1_lbl2=MathTex("y").scale(0.6).next_to(dot1, RIGHT)
        dot2_lbl2=MathTex("x").scale(0.6).next_to(dot2, LEFT)
        dot3_lbl=MathTex("z").scale(0.6).next_to(dot3, UP)

        line1_lbl=MathTex("Z").scale(0.6).next_to(line1, DOWN).set_color(REANLEA_YELLOW)
        line2_lbl=MathTex("Y").scale(0.6).next_to(line2, aligned_edge=RIGHT, direction=LEFT, buff=0.4).rotate(line2.get_angle()).set_color(REANLEA_GREEN)
        line3_lbl=MathTex("X").scale(0.6).next_to(line3, aligned_edge=LEFT, direction=RIGHT, buff= 0.6).rotate(line3.get_angle()).set_color(REANLEA_VIOLET_LIGHTER)

        brace_line2=Brace(Line(start=dot2.get_center(), end=np.array((dot3.get_center()[0],0,0)))).set_color(REANLEA_GREEN).set_opacity(0.8).set_z_index(-1)
        brace_line3=Brace(Line(start=np.array((dot3.get_center()[0],0,0)), end=dot1.get_center())).set_color(REANLEA_VIOLET).set_opacity(0.8).set_z_index(-1)
        
        brace_line2_lbl=MathTex(r"Y.cos\theta_{1}").next_to(brace_line2, .4*DOWN).scale(.5).set_color(REANLEA_GREEN)
        brace_line3_lbl=MathTex(r"X.cos\theta_{2}").next_to(brace_line3, .4*DOWN).scale(.5).set_color(REANLEA_VIOLET_LIGHTER).set_sheen(0.2,DR)
        
        angle_12_lbl=MathTex(r"\theta_{1}").move_to(
            Angle(
                line1, line2, radius=0.7, other_angle=False
            ).point_from_proportion(0.5)
        ).scale(0.5).set_color(REANLEA_AQUA_GREEN)

        angle_13_lbl=MathTex(r"\theta_{2}").move_to(
            Angle(
                line3, line1, radius=0.85, quadrant=(-1,-1),other_angle=False
            ).point_from_proportion(0.5)
        ).scale(0.5).set_color(REANLEA_VIOLET_LIGHTER)


        # GROUP REGION
        grp=VGroup(line1,dot1,dot2)
        grp2=VGroup(line1,line2,line3, dot1,dot2,dot3)
        dot_lbl_grp=VGroup(dot1_lbl2,dot2_lbl2,dot3_lbl)
        grp_line23=VGroup(line2,line3).set_z_index(-1)

        line_lbl=VGroup(line1_lbl,line2_lbl,line3_lbl)
        brace_lbl=VGroup(brace_line2,brace_line3,brace_line2_lbl,brace_line3_lbl)

        angle_grp=VGroup(angle_12, angle_13)
        angle_lbl_grp=VGroup(angle_12_lbl,angle_13_lbl)

        z_lbl=VGroup(line3_lbl,brace_line2_lbl,brace_line3_lbl)



        # TEXT & EQUN REGION 

        eq1 = MathTex("Z", "=", r"Y.cos\theta_{1}", "+",r"X.cos\theta_{2}").move_to(2.5*DOWN).scale(.7).set_color(REANLEA_TXT_COL)
        eq2 = MathTex("\leq", "Y", "+", "X").scale(.7).set_color_by_gradient(REANLEA_GREY).next_to(eq1)
        eq3= MathTex("-1","\leq",r"cos\theta","\leq","1").scale(0.85).set_color_by_gradient(REANLEA_GREEN_AUQA).move_to(3*UP+4*RIGHT)
        eq4= MathTex("d(x,y)","\leq",r"d(x,z)","+","d(z,y)").scale(0.7).set_color_by_gradient(REANLEA_BLUE_LAVENDER).move_to(3*DOWN+ 0.75*RIGHT)
        
        eq_grp1=VGroup(eq1,eq2)

        sr_rect=SurroundingRectangle(eq4, color=REANLEA_WARM_BLUE, buff=0.25, corner_radius=0.25)


        # MORE GROUPS

        eq_grp1=VGroup(eq1,eq2)

        uncreate_grp=VGroup(eq1,eq2,eq3,eq4,sr_rect,brace_lbl,dot_lbl_grp,line_lbl,circ,projec_line,angle_grp,angle_lbl_grp)


        # ADD UPDATER REGION

        


        # PLAY REGION
        self.add(water_mark)

        self.add(grp,dot1_lbl,dot2_lbl)
        self.wait(2)

        self.play(
            dot1.animate.scale(.5),
            dot2.animate.scale(.5),
            line1.animate.set_stroke(width=5),
            FadeOut(dot1_lbl),
            FadeOut(dot2_lbl),
            FadeIn(dot3)
        )
        self.wait()
        self.play(
            Create(grp_line23),
            run_time=1.5,
            buff=10
        )
        self.wait()
        

        self.play(Write(line_lbl))
        self.wait()
        self.play(Write(angle_grp))
        self.play(Write(angle_lbl_grp))
        self.wait(2)
        self.play(Create(projec_line))
        self.wait()
        
        self.play(
            line1_lbl.animate.shift(.5*DOWN),
            Write(brace_line2),
            Write(brace_line3),
            buff=1,
            run_time=3
        )
        self.play(Write(brace_line2_lbl),
            Write(brace_line3_lbl),
            run_time=2
        )
        self.wait()
        self.play(
            TransformMatchingShapes(z_lbl.copy(),eq1)
        )
        self.wait()
        self.play(Write(eq3))
        self.play(Write(eq2))
        self.wait()

        self.play(Write(line1_p1))
        self.wait()
        self.play(Write(circ))
        self.wait()

        line1_p1.add_updater(
            lambda x: x.become(line1_p1_ref.copy()).rotate(
                line1_p1_tracker.get_value() , about_point=dot2.get_center()
            )
        )


        self.play(
            #line1_p1.animate.rotate(line2.get_angle(), about_point=dot2.get_center()),
            #Rotate(line1_p1, angle=line2.get_angle(), about_point=line1_p1.get_start())
            line1_p1_tracker.animate.set_value(line2.get_angle())
        )
        self.wait(3)
        self.play(line1_p1_tracker.animate.set_value(line1.get_angle()))
        self.wait()
        self.play(Uncreate(line1_p1))
        self.wait(2)
        self.play(Write(dot_lbl_grp))
        self.wait()
        self.play(
            eq_grp1.animate.shift(.35*UP).set_opacity(0.65),
            Write(eq4)
        )
        self.wait()
        self.play(
            Create(sr_rect)
        )
        self.wait(3)
        self.play(
            Uncreate(uncreate_grp)
        )
        self.wait(2)


        # length tracker updaters


        dot3.add_updater(
            lambda x : x.move_to(line2.get_end())
        )

        self.play(
            theta_tracker.animate.set_value(30),
            scale_tracker.animate.set_value(.75),
        )

        self.wait(2)

        self.play(
            theta_tracker.animate.set_value(130),
            scale_tracker.animate.set_value(.85),
        )

        self.wait(2)

        self.play(
            theta_tracker.animate.set_value(270),
            scale_tracker.animate.set_value(1.25)
        )

        self.wait(2)

        self.play(
            theta_tracker.animate.set_value(0),
            scale_tracker.animate.set_value(0)
        )

        self.wait(2)

        self.play(
            theta_tracker.animate.set_value(-30),
            scale_tracker.animate.set_value(1)
        )

        self.wait(4)

        
       

        



        
        # manim -pqh anim.py Scene6

        # manim -pql anim.py Scene6

        # manim -sqk anim.py Scene6