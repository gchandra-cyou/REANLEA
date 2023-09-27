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


class Trailer(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        self.wait()


        # MAIN SCENE

        '''with RegisterFont("Homemade Apple") as fonts:
            txt_cs_0 = Text("EUCLIDEAN SPACE" , font=fonts[0])
            

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
            txt_cs_grp_0.animate.shift(2.5*UP).scale(.5)
        )'''

        rect=Rectangle(height=4.5, width=8).scale(.8).shift(.75*UP+3*RIGHT).set_stroke(opacity=1, color=[REANLEA_VIOLET,REANLEA_AQUA])

        self.play(
            Create(rect)
        )

        
        transition_points = [
            # use a list if we want multiple lines
            ["Distance"],
            ["Space"],
            ["Cartesian Product"],
            ["Dimension"],
            ["Pythagoras Theorem"]
        ]

        transition_points_ref = [
            # use a list if we want multiple lines
            ["Distance"],
            ["Space"],
            ["Cartesian" ,"Product"],
            ["Dimension"],
            ["Pythagoras","Theorem"]
        ]

        ch_points = [
            # use a list if we want multiple lines
            ["static point and vector","geometry of distance","generalization to metric"],
            ["1+1= ? & how -(-1)=1","appliation on vectors","properties of vector space"],
            ["define cartesian product","new space construction","relation wih vector addition & coordinate representation"],
            ["geometry of function","geometry of linearity","base and dimension"],
            ["geometric proof","how to determine norm","Notion of vector addition and components"]
        ]

        for i in range(len(transition_points)):
            self.transition(
                transition_name=transition_points[i],
                transition_name_ref=transition_points_ref[i],
                ch_name=ch_points[i],
                index=i + 1,
                total=len(transition_points),
            )



    def transition(self, transition_name, transition_name_ref, ch_name, index, total):
        """
        Create transitions easily.

        - Transition name — string, self explanatory
        - Index correspond to the position of this transition on the video
        - Total corresponds to the total amount of transitions there will be

        Total will generate a number of nodes and index will highlight that specific
        node, showing the progress.
        """

        if isinstance(transition_name, list):
            with RegisterFont("Courier Prime") as fonts:
                subtitles = [
                    Text(t,font=fonts[0])
                    for t in transition_name
                ]

                title = (
                    VGroup(*subtitles)
                    .scale(.45)
                    .arrange(DOWN)
                )

                subtitles_ref = [
                    Text(t,font=fonts[0])
                    for t in transition_name_ref
                ]

                title_ref = (
                    VGroup(*subtitles_ref)
                    .scale(.45)
                    .arrange(DOWN)
                )
        else:

            title = (
                MarkupText(transition_name, weight=BOLD)
                .set_stroke(BLACK, width=10, background=True)
                .scale_to_fit_width(config.frame_width - 3)
                .shift(UP)
            )

            title_ref = (
                MarkupText(transition_name_ref, weight=BOLD)
                .set_stroke(BLACK, width=10, background=True)
                .scale_to_fit_width(config.frame_width - 3)
                .shift(UP)
            )

        title_ref.shift(5.75*LEFT)

        if isinstance(ch_name, list):
            with RegisterFont("Courier Prime") as fonts:
                ch_subtitles = [
                    Text(t,font=fonts[0])
                    for t in ch_name
                ]

                ch_title = (
                    VGroup(*ch_subtitles)
                    .scale(.35)
                    #.arrange(DOWN)
                )
        else:
            ch_title = (
                MarkupText(ch_name, weight=BOLD)
                .set_stroke(BLACK, width=10, background=True)
                .scale_to_fit_width(config.frame_width - 3)
                .shift(UP)
            )

        nodes_and_lines = VGroup()
        for n in range(1, total + 1):
            if n == index:
                node = (
                    Circle()
                    .scale(0.2)
                    .set_stroke(REANLEA_RED)
                    .set_fill(PURE_GREEN, opacity=1)
                )
                nodes_and_lines.add(node)
                

            else:
                nodes_and_lines.add(
                    Circle()
                    .scale(0.2)
                    .set_stroke(REANLEA_PURPLE)
                    .set_fill(REANLEA_AQUA, opacity=1)
                )


            nodes_and_lines.add(Line().set_color(REANLEA_PURPLE))
            

        nodes_and_lines.remove(nodes_and_lines[-1])

        nodes_and_lines.arrange(RIGHT, buff=0.5).scale_to_fit_width(
            config.frame_width - 5
        ).to_edge(DOWN, buff=1)


        if index == 2 :
            global nodes_and_lines_ref

            nodes_and_lines_ref=nodes_and_lines.copy()
            nodes_and_lines_ref[0:2].set_opacity(0)
            nodes_and_lines_ref[7:].set_opacity(0)   

        title.next_to(nodes_and_lines[2*(index-1)],UP)

        rect_2=Rectangle(height=4.5, width=8).scale(.8).shift(.75*UP+3*RIGHT).set_stroke(opacity=1, color=[REANLEA_VIOLET,REANLEA_AQUA])

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        

        
        

        self.play(
            AnimationGroup(
                LaggedStartMap(FadeIn, nodes_and_lines),
                AnimationGroup(
                    FadeIn(title),
                    FadeIn(title_ref),
                ),
                lag_ratio=.35
            )
        )

        ###

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

        s_grp=VGroup(s1,s2_0,s2_1,s3)

        self.add(s_grp)


        ###

        grid=NumberPlane()

        dt_1=Dot(grid.polar_to_point(2, PI+(5*PI/12)+(2*PI/7)+(PI/18)-5*DEGREES)).shift(6*LEFT).set_color(PURE_GREEN).set_sheen(-.4,DOWN)

        dt_2=Dot(grid.polar_to_point(2, PI+(5*PI/12)+(2*PI/7)+(PI/18)+(3*PI/14)+(PI/18)-5*DEGREES)).shift(6*LEFT).set_color(REANLEA_WHITE).set_sheen(-.4,DOWN)

        dt_3=Dot(grid.polar_to_point(2, PI+(5*PI/12)+(2*PI/7)+((3*PI/7)+(PI/18))+(2*PI/18)-5*DEGREES)).shift(6*LEFT).set_color(REANLEA_YELLOW ).set_sheen(-.4,DOWN)

        dt_grp=VGroup(dt_1,dt_2,dt_3)

        

        self.play(
            AnimationGroup(
                x_trac_1.animate.set_value((3*PI/10)+(5*PI/12)),
                x_trac_2_0.animate.set_value((3*PI/10)+(5*PI/12)+(2*PI/7)+(PI/18)),
                x_trac_2_1.animate.set_value((3*PI/10)+(5*PI/12)+(2*PI/7)+(PI/18)+(3*PI/14)+(PI/18)),
                x_trac_3.animate.set_value((3*PI/10)+(5*PI/12)+(2*PI/7)+((3*PI/7)+(PI/18))+(2*PI/18))
            ),
            run_time=1.35
        )

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

        ch_title[0].set_color(REANLEA_YELLOW).next_to(dt_3)
        ch_title[1].set_color(REANLEA_WHITE).next_to(dt_2)
        ch_title[2].set_color(PURE_GREEN).next_to(dt_1)

        self.play(
            Write(ch_title[0])
        )
        self.wait(2)

        self.play(
            Write(ch_title[1])
        )
        self.wait(2)

        self.play(
            Write(ch_title[2])
        )
        self.wait(2)

        
        

        if index != 5:
            self.play(
                AnimationGroup(
                    FadeOut(title),
                    FadeOut(title_ref),
                    FadeOut(dt_grp),
                    FadeOut(s_grp),
                    FadeOut(ch_title)
                ),
                run_time=2
            )
        else:
            self.play(
                FadeIn(rect_2),
                FadeIn(water_mark),
                AnimationGroup(
                    *[FadeOut(mobj) for mobj in self.mobjects],
                    FadeIn(nodes_and_lines_ref)
                ),
                run_time=2
            )
            self.wait(2)
        
        

        
# manim -pqk anim3.py Trailer

# manim -pqh anim3.py Trailer

# manim -sqk anim3.py Trailer


class Trailer_1(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        self.wait()


        # MAIN SCENE

        '''with RegisterFont("Homemade Apple") as fonts:
            txt_cs_0 = Text("EUCLIDEAN SPACE" , font=fonts[0])
            

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
            txt_cs_grp_0.animate.shift(2.5*UP).scale(.5)
        )'''

        rect=Rectangle(height=4.5, width=8).scale(.8).shift(.75*UP+3*RIGHT).set_stroke(opacity=1, color=[REANLEA_VIOLET,REANLEA_AQUA])

        self.play(
            Create(rect)
        )



        
        transition_points = [
            # use a list if we want multiple lines
            [""],
            ["Inner Product"],
            ["Norm"],
            ["Cauchy-Schwarz Inequality"],
            [""]
        ]

        transition_points_ref = [
            # use a list if we want multiple lines
            [""],
            ["Inner","Product"],
            ["Norm"],
            ["Cauchy-Schwarz","Inequality"],
            [""]
        ]

        ch_points = [
            # use a list if we want multiple lines
            ["","",""],
            ["projection and generalization","geometry of inner product","defining euclidean space"],
            ["norms & inner product","coordinate representation","norm, inner product, Pythagoras theorem and law of cosines"],
            ["geometric proof","Triangle inequality","Notion of d–infinite metric"],
            ["","",""]
        ]

        for i in range(len(transition_points)):
            self.transition(
                transition_name=transition_points[i],
                transition_name_ref=transition_points_ref[i],
                ch_name=ch_points[i],
                index=i + 1,
                total=len(transition_points),
            )



    def transition(self, transition_name, transition_name_ref, ch_name, index, total):
        """
        Create transitions easily.

        - Transition name — string, self explanatory
        - Index correspond to the position of this transition on the video
        - Total corresponds to the total amount of transitions there will be

        Total will generate a number of nodes and index will highlight that specific
        node, showing the progress.
        """

        if isinstance(transition_name, list):
            with RegisterFont("Courier Prime") as fonts:
                subtitles = [
                    Text(t,font=fonts[0])
                    for t in transition_name
                ]

                title = (
                    VGroup(*subtitles)
                    .scale(.45)
                    .arrange(DOWN)
                )

                subtitles_ref = [
                    Text(t,font=fonts[0])
                    for t in transition_name_ref
                ]

                title_ref = (
                    VGroup(*subtitles_ref)
                    .scale(.45)
                    .arrange(DOWN)
                )
        else:

            title = (
                MarkupText(transition_name, weight=BOLD)
                .set_stroke(BLACK, width=10, background=True)
                .scale_to_fit_width(config.frame_width - 3)
                .shift(UP)
            )

            title_ref = (
                MarkupText(transition_name_ref, weight=BOLD)
                .set_stroke(BLACK, width=10, background=True)
                .scale_to_fit_width(config.frame_width - 3)
                .shift(UP)
            )

        title_ref.shift(5.75*LEFT)

        if isinstance(ch_name, list):
            with RegisterFont("Courier Prime") as fonts:
                ch_subtitles = [
                    Text(t,font=fonts[0])
                    for t in ch_name
                ]

                ch_title = (
                    VGroup(*ch_subtitles)
                    .scale(.325)
                    #.arrange(DOWN)
                )
        else:
            ch_title = (
                MarkupText(ch_name, weight=BOLD)
                .set_stroke(BLACK, width=10, background=True)
                .scale_to_fit_width(config.frame_width - 3)
                .shift(UP)
            )

        nodes_and_lines = VGroup()
        for n in range(1, total + 1):
            if n == index:
                node = (
                    Circle()
                    .scale(0.2)
                    .set_stroke(REANLEA_RED)
                    .set_fill(PURE_GREEN, opacity=1)
                )
                nodes_and_lines.add(node)
                

            else:
                nodes_and_lines.add(
                    Circle()
                    .scale(0.2)
                    .set_stroke(REANLEA_PURPLE)
                    .set_fill(REANLEA_AQUA, opacity=1)
                )


            nodes_and_lines.add(Line().set_color(REANLEA_PURPLE))
            

        nodes_and_lines.remove(nodes_and_lines[-1])
        nodes_and_lines[0:2].set_opacity(0)
        nodes_and_lines[7:].set_opacity(0)

        nodes_and_lines.arrange(RIGHT, buff=0.5).scale_to_fit_width(
            config.frame_width - 5
        ).to_edge(DOWN, buff=1)


           

        title.next_to(nodes_and_lines[2*(index-1)],UP)

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)

        

        if index == 2 :
            self.play(
                FadeIn(nodes_and_lines)
            )
            self.wait(4)
            self.play(
                AnimationGroup(
                    FadeIn(title),
                    FadeIn(title_ref)
                )
            )
        else:
            self.play(
                AnimationGroup(
                    FadeIn(nodes_and_lines),
                    FadeIn(title),
                    FadeIn(title_ref)
                )
            )



        ###

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

        s_grp=VGroup(s1,s2_0,s2_1,s3)

        self.add(s_grp)


        ###

        grid=NumberPlane()

        dt_1=Dot(grid.polar_to_point(2, PI+(5*PI/12)+(2*PI/7)+(PI/18)-5*DEGREES)).shift(6*LEFT).set_color(PURE_GREEN).set_sheen(-.4,DOWN)

        dt_2=Dot(grid.polar_to_point(2, PI+(5*PI/12)+(2*PI/7)+(PI/18)+(3*PI/14)+(PI/18)-5*DEGREES)).shift(6*LEFT).set_color(REANLEA_WHITE).set_sheen(-.4,DOWN)

        dt_3=Dot(grid.polar_to_point(2, PI+(5*PI/12)+(2*PI/7)+((3*PI/7)+(PI/18))+(2*PI/18)-5*DEGREES)).shift(6*LEFT).set_color(REANLEA_YELLOW ).set_sheen(-.4,DOWN)

        dt_grp=VGroup(dt_1,dt_2,dt_3)

        

        self.play(
            AnimationGroup(
                x_trac_1.animate.set_value((3*PI/10)+(5*PI/12)),
                x_trac_2_0.animate.set_value((3*PI/10)+(5*PI/12)+(2*PI/7)+(PI/18)),
                x_trac_2_1.animate.set_value((3*PI/10)+(5*PI/12)+(2*PI/7)+(PI/18)+(3*PI/14)+(PI/18)),
                x_trac_3.animate.set_value((3*PI/10)+(5*PI/12)+(2*PI/7)+((3*PI/7)+(PI/18))+(2*PI/18))
            ),
            run_time=1.35
        )

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

        ch_title[0].set_color(REANLEA_YELLOW).next_to(dt_3)
        ch_title[1].set_color(REANLEA_WHITE).next_to(dt_2)
        ch_title[2].set_color(PURE_GREEN).next_to(dt_1)

        self.play(
            Write(ch_title[0])
        )
        self.wait(2)

        self.play(
            Write(ch_title[1])
        )
        self.wait(2)

        self.play(
            Write(ch_title[2])
        )
        self.wait(2)



        self.wait(2)
        

        if index != 4:
            self.play(
                AnimationGroup(
                    FadeOut(title),
                    FadeOut(title_ref),
                    FadeOut(dt_grp),
                    FadeOut(s_grp),
                    FadeOut(ch_title)
                ),
                run_time=2
            )
        else:
            self.play(
                FadeIn(water_mark),
                AnimationGroup(
                    *[FadeOut(mobj) for mobj in self.mobjects]
                ),
                run_time=2
            )
            self.wait(2)
        
        

        


# manim -pqh anim3.py Trailer_1

# manim -pqk anim3.py Trailer_1

# manim -sqk anim3.py Trailer_1

###################################################################################################################

class Scene1_intro_1(MovingCameraScene):
    def construct(self):

        self.camera.frame.save_state()

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        water_mark.save_state()
        self.add(water_mark)
        

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

        # horizontal dashed line
        d_line=DashedDoubleArrow(
            start=p5, end=p8, dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA)

        d_line_label= MathTex("d(x,y)").next_to(d_line, .1*UP).scale(0.45).set_color(REANLEA_GREY)

        # Vertical dashed line
        v_line1=DashedLine(
            start=p4, end=p5, stroke_width=1
        ).set_color(RED_D)

        v_line2=DashedLine(
            start=p7, end=p8, stroke_width=1
        ).set_color(RED_D)

        #glowing circle 

        #1

        glow_circ_grp_1_1 = VGroup(dot1,dot2)
        
        glowing_circles_1=VGroup()                   # VGroup( doesn't have append method) 

        for dot in glow_circ_grp_1_1:
            glowing_circle=get_glowing_surround_circle(dot[0], color=REANLEA_YELLOW)
            glowing_circle.save_state()
            glowing_circles_1 += glowing_circle
        
        glowing_circles_1.save_state()



       
        # play region
        #self.add(water_mark)
        self.wait()

        self.play(
            DrawBorderThenFill(dumy_line)
        )
        self.add(line)
        self.play(Uncreate(dumy_line))
        self.play(Create(zero_tick))
        self.play(Create(dot1[0]))
        #self.wait(5)
        self.play(
            Create(dot2[0])
        )
        #self.wait(5)
        self.play(
            AnimationGroup(
                Create(dot1[1]),
                Create(dot2[1])
            )
        )
        #self.wait(2)

        self.play(
            self.camera.frame.animate.scale(0.5).move_to(DOWN + 1.5*RIGHT),
            water_mark.animate.scale(0.5).move_to(0.5*UP + LEFT),
        )
        #self.wait(5)

        self.play(
            AnimationGroup(
                Create(v_line1),
                Create(v_line2)
            )
        )
        #self.wait()
        self.play(Write(d_line))
        #self.wait(2)
        self.play(Write(d_line_label))
        #self.wait(5)

        self.play(Restore(self.camera.frame),Restore(water_mark))
        #self.wait(5)

        self.play(
            FadeIn(*glowing_circles_1),
        )
        #self.wait(5)


        #### last scene of Scene1_intro_1

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


        grp_intro=VGroup(line,zero_tick,dot1,dot2,d_line,d_line_label,v_line1,v_line2)

        tex=MathTex("d(x,y)=")#.set_color(REANLEA_GREEN_LIGHTER).set_sheen(-0.4,DR)
        value=DecimalNumber(1).set_color_by_gradient(REANLEA_MAGENTA).set_sheen(-0.1,LEFT)

        value.add_updater(
            lambda x : x.set_value(((1-(dot2.get_center()[0]/3))/2)+1)
        )
        grp2=VGroup(tex,value).arrange(RIGHT, buff=0.3).move_to(2*UP).set_color_by_gradient(REANLEA_GREEN,REANLEA_MAGENTA)

        '''self.play(
            AnimationGroup(
                Transform(grp_intro,grp_last.reverse_direction(), run_time=1.5),
                FadeOut(*glowing_circles_1)
            )
        )
        self.wait(5)'''

        self.play(
            AnimationGroup(
                AnimationGroup(
                    ReplacementTransform(line,line_last),
                    ReplacementTransform(dot2[0],dot1_last),
                    ReplacementTransform(dot1[0],dot2_last),
                    ReplacementTransform(dot2[1],dot1_last_lbl),
                    ReplacementTransform(dot1[1],dot2_last_lbl),
                    ReplacementTransform(v_line1,v_line1_last),
                    ReplacementTransform(v_line2,v_line2_last),
                    ReplacementTransform(d_line,d_line_last),
                    TransformMatchingShapes(d_line_label,grp2)
                ),
                AnimationGroup(
                    FadeOut(*glowing_circles_1),
                    FadeOut(zero_tick)
                )
            ),
            run_time=1.5
        )
        #self.wait(5)

        
        
        
        
        
        


         # manim -pqk anim3.py Scene1_intro_1

         # manim -pqh anim3.py Scene1_intro_1

         # manim -sqk anim3.py Scene1_intro_1


class Scene1_intro_2(Scene):
    def construct(self):
        # WATER-MARK
        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

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

        


        # TEXT

        tex=MathTex("d(x,y)=")#.set_color(REANLEA_GREEN_LIGHTER).set_sheen(-0.4,DR)
        grp2=VGroup(tex,value).arrange(RIGHT, buff=0.3).move_to(2*UP).set_color_by_gradient(REANLEA_GREEN,REANLEA_MAGENTA)


        

        


        # play region

        self.add(grp1, dot1_lbl,dot3_lbl,grp2)
        self.wait()
        self.add(dot3)
        #self.wait(5)
        '''self.play(
            MoveAlongPath(dot2, line1, rate_func=rate_functions.ease_in_out_sine),
            run_time=3
        )'''
        self.play(
            x.animate.set_value(dot1.get_center()[0]),
            run_time=2.5
        )
        #self.wait(5)
        
        self.play(
            x.animate.set_value(dot3.get_center()[0] + dot1.get_center()[0]/4)
        )
        self.play(
            x.animate.set_value(dot3.get_center()[0]/2 + dot1.get_center()[0]),
        )
        self.play(
            x.animate.set_value(dot3.get_center()[0] + dot1.get_center()[0]/3)
        )
        self.play(
            x.animate.set_value(dot3.get_center()[0]/4 + dot1.get_center()[0])
        )
        
        self.play(
            x.animate.set_value(dot3.get_center()[0] + dot1.get_center()[0]/2.75),
            #run_time=5
        )

        self.play(
            x.animate.set_value(dot3.get_center()[0]/4 + dot1.get_center()[0]),
            #run_time=5
        )

        self.play(
            x.animate.set_value(dot3.get_center()[0]),
            #run_time=3
        )

        grp1_2=VGroup(d_line,v_line1, v_line2)

        dot1_ref=Dot(radius=0.25, color=REANLEA_GREEN).move_to(3*RIGHT).set_sheen(-0.6,DOWN)
        dot2_ref=Dot(radius=0.25, color=REANLEA_VIOLET_LIGHTER).move_to(3*LEFT).set_sheen(-0.4,DOWN)
        dot4_ref=Dot(radius=0.125, color=REANLEA_YELLOW).move_to(LEFT+2.5*UP).set_sheen(-0.6,DOWN)
        line1_ref=Line(start=dot2.get_center(), end=dot1.get_center()).set_color(REANLEA_YELLOW_DARKER).set_stroke(width=10).set_z_index(-10)

        dot1_lbl2=MathTex("y").scale(0.6).next_to(dot1_ref, RIGHT)
        dot2_lbl2=MathTex("x").scale(0.6).next_to(dot2_ref, LEFT)
        dot4_ref_lbl2=MathTex("z").scale(0.6).next_to(dot4_ref, UP)

        self.play(
            AnimationGroup(
                FadeOut(grp2),
                FadeOut(grp1_2)
            )
        )

        self.play(
            AnimationGroup(
                ReplacementTransform(VGroup(dot1,dot2),VGroup(dot1_ref,dot2_ref))
            ),
            ReplacementTransform(line,line1_ref),
            AnimationGroup(
                FadeIn(dot4_ref),
                Write(dot4_ref_lbl2),
                lag_ratio=.5
            ),
            AnimationGroup(
                ReplacementTransform(dot1_lbl,dot1_lbl2),
                ReplacementTransform(dot3_lbl,dot2_lbl2)
            )
        )
        self.wait(2)



    # manim -pqk anim3.py Scene1_intro_2

    # manim -pqh anim3.py Scene1_intro_2

    # manim -sqk anim3.py Scene1_intro_2


class Scene1_intro_3(Scene):
    def construct(self):

        # WATER MARK 

        '''water_mark=ImageMobject("background_4.png").scale(.5).set_z_index(-10)
        self.add(water_mark)'''

        water_mark_1=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark_1)

        
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

        ln_grp=VGroup(line1).set_z_index(-10)

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

        self.add(dt_grp,ln_grp,lbl_grp)
        self.wait()

        self.play(
            Write(cir_grp)
        )
        self.play(
            Write(brc_grp)
        )
        self.play(
            Write(ind_ln_0)
        )
        self.play(
            Write(eq145)
        )

        self.wait(2)

        water_mark_2=water_mark_1.copy()
        eq145_ref=eq145.copy()

        self.play(            
            AnimationGroup(
                *[FadeOut(mobj) for mobj in self.mobjects],
            ),
            FadeIn(water_mark_2),
            eq145_ref.animate.scale(.775).move_to(1.55*UP+4*LEFT),
            run_time=1.75
        )

        dumy_ln_2=Line().rotate(-90*DEGREES).set_stroke(width=2, color=[REANLEA_BLUE_SKY,REANLEA_VIOLET]).scale(1.5).move_to(.5*DOWN+4*LEFT)

        bulet_1=Dot(radius=DEFAULT_DOT_RADIUS/1.25, color=REANLEA_WHITE).set_sheen(-.4,DOWN).move_to(.5*UP+4*LEFT)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_sym_1=Text("How are the metric and metric spaces  defined? ", font=fonts[0]).scale(.65).set_color(REANLEA_CYAN_LIGHT).next_to(bulet_1,RIGHT)

        bulet_2=Dot(radius=DEFAULT_DOT_RADIUS/1.25, color=REANLEA_WHITE).set_sheen(-.4,DOWN).next_to(bulet_1,DOWN).shift(.75*DOWN)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_sym_2=Text("how does the concept of vectors make a difference from a point?", font=fonts[0]).scale(.65).set_color(REANLEA_CYAN_LIGHT).next_to(bulet_2,RIGHT)

        bulet_3=Dot(radius=DEFAULT_DOT_RADIUS/1.25, color=REANLEA_WHITE).set_sheen(-.4,DOWN).next_to(bulet_2,DOWN).shift(.75*DOWN)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_sym_3=Text("How we can reflect the concept of distance in the space of vectors", font=fonts[0]).scale(.65).set_color(REANLEA_CYAN_LIGHT).next_to(bulet_3,RIGHT)

            txt_sym_4=Text(" and generalise it?", font=fonts[0]).scale(.65).set_color(REANLEA_CYAN_LIGHT).next_to(bulet_3,DOWN).shift(1.625*RIGHT)

    
        self.play(
            Create(dumy_ln_2)
        )

        self.play(
            Write(bulet_1)
        )
        self.play(
            Create(txt_sym_1),
            run_time=2
        )
        self.wait()

        self.play(
            Write(bulet_2)
        )
        self.play(
            Create(txt_sym_2),
            run_time=2
        )
        self.wait()

        self.play(
            Write(bulet_3)
        )
        self.play(
            Create(txt_sym_3),
            run_time=2
        )
        self.play(
            Create(txt_sym_4)
        )
        self.wait(2)

        water_mark_3=water_mark_2.copy()

        self.play(            
            AnimationGroup(
                *[FadeOut(mobj) for mobj in self.mobjects],
            ),
            FadeIn(water_mark_3),
            run_time=1.75
        )

        self.wait(5)



    
    # manim -pqk anim3.py Scene1_intro_3

    # manim -pqh anim3.py Scene1_intro_3

    # manim -sqk anim3.py Scene1_intro_3


###################################################################################################################

class Scene2_intro_0(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        water_mark_1=water_mark.copy()



        # OBJECTS

        promo_img=ImageMobject("promo.jpg").scale(.545).set_z_index(-10)
        self.add(promo_img)
        self.wait(2)
        self.play(
            FadeOut(promo_img),
            run_time=.25
        )
        self.wait(.25)

        with RegisterFont("Courier Prime") as fonts:
            txt_1=Text("Lec. - II", font=fonts[0]).set_color_by_gradient(REANLEA_TXT_COL).scale(0.5)
        
        
        self.play(
            AddTextLetterByLetter(txt_1),
            run_time=1.5
        )
        self.wait(.5)
        self.play(
            FadeOut(txt_1)
        )
        self.wait(1.5)

        eq_1_1=MathTex("1","+","1","=","2").scale(1.3).set_color(REANLEA_CYAN_LIGHT)
        eq_1_2=MathTex("-(-1)","=","1").scale(1.3).set_color(REANLEA_CYAN_LIGHT).next_to(eq_1_1,DOWN).shift(.215*LEFT)
        self.play(
            Create(eq_1_1)
        )
        self.play(
            AnimationGroup(
                eq_1_1.animate.shift(.5*UP),
                Create(eq_1_2)
            )
        )
        eq_1=VGroup(eq_1_1,eq_1_2)

        sep_ln_1=Line().rotate(PI/2).scale(1).set_stroke(width=2, color=REANLEA_PURPLE_LIGHTER).next_to(eq_1,RIGHT).shift(RIGHT)

        self.play(
            Create(sep_ln_1.reverse_direction())
        )

        with RegisterFont("Courier Prime") as fonts:
            txt_2=Text("How?", font=fonts[0]).set_color_by_gradient(REANLEA_CYAN_LIGHT).scale(1.75)

            txt_2.next_to(sep_ln_1).shift(RIGHT)
        
        self.play(
            AddTextLetterByLetter(txt_2),
            run_time=1.25
        )

        sep_ln_2=Line().rotate(-90*DEGREES).set_stroke(width=2.5, color=[REANLEA_BLUE_SKY,REANLEA_VIOLET]).scale(.65).shift(2.15*DOWN)

        self.play(
            Create(sep_ln_2)
        )

        bulet_1=Dot(radius=DEFAULT_DOT_RADIUS/1.25, color=REANLEA_WHITE).set_sheen(-.4,DOWN).shift(1.85*DOWN)

        self.play(
            Write(bulet_1)
        )

        with RegisterFont("Reenie Beanie") as fonts:
            txt_x_1=Text("Vector Addition", font=fonts[0]).scale(.65).set_color(REANLEA_WHITE).next_to(bulet_1,RIGHT)

        self.play(
            Write(txt_x_1)
        )

        bulet_2=Dot(radius=DEFAULT_DOT_RADIUS/1.25, color=REANLEA_WHITE).set_sheen(-.4,DOWN).next_to(bulet_1,DOWN).shift(.35*DOWN)

        self.play(
            Write(bulet_2)
        )

        with RegisterFont("Reenie Beanie") as fonts:
            txt_x_2=Text("Vector Scale", font=fonts[0]).scale(.65).set_color(REANLEA_WHITE).next_to(bulet_2,RIGHT)

        self.play(
            Write(txt_x_2)
        )

        sub_def_grp=VGroup(sep_ln_2,bulet_1,bulet_2,txt_x_1,txt_x_2)


        grp_1=VGroup(eq_1,sep_ln_1,txt_2,sub_def_grp)
        sur_grp_1=Circle(radius=2.15).set_stroke(width=3, color=[PURE_GREEN,REANLEA_AQUA])

        self.play(
            AnimationGroup(
                grp_1.animate.scale(.45).move_to(ORIGIN),
                Create(sur_grp_1),
                lag_ratio=.8
            )
        )

        grp_1_1=VGroup(grp_1,sur_grp_1)
        self.play(
            grp_1_1.animate.shift(2*LEFT)
        )

        ln_1=Line().set_stroke(width=1).rotate(15*DEGREES).scale(1.5).shift(RIGHT)

        self.play(
            Create(ln_1)
        )

        with RegisterFont("Cousine") as fonts:
            vsp_exp_1=Text("Vector Space", font=fonts[0]).scale(0.65).set_color_by_gradient(REANLEA_GREEN,REANLEA_AQUA).next_to(ln_1).shift(.4*UP)
        
        self.play(
            Write(vsp_exp_1)
        )

        l_1=Line().rotate(PI/2).set_stroke(width=3, color=(REANLEA_PINK,REANLEA_YELLOW)).scale(0.75).next_to(vsp_exp_1,DOWN)

        self.play(
            Create(l_1.reverse_direction())
        )

        with RegisterFont("Pacifico") as fonts:
            fld_exp_1=Text("Field", font=fonts[0]).scale(0.85).set_color_by_gradient(REANLEA_PINK,REANLEA_MAGENTA).next_to(l_1,DOWN)

        
        self.play(
            Write(fld_exp_1)
        )

        with RegisterFont("Cousine") as fonts:
            txt_3=Text("* Peano's Axiom", font=fonts[0]).set_color_by_gradient(REANLEA_CYAN_LIGHT).scale(.2).next_to(sur_grp_1,DOWN).shift(.25*DOWN)

            txt_4=Text("** Construction of Natural Numbers", font=fonts[0]).set_color_by_gradient(REANLEA_CYAN_LIGHT).scale(.2).next_to(txt_3,DOWN).shift(.67*RIGHT)

        self.play(
            Write(txt_3)
        )

        self.play(
            Write(txt_4)
        )

        self.wait(2)

        self.play(            
            AnimationGroup(
                *[FadeOut(mobj) for mobj in self.mobjects],
            ),
            FadeIn(water_mark_1),
            run_time=1.75
        )

        with RegisterFont("Courier Prime") as fonts:
            txt_5=Text("Subscribe to stay connected with us.", font=fonts[0]).set_color_by_gradient(REANLEA_TXT_COL).scale(0.5)
        
        
        self.play(
            AddTextLetterByLetter(txt_5),
            run_time=3
        )
        self.wait(.75)
        
        self.play(
            FadeOut(txt_5)
        ) 

        self.wait(5)
            



        # manim -pqh anim3.py Scene2_intro_0

        # manim -pqk anim3.py Scene2_intro_0

        # manim -sqk anim3.py Scene2_intro_0

###################################################################################################################

class Scene3_intro_0(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        water_mark_1=water_mark.copy()



        # OBJECTS

        decarte_img=ImageMobject("de_carte.jpg").scale(1.25).shift(.75*UP)
        self.play(
            FadeIn(decarte_img)
        )
        with RegisterFont("Courier Prime") as fonts:
            cap_decarte=Text("René Descartes", font=fonts[0]).set_color_by_gradient(REANLEA_TXT_COL).scale(0.65).next_to(decarte_img,DOWN).shift(.25*DOWN)
            cap_decarte_date=Text("(31 March 1596 – 11 February 1650)", font=fonts[0]).set_color_by_gradient(REANLEA_TXT_COL).scale(0.175).next_to(cap_decarte,DOWN)
        
        
        self.play(
            AnimationGroup(
            Write(cap_decarte),
            FadeIn(cap_decarte_date)
            )
        )
        
        self.wait(2)

        decarte_grp=Group(decarte_img,cap_decarte,cap_decarte_date)
        sur_grp_1=Circle(radius=2.15).set_stroke(width=3, color=[PURE_GREEN,REANLEA_AQUA]).shift(4*LEFT+.2*UP).set_z_index(10)

        self.play(
            AnimationGroup(
            decarte_grp.animate.shift(4*LEFT).scale(.65),
            Create(sur_grp_1),
            lag_ratio=.35
            )
        )
        self.wait(2)

        #PORTION-II

        with RegisterFont("Courier Prime") as fonts:
            txt_1=Text("Cartesian Coordinates", font=fonts[0]).set_color_by_gradient(REANLEA_CYAN_LIGHT).scale(.35).shift(RIGHT)

        sep_ln_1=Line().rotate(PI/2).scale(1).set_stroke(width=2, color=REANLEA_PURPLE_LIGHTER).next_to(txt_1,RIGHT).shift(.5*RIGHT)

        ax_1=Axes(
            x_range=[-1.5,5.5],
            y_range=[-1.5,4.5],
            y_length=(round(config.frame_width)-2)*6/7,
            tips=False, 
            axis_config={
                "font_size": 24,
                #"include_ticks": False,
            }, 
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.15).set_z_index(2).next_to(sep_ln_1).shift(.5*RIGHT)

        np_1=NumberPlane(
            background_line_style={
                "stroke_opacity": 0.5
            }
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.2).set_z_index(2).next_to(sep_ln_1).shift(.5*RIGHT)

        self.play(
            AnimationGroup(
                Write(np_1),
                Create(txt_1)
            )
        )
        self.wait(2)

        self.play(
            Create(sep_ln_1.reverse_direction())
        )
        self.wait(2)
        
        self.play(
            txt_1.animate.shift(.5*UP)
        )
        self.wait(2)

        sep_ln_2=Line().rotate(-90*DEGREES).set_stroke(width=2.5, color=[REANLEA_BLUE_SKY,REANLEA_WHITE]).next_to(txt_1,DOWN).scale(.65).shift(.25*UP+.5*LEFT)

        self.play(
            Create(sep_ln_2)
        )
        self.wait(2)

        bulet_1=Dot(radius=DEFAULT_DOT_RADIUS/1.25, color=REANLEA_WHITE).set_sheen(-.4,DOWN).move_to(sep_ln_2.get_start()).shift(.35*DOWN)

        self.play(
            Write(bulet_1)
        )
        self.wait(2)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_x_1=Text("Geometry", font=fonts[0]).scale(.5).set_color(REANLEA_WHITE).next_to(bulet_1,RIGHT)

        self.play(
            Write(txt_x_1)
        )
        self.wait(2)

        bulet_2=Dot(radius=DEFAULT_DOT_RADIUS/1.25, color=REANLEA_WHITE).set_sheen(-.4,DOWN).next_to(bulet_1,DOWN).shift(.35*DOWN)

        self.play(
            Write(bulet_2)
        )
        self.wait(2)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_x_2=Text("Algebra", font=fonts[0]).scale(.5).set_color(REANLEA_WHITE).next_to(bulet_2,RIGHT)

        self.play(
            Write(txt_x_2)
        )
        self.wait(2)

        sub_def_grp=VGroup(sep_ln_2,bulet_1,bulet_2,txt_x_1,txt_x_2)

        grp_cp=VGroup(sub_def_grp,txt_1,sep_ln_1,np_1)

        ind_ln_1=Line().rotate(30*DEGREES).scale(2.15).set_stroke(width=2, color=REANLEA_PURPLE_LIGHTER).next_to(sur_grp_1,RIGHT).shift(.6*LEFT+.75*UP).set_z_index(5)

        self.play(
            grp_cp.animate.scale(.5).shift(2.5*UP),
            Create(ind_ln_1)
        )
        self.wait(2)

        sur_grp_cp=SurroundingRectangle(grp_cp, corner_radius=.12).set_stroke(width=1, color=[REANLEA_WHITE,REANLEA_PINK]).scale(.75)

        self.play(
            Create(sur_grp_cp)
        )
        self.wait(2)

        ind_ln_5=Line().rotate(90*DEGREES).scale(.45).set_stroke(width=2, color=REANLEA_PURPLE_LIGHTER).next_to(grp_cp,DOWN).shift(1.5*RIGHT+.1*DOWN).set_z_index(5)

        self.play(
            Create(ind_ln_5.reverse_direction())
        )
        self.wait(2)

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

        with RegisterFont("Courier Prime") as fonts:
            txt_2=Text("Cartesian Product (Set Theory)", font=fonts[0]).set_color_by_gradient(REANLEA_CYAN_LIGHT).scale(.65).next_to(eqn_1,DOWN).shift(.25*DOWN)

        set_a_grp=VGroup(grph_1,grph_1_lbl,dot_1,dot_1_lbl).shift(2*RIGHT)
        set_b_grp=VGroup(grph_2,grph_2_lbl,dot_2,dot_2_lbl).shift(2*LEFT)
        cp_grp_1=VGroup(set_a_grp,set_b_grp,eqn_1,txt_2).scale(.35).move_to(4.5*RIGHT).shift(.5*DOWN+.5*RIGHT)


        self.play(
            Create(cp_grp_1),
            run_time=2.5
        )
        self.wait(2)


        sur_cp_grp_1= SurroundingRectangle(cp_grp_1, corner_radius=.12).set_stroke(width=1, color=[REANLEA_WHITE,REANLEA_YELLOW_GREEN]).scale(1.25)

        self.play(
            Create(sur_cp_grp_1)
        )
        self.wait(2)

        with RegisterFont("Courier Prime") as fonts:
            txt_3=Text("Vector Addition", font=fonts[0]).set_color_by_gradient(REANLEA_CYAN_LIGHT).scale(.35).next_to(grp_cp,DOWN).shift(4.65*DOWN+1.5*LEFT)

        ind_ln_2=Line().rotate(90*DEGREES).scale(2.15).set_stroke(width=2, color=REANLEA_PURPLE_LIGHTER).next_to(grp_cp,DOWN).shift(1.5*LEFT+.1*DOWN).set_z_index(5)

        self.play(
            Create(ind_ln_2.reverse_direction())
        )
        self.wait(2)

        self.play(
            Create(txt_3)
        )
        self.wait(2)

        with RegisterFont("Courier Prime") as fonts:
            txt_4=Text("Dimension", font=fonts[0]).set_color_by_gradient(REANLEA_CYAN_LIGHT).scale(.35).next_to(txt_3,RIGHT).shift(2.5*RIGHT)


        ind_ln_3=Line().rotate(90*DEGREES).scale(.6).set_stroke(width=2, color=REANLEA_PURPLE_LIGHTER).next_to(txt_4,UP).shift(.015*DOWN).set_z_index(5)

        self.play(
            Create(ind_ln_3.reverse_direction())
        )
        self.wait(2)

        self.play(
            Create(txt_4)
        )
        self.wait(2)

        ind_ln_4=Line().scale(1.15).set_stroke(width=2, color=REANLEA_PURPLE_LIGHTER).next_to(txt_3,RIGHT).set_z_index(5)

        self.play(
            Create(ind_ln_4)
        )
        self.wait(2)

        self.play(            
            AnimationGroup(
                *[FadeOut(mobj) for mobj in self.mobjects],
            ),
            FadeIn(water_mark_1),
            run_time=1.75
        )


        


        
        self.wait(5)
            



        # manim -pqh anim3.py Scene3_intro_0

        # manim -pqk anim3.py Scene3_intro_0

        # manim -sqk anim3.py Scene3_intro_0

###################################################################################################################

class Scene4_intro_0(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        self.wait()

        water_mark_1=water_mark.copy()

        # OBJECTS

        dt_1=Dot(point=4*LEFT,radius=.1,color=REANLEA_YELLOW)
        dt_2=Dot(point=4*RIGHT,radius=.1,color=PURE_GREEN)
        self.play(
            Write(VGroup(dt_1,dt_2))
        )
        self.wait()

        ln_1=Line(4*LEFT,4*RIGHT).set_stroke(width=7.5, color=[PURE_GREEN,REANLEA_YELLOW])
        self.play(Create(ln_1))
        self.wait()

        with RegisterFont("Courier Prime") as fonts:
            txt_1=Text("dimension = 1", font=fonts[0]).set_color_by_gradient(REANLEA_CYAN_LIGHT).scale(1).next_to(ln_1,DOWN).shift(1.5*DOWN)

        self.play(
            AnimationGroup(Write(txt_1),
                AnimationGroup(dt_1.animate.scale(0),dt_2.animate.scale(0))
            )
        )
        self.wait(2)

        def func(t):
            return [t,np.exp(-t ** 2),0]
        
        f = ParametricFunction(func, t_range=np.array([-3, 3]), fill_opacity=0).set_stroke(width=7.5, color=[PURE_GREEN,REANLEA_YELLOW])
        self.play(ReplacementTransform(ln_1,f))




        # manim -pqh anim3.py Scene4_intro_0

        # manim -pqk anim3.py Scene4_intro_0

        # manim -sqk anim3.py Scene4_intro_0

###################################################################################################################

class trailer_0(Scene):
    def construct(self):

        # water mark 
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
        ).set_color(REANLEA_TXT_COL_DARKER).scale(.5).set_z_index(-5)

        dt_0=Dot().set_color(REANLEA_AQUA).move_to(ax_1.c2p(0,0)).set_z_index(2)

        self.play(
            Write(ax_1),
            run_time=2
        )
        self.play(
            Write(dt_0)
        )
        
        self.wait(2)
        
        
        dt_1=Dot().set_color(REANLEA_GOLD).move_to(ax_1.c2p(3,2)).set_z_index(2)
        dt_2=Dot().set_color(REANLEA_SLATE_BLUE).move_to(ax_1.c2p(2,1)).set_z_index(2)
        dt_2_ref=Dot().set_color(REANLEA_BLUE).move_to(ax_1.c2p(2,1)).set_z_index(2)

        dt_1_lbl=MathTex("(","3",",","2",")").scale(.45).set_color(REANLEA_GOLD).next_to(dt_1,UR, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2)
        dt_2_lbl=MathTex("(","2",",","1",")").scale(.45).set_color(REANLEA_SLATE_BLUE).next_to(dt_2,RIGHT, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2)
        dt_2_lbl_ref=MathTex("(","2",",","1",")").scale(.45).set_color(REANLEA_BLUE).next_to(dt_2,RIGHT, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2)

        ln_1=Line(start=dt_0.get_center(), end=dt_1.get_center()).set_stroke(width=5, color=[REANLEA_GOLD,REANLEA_AQUA])
        ln_2=Line(start=dt_0.get_center(), end=dt_2.get_center()).set_stroke(width=5, color=[REANLEA_SLATE_BLUE_LIGHTER,REANLEA_AQUA])
        ln_2_ref=Line(start=dt_0.get_center(), end=dt_2.get_center()).set_stroke(width=5, color=[REANLEA_BLUE,REANLEA_AQUA])

        self.play(
            Write(dt_1),
            Write(dt_2)
        )
        self.wait(2)
        self.play(
            AnimationGroup(
                Create(dt_1_lbl),
                Create(dt_2_lbl)
            ),
            AnimationGroup(
                Create(ln_1),
                Create(ln_2)
            )
        )
        self.wait(2)

        txt_1=MathTex(r"\langle","(3,2)",",","(2,1)",r"\rangle").scale(.65).shift(2.75*UP+3*RIGHT).set_color(REANLEA_CYAN_LIGHT)
        txt_1[1].set_color(REANLEA_GOLD)
        txt_1[3].set_color(REANLEA_SLATE_BLUE_LIGHTER)

        self.play(
            ReplacementTransform(VGroup(dt_1_lbl.copy(),dt_2_lbl.copy()),txt_1[1:4]),
            FadeIn(txt_1[0]),
            FadeIn(txt_1[4])
        )
        self.wait(2)

        ind_ln_0=Line().scale(.25).set_stroke(width=1).rotate(-35*DEGREES).next_to(txt_1,DOWN).shift(.35*RIGHT)

        with RegisterFont("Cousine") as fonts:
            ind_ln_0_lbl = VGroup(*[Text(x, font=fonts[0]) for x in (
                "measures the projection",
                "of one vector onto another"
            )]).arrange_submobjects(DOWN).scale(.25).set_color_by_gradient(REANLEA_BLUE_LAVENDER).next_to(ind_ln_0.get_end()).shift(.1875*LEFT+.1875*DOWN)
            ind_ln_0_lbl[1].shift(.15*RIGHT)

        self.play(
            Create(ind_ln_0)
        )
        self.play(
            Write(ind_ln_0_lbl),
            run_time=2
        ) 
        self.wait(2)
        
        dt_3=Dot().set_color(REANLEA_SLATE_BLUE).move_to(ax_1.c2p(1,0)).set_z_index(2)
        dt_3_lbl=MathTex("(","1",",","0",")").scale(.45).set_color(REANLEA_SLATE_BLUE).next_to(dt_3,DR, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2)
        ln_3=Line(start=dt_0.get_center(), end=dt_3.get_center()).set_stroke(width=5, color=[REANLEA_SLATE_BLUE_LIGHTER,REANLEA_AQUA])

        txt_2=MathTex(r"\langle","(3,2)",",","(1,0)",r"\rangle").scale(.65).shift(2.75*UP+3*RIGHT).set_color(REANLEA_CYAN_LIGHT)
        txt_2[1].set_color(REANLEA_GOLD)
        txt_2[3].set_color(REANLEA_SLATE_BLUE_LIGHTER)

        txt_2_ref=MathTex(r"\langle","(3,2)",",","(2,1)",r"\rangle").scale(.65).shift(2.75*UP+3*RIGHT).set_color(REANLEA_CYAN_LIGHT)
        txt_2_ref[1].set_color(REANLEA_GOLD)
        txt_2_ref[3].set_color(REANLEA_BLUE)

        self.play(
            AnimationGroup(
                ReplacementTransform(dt_2,dt_3),
                ReplacementTransform(ln_2,ln_3),
                ReplacementTransform(dt_2_lbl,dt_3_lbl)
            ),
            ReplacementTransform(txt_1,txt_2)
        )
        self.wait(2)

        dissipating_dt_1=Dot().move_to(ax_1.c2p(3,2)).set_opacity(opacity=0)
        dissipating_path_1 = TracedPath(dissipating_dt_1.get_center, dissipating_time=0.5, stroke_color=[REANLEA_GOLD],stroke_opacity=[1, 0])
        self.add(dissipating_dt_1,dissipating_path_1)
        ln_dis_1=Line(ax_1.c2p(3,2),ax_1.c2p(3,0))

        dt_4=Dot().set_color(PURE_RED).move_to(ax_1.c2p(3,0)).set_z_index(2)
        dt_4_lbl=MathTex("(","3",",","0",")").scale(.45).set_color(PURE_RED).next_to(dt_4,DOWN, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2)

        self.play(
            AnimationGroup(
                MoveAlongPath(dissipating_dt_1,ln_dis_1),
                AnimationGroup(
                    Flash(point=Dot().move_to(ax_1.c2p(3,0)), color=PURE_GREEN),
                    FadeIn(dt_4)
                ),               
                lag_ratio=0.5
            )
        )
        self.play(
            Write(dt_4_lbl)
        )

        d_ln_1=DashedDoubleArrow(
            start=dt_0.get_center(), end=dt_4.get_center(), dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).shift(.75*DOWN).set_color_by_gradient(REANLEA_AQUA,REANLEA_GOLD)

        d_ln_1_lbl=MathTex("3","=",r"\sqrt{3^{2} + 2^{2}}",r"\cdot",r"cos(tan^{-1}(\frac{2}{3}))").scale(.4).next_to(d_ln_1,DOWN).shift(1.2*RIGHT+.175*UP)

        self.play(
            Create(d_ln_1)
        )
        self.play(
            Write(d_ln_1_lbl[0])
        )
        self.wait(2)

        self.play(
            Write(d_ln_1_lbl[1:])
        )
        self.wait(2)

        self.play(
            AnimationGroup(
                Indicate(dt_1),
                Indicate(dt_3),
            ),
            AnimationGroup(
                Indicate(dt_1_lbl),
                Indicate(dt_3_lbl),
            ),
            AnimationGroup(
                Indicate(txt_2[1]),
                Indicate(txt_2[3]),
            ),
            run_time=1.25
        )
        self.wait(2)

        txt_3= MathTex(r"=",r"\lVert (3,2) \rVert",r"\cdot",r"\lVert (1,0) \rVert",r"\cdot",r"cos\theta").scale(.475).next_to(txt_2).set_color(REANLEA_CYAN_LIGHT)
        txt_3[1][1:6].set_color(REANLEA_GOLD)
        txt_3[3][1:6].set_color(REANLEA_SLATE_BLUE_LIGHTER)
        txt_3[5][-1].set_color(REANLEA_YELLOW_GREEN)


        self.play(
            Write(txt_3)
        )

        angl_1=Angle(ln_2,ln_1).set_color(REANLEA_YELLOW_GREEN).set_stroke(width=3.5).set_z_index(-1)
        
        angl_1_lbl=MathTex(r"\theta").scale(.4).set_color(REANLEA_YELLOW_GREEN).next_to(angl_1,UR, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2).shift(.25*DOWN).set_z_index(2)

        self.play(
            Create(angl_1)
        )
        self.play(
            Write(angl_1_lbl)
        )
        self.wait(2)



        eqn_1_0=txt_2.copy().scale(.8).move_to(4.25*LEFT+2.75*UP)

        ind_ln_1_0=Line().scale(.75).set_stroke(width=1).rotate(-135*DEGREES).next_to(eqn_1_0,DOWN).shift(.7*LEFT)

        eqn_1_1=txt_3[1:].copy().scale(.9).next_to(ind_ln_1_0.get_end()).shift(.25*DOWN+1.35*LEFT)

        ind_ln_1_1=Line().scale(.75).set_stroke(width=1).rotate(-45*DEGREES).next_to(ind_ln_1_0)

        ind_ln_1_2=Line().scale(.35).set_stroke(width=1).next_to(eqn_1_1)

        eqn_1_2= MathTex(r"3 \cdot 1",r"+",r"2 \cdot 0").scale(.4275).next_to(ind_ln_1_2).set_color(REANLEA_CYAN_LIGHT)
        eqn_1_2[0][0].set_color(REANLEA_GOLD)
        eqn_1_2[2][0].set_color(REANLEA_GOLD)
        eqn_1_2[0][2].set_color(REANLEA_SLATE_BLUE_LIGHTER)
        eqn_1_2[2][2].set_color(REANLEA_SLATE_BLUE_LIGHTER)
        
        
        self.play(
            ReplacementTransform(txt_2.copy(),eqn_1_0)
        )
        self.play(
            Create(ind_ln_1_0)
        )
        self.play(
            ReplacementTransform(txt_3[1:],eqn_1_1),
            FadeOut(txt_3[0])
        )
        self.wait(2)

        txt_4= MathTex(r"=",r"3 \cdot 1",r"+",r"2 \cdot 0").scale(.65).next_to(txt_2).set_color(REANLEA_CYAN_LIGHT)
        txt_4[1:].shift(.1*RIGHT)
        txt_4[1][0].set_color(REANLEA_GOLD)
        txt_4[3][0].set_color(REANLEA_GOLD)
        txt_4[1][2].set_color(REANLEA_SLATE_BLUE_LIGHTER)
        txt_4[3][2].set_color(REANLEA_SLATE_BLUE_LIGHTER)

        txt_4_ref= MathTex(r"=",r"3 \cdot 2",r"+",r"2 \cdot 1").scale(.65).next_to(txt_2).set_color(REANLEA_CYAN_LIGHT)
        txt_4_ref[1:].shift(.1*RIGHT)
        txt_4_ref[1][0].set_color(REANLEA_GOLD)
        txt_4_ref[3][0].set_color(REANLEA_GOLD)
        txt_4_ref[1][2].set_color(REANLEA_BLUE)
        txt_4_ref[3][2].set_color(REANLEA_BLUE)

        self.play(
            AnimationGroup(
                AnimationGroup(
                    Create(ind_ln_1_1),
                    Create(ind_ln_1_2)
                ),
                AnimationGroup(
                    Write(txt_4),
                    Write(eqn_1_2)
                ),
                lag_ratio=.75
            )
        )
        self.wait(2)

        ind_ln_2_1=Line().scale(.75).set_stroke(width=1).rotate(-45*DEGREES).next_to(eqn_1_1,DOWN).shift(.61*RIGHT+.125*UP) 

        ind_ln_2_2=Line().scale(.75).set_stroke(width=1).rotate(-135*DEGREES).next_to(ind_ln_2_1)

        eqn_2_0=MathTex(r"\sqrt{3^{2} + 2^{2}}",r"\cdot",r"cos(tan^{-1}(\frac{2}{3}))").scale(.52).next_to(ind_ln_2_2.get_end(),DOWN).shift(.2*LEFT+.175*UP)
        eqn_2_0[0][2].set_color(REANLEA_GOLD)
        eqn_2_0[0][5].set_color(REANLEA_GOLD)
        eqn_2_0[2][4:14].set_color(REANLEA_YELLOW_GREEN)

        self.play(
            AnimationGroup(
                AnimationGroup(
                    Create(ind_ln_2_1),
                    Create(ind_ln_2_2)
                ),
                ReplacementTransform(d_ln_1_lbl[2:].copy(),eqn_2_0),
                lag_ratio=.75
            )
        )
        self.wait(2)

        sr_eqn_2_0=SurroundingRectangle(eqn_2_0[0], color=REANLEA_PURPLE, corner_radius=.075).set_stroke(width=2)
      
        self.play(
            AnimationGroup(
                eqn_2_0[1:].animate.shift(.1*RIGHT),
                Write(sr_eqn_2_0),
                lag_ratio=.75
            )
        )

        indct_arr_1=MathTex(r"\rightarrow").set_stroke(width=3, color=[REANLEA_WARM_BLUE,REANLEA_PINK]).rotate(-PI/2).scale(.75).next_to(sr_eqn_2_0,DOWN)

        self.play(
            Create(indct_arr_1)
        )
        with RegisterFont("Cousine") as fonts:
            txt_pt_0 = Text("Pythagoras Theorem" , font=fonts[0]).next_to(indct_arr_1,DOWN).shift(.25*UP).scale(.3)
        
        self.play(
            Write(txt_pt_0)
        )

        rel_ln_0=Line().rotate(-PI/2).scale(.35).next_to(txt_pt_0,DOWN).shift(.1*UP).set_stroke(width=2, color=[REANLEA_BLUE_SKY,REANLEA_PURPLE])
        self.play(
            Create(rel_ln_0)
        )
        

        with RegisterFont("Cousine") as fonts:
            txt_pt_1 = Text("Inner Product" , font=fonts[0]).next_to(rel_ln_0,DOWN).shift(.25*UP).scale(.3)
        
        self.play(
            Write(txt_pt_1)
        )

        rel_ln_1=Line().scale(.45).next_to(txt_pt_1).shift(.1*LEFT).set_stroke(width=2, color=[REANLEA_BLUE_SKY,REANLEA_PURPLE])
        self.play(
            Create(rel_ln_1)
        )
        


        with RegisterFont("Cousine") as fonts:
            txt_pt_2 = Text("Norm/Length" , font=fonts[0]).scale(.3).next_to(rel_ln_1)
        
        self.play(
            Write(txt_pt_2)
        )

        rel_ln_2=Line().rotate(160*DEGREES).set_stroke(width=2, color=[REANLEA_BLUE_SKY,REANLEA_PURPLE]).next_to(txt_pt_2,UP).shift(1.2*LEFT+.1*DOWN)
        self.play(
            Create(rel_ln_2)
        )
        self.wait(2)
        

        arr_circ_1=MathTex(r"\circlearrowleft").set_color(REANLEA_BLUE_SKY).scale(1.25).set_stroke(width=4.5, color=[REANLEA_BLUE_LAVENDER,REANLEA_BLUE_SKY,REANLEA_PURPLE,REANLEA_GOLD]).next_to(rel_ln_0).shift(.2*RIGHT+.125*DOWN)

        self.play(
            Write(arr_circ_1)
        )
        self.wait(2)

        self.play(
            AnimationGroup(
                ReplacementTransform(dt_3.copy(),dt_2_ref),
                ReplacementTransform(ln_3.copy(),ln_2_ref),
                ReplacementTransform(dt_3_lbl.copy(),dt_2_lbl_ref)
            ),
            AnimationGroup(
                ReplacementTransform(txt_2,txt_2_ref),
                ReplacementTransform(txt_4,txt_4_ref)
            )
        )
        self.wait(2)

        with RegisterFont("Courier Prime") as fonts:
            txt_ip_0 = Text("Inner Product" , font=fonts[0]).scale(.55).set_color(REANLEA_CYAN_LIGHT).shift(4.75*RIGHT+.5*DOWN)

        dumy_ln_2=Line().rotate(-90*DEGREES).set_stroke(width=2, color=[REANLEA_BLUE_SKY,REANLEA_VIOLET]).scale(.65).shift(4.2*RIGHT+1.5*DOWN)

        bulet_1=Dot(radius=DEFAULT_DOT_RADIUS/1.25, color=REANLEA_WHITE).set_sheen(-.4,DOWN).shift(4.2*RIGHT+1.15*DOWN)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_sym_1=Text("Symmetric", font=fonts[0]).scale(.5).set_color(REANLEA_CYAN_LIGHT).next_to(bulet_1,RIGHT)

        bulet_2=Dot(radius=DEFAULT_DOT_RADIUS/1.25, color=REANLEA_WHITE).set_sheen(-.4,DOWN).next_to(bulet_1,DOWN).shift(.05*DOWN)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_sym_2=Text("Bilinear", font=fonts[0]).scale(.5).set_color(REANLEA_CYAN_LIGHT).next_to(bulet_2,RIGHT)

        bulet_3=Dot(radius=DEFAULT_DOT_RADIUS/1.25, color=REANLEA_WHITE).set_sheen(-.4,DOWN).next_to(bulet_2,DOWN).shift(.05*DOWN)

        with RegisterFont("Reenie Beanie") as fonts:
            txt_sym_3=Text("Positive Definite", font=fonts[0]).scale(.5).set_color(REANLEA_CYAN_LIGHT).next_to(bulet_3,RIGHT)

        self.play(
            ReplacementTransform(txt_pt_1.copy(),txt_ip_0)
        )
        self.wait()

        self.play(
            Create(dumy_ln_2)
        )

        self.play(
            Write(bulet_1)
        )
        self.play(
            Create(txt_sym_1)
        )
        self.wait()

        self.play(
            Write(bulet_2)
        )
        self.play(
            Create(txt_sym_2)
        )
        self.wait()

        self.play(
            Write(bulet_3)
        )
        self.play(
            Create(txt_sym_3)
        )
        self.wait(2)

        dt_0_lbl= MathTex("O").scale(.45).set_color(REANLEA_AQUA).next_to(dt_0,DL, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2)
        dt_1_lbl_1= MathTex("=","v").scale(.65).set_color(REANLEA_GOLD).next_to(dt_1_lbl, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2)

        d_ln_2=DashedDoubleArrow(
            start=dt_0.get_center(), end=dt_1.get_center(), dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).set_color_by_gradient(REANLEA_AQUA,REANLEA_GOLD).shift(.1*LEFT+.15*UP)

        ln_1_lbl=MathTex("d(O,v)").scale(.5).rotate(ln_1.get_angle()).next_to(ln_1, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER/2).shift(2*LEFT+.3*UP)
        ln_1_lbl[0][2].set_color(REANLEA_AQUA)
        ln_1_lbl[0][4].set_color(REANLEA_GOLD)

        self.play(
            AnimationGroup(
                Write(dt_0_lbl),
                Write(dt_1_lbl_1)
            )
        )

        self.play(
            AnimationGroup(
                Create(d_ln_2),
                Write(ln_1_lbl),
                lag_ratio=.5
            )
        )
        self.wait(2)

        water_mark_2=water_mark.copy()
        im_1 = ImageMobject("title_1.png").scale(.5)

        self.play(            
            AnimationGroup(
                *[FadeOut(mobj) for mobj in self.mobjects],
            ),
            FadeIn(water_mark_2),
            run_time=3
        )
        self.play(
            FadeIn(im_1),
            run_time=2
        )
        self.wait(2)

        self.play(
            FadeOut(im_1)
        )





        

        self.wait(4)


    # manim -pqh anim3.py trailer_0

    # manim -pqk anim3.py trailer_0

    # manim -sqk anim3.py trailer_0



###################################################################################################################

# Changing FONTS : import any font from Google
# some of my fav fonts: Cinzel,Kalam,Prata,Kaushan Script,Cormorant, Handlee,Monoton, Bad Script, Reenie Beanie, 
# Poiret One,Merienda,Julius Sans One,Merienda One,Cinzel Decorative, Montserrat, Cousine, Courier Prime , The Nautigal
# Marcellus SC,Contrail One,Thasadith,Spectral SC,Dongle,Cormorant SC,Comfortaa, Josefin Sans (LOVE), Fuzzy Bubbles

###################################################################################################################


                     #### completed on 15th March,2023 | 03:05am  ####


###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix" 