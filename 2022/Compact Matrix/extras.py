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



config.background_color= BLACK

###################################################################################################################


class Trailer_1(Scene):
    def construct(self):

        rect=Rectangle(height=4.5, width=8).scale(.5).shift(.75*UP+3*RIGHT).set_stroke(opacity=1, color=[REANLEA_VIOLET,REANLEA_AQUA])

        circ=Circle(radius=2).set_stroke(opacity=1, color=[REANLEA_VIOLET,REANLEA_AQUA]).move_to(.75*UP+3.25*RIGHT)


        with RegisterFont("Comic Neue") as fonts:
            txt_1=Text("Svādhyāya", font=fonts[0]).set_color_by_gradient(REANLEA_TXT_COL).scale(1).move_to(circ.get_center())

        self.play(
            Write(txt_1)
        )

        self.play(
            Create(circ)
        )



        
        transition_points = [
            [""],
            ["MTTS Values"],
            ["How to Solve it - George Pólya"],
            ["Nurturing Critical Thinking"],
            [""]
        ]

        transition_points_ref = [
            # use a list if we want multiple lines
            [""],
            ["MTTS","Values"],
            ["How to", "Solve it"],
            ["Nurturing","Thinking"],
            [""]
        ]

        ch_points = [
            # use a list if we want multiple lines
            ["","",""],
            ["Cognitive Enrichment & Creativity","Collaborative Communication","Ethical Character"],
            ["Understanding the Problem","Devising Plans & Implementation","Reflection"],
            ["Clarity & Exactitude","Pertinence","Comprehensiveness"],
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
                AnimationGroup(
                    *[FadeOut(mobj) for mobj in self.mobjects]
                ),
                run_time=2
            )
            self.wait(2)
        


        # manim -pqh extras.py Trailer_1

        # manim -pqk extras.py Trailer_1

        # manim -sqk extras.py Trailer_1


def get_solar_ray(
        factor=3, color=REANLEA_GREY, n=10
):
    ln=Line(ORIGIN,RIGHT).scale(factor).set_stroke(width=1.25,color=color)
    dt=Dot(radius=DEFAULT_DOT_RADIUS/1.25).move_to(ln.get_end()).set_color(WHITE)
    ln_grp=VGroup(ln,dt)
    angl=360/n

    rays=VGroup(
        *[
            ln_grp.copy().rotate(i*angl*DEGREES, about_point=ln.get_start())
            for i in np.linspace(0,n,n+1)
        ]
    )

    return rays


class solar_ray_test(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.25).set_z_index(-100)
        self.add(water_mark)

        rays_1=get_solar_ray().move_to(ORIGIN)
        rays_2=get_solar_ray(factor=2.5).move_to(ORIGIN).rotate(9*DEGREES)
        rays_3=get_solar_ray(factor=1.5).move_to(ORIGIN).rotate(2*9*DEGREES)
        rays_4=get_solar_ray(factor=2).move_to(ORIGIN).rotate(3*9*DEGREES)

        ray_grp=VGroup(rays_1,rays_2,rays_3,rays_4)

        dot_1=Dot(radius=0.3, color=BLACK).move_to(ORIGIN).set_sheen(-0.4,DOWN).set_z_index(3)
        glowing_circle_1=get_glowing_surround_circle(dot_1,buff_max=.15, color=WHITE).set_z_index(-20)
        glowing_circle_1_1=get_glowing_surround_circle(dot_1,buff_max=.45, color=WHITE).set_z_index(-20)
        glowing_circle_1_2=get_glowing_surround_circle(dot_1,buff_max=.75, color=WHITE).set_z_index(-20)

        self.play(
            Create(ray_grp)
        )
        self.play(
            Create(dot_1)
        )
        self.play(
            FadeIn(glowing_circle_1),
            FadeIn(glowing_circle_1_1),
            FadeIn(glowing_circle_1_2)
        )
        self.wait(4)


        # manim -pqh extras.py solar_ray_test

        # manim -sqk extras.py solar_ray_test
        
###################################################################################################################

# Playfair Display SC , Great Vibes , Merienda , Tangerine , Shalimar , Parisienne , Allura , Playball , Bad Script
# Cormorant SC, Montserrat, MonteCarlo , Kolker Brush

###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix"