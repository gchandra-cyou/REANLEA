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
from calendar import c
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
        theta_tracker_1=40

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
        dot_2=Dot(radius=0.2, color=REANLEA_VIOLET_LIGHTER).move_to(line_1.n2p(-2)).set_sheen(-0.4,DOWN).set_z_index(1).set_opacity(0.4)
        dot_1_2=dot_1.copy().move_to(line_1.n2p(-1))






        line_2=Line().move_to(UP).set_color(PURE_RED).set_z_index(4)
        line_2.save_state()

        brace_line_2=Brace(line_2, stroke_width=.01).set_color(PURE_GREEN).set_opacity(0.5)
        brace_line_2_label=brace_line_2.get_tex("1").scale(0.65).set_color(REANLEA_GREEN).shift(.25*UP)


        line_3=line_2.copy().move_to(0.5*UP+1.5*LEFT)
        line_4=line_2.copy().move_to(0.5*UP+1.5*RIGHT)
        line_3_4=Line(start=line_1.n2p(-2), end=line_1.n2p(0))



        d_line_1=DashedDoubleArrow(
            start=line_1.n2p(-2), end=line_1.n2p(0), dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA).shift(0.5*UP)

        d_line_1_label=MathTex("1+1").set_color_by_gradient(REANLEA_TXT_COL_LIGHTER).scale(0.5).next_to(d_line_1,0.5*UP)

        d_line_2=DashedDoubleArrow(
            start=line_1.n2p(-2), end=line_1.n2p(-1), dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA).shift(0.5*UP)

        




        brace_line_3_4=Brace(line_3_4, stroke_width=.01).set_color(REANLEA_GREY).set_opacity(0.5).shift(0.75*DOWN)
        brace_line_3_4_label=MathTex("2",r"\times","1").scale(0.65).set_color(REANLEA_TXT_COL_LIGHTER).next_to(brace_line_3_4,0.5*DOWN)

        brace_line_3=Brace(Line(start=line_1.n2p(-2), end=line_1.n2p(-1)), stroke_width=.01).set_color(REANLEA_YELLOW_DARKER).set_opacity(1).shift(0.75*DOWN)
        brace_line_3_label=MathTex("1").scale(0.65).set_color(REANLEA_YELLOW).next_to(brace_line_3,0.5*DOWN)





        vect_1=Arrow(start=line_1.n2p(-2),end=line_1.n2p(-1),max_tip_length_to_length_ratio=0.125, buff=0).set_color(PURE_RED).set_opacity(1)
        vect_1.set_z_index(4)
        vect_1_lbl=MathTex("u").scale(.85).next_to(vect_1,0.5*DOWN).set_color(PURE_RED)

        
        vect_2=Arrow(start=line_1.n2p(-2),end=line_1.n2p(-1),max_tip_length_to_length_ratio=0.125, buff=0).set_color(REANLEA_MAGENTA).set_opacity(0.85)
        vect_2.set_z_index(3)




        vect_1_moving=Arrow(start=line_1.n2p(-2),end=line_1.n2p(-1),max_tip_length_to_length_ratio=0.125, buff=0).set_color(REANLEA_SLATE_BLUE)
        vect_1_ref=vect_1_moving.copy()
        vect_1_moving.rotate(
            theta_tracker_1.get_value() * DEGREES, about_point=vect_1_moving.get_start()
        )

        ang=Angle(vect_1, vect_1_moving, radius=0.5, other_angle=False).set_stroke(color=PURE_GREEN, width=3).set_z_index(-1)
        ang_lbl = MathTex(r"\theta =").move_to(
            Angle(
                vect_1, vect_1_moving, radius=.85 + 3 * SMALL_BUFF, other_angle=False
            ).point_from_proportion(.5)                         # Gets the point at a proportion along the path of the VMobject.
        ).scale(.5).set_color(PURE_GREEN)

        ang_theta=DecimalNumber(unit="^o").scale(.5).set_color(PURE_GREEN)

        ang_theta_cos_demo=Variable(theta_tracker_1, MathTex(r"\theta"), num_decimal_places=2)
        ang_theta_cos=Variable(np.cos(theta_tracker_1*DEGREES), MathTex(r"cos(\theta)"), num_decimal_places=3)

        projec_line=always_redraw(
            lambda : DashedLine(start=vect_1_moving.get_end(), end=np.array((vect_1_moving.get_end()[0],line_1.n2p(0)[1],0))).set_stroke(color=REANLEA_AQUA_GREEN, width=1)
        )







        glow_dot_1=get_glowing_surround_circle(dot_1_2)
        glow_dot_2=get_glowing_surround_circle(dot_2, opacity_multiplier=0.04)




        bez_arr_1=bend_bezier_arrow().flip(DOWN).move_to(2.5*LEFT + 0.1*UP).flip(LEFT).rotate(45*DEGREES).set_z_index(-1)



        # TEXT REGION 

        so_on_txt_symbol=Text("...").move_to(0.9*DOWN+6.9*RIGHT).scale(0.5).set_color(REANLEA_GREEN)

        with RegisterFont("Fuzzy Bubbles") as fonts:
            txt_1=Text("unit vector", font=fonts[0]).scale(0.4)
            txt_1.set_color_by_gradient(REANLEA_TXT_COL).shift(3*RIGHT)
        txt_1.move_to(.75*LEFT+ 0.2*UP).rotate(20*DEGREES)

        # EQUATION REGION

        eq_1_1=MathTex("1","+","1","=","?").scale(1.3).set_color(REANLEA_PURPLE_LIGHTER)#.set_color_by_gradient(REANLEA_GREEN_AUQA,REANLEA_PURPLE)
        eq_1_2=MathTex("1","+","1","=","2").scale(1.3).set_color(REANLEA_PURPLE_LIGHTER)#.set_color_by_gradient(REANLEA_GREEN_AUQA,REANLEA_PURPLE)
        eq_1_3=MathTex("=","2",r"\times","1").scale(1.3).next_to(eq_1_2).set_color(REANLEA_PURPLE_LIGHTER)#.set_color_by_gradient(REANLEA_PURPLE,REANLEA_PINK_DARKER)

        # UPDATER REGION

        dot_1.add_updater(lambda z : z.move_to(line_1.n2p(x.get_value())))


        vect_1_moving.add_updater(
            lambda x: x.become(vect_1_ref.copy()).rotate(
                theta_tracker_1.get_value() * DEGREES, about_point=vect_1_moving.get_start()
            )
        )
        

        ang.add_updater(
            lambda x: x.become(Angle(vect_1, vect_1_moving, radius=0.5, other_angle=False).set_stroke(color=PURE_GREEN, width=3).set_z_index(-1))
        )


        ang_lbl.add_updater(
            lambda x: x.move_to(
                Angle(
                    vect_1, vect_1_moving, radius=0.85 + 13*SMALL_BUFF, other_angle=False
                ).point_from_proportion(0.5),
                aligned_edge=RIGHT
            )
        )

        ang_theta.add_updater( lambda x: x.set_value(theta_tracker_1.get_value()).next_to(ang_lbl, RIGHT))

        ang_theta_cos.add_updater(lambda v: v.tracker.set_value(np.cos(ang_theta_cos_demo.tracker.get_value()*DEGREES)))



        # GROUP REGION

        eq_1_grp=VGroup(eq_1_1,eq_1_2,eq_1_3)
        line_tick_grp_1=VGroup(line_1,zero_tick,one_tick,two_tick,three_tick,four_tick,five_tick,minus_one_tick,dot_1,dot_2)
        glow_dot_1_2_grp=VGroup(glow_dot_1,glow_dot_2)










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

        self.play(
            Uncreate(brace_line_3_4),
            Uncreate(brace_line_3_4_label),
            Unwrite(d_line_1.reverse_direction(), run_time=0.85),
            Uncreate(d_line_1_label),
            FadeOut(line_3),
            FadeOut(line_4)
        )
        self.play(
            x.animate.set_value(-1),
            run_time=1.5
        )
        self.play(
            Write(vect_1),
            lag_ratio=0.5,
            run_time=1.35
        )
        self.wait(2)
        self.play(
            Write(d_line_2),
            FadeIn(glow_dot_1_2_grp)
        )
        self.play(FadeOut(glow_dot_1_2_grp))
        self.wait(2)
        self.play(
            Write(brace_line_3),
            Write(brace_line_3_label)
        )
        self.play(
            Create(bez_arr_1)
        )
        self.play(
            Write(txt_1)
        )
        self.wait(2)
        self.play(
            Uncreate(txt_1),
            Uncreate(bez_arr_1),
            Unwrite(brace_line_3),
            Unwrite(brace_line_3_label),
            Uncreate(d_line_2),
            Uncreate(dot_1),
            Uncreate(dot_2)
        )
        self.wait(2)

        self.play(
            vect_1.animate.shift(4.87*RIGHT)
        )
        self.wait()
        self.play(
            vect_1.animate.shift(2.45*LEFT)
        )
        self.wait()
        self.play(
            vect_1.animate.shift(4.71*RIGHT)
        )
        self.wait()
        self.play(
            vect_1.animate.shift(10*LEFT)
        )
        self.wait(2)
        self.play(
            vect_1.animate.shift(8.34546*RIGHT)
        )
        self.wait(2)

        self.play(
            ReplacementTransform(vect_1.copy(), vect_2)
        )

        dot_ind_1=Dot(vect_1.get_start(), color=REANLEA_YELLOW_DARKER)
        dot_ind_2=Dot(vect_1.get_end(), color=REANLEA_YELLOW_DARKER)
        self.play(
            Indicate(dot_ind_1,scale_value=2.75, color=YELLOW_C),
            Indicate(dot_ind_2,scale_value=2.75, color=YELLOW_C)
        )
        self.play(FadeOut(dot_ind_1),FadeOut(dot_ind_2))
        
        self.wait(2)
        self.play(
            vect_1.animate.move_to(line_1.n2p(-1.5))
        )
        self.play(FadeOut(vect_2))

        wiggle_line_1=Line(line_1.n2p(-3),line_1.n2p(-2)).set_color(REANLEA_YELLOW_CREAM)
        self.play(
            Wiggle(
                wiggle_line_1
            )
        )
        self.play(FadeOut(wiggle_line_1))
        self.wait()
        self.play(
            Wiggle(zero_tick)
        )

        sgn_pos_1=MathTex("+").scale(.75).set_color(PURE_GREEN).move_to(6.5*RIGHT)
        sgn_pos_2=Circle(radius=0.2, color=PURE_GREEN).move_to(sgn_pos_1.get_center()).set_stroke(width= 1)
        sgn_pos=VGroup(sgn_pos_1,sgn_pos_2)

        sgn_neg_1=MathTex("-").scale(.75).set_color(REANLEA_YELLOW).move_to(6.5*LEFT)
        sgn_neg_2=Circle(radius=0.2, color=REANLEA_YELLOW).move_to(sgn_neg_1.get_center()).set_stroke(width= 1)
        sgn_neg=VGroup(sgn_neg_1,sgn_neg_2)

        self.play(
            Write(sgn_pos),
            Write(sgn_neg)
        )
        self.wait()
        self.play(
            vect_1.animate.move_to(line_1.n2p(-2.5))
        )
        self.wait(2)
        self.play(
            vect_1.animate.move_to(line_1.n2p(-1.5))
        )
        self.wait()
        self.play(Write(vect_1_lbl))
        self.wait()


        self.play(Write(vect_1_moving))
        self.play(theta_tracker_1.animate.set_value(40))
        self.wait()
        self.play(
            Create(ang)
        )
        self.play(
            Write(ang_lbl),
            Write(ang_theta),
            lag_ratio=0.5
        )
        self.wait(1.25)
        bra_1=always_redraw(
            lambda : BraceBetweenPoints(
                point_1=vect_1.get_start(),
                point_2=np.array((vect_1_moving.get_end()[0],0,0)),
                direction=DOWN,
                color=REANLEA_SLATE_BLUE
            ).set_stroke(width=0.1).set_z_index(5)
        )
        self.play(
            Write(projec_line),
            Create(bra_1),
            vect_1_lbl.animate(run_time=.5).shift(.5*DOWN)
        )
        self.wait(2)
        self.play(
            Write(ang_theta_cos)
        )
        self.play(
            theta_tracker_1.animate.increment_value(90),
            ang.animate(run_time=5/3).set_stroke(color=REANLEA_YELLOW, width=3),
            ang_lbl.animate(run_time=5/3).set_color(REANLEA_YELLOW),
            ang_theta.animate(run_time=5/3).set_color(REANLEA_YELLOW),
            ang_theta_cos_demo.tracker.animate.set_value(130),
            run_time=3
        )
        self.wait()
        self.play(
            theta_tracker_1.animate(rate_functions=smooth).increment_value(50),
            ang_theta_cos_demo.tracker.animate.set_value(180),
            run_time=3
        )
        self.wait()
        
        '''self.play(
            Uncreate(projec_line),
            Uncreate(bra_1)
        )'''
        self.wait(2)

        


        # manim -pqh anim1.py Scene1

        # manim -pql anim1.py Scene1

        # manim -sqk anim1.py Scene1



###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix"