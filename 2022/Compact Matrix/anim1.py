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
from func import EmojiImageMobject

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
        theta_tracker_1=ValueTracker(0.01)
        theta_tracker_2=40

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

        d_line_3=DashedDoubleArrow(
            start=line_1.n2p(-1.5)+0.4*DOWN, end=0.95*UP+RIGHT, dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).set_color_by_gradient(REANLEA_SLATE_BLUE,PURE_RED).shift(0.3*LEFT)



        




        brace_line_3_4=Brace(line_3_4, stroke_width=.01).set_color(REANLEA_GREY).set_opacity(0.5).shift(0.75*DOWN)
        brace_line_3_4_label=MathTex("2",r"\times","1").scale(0.65).set_color(REANLEA_TXT_COL_LIGHTER).next_to(brace_line_3_4,0.5*DOWN)

        brace_line_3=Brace(Line(start=line_1.n2p(-2), end=line_1.n2p(-1)), stroke_width=.01).set_color(REANLEA_YELLOW_DARKER).set_opacity(1).shift(0.75*DOWN)
        brace_line_3_label=MathTex("1").scale(0.65).set_color(REANLEA_YELLOW).next_to(brace_line_3,0.5*DOWN)





        vect_1=Arrow(start=line_1.n2p(-2),end=line_1.n2p(-1),max_tip_length_to_length_ratio=0.125, buff=0).set_color(PURE_RED).set_opacity(1)
        vect_1.set_z_index(4)
        vect_1_lbl=MathTex("u").scale(.85).next_to(vect_1,0.5*DOWN).set_color(PURE_RED)
        vect_1_lbl_vec=MathTex(r"\vec{1}").scale(.85).set_color(PURE_RED).move_to(line_1.n2p(-1.5)+ 0.85*DOWN)

        
        vect_2=Arrow(start=line_1.n2p(-2),end=line_1.n2p(-1),max_tip_length_to_length_ratio=0.125, buff=0).set_color(REANLEA_MAGENTA).set_opacity(0.85)
        vect_2.set_z_index(3)




        vect_1_moving=Arrow(start=line_1.n2p(-2),end=line_1.n2p(-1),max_tip_length_to_length_ratio=0.125, buff=0).set_color(REANLEA_SLATE_BLUE).set_z_index(3)
        vect_1_ref=vect_1_moving.copy()
        vect_1_moving.rotate(
            theta_tracker_1.get_value() * DEGREES, about_point=vect_1_moving.get_start()
        )

        ang=Angle(vect_1, vect_1_moving, radius=0.5, other_angle=False).set_stroke(color=REANLEA_WARM_BLUE, width=3).set_z_index(-1)
        ang_lbl = MathTex(r"\theta =").move_to(
            Angle(
                vect_1, vect_1_moving, radius=.85 + 3 * SMALL_BUFF, other_angle=False
            ).point_from_proportion(.5)                         # Gets the point at a proportion along the path of the VMobject.
        ).scale(.5).set_color(PURE_GREEN)

        ang_theta=DecimalNumber(unit="^o").scale(.5).set_color(PURE_GREEN)



        ang_theta_cos_demo=Variable(theta_tracker_2, MathTex(r"\theta"), num_decimal_places=2)
        ang_theta_cos=Variable(np.cos(theta_tracker_2*DEGREES), '', num_decimal_places=3).set_color(REANLEA_SLATE_BLUE).move_to(UP+3*RIGHT)
        
        ang_theta_cos_lbl_left=MathTex("u","\cdot",r"cos(\theta)").arrange(RIGHT,buff=0.2).move_to(UP +RIGHT)
        ang_theta_cos_lbl_right=MathTex("\cdot","u").arrange(RIGHT, buff=0.2).set_color(PURE_RED).move_to(UP +4.55*RIGHT)
        ang_theta_cos_lbl_left[0:2].set_color(PURE_RED)
        ang_theta_cos_lbl_left[2][0:3].set_color(REANLEA_WARM_BLUE)
        ang_theta_cos_lbl_left[2][4].set_color(PURE_GREEN)
        ang_theta_cos_grp=VGroup(ang_theta_cos_lbl_left,ang_theta_cos,ang_theta_cos_lbl_right).scale(.65)
        sur_ang_theta_cos_grp=SurroundingRectangle(ang_theta_cos_grp, color=REANLEA_TXT_COL,corner_radius=0.125, buff=0.2)





        projec_line=always_redraw(
            lambda : DashedLine(start=vect_1_moving.get_end(), end=np.array((vect_1_moving.get_end()[0],line_1.n2p(0)[1],0))).set_stroke(color=REANLEA_AQUA_GREEN, width=1)
        )







        glow_dot_1=get_glowing_surround_circle(dot_1_2)
        glow_dot_2=get_glowing_surround_circle(dot_2, opacity_multiplier=0.04)




        bez_arr_1=bend_bezier_arrow().flip(DOWN).move_to(2.5*LEFT + 0.1*UP).flip(LEFT).rotate(45*DEGREES).set_z_index(-1)


        sgn_pos_1=MathTex("+").scale(.75).set_color(PURE_GREEN).move_to(6.5*RIGHT)
        sgn_pos_2=Circle(radius=0.2, color=PURE_GREEN).move_to(sgn_pos_1.get_center()).set_stroke(width= 1)
        sgn_pos=VGroup(sgn_pos_1,sgn_pos_2)

        sgn_neg_1=MathTex("-").scale(.75).set_color(REANLEA_YELLOW).move_to(6.5*LEFT)
        sgn_neg_2=Circle(radius=0.2, color=REANLEA_YELLOW).move_to(sgn_neg_1.get_center()).set_stroke(width= 1)
        sgn_neg=VGroup(sgn_neg_1,sgn_neg_2)

        sgn_grp=VGroup(sgn_pos,sgn_neg)

        bez_arr_2=bend_bezier_arrow().rotate(-30*DEGREES).scale(0.45).set_color(REANLEA_TXT_COL).move_to(4*RIGHT+0.425*UP)



        # TEXT REGION 

        so_on_txt_symbol=Text("...").move_to(0.9*DOWN+6.9*RIGHT).scale(0.5).set_color(REANLEA_GREEN)

        with RegisterFont("Fuzzy Bubbles") as fonts:
            txt_1=Text("unit vector", font=fonts[0]).scale(0.4)
            txt_1.set_color_by_gradient(REANLEA_TXT_COL).shift(3*RIGHT)
        txt_1.move_to(.75*LEFT+ 0.2*UP).rotate(20*DEGREES)

        txt_2=MathTex("-u").set_color(REANLEA_SLATE_BLUE).move_to(4.7*RIGHT + 0.15*UP).scale(0.85)

        txt_2_vect=MathTex(r"- \vec{1}").scale(0.85).set_color(REANLEA_SLATE_BLUE).move_to(line_1.n2p(-2.55)+ 0.85*DOWN) 




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
            lambda x: x.become(Angle(vect_1, vect_1_moving, radius=0.5, other_angle=False).set_stroke(color=REANLEA_WARM_BLUE, width=3).set_z_index(-1))
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
            Write(d_line_3),
            run_time=0.35
        )
        self.play(
            TransformMatchingShapes(vect_1_lbl.copy(),ang_theta_cos_grp)
        )
        self.play(
            Uncreate(d_line_3.reverse_direction()),
            Create(sur_ang_theta_cos_grp)
        )
        self.wait(2)
        self.play(
            theta_tracker_1.animate.increment_value(90),
            ang_lbl.animate(run_time=5/3).set_color(REANLEA_YELLOW),
            ang_theta.animate(run_time=5/3).set_color(REANLEA_YELLOW),
            ang_theta_cos_demo.tracker.animate.set_value(130),
            ang_theta_cos_lbl_left[2][4].animate().set_color(REANLEA_YELLOW),
            run_time=3
        )
        self.wait()
        self.play(
            theta_tracker_1.animate(rate_functions=smooth).increment_value(50),
            ang_theta_cos_demo.tracker.animate.set_value(180),
            run_time=3
        )
        self.wait(2)
        self.play(
            Uncreate(projec_line)
        )
        self.play(
            Create(bez_arr_2),
        )
        self.play(
            Create(txt_2),
        )
        self.wait()
        self.play(
            Uncreate(bez_arr_2.reverse_direction())
        )
        self.play(
            txt_2.animate.move_to(line_1.n2p(-2.55)+ 0.85*DOWN)
        )
        self.wait(2.75)

        self.play(
            Uncreate(bra_1),
            Uncreate(ang_theta_cos_grp),
            Uncreate(sur_ang_theta_cos_grp),
            Uncreate(ang),
            Uncreate(ang_lbl),
            Uncreate(ang_theta),
            Uncreate(eq_1_grp)
        )
        self.wait(2)

        self.play(
            Transform(vect_1_lbl,vect_1_lbl_vec),
            Transform(txt_2,txt_2_vect)
        )




        self.wait(4)

        


        # manim -pqh anim1.py Scene1

        # manim -pql anim1.py Scene1

        # manim -sqk anim1.py Scene1

        # manim -sql anim1.py Scene1



###################################################################################################################


class Scene2(Scene):
    def construct(self):
        
        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)





        # PREVIOUS SCENE REGION

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

        so_on_txt_symbol=Text("...").move_to(0.9*DOWN+6.9*RIGHT).scale(0.5).set_color(REANLEA_GREEN)

        line_grp=VGroup(line_1, minus_one_tick, zero_tick,one_tick,two_tick,three_tick,four_tick,five_tick,so_on_txt_symbol)
    



        vect_1=Arrow(start=line_1.n2p(-2),end=line_1.n2p(-1),max_tip_length_to_length_ratio=0.125, buff=0).set_color(PURE_RED).set_opacity(1).set_z_index(4)
        vect_1_lbl=MathTex(r"\vec{1}").scale(.85).set_color(PURE_RED).move_to(line_1.n2p(-1.5)+ 0.85*DOWN)
        vect_1_moving=Arrow(start=line_1.n2p(-2),end=line_1.n2p(-1),max_tip_length_to_length_ratio=0.125, buff=0).set_color(REANLEA_SLATE_BLUE).set_z_index(3).rotate(180* DEGREES, about_point=line_1.n2p(-2))
        vect_1_moving_lbl=MathTex(r"- \vec{1}").scale(.85).set_color(REANLEA_SLATE_BLUE).move_to(line_1.n2p(-2.55)+ 0.85*DOWN)

        

        vect_grp=VGroup(vect_1,vect_1_lbl,vect_1_moving,vect_1_moving_lbl)

        self.add(line_grp, vect_grp)
        


        #
        #
        #


        # MOBJECT REGION
        
        

        mirror_1=get_mirror().move_to(DOWN + 4.12*LEFT)#.shift(line_1.n2p(-2.335))

        bend_bez_arrow=bend_bezier_arrow().rotate(-10*DEGREES).scale(0.75).set_color(REANLEA_BLUE_SKY).move_to(UP + 2.75*LEFT).flip(UP)
        
        indicate_line_1=Line(line_1.n2p(-2),line_1.get_end()).set_color(REANLEA_YELLOW_CREAM)
        indicate_line_1_hlgt=line_highlight(buff_max=indicate_line_1.get_length(), factor=.15, opacity_factor=.25, color=PURE_GREEN).move_to(indicate_line_1.get_center())
        

        dot_1=Dot(radius=0.1, color=REANLEA_MAGENTA).move_to(line_1.n2p(-2)).set_sheen(-0.4,DOWN).set_z_index(6)

        vect_mov_1=Arrow(start=line_1.n2p(-2),end=line_1.n2p(-0.15),buff=0, tip_length=0.25).set_color(REANLEA_RED_LIGHTER).set_opacity(1).set_z_index(7)
        
        vect_mov=always_redraw(
            lambda : Arrow(start=line_1.n2p(-2),end=dot_1.get_center(),buff=0, tip_length=0.25).set_color(REANLEA_RED_LIGHTER).set_opacity(1).set_z_index(7)
        )

        d_d_arr_1=DashedDoubleArrow(
            start=line_1.n2p(-2), end=line_1.n2p(-0.5),dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).shift(.2*UP).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA)

        dot_1_mir=Dot(radius=0.1, color=REANLEA_SLATE_BLUE_LIGHTEST).move_to(line_1.n2p(-3.5)).set_sheen(-0.4,DOWN).set_z_index(6)

        vect_mov_1_mir=Arrow(start=line_1.n2p(-2),end=line_1.n2p(-3.5),buff=0, tip_length=0.25).set_color(REANLEA_TXT_COL_DARKER).set_opacity(1).set_z_index(7)


        sgn_pos_1=MathTex("+").scale(.75).set_color(PURE_GREEN).move_to(6.5*RIGHT)
        sgn_pos_2=Circle(radius=0.2, color=PURE_GREEN).move_to(sgn_pos_1.get_center()).set_stroke(width= 1)
        sgn_pos=VGroup(sgn_pos_1,sgn_pos_2)

        sgn_neg_1=MathTex("-").scale(.75).set_color(REANLEA_YELLOW).move_to(6.5*LEFT)
        sgn_neg_2=Circle(radius=0.2, color=REANLEA_YELLOW).move_to(sgn_neg_1.get_center()).set_stroke(width= 1)
        sgn_neg=VGroup(sgn_neg_1,sgn_neg_2)

        sgn_grp=VGroup(sgn_pos,sgn_neg)




        vect_2=Arrow(start=line_1.n2p(-2),end=line_1.n2p(-1),max_tip_length_to_length_ratio=0.125, buff=0).set_color(REANLEA_CHARM).set_opacity(1).set_z_index(3)
        
        
        dot_2=Dot(radius=0.2, color=REANLEA_PURPLE).move_to(line_1.n2p(-2)).set_sheen(-0.4,DOWN).set_z_index(6)
        push_arr=Arrow(start=line_1.n2p(-2.5),end=line_1.n2p(-2.3),max_tip_length_to_length_ratio=.5, buff=0).set_color(REANLEA_YELLOW_GREEN).set_opacity(1).set_z_index(6)

        vect_3=Arrow(start=line_1.n2p(-2),end=line_1.n2p(0),tip_length=0.25, buff=0).set_color(REANLEA_YELLOW_GREEN).set_opacity(.5).set_z_index(7)

        
        d_line_1=DashedLine(line_1.n2p(-2), end=line_1.n2p(-2)+1.5*UP, stroke_width=1).set_color(PURE_RED)
        d_line_2=DashedLine(line_1.n2p(0), end=line_1.n2p(0)+1.5*UP, stroke_width=1).set_color(PURE_RED)
        
        d_d_arr_2=DashedDoubleArrow(
            start=line_1.n2p(-2), end=line_1.n2p(0),dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).shift(UP).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA)


        zo=ValueTracker(0)
        

        d_d_arr_3=DashedDoubleArrow(
            start=line_1.n2p(-2), end=line_1.n2p(-1),dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).shift(1.3*UP).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA)

        d_d_arr_3_ref=d_d_arr_3.copy()

        
        d_d_arr_3.add_updater(
            lambda x: x.become(d_d_arr_3_ref.copy()).rotate(
                zo.get_value()*DEGREES , about_point=line_1.n2p(-1)+1.3*UP
            )
        )


        d_d_arr_3_dumy=DashedDoubleArrow(
            start=line_1.n2p(-2), end=line_1.n2p(-1),dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).shift(0.3*UP).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA)


        bez=bend_bezier_arrow_indicate().flip(RIGHT).move_to(1.4*UP+ 0.5*LEFT).scale(.75).rotate(-20*DEGREES).set_color(REANLEA_TXT_COL)


        vect_1_scale_fact=ValueTracker(0)

        d_d_arr_4=DashedDoubleArrow(
            start=line_1.n2p(-1.2)+ DOWN, end=line_1.n2p(2) +  UP ,dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).shift(1.3*UP).set_color_by_gradient(REANLEA_RED_LIGHTER,REANLEA_GREEN_AUQA).rotate(5*DEGREES).set_z_index(10)


        value=DecimalNumber().set_color_by_gradient(REANLEA_WARM_BLUE).set_sheen(-0.1,LEFT).move_to(line_1.n2p(2) + 2.9* UP ).scale(0.85)

        value.add_updater(
            lambda x : x.set_value(1+vect_1_scale_fact.get_value())
        )

        txt_vec_1_val=MathTex(r"\cdot",r"\vec{1}").scale(.95).set_color(PURE_RED).move_to(line_1.n2p(2.3) + 2.975* UP)

        vect_1_scale=VGroup(value,txt_vec_1_val)

        sr_rec_vec_1_scale=SurroundingRectangle(vect_1_scale, color=REANLEA_BLUE_DARKER, corner_radius=0.15, buff=.25)
        
        d_line_3=DashedLine(line_1.n2p(1.77), end=line_1.n2p(1.77)+.3*UP, stroke_width=1).set_color(PURE_RED)

        d_d_arr_5=DashedDoubleArrow(
            start=line_1.n2p(-2), end=line_1.n2p(1.77), dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).shift(0.3*UP).set_color_by_gradient(REANLEA_YELLOW_GREEN)

        d_d_arr_6=DashedDoubleArrow(
            start=line_1.n2p(-2), end=line_1.n2p(-1), dash_length=2.0,stroke_width=2, 
            max_tip_length_to_length_ratio=0.015, buff=10
        ).shift(0.6*UP).set_color_by_gradient(PURE_RED)

        d_d_arr_5_lbl=MathTex("x").set_color(REANLEA_YELLOW_GREEN).scale(0.5).next_to(d_d_arr_5, 0.35*UP)

        x_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_GREEN),
            MathTex("x").scale(.5),
        )
        x_tick[0].move_to(line_1.n2p(1.77))
        x_tick[1].next_to(x_tick[0], DOWN).set_color(REANLEA_YELLOW_GREEN)
        x_tick.set_z_index(3)


        vect_4=Arrow(start=line_1.n2p(-2),end=line_1.n2p(1.77),tip_length=0.25, buff=0).set_color(REANLEA_YELLOW_GREEN).set_opacity(.5).set_z_index(7)

        vect_5=Arrow(start=line_1.n2p(-2),end=line_1.n2p(0),tip_length=0.25, buff=0).set_color(REANLEA_WARM_BLUE).set_z_index(7)

        vect_6=Arrow(start=line_1.n2p(-2),end=line_1.n2p(1),tip_length=0.25, buff=0).set_color(REANLEA_AQUA_GREEN).set_z_index(8)

        
        d_line_4=Line(LEFT, RIGHT, stroke_width=2).set_color(REANLEA_BLUE_LAVENDER).move_to(1.65*UP + 5*RIGHT)

        z1=ValueTracker(0)

        vect_7=Arrow(start=line_1.n2p(0),end=line_1.n2p(3),tip_length=0.25, buff=0).set_color(REANLEA_AQUA_GREEN).set_z_index(8)


        vect_7_ref=vect_7.copy()

        vect_7.add_updater(
            lambda x : x.become(vect_7_ref.copy()).rotate(
                z1.get_value()*DEGREES, about_point=line_1.n2p(0)
            )
        )

        

        # LABEL REGION

        vect_1_mir_lbl=MathTex(r"-(-\vec{1})").scale(.85).set_color(PURE_RED).move_to(line_1.n2p(-1.5)+ 1.3*DOWN)
        vect_2_mir_lbl=MathTex("=",r"-(-\vec{1})").scale(.85).set_color(PURE_RED).move_to(line_1.n2p(-1.15)+ 1.3*DOWN)
        vect_2_lbl=MathTex(r"\vec{1}").scale(.85).set_color(REANLEA_CHARM).move_to(line_1.n2p(-0.5)+ 0.85*DOWN)
        vect_3_lbl=MathTex(r"\vec{2}").scale(.85).set_color(REANLEA_YELLOW_GREEN).move_to(line_1.n2p(-1)+ 0.9*UP)


        


        
        
        # TEXT REGION 

        with RegisterFont("Fuzzy Bubbles") as fonts:
            txt_mir=Text("MIRROR", font=fonts[0]).scale(0.4)
            txt_mir.set_color_by_gradient(REANLEA_YELLOW_LIGHTER).shift(3*RIGHT)
        txt_mir.move_to(1.75*UP + 1.9*LEFT)



        with RegisterFont("Cousine") as fonts:
            text_1 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Scaling Factor",
            )]).scale(0.24).set_color(REANLEA_GREY)

        text_1.move_to(1.7*UP+RIGHT)

        txt_blg_1=MathTex(r"\in", r"\mathbb{R}").set_color(REANLEA_TXT_COL).scale(0.7).move_to(1.35*UP+1.1*RIGHT)
        txt_blg_1[0].scale(0.65)
        txt_blg_1[1].set_color(REANLEA_BLUE_SKY)


        # EQUATION REGION

        vect_3_lbl_eqn_dumy=MathTex(r"\vec{2}","=",r"2 \cdot \vec{1}").scale(.85).set_color(REANLEA_YELLOW_GREEN).move_to(line_1.n2p(-1)+ 2.9*UP)
        vect_3_lbl_eqn=MathTex(r"\vec{2}","=",r"2 \cdot \vec{1}").scale(.85).set_color(REANLEA_YELLOW_GREEN).move_to(line_1.n2p(-1)+ 2.9*UP)
        vect_3_lbl_eqn.shift(vect_3_lbl.get_center()+UP - vect_3_lbl_eqn_dumy[0].get_center())



        vec_1_2_eqn_grp=VGroup(vect_1_lbl,vect_2_lbl,vect_3_lbl_eqn)

        eq_1=MathTex(r"\vec{1}","+",r"\vec{1}","=",r"\vec{2}","=",r"2 \cdot \vec{1}").scale(.85).set_color(REANLEA_PURPLE_LIGHTER).move_to(3.25*UP)
        eq_2=MathTex(r"\vec{1}","+",r"\vec{1}","=",r"2 \cdot \vec{1}").scale(.85).set_color(REANLEA_PURPLE_LIGHTER).move_to(3.25*UP)
        
        stripe_1=get_stripe(factor=0.05,buff_max=2.4).move_to(2.925*UP)


        vect_4_lbl_eqn=MathTex(r"\vec{x}","=",r"x \cdot \vec{1}").scale(0.85).move_to(line_1.n2p(-1)+ 2.9*UP).set_color(PURE_RED)
        vect_4_lbl_eqn.shift(vect_3_lbl.get_center()+UP - vect_3_lbl_eqn_dumy[0].get_center())
        vect_4_lbl_eqn[0].set_color(PURE_GREEN)
        vect_4_lbl_eqn[2][0].set_color(PURE_GREEN)

        vect_5_lbl_eqn=MathTex(r"\vec{2}","=",r"2 \cdot \vec{1}").scale(0.85).move_to(UP).set_color(PURE_RED)
        vect_5_lbl_eqn[0].set_color(REANLEA_WARM_BLUE)
        vect_5_lbl_eqn[2][0].set_color(REANLEA_WARM_BLUE)

        vect_6_lbl_eqn=MathTex(r"\vec{3}","=",r"3 \cdot \vec{1}").scale(0.85).move_to(UP).set_color(PURE_RED)
        vect_6_lbl_eqn[0].set_color(REANLEA_AQUA_GREEN)
        vect_6_lbl_eqn[2][0].set_color(REANLEA_AQUA_GREEN)

        eq_3=MathTex(r"\vec{2}","+",r"\vec{3}","=",r"2 \cdot \vec{1}","+",r"3 \cdot \vec{1}").scale(0.85).move_to(.5*UP).set_color(PURE_RED)
        eq_3[0].set_color(REANLEA_WARM_BLUE)
        eq_3[2].set_color(REANLEA_AQUA_GREEN)
        eq_3[4][0].set_color(REANLEA_WARM_BLUE)
        eq_3[6][0].set_color(REANLEA_AQUA_GREEN)


        eq_4=MathTex(r"\vec{2}","+",r"\vec{3}","=",r"2 \cdot \vec{1}","+",r"3 \cdot \vec{1}","=",r"(2+3) \cdot \vec{1}").scale(0.85).set_color(PURE_RED)
        eq_4[0].set_color(REANLEA_WARM_BLUE)
        eq_4[2].set_color(REANLEA_AQUA_GREEN)
        eq_4[4][0].set_color(REANLEA_WARM_BLUE)
        eq_4[6][0].set_color(REANLEA_AQUA_GREEN)
        eq_4[8][0].set_color(REANLEA_GREY_DARKER)
        eq_4[8][4].set_color(REANLEA_GREY_DARKER)
        eq_4[8][1].set_color(REANLEA_WARM_BLUE)
        eq_4[8][3].set_color(REANLEA_AQUA_GREEN)
        eq_4.shift(eq_3[0].get_center() - eq_4[0].get_center())


        eq_5=MathTex(r"\vec{2}","+",r"\vec{3}","=",r"2 \cdot \vec{1}","+",r"3 \cdot \vec{1}","=",r"5 \cdot \vec{1}").scale(0.85).set_color(PURE_RED)
        eq_5[0].set_color(REANLEA_WARM_BLUE)
        eq_5[2].set_color(REANLEA_AQUA_GREEN)
        eq_5[4][0].set_color(REANLEA_WARM_BLUE)
        eq_5[6][0].set_color(REANLEA_AQUA_GREEN)
        eq_5[8][0].set_color(REANLEA_VIOLET_LIGHTER)
        eq_5.shift(eq_4[0].get_center() - eq_5[0].get_center())


        eq_6=MathTex(r"\vec{2}","+",r"\vec{3}","=",r"2 \cdot \vec{1}","+",r"3 \cdot \vec{1}","=",r"\vec{5}").scale(0.85).set_color(PURE_RED)
        eq_6[0].set_color(REANLEA_WARM_BLUE)
        eq_6[2].set_color(REANLEA_AQUA_GREEN)
        eq_6[4][0].set_color(REANLEA_WARM_BLUE)
        eq_6[6][0].set_color(REANLEA_AQUA_GREEN)
        eq_6[8].set_color(REANLEA_VIOLET_LIGHTER)
        eq_6.shift(eq_5[0].get_center() - eq_6[0].get_center())



        eq_7=MathTex(r"\vec{5}","=",r"5 \cdot \vec{1}").scale(0.85).set_color(PURE_RED)
        eq_7[2][0].set_color(REANLEA_VIOLET_LIGHTER)
        eq_7[0].set_color(REANLEA_VIOLET_LIGHTER)
        eq_7.move_to(1.5*UP+5*RIGHT)


        eq_8=MathTex(r"\vec{2}","-",r"\vec{3}","=",r"2 \cdot \vec{1}","+",r"(-3) \cdot \vec{1}","=",r"\vec{-1}").scale(0.85).set_color(PURE_RED).move_to(0.5*UP)
        eq_8[0].set_color(REANLEA_WARM_BLUE)
        eq_8[2].set_color(REANLEA_AQUA_GREEN)
        eq_8[4][0].set_color(REANLEA_WARM_BLUE)
        eq_8[6][1:3].set_color(REANLEA_AQUA_GREEN)
        eq_8[8].set_color(REANLEA_VIOLET_LIGHTER)
        eq_8.shift(eq_5[0].get_center() - eq_6[0].get_center())


        eq_9=MathTex(r"\vec{-3}","=",r"(-3) \cdot \vec{1}").scale(0.85).move_to(2*UP+5.175*RIGHT).set_color(PURE_RED)
        eq_9[0].set_color(REANLEA_AQUA_GREEN)
        eq_9[2][1:3].set_color(REANLEA_AQUA_GREEN)


        eq_10=MathTex(r"\vec{-1}","=",r"(-1) \cdot \vec{1}").scale(0.85).set_color(PURE_RED)
        eq_10[2][1:3].set_color(REANLEA_VIOLET_LIGHTER)
        eq_10[0].set_color(REANLEA_VIOLET_LIGHTER)
        eq_10.move_to(1.25*UP+5.175*RIGHT)






        # GROUP REGION

        grp_1=VGroup(vect_4_lbl_eqn, text_1,txt_blg_1, bez)
        grp_2=VGroup(vect_5_lbl_eqn,vect_6_lbl_eqn)
        grp_3=VGroup(eq_4, vect_4_lbl_eqn)
        grp_4=VGroup(eq_3,eq_6)

        # EXTRAS

        sr_grp_4=SurroundingRectangle(grp_4, color=REANLEA_WELDON_BLUE, buff=0.25, corner_radius=.125).move_to(0.5*UP)

        sr_eq_8=SurroundingRectangle(eq_8, color=REANLEA_WELDON_BLUE, buff=0.25, corner_radius=.125).move_to(0.5*UP)


        
        


        # PLAY REGION
        self.wait(1.5)
        self.play(
            Wiggle(zero_tick)
        )
        self.play(
            Write(mirror_1)
        )
        self.wait()
        self.play(
            Create(bend_bez_arrow)
        )
        self.play(
            Write(txt_mir)
        )
        self.wait(2)
        self.play(
            mirror_1.animate.flip(DOWN).move_to(DOWN + 3.88*LEFT),
        )
        self.play(
            Write(vect_1_mir_lbl)
        )
        self.wait()
        self.play(
            TransformMatchingShapes(vect_1_mir_lbl,vect_2_mir_lbl)
        )
        self.wait(2)

        self.play(
            FadeIn(indicate_line_1_hlgt),
            Create(dot_1)
        )
        self.wait(2.5)
        self.play(
            dot_1.animate.move_to(line_1.n2p(-0.15))
        )


        self.play(
            Write(vect_mov_1),
            vect_1.animate.set_opacity(0.3)
        )
        self.play(
            FadeIn(vect_mov),
        )
        self.play(
            FadeOut(vect_mov_1)
        )


        self.play(
            dot_1.animate.move_to(line_1.n2p(2.5))
        )
        self.play(
            dot_1.animate.move_to(line_1.n2p(0.5))
        )
        self.play(
            dot_1.animate.move_to(line_1.n2p(3))
        )
        self.play(
            dot_1.animate.move_to(line_1.n2p(0))
        )
        self.play(
            dot_1.animate.move_to(line_1.n2p(-0.5))
        )
        self.wait(1.5)
        self.play(
            Write(d_d_arr_1)
        )
        self.wait(2)
        self.play(
            d_d_arr_1.animate.shift(3*LEFT)
        )
        self.wait()
        self.play(
            Write(dot_1_mir)
        )
        self.wait()
        self.play(
            Write(vect_mov_1_mir),
            vect_1_moving.animate.set_opacity(0.3)
        )
        self.wait()

        self.play(
            Write(sgn_grp)
        )
        self.play(
            Wiggle(sgn_pos)
        )
        self.play(
            Wiggle(sgn_neg)
        )
        self.wait(2)

        self.play(
            Indicate(mirror_1)
        )
        self.wait()
        self.play(
            Wiggle(vect_mov, color=REANLEA_RED_LIGHTER),
            Wiggle(vect_mov_1_mir, color=REANLEA_TXT_COL_DARKER)
        )
        self.wait(2)

        uncrt_grp_1=VGroup(mirror_1,bend_bez_arrow,txt_mir,indicate_line_1_hlgt,d_d_arr_1,vect_mov,vect_mov_1_mir, dot_1,dot_1_mir, vect_2_mir_lbl)

        self.play(
            Uncreate(uncrt_grp_1),
            lag_ratio=.1,
        )
        self.play(
            vect_1.animate.set_opacity(1),
            vect_1_moving.animate.set_opacity(1),
        )
        self.wait(2)



        ### PART-II ###

        self.add(vect_2)
        self.play(
            vect_2.animate.shift(2*1.75*RIGHT)
        )
        self.wait()
        self.play(
            ShowPassingFlash(Underline(vect_1, color=PURE_RED)),
            ShowPassingFlash(Underline(vect_2, color=REANLEA_CHARM))
        )
        self.wait(2)
        self.play(
            vect_2.animate.shift(2*0.75*LEFT)
        )
        self.wait()
        self.play(
            Write(vect_2_lbl)
        )
        self.wait(2)

        self.play(
            Create(dot_2)
        )
        self.wait()
        self.play(
            push_arr.animate.move_to(line_1.n2p(-2.2))
        )
        self.play(
            dot_2.animate.move_to(line_1.n2p(-1))
        )
        self.play(
            FadeOut(push_arr)
        )
        self.wait()

        push_arr.move_to(line_1.n2p(-1.4))
        self.play(
            push_arr.animate.move_to(line_1.n2p(-1.2))
        )
        self.play(
            dot_2.animate.move_to(line_1.n2p(0))
        )
        self.play(
            FadeOut(push_arr)
        )
        self.wait()


        self.play(
            dot_2.animate.set_z_index(2),
        )
        self.play(
            Write(vect_3)
        )
        self.wait()
        self.play(
            vect_3.animate.shift(2*0.25*UP).set_opacity(1)
        )

        self.play(
            Write(vect_3_lbl)
        )
        self.wait()


        self.play(
            Write(d_line_1),
            Write(d_line_2)
        )
        self.play(
            vect_3_lbl.animate.shift(UP),
            Write(d_d_arr_2)
        )
        self.wait()

        self.play(
            Write(d_d_arr_3_dumy)
        )
        self.wait()
        self.play(
            d_d_arr_3_dumy.animate.shift(UP)
        )
        self.wait()
        self.add(d_d_arr_3)
        self.play(
            d_d_arr_3_dumy.animate.set_opacity(0)
        )

        self.play(
            zo.animate.set_value(-180)
        )
        self.wait(2)

        self.play(
            TransformMatchingShapes(vect_3_lbl, vect_3_lbl_eqn)
        )
        self.wait()

        self.play(
            Indicate(vect_3_lbl_eqn[2], color=REANLEA_AQUA_GREEN)
        )
        self.wait()

        self.play(
            Indicate(vect_3_lbl_eqn[2][0], color=REANLEA_BLUE_SKY)
        )
        self.wait()

        self.play(
            Create(bez)
        )
        self.play(
            Write(text_1)
        )
        self.wait()
        self.play(
            Write(txt_blg_1)
        )
        self.wait(2)

        self.play(
            Write(d_d_arr_4)
        )
        self.play(
            Write(vect_1_scale)
        )
        self.play(
            Write(sr_rec_vec_1_scale)
        )
        self.wait(2)


        vect_1.add_updater(
            lambda z : z.become(
                Arrow(start=line_1.n2p(-2),end=np.array((line_1.n2p(-1)[0]*(1-vect_1_scale_fact.get_value()),line_1.n2p(-1)[1],line_1.n2p(-1)[2])),max_tip_length_to_length_ratio=0.125, buff=0).set_color(PURE_RED).set_opacity(1).set_z_index(4)
            )
        )
        self.play(
            vect_1_scale_fact.animate.set_value(1),  
        )
        self.wait()

        self.play(
            TransformMatchingShapes(vec_1_2_eqn_grp.copy(),eq_1),
            Write(stripe_1)
        )
        self.play(
            ReplacementTransform(eq_1,eq_2)
        )
        self.wait()

        self.play(
            vect_1_scale_fact.animate.set_value(0),  
        )
        self.wait()

        
        self.play(
            Uncreate(sr_rec_vec_1_scale),
            Uncreate(vect_1_scale),
            Uncreate(d_d_arr_4)
        )
        self.wait()

        self.play(
            Uncreate(vect_2),
            Uncreate(vect_2_lbl),
            Uncreate(d_line_1),
            Uncreate(d_line_2),
            Uncreate(d_d_arr_2),
            Uncreate(d_d_arr_3),
            vect_3.animate.shift(0.75*UP),
            dot_2.animate.shift(2*1.77*RIGHT).scale(0.5).set_z_index(7)
        )
        self.wait()
        
        self.play(
            Create(d_line_3)
        )
        self.play(
            Write(d_d_arr_5)
        )
        self.play(
            Write(d_d_arr_6)
        )
        self.wait()

        self.play(
            Create(d_d_arr_5_lbl)
        )
        self.wait()

        self.play(
            TransformMatchingShapes(d_d_arr_5_lbl.copy(), x_tick)
        )
        self.wait()

        self.play(
            ReplacementTransform(vect_3,vect_4),
            ReplacementTransform(vect_3_lbl_eqn,vect_4_lbl_eqn),
            vect_1.animate.set_opacity(0.25)
        )
        self.wait()

        self.play(
            Indicate(vect_4_lbl_eqn[0], color=REANLEA_PINK_LIGHTER),
            Indicate(vect_4_lbl_eqn[2][0], color=REANLEA_PINK_LIGHTER)
        )
        self.wait(2)

        self.play(
            Uncreate(d_d_arr_5),
            Unwrite(d_d_arr_5_lbl),
            Uncreate(d_line_3),
            Uncreate(vect_4),
            d_d_arr_6.animate.shift(0.9*DOWN)
        )

        self.play(
            Uncreate(vect_1),
            Uncreate(vect_1_lbl),
            grp_1.animate.shift(3*LEFT+.5*UP)
        )
        self.wait()
        self.play(
            ReplacementTransform(vect_4_lbl_eqn.copy(), vect_5_lbl_eqn),
            dot_2.animate.move_to(line_1.n2p(0))
        )
        self.wait(1.5)

        self.play(
            Create(vect_5)
        )
        self.wait(1.5)

        self.play(
            vect_5_lbl_eqn.animate.move_to(2.5*UP+5*RIGHT)
        )
        self.wait(2)

        self.play(
            ReplacementTransform(vect_4_lbl_eqn.copy(), vect_6_lbl_eqn),
            dot_2.animate.move_to(line_1.n2p(1))
        )
        self.wait(1.5)

        self.play(
            Create(vect_6)
        )
        self.wait(1.5)

        self.play(
            vect_6_lbl_eqn.animate.move_to(2*UP+5*RIGHT)
        )
        self.wait(2)

        self.play(
            TransformMatchingShapes(grp_2.copy(), eq_3),
            vect_6.animate.shift(4*RIGHT),
            dot_2.animate.move_to(line_1.n2p(3))
        )
        self.wait(2)

        self.play(
            Write(eq_4)
        )
        self.wait()

        self.play(
            ReplacementTransform(eq_4,eq_5)
        )
        self.wait()

        self.play(
            Indicate(vect_4_lbl_eqn),
            Indicate(eq_5[8])
        )
        self.wait()
        self.play(
            ReplacementTransform(grp_3.copy(), eq_7)
        )
        self.wait()
        self.play(
            ReplacementTransform(eq_5,eq_6)
        )
        self.play(
            grp_4.animate.move_to(0.5*UP)
        )
        self.play(
            Create(sr_grp_4)
        )
        self.wait()

        self.play(
            eq_7.animate.shift(0.1*DOWN),
            Create(d_line_4)
        )
        self.add(vect_7)
        self.play(
            FadeOut(vect_6)
        )

        self.play(
            z1.animate.set_value(180),
            dot_2.animate.move_to(line_1.n2p(-3)),
            ReplacementTransform(grp_4,eq_8),
            ReplacementTransform(eq_7,eq_10),
            ReplacementTransform(sr_grp_4,sr_eq_8),
            ReplacementTransform(vect_6_lbl_eqn,eq_9)
        )
        self.wait()

        self.play(
            dot_2.animate.move_to(line_1.n2p(-2)).set_z_index(10)
        )
        self.wait(2)

        z2=ValueTracker(0)
        value2=DecimalNumber().set_color_by_gradient(REANLEA_PINK_LIGHTER).move_to(line_1.n2p(-1) + .5* UP ).scale(0.65)

        value2.add_updater(
            lambda x : x.set_value(z2.get_value())
        )

        self.play(
            Write(value2)
        )

        dot_2.set_z_index(10)

        self.play(
            dot_2.animate.move_to(line_1.n2p(0)),
            z2.animate.set_value(2)
        )
        self.wait()
        z2.set_value(0)
        self.wait()

        self.play(
            dot_2.animate.move_to(line_1.n2p(-3)),
            z2.animate.set_value(-3)
        )
        self.wait(2)

        self.play(
            Uncreate(dot_2),
            Unwrite(value2),
            Unwrite(eq_9),
            Uncreate(eq_8),
            Uncreate(sr_eq_8),
            Uncreate(eq_10),
            Uncreate(vect_7),
            Uncreate(d_line_4),
            Uncreate(vect_5),
            Uncreate(vect_5_lbl_eqn),
            Uncreate(vect_1_moving),
            Uncreate(vect_1_moving_lbl),
            Unwrite(d_d_arr_6),
            FadeOut(eq_2),
            FadeOut(stripe_1),
            run_time=2.5
        )
        self.wait()






        self.wait(4)





        



        # manim -pqh anim1.py Scene2

        # manim -pql anim1.py Scene2

        # manim -sqk anim1.py Scene2

        # manim -sql anim1.py Scene2





###################################################################################################################


class Scene3(Scene):
    def construct(self):
        
        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)





        # PREVIOUS SCENE REGION

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

        x_tick = VGroup(
            Line(0.15 * UP, 0.15 * DOWN, stroke_width=2.0, color=REANLEA_GREEN),
            MathTex("x").scale(.5),
        )
        x_tick[0].move_to(line_1.n2p(1.77))
        x_tick[1].next_to(x_tick[0], DOWN).set_color(REANLEA_YELLOW_GREEN)
        x_tick.set_z_index(3)

        so_on_txt_symbol=Text("...").move_to(0.9*DOWN+6.9*RIGHT).scale(0.5).set_color(REANLEA_GREEN)

        line_grp=VGroup(line_1, minus_one_tick, zero_tick,one_tick,two_tick,three_tick,four_tick,five_tick,x_tick,so_on_txt_symbol)



        sgn_pos_1=MathTex("+").scale(.75).set_color(PURE_GREEN).move_to(6.5*RIGHT)
        sgn_pos_2=Circle(radius=0.2, color=PURE_GREEN).move_to(sgn_pos_1.get_center()).set_stroke(width= 1)
        sgn_pos=VGroup(sgn_pos_1,sgn_pos_2)

        sgn_neg_1=MathTex("-").scale(.75).set_color(REANLEA_YELLOW).move_to(6.5*LEFT)
        sgn_neg_2=Circle(radius=0.2, color=REANLEA_YELLOW).move_to(sgn_neg_1.get_center()).set_stroke(width= 1)
        sgn_neg=VGroup(sgn_neg_1,sgn_neg_2)

        sgn_grp=VGroup(sgn_pos,sgn_neg)



        vect_4_lbl_eqn=MathTex(r"\vec{x}","=",r"x \cdot \vec{1}").scale(0.85).move_to(line_1.n2p(-1)+ 2.9*UP).set_color(PURE_RED)
        vect_3_lbl=MathTex(r"\vec{2}").scale(.85).set_color(REANLEA_YELLOW_GREEN).move_to(line_1.n2p(-1)+ 0.9*UP)
        vect_3_lbl_eqn_dumy=MathTex(r"\vec{2}","=",r"2 \cdot \vec{1}").scale(.85).set_color(REANLEA_YELLOW_GREEN).move_to(line_1.n2p(-1)+ 2.9*UP)
        vect_4_lbl_eqn.shift(vect_3_lbl.get_center()+UP - vect_3_lbl_eqn_dumy[0].get_center())
        vect_4_lbl_eqn[0].set_color(PURE_GREEN)
        vect_4_lbl_eqn[2][0].set_color(PURE_GREEN)

        with RegisterFont("Cousine") as fonts:
            text_1 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Scaling Factor",
            )]).scale(0.24).set_color(REANLEA_GREY)

        text_1.move_to(1.7*UP+RIGHT)

        txt_blg_1=MathTex(r"\in", r"\mathbb{R}").set_color(REANLEA_TXT_COL).scale(0.7).move_to(1.35*UP+1.1*RIGHT)
        txt_blg_1[0].scale(0.65)
        txt_blg_1[1].set_color(REANLEA_BLUE_SKY)


        bez=bend_bezier_arrow_indicate().flip(RIGHT).move_to(1.4*UP+ 0.5*LEFT).scale(.75).rotate(-20*DEGREES).set_color(REANLEA_TXT_COL)

        grp_1=VGroup(vect_4_lbl_eqn, text_1,txt_blg_1,bez).shift(3*LEFT+.5*UP)

        prv_grp=VGroup(line_grp,sgn_grp, grp_1)

        self.add(prv_grp)


        #
        #
        #






        # MOBJECT REGION

        vec_1= Arrow(start=LEFT,end=RIGHT,tip_length=0.125, max_stroke_width_to_length_ratio=2, buff=0)
        vec_1.set_color(REANLEA_WARM_BLUE).rotate(55*DEGREES).shift(1.7347*RIGHT)


        dts= VGroup(*[Dot().shift(i*0.15*RIGHT*np.random.uniform(-6,6)) for i in range(-15,15)])
        dts.shift(DOWN).set_color_by_gradient(REANLEA_BLUE, PURE_GREEN, REANLEA_GREY_DARKER,REANLEA_VIOLET,REANLEA_AQUA_GREEN).set_z_index(5)


        l_1=Line().rotate(PI/2).set_stroke(width=5, color=(PURE_GREEN,REANLEA_BLUE_SKY)).scale(0.5).shift(0.5*RIGHT+ 0.5*UP)
        


        # TEXT REGION

        with RegisterFont("Cousine") as fonts:
            txt_1 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "Space of Vectors",
            )]).scale(0.3).set_color(REANLEA_GREY).move_to(1.05*UP+2.55*RIGHT).rotate(-35*DEGREES)
            
            



        with RegisterFont("Kalam") as fonts:
            txt_2 = VGroup(*[Text(x, font=fonts[0], weight=BOLD) for x in (
                "I. Vector Addition",
                "II. Scalar Multiplication"
            )]).scale(0.5).set_color(REANLEA_GREY).arrange_submobjects(1.5*DOWN).shift(0.5*UP)

            txt_2[1].shift(0.3*RIGHT)#.shift(txt_2[0][0].get_center()-txt_2[1][0].get_center())


        r_1=MathTex("\mathbb{R}").set_color(PURE_GREEN).scale(0.7).move_to(5.25*RIGHT+0.5*DOWN)

        r_1_copy=MathTex("\mathbb{R}").set_color(PURE_GREEN).scale(0.7).shift(0.5*RIGHT+ 1.25*UP)

        txt_blg_1_copy=MathTex("\mathbb{R}").set_color(REANLEA_BLUE_SKY).scale(0.7).shift(0.5*RIGHT+.25*DOWN)


        with RegisterFont("Cousine") as fonts:
            txt_vs = Text(" - Vector Space", font=fonts[0]).scale(0.3).set_color(PURE_GREEN).next_to(r_1_copy,RIGHT).shift(LEFT)
            txt_fld= Text("- Scalar Field", font=fonts[0]).scale(0.3).set_color(REANLEA_BLUE_SKY).next_to(txt_blg_1_copy, RIGHT).shift(LEFT)

            txt_3 = VGroup(*[Text(x, font=fonts[0]) for x in (
                "each of its points",
                "represents a vector"
            )]).scale(0.2).set_color(REANLEA_GREY).arrange_submobjects(.25*DOWN).move_to(1.75*UP+3.75*RIGHT)


        # INDICATE REGION

        sr_txt_2=SurroundingRectangle(txt_2, color=REANLEA_WELDON_BLUE, buff=0.25, corner_radius=0.15).scale(0.75).shift(2.5*UP+RIGHT)
        sr_bez_1=get_surround_bezier(txt_3).set_color(REANLEA_GREY_DARKER)

        # GROUP REGION

        grp_1=VGroup(vec_1,txt_1, txt_3,sr_bez_1)
        r_grp=VGroup(r_1_copy,txt_blg_1_copy,l_1)


        





        # PLAY REGION 

        self.wait(2)

        self.play(
            Indicate(line_1),
            color=PURE_RED,
            run_time=2
        )
        self.play(
            Write(vec_1)
        )
        self.play(
            Write(txt_1)
        )
        
        self.play(
            Write(r_1),
        )
        
        self.play(
            Write(txt_3),
            Create(dts),
            run_time=3,
            lag_ratio=.1
        )
        
        
        self.play(
            Create(sr_bez_1),
            lag_ratio=.95
        )
        self.wait()

        
        self.play(
            grp_1.animate.shift(2*RIGHT),
            AddTextLetterByLetter(txt_2[0])
        )
        
        self.wait(1.35)
        self.play(
            AddTextLetterByLetter(txt_2[1])
        )
        self.wait(2)

        self.play(
            txt_2.animate.scale(0.75).shift(2.5*UP+RIGHT)
        )

        self.play(
            Write(sr_txt_2)
        )
        self.play(
            TransformMatchingShapes(r_1.copy(),r_1_copy)
        )
        self.play(
            Create(l_1.reverse_direction())
        )
        self.play(
            TransformMatchingShapes(txt_blg_1[1].copy(),txt_blg_1_copy)
        )
        self.wait()
        self.play(
            r_grp.animate.shift(LEFT)
        )

        self.play(
            Write(txt_vs)
        )
        self.play(
            Write(txt_fld)
        )




        self.wait(4)













        # manim -pqh anim1.py Scene3

        # manim -pql anim1.py Scene3

        # manim -sqk anim1.py Scene3

        # manim -sql anim1.py Scene3



###################################################################################################################


# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix"