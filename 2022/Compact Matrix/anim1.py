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
        
        d_line_1=DashedLine(start=2*LEFT, end=2*RIGHT, stroke_width=2).shift(line_1.n2p(-2)).rotate(PI/2)
        d_line_1.set_color_by_gradient(REANLEA_AQUA_GREEN,REANLEA_MAGENTA_LIGHTER,REANLEA_SLATE_BLUE_LIGHTER,REANLEA_TXT_COL_LIGHTER,REANLEA_VIOLET_LIGHTER).set_z_index(-5)

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
        
        


        # LABEL REGION

        vect_1_mir_lbl=MathTex(r"-(-\vec{1})").scale(.85).set_color(PURE_RED).move_to(line_1.n2p(-1.5)+ 1.3*DOWN)
        vect_2_mir_lbl=MathTex("=",r"-(-\vec{1})").scale(.85).set_color(PURE_RED).move_to(line_1.n2p(-1.15)+ 1.3*DOWN)
        
        # TEXT REGION 

        with RegisterFont("Fuzzy Bubbles") as fonts:
            txt_mir=Text("MIRROR", font=fonts[0]).scale(0.4)
            txt_mir.set_color_by_gradient(REANLEA_YELLOW_LIGHTER).shift(3*RIGHT)
        txt_mir.move_to(1.75*UP + 1.9*LEFT)


        # EQUATION REGION

        
        





        # GROUP REGION

        
        


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
            Write(vect_mov_1)
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
            Write(vect_mov_1_mir)
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
            run_time=3
        )

        self.wait(4)


        #
        #
        ### PART - II ###




        # manim -pqh anim1.py Scene2

        # manim -pql anim1.py Scene2

        # manim -sqk anim1.py Scene2

        # manim -sql anim1.py Scene2





###################################################################################################################


# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix"