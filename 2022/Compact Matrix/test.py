from __future__ import annotations
from audioop import add
from cProfile import label
from distutils import text_file


import fractions
from imp import create_dynamic
from multiprocessing import context
from multiprocessing.dummy import Value
from numbers import Number
from operator import is_not
from tkinter import CENTER, Y, Label, Scale
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


config.background_color= REANLEA_BACKGROUND_COLOR


###############################################################################################################


class Anagram(Scene):
    def construct(self):
        src=Tex("Welcome to REANLEA.com")
        trgt=Tex("Here we'll make your Imagination into Reality!")
        self.play(Write(src))
        self.wait()
        self.play(TransformMatchingShapes(src,trgt,path_arc=PI/2))
        self.wait(2)
        self.play(*[FadeOut(mobj) for mobj in self.mobjects])
        self.wait(0.5)
 
        s=Square()
        circ=Circle()
        s.save_state()
        self.play(FadeIn(s),
            run_time=2)
        self.play(s.animate.set_color(PURPLE).set_opacity(0.5).shift(2*LEFT).scale(4))
        self.play(s.animate.shift(5*DOWN).rotate(PI/4),
            run_time=2  )
        self.play(
            Restore(s),
            run_time=4
        )
        def fn(x):
            x.scale(0.5)
            x.shift(UP*3)
            return x
 
        self.play(
            ApplyFunction(fn,s),
            run_time=5
        )
        self.wait()
 
        self.play(
            Transform(s,circ)
        )
   
        def fn(x):
            x.scale(0.5)
            x.shift(2*UP+4*RIGHT)
            x.set_fill(color=GREEN, opacity=0.5)
            return x
        self.play(
            *[FadeOut(mobj) for mobj in self.mobjects],
            ApplyFunction(fn,circ),
            run_time=3
        )    
        self.play(*[FadeOut(mobj) for mobj in self.mobjects])
       
 
 
        variables = VGroup(MathTex("a"), MathTex("b"), MathTex("c")).arrange_submobjects().shift(UP)
 
        eq1 = MathTex("{{x}}^2", "+", "{{y}}^2", "=", "{{z}}^2")
        eq2 = MathTex("{{a}}^2", "+", "{{b}}^2", "=", "{{c}}^2")
        eq3 = MathTex("{{a}}^2", "=", "{{c}}^2", "-", "{{b}}^2")
 
        self.add(eq1)
        self.wait()
        self.play(TransformMatchingTex(Group(eq1, variables), eq2))
        self.wait(0.5)
        self.play(TransformMatchingTex(eq2, eq3))
        self.wait()
        self.play(*[FadeOut(mobje) for mobje in self.mobjects])
 
 
 
 
        mob=Circle(radius=4,color=TEAL_A)
        self.play(Write(Tex("Join Us Now!"),
            run_time=1.25),
            Broadcast(mob))
        self.wait()
        self.play(*[FadeOut(mobjec) for mobjec in self.mobjects])


        # manim -pqh test.py Anagram
 

class LineExample(Scene):
                def construct(self):
                    d = VGroup()
                    for i in range(0,10):
                        d.add(Dot())
                    d.arrange_in_grid(buff=1)
                    self.add(d)
                    l= Line(d[0], d[1])
                    self.add(l)
                    self.wait()
                    l.put_start_and_end_on(d[1].get_center(), d[2].get_center())
                    self.wait()
                    l.put_start_and_end_on(d[4].get_center(), d[7].get_center())
                    self.wait()


                    # manim -pqh test.py LineExample



class DashedLineExample(Scene):
            def construct(self):
                # dash_length increased
                dashed_1 = DashedLine(start=LEFT, end=RIGHT, dash_length=2.0).shift(UP*2)
                # normal
                dashed_2 = DashedLine(config.left_side, config.right_side)
                # dashed_ratio decreased
                dashed_3 = DashedLine(config.left_side, config.right_side, dashed_ratio=0.1).shift(DOWN*2)
                self.add(dashed_1, dashed_2, dashed_3)


                # manim -pqh test.py DashedLineExample


class DasAr(Scene):
    def construct(self):
           arr1=DashedArrow(start=LEFT, end=RIGHT, dash_length=2.0, max_tip_length_to_length_ratio=0.15, color=RED)

           def fun(x):
            return Write(x)
            

           with RegisterFont("Kalam") as fonts:
            text_1 = VGroup(*[Text(x, font=fonts[0],weight=BOLD) for x in (
                "Arrow is",
                "Always RIGHT"
            )]).arrange_submobjects(DOWN).to_edge(UP).shift(0.5*DOWN).scale(0.6).set_color(REANLEA_TXT_COL)

            with RegisterFont("Montserrat") as fonts:
                a=Text("R E A N L E A",font=fonts[0], weight=THIN).set_color(REANLEA_TXT_COL).to_edge(UP).shift(0.5*DOWN).scale(0.5)
                b=Text("P A C I F I C   R E S I L I E N C E",font=fonts[0]).set_color(REANLEA_TXT_COL).to_edge(UP).shift(0.5*DOWN).scale(0.5)


            g=VGroup(text_1,a)



            self.play(Create(arr1))
            self.play(
                Create(text_1)
            )
            self.wait(2)
            self.play(Transform(text_1,a))
            self.wait()
            #self.play(FadeOut(g))
            #self.play(Write(b))
            self.play(
                FadeOut(g),
                Write(b)
            )
    
            self.wait()

           # manim -pqh test.py DasAr






class FontCheck(Scene):
    def construct(self):
         with RegisterFont("Montserrat") as fonts:
            tx=MarkupText("WHAT   ABOUT   D I S T A N C E ?", font=fonts[0]).scale(0.6)

         self.play(Write(tx))

         # manim -pqh test.py FontChek


class DasAr1(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()
        arr1=DashedDoubleArrow(
            start=LEFT, end=RIGHT, dash_length=2.0,
            stroke_width=1, max_tip_length_to_length_ratio=0.05, color=RED
        )

        with RegisterFont("Montserrat") as fonts:
            text_1=Text("D I S T A N C E ", font=fonts[0], weight=BOLD).scale(0.6).to_edge(UP).shift(0.5*DOWN)
            text_1.set_color_by_gradient(REANLEA_BLUE_SKY,REANLEA_TXT_COL_DARKER)

        
        self.add(text_1)
        self.play(Create(arr1))
        self.play(self.camera.frame.animate.scale(0.5).move_to(DOWN))
        self.wait(2)

           #  manim -pqh test.py DasAr1



                
class DasAr2(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()

        arr1=DashedDoubleArrow(
            start=LEFT, end=RIGHT, dash_length=2.0,
            stroke_width=1, max_tip_length_to_length_ratio=0.05, color=RED
        ).shift(DOWN)

        dot1=Dot(radius=.2).move_to(UP).set_color(REANLEA_GREEN).set_sheen(-0.6,DOWN)

        dot1.save_state()

        with RegisterFont("Montserrat") as fonts:
            text_1=Text("R E A N L E A ", font=fonts[0]).scale(0.6).to_edge(UP).shift(.5*DOWN)             # to_edge(UP) == move_to(3.35*UP)
            text_1.set_color_by_gradient(REANLEA_GREY_DARKER,REANLEA_TXT_COL_DARKER)

        text_1.save_state()

        
        self.add(text_1, dot1)
        self.wait()
        self.play(Create(arr1))
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(DOWN),   
            text_1.animate.scale(0.5).move_to(0.425*UP),
            dot1.animate.move_to(.5*DOWN)
        )
        self.wait(2)
        self.play(Restore(self.camera.frame), Restore(text_1), Restore(dot1))
        self.wait(2)

           #  manim -pqh test.py DasAr2





class CircleWithContent(VGroup):
    def __init__(self, content):
        super().__init__()
        self.circle = Circle(radius=3)
        self.content = content
        self.add(self.circle, content)
        content.move_to(self.circle.get_center())

    def clear_content(self):
        self.remove(self.content)
        self.content = None

    @override_animate(clear_content)
    def _clear_content_animation(self, anim_args=None):
        if anim_args is None:
            anim_args = {}
        anim = Uncreate(self.content, **anim_args)
        self.clear_content()
        return anim

class AnimationOverrideExample2(Scene):
    def construct(self):
        t = Text("R E A N L E A", font="Comic Sans")
        my_mobject = CircleWithContent(t)
        self.play(Create(my_mobject))
        self.wait(3)
        self.play(my_mobject.animate.clear_content())
        self.wait()


        # manim -pqh test.py AnimationOverrideExample2




class DasAr3(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()

        arr1=DashedDoubleArrow(
            start=LEFT, end=RIGHT, dash_length=2.0,
            stroke_width=1, max_tip_length_to_length_ratio=0.05, color=RED
        ).shift(DOWN)
        arr2=DashedDoubleArrow(
            start=LEFT, end=RIGHT, dash_length=2.0,
            stroke_width=1, max_tip_length_to_length_ratio=0.05, color=RED
        ).shift(DOWN)
        arr3=DashedDoubleArrow(
            start=LEFT, end=RIGHT, dash_length=2.0,
            stroke_width=1, max_tip_length_to_length_ratio=0.05, color=RED
        ).shift(DOWN)


        grp=VGroup(arr1,arr2,arr3)
        grp.save_state()

        with RegisterFont("Montserrat") as fonts:
            text_1=Text("R E A N L E A ", font=fonts[0]).scale(0.6).to_edge(UP).shift(.5*DOWN)             # to_edge(UP) == move_to(3.35*UP)
            text_1.set_color_by_gradient(REANLEA_GREY_DARKER,REANLEA_TXT_COL_DARKER)

        text_1.save_state()

        
        self.add(text_1)
        self.wait()
        self.play(Create(grp))
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(DOWN + 1.5*RIGHT),   
            text_1.animate.scale(0.5).move_to(0.425*UP +1.5*RIGHT),
            arr2.animate.move_to(DOWN+1.5*RIGHT),
            arr3.animate.move_to(DOWN+3*RIGHT)
        )
        self.wait(2)
        self.play(Restore(self.camera.frame), Restore(text_1), Restore(grp))
        self.wait(2)

           #  manim -pqh test.py DasAr3



class DasArGlowCircle(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()

        arr1=DashedDoubleArrow(
            start=LEFT, end=RIGHT, dash_length=2.0,
            stroke_width=1, max_tip_length_to_length_ratio=0.05, color=RED
        ).shift(DOWN)

        dot1=Dot(radius=.2).move_to(UP).set_color(REANLEA_GREEN).set_sheen(-0.4,DOWN)

        dot1.save_state()

        with RegisterFont("Montserrat") as fonts:
            text_1=Text("R E A N L E A ", font=fonts[0]).scale(0.6).to_edge(UP).shift(.5*DOWN)             # to_edge(UP) == move_to(3.35*UP)
            text_1.set_color_by_gradient(REANLEA_GREY_DARKER,REANLEA_TXT_COL_DARKER)

        text_1.save_state()

        glowing_circles=[]
        glow_circle=get_glowing_surround_circle(dot1)
        glow_circle.save_state()
        glowing_circles.append(FadeIn(glow_circle))

        
        self.add(text_1, dot1)
        self.wait()
        self.play(Create(arr1), FadeIn(glow_circle))
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(DOWN),   
            text_1.animate.scale(0.5).move_to(0.425*UP),
            dot1.animate.move_to(.5*DOWN),
            glow_circle.animate().move_to(0.5*DOWN)
        )
        self.wait(2)
        self.play(Restore(self.camera.frame), Restore(text_1), Restore(dot1), Restore(glow_circle))
        self.wait(2)

           #  manim -pqh test.py DasArGlowCircle