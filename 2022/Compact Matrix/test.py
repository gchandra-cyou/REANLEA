from __future__ import annotations
from asyncore import poll2
from audioop import add
from cProfile import label
from distutils import text_file


import fractions
from imp import create_dynamic
from multiprocessing import context
from multiprocessing.dummy import Value
from numbers import Number
from operator import is_not
from pickle import TRUE
from tkinter import CENTER, Y, Label, Scale
from imp import create_dynamic
from tracemalloc import start
from turtle import degrees, dot, width
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



class DasAr5(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()

        arr1=DashedDoubleArrow(
            start=LEFT, end=RIGHT, dash_length=2.0,
            stroke_width=1, max_tip_length_to_length_ratio=0.05, color=RED
        ).shift(DOWN)

        dot1=Dot(radius=.2).move_to(UP).set_color(REANLEA_GREEN).set_sheen(-0.4,DOWN)

        dot1.save_state()

        dot2=Dot(radius=.4).move_to(UP + 3*LEFT).set_color(REANLEA_VIOLET_LIGHTER).set_sheen(-0.4,DOWN)

        dot2.save_state()

        with RegisterFont("Montserrat") as fonts:
            text_1=Text("R E A N L E A ", font=fonts[0]).scale(0.6).to_edge(UP).shift(.5*DOWN)             # to_edge(UP) == move_to(3.35*UP)
            text_1.set_color_by_gradient(REANLEA_GREY_DARKER,REANLEA_TXT_COL_DARKER)

        text_1.save_state()

        
        glow_circle1=get_glowing_surround_circle(dot1)
        glow_circle1.save_state()
        glow_circle2=get_glowing_surround_circle(dot2, color=REANLEA_CHARM)
        glow_circle2.save_state()
        

        
        self.add(text_1, dot1,dot2)
        self.wait()
        self.play(Create(arr1), FadeIn(glow_circle1),FadeIn(glow_circle2))
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(DOWN),   
            text_1.animate.scale(0.5).move_to(0.425*UP),
            dot1.animate.move_to(.5*DOWN),
            dot2.animate.move_to(.5*DOWN+1.5*LEFT),
            glow_circle1.animate().move_to(0.5*DOWN),
            glow_circle2.animate().move_to(0.5*DOWN+1.5*LEFT)
        )
        self.wait(2)
        self.play(Restore(self.camera.frame), Restore(text_1), Restore(dot1), Restore(glow_circle1))
        self.wait(2)

           #  manim -pqh test.py DasAr5



class DasAr6(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()

        arr1=DashedDoubleArrow(
            start=LEFT, end=RIGHT, dash_length=2.0,
            stroke_width=1, max_tip_length_to_length_ratio=0.05, color=RED
        ).shift(DOWN)

        dots=[]

        dot1=Dot(radius=.2).move_to(UP).set_color(REANLEA_GREEN).set_sheen(-0.4,DOWN)

        dot1.save_state()
        dots.append(dot1)

        dot2=Dot(radius=.4).move_to(UP + 3*LEFT).set_color(REANLEA_VIOLET_LIGHTER).set_sheen(-0.4,DOWN)

        dot2.save_state()
        dots.append(dot2)

        with RegisterFont("Montserrat") as fonts:
            text_1=Text("R E A N L E A ", font=fonts[0]).scale(0.6).to_edge(UP).shift(.5*DOWN)             # to_edge(UP) == move_to(3.35*UP)
            text_1.set_color_by_gradient(REANLEA_GREY_DARKER,REANLEA_TXT_COL_DARKER)

        text_1.save_state()

        colorz=[REANLEA_YELLOW,REANLEA_CHARM]
        
        glowing_circles=[]
        for i,dot in enumerate(dots):
            glowing_circle=get_glowing_surround_circle(dot, color=colorz[i])
            glowing_circle.save_state()
            glowing_circles.append(glowing_circle)

        #glowing_circles.save_state() doesn't work -> So let's think about the formation with VGroup()
        
        self.add(text_1, dot1,dot2)
        self.wait()
        self.play(Create(arr1), FadeIn(*glowing_circles))
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(DOWN),   
            text_1.animate.scale(0.5).move_to(0.425*UP),
            dots[0].animate.move_to(.5*DOWN),
            dot2.animate.move_to(.5*DOWN+1.5*LEFT),
        )
        self.wait(2)
        self.play(Restore(self.camera.frame), Restore(text_1), Restore(dots[0]), Restore(dots[1]))
        self.wait(2)

           # But here we can't restore all the elements in list 'glowing_circles' at once by calling 'Resore()'
           # even we can't restore all the mobjets in the list by save_state(), because 'list' object has no attribute 'save_state'

           #  manim -pqh test.py DasAr6




class Restorex(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()

        dots=VGroup()
        for i in np.arange(1,6,1):
            dot1 = Dot(radius=0.1*i).move_to(RIGHT*i)
            dot1.save_state()
            dots += dot1
            dot2 = Dot(radius=0.1*i).move_to(LEFT*i)
            dot2.save_state()
            dots += dot2

        dots.save_state()


        with RegisterFont("Montserrat") as fonts:
            text_1=Text("R E A N L E A ", font=fonts[0]).scale(0.6).to_edge(UP).shift(.5*DOWN)             # to_edge(UP) == move_to(3.35*UP)
            text_1.set_color_by_gradient(REANLEA_GREY_DARKER,REANLEA_TXT_COL_DARKER)

        text_1.save_state()


        self.add(text_1)
        self.play(FadeIn(*dots))
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(DOWN),   
            text_1.animate.scale(0.5).move_to(0.425*UP),
            dots[0].animate.move_to(2.5*DOWN)
        )
        self.wait()
        self.play(Restore(self.camera.frame), Restore(text_1), Restore(dots))


        #  manim -pqh test.py Restorex
            
        


class DasAr7(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()

        arr1=DashedDoubleArrow(
            start=LEFT, end=RIGHT, dash_length=2.0,
            stroke_width=1, max_tip_length_to_length_ratio=0.05, color=RED
        ).shift(DOWN)

        color1=[REANLEA_GREEN, REANLEA_VIOLET_LIGHTER]
        dots=VGroup()
        for i in np.arange(1,3,1):
            dot=Dot(radius=0.2*i, color=color1[i-1]).move_to(UP + 2*(i-1)*LEFT).set_sheen(-0.4, DOWN)
            dot.save_state()
            dots += dot

        dots.save_state()


        '''dot1=Dot(radius=.2).move_to(UP).set_color(REANLEA_GREEN).set_sheen(-0.4,DOWN)

        dot1.save_state()
        dots.append(dot1)

        dot2=Dot(radius=.4).move_to(UP + 3*LEFT).set_color(REANLEA_VIOLET_LIGHTER).set_sheen(-0.4,DOWN)

        dot2.save_state()
        dots.append(dot2)'''          # we don't have to define separately now

        with RegisterFont("Montserrat") as fonts:
            text_1=Text("R E A N L E A ", font=fonts[0]).scale(0.6).to_edge(UP).shift(.5*DOWN)             # to_edge(UP) == move_to(3.35*UP)
            text_1.set_color_by_gradient(REANLEA_GREY_DARKER,REANLEA_TXT_COL_DARKER)
            water_mark=Text("R E A N L E A ", font=fonts[0]).scale(0.3).to_edge(UP).shift(.5*DOWN + 5*LEFT).set_opacity(.15)            # to_edge(UP) == move_to(3.35*UP)
            water_mark.set_color_by_gradient(REANLEA_GREY)

        text_1.save_state()
        water_mark.save_state()

        eq1 = MathTex("d(0,x)", "+", "d(x,y)", "=", "d(0,y)").move_to(3*DOWN)
        #eq2 = MathTex("\Rightarrow", "d(x,y)", "=", "d(0,y)", "-", "d(0,x)").move_to(3*DOWN)
        eq2 = MathTex("d(x,y)", "=", "d(0,y)", "-", "d(0,x)").move_to(3*DOWN)
        r_arr= MathTex("\Rightarrow").next_to(eq2, LEFT)
        eq3 = MathTex("d(x,y)", "=", "y", "-", "x").move_to(2.15*DOWN).set_color(REANLEA_BLUE_LAVENDER)




        color2=[REANLEA_YELLOW,REANLEA_CHARM]
        
        glowing_circles=VGroup()                   # VGroup( doesn't have append method) 

        for i,dot in enumerate(list(dots)):
            glowing_circle=get_glowing_surround_circle(dot, color=color2[i])
            glowing_circle.save_state()
            glowing_circles += glowing_circle
        
        glowing_circles.save_state()
        
        
        self.add(text_1, water_mark, *dots)
        self.wait()
        self.play(Create(arr1), FadeIn(*glowing_circles))
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(DOWN),   
            text_1.animate.scale(0.5).move_to(0.425*UP),
            water_mark.animate.scale(0.5).move_to(0.465*UP + 2.5*LEFT),
            dots[0].animate.move_to(.5*DOWN),
            dots[1].animate.move_to(.5*DOWN+1.5*LEFT),
            glowing_circles[0].animate.move_to(.5*DOWN),
            glowing_circles[1].animate.move_to(.5*DOWN +1.5*LEFT),
        )
        self.wait(2)
        self.play(Restore(self.camera.frame), Restore(text_1), Restore(dots), Restore(glowing_circles), Restore(water_mark))
        self.wait(2)

        self.wait()
        self.play(Write(eq1))
        self.wait()
        self.play(
            eq1.animate.scale(0.75).move_to(.25*LEFT + 2.15*DOWN).set_fill(color=REANLEA_GREY_DARKER, opacity=0.75),
            ReplacementTransform(eq1.copy(),eq2),
            FadeIn(r_arr)
        )
        self.wait()
        self.play(
            FadeOut(eq1),
            FadeOut(r_arr),
            #ReplacementTransform(eq2,eq3)
            eq2.animate.move_to(2.15*DOWN).scale(1.1).set_fill(color=REANLEA_BLUE_LAVENDER)
        )
        self.play(
            Circumscribe(eq2, color=REANLEA_CHARM, run_time=1.5)
        )
        self.play(
            Transform(eq2, eq3)
        )
        self.wait(3)


        #  manim -pqh test.py DasAr7

        # This gives us exact result as expected....   ***




class DasAr8(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()

        arr1=DashedDoubleArrow(
            start=LEFT, end=RIGHT, dash_length=2.0,
            stroke_width=1, max_tip_length_to_length_ratio=0.05, color=RED
        ).shift(DOWN)

        color1=[REANLEA_GREEN, REANLEA_VIOLET_LIGHTER]
        dots=VGroup()
        for i in np.arange(1,3,1):
            dot=Dot(radius=0.2*i, color=color1[i-1]).move_to(UP + 2*(i-1)*LEFT).set_sheen(-0.4, DOWN)
            dot.save_state()
            dots += dot

        dots.save_state()       

        with RegisterFont("Montserrat") as fonts:
            text_1=Text("R E A N L E A ", font=fonts[0]).scale(0.6).to_edge(UP).shift(.5*DOWN)             # to_edge(UP) == move_to(3.35*UP)
            text_1.set_color_by_gradient(REANLEA_GREY_DARKER,REANLEA_TXT_COL_DARKER)

        text_1.save_state()

        color2=[REANLEA_YELLOW,REANLEA_CHARM]

        
        def GlowCircFun(x):
            glowing_circles=VGroup()
            for i,dot in enumerate(list(x)):
                 glowing_circle=get_glowing_surround_circle(dot, color=color2[i])
                 #glowing_circle.save_state()
                 glowing_circles += glowing_circle
                 glowing_circles += dot

            glowing_circles.save_state()
            return glowing_circles

                            
        
        
        self.add(text_1, *dots)
        self.wait()
        self.play(Create(arr1), ApplyFunction(GlowCircFun,dots))
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(DOWN),   
            text_1.animate.scale(0.5).move_to(0.425*UP).set_fill(opacity=0.5),
            
        )
        self.wait(2)
        self.play(Restore(self.camera.frame), Restore(text_1), Restore(dots))
        self.wait(2)


        #  manim -pqh test.py DasAr8



class ReplacementTransformOrTransform(Scene):
            def construct(self):
                # set up the numbers
                r_transform = VGroup(*[Integer(i) for i in range(1,4)])
                text_1 = Text("ReplacementTransform", color=REANLEA_BLUE_LAVENDER)
                r_transform.add(text_1)

                transform = VGroup(*[Integer(i) for i in range(4,7)])
                text_2 = Text("Transform", color=REANLEA_GREEN_AUQA)
                transform.add(text_2)

                ints = VGroup(r_transform, transform)
                texts = VGroup(text_1, text_2).scale(0.75)
                r_transform.arrange(direction=UP, buff=1)
                transform.arrange(direction=UP, buff=1)

                ints.arrange(buff=2)
                self.add(ints, texts)

                # The mobs replace each other and none are left behind
                self.play(ReplacementTransform(r_transform[0], r_transform[1]))
                self.play(ReplacementTransform(r_transform[1], r_transform[2]))

                # The mobs linger after the Transform()
                
                self.play(Transform(transform[1], transform[2]))
                self.play(Transform(transform[0], transform[1]))
                self.wait()


                # manim -pqh test.py ReplacementTransformOrTransform



class FontCheck1(Scene):
    def construct(self):
    
    
        self.camera.frame.save_state()

        with RegisterFont("Montserrat") as fonts:
            text_1=Text("R E A N L E A ", font=fonts[0]).scale(0.6).to_edge(UP).shift(.5*DOWN)             # to_edge(UP) == move_to(3.35*UP)
            text_1.set_color_by_gradient(REANLEA_GREY_DARKER,REANLEA_TXT_COL_DARKER)

        text_1.save_state()


        self.add(text_1)
        self.wait()
        
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(DOWN),   
            text_1.animate.scale(0.5).move_to(0.425*UP),
        )
        self.wait(5)


        # manim -pqh test.py FontCheck1



class DasAr9(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()

        with RegisterFont("Montserrat") as fonts:
            text_1=Text("R E A N L E A ", font=fonts[0]).scale(0.3).to_edge(UP).shift(.5*DOWN + 5*LEFT).set_opacity(.15)            # to_edge(UP) == move_to(3.35*UP)
            text_1.set_color_by_gradient(REANLEA_GREY)
        text_1.save_state()

        text_2= MathTex("\pi", "\Rightarrow")
        text_2.save_state()
        


        eq1 = MathTex("d(0,x)", "+", "d(x,y)", "=", "d(0,y)").move_to(3*DOWN)
        #eq2 = MathTex("\Rightarrow", "d(x,y)", "=", "d(0,y)", "-", "d(0,x)").move_to(3*DOWN)
        eq2 = MathTex("d(x,y)", "=", "d(0,y)", "-", "d(0,x)").move_to(3*DOWN)
        r_arr= MathTex("\Rightarrow").next_to(eq2, LEFT)
        eq3 = MathTex("d(x,y)", "=", "y", "-", "x").move_to(2.15*DOWN).set_color(REANLEA_BLUE_LAVENDER)
        eq4 = MathTex("d(x,y)", "=", "(1-t).d(x,y)").move_to(2.5*DOWN).set_color(REANLEA_BLUE_LAVENDER)

        a=Tex("A")
        b=Tex("B")
        c=Tex("C")

        text_3=VGroup(
            Text(", where" ).scale(0.5),
            MathTex("t"),
            Text(" is the distance ratio").scale(0.5)
        ).arrange(buff=0.25).next_to(eq4, DOWN)

        


        abc_grp=VGroup(a,b,c).arrange(direction=RIGHT).move_to(DOWN)
        

        grp=VGroup(eq1,eq2)

        
        self.add(text_1,text_2)
        self.wait()
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(DOWN),   
            text_1.animate.scale(0.5).move_to(0.465*UP + 2.5*LEFT),
        )
        self.wait()
        self.play(Restore(self.camera.frame),Restore(text_1), Restore(text_2))
        self.wait()

        #############

        self.play(
            Create(abc_grp)
        )
        self.wait(2)
        self.play(
            Transform(a, eq1[0]),
            Transform(b, eq1[2]),
            Transform(c, eq1[4])
        )
        self.wait(2)
        self.play(Write(eq1))
        self.play(FadeOut(abc_grp))
        self.wait()
        self.play(
            eq1.animate.scale(0.75).move_to(.25*LEFT + 2.15*DOWN).set_fill(color=REANLEA_GREY_DARKER, opacity=0.75),
            ReplacementTransform(eq1.copy(),eq2),
            FadeIn(r_arr)
        )
        self.wait()
        self.play(
            FadeOut(eq1),
            FadeOut(r_arr),
            #ReplacementTransform(eq2,eq3)
            eq2.animate.move_to(2.15*DOWN).scale(1.1).set_fill(color=REANLEA_BLUE_LAVENDER)
        )
        self.play(
            Circumscribe(eq2, color=REANLEA_CHARM, run_time=1.5)
        )
        self.play(
            ReplacementTransform(eq2, eq3)                    # ReplacementTransform doesn't treat eq3 as eq2 after transformation.
        )
        self.wait(2)
        
        self.play(Transform(eq3,eq4))
        self.play(Create(text_3))
        self.wait(3)

           #  manim -pqh test.py DasAr9



class MovingDotEx(Scene):
    def construct(self):

        # WATER-MARK
        with RegisterFont("Montserrat") as fonts:
            water_mark=Text("R E A N L E A ", font=fonts[0]).scale(0.3).to_edge(UP).shift(.5*DOWN + 5*LEFT).set_opacity(.15)            # to_edge(UP) == move_to(3.35*UP)
            water_mark.set_color_by_gradient(REANLEA_GREY)
        water_mark.save_state()

        # Tracker 
        x=ValueTracker(-3)

        # MOBJECTS
        line1=Line(3*LEFT,3*RIGHT).set_color(REANLEA_PINK_DARKER).set_opacity(0.6)
        
        dot1=Dot(radius=0.25, color=REANLEA_VIOLET_LIGHTER).move_to(3*RIGHT).set_sheen(-0.6,DOWN)
        dot2=Dot(radius=0.25, color=REANLEA_BLUE_LAVENDER).move_to(3*LEFT).set_sheen(-0.6,DOWN)
        dot3=dot2.copy().set_opacity(0.4)


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
            start=p2-DOWN, end=p2, stroke_width=1
        ).set_color(RED_D)


        # GROUPS
        grp1=VGroup(line1,dot1,dot2, d_line, v_line1, v_line2)


        #value updater
        value=DecimalNumber().set_color_by_gradient(REANLEA_GREEN_LIGHTER).set_sheen(-0.4,DR)

        value.add_updater(
            lambda x : x.set_value(1-(dot2.get_center()[0]/3))
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

        tex=MathTex("d(x,y)=").set_color(REANLEA_GREEN_LIGHTER).set_sheen(-0.4,DR)
        grp2=VGroup(tex,value).arrange(RIGHT, buff=0.3).move_to(2*UP)
        



        # play region

        self.add(water_mark)
        self.play(Create(grp1))
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


        #  manim -pqh test.py MovingDotEx



class MovingDotEx2(Scene):
    def construct(self):

        # WATER-MARK
        with RegisterFont("Montserrat") as fonts:
            water_mark=Text("R E A N L E A ", font=fonts[0]).scale(0.3).to_edge(UP).shift(.5*DOWN + 5*LEFT).set_opacity(.15)            # to_edge(UP) == move_to(3.35*UP)
            water_mark.set_color_by_gradient(REANLEA_GREY)
        water_mark.save_state()

        # Tracker 
        x=ValueTracker(-3)

        # MOBJECTS
        line=NumberLine(
            x_range=[-3,3],
            include_ticks=False,
            include_tip=False
        ).set_color(REANLEA_PINK_DARKER).set_opacity(0.6)
        
        dot1=Dot(radius=0.25, color=REANLEA_VIOLET_LIGHTER).move_to(3*RIGHT).set_sheen(-0.6,DOWN)
        dot2=Dot(radius=0.25, color=REANLEA_BLUE_LAVENDER).move_to(3*LEFT).set_sheen(-0.6,DOWN)
        dot3=dot2.copy().set_opacity(0.4)


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
            start=p2-DOWN, end=p2, stroke_width=1
        ).set_color(RED_D)


        # GROUPS
        grp1=VGroup(line,dot1,dot2, d_line, v_line1, v_line2)


        #value updater
        value=DecimalNumber().set_color_by_gradient(REANLEA_GREEN_LIGHTER).set_sheen(-0.4,DR)

        value.add_updater(
            lambda x : x.set_value(1-(dot2.get_center()[0]/3))
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

        tex=MathTex("d(x,y)=").set_color(REANLEA_GREEN_LIGHTER).set_sheen(-0.4,DR)
        grp2=VGroup(tex,value).arrange(RIGHT, buff=0.3).move_to(2*UP)
        



        # play region

        self.add(water_mark)
        self.play(Create(grp1))
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


        #  manim -pqh test.py MovingDotEx2



class AnnuFont(Scene):
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
        self.wait()


        #  manim -pqh test.py AnnuFont



class CircEx(Scene):
    def construct(self):
        n=40
        buff_min=0
        buff_max=0.15
        glowing_circle = VGroup(
            *[
                Circle(radius=.25+interpolate(buff_min, buff_max, b))
                for b in np.linspace(0, 1, n)
            ]
        )
        for i, c in enumerate(glowing_circle):
            c.set_stroke(REANLEA_BLUE_LAVENDER, width=0.5, opacity=1- i / n)


        self.play(
            Create(glowing_circle),
            run_time=5
        )
        self.wait()


        #  manim -pqh test.py CircEx




class BezierEx(Scene):
    def construct(self):
        with RegisterFont("Montserrat") as fonts:
            text_1=Text("R E A N L E A ", font=fonts[0]).scale(.55)
            text_1.set_color_by_gradient(REANLEA_GREY)
        text_1.save_state()
        p1 = ParametricFunction(
            lambda t: bezier(np.array([
                [.3,1.13,0],
                [.38, .09, 0],
                [2.20, 0.95, 0],
                [2.08, 1.02, 0],  
            ]))(t),
            [0, 1],
            color=REANLEA_CHARM,
        ).reverse_direction()
        p2 = ParametricFunction(
            lambda t: bezier(np.array([
                [.3, 1.13, 0],
                [.43, 1.91, 0],
                [3.21, 1.89, 0],
                [1.52, .62, 0],
            ]))(t),
            [0, 1],
            color=REANLEA_CHARM,
        )

        p = ParametricFunction(
            lambda t: bezier(np.array([
                [2.08, 1.02, 0],
                [2.20, 0.95, 0],
                [.38, .09, 0],
                [.3,1.13,0],
                [.3, 1.13, 0],
                [.43, 1.91, 0],
                [3.21, 1.89, 0],
                [1.52, .62, 0],
            ]))(t),
            [0, 1],
            color=REANLEA_CHARM,
        ).flip(axis=RIGHT).scale(1.85).move_to(text_1.get_center()).rotate(10*DEGREES)


        p3 = ParametricFunction(
            lambda t: bezier(np.array([
                [1.68, .67, 0],
                [1.65, 0.35, 0],
                [.21, .67, 0],
                [.3,1.13,0],
                [.47, 1.65, 0],
                [2.93, 0.26, 0],
                [.97, .59, 0],
            ]))(t),
            [0, 1],
            color=REANLEA_CHARM,
        ).flip(axis=RIGHT).scale(2.5).move_to(text_1.get_center()).rotate(-15*DEGREES)


        grp=VGroup(p1,p2).flip(axis=RIGHT).rotate(15*DEGREES).scale(1.85).move_to(text_1.get_center())

        p4 = ParametricFunction(
            lambda t: bezier(np.array([
                [1.68,.67,0],
                [1.65,.35, 0],
                [.21,.67, 0],
                [.3,1.13, 0],  
            ]))(t),
            [0, 1],
            color=REANLEA_CHARM,
        )
        p5 = ParametricFunction(
            lambda t: bezier(np.array([
                [.3, 1.13, 0],
                [.47, 1.65, 0],
                [2.93, .26, 0],
                [.93, .58, 0],
            ]))(t),
            [0, 1],
            color=REANLEA_CHARM,
        )
        grp1=VGroup(p4,p5).scale(1.95).move_to(text_1.get_center()).flip(axis=RIGHT).rotate(-15*DEGREES)

        self.play(
            Write(text_1)
        )
        self.play(Create(grp1))
        self.wait(2)


        #  manim -pqh test.py BezierEx



class AddTextLetterByLetterz(Scene):
    def construct(self):

        # WATER-MARK
        with RegisterFont("Montserrat") as fonts:
            water_mark=Text("R E A N L E A ", font=fonts[0]).scale(0.3).to_edge(UP).shift(.5*DOWN + 5*LEFT).set_opacity(.15)            # to_edge(UP) == move_to(3.35*UP)
            water_mark.set_color_by_gradient(REANLEA_GREY)
        water_mark.save_state()

        # HEADING
        with RegisterFont("Courier Prime") as fonts:
            text_1 = Text("You MAY KNOW almost EVERYTHING if you want.", font=fonts[0]).set_color_by_gradient(REANLEA_GREY).scale(.4)
            text_2 = Text("But without investing ENOUGH TIME you CAN'T LEARN anything ...", font=fonts[0]).set_color_by_gradient(REANLEA_GREY).scale(.4)
            

            
        grp=VGroup(text_1,text_2).arrange(DOWN)

        s1=AnnularSector(inner_radius=2, outer_radius=2.75, angle=2*PI, color=REANLEA_GREY_DARKER).set_opacity(0.3).move_to(5.5*LEFT)
        s2=AnnularSector(inner_radius=.2, outer_radius=.4, angle=2*PI, color=REANLEA_GREY).set_opacity(0.3).move_to(UP + 5*RIGHT)
        s3=AnnularSector(inner_radius=1, outer_radius=1.5, color=REANLEA_SLATE_BLUE).set_opacity(0.6).move_to(3.5*DOWN + 6.5*RIGHT).rotate(PI/2)
        ann=VGroup(s1,s2,s3)


        self.add(water_mark)
        self.play(
            AddTextLetterByLetter(text_1)
        )
        self.play(
            AddTextLetterByLetter(text_2)
        )

        self.wait(2)

        self.play(
            *[FadeOut(mobj) for mobj in self.mobjects]
        )


        #  manim -pqh test.py AddTextLetterByLetterz


class ex2(Scene):
    def construct(self):

        text1=Text("R E A N L E A ").scale(2)

        a=get_surround_bezier(text1)

        self.add(text1)
        self.play(
            Create(a),
            text1.animate.set_opacity(0.5)
        )

         #  manim -pqh test.py ex2



class BezierEx2(Scene):
    def construct(self):
        with RegisterFont("Montserrat") as fonts:
            text_1=Text("R E A N L E A ", font=fonts[0]).scale(.55)
            text_1.set_color_by_gradient(REANLEA_GREY)
        text_1.save_state()

        grp=VGroup()

        p1 = ParametricFunction(
            lambda t: bezier(np.array([
                [2.21,1.50,0],
                [2.19, .31, 0],
                [.33, 1.83, 0],
                [.40, .58, 0],  
            ]))(t),
            [0, 1],
            #color=REANLEA_CHARM,
        ).flip(RIGHT)

        grp += p1


        ar= Arrow(max_stroke_width_to_length_ratio=0,max_tip_length_to_length_ratio=0.15).move_to(p1.get_end()+.55*DOWN).rotate(PI/2)
        

        
        grp += ar

        grp.set_color_by_gradient(REANLEA_BLUE_SKY)

        grp.next_to(text_1, DOWN)


        self.play(
            Write(text_1)
        )
        self.play(Create(grp))
        self.wait(2)


        #  manim -pqh test.py BezierEx2



class ex3(Scene):
    def construct(self):

        with RegisterFont("Montserrat") as fonts:
            text1=Text("R E A N L E A ", font=fonts[0]).scale(.55)
            text1.set_color_by_gradient(REANLEA_GREY).set_opacity(0.6)
        

        #a=ArrowCubicBezierUp(text1)[0].set_color(REANLEA_BLUE_SKY)
        #b=ArrowCubicBezierUp(text1)[1].set_color(REANLEA_CHARM)
        c=ArrowCubicBezierUp()#.set_color(REANLEA_BLUE_SKY)
        #arcu=CurvesAsSubmobjects(a)

        #arcu.set_color_by_gradient(REANLEA_GREEN,REANLEA_CHARM)

        #grp=VGroup(b,arcu)


        self.play(Create(text1), rate_functions=low_frame_rate)
        self.play(
            Create(c),
            lag_ratio=0.2,                                            # to run all the mobjects in a VGroup simultaneously we've to use 'lag_ratio=0'
        )

         #  manim -pqh test.py ex3


class ex4(Scene):
    def construct(self):

        with RegisterFont("Montserrat") as fonts:
            text1=Text("R E A N L E A ", font=fonts[0]).scale(.55)
            text1.set_color_by_gradient(REANLEA_GREY).set_opacity(0.6)
        

        
        c=ArrowQuadricBezierDown(text1).shift(.7*RIGHT)
        

        self.play(
            Write(text1),
            Create(c),
            lag_ratio=0.2,                            
        )
        self.wait(2)

         #  manim -pqh test.py ex4



class BezierEx3(Scene):
    def construct(self):
        with RegisterFont("Montserrat") as fonts:
            text_1=Text("R E A N L E A ", font=fonts[0]).scale(.55)
            text_1.set_color_by_gradient(REANLEA_GREY)
        text_1.save_state()

        grp=VGroup()

        p1 = ParametricFunction(
            lambda t: bezier_updated(
                np.array([
                    [3,10,0],
                    [6, 4, 0],
                    [13, 4, 0],
                    [16, 10, 0],
                    [13,15,0],
                    [6,16,0],
                    [6,20,0]
                ],
            ))(t),
            [0, 1],
            #color=REANLEA_CHARM,
        ).flip(DOWN).scale(.3).set_stroke(width=10, opacity=0.35)

        

        grp += p1

        grp.set_color_by_gradient(REANLEA_BLUE_SKY)

        grp.move_to(text_1.get_center())


        self.play(
            Write(text_1)
        )
        self.play(Create(grp))
        self.wait(2)


        #  manim -pqh test.py BezierEx3


######## Discord Solution of bezier_updated ###########


def bezier_updated1(t,
    points: np.ndarray,
    weights: np.ndarray,
):
    n = len(points) - 1


    '''if n == 3:
        return (
             (1 - t) ** 3 * points[0]*weights[0]
            + 3 * t * (1 - t) ** 2 * points[1]*weights[1]
            + 3 * (1 - t) * t**2 * points[2]*weights[2]
            + t**3 * points[3]*weights[3]
        )

    else:'''
    return  sum(
        ((1 - t) ** (n - k)) * (t**k) * choose(n, k) * point * weights[k]  for k, point in enumerate(points) 
    )

class BezierEx4(Scene):
    def construct(self):
        
        '''for t in np.linspace(0,1,10):
            print("t={:3.1f} -> {}".format(t,bezier_updated1(t,
                np.array([
                    [3,10,0],
                    [6, 4, 0],
                    [13, 4, 0],
                    [16, 10, 0],
                ]),
                np.array([8,2,8,2]))))'''

        p1 = ParametricFunction(
            lambda t: 0.1*bezier_updated1(t,
                np.array([
                    [3,10,0],
                    [6, 4, 0],
                    [13, 4, 0],
                    [16, 10, 0],
                    [13,15,0],
                    [6,16,0],
                    [3,10,0]
                ]),
                np.array([1,8,1,8,1,8,1])),
            t_range=[0, 1],
            color=REANLEA_CHARM,
        )
        
        p1.move_to(ORIGIN).flip(UP).rotate(PI/2)

        self.play(Create(p1))
        self.wait(2)


        #  manim -pqh test.py BezierEx4



class BezierEx5(Scene):
    def construct(self):

        grp=VGroup()
        p1 = ParametricFunction(
            lambda t: bezier_updated1(t,
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
        p.set_color_by_gradient(REANLEA_GREEN,REANLEA_BLUE,REANLEA_CHARM).set_stroke(width=15)

        grp += p


        ar= Arrow(max_stroke_width_to_length_ratio=0,max_tip_length_to_length_ratio=0.65).move_to(p1.get_end()+.55*DOWN).rotate(PI/2)
        ar.set_color(REANLEA_CHARM)

        
        grp += ar

        
        

        self.play(Create(grp))
        self.wait(2)


        #  manim -pqh test.py BezierEx5


# ROUND CORNERS   
# VISIT : https://github.com/GarryBGoode/Manim_CAD_Drawing_utils
class Test_round(Scene):
    def construct(self):
        mob1 = RegularPolygon(n=4,radius=1.5,color=REANLEA_GREY_DARKER).rotate(PI/4)
        mob1.set_stroke(width=50, opacity=.4).scale(2.5)
        mob2 = Triangle(radius=1.5,color=REANLEA_BLUE_LAVENDER).scale(2.5)
        mob2#.set_fill(opacity=0.1).set_z_index(10)
        crbase = Rectangle(height=0.5,width=3)
        mob3 = Union(crbase.copy().rotate(PI/4),crbase.copy().rotate(-PI/4),color=BLUE)
        mob4 = Circle(radius=1.3)
        mob2#.shift(2.5*UP)
        mob3.shift(2.5*DOWN)
        mob1.shift(2.5*LEFT)
        mob4.shift(2.5*RIGHT)

        mob1 = Round_Corners(mob1, 0.25)
        #mob2 = Round_Corners(mob2, 0.25)
        mob2_1=mob2.copy().scale(1.5)
        mob3 = Round_Corners(mob3, 0.25)
        mob4 = Round_Corners (mob4, 0.25)
        #self.add(mob1,mob2,mob3,mob4)
        grp=VGroup(mob1,mob2,mob3,mob4)

        self.play(
            Write(grp)
        )
        self.wait(2)


        # manim -pqh test.py Test_round


class  test_dimension_pointer(Scene):
    def construct(self):
        mob1 = Round_Corners(Triangle().scale(2),0.3)
        p = ValueTracker(0)
        dim1 = Pointer_To_Mob(mob1,p.get_value(),r'triangel')
        dim1.add_updater(lambda mob: mob.update_mob(mob1,p.get_value()))
        dim1.update()
        PM = Path_mapper(mob1)
        self.play(Create(mob1),rate_func=PM.equalize_rate_func(smooth))
        self.play(Create(dim1))
        self.play(p.animate.set_value(1),run_time=10)
        self.play(Uncreate(mob1,rate_func=PM.equalize_rate_func(smooth)))
        self.play(Uncreate(dim1))
        self.wait()

         # manim -pqh test.py test_dimension_pointer


class test_dimension_base(Scene):
    def construct(self):
        mob1 = Round_Corners(Triangle().scale(2),0.3)
        dim1 = Linear_Dimension(mob1.get_critical_point(UP),
                                mob1.get_critical_point(DOWN),
                                direction=RIGHT,
                                offset=3,
                                color=RED)
        dim2 = Linear_Dimension(mob1.get_critical_point(RIGHT),
                                mob1.get_critical_point(LEFT),
                                direction=UP,
                                offset=-3,
                                color=RED)
        #self.add(mob1,dim1,dim2)
        grp=VGroup(mob1,dim1,dim2)

        self.play(Write(grp))
        self.wait(2)


        # manim -pqh test.py test_dimension_base



class test_dash(Scene):
    def construct(self):
        mob1 = Round_Corners(Square().scale(3),radius=0.8).shift(DOWN*0)
        vt = ValueTracker(0)
        dash1 = Dashed_line_mobject(mob1,num_dashes=36,dashed_ratio=0.5,dash_offset=0)
        def dash_updater(mob):
            offset = vt.get_value()%1
            dshgrp = mob.generate_dash_mobjects(
                **mob.generate_dash_pattern_dash_distributed(36, dash_ratio=0.5, offset=offset)
            )
            mob['dashes'].become(dshgrp)
        dash1.add_updater(dash_updater)

        self.add(dash1)
        self.play(vt.animate.set_value(2),run_time=6)
        self.wait(0.5)



         # manim -pqh test.py test_dash


class Test_chamfer(Scene):
    def construct(self):
        mob1 = RegularPolygon(n=4,radius=1.5,color=PINK).rotate(PI/4)
        mob2 = Triangle(radius=1.5,color=TEAL)
        crbase = Rectangle(height=0.5,width=3)
        mob3 = Union(crbase.copy().rotate(PI/4),crbase.copy().rotate(-PI/4),color=BLUE)
        mob4 = Circle(radius=1.3)
        mob2.shift(2.5*UP)
        mob3.shift(2.5*DOWN)
        mob1.shift(2.5*LEFT)
        mob4.shift(2.5*RIGHT)

        mob1 = Chamfer_Corners(mob1, 0.25)
        mob2 = Chamfer_Corners(mob2,0.25)
        mob3 = Chamfer_Corners(mob3, 0.25)
        #self.add(mob1,mob2,mob3,mob4)

        grp=VGroup(mob1,mob2,mob3,mob4)

        self.play(
            Create(mob2)
        )
        self.wait(2)


        # manim -pqh test.py Test_chamfer




class RopeEx(Scene):
    def construct(self):

        theta_tracker=ValueTracker(0)
    
        dot1=Dot(radius=0.15, color=REANLEA_GREEN_LIGHTER).move_to(3*LEFT).set_sheen(-0.6,DOWN)
        dot2=Dot(radius=0.15, color=REANLEA_BLUE_SKY).move_to(3*RIGHT).set_sheen(-0.6,DOWN)

        dot3=Dot(radius=0.15, color=REANLEA_BLUE_SKY).move_to(4*RIGHT+UP)
        dot4=Dot(radius=0.15, color=REANLEA_BLUE_SKY).move_to(4*LEFT+DOWN)



        p1= dot1.get_center()
        p2= dot2.get_center()

        line=Line(start=p1,end=p2).set_color(REANLEA_PURPLE)

        length = line.get_length()
        angle1  = line.get_angle()

        line1=line.copy()
        line1_ref=line1.copy().set_stroke(width=15)

        
        '''line.add_updater(lambda z : z.become(
            Line(start=dot1.get_center(), end=dot2.get_center())
        ))'''

        line=always_redraw(lambda : Line(start=dot1.get_center(), end=dot2.get_center()) )

        line1.rotate(
            theta_tracker.get_value()*DEGREES, about_point=dot1.get_center()
        )

        line1.add_updater(
            lambda x: x.become(line1_ref.copy().rotate(
                theta_tracker.get_value(), about_point=dot1.get_center()
            ))
        )

        


        grp=VGroup(line,line1,dot1,dot2)




        self.play(Write(grp))
        mxg=dot2.animate.move_to(dot3.get_center()[0])
        self.play(
           mxg,
           theta_tracker.animate.set_value(line.get_angle())
        )
        self.play(theta_tracker.animate.set_value(line.get_angle()))
        
        self.wait(2)

        


        # manim -pqh test.py RopeEx



class TransformPathArc(Scene):
            def construct(self):
                def make_arc_path(start, end, arc_angle):
                    points = []
                    p_fn = path_along_arc(arc_angle)
                    # alpha animates between 0.0 and 1.0, where 0.0
                    # is the beginning of the animation and 1.0 is the end.
                    for alpha in range(0, 11):
                        points.append(p_fn(start, end, alpha / 10.0))
                    path = VMobject(stroke_color=YELLOW)
                    path.set_points_smoothly(points)
                    return path

                left = Circle(stroke_color=BLUE_E, fill_opacity=1.0, radius=0.5).move_to(LEFT * 2)
                colors = [TEAL_A, TEAL_B, TEAL_C, TEAL_D, TEAL_E, GREEN_A]
                # Positive angles move counter-clockwise, negative angles move clockwise.
                examples = [-90, 0, 30, 90, 180, 270]
                anims = []
                for idx, angle in enumerate(examples):
                    left_c = left.copy().shift((3 - idx) * UP)
                    left_c.fill_color = colors[idx]
                    right_c = left_c.copy().shift(4 * RIGHT)
                    path_arc = make_arc_path(left_c.get_center(), right_c.get_center(),
                                             arc_angle=angle * DEGREES)
                    desc = Text('%d°' % examples[idx]).next_to(left_c, LEFT)
                    # Make the circles in front of the text in front of the arcs.
                    self.add(
                        path_arc.set_z_index(1),
                        desc.set_z_index(2),
                        left_c.set_z_index(3),
                    )
                    anims.append(Transform(left_c, right_c, path_arc=angle * DEGREES))
                    anims

                self.play(*anims, run_time=2)
                self.wait()


                 # manim -pqh test.py TransformPathArc  



class sqEx(Scene):
    def construct(self):
        sq= Square(side_length=1)

        def update_sq(mob):
            mob.scale(1.5)
            mob.rotate(PI/3)
            return mob

        self.play(Create(sq))

        self.play(
                ApplyFunction(update_sq, sq),
                run_time=4
                )
        
        self.wait()

        # manim -pqh test.py sqEx