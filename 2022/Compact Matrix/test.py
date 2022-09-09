from __future__ import annotations
from cProfile import label


import fractions
from imp import create_dynamic
from multiprocessing import context
from multiprocessing.dummy import Value
from numbers import Number
from tkinter import Y, Label
from imp import create_dynamic
from tracemalloc import start
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



class ItrRiemannRects(Scene):
    def construct(self):
            ax=Axes(
                x_range=[-1,10],
                y_range=[-1,8],
                tips=False
            )
            x_label=ax.get_x_axis_label(
                Tex("$x$").scale(0.65)
            )
            y_label=ax.get_y_axis_label(
                Tex("$y$").scale(0.65)
            )
            ax_label=VGroup(ax,x_label,y_label)  #represent axes with label
            quad=ax.plot(lambda x : 0.1*(x-2)*(x-8)*(x-5)+5 , x_range=[0.3,9.2]) #define function

            flat_rects=ax.get_riemann_rectangles(  
                quad, 
                x_range=[2,8],
                dx=0.5,
                color=[REANLEA_BLUE, REANLEA_YELLOW],
                stroke_width=0.5,
                fill_opacity=0.75,
            )                                          #get riemann rectangles
            grrl=[]                                    #create empty list to add series of riemann rectangles
            for n in range(6):
                a=ax.get_riemann_rectangles(
                quad, 
                x_range=[2,8],
                dx=(0.5)/(2**n),
                color=[REANLEA_BLUE,REANLEA_YELLOW],
                stroke_width=(0.25)/(2**n),
                fill_opacity=0.75,
                )
                grrl.append(a)                          #add created rectangle to previous(appended) list
                

            grp=VGroup(ax_label,quad,*grrl)             #making a vector group to transform it to another object
            txt1=Tex("REANLEA.com").set_color_by_gradient(REANLEA_ORANGE,REANLEA_VIOLET,REANLEA_GREEN_LIGHTER)
            txt2=Tex("HAR GHAR TIRANGA").set_color_by_gradient(REANLEA_ORANGE,REANLEA_VIOLET,REANLEA_GREEN_LIGHTER)

            

            self.play(
                Create(ax_label)
            )
            self.play(
                Create(quad, run_time=3)
            )
            self.play(
                Create(flat_rects)
            )
            self.wait()
            self.play(
                Transform(flat_rects,grrl[0],replace_mobject_with_target_in_scene=True)   
            )
            for r in range (1,len(grrl)):
                self.play(
                    Transform(
                        grrl[r-1],
                        grrl[r],
                        replace_mobject_with_target_in_scene=True
                    ),
                    run_time=3
                )
            self.wait(2)

            self.play(
                Transform(grp,txt1),
                run_time=2
            )
            self.wait(2)
            self.play(
                *[FadeOut(mobj) for mobj in self.mobjects],    #make fading out and transformation in one scene
                Transform(txt1,txt2),
                run_time=2
            )
            self.wait(2)


            # manim -pqh test.py ItrRiemannRects


class FontCheck(Scene):
    def construct(self):
         with RegisterFont("Montserrat") as fonts:
            tx=MarkupText("WHAT   ABOUT   D I S T A N C E ?", font=fonts[0]).scale(0.6)

         self.play(Write(tx))

         # manim -pqh test.py FontCheck