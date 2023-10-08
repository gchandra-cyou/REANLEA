from __future__ import annotations

import sys
sys.path.insert(1,'C:\\Users\\gchan\\Desktop\\REANLEA\\2023\\common')

from reanlea_colors  import*
from func import*

from asyncore import poll2
from audioop import add
from cProfile import label
from distutils import text_file

import math
from math import pi


from manim import *
from manim_physics import *
import pandas


from numpy import array
import numpy as np
import random as rd
from dataclasses import dataclass
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

from manim.mobject.geometry.arc import Circle
from manim.mobject.geometry.polygram import Square, Triangle
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.types.vectorized_mobject import VMobject
from manim.utils.space_ops import angle_of_vector
from manim.mobject.geometry.tips import ArrowTriangleFilledTip
from manim.mobject.geometry.tips import ArrowTip
from manim.mobject.geometry.tips import ArrowTriangleTip
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import root_scalar
import requests
import io
from PIL import Image
from random import choice, seed
from random import random, seed
from enum import Enum
from scipy.stats import norm, gamma
from scipy.optimize import fsolve

from manim.utils.unit import Pixels

from manim import *
from random import random, seed
from enum import Enum

config.background_color= REANLEA_BACKGROUND_COLOR
config.max_files_cached=500


###################################################################################################################

class ArgMinExample(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-4.5, 4.5], y_range=[0, 30, 10], axis_config={"include_tip": False}
        )
        labels = ax.get_axis_labels(x_label="x", y_label="f(x)")


        def func(x):
            return (0.1*(x**4)) - (x**2) + .5*x + 9 + 5*np.sin(x)
        graph = ax.plot(func, x_range=[-4,4], color=REANLEA_BLUE)
        xt = ValueTracker(-4)
        yt = ValueTracker(30)

        initial_point = [ax.coords_to_point(xt.get_value(), yt.get_value())]
        dot = Dot(point=initial_point)

        def updater1(mobj):
            mobj.move_to(ax.c2p(xt.get_value(), yt.get_value()))

        dot.add_updater(updater1)
        self.add(ax,labels,graph,dot)
        self.wait(1)
        self.play(yt.animate.set_value(func(xt.get_value())))
        self.remove_updater(updater1)

        dot.add_updater(lambda x: x.move_to(ax.c2p(xt.get_value(), func(xt.get_value()))))
        #x_space = np.linspace(*ax.x_range[:2],200)
        x_space=np.linspace(-4,3,200)
        minimum_index = func(x_space).argmin()

        '''The animation sets the X value of the dot to the X value of the minimum of the function. 
        This x-index for the minimum is computed and extracted using the .argmin() method.'''

        self.add(ax, labels, graph, dot)
        self.play(xt.animate.set_value(x_space[minimum_index]))
        self.wait()

        # manim -pqh anim.py ArgMinExample


        # visit : https://towardsdatascience.com/take-your-python-visualizations-to-the-next-level-with-manim-ce9ad7ff66bf



class ArgMinEx9(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-4.5, 4.5], y_range=[0, 30, 10], axis_config={"include_tip": False}
        )
        labels = ax.get_axis_labels(x_label="x", y_label="f(x)")
        def func(x):
            return (0.1*(x**4)) - (x**2) + .5*x + 9 + 5*np.sin(x)
        graph = ax.plot(func, x_range=[-4,4], color=REANLEA_BLUE)

        xts = [ValueTracker(i) for i in range (-4,5)]
        yts = [ValueTracker(30) for i in range (9)]

        points=[[xts[i].get_value(), yts[i].get_value()] for i in range(9)]
        dots=[Dot(point=ax.c2p(*p)) for p in points]
        dot_grp=VGroup(*dots)

        def updater_1(mobj, idx):
            mobj.add_updater(
                lambda x: x.move_to(ax.c2p(xts[idx].get_value(),yts[idx].get_value()))
            )

        for i,d in enumerate(dot_grp):
            updater_1(d,i)

        falling_dots=[y.animate.set_value(func(x.get_value())) for x,y in zip(xts,yts)]

        self.add(ax,labels,graph,dot_grp)
        self.wait()

        self.play(LaggedStart(*falling_dots, lag_ratio=0.2))

        for dot in dot_grp:
            dot.remove_updater(updater_1)
        

        def updater_2(mobj, idx):
            mobj.add_updater(
                lambda x: x.move_to(ax.c2p(xts[idx].get_value(), func(xts[idx].get_value())))
            )

        for i, d in enumerate(dot_grp):
            updater_2(d,i)

        x_space_left=np.linspace(-4, 1.32, 300)
        x_space_right=np.linspace(1.32, 4, 150)
        min_idx_left= func(x_space_left).argmin()
        min_idx_right= func(x_space_right).argmin()

        rolling_dot_left=[x.animate.set_value(x_space_left[min_idx_left]) for x in xts if x.get_value() < 1.32]
        rolling_dot_right=[x.animate.set_value(x_space_right[min_idx_right]) for x in xts if x.get_value() >= 1.32]

        rolling_dots=rolling_dot_left+rolling_dot_right

        self.play(LaggedStart(*rolling_dots, lag_ratio=0.03), run_time=2.5)
        self.wait(2)


        # manim -pqh anim.py ArgMinEx9



class ArgMinEx99(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-4.5, 4.5], y_range=[0, 30, 10], axis_config={"include_tip": False}
        )
        labels = ax.get_axis_labels(x_label="x", y_label="f(x)")
        def func(x):
            return (0.1*(x**4)) - (x**2) + .5*x + 9 + 5*np.sin(x)
        graph = ax.plot(func, x_range=[-4,4], color=REANLEA_BLUE)

        xts = [ValueTracker(i) for i in np.linspace(-4,4,99)]
        yts = [ValueTracker(30) for i in range (99)]

        points=[[xts[i].get_value(), yts[i].get_value()] for i in range(99)]
        dots=[Dot(point=ax.c2p(*p)) for p in points]
        dot_grp=VGroup(*dots)

        def updater_1(mobj, idx):
            mobj.add_updater(
                lambda x: x.move_to(ax.c2p(xts[idx].get_value(),yts[idx].get_value()))
            )

        for i,d in enumerate(dot_grp):
            updater_1(d,i)

        falling_dots=[y.animate.set_value(func(x.get_value())) for x,y in zip(xts,yts)]

        self.add(ax,labels,graph,dot_grp)
        self.wait()

        self.play(LaggedStart(*falling_dots, lag_ratio=0.01))

        for dot in dot_grp:
            dot.remove_updater(updater_1)
        

        def updater_2(mobj, idx):
            mobj.add_updater(
                lambda x: x.move_to(ax.c2p(xts[idx].get_value(), func(xts[idx].get_value())))
            )

        for i, d in enumerate(dot_grp):
            updater_2(d,i)

        x_space_left=np.linspace(-4, 1.32, 300)
        x_space_right=np.linspace(1.32, 4, 150)
        min_idx_left= func(x_space_left).argmin()
        min_idx_right= func(x_space_right).argmin()

        rolling_dot_left=[x.animate.set_value(x_space_left[min_idx_left]) for x in xts if x.get_value() < 1.32]
        rolling_dot_right=[x.animate.set_value(x_space_right[min_idx_right]) for x in xts if x.get_value() >= 1.32]

        rolling_dots=rolling_dot_left+rolling_dot_right

        self.play(LaggedStart(*rolling_dots, lag_ratio=0.003), run_time=2.5)
        self.wait(2)


        # manim -pqh anim.py ArgMinEx99


class Sierpinski(Scene):

   config.disable_caching_warning=True
  
   def subdivide(self, square, n):

   # Subdivide a square of size n into 4 equal parts of size n/2.
   # Return the three subsquares in the upper left, lower left, and lower right corners as a VGroup.

        ULsq = Square(
            side_length=n/2, 
            color=BLACK, 
            fill_color=YELLOW, 
            fill_opacity=1,
            stroke_width=0.25
        ).align_to(square,LEFT+UP)
        LLsq = ULsq.copy().shift(DOWN*n/2)
        LRsq = LLsq.copy().shift(RIGHT*n/2)
        sqs = VGroup(ULsq,LLsq,LRsq)
        return sqs
 
    
   def construct(self):

        size = 6  # size of initial square
        orig_size = size
        iterations = 6  # numeber of iterations in construction

        title = Text("Sierpinski Triangle").to_edge(UP)
        self.play(Create(title))
        self.wait()
        S = Square(
            side_length=size, 
            color=BLACK, 
            fill_color=YELLOW, 
            fill_opacity=1,
            stroke_width=0.5
            ).to_edge(LEFT,buff=.75).shift(DOWN*0.3)
        text1 = Text("Start with a square", font_size=24).move_to([2,2,0])
        text2 = Text("Divide into 4 equal subsquares", font_size=24).align_to(text1,LEFT).shift(UP)
        text3 = Text("Remove the upper right square",font_size=24).align_to(text1,LEFT)
        text4 = Text("Repeat with each remaining subsquare", font_size=24).align_to(text1,LEFT).shift(DOWN*3) 
        textct = Text("Iteration 1",color=YELLOW).align_to(text1,LEFT).shift(DOWN*2)   

        # First iteration with instructions                   
        self.add(textct)
        self.wait(1)       
        self.add(text1)
        self.play(FadeIn(S))
        self.wait(1)
        self.add(text2)
        self.wait(0.2)
        vertLine = Line(S.get_left(), S.get_right(), color=BLACK,stroke_width=1)
        horizLine = Line(S.get_top(), S.get_bottom(), color=BLACK,stroke_width=1)
        self.play(Create(vertLine), Create(horizLine), run_time=2)                 
        B=[0]
        B[0] = self.subdivide(S,size)
        self.wait(1)
        self.add(text3)
        self.wait(0.5)         
        self.add(*B[0])
        self.play(FadeOut(S), run_time=1.5)
        self.wait(1)
 

        # temporarily split off the three subsquares to illustrate construction on each subsquare
        # and draw lines to split each into 4 additional subsquares
        self.remove(textct)
        textct = Text("Iteration "+str(2), color=YELLOW).align_to(text1,LEFT).shift(DOWN*2)
        self.add(textct)
        self.wait(1)
        self.add(text4)   
        self.wait(1) 
        self.play(B[0][0].animate.shift(UP*0.5),B[0][2].animate.shift(RIGHT*0.5))
        self.wait(1)
        for k in range(3):
           vertLine = Line(B[0][k].get_left(), B[0][k].get_right(), color=BLACK,stroke_width=1)
           horizLine = Line(B[0][k].get_top(), B[0][k].get_bottom(), color=BLACK,stroke_width=1)
           self.play(Create(vertLine),Create(horizLine), run_time=0.33)
           self.wait(0.5)

        # Remaining iterations
        for m in range(0,iterations-1):
           size=size/2
           C = [0]*(3**(m+1))
           if (m > 0): self.wait(1.5)
           for k in range(3**m):
              C[3*k]=self.subdivide(B[k][0],size)
              C[3*k+1]=self.subdivide(B[k][1],size)
              C[3*k+2]=self.subdivide(B[k][2],size)
              self.add(*C[3*k],*C[3*k+1],*C[3*k+2])             
              self.remove(*B[k]) 
           if (m == 0): # recombine the squares of iteration 1 back into place
              self.wait(1)
              self.play(C[0].animate.shift(DOWN*0.5),C[2].animate.shift(LEFT*0.5))
           self.remove(textct)
           textct = Text("Iteration "+str(m+2), color=YELLOW).align_to(text1,LEFT).shift(DOWN*2)
           self.add(textct)            
           if (m < iterations-2): B = C.copy()
        self.wait(2)

        # Demonstrate self-similarity
        self.remove(text1,text2,text3,text4,textct)
        self.wait(1)

        VGTL = VGroup()  # top left corner
        VGBL = VGroup()  # bottom left corner
        VGBR = VGroup()  # bottom right corner
        m = 3**(iterations-2)
        for k in range(m):
           VGTL += C[k] 
        for k in range(m,2*m):
           VGBL += C[k]
        for k in range(2*m,3*m):
           VGBR += C[k]

        # Method 1 - show each corner is self-similar

        # set colors
        text4 = Text("Three self-similar pieces", color=YELLOW).move_to([2,2,0])
        self.add(text4)

        self.play(VGTL.animate.set_color(PURE_RED)) 
        self.play(VGBL.animate.set_color(PURE_GREEN)) 
        self.play(VGBR.animate.set_color(ORANGE)) 
        VGTL.save_state()  # save corners in current colors for method 2
        VGBL.save_state()
        VGBR.save_state()
        # shift the three corners apart to illustrate self-similarity
        self.play(VGTL.animate.shift(UP*0.5),VGBR.animate.shift(RIGHT*0.5))
        self.wait(1.5)
        # shift back and restore colors
        self.play(VGTL.animate.shift(DOWN*0.5),VGBR.animate.shift(LEFT*0.5))
        self.play(
            VGTL.animate.set_color(YELLOW).set_fillcapacity(1), 
            VGBL.animate.set_color(YELLOW).set_fillcapacity(1), 
            VGBR.animate.set_color(YELLOW).set_fillcapacity(1)
            )    
        self.wait(1.5)  

        # Method 2 - iterated function system

        # Combine all corners into one mobject and make a copy
        VGall = VGroup(*VGTL,*VGBL,*VGBR)
        VGallcp = VGall.copy().set_color(GRAY)       
        VGall.save_state()
        
        self.remove(text4)
        text5 = Text("Iterated Function System", color=YELLOW).move_to([2,2,0])
        self.add(text5)
        self.wait(1)

        # first scaling and translation to upper left corner
        text5 = MarkupText(f'<span fgcolor="{PURE_RED}" weight="{BOLD}">1.</span> Scale by 1/2, translate up', 
                           color=YELLOW, font_size=30
                           ).move_to([2.75,1,0])
        self.add(text5)
        self.add(VGallcp,VGall)
        self.play(Transform(VGall,VGBL),run_time=2)  
        self.play(VGall.animate.set_color(PURE_RED))
        self.play(VGall.animate.shift(UP*orig_size/2))
        self.wait(2)

        # second scaling to lower left corner (no translation)
        text6 = MarkupText(f'<span fgcolor="{PURE_GREEN}" weight="{BOLD}">2.</span> Scale by 1/2', 
                           color=YELLOW, font_size=30
                           ).align_to(text5, LEFT)        
        VGall.restore()      
        self.add(text6)
        self.play(Transform(VGall,VGBL),run_time=2)
        self.play(VGall.animate.set_color(PURE_GREEN))        
        self.wait(2)

        # third scaling and translation to lower right corner
        text7 = MarkupText(f'<span fgcolor="{ORANGE}" weight="{BOLD}">3.</span> Scale by 1/2, translate right', 
                           color=YELLOW, font_size=30
                           ).align_to(text5, LEFT).shift(DOWN)       
        VGall.restore()
        self.add(text7)
        self.play(Transform(VGall,VGBL),run_time=2)
        self.play(VGall.animate.set_color(ORANGE))
        self.play(VGall.animate.shift(RIGHT*orig_size/2))
        self.wait(2)

        # restore the three (self-similar) corners to their individual colors
        VGTL.restore()
        VGBL.restore()
        VGBR.restore()
        self.wait(4)


        # manim -sqk anim.py Sierpinski

        # manim -pqh anim.py Sierpinski



class StereographicProjection(ThreeDScene):
        def construct(self):
                resolution_fa = 1
                self.set_camera_orientation(phi=75 * DEGREES, theta=-160 * DEGREES)
                axes = ThreeDAxes(x_range=(-5, 5, 1), y_range=(-5, 5, 1), z_range=(-1, 1, 0.5))
                # def param_surface(u, v):
                #       x = u
                #       y = v

                #       return z
                def pointOnSphere(x, y):
                        modz = np.sqrt(x**2+y**2)
                        return [4*x/(modz**2+4), 4*y/(modz**2+4), 2*modz**2/(modz**2+4)]
                surface_plane = Surface(
                        lambda u, v: axes.c2p(u, v, 0),
                        resolution=(resolution_fa, resolution_fa),
                        v_range=[-5, 5],
                        u_range=[-5, 5],
                        ).set_color(REANLEA_PURPLE)
                surface_plane.set_style(fill_opacity=1)
                sphere = Sphere(center = [0, 0, 1], resolution = (32, 32), fill_opacity = 0.1).set_color(GRAY)
                # surface_plane.set_fill_by_value(axes=axes, colors=[(RED, -0.5), (YELLOW, 0), (GREEN, 0.5)], axis=2)
                line = Line(start = [-3, -3, 0], end = [0, 0, 2], **{"stroke_width": 1}).set_color(REANLEA_GREEN)
                point = [-3, -3, 0]
                dot1 = Sphere(center = point, radius = 0.05, resolution = (5, 5))
                dot2 = Sphere(center = pointOnSphere(-3, -3), radius = 0.05, resolution = (5, 5)).set_color(REANLEA_BLUE)
                # print(dot1Temp.get_center(random.randit))
                # dot1.add_updater(
                #       lambda mobject: mobject.move_to(dot1Temp.get_center()))
                dot2.add_updater(
                        lambda mobject: mobject.move_to(pointOnSphere(dot1.get_center()[0], dot1.get_center()[1]))
                        )
                line = always_redraw(lambda: Line(start = [0, 0, 2], end = dot1.get_center(), **{"stroke_width": 1}))
                self.add(axes, surface_plane, sphere, line, dot1, dot2)
                self.wait(2)
                for i in range(10):
                        self.play(dot1.animate.move_to([np.random.randint(-5, 5), np.random.randint(-5, 5), 0]))
                self.wait()
                self.play(dot1.animate.move_to([10,10,0]), run_time = 2)
                self.play(dot1.animate.move_to([-10, 10, 0]), run_time = 2)
                self.play(dot1.animate.move_to([-10, -10, 0]), run_time = 2)
                self.play(dot1.animate.move_to([10, -10, 0]), run_time = 2)
                self.play(dot1.animate.move_to([100, -100, 0]), run_time = 3)
                self.wait()


        # manim -pqh anim.py StereographicProjection   


class PCA(ThreeDScene):
    def construct(self):
        self.begin_ambient_camera_rotation(0.2)
        self.set_camera_orientation(phi=45 * DEGREES, theta=-45 * DEGREES)
        self.add(ThreeDAxes())
        centers = [[1, 1.2, 1.6], [0.9, 0.2, 2.9], [0.4, 0.5, 1.8]]
        # create dataset6
        X, y = make_blobs(
            n_samples=40, n_features=3,
            centers=centers, cluster_std=0.4,
            shuffle=False, random_state=0
        )
        A = X.T[0]
        B = X.T[1]
        C = X.T[2]

        Dots = VGroup(*[Dot(point=(a, b, c)) for a, b, c in zip(A, B, C)])
        Dots_shaddow = Dots.copy().set_opacity(0.3)
        Dots_z0 = Dots.copy().set_color(REANLEA_GREEN)
        rect = Rectangle(width=3, height=3, fill_color=REANLEA_AQUA, fill_opacity=0.2, stroke_color=WHITE).set_shade_in_3d(True)
        rect.shift(UP + RIGHT)
        self.add(rect)
        self.add(Dots)
        self.add(Dots_shaddow)
        [objec.set_z(0) for objec in Dots_z0.submobjects]
        self.wait()
        self.play(Transform(Dots, Dots_z0), run_time=1)
        self.wait(3)


        # manim -pqh anim.py PCA



class Lorenz_Attractor_0(ThreeDScene):
    def construct(self):
        r = 50
        x = float(25)
        y = float(15)
        z = float(15)
        a = 10
        b = 28
        c = 8/3
        camP = PI/3
        i = 0
        di = 0.01
        self.set_camera_orientation(phi=camP,theta=-PI/4)
        axis = ThreeDAxes(x_range=(- r, r , r/10), y_range=(- r, r, r/10), z_range=(- r, r, r/10))
        self.add(axis)
        points = []
        while i<10:
            pnt = axis.coords_to_point(x, y, z)
            points.append(pnt)
            i = i + di
            dx = a*(y-x)
            dy = x*(b-z) - y
            dz = x*y - c*z
            x += di*dx
            y += di*dy
            z += di*dz
        LorenzPath = VMobject(color=RED)
        LorenzPath.set_points_smoothly(points)

        self.begin_ambient_camera_rotation(rate=0.25)
        self.play(
            Create(LorenzPath), rate_func=linear, run_time=10
        )


        # manim -pqh anim.py Lorenz_Attractor_0


class Lorenz_Attractor(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(x_range=[-3.5,3.5],y_range=[-3.5,3.5],z_range=[0,6],axis_config={"include_tip": True,"include_ticks":True,"stroke_width":1})
        dot = Sphere(radius=0.05,fill_color=BLUE).move_to(0*RIGHT + 0.1*UP + 0.105*OUT)
        
         
        self.set_camera_orientation(phi=5 * DEGREES,theta=30*DEGREES,gamma = 90*DEGREES)  
        self.begin_ambient_camera_rotation(rate=0.5)            #Start move camera
 
        dtime = 0.01
        numsteps = 30
 
        self.add(axes,dot)
 
        def lorenz(x, y, z, s=10, r=28, b=2.667):
            x_dot = s*(y - x)
            y_dot = r*x - y - x*z
            z_dot = x*y - b*z
            return x_dot, y_dot, z_dot
 
        def update_trajectory(self, dt):
            new_point = dot.get_center()
            if np.linalg.norm(new_point - self.points[-1]) > 0.01:                             #numpy method for get_norm 
                self.add_smooth_curve_to(new_point)
 
        traj = VMobject()
        traj.start_new_path(dot.get_center())
        traj.set_stroke(REANLEA_BLUE_LAVENDER, 1.5, opacity=0.8)
        traj.add_updater(update_trajectory)
        self.add(traj)
 
        def update_position(self,dt):
            x_dot, y_dot, z_dot = lorenz(dot.get_center()[0]*10, dot.get_center()[1]*10, dot.get_center()[2]*10)
            x = x_dot * dt/10
            y = y_dot * dt/10
            z = z_dot * dt/10
            self.shift(x/10*RIGHT + y/10*UP + z/10*OUT)
 
        dot.add_updater(update_position)
        self.wait(5)


        # manim -pqh anim.py Lorenz_Attractor


             


                             #########    Curious Walk Work    ###########

class Cycloid(Scene):

    def construct(self):

        CycloidTxt = Text("Cycloid", font="TeX Gyre Termes").scale(1.5).to_edge(UP)

        r = 3 / PI
        corr = 1 / config.frame_rate  # missed frame correction

        BL = NumberLine().shift(DOWN * r * 2)  # Base Line

        C = Circle(r, color="#F72119").next_to(BL.n2p(-6), UP, buff=0)
        DL = DashedLine(C.get_center(), C.get_top(), color="#A5ADAD")
        CP = Dot(DL.get_start(), color="#ff3503")  # Center Point
        TP = Dot(DL.get_end(), color="#00EAFF").scale(1.2)  # Tracing Point

        RC = VGroup(C, DL, CP, TP)  # Rolling Circle

        self.dir = 1  # direction of motion

        def Rolling(m, dt):  # update_function
            theta = self.dir * -PI
            m.rotate(dt * theta, about_point=m[0].get_center()).shift(dt * LEFT * theta * r)

        Cycloid = TracedPath(TP.get_center, stroke_width=6.5, stroke_color="#4AF1F2")

        self.add(CycloidTxt, BL, Cycloid, RC)

        RC.add_updater(Rolling)
        self.wait(4 + corr)

        RC.suspend_updating(Rolling)
        Cycloid.clear_updaters()

        self.wait()
        self.dir = -1  # direction change, rolling back

        RC.resume_updating(Rolling)
        self.play(Uncreate(Cycloid, rate_func=lambda t: linear(1 - t), run_time=4 + corr))
            
        RC.clear_updaters()
        self.wait()


        # manim -pqh anim.py Cycloid



class FibonacciSpiral(Scene):
    def construct(self):

        squares = VGroup(Square(1 * 0.3))
        next_dir = [RIGHT, UP, LEFT, DOWN]
        FSeq = [1, 2, 3, 5, 8, 13, 21]

        for j, i in enumerate(FSeq):
            d = next_dir[j % 4]
            squares.add(Square(i * 0.3).next_to(squares, d, buff=0))

        squares.center()

        direction = [1, -1, -1, 1]
        corner = [[UL, -UL], [UR, -UR]]
        spiral = VGroup()

        for j, i in enumerate(squares):
            c = corner[j % 2]
            d = direction[j % 4]
            arc = ArcBetweenPoints(
                i.get_corner(c[0]),
                i.get_corner(c[1]),
                angle=PI / 2 * d,
                color="#04d9ff",
                stroke_width=6,
            )
            if direction[j % 4] != 1:
                arc = arc.reverse_direction()
            spiral.add(arc)

        self.play(
            LaggedStart(
                FadeIn(squares, lag_ratio=1), Create(spiral, lag_ratio=1), run_time=5
            )
        )
        self.wait()

        self.play(FadeOut(squares), Uncreate(spiral[::-1]), run_time=1.5)



        # manim -pqh anim.py FibonacciSpiral




class CW_KochCurve(Scene):
    def construct(self):
        def KochCurve(
            n, length=12, stroke_width=8, color=("#0A68EF", "#4AF1F2", "#0A68EF")
        ):

            l = length / (3 ** n)

            LineGroup = Line().set_length(l)

            def NextLevel(LineGroup):
                return VGroup(
                    *[LineGroup.copy().rotate(i) for i in [0, PI / 3, -PI / 3, 0]]
                ).arrange(RIGHT, buff=0, aligned_edge=DOWN)

            for _ in range(n):
                LineGroup = NextLevel(LineGroup)

            KC = (
                VMobject(stroke_width=stroke_width)
                .set_points(LineGroup.get_all_points())
                .set_color(color)
            )
            return KC

        level = Variable(0, Tex("level"), var_type=Integer).set_color("#4AF1F2")
        txt = (
            VGroup(Tex("Koch Curve", font_size=60), level)
            .arrange(DOWN, aligned_edge=LEFT)
            .to_corner(UL)
        )
        kc = KochCurve(0, stroke_width=12).to_edge(DOWN, buff=2.5)

        self.add(txt, kc)
        self.wait()

        for i in range(1, 6):
            self.play(
                level.tracker.animate.set_value(i),
                kc.animate.become(
                    KochCurve(i, stroke_width=12 - (2 * i)).to_edge(DOWN, buff=2.5)
                ),
            )
            self.wait()

        for i in range(4, -1, -1):
            self.play(
                level.tracker.animate.set_value(i),
                kc.animate.become(
                    KochCurve(i, stroke_width=12 - (2 * i)).to_edge(DOWN, buff=2.5)
                ),
            )
            self.wait()


            #  manim -pqh anim.py CW_KochCurve




class Sunflower1(Scene):
    def construct(self):
        # Golden Angle
        GA = PI * (3 - 5 ** 0.5)

        def Bloom(a=GA, n=300):
            Plane = PolarPlane()
            flower = VGroup()
            for i in range(n):
                r = 0.025 + (0.00043 * i)
                pos = Plane.polar_to_point(0.03 + (0.01 * i), a * (n - i))
                seed = Dot(
                    point=pos,
                    radius=r,
                    color=interpolate_color("#c3ff12", "#ffc512", i / (n - 1)),
                )
                flower.add(seed)
            return flower

        val = ValueTracker(3 * PI / 4)

        ang = MathTex(r"\theta = ", font_size=55)
        num = DecimalNumber(
            val.get_value() * 180 / PI,
            5,
            show_ellipsis=True,
            unit="^\circ",
            font_size=53,
        ).next_to(ang)
        ang.add(num).to_corner(UL)

        sf = Bloom(val.get_value())

        # Updaters
        num.add_updater(lambda m: m.set_value(val.get_value() * 180 / PI))
        sf.add_updater(lambda m: m.become(Bloom(val.get_value())))
        self.add(ang, sf)

        self.wait()
        self.play(val.animate(rate_func=linear, run_time=15).set_value(GA))

        VGroup(sf, num).clear_updaters()
        self.wait()



        # manim -pqh anim.py Sunflower1


class Sunflower2(Scene):
    def construct(self):
        # Golden Angle
        GA = PI * (3 - 5 ** 0.5)

        def Bloom(a=GA, n=300):
            Plane = PolarPlane()
            flower = VGroup()
            for i in range(n):
                r = 0.025 + (0.00043 * i)
                pos = Plane.polar_to_point(0.03 + (0.01 * i), a * (n - i))
                seed = Dot(
                    point=pos,
                    radius=r,
                    color=interpolate_color("#c3ff12", "#ffc512", i / (n - 1)),
                )
                flower.add(seed)
            return flower

        sf = Bloom()
        self.add(sf)

        self.wait()
        self.play(*[FadeToColor(i, "#F0DFC5") for i in sf], lag_ratio=0.005)
        self.wait()

        # list of colors
        c = ["#ff0000", "#00ffff", "#00ff00", "#ff00ff", "#0000ff", "#ffff00"]

        # colorful spirals animation
        for n, m in [[21, 8], [34, 13]]:  # Fibonacci Numbers
            Anim1 = AnimationGroup(
                *[
                    FadeToColor(sf[(i * m) % n :: n], c[i % 6], lag_ratio=0.2)
                    for i in range(n)
                ],
                lag_ratio=0.15
            )
            Anim2 = AnimationGroup(
                *[
                    FadeToColor(sf[(i * m) % n :: n], "#F0DFC5", lag_ratio=0.5)
                    for i in range(n)
                ]
            )
            self.play(Anim1)
            self.wait(3)

            self.play(Anim2)
            self.wait(2)



            # manim -pqh anim.py Sunflower2




class SubstitutionSystem(Scene):
    def construct(self):
        depth = 6
        base_s = Square(
            side_length=config["pixel_height"] * Pixels, fill_color=REANLEA_BACKGROUND_COLOR, fill_opacity=1, stroke_width=0
        )
        self.add(base_s)
        fractal = VGroup(base_s)
        old_fractal = VGroup()

        for _ in range(depth):
            old_fractal = fractal.copy()
            print(fractal)
            fractal = VGroup()
            color = random_bright_color()
            for s in old_fractal:
                s_g = VGroup(*[
                        s.copy().scale(0.5)
                        for _ in range(4)
                    ]).arrange_in_grid(cols=2, rows=2, buff=0)
                s_g[0].set_fill(color, 1)
                fractal.add(s_g)

            self.play(FadeTransform(old_fractal, fractal))



            # manim -pqh anim.py SubstitutionSystem



#----------------------------------------------------------------------------------------#

##########  https://slama.dev/manim/3d-and-the-other-graphs/  #########

class MoveAndFade(Animation):
    def __init__(self, mobject: Mobject, path: VMobject, **kwargs):
        self.path = path
        self.original = mobject.copy()
        super().__init__(mobject, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        point = self.path.point_from_proportion(self.rate_func(alpha))

        # this is not entirely clean sice we're creating a new object
        # this is because obj.fade() doesn't add opaqueness but adds it
        self.mobject.become(self.original.copy()).move_to(point).fade(alpha)




def create_graph(x_values, y_values):
        """Build a graph with the given values."""
        y_values_all = list(range(0, (max(y_values) or 1) + 1))
        n=9
        axes = (
            Axes(
                x_range=[-n // 2 + 1, n // 2, 1],
                y_range=[0, max(y_values) or 1, 1],
                x_axis_config={"numbers_to_include": x_values},
                tips=False,
            )
            .scale(0.45)
            .shift(LEFT * 3.0)
        )

        graph = axes.plot_line_graph(x_values=x_values, y_values=y_values)

        return graph, axes


class coordsysDistributionSimulation(Scene):
    def construct(self):
        seed(0xDEADBEEF2)  # hezčí vstupy :)

        radius = 0.13
        x_spacing = radius * 1.5
        y_spacing = 4 * radius

        n = 9
        pyramid = VGroup()
        pyramid_values = []  # how many marbles fell where

        # build the pyramid
        for i in range(1, n + 1):
            row = VGroup()

            for j in range(i):
                obj = Dot()

                # if it's the last row, make the rows numbers instead
                if i == n:
                    obj = Tex("0")
                    pyramid_values.append(0)

                row.add(obj)

            row.arrange(buff=2 * x_spacing)

            if len(pyramid) != 0:
                row.move_to(pyramid[-1]).shift(DOWN * y_spacing)

            pyramid.add(row)

        pyramid.move_to(RIGHT * 3.4)

        x_values = np.arange(-n // 2 + 1, n // 2 + 1, 1)

        graph, axes = create_graph(x_values, pyramid_values)

        self.play(Write(axes), Write(pyramid), Write(graph), run_time=1.5)

        for iteration in range(120):
            circle = (
                Circle(fill_opacity=1, stroke_opacity=0)
                .scale(radius)
                .next_to(pyramid[0][0], UP, buff=0)
            )

            # go faster and faster
            run_time = (
                0.5
                if iteration == 0
                else 0.1
                if iteration == 1
                else 0.02
                if iteration < 20
                else 0.003
            )

            self.play(FadeIn(circle, shift=DOWN * 0.5), run_time=run_time * 2)

            x = 0
            for i in range(1, n):
                next_position = choice([0, 1])
                x += next_position

                dir = LEFT if next_position == 0 else RIGHT

                circle_center = circle.get_center()

                # behave normally when it's not the last row
                if i != n - 1:
                    b = CubicBezier(
                        circle_center,
                        circle_center + dir * x_spacing,
                        circle_center + dir * x_spacing + DOWN * y_spacing / 2,
                        circle.copy().next_to(pyramid[i][x], UP, buff=0).get_center(),
                    )

                    self.play(
                        MoveAlongPath(circle, b, rate_func=rate_functions.ease_in_quad),
                        run_time=run_time,
                    )

                # if it is, animate fadeout and add
                else:
                    b = CubicBezier(
                        circle_center,
                        circle_center + dir * x_spacing,
                        circle_center + dir * x_spacing + DOWN * y_spacing / 2,
                        pyramid[i][x].get_center(),
                    )

                    pyramid_values[x] += 1

                    n_graph, n_axes = create_graph(x_values, pyramid_values)

                    self.play(
                        AnimationGroup(
                            AnimationGroup(
                                MoveAndFade(
                                    circle, b, rate_func=rate_functions.ease_in_quad
                                ),
                                run_time=run_time,
                            ),
                            AnimationGroup(
                                pyramid[i][x]
                                .animate(run_time=run_time)
                                .become(
                                    Tex(str(pyramid_values[x])).move_to(pyramid[i][x])
                                ),
                                graph.animate.become(n_graph),
                                axes.animate.become(n_axes),
                                run_time=run_time,
                            ),
                            lag_ratio=0.3,
                        )
                    )

        self.play(FadeOut(axes), FadeOut(pyramid), FadeOut(graph), run_time=1)



        # manim -pqh anim.py coordsysDistributionSimulation




# inspired by https://softologyblog.wordpress.com/2019/12/28/3d-cellular-automata-3/


class Grid:
    class ColorType(Enum):
        FROM_COORDINATES = 0
        FROM_PALETTE = 1

    def __init__(
        self,
        scene,
        grid_size,
        survives_when,
        revives_when,
        state_count=2,
        size=1,
        palette=["#000b5e", "#001eff"],
        color_type=ColorType.FROM_PALETTE,
    ):
        self.grid = {}
        self.scene = scene
        self.grid_size = grid_size
        self.size = size
        self.survives_when = survives_when
        self.revives_when = revives_when
        self.state_count = state_count
        self.palette = palette
        self.color_type = color_type

        self.bounding_box = Cube(side_length=self.size, color=GRAY, fill_opacity=0.05)
        self.scene.add(self.bounding_box)

    def fadeIn(self):
        self.scene.play(
            FadeIn(self.bounding_box),
        )

    def fadeOut(self):
        self.scene.play(
            FadeOut(self.bounding_box),
            *[FadeOut(self.grid[index][0]) for index in self.grid],
        )

    def __index_to_position(self, index):
        """Convert the index of a cell to its position in 3D."""
        dirs = [RIGHT, UP, OUT]

        # be careful!
        # we can't just add stuff to ORIGIN, since it doesn't create new objects,
        # meaning we would be moving the origin, which messes with the animations
        result = list(ORIGIN)
        for dir, value in zip(dirs, index):
            result += ((value - (self.grid_size - 1) / 2) / self.grid_size) * dir * self.size

        return result

    def __get_new_cell(self, index):
        """Create a new cell"""
        cell = (
            Cube(
                side_length=1 / self.grid_size * self.size, color=BLUE, fill_opacity=1
            ).move_to(self.__index_to_position(index)),
            self.state_count - 1,
        )

        self.__update_cell_color(index, *cell)

        return cell

    def __return_neighbouring_cell_coordinates(self, index):
        """Return the coordinates of the neighbourhood of a given index."""
        neighbourhood = set()
        for dx in range(-1, 1 + 1):
            for dy in range(-1, 1 + 1):
                for dz in range(-1, 1 + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue

                    nx = index[0] + dx
                    ny = index[1] + dy
                    nz = index[2] + dz

                    # don't loop around (although we could)
                    if (
                        nx < 0
                        or nx >= self.grid_size
                        or ny < 0
                        or ny >= self.grid_size
                        or nz < 0
                        or nz >= self.grid_size
                    ):
                        continue

                    neighbourhood.add((nx, ny, nz))

        return neighbourhood

    def __count_neighbours(self, index):
        """Return the number of neighbouring cells for a given index (excluding itself)."""
        total = 0
        for neighbour_index in self.__return_neighbouring_cell_coordinates(index):
            if neighbour_index in self.grid:
                total += 1

        return total

    def __return_possible_cell_change_indexes(self):
        """Return the indexes of all possible cells that could change."""
        changes = set()
        for index in self.grid:
            changes |= self.__return_neighbouring_cell_coordinates(index).union({index})
        return changes

    def toggle(self, index):
        """Toggle a given cell."""
        if index in self.grid:
            self.scene.remove(self.grid[index][0])
            del self.grid[index]
        else:
            self.grid[index] = self.__get_new_cell(index)
            self.scene.add(self.grid[index][0])

    def __update_cell_color(self, index, cell, age):
        """Update the color of the specified cell."""
        if self.color_type == self.ColorType.FROM_PALETTE:
            state_colors = color_gradient(self.palette, self.state_count - 1)

            cell.set_color(state_colors[age - 1])
        else:

            def coordToHex(n):
                return hex(int(n * (256 / self.grid_size)))[2:].ljust(2, "0")

            cell.set_color(
                f"#{coordToHex(index[0])}{coordToHex(index[1])}{coordToHex(index[2])}"
            )

    def do_iteration(self):
        """Perform the automata generation, returning True if a state of any cell changed."""
        new_grid = {}
        something_changed = False

        for index in self.__return_possible_cell_change_indexes():
            neighbours = self.__count_neighbours(index)

            # alive rules
            if index in self.grid:
                cell, age = self.grid[index]

                # always decrease age
                if age != 1:
                    age -= 1
                    something_changed = True

                # survive if within range or age isn't 1
                if neighbours in self.survives_when or age != 1:
                    self.__update_cell_color(index, cell, age)
                    new_grid[index] = (cell, age)
                else:
                    self.scene.remove(self.grid[index][0])
                    something_changed = True

            # dead rules
            else:
                # revive if within range
                if neighbours in self.revives_when:
                    new_grid[index] = self.__get_new_cell(index)
                    self.scene.add(new_grid[index][0])
                    something_changed = True

        self.grid = new_grid

        return something_changed


class GOLFirst(ThreeDScene):
    def construct(self):
        seed(0xDEADBEEF)

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=-0.20)

        grid_size = 16
        size = 3.5

        grid = Grid(
            self,
            grid_size,
            [4, 5],
            [5],
            state_count=2,
            size=size,
            color_type=Grid.ColorType.FROM_COORDINATES,
        )

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if random() < 0.2:
                        grid.toggle((i, j, k))

        grid.fadeIn()

        self.wait(1)

        for i in range(50):
            something_changed = grid.do_iteration()

            if not something_changed:
                break

            self.wait(0.2)

        self.wait(2)

        grid.fadeOut()

        # manim -pqh anim.py GOLFirst


class GOLSecond(ThreeDScene):
    def construct(self):
        seed(0xDEADBEEF)

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.15)

        grid_size = 16
        size = 3.5

        grid = Grid(
            self,
            grid_size,
            [2, 6, 9],
            [4, 6, 8, 9],
            state_count=10,
            size=size,
            color_type=Grid.ColorType.FROM_PALETTE,
        )

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if random() < 0.3:
                        grid.toggle((i, j, k))

        self.wait(2)

        for i in range(70):
            something_changed = grid.do_iteration()

            if not something_changed:
                break

            self.wait(0.1)

        self.wait(2)

        grid.fadeOut()


        # manim -pqh anim.py GOLSecond

#-------------------------------------------------------------------------------------#

class Sierpinski_01(Scene):

   config.disable_caching_warning=True
  
   def subdivide(self, square, n):
        ULsq = Square(
            side_length=n/2, 
            color=BLACK, 
            fill_color=YELLOW, 
            fill_opacity=1,
            stroke_width=0.25
        ).align_to(square,LEFT+UP)
        LLsq = ULsq.copy().shift(DOWN*n/2)
        LRsq = LLsq.copy().shift(RIGHT*n/2)
        sqs = VGroup(ULsq,LLsq,LRsq)
        return sqs
 
    
   def construct(self):

        size = 6  
        orig_size = size
        iterations = 7

        S = Square(
            side_length=size, 
            color=BLACK, 
            fill_color=YELLOW, 
            fill_opacity=1,
            stroke_width=0.5
            )                    
        
        self.play(FadeIn(S))
        self.wait(1)
                         
        B=[0]
        B[0] = self.subdivide(S,size)
        self.wait(1)
       
        self.add(*B[0])
        self.play(FadeOut(S), run_time=1.5)
        self.wait(1)


        # Remaining iterations
        for m in range(0,iterations-1):
           size=size/2
           C = [0]*(3**(m+1))
           if (m > 0): self.wait(1.5)
           for k in range(3**m):
              C[3*k]=self.subdivide(B[k][0],size)
              C[3*k+1]=self.subdivide(B[k][1],size)
              C[3*k+2]=self.subdivide(B[k][2],size)
              self.add(*C[3*k],*C[3*k+1],*C[3*k+2])             
              self.remove(*B[k]) 


           if (m == 0): 
              self.wait(.5)
                           
           if (m < iterations-2): B = C.copy()

        self.wait(2)


        # manim -sqk anim.py Sierpinski_01

        # manim -pqh anim.py Sierpinski_01


class Sierpinski_carpet(Scene):

   config.disable_caching_warning=True
  
   def subdivide(self, square, n):
        mid_sq = Square(
            side_length=n/3, 
            fill_color="#cca300", 
            fill_opacity=1,
            stroke_width=0
        ).move_to(square.get_center())
        sq_R = mid_sq.copy().shift(RIGHT*n/3)
        sq_UR = sq_R.copy().shift(UP*n/3)
        sq_U = mid_sq.copy().shift(UP*n/3)
        sq_UL = sq_U.copy().shift(LEFT*n/3)
        sq_L = mid_sq.copy().shift(LEFT*n/3)
        sq_DL = sq_L.copy().shift(DOWN*n/3)
        sq_D = mid_sq.copy().shift(DOWN*n/3)
        sq_DR = sq_D.copy().shift(RIGHT*n/3)
        
        
        mid_sq.set_fill("#cca300")
        
        sqs = VGroup(sq_R,sq_UR,sq_U,sq_UL,sq_L,sq_DL,sq_D,sq_DR,mid_sq)
        return sqs
 
    
   def construct(self):

        size = 6  
        iterations = 6

        S = Square(
            side_length=size,  
            fill_color="#00673A", 
            fill_opacity=1,
            stroke_width=0.5
            )                      
        
        self.play(FadeIn(S))
        self.wait(1)
                         
        B=[0]
        B[0] = self.subdivide(S,size)
        #self.play(FadeIn(B[0][-1]))
        
        
        # Remaining iterations
        
        for m in range(0,iterations-1):
           grp=VGroup()
           size=size/3
           C = [0]*(8**(m+1))
           if (m > 0): self.wait(1.5)
           for k in range(8**m):
              C[8*k]=self.subdivide(B[k][0],size)
              C[8*k+1]=self.subdivide(B[k][1],size)
              C[8*k+2]=self.subdivide(B[k][2],size)
              C[8*k+3]=self.subdivide(B[k][3],size)
              C[8*k+4]=self.subdivide(B[k][4],size)
              C[8*k+5]=self.subdivide(B[k][5],size)
              C[8*k+6]=self.subdivide(B[k][6],size)
              C[8*k+7]=self.subdivide(B[k][7],size)
              
              #self.add(*C[3*k],*C[3*k+1],*C[3*k+2])     
              #self.remove(*B[k]) 
              #self.add(*B[k][-1])
              
              #grp += VGroup(*B[k-1][-1]) 

              grp += VGroup(*B[k-1][-1])

           self.play(Write(grp))  
             
                   
           if (m == 0): # recombine the squares of iteration 1 back into place
              self.wait(.5)          
                           
           if (m < iterations-2): B = C.copy()
            
          
        self.wait(2)


        # manim -sqk anim.py Sierpinski_carpet

        # manim -pqh anim.py Sierpinski_carpet


class SierpinskiTriangle(Scene):
        
        def sub_triangle(self,triangle):
            vertices = triangle.get_vertices()
            a=vertices[0]
            b=vertices[1]
            c=vertices[2]
            tri_0=Polygon((a+b)/2,(b+c)/2,(c+a)/2).set_fill(color="#00673A",opacity=1).set_stroke(color="#00673A",width=0)
            tri_1=Polygon((a+b)/2,a,(c+a)/2)
            tri_2=Polygon((a+b)/2,(b+c)/2,b)
            tri_3=Polygon(c,(b+c)/2,(c+a)/2)

            tris=VGroup(tri_1,tri_2,tri_3,tri_0)

            return tris
        
        def construct(self):
            iterations=8

            Tri=Triangle().scale(4).set_stroke(width=0).set_fill(color="#cca300",opacity=1)

            self.play(FadeIn(Tri))
            self.wait()
            

            B=[0]
            B[0] = self.sub_triangle(Tri)

            for m in range(0,iterations-1):
                grp=VGroup()
                C = [0]*(3**(m+1))
                if (m > 0): self.wait(1.5)
                for k in range(3**m):
                    C[3*k]=self.sub_triangle(B[k][0])
                    C[3*k+1]=self.sub_triangle(B[k][1])
                    C[3*k+2]=self.sub_triangle(B[k][2])
                     
                    grp += VGroup(*B[k-1][-1])

                self.play(Write(grp))  
             
                   
                if (m == 0): # recombine the squares of iteration 1 back into place
                 self.wait(.5)          
                           
                if (m < iterations-2): B = C.copy()
            
          
            self.wait(2)



        # manim -pqh anim.py SierpinskiTriangle



config.background_color=REANLEA_BACKGROUND_COLOR_GHEE
class Cantor_Set(Scene):
        def subdivide(self, line):
            len=line.get_length()/3

            ln_0=Line().set_stroke(color=REANLEA_WARM_BLUE_DARKER,width=8).set_length(len).move_to(line.get_center())

            ln_1=ln_0.copy().shift(LEFT*len)
            ln_2=ln_0.copy().shift(RIGHT*len)

            ln_0.set_stroke(width=9, color=REANLEA_BACKGROUND_COLOR_GHEE)

            lns=VGroup(ln_1,ln_2,ln_0)
            return lns
        
        def construct(self):

            water_mark=ImageMobject('C:\\Users\\gchan\\Desktop\\REANLEA\\2023\\common\\watermark_ghee.png').scale(0.075).move_to(5*LEFT+3*UP).set_opacity(1).set_z_index(-100)
            self.add(water_mark)

            #anim zone 

            length=12
            iterations=9

            level = Variable(0, Tex("iterations:"), var_type=Integer).set_color(REANLEA_BACKGROUND_COLOR_OXFORD_BLUE)
            txt = (
                VGroup(Tex("Cantor Set", font_size=60).set_color(REANLEA_BACKGROUND_COLOR_OXFORD_BLUE), level).arrange(DOWN, aligned_edge=LEFT).to_corner(UR)
            )
            self.play(Write(txt))

            line=Line(color=WHITE).set_stroke(color=REANLEA_WARM_BLUE_DARKER,width=8).set_length(length)

            self.play(Create(line))
            self.wait()

            B=[0]
            B[0]=self.subdivide(line)
            

            for m in range(0,iterations-1):
                grp=VGroup()
            
                C = [0]*(2**(m+1))
                if (m > 0): self.wait(1.5)
                for k in range(2**m):
                    C[2*k]=self.subdivide(B[k][0])
                    C[2*k+1]=self.subdivide(B[k][1])
                                        
                    grp += VGroup(*B[k-1])
                
                grp.move_to(.2*m*DOWN)
                                         
                self.play(FadeIn(grp),level.tracker.animate.set_value(m+1))  
                
                   
                if (m == 0):
                 grp+=VGroup(B[0])

                self.play(grp.animate.shift(.2*DOWN))
                self.wait(.15)          
                           
                if (m < iterations-2): B = C.copy()
            
            self.wait(2)
            



        #  manim -pqh anim.py Cantor_Set

        #  manim -sqk anim.py Cantor_Set


###################################################################################################################

# Changing FONTS : import any font from Google
# some of my fav fonts: Cinzel,Kalam,Prata,Kaushan Script,Cormorant, Handlee,Monoton, Bad Script, Reenie Beanie, 
# Poiret One,Merienda,Julius Sans One,Merienda One,Cinzel Decorative, Montserrat, Cousine, Courier Prime , The Nautigal
# Marcellus SC,Contrail One,Thasadith,Spectral SC,Dongle,Cormorant SC,Comfortaa, Josefin Sans (LOVE), Fuzzy Bubbles

###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2023\lab" 