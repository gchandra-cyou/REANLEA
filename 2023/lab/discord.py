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
import random


config.background_color= REANLEA_BACKGROUND_COLOR


###################################################################################################################

class Charge3D(VGroup):
    def __init__(
        self,
        magnitude: float = 1,
        point: np.ndarray = ORIGIN,
        add_glow: bool = True,
        **kwargs,
    ) -> None:
        
        VGroup.__init__(self, **kwargs)
        self.magnitude = magnitude
        self.point = point
        self.radius = (abs(magnitude) * 0.4 if abs(magnitude) < 2 else 0.8) * 0.3

        if magnitude > 0:
            label = VGroup(Prism(dimensions=[.2,.05,.05],fill_color=WHITE,fill_opacity=1).set_z_index(6),
                           Prism(dimensions=[.05,.05,.2],fill_color=WHITE,fill_opacity=1).set_z_index(6))
            color = RED
            layer_colors = [RED_D, RED_A]
            layer_radius = 2
        else:
            label = Prism(dimensions=[.2,.1,.1],fill_color=WHITE,fill_opacity=1).set_z_index(6)
            color = BLUE
            layer_colors = ["#3399FF", "#66B2FF"]
            layer_radius = 2

        if add_glow:  # use many arcs to simulate glowing
            layer_num = 50
            color_list = color_gradient(layer_colors, layer_num)
            opacity_func = lambda t: 200 * (1 - abs(t - 0.009) ** 0.0001)
            rate_func = lambda t: t ** 2

            for i in range(layer_num):
                self.add(
                    Sphere(
                        radius=layer_radius * rate_func((0.5 + i) / layer_num),
                        stroke_width=0,
                        checkerboard_colors=[color_list[0],color_list[0]],
                        color=color_list[i],
                        fill_opacity=opacity_func(rate_func(i / layer_num)),
                    ).shift(point)
                )

        for mob in self:
            mob.set_z_index(1)
        self.add(Dot3D(point=self.point, radius=self.radius, color=color).set_z_index(5))
        self.add(label.scale(self.radius / 0.3).shift(point))



class ElectricField3D(StreamLines):
    def __init__(self, *charges: Charge, **kwargs) -> None:
        self.charges = charges
        super().__init__(lambda p: self._field_func(p), three_dimensions=True,x_range=[-4,4],y_range=[-4,4],colors=['#191919', WHITE],**kwargs)


    def _field_func(self, p: np.ndarray) -> np.ndarray:
        vec=np.zeros(3)
        for charge in self.charges:

            if np.sqrt((p[0]-charge.get_x())**2 + (p[1]-charge.get_y())**2+ (p[2]-charge.get_z())**2) < .3 and charge.magnitude<0:
                return np.zeros(3)
            
            if any(charge.get_center()-p)>0:
                vec+=charge.magnitude/(np.linalg.norm(charge.get_center()-p)**2) * (p - charge.get_center()) / np.linalg.norm(p - charge.get_center())
            else:
                return np.zeros(3)
        return vec
        


class cu(ThreeDScene):

    def construct(self):
        self.camera.background_color = '#191919'
        self.set_camera_orientation(phi= 75* DEGREES, theta=45 * DEGREES)
        #self.add(ThreeDAxes())
        
        carga1 = Charge(point=IN+RIGHT,magnitude=-1)
        carga2 = Charge(point=OUT+LEFT)
        #stream_lines=StreamLines(lambda pos: carga1.magnitude/(np.linalg.norm(carga1.get_center()-pos)**2) * (pos - carga1.get_center()) / np.linalg.norm(pos - carga2.get_center()) + carga2.magnitude/(np.linalg.norm(carga2.get_center()-pos)**2) * (pos - carga2.get_center()) / np.linalg.norm(pos - carga2.get_center()) if any(carga1.get_center()-pos)>0 and any(carga2.get_center()-pos)!=0 else np.zeros(3), color_scheme=lambda p: np.linalg.norm(p),colors=['#191919',WHITE],x_range=[-4,4],y_range=[-4,4], stroke_width=3)
        #self.add(stream_lines)
        stream_lines=ElectricField3D(carga1, carga2)
        self.add(carga1, carga2)
        self.add(stream_lines)
        carga1.become(Charge3D(point=RIGHT,magnitude=-1))
        carga2.become(Charge3D(point=OUT))
        stream_lines.start_animation(warm_up=False, flow_speed=1.5)
        self.begin_ambient_camera_rotation(rate=2*PI/(stream_lines.virtual_time / stream_lines.flow_speed / .4))
        self.wait(2*PI)


        # manim -pqh discord.py cu


class Throwbot(Scene):
    def construct(self):
        circle = Circle(radius=1)
        arc = Arc(radius=0.275, start_angle=0, angle=-PI).shift(4*LEFT)
        arm = Line([-2, 0, 0], [2, 0, 0])
        pivot = Dot(radius=0.1).shift(0.275*DOWN)
        arm.shift(2*LEFT + 0.275*DOWN)
        machine = Rectangle(height=2, width=3).shift((0.8+0.275)*DOWN + 1.3*LEFT)
        arm_group = VGroup(arc, arm, pivot)

        self.play(Create(circle.move_to(2*UP)))
        self.play(circle.animate.move_to(4*LEFT).scale(0.2), run_time = 1.5)
        self.play(Create(arm_group), Create(machine))
        self.play(VGroup(machine, arm_group, circle).animate.scale(0.5).to_corner(LEFT + DOWN))

        arm_group.add(circle)
        arm_group.pivot = pivot
        arm_group.arm = arm
        arm_group.wp = -60
        arm_group.w = 0

        def rotation_updater(mob, dt):
            if(mob.arm.get_angle() > -PI/4):
                mob.rotate(0.5 * mob.wp * dt*2 + mob.w*dt, about_point=mob.pivot.get_center())
                mob.w = mob.w + mob.wp * dt

        arm_group.add_updater(rotation_updater)

        def stop_func():
            return arm.get_angle() <= -PI/4

        self.wait(stop_condition=(stop_func))

        # manim -pqh discord.py Throwbot


from collections import deque

class dot_wave(Scene):
    def construct(self):
        dots = VGroup(*[Dot() for _ in range(20)]).arrange(RIGHT, buff=0.5)
        self.add(dots)

        def dot_updater(dot, previous, lag=2):
            dot.last_positions = deque([dot.get_y()]*lag, maxlen=lag)
            def updater(dot, dt):
                dot.last_positions.append(dot.get_y())
                if previous is None:
                    return                
                dot.move_to([dot.get_x(), previous.last_positions[0], 0])
            return updater
    
        previous = None
        for i, dot in enumerate(dots):
            dot.add_updater(dot_updater(dot, previous, lag=i//3+1))
            previous = dot
        
        def first_dot_updater(amplitude, frequency):
            t = 0
            def updater(m,dt):
                nonlocal t
                t += dt
                m.move_to([m.get_x(), amplitude*np.sin(t*frequency), 0])
            return updater

        dots[0].add_updater(first_dot_updater(1,5))
        self.wait(10)

        # manim -pqh discord.py dot_wave


class defFun(Scene):
    def construct(self):
        def f(x):
            if x < 0.5:
                return 0
            else:
                return 2*(x-0.5)

        ax = Axes(
            x_range = [0, 1, 1],
            y_range = [0, 1, 1],
            tips=False
        )

        plt = ax.plot(f, discontinuities = [0.5]).set_stroke(width=15, color=[PURE_GREEN,REANLEA_WARM_BLUE])


        self.play(
            Create(plt)
        )

        self.wait(3)


        # manim -pqh discord.py defFun


class bigsum(Scene):
    def construct(self):
        leftHandSide_string = r"\sum_{{i=1}}^{} \left(\frac{{d}}{{2}}\right)^i="  

        rhs_string = ""
        for i in range(1, 7+1):
            rhs_string += r"  \frac{{d}}{{{:.0f}}}".format(2**i)
            if i < 7:
                rhs_string += r"  +"
        
        eqn0 = MathTex(leftHandSide_string.format("n"), *rhs_string.split("  ")).to_edge(LEFT)

        lhss = [
            MathTex(leftHandSide_string.format("n")),
            *[MathTex(leftHandSide_string.format(i)) for i in range(1, 7+1)]
        ]
        # something is different between eqn0[0] and lhss[0]
        eqn0[0].become( lhss[0].move_to(eqn0[0].get_center()))

        self.play(Write(eqn0[0]))
        self.wait()

        for i in range(1,7+1):
            self.play(
                Transform(eqn0[0], lhss[i].move_to(eqn0[0].get_center()))
            )
            self.play(
                Write(eqn0[2*i-2:2*i] if i>1 else eqn0[i])
            )

            self.wait()


            # manim -pqh discord.py bigsum


# based on https://stackoverflow.com/questions/15191088/how-to-do-a-polynomial-fit-with-fixed-points  
      
def polyfit_with_fixed_points(n, x, y, xf, yf) :
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x**np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf**np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:n + 1]

class curve_fix_points(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0,1,0.2],
            y_range=[-5,5,1],
            tips=False,
        ).add_coordinates()
        self.add(ax)

        n, d, f = 50, 8, 3
        x = np.random.rand(n)
        xf = np.random.rand(f)
        poly = np.polynomial.Polynomial(np.random.rand(d + 1))
        y = poly(x) + np.random.rand(n) - 0.5
        yf = np.random.uniform(np.min(y), np.max(y), size=(f,))

        fixpoints = VGroup()
        for i in range(f):
            fixpoints += Dot(ax.c2p(xf[i],yf[i])).set_color(RED)
        self.add(fixpoints)

        params = polyfit_with_fixed_points(d, x , y, xf, yf)
        poly = np.polynomial.Polynomial(params)

        pl = ax.plot(poly, x_range=[0,1,.01])
        self.play(
            Create(pl)
        )

        # manim -pqh discord.py curve_fix_points


class polyfit(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0,10,2],
            y_range=[-5,5,1],
            tips=False,
        ).add_coordinates()
        self.add(ax)
        
        npoints = 6
        xs = np.linspace(0, 10, npoints, endpoint=True)
        ys = np.random.uniform(low=-2, high=2, size=npoints)

        y2 = ValueTracker(-2)

        fixpoints = VGroup()
        def fixpoints_updater(mobj):
            dummy = VGroup()
            ys[2] = y2.get_value()
            for i in range(npoints):
                dummy += Dot(ax.c2p(xs[i],ys[i])).set_color(BLUE if i==2 else RED)
            mobj.become(dummy)
        fixpoints.add_updater(fixpoints_updater, call_updater=True)    
        self.add(fixpoints)  

        pl = VMobject()
        def pl_updater(mobj):
            poly = np.polynomial.Polynomial.fit(x = xs, y = ys, deg = npoints)
            pl.become(ax.plot(poly, x_range=[0,10,.05]))
        pl.add_updater(pl_updater, call_updater=True)
        self.add(pl)

        self.play(y2.animate.set_value(2), run_time=3)

        # manim -pqh discord.py polyfit


class color_grad_rotate(Scene):
    def construct(self):
        sheen = ValueTracker(0)
        pi = MathTex(r"\pi").scale(10).set_color([BLUE, YELLOW, RED])
        def pi_updater(mobj):
            mobj.set_sheen_direction(RIGHT).rotate_sheen_direction(sheen.get_value())
        pi.add_updater(pi_updater)    
        self.play(Write(pi))
        self.wait()
        self.camera.background_color=REANLEA_BACKGROUND_COLOR_OXFORD_BLUE       
        self.play(sheen.animate.set_value(2*PI))
        colors = color_gradient([REANLEA_BACKGROUND_COLOR_OXFORD_BLUE,REANLEA_BACKGROUND_COLOR],16)
        self.time = 0
        def sceneUpdater(dt):
            self.camera.background_color = colors[int(self.time*15) if self.time<1 else 15]
            self.time += dt
        self.add_updater(sceneUpdater)    
        self.wait(2)

        # manim -pqh discord.py color_grad_rotate


class rotate_line(Scene):
    def construct(self):
        ax = NumberPlane(y_range=[-6, 6], background_line_style={"stroke_opacity": 0.4})
        ax.add_coordinates()
        k = 1
        b = 0
        function= MathTex("y = kx + b")
        k_var= Variable(k,"k",num_decimal_places=2)
        b_var =Variable(b,"b",num_decimal_places=1)
        Group(function,k_var,b_var).arrange(DOWN)
        Group(function,k_var,b_var).align_on_border(UL,buff=0.5)
        
        def func(x):
            return k_var.tracker.get_value()*x + b 
        #curve.add_updater(lambda y: )
        k_var.add_updater(lambda z: z.tracker.set_value(k_var.tracker.get_value()))
        curve = always_redraw(lambda : ax.plot(func, color=YELLOW))

        self.play(
            Write(ax),
            Write(function),
            Write(k_var),
            Write(curve)
                  )
        self.wait(4)
        self.play(
            k_var.tracker.animate.set_value(10),run_time=10,rate_func=there_and_back_with_pause
            )
        
        # manim -pqh discord.py rotate_line


class movingCursor(Scene):
    def construct(self):
        longEqn = MathTex(r"f(x)=",
            *[r"\frac{{1}}{{ {:.0f} }}\,x^{{ {:.0f} }} +".format(i,i) for i in range(1,10)]
        ).scale_to_fit_width(12)
        self.play(Write(longEqn))

        cursor = VGroup(
            *[Line(0.1*RIGHT, 0.3*RIGHT, stroke_width=1).rotate(i*PI/2, about_point=ORIGIN) for i in range(4)]
        ).move_to([-6,-3,0])

        def longEqnUpdater(mobj):
            for i in range(1,10):
                if np.linalg.norm(cursor.get_center()-longEqn[i].get_critical_point(DL)) < 0.1:
                    longEqn[i].set_stroke(width=2, color=YELLOW)
        longEqn.add_updater(longEqnUpdater)

        self.add(cursor)
        self.wait()

        path = VMobject().set_points_as_corners([cursor.get_center(), longEqn[1].get_critical_point(DL)])

        for i in range(1,9,2):
            path.add_quadratic_bezier_curve_to(
                (longEqn[i].get_critical_point(DL)+longEqn[i+2].get_critical_point(DL))/2+2*DOWN, longEqn[i+2].get_critical_point(DL)
            )

        self.add(path)
        self.play(FadeOut(path))

        self.play(MoveAlongPath(cursor, path), rate_func=rate_functions.linear, run_time=10)

        self.wait()

        # manim -pqh discord.py movingCursor


def move_obj_in_arc_towards(obj: Mobject, to_obj: Mobject):
    obj.generate_target()
    obj.target.move_to(to_obj.get_center())
    anim = MoveToTarget(obj, path_arc=PI/2)
    return anim

class target_object_class(Scene):
    def construct(self):
        sq = Square().to_corner(UR)
        c = Circle().to_edge(LEFT)
        self.play(Create(sq), Create(c))
        self.wait()
        self.play(move_obj_in_arc_towards(c, sq))
        self.wait()

    # manim -pqh discord.py target_object_class


class Gosper(VMobject):
    def __init__(self, order:int = 3, size = 1, stroke_color=YELLOW, stroke_opacity=1, stroke_width=4, **kwargs):
        super().__init__(stroke_color=stroke_color, stroke_opacity=stroke_opacity, stroke_width=stroke_width, **kwargs)
        self.turtleangle = 0
        self.turtlestep = size
        self.turtlepoints = [np.array([0,0,0])]
        def gosper_curve(order: int, size: int, is_A: bool = True) -> None:
            """Draw the Gosper curve."""
            if order == 0:
                self.turtlepoints.append(self.turtlepoints[-1] + self.turtlestep * [np.cos(self.turtleangle),np.sin(self.turtleangle),0])
                return
            for op in "A-B--B+A++AA+B-" if is_A else "+A-BB--B-A++A+B":
                if op == "A":
                    gosper_curve(order=order-1, size=size, is_A=True)
                elif op == "B":
                    gosper_curve(order=order-1, size=size, is_A=False)
                elif op == "-":
                    self.turtleangle -= 60*DEGREES
                else: 
                    self.turtleangle += 60*DEGREES
        gosper_curve(order=order, size=size)
        self.set_points_as_corners(self.turtlepoints)

class GosperTest(Scene):
    def construct(self):
        gosper = Gosper(order=3, size=1).scale(0.3).move_to(ORIGIN)
        self.play(Create(gosper,run_time=5))

        # manim -pqh discord.py Gosper


class Gauss_vector_field(Scene):
    def construct(self):
        def func(x):
            a = np.array([0,0])
            if x[0]**2+x[1]**2 <= 4:
                a[0] = x[0]*5
                a[1] = x[1]*5
            else:
                a[0] = np.sign(x[0])*10
                a[1] = np.sign(x[1])*10
            return a[0]*RIGHT + a[1]*UP
        field = ArrowVectorField(func,x_range=[-3,3,0.2])
        self.play(
            Write(field)
        )

        # manim -pqh discord.py Gauss_vector_field

class Gauss_vector_field_1(Scene):
    def construct(self):
        def func(x):
            return [
                x[0]/4 if x[0]**2 <= 1 else 0.25,
                x[1]/4 if x[1]**2 <= 1 else 0.25,
                0
            ]
        field = ArrowVectorField(func,x_range=[-3,3,0.2])
        self.play(
            Write(field)
        )

        # manim -pqh discord.py Gauss_vector_field_1


def disc_func(x):
    return np.cos(x) + 1

class func_discontinuity(Scene):
    def construct(self):
        disc = (2, -2)
        axes = Axes(x_range=[-5, 5, 1], y_range=[-3, 3, 1]).add_coordinates()
        plot = axes.plot(lambda x: disc_func(x))
        disc_point = VGroup(
            Dot(
                axes.c2p(disc[0], disc_func(disc[0])), fill_color=BLACK, stroke_width=1
            ),
            Dot(axes.c2p(disc[0], disc[1])),
        )

        self.add(axes)
        self.play(
            Create(plot)
        )
        self.play(
            Create(disc_point)
        )

        # manim -pqh discord.py func_discontinuity


class hyperbola_one_by_x(Scene):
    def construct(self):
        def hyperbola_with_zero(x):
            return 1 / x if x != 0 else 0
        
        ax = Axes(
            x_range=[0,10,2],
            y_range=[-5,5,1],
            tips=False,
        ).add_coordinates()
        self.add(ax)

        plot = ax.plot(lambda x: hyperbola_with_zero(x))

        self.play(
            Create(plot)
        )
        self.wait(2)
        
    
    # manim -pqh discord.py hyperbola_one_by_x

white = {100: "#ffffff", 300: "#fafafa"}

black = {800: "#1a1b1c", 900: "#000000"}

success = {
    100: "#dff6db",
    200: "#a3e596",
    300: "#5fd149",
    400: "#138a0f",
    500: "#138a0f",
    600: "#0c6d18",
    700: "#065220",
    800: "#034422",
    900: "#022c16",
    1000: "#011c0e",
}
primary = {
    100: "#e9f1ff",
    200: "#bdd5ff",
    300: "#81bdff",
    400: "#3fa2ff",
    500: "#0376e2",
    600: "#075bbb",
    700: "#094493",
    800: "#07397a",
    900: "#052550",
    1000: "#031833",
}
class positioning4(Scene):
    def construct(self):
        squareObjs = VGroup()

        for i in range(10):
            column = success[300] if i % 2 else success[200]
            square = RoundedRectangle(0.1, height=0.6, width=0.6).set_fill(column, 1)
            square.set_stroke(column, 1)
            num = DecimalNumber(
                number=i,
                num_decimal_places=0,
                color=primary[800], 
                font_size=36,
                stroke_width=2,
                edge_to_fix=DOWN,
            ).move_to(square)
            squareObjs.add(VGroup(square,num))

        squareObjs.arrange(DOWN, buff=0.05).shift(RIGHT * 4)

        self.add(squareObjs)
        self.wait(1)
        self.play(
            squareObjs[4][1].animate.set_value(14),
            run_time=1
        )
        self.wait()

        # manim -pqh discord.py positioning4

class pieces(Scene):
  def construct(self):
# Making the graph
        ax = Axes(
            x_range=[0, 24, 5],
            y_range=[-15, 15, 5],
            x_axis_config={"include_tip": False},
            y_axis_config={"include_tip": False},
        )
        ax.add_coordinates()

        graph = VGroup(
            ax.plot(
                lambda x: 6,
                x_range=[0,10,0.01],
                color=RED,
            ),
            ax.plot(
                lambda x: -3*(x-10)+6,
                x_range=[10,14,0.01],
                color=BLUE,
            ),
            ax.plot(
                lambda x: -4*(x-16)**2+10,
                x_range=[14,18,0.01],
                color=GREEN,
            ),
            ax.plot(
                lambda x: 3,
                x_range=[18,22,0.01],
                color=PURPLE,
            ),
        )

        # Animating:
        self.play(Create(ax))
        self.play(Write(graph))
        self.wait(2)

        # manim -pqh discord.py pieces


class pieces2(Scene):
  def construct(self):
# Making the graph
        ax = Axes(
            x_range=[0, 24, 5],
            y_range=[-15, 15, 5],
            x_axis_config={"include_tip": False},
            y_axis_config={"include_tip": False},
        )
        ax.add_coordinates()

        graph = ax.plot(
                lambda x: 6 if x < 10 else
                -3*(x-10)+6 if x < 14 else
                -4*(x-16)**2+10 if x < 18 else
                3,
                x_range=[0,25,0.01],
                color=PURPLE,
            )

        # Animating:
        self.play(Create(ax))
        self.play(Write(graph))
        self.wait(2)

        # manim -pqh discord.py pieces2

class pieces3(Scene):
  def construct(self):
# Making the graph
        ax = Axes(
            x_range=[0, 24, 5],
            y_range=[-15, 15, 5],
            x_axis_config={"include_tip": False},
            y_axis_config={"include_tip": False},
        )
        ax.add_coordinates()
        def func(x):
            return np.piecewise(
                x,
                [   
                    x < 10, 
                    x >= 10, 
                    x >= 14,
                    x >= 18
                ],
                [
                    6,
                    lambda x: -3*(x-10)+6,
                    lambda x: -4*(x-16)**2+10,
                    3    # default value
                ]
            )
        graph = ax.plot(
                func,
                x_range=[0,25,0.01],
                color=PURPLE,
            )

        # Animating:
        self.play(Create(ax))
        self.play(Write(graph))
        self.wait(2)

        # manim -pqh discord.py pieces3


class ChangingDots(Scene):
    def construct(self):

        moving_line = Line([-7, -5, 0], [-6, 5, 0])
        moving_line.nv = np.array([10, -1, 0])

        def color_updater(obj):
            if np.dot(moving_line.get_start(), moving_line.nv) > np.dot(obj.get_center(), moving_line.nv):
                obj.set_color(BLUE)
            else:
                obj.set_color(YELLOW)


        for i in range(30):
            p = Dot().move_to([random.uniform(-6, 6), random.uniform(-4, 4), 0])
            p.add_updater(color_updater)
            self.add(p)
        
    
        self.play(moving_line.animate.shift(14*RIGHT), run_time=5)
        self.play(moving_line.animate.shift(14*LEFT), run_time=5)

        # manim -pqh discord.py ChangingDots




def summation(func: Callable[[int], Any], i: int, n=None):
    """This is a sigma summation.

    Parameters
    ----------

    func
        The formula for the terms.
    i
        The starting index.
    n
        The last index value. 

        If you want that n approaches infinity it needs to be ``None`` (the default).
    
    """
    sigma = 0
    if n is None:
        # Infinity Sum
        while True:
            if func(i) == 0:
                break
            sigma += func(i)
            i += 1
    else:
        # Finity Sum
        for x in range(i, n + 1):
            sigma += func(x)
    return sigma

class Dicotomia(Scene):
    def construct(self):
        d = 64
        b = ValueTracker(0)

        sum = lambda n=None: summation(lambda x: (1/2)**x, 1, n)

        scale = lambda a: 1/(a+1)

        InvisibleLine = NumberLine(x_range = [0, d, 2], length = 10).set_opacity(0)

        tracker = ValueTracker(0)
        
        Dline = Line(
            InvisibleLine.get_left(), InvisibleLine.get_right()
        ).set_color(YELLOW) #It is the yellow line
        DlineParts = always_redraw(
            lambda: Line(
            Dline.get_left(), InvisibleLine.n2p(tracker.get_value())
        ).set_color(PURE_RED) #This the red line
        )
        Divide = Line(
            DOWN, 
            UP
        ).move_to(Dline.get_left()).set_color(PURE_BLUE).set_stroke(width = InvisibleLine.get_stroke_width()) #Ant this the divisor

        SystemDichoGroup = VGroup(InvisibleLine, Dline, Divide).move_to(DOWN)


        self.play(Create(Dline), FadeIn(DlineParts), run_time = 2)
        self.play(Write(Divide))
        self.play(Divide.copy().animate.move_to(Dline.get_right()), run_time = 2) #Another divisor
        Divide2 = always_redraw(lambda:
            Divide.copy().scale(scale(b.get_value())).move_to(
                InvisibleLine.n2p(tracker.get_value())
            )
        )

        def formulaWithResults(n: int=1):
            left_part = rf"\sum_{{i=1}}^{{{n}}} \left(\frac{{1}}{{2}}\right)^i = "
            result = rf"{{{sum(n)}}}"
            eq = left_part + result
            mobj = MathTex(eq)
            mobj[0][6].set_color(PURE_GREEN)
            return mobj
        

        eq1 = formulaWithResults().move_to(UP)

        Veces = 9 #Cuantas veces se repita la suma

        self.add(Divide2)
        for Idk in range(Veces):
            eq1_new = formulaWithResults(Idk+1).move_to(UP)
            self.wait(4/1.5**Idk)
            if Idk == 0:
                self.play(Write(eq1))
                br = Brace(Dline).shift(DOWN)
                Lenght = MathTex(f"{{1}}").shift(br.get_bottom() + DOWN/2).set_color(PURE_GREEN)
                self.play(FadeIn(br, Lenght))
            self.play(
                ReplacementTransform(eq1, eq1_new),
                FadeIn(
                    Divide.copy().scale(scale(b.get_value())).move_to(
                        InvisibleLine.n2p(
                            tracker.get_value()
                        )
                    ) #The copies of the blue line 🐒
                ),
                tracker.animate.set_value(sum(Idk+1)*d),
                b.animate.set_value(Idk+1)
            )
            eq1 = eq1_new

        final_txt1 =  r"\lim_{n \to \infty}\sum_{i=1}^{n}\left(\frac{1}{2}\right)^i = 1"
        final_eq1 = MathTex(final_txt1).move_to(UP)
        final_eq1[0][12].set_color(PURE_GREEN)
        final_eq1[0][-1].set_color(PURE_GREEN)
        self.play(
            ReplacementTransform(eq1, final_eq1),
            tracker.animate.set_value(sum()*d),
            run_time=3)
        self.wait(2)

        #Aquí se eliminan los objetos que estorban

        MobjectsToNotRemove = [Dline, DlineParts, Divide, Divide2]

        LenghtD = MathTex(f"{{d}}").shift(br.get_bottom() + DOWN/2).set_color(PURE_GREEN)

        self.play(
            *[FadeOut(obj) 
                for obj in self.mobjects + self.foreground_mobjects
                    if obj not in MobjectsToNotRemove],
            tracker.animate.set_value(0),
            b.animate.set_value(0),
            run_time = 4
        )

        #Aquí empieza la generalización
        self.play(Divide.copy().animate.move_to(Dline.get_right()), run_time = 2)

        def formula(n=1):
            left_part = rf"\sum_{{i=1}}^{{{n}}} \left(\frac{{d}}{{2}}\right)^i ="
            terms = [rf"\frac{{d}}{{{2**i}}}" for i in range(1, n+1)]
            eq = left_part + " + ".join(terms)
            mobj = MathTex(eq)
            mobj[0][6].set_color(PURE_GREEN)
            index = 12
            for i in range(1, n+1):
                mobj[0][index].set_color(PURE_GREEN)
                index += 4 + int(np.log10(2**i))
            return mobj
        
        
        eq = formula().move_to(UP)

        Veces = 6

        for Goofy in range(Veces):
            eq_new = formula(Goofy+1).move_to(UP)
            self.wait(4/1.5**Goofy)
            if Goofy == 0:
                self.play(Write(eq))
                self.play(FadeIn(br, LenghtD))
            self.play(
                ReplacementTransform(eq, eq_new),
                FadeIn(
                    Divide.copy().scale(scale(b.get_value())).move_to(
                        InvisibleLine.n2p(
                            tracker.get_value()
                        )
                    ) #The copies of the blue line 🐒
                ),
                tracker.animate.set_value(sum(Goofy+1)*d),
                b.animate.set_value(Goofy+1)
            )
            eq = eq_new
        final_txt =  r"\lim_{n \to \infty}\sum_{i=1}^{n}\left(\frac{d}{2}\right)^i = d"
        final_eq = MathTex(final_txt).move_to(UP)
        final_eq[0][12].set_color(PURE_GREEN)
        final_eq[0][-1].set_color(PURE_GREEN)
        self.play(
            ReplacementTransform(eq, final_eq),
            tracker.animate.set_value(sum()*d),
            run_time=3)
        self.wait(2)


        # manim -pqh discord.py Dicotomia



class AnimSine(Scene):
    def construct(self):
        #self.camera.background_color = "#06080d"
        line = Arrow(start=LEFT*2.5, end=RIGHT*2.5, color=WHITE, max_tip_length_to_length_ratio=0)
        ax =  Axes([-10,10,2], [-5,5,1]).add_coordinates()
        self.play(Create(ax))

        a = ValueTracker(0)
        b = ValueTracker(0)
        c = ValueTracker(0)
        d = ValueTracker(0)

        sin_func = always_redraw(lambda:
            ax.plot(lambda x: a.get_value() * np.sin( b.get_value()*x + c.get_value()) + d.get_value())
        )
        self.play(Create(sin_func))

        self.wait(1)
        self.play(d.animate.set_value(2)) # move up to y=2

        self.wait(1)
        self.play(
            a.animate.set_value(1),
            b.animate.set_value(1),
            c.animate.set_value(0),
            d.animate.set_value(0),
        ) # change to y=sin(x)

        self.wait(1)
        self.play(
            a.animate.set_value(1),
            b.animate.set_value(1),
            c.animate.set_value(1),
            d.animate.set_value(0),
        ) # change to y=sin(x+1)

        self.wait(2)

        # manim -pqh discord.py AnimSine


class sin1byxtob(Scene):
    def construct(self):
        b = ValueTracker(1)
        numPlane = NumberPlane(
            x_range=[-10, 10, 1], x_length=7, y_range=[-16, 16, 1], y_length=7
        ).to_edge(DOWN)
        parab = always_redraw(
            lambda: numPlane.plot(
                lambda x: (np.sin(1/x**b.get_value()) * 5),
                x_range=[-10, 10] if int(b.get_value()) == b.get_value() else [0.1, 10],
                discontinuities=[0],
                dt = 0.1,
                color=GREEN,
            )
        )
        self.play(DrawBorderThenFill(numPlane))
        self.play(Create(parab))
        #self.play(b.animate.set_value(50), run_time=10, rate_func=slow_into)
        self.wait()

        # manim -pqh discord.py sin1byxtob



def set_background(self) :
    background = Rectangle(
        width = 1920,
        height = 1080,
        stroke_width = 0,
        fill_color = "#1E3264",
        fill_opacity = 1
    )
    self.add(background)

class Grafik(MovingCameraScene):       
    def construct(self): 
        t5 = Tex (
            'Pertimbangkan sisi kanan dari ',
            r'$sin(\frac{1}{x})$'
            )
        axes2 = Axes(
            x_range = [0.001,3,0.01],
            y_range = [-2,2,1.0],
            x_axis_config = {
                "numbers_to_include": [0,1,2,3],
                "include_ticks" : False
            },
            y_axis_config = {"numbers_to_include": [-2,-1,0,1,2]},
            tips = False
        )
        
        #Defining graph function
        def func2(x):
            if x == 0:
                return 0
            else:
                return np.sin(1/x)
            

        #Get the graph
        graph2 = axes2.plot(func2)
        graph2.set_stroke(width = 1.5)
        #graph2.reverse_direction()
        
        #Set up its label
        axes_labels2 = axes2.get_axis_labels()
        
        #Show animate
        self.play(Create(axes2), run_time=2)
        #self.play(Create(axes_labels2))
        #self.wait(0.25)
        self.play(Create(graph2),run_time=5)
        self.wait(2)

        # manim -pqh discord.py Grafik


class Grafik_1(MovingCameraScene):       
    def construct(self): 

        ax = Axes(
            x_range = [0.0001,1.5,0.01],
            y_range = [-2,2,1.0],
            x_axis_config = {
                "numbers_to_include": [0,1,2,3],
                "include_ticks" : False
            },
            y_axis_config = {"numbers_to_include": [-2,-1,0,1,2]},
            tips = False
        ).set_color(REANLEA_GREY)

        dt=Dot(radius=.125/4,color=PURE_RED).move_to(ax.c2p(0,0)).set_z_index(2)
        
        #Defining graph function
        def func2(x):
            if x == 0:
                return 0
            else:
                return np.sin(1/x)
            

        #Get the graph
        graph2 = ax.plot(func2)

        graph2.set_stroke(width = 1.5, color=REANLEA_WARM_BLUE)
        
        #Show animate
        self.play(Create(ax), run_time=2)
        self.play(Write(dt))
        self.play(Create(graph2),run_time=5)
        self.wait(2)


        # manim -pqh discord.py Grafik_1

        # manim -sqk discord.py Grafik_1


class line_with_moving_dot(Scene):
    def construct(self):
        axes = Axes(x_range=[0,10,1], x_length=9, y_range=[0,20,5], y_length=6, axis_config={"include_numbers":False, "include_tip":False}).to_edge(DL).scale(1.2).shift(RIGHT * 1.5)
        func = lambda x: x**3
        graph = axes.plot(lambda x: 0.1 * ( x - 2 ) * (x - 5) * (x - 7) + 7)
        labels = axes.get_axis_labels(
            Tex("x").scale(0.7), Text("f(x)").scale(0.45)
        )

        x1 = ValueTracker(9.0)
        x2 = ValueTracker(2.5)


        p1 = always_redraw(lambda:Dot(axes.c2p(x1.get_value(), graph.underlying_function(x1.get_value())), color=RED))

        p2 = always_redraw(lambda:
            Dot(axes.c2p(x2.get_value(), graph.underlying_function(x2.get_value())), color=BLUE)
        )

        line = Line(start=axes.c2p(x1.get_value(),graph.underlying_function(x1.get_value())), end=axes.c2p(x2.get_value(),graph.underlying_function(x2.get_value())), color = GREEN_C)
        line.add_updater(lambda mob: mob.become(Line(start=axes.c2p(x1.get_value(),graph.underlying_function(x1.get_value())), end=axes.c2p(x2.get_value(),graph.underlying_function(x2.get_value())), color = GREEN_C)))

        self.add(axes)
        self.add(labels)
        self.add(graph)
        self.add(p2)
        self.add(p1)

        self.play(Write(line))

        dy = always_redraw(lambda: DashedLine(start=axes.c2p(x2.get_value(),graph.underlying_function(x2.get_value())), end=axes.c2p(x2.get_value(),graph.underlying_function(x1.get_value()))))
        dx = always_redraw(lambda: DashedLine(start=axes.c2p(x1.get_value(),graph.underlying_function(x1.get_value())), end=axes.c2p(x2.get_value(),graph.underlying_function(x1.get_value()))))

        self.play(AnimationGroup(x1.animate.set_value(2.5)), rate_func=linear, run_time=8)
        self.wait()

        # manim -pqh discord.py line_with_moving_dot

        # manim -sqk discord.py line_with_moving_dot


class changing_x_axis(Scene):
    def construct(self):
        Tmax = ValueTracker(6)

        def plot_updater(mob, dt):
            ax = Axes([-Tmax.get_value(),Tmax.get_value()],[-1.5,1.5])
            f = ax.plot(lambda t: np.cos(t))
            ax.add(f)
            mob.become(ax)

        ax1 = Axes()
        self.add(ax1)
        ax1.add_updater(plot_updater)
        self.play(Tmax.animate.set_value(12),run_time=4)
        self.wait()

        # manim -pqh discord.py changing_x_axis

        # manim -sqk discord.py changing_x_axis


class idx_lambda_act(Scene):
    def construct(self):
        
        colors_list = [PURPLE, YELLOW]
        directions_list = [RIGHT, LEFT]
        shift_directions = [UP, DOWN]
        axes = Axes()
        self.add(axes)
        k = ValueTracker(0)

        for idx in range(2):
            self.add(
                always_redraw(
                    lambda idx=idx: Dot(directions_list[idx], fill_color = colors_list[idx]).shift(k.get_value()*shift_directions[idx])
                )
            )
        self.play(k.animate.set_value(5), run_time = 5)

        # manim -pqh discord.py idx_lambda_act

        # manim -sqk discord.py idx_lambda_act


def search_shape_in_text(text:VMobject, shape:VMobject):
    T = TransformMatchingShapes
    results = []
    l = len(shape.submobjects[0])
    shape_aux = VMobject()
    shape_aux.points = np.concatenate([p.points for p in shape.submobjects[0]])
    for i in range(len(text.submobjects[0])):
        subtext = VMobject()
        subtext.points = np.concatenate([p.points for p in text.submobjects[0][i:i+l]])
        if T.get_mobject_key(subtext) == T.get_mobject_key(shape_aux):
            results.append(slice(i, i+l))
    return results

class latex_tex_shape_template(Scene):
    def construct(self):
        myTexTemplate = TexTemplate()
        myTexTemplate.add_to_preamble(r"\usepackage{mathptmx}")

        def color_equation(equation):
            for string, color in zip(strings, colors):
                tex = MathTex(string, tex_template=myTexTemplate)
                results = search_shape_in_text(equation, tex)
                for result in results:
                    equation[0][result].set_color(color)
        strings = ["x", "y", "z"]
        colors = [RED, GREEN, BLUE]

        eq_0 = MathTex(r"x^y + y = 3", tex_template=myTexTemplate)
        color_equation(eq_0)

        self.add(eq_0)
        self.wait(1)


        # manim -pqh discord.py latex_tex_shape_template

        # manim -sqk discord.py latex_tex_shape_template

class parametric_func_example_1(Scene):
    def construct(self):
        def func(t):
            return [t,np.exp(-t ** 2),0]
        
        f = ParametricFunction(func, t_range=np.array([-3, 3]), fill_opacity=0).set_color(BLUE)
        self.play(Write(f.scale(3)))

        # manim -pqh discord.py parametric_func_example_1

        # manim -sqk discord.py parametric_func_example_1


class SquigglyArrowEx(Scene):
    def construct(self):
        self.add(NumberPlane())
        lines = VGroup(*[
            SquigglyArrow(start=ORIGIN, end=RIGHT*(1+i/10), color=WHITE)
            for i in range(10)
        ]).arrange(DOWN, center=False, aligned_edge=LEFT).shift(2*UP)
        self.add(lines)

        lines = VGroup(*[
            SquigglyArrow(start=ORIGIN, end=DOWN*(1+i/10), color=RED, buff=0.2)
            for i in range(10)
        ]).arrange(LEFT, center=False, aligned_edge=UP).shift(LEFT)
        self.add(lines)

        lines = VGroup(*[
            SquigglyArrow(start=ORIGIN, end=2.5*(RIGHT*np.sin(2*PI/10*i) + UP*np.cos(2*PI/10*i)),
                          color=YELLOW, buff=0.3, period=.5)
            for i in range(10)
        ]).shift(4*RIGHT)
        self.add(lines)

        # manim -pqh discord.py SquigglyArrowEx

        # manim -sqk discord.py SquigglyArrowEx


class SquigglyArrowEx_1(Scene):
    def construct(self):
        self.add(NumberPlane())
        l = SquigglyArrow(start=ORIGIN, end=4*RIGHT, color=RED, num_wiggles=5)
        l2 = SquigglyArrow(start=ORIGIN, end=4*RIGHT, color=BLUE, num_wiggles=2).next_to(l, DOWN)
        self.add(l, l2)

        # manim -pqh discord.py SquigglyArrowEx_1

        # manim -sqk discord.py SquigglyArrowEx_1



class ProgressBar(VGroup):
    def __init__(self, color=WHITE, fill_color=WHITE, fill_opacity=0.5, **kwargs):
        super().__init__(**kwargs)
        self.add(Rectangle(color=color).set_fill(color, 0))
        self.add(Rectangle().set_fill(fill_color, fill_opacity).set_stroke(width=0))
        self.progress = 0
        self.update_progress()

    def set_progress(self, progress):
        self.progress = progress
        self.update_progress()
        return self

    def update_progress(self):
        p = self.progress or 0.001
        w = p*self.submobjects[0].width
        self.submobjects[1].stretch_to_fit_width(w)
        self.submobjects[1].move_to(self.submobjects[0].get_left(), aligned_edge=LEFT)
        return self
    
class ProgressBarScene(Scene):
    def tear_down(self):
        with open("aux_time.txt", "w") as f:
            f.write(str(self.time))
        if self.time != self.total_time:
            print("Total time has changed. You need to rerender the scene")
        super().tear_down()
    
    def setup(self):
        super().setup()
        try:
            with open("aux_time.txt", "r") as f:
                self.total_time = float(f.read())
        except:
            self.total_time = 1e3 # Not important

        progress_bar = (ProgressBar()
                        .stretch_to_fit_width(config.frame_width*0.9)
                        .stretch_to_fit_height(0.2)
        )
        progress_bar.to_edge(DOWN).set_z_index(50)
        self.add(progress_bar)
        self.time = 0

        def bar_updater(total_time):
            def update(bar, dt):
                self.time += dt
                frac = self.time/total_time
                bar.set_progress(frac)
            return update
        progress_bar.add_updater(bar_updater(self.total_time))


class ProgressBarSceneEx(ProgressBarScene):
    def construct(self):
        t = Text("Hello!")
        self.play(Write(t), run_time=5)
        self.wait(2)
        self.play(FadeOut(t), run_time=2)

    # manim -pqh discord.py ProgressBarSceneEx

    # manim -sqk discord.py ProgressBarSceneEx

class ProgressBar_1(VGroup):
    def __init__(self, color=WHITE, fill_color=WHITE, fill_opacity=0.5, **kwargs):
        super().__init__(**kwargs)
        self.add(Rectangle(color=color).set_fill(color, 0))
        self.add(Rectangle().set_fill(fill_color, fill_opacity).set_stroke(width=0))
        self.progress = 0
        self.update_progress()

    def set_progress(self, progress):
        self.progress = progress
        self.update_progress()
        return self

    def update_progress(self):
        p = self.progress or 0.001
        w = p*self.submobjects[0].width
        self.submobjects[1].stretch_to_fit_width(w)
        self.submobjects[1].move_to(self.submobjects[0].get_left(), aligned_edge=LEFT)
        return self

class ProgressBarSceneEx_1(Scene):
  def construct(self):
    progr = ProgressBar_1().stretch_to_fit_width(5).shift(UP)
    self.add(progr)
    self.play(progr.animate.set_progress(1), run_time=3)

    # manim -pqh discord.py ProgressBarSceneEx_1

    # manim -sqk discord.py ProgressBarSceneEx_1


class atomic_sunflower(Scene):
    def construct(self):
        atomic_radius = 0.2

        sunflower_seed_holder = Circle(fill_opacity=1, color=DARK_BROWN).scale(3.6)
        self.add(sunflower_seed_holder)
        atoms = VGroup()
        for y in np.arange(start = 0, stop = 4, step = 2*atomic_radius):
            for x in np.arange(start = sunflower_seed_holder.get_right()[0] - atomic_radius, stop = sunflower_seed_holder.get_left()[0], step = -2*atomic_radius):
                dot = Dot(radius=atomic_radius).set_color(BLUE).move_to([x,y,0])
                if np.linalg.norm(Intersection(dot, sunflower_seed_holder).get_center() - dot.get_center()) < 1e-3:
                    atoms += dot
                dot = Dot(radius=atomic_radius).set_color(BLUE).move_to([x,-y,0])
                if np.linalg.norm(Intersection(dot, sunflower_seed_holder).get_center() - dot.get_center()) < 1e-3:
                    atoms += dot
        self.play(Create(atoms))
        self.wait()


        # manim -pqh discord.py atomic_sunflower

        # manim -sqk discord.py atomic_sunflower


class fiboSpiral(Scene):
    def construct(self):
        sunflower_seed_holder = Circle(fill_opacity=1, color=DARK_BROWN).scale(3.6)
        self.add(sunflower_seed_holder)
        alpha = 0.618 
        v0 = 0.03
        angle = 2*PI*alpha
        radius = 0
        theta = 0
        seeds = VGroup()
        for i in range(100):
            seeds += Dot(radius=0.1).set_color(BLUE).move_to([radius*np.cos(theta),radius*np.sin(theta),0])
            theta += angle
            radius += v0
        self.play(Create(seeds))
        self.wait()

        # manim -pqh discord.py fiboSpiral

        # manim -sqk discord.py fiboSpiral

class track_mobject(Scene):
    def construct(self):
        center_dot = Dot()
        red_rect = VMobject()
        def track_center_dot(mob):
            x = 1 #((mob.get_x() / 3) + 1) / 2
            y = 1 #((mob.get_y() / 3) + 1) / 2
            mob.become(Rectangle(width=x, height=y, color=RED,
                                 fill_opacity=0.5))
            mob.move_to(center_dot.get_center() - np.array([-mob.width / 2, mob.height / 2, 0]))
            
        red_rect.add_updater(track_center_dot)
        self.add(red_rect, center_dot)
        self.wait()
        self.play(center_dot.animate.shift(DOWN))
        self.wait(2) 

        # manim -pqh discord.py track_mobject

        # manim -sqk discord.py track_mobject


class PhyllotaxisRadiusTest(Scene):
    def construct(self):
        sunflower_seed_holder = Circle(fill_opacity=1, color=DARK_BROWN).scale(3.6)
        a = 90 * (np.pi / 180)
        angle = ValueTracker(a)

        radius = Line().add_updater(lambda m: m.become(Line(start=sunflower_seed_holder.get_center(), end=sunflower_seed_holder.point_at_angle(angle.get_value()))), call_updater=True)

        self.add(sunflower_seed_holder)
        self.add(radius)
        for stepangle in range(90, 721, 30):
            self.add(Dot().move_to(radius.get_end()))
            self.play(angle.animate.set_value(stepangle*DEGREES), run_time = 1, rate_func=linear) 

        # manim -pqh discord.py PhyllotaxisRadiusTest

        # manim -sqk discord.py PhyllotaxisRadiusTest


class surf_circ(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=(-1.5, 1.5),
            y_range=(-1.5, 1.5),
            z_range=(0, 1, 1),
            x_length=10,
            y_length=10,
            z_length=3,
        )
        def f(u, v):
            k = 50
            z = 0.2*np.exp(-k*(u + v)**2)
            return z        
        surface = Surface(
            lambda u, v: axes.c2p(u*np.cos(v), u*np.sin(v), f(u*np.cos(v),u*np.sin(v))),
            u_range=(0, 1),
            v_range=(0, 2*PI),
            color=REANLEA_VIOLET,
            fill_color = [REANLEA_BLUE_LAVENDER],
            checkerboard_colors=None,
            fill_opacity = 0.5,
            stroke_opacity=0,
        ).set_color_by_gradient(REANLEA_BLUE_LAVENDER,REANLEA_AQUA_GREEN,REANLEA_VIOLET)  
        self.set_camera_orientation(phi=75 * DEGREES, theta=-50 * DEGREES)        
        self.add(surface) 

        # manim -pqh discord.py surf_circ

        # manim -sqk discord.py surf_circ


class cancelling(Scene):
    def construct(self):
        equation = MathTex(
            *r"\lim_{n\rightarrow\infty}  \left[  \left(  1  -  \frac{1}{2}  \right)  +  \left(  \frac{1}{2}  -  \frac{1}{3}  \right)  +  \left(  \frac{1}{3}  -  \frac{1}{4}  \right)  +  \dots  +  \left(  \frac{1}{n}  -  \frac{1}{n+1}  \right)  \right]  =  1  -  \frac{1}{n+1}  =  1".split("  ")
        ).scale_to_fit_width(12)
        self.add(equation)      
        # lets identify the individual parts of the equation:
        self.add(index_labels(equation).set_color(RED).shift(0.5*UP))  

        # from this we get the cancelling pairs 5/9, 11/15, 17/-, -/23
        cancelLines = VGroup()
        for set in [(5,9,DL,UR,PURPLE), (11,15,UL,DR,RED), (17,None,DL,UR,BLUE), (None,23,UL,DR,PINK)]:
            if set[0] != None:
                cancelLines += VGroup(
                    Line(equation[set[0]].get_critical_point(set[2]),equation[set[0]].get_critical_point(set[3]),color=set[4])
                )
            if set[1] != None:
                cancelLines += VGroup(
                    Line(equation[set[1]].get_critical_point(set[2]),equation[set[1]].get_critical_point(set[3]),color=set[4])
                )            
        self.play(Create(cancelLines), run_time=3)    
        self.wait()

        # manim -pqh discord.py cancelling

        # manim -sqk discord.py cancelling


def search_shape_in_text(text:VMobject, shape:VMobject):
    T = TransformMatchingShapes
    results = []
    l = len(shape.submobjects[0])
    shape_aux = VMobject()
    shape_aux.points = np.concatenate([p.points for p in shape.submobjects[0]])
    for i in range(len(text.submobjects[0])):
        subtext = VMobject()
        subtext.points = np.concatenate([p.points for p in text.submobjects[0][i:i+l]])
        if T.get_mobject_key(subtext) == T.get_mobject_key(shape_aux):
            results.append(slice(i, i+l))
    return results

def search_shapes_in_text(text:VMobject, shapes:list[VMobject]):
    results = []
    for shape in shapes:
        results += search_shape_in_text(text, shape)
    return results

def cross(mob, slant=1, **kwargs):
    line = Line(mob.get_corner(UP + RIGHT*slant), mob.get_corner(DOWN + LEFT*slant), **kwargs)
    return line

class cancelling_1(Scene):
    def construct(self):
        eq = r"""
        \lim_{n\rightarrow\infty}\left[
            \left(1-\frac{1}{2}\right) +
            \left(\frac{1}{2}-\frac{1}{3}\right) +
            \left(\frac{1}{3}-\frac{1}{4}\right) +
            \dots +
            \left(\frac{1}{n}-\frac{1}{n+1}\right)
            \right] = 1 - \frac{1}{n+1} = 1
            """
        equation = MathTex(eq).scale_to_fit_width(12)
        self.add(equation)
        frac = MathTex(r"\frac{1}{\phantom{2}}")
        fracn = MathTex(r"\frac{1}{\phantom{n}}")
        colors = [YELLOW, BLUE, GREEN, RED, PURPLE, ORANGE, PINK]
        fracs = search_shapes_in_text(equation, (frac, fracn))
        slant = 1
        for g1, g2, color in zip(fracs[:-2:2], fracs[1:-2:2], colors):
            f1 = equation[0][g1.start:g1.stop+1]
            f2 = equation[0][g2.start:g2.stop+1]
            self.play(Wiggle(f1, scale_value=1.5), 
                      Wiggle(f2, scale_value=1.5))
            self.play(Create(cross(f1, slant, color=color)),
                      Create(cross(f2, slant, color=color)))
            slant *= -1
        for g, color in zip(fracs[-2:], colors[-2:]):
            f = equation[0][g.start:g.stop+1]
            self.play(Wiggle(f, scale_value=1.5))
            self.play(Create(cross(f, slant, color=color)))
            slant *= -1

        # manim -pqh discord.py cancelling_1

        # manim -sqk discord.py cancelling_1



class Charge3D(VGroup):
    def __init__(
        self,
        magnitude: float = 1,
        point: np.ndarray = ORIGIN,
        add_glow: bool = True,
        **kwargs,
    ) -> None:
        
        VGroup.__init__(self, **kwargs)
        self.magnitude = magnitude
        self.point = point
        self.radius = (abs(magnitude) * 0.4 if abs(magnitude) < 2 else 0.8) * 0.3

        if magnitude > 0:
            label = VGroup(Prism(dimensions=[.2,.05,.05],fill_color=WHITE,fill_opacity=1).set_z_index(6),
                           Prism(dimensions=[.05,.05,.2],fill_color=WHITE,fill_opacity=1).set_z_index(6))
            color = RED
            layer_colors = [RED_D, RED_A]
            layer_radius = 2
        else:
            label = Prism(dimensions=[.2,.1,.1],fill_color=WHITE,fill_opacity=1).set_z_index(6)
            color = BLUE
            layer_colors = ["#3399FF", "#66B2FF"]
            layer_radius = 2

        if add_glow:  # use many arcs to simulate glowing
            layer_num = 50
            color_list = color_gradient(layer_colors, layer_num)
            opacity_func = lambda t: 200 * (1 - abs(t - 0.009) ** 0.0001)
            rate_func = lambda t: t ** 2

            for i in range(layer_num):
                self.add(
                    Sphere(
                        radius=layer_radius * rate_func((0.5 + i) / layer_num),
                        stroke_width=0,
                        checkerboard_colors=[color_list[0],color_list[0]],
                        color=color_list[i],
                        fill_opacity=opacity_func(rate_func(i / layer_num)),
                    ).shift(point)
                )

        for mob in self:
            mob.set_z_index(1)
        self.add(Dot3D(point=self.point, radius=self.radius, color=color).set_z_index(5))
        self.add(label.scale(self.radius / 0.3).shift(point))



class ElectricField3D(StreamLines):
    def __init__(self, *charges: Charge, **kwargs) -> None:
        self.charges = charges
        super().__init__(lambda p: self._field_func(p), three_dimensions=True,x_range=[-4,4],y_range=[-4,4],colors=['#191919', WHITE],**kwargs)


    def _field_func(self, p: np.ndarray) -> np.ndarray:
        vec=np.zeros(3)
        for charge in self.charges:

            if np.sqrt((p[0]-charge.get_x())**2 + (p[1]-charge.get_y())**2+ (p[2]-charge.get_z())**2) < .3 and charge.magnitude<0:
                return np.zeros(3)
            
            if any(charge.get_center()-p)>0:
                vec+=charge.magnitude/(np.linalg.norm(charge.get_center()-p)**2) * (p - charge.get_center()) / np.linalg.norm(p - charge.get_center())
            else:
                return np.zeros(3)
        return vec
        


class cu_vector_field(ThreeDScene):

    def construct(self):
        self.camera.background_color = '#191919'
        self.set_camera_orientation(phi= 75* DEGREES, theta=45 * DEGREES)
        #self.add(ThreeDAxes())
        
        carga1 = Charge(point=IN+RIGHT,magnitude=-1)
        carga2 = Charge(point=OUT+LEFT)
        #stream_lines=StreamLines(lambda pos: carga1.magnitude/(np.linalg.norm(carga1.get_center()-pos)**2) * (pos - carga1.get_center()) / np.linalg.norm(pos - carga2.get_center()) + carga2.magnitude/(np.linalg.norm(carga2.get_center()-pos)**2) * (pos - carga2.get_center()) / np.linalg.norm(pos - carga2.get_center()) if any(carga1.get_center()-pos)>0 and any(carga2.get_center()-pos)!=0 else np.zeros(3), color_scheme=lambda p: np.linalg.norm(p),colors=['#191919',WHITE],x_range=[-4,4],y_range=[-4,4], stroke_width=3)
        #self.add(stream_lines)
        stream_lines=ElectricField3D(carga1, carga2)
        self.add(carga1, carga2)
        self.add(stream_lines)
        carga1.become(Charge3D(point=RIGHT,magnitude=-1))
        carga2.become(Charge3D(point=OUT))
        stream_lines.start_animation(warm_up=False, flow_speed=1.5)
        self.begin_ambient_camera_rotation(rate=2*PI/(stream_lines.virtual_time / stream_lines.flow_speed / .4))
        self.wait(2*PI)


        # manim -pqh discord.py cu_vector_field

        # manim -sqk discord.py cu_vector_field


class randomFlag(Scene):
    def construct(self):
        flagnames = [
            r"img\earth.png",
            r"img\ganesh.png",            
        ]
        flags = Group(
            Square(side_length=3, color=BLACK, fill_opacity=1).set_z_index(0), # visual blocker
            *[ImageMobject(filename).scale_to_fit_width(2).set_z_index(-10) for filename in flagnames]
        )
        self.add(flags)
        previousFlag = -1
        for i in range(20):
            flagNum = np.random.random_integers(1, len(flagnames))
            if previousFlag > 0:
                flags[previousFlag].set_z_index(-10)
            flags[flagNum].set_z_index(1)
            previousFlag = flagNum
            self.wait(3/15)


        # manim -pqh discord.py randomFlag

        # manim -sqk discord.py randomFlag

class get_length_Test(Scene):
    def construct(self):
        line = Line([-3,2,0], [2,-2,0]) 

        print(line.get_length())
        print(np.linalg.norm(line.get_start()))               
        print(np.linalg.norm(line.get_start()))               
        print(np.linalg.norm(line.get_start()-line.get_end())) 

        a=MathTex(line.get_length())
        b=MathTex(np.linalg.norm(line.get_start()))
        c=MathTex(np.linalg.norm(line.get_start()))
        d=MathTex(np.linalg.norm(line.get_start()-line.get_end()))

        grp=VGroup(a,b,c,d).arrange_submobjects(DOWN)

        self.add(grp)

        # manim -pqh discord.py get_length_Test

        # manim -sqk discord.py get_length_Test


class Rot_AdS_normal_vec(AnimationGroup):
    def __init__(self, vgroup, action = None, **kwargs):
        self.vgroup = vgroup
        center = vgroup[0].get_center()
        if action == "RandalSundrum":
                super().__init__(
                    Succession(
                    vgroup[-1].animate.move_to(center).build(),
                    Rotate(vgroup[-1], angle= PI, about_point= center),
                    vgroup[-1].animate.rotate(PI).move_to(4*RIGHT).build()),
                **kwargs)

class AdS_Jc(VGroup):   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #Geometry
        brane = Line(start= [0,-2.5,0], end = [0,2.5,0], color = GREEN, stroke_width = 3)
        adskp= RoundedRectangle(corner_radius=0.2, height=4,  width=4, stroke_width=1, color= DARK_BLUE, fill_opacity=0.2)
        adskm= RoundedRectangle(corner_radius=0.2, height=4,  width=4, stroke_width=1, color= RED_D, fill_opacity=0.2)
        adskp.shift([2.1,0,0])
        adskm.shift([-2.1,0,0])
        
        #Text
        in_text = MathTex("-6 k_{-}^{2}").move_to(adskm.get_center())
        out_text = MathTex("-6 k_{+}^{2}").move_to(adskp.get_center())
        sym = MathTex("\mathbb{Z}_{2}").move_to(brane.get_corner(UL))
        
        #Arrow
        arrow = Arrow(max_tip_length_to_length_ratio=2, color= BLACK, start = LEFT, end= [0.5,0,0]).move_to(adskm.get_left())
        
        self.add(brane, adskm, in_text, adskp, out_text, sym, arrow)
        
class arr_rect_continuous_Test(Scene):
    def construct(self):
        self.camera.background_color = "#FFFFDC"
        Tex.set_default(color = BLACK, font_size = 25)
        MathTex.set_default(color = BLACK, font_size = 35)
        qua = AdS_Jc()
        self.add (qua[:-1])
        self.wait()
        self.play(Rot_AdS_normal_vec(qua, action = "RandalSundrum"))


        # manim -pqh discord.py arr_rect_continuous_Test

        # manim -sqk discord.py arr_rect_continuous_Test


class AdS_Jc_1(VGroup):   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #Geometry
        brane = Line(start= [0,-2.5,0], end = [0,2.5,0], color = GREEN, stroke_width = 3)
        adskp= RoundedRectangle(corner_radius=0.2, height=4,  width=4, stroke_width=1, color= DARK_BLUE, fill_opacity=0.2)
        adskm= RoundedRectangle(corner_radius=0.2, height=4,  width=4, stroke_width=1, color= RED_D, fill_opacity=0.2)
        adskp.shift([2.1,0,0])
        adskm.shift([-2.1,0,0])
        
        #Text
        in_text = MathTex("-6 k_{-}^{2}").move_to(adskm.get_center())
        out_text = MathTex("-6 k_{+}^{2}").move_to(adskp.get_center())
        sym = MathTex("\mathbb{Z}_{2}").move_to(brane.get_corner(UL))
        
        #Arrow
        arrow = Arrow(max_tip_length_to_length_ratio=2, color= BLACK, start = LEFT, end= [0.5,0,0]).move_to(adskm.get_left())
        
        self.add(brane, adskm, in_text, adskp, out_text, sym, arrow)
        
class arr_rect_continuous_Test_1(Scene):
    def construct(self):
        self.camera.background_color = "#FFFFDC"
        Tex.set_default(color = BLACK, font_size = 25)
        MathTex.set_default(color = BLACK, font_size = 35)
        qua = AdS_Jc_1()
        self.add (qua[:-1])
        self.wait()

        x_pos = ValueTracker(-4.1)
        def arrowUpdater(mobj):
            x = x_pos.get_value()
            angle = 0 if x < -0.2 else PI/0.4*(x+0.2) if x < 0.2 else PI
            mobj.rotate(angle = angle-mobj.get_angle(), about_point=mobj.get_center())
            mobj.shift((x-mobj.get_center()[0])*RIGHT)
        qua[-1].add_updater(arrowUpdater)

        self.add(qua[-1])
        self.play(x_pos.animate.set_value(4.1), rate_func=rate_functions.linear, run_time=4)
        self.wait()

        # manim -pqh discord.py arr_rect_continuous_Test_1

        # manim -sqk discord.py arr_rect_continuous_Test_1


class squaresAlongLine(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0,20,1],
            y_range=[0,20,1],
            x_length=7,
            y_length=7,
        )            
        self.add(ax)
        unitLengths = ax.c2p(1,1)-ax.c2p(0,0)
        
        x1 = ValueTracker(20)
        y1 = ValueTracker(5)

        line = always_redraw(lambda:
            Line(ax.c2p(0,0), ax.c2p(x1.get_value(),y1.get_value()))            
        )
        
        squares = VGroup()
        def squaresUpdater(mobj):
            group = VGroup()
            x0 = 0
            y0 = 0
            dx = abs(x1.get_value() - x0)
            sx = 1 if x0 < x1.get_value() else -1
            dy = -abs(y1.get_value() - y0)
            sy = 1 if y0 < y1.get_value() else -1
            error = dx + dy
            
            while True:

                group += Rectangle(width=unitLengths[0],height=unitLengths[1]).move_to(ax.c2p(x0,y0))
                if (x0 >= x1.get_value()) and (y0 >= y1.get_value()):
                    break
                e2 = 2 * error
                if e2 >= dy:
                    if x0 == x1.get_value():
                        break
                    error = error + dy
                    x0 = x0 + sx
                if e2 <= dx:
                    if y0 == y1.get_value():
                        break
                    error = error + dx
                    y0 = y0 + sy
            mobj.become(group)
        squares.add_updater(squaresUpdater, call_updater=True)
        
        self.add(squares, line)
        self.wait()
        
        self.play(y1.animate.set_value(20), run_time=5, rate_func=rate_functions.linear)
        self.play(x1.animate.set_value(5), run_time=5, rate_func=rate_functions.linear)
        self.wait()


        # manim -pqh discord.py squaresAlongLine

        # manim -sqk discord.py squaresAlongLine

class showBresenham(Scene):
    def construct(self):

        unitLengths = [0.2, 0.2]
        
        start = Dot(color=YELLOW).move_to([-5,-2,0])
        end   = Dot(color=BLUE).move_to([6,3,0])
        
        line = always_redraw(lambda:
            Line(start=start.get_center(), end=end.get_center())            
        )
        
        squares = VGroup()
        def squaresUpdater(mobj):
            group = VGroup()
            x0 = int(start.get_center()[0]/unitLengths[0])
            y0 = int(start.get_center()[1]/unitLengths[1])
            x1 = int(end.get_center()[0]/unitLengths[0])
            y1 = int(end.get_center()[1]/unitLengths[1])
            dx = abs(x1 - x0)
            sx = 1 if x0 < x1 else -1
            dy = -abs(y1 - y0)
            sy = 1 if y0 < y1 else -1
            error = dx + dy
            
            while True:
                group += Rectangle(width=unitLengths[0],height=unitLengths[1]).move_to([x0*unitLengths[0],y0*unitLengths[1],0])
                if (x0 == x1) and (y0 == y1):
                    break
                e2 = 2 * error
                if e2 >= dy:
                    if x0 == x1:
                        break
                    error = error + dy
                    x0 = x0 + sx
                if e2 <= dx:
                    if y0 == y1:
                        break
                    error = error + dx
                    y0 = y0 + sy
            mobj.become(group)
        squares.add_updater(squaresUpdater, call_updater=True)
        
        self.add(start, end, squares, line)
        self.wait()
        for i in range(6):
            self.play(end.animate.move_to([np.random.uniform(low=-7,high=7),np.random.uniform(low=-4,high=4),0]), run_time=2, rate_func=rate_functions.linear)
            self.play(start.animate.move_to([np.random.uniform(low=-7,high=7),np.random.uniform(low=-4,high=4),0]), run_time=2, rate_func=rate_functions.linear)
        self.wait()

        # manim -pqh discord.py showBresenham

        # manim -sqk discord.py showBresenham


class gaussLaw(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(x_range = [-10, 10],
                          y_range = [-10, 10],
                          z_range = [-10, 10],
                          x_length = 20,
                          y_length = 20,
                          z_length = 20,
                          z_axis_config= {"include_ticks":True},
                          axis_config = {"include_ticks":True})
        self.play(Create(axes))
        self.move_camera(phi = 60 * DEGREES, theta = 30 * DEGREES, zoom = 0.5, run_time = 2)
        self.wait(1)
        self.move_camera(zoom = 0.5, run_time = 2)


        #objects to be created
        charge1 = Dot3D(point = axes.coords_to_point(0, 0, 0), color = YELLOW)
        gaussianSurface = Sphere(charge1.get_center(), radius = 3, fill_opacity = 0.5, checkerboard_colors = None, stroke_width = 0)

        #func = lambda pos: (pos * (1 / ((pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2 + 1e-10) ** (3/2))))
        func1 = lambda pos: (100 * axes.c2p(*pos) * (1 / ((axes.c2p(*pos)[0] ** 2 + axes.c2p(*pos)[1] ** 2 + axes.c2p(*pos)[2] ** 2 + 1e-10) ** (3/2))))
        
        
        self.play(Create(ArrowVectorField(func1, x_range=[-2, 2, 1], y_range=[-2, 2, 1],z_range=[-2,2, 1], vector_config = {})))

        self.play(Create(charge1))
        self.move_camera(frame_center = charge1.get_center(), zoom = 0.8)
        self.play(Create(gaussianSurface))
        self.wait(1)
        self.begin_ambient_camera_rotation(rate = 0.2)
        self.wait(3)
        self.move_camera(zoom = 4)


        # manim -pqh discord.py gaussLaw

        # manim -sqk discord.py gaussLaw


class Spring(Line):
    def __init__(self, start=LEFT, end=RIGHT, num_loops=5, loop_radius=0.3, color=GREY, **kwargs):
        self.num_loops = num_loops
        self.loop_radius = loop_radius        
        super().__init__(start, end, color=color, **kwargs)

    def init_points(self):
        ## self.start, self.end
        length = np.linalg.norm(self.end-self.start)
        angle = angle_of_vector(self.end-self.start)
        micro_radius = self.loop_radius/length
        m = 2*np.pi*(self.num_loops+0.5)
        def loop(t):
            return t*RIGHT + micro_radius*(
                RIGHT + np.cos(m*t)*LEFT + np.sin(m*t)*UP
            )
        new_epsilon = 0.1/(m*micro_radius)/length
        p = ParametricFunction(loop, t_range=[0,1,new_epsilon])
        self.points = p.points.copy()
        self.move_to(self.start, aligned_edge=LEFT)
        self.scale(length/(1+2*micro_radius), about_point=self.start)
        self.rotate(angle, about_point=self.start)

    generate_points=init_points
    

class Spring_Test(Scene):
    def construct(self):
        bottom = ValueTracker(0)
        spring = always_redraw(lambda:
                    Spring(2*UP, [0, bottom.get_value(), 0], num_loops=4, loop_radius=0.5, color=RED)
        )
        self.add(spring)
        self.wait()
        self.play(bottom.animate.set_value(-4), run_time=2)
        self.wait()


        # manim -pqh discord.py Spring_Test

        # manim -sqk discord.py Spring_Test





class ColoredPlanes(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()



# Start: code for horizontal and vertical plane coloring

        # Create the xy plane
        xy_plane = Polygon(
            RIGHT*4 + DOWN*4, RIGHT*4 + UP*4, UP*4 , DOWN*4,
            color=GRAY, fill_opacity=0.5
        )

        # Create the yz plane
        yz_plane = Polygon(
            DOWN * 4 , UP * 4 , UP * 4 + OUT * 4, OUT * 4 + DOWN * 4,
            color=GRAY, fill_opacity=0.5
        )

        # # Create the xz plane
        # xz_plane = Polygon(
        #     ORIGIN, RIGHT * 4, RIGHT * 4 + OUT * 4, OUT * 4,
        #     color=GREEN, fill_opacity=0.5
        # )
        plane=VGroup(xy_plane,yz_plane,axes)
        self.set_camera_orientation(phi=60 * DEGREES, theta=-70 * DEGREES)
        self.add(plane)

# End : code for horizontal and plane colouring 


# start: Code for creating shape or plane
        k=int(input("enter no of sides of plane "))
        shape=RegularPolygon(n=k,color=RED,fill_color=YELLOW,fill_opacity=1,)
        shape.move_to([2,2,2])
        self.add(shape)
# End : Code for creating plane or shape

# Start : code for drawing projections line onto HP and VP       
        i=0
        shape_points=[]
        t_points=[]
        f_points=[]
        for vertex in shape.get_vertices():
            print(f"Vertex is {vertex}")
            x1,y1,z1 = vertex
            shape_point=[x1,y1,z1]
            shape_points.append(shape_point)

            t_point=[x1,y1,0]
            t_line=Line3D(start=shape_point,end=t_point,color=BLUE)
            t_dot=Dot(t_point)
            t_points.append(t_point)
            
          


            f_point=[0,y1,z1]
            f_line=Line3D(start=shape_point,end=f_point,color=BLUE)
            f_dot=Dot(f_point)
            f_points.append(f_point)
            
            

            self.add(t_dot,f_dot)
            self.add(t_line,f_line)
            i=i+1
# End : code for drawing projections line onto HP and VP                


# Start : code for creating Top View                        
        for l in range(len(t_points)):
            
            t_view=Line3D(start=t_points[l-1],end=t_points[l])
            self.add(t_view)
# End : code for creating Top view  

#Start : Code for creating front view
        for l in range(len(f_points)):
            
            f_view=Line3D(start=f_points[l-1],end=f_points[l])
            self.add(f_view)
#End : COde for creating front view

        # self.begin_ambient_camera_rotation(rate=0.2)
        # self.wait(10)
        # self.stop_ambient_camera_rotation()
        self.wait()


        # manim -pqh discord.py ColoredPlanes

        # manim -sqk discord.py ColoredPlanes



class plot_product_grph(Scene):
    def construct(self):
        axes = Axes(
            x_range=[0, 200,10],
            y_range=[0, 5,],
            axis_config={"include_numbers" : True,"color": BLUE}
        )
        axes.center()
        approximation = axes.plot(lambda x: 2*np.prod([(4*n**2)/(4*n**2-1) for n in range(1,int(x)+1)]),
                                  x_range= [0,200,10])
        pidroite = axes.plot(lambda x : np.pi,
                             x_range=[0,200])
        tex = MathTex(r"f:x \rightarrow \prod_{k =1}{x}\frac{4x^2}{4x^2-1}")
        self.add(axes, pidroite, tex)

        self.play(
            Write(approximation )
        )


        # manim -pqh discord.py plot_product_grph

        # manim -sqk discord.py plot_product_grph

class oneoverx(Scene):
    def construct(self):
        axes = Axes(x_range=[-0.5,7], y_range=[0,7])
        graph = axes.plot(lambda x: 1 / x, discontinuities=[0], dt=0.1)
        graph_1 = axes.plot(lambda x: 2 / x, discontinuities=[0], dt=0.5)
        self.add(axes)
        self.play(
            Create(graph)
        )
        self.play(
            Create(graph_1)
        )

        # manim -pqh discord.py oneoverx

        # manim -sqk discord.py oneoverx



def GetDragon(start=LEFT/2, end=RIGHT/2, times=13, DexOrLev=1, color='#F08080', stroke_width=2, stroke_width2=4):
    dol = (+1,-1)if DexOrLev else (-1,+1)
    a,b = [start, end],[]
    for i in range(times-1):
        for j in range(len(a)):
            if j!=0:
                midpoint = (a[j]+a[j-1])/2
                b.append(rotate_vector(a[j]-midpoint, PI/2*dol[j%2!=0])+midpoint)
            b.append(a[j])
        a, b = b, []
    r = Polygram(color=color, stroke_width=stroke_width)
    r.start_new_path(a[0])
    r.add_points_as_corners(a[1:])
    return r
config.background_color=WHITE

class get_dragon(MovingCameraScene):
    def construct(self):
        self.camera.frame.move_to([2,0,0])
        self.camera.frame.scale(.7)
        colors = [BLUE_E, BLUE_A, BLUE_D, GREEN_A, GREEN_E, PURE_GREEN]
        My_colors = color_gradient(colors,11)[1:-2]
        steps = []
        for times in range(1, 16):
            stroke = 20/times
            DragonCurveg = VGroup(
                GetDragon(start=DL,end=UL,DexOrLev=0, times=times, stroke_width=stroke).set_color(My_colors[0]),
                GetDragon(start=UR,end=UL,DexOrLev=0, times=times, stroke_width=stroke).set_color(My_colors[1]),
                GetDragon(start=UR,end=DR,DexOrLev=0, times=times, stroke_width=stroke).set_color(My_colors[2]),
                GetDragon(start=DL,end=DR,DexOrLev=0, times=times, stroke_width=stroke).set_color(My_colors[3]),
                ).set_stroke(opacity=0.8)
            tesselation = VGroup()
            for x in [-6, -2, 2]:
                for y in [-4, -2, 0, 2, 4]:
                    g = DragonCurveg.copy()
                    g.move_to(x*LEFT+y*UP)
                    if y%4 == 0:
                        for i in range(4,8):
                            g[i-4].set_color(My_colors[i])
                            g[i-4].set_color(My_colors[i])
                    tesselation.add(g)
            steps.append(tesselation.rotate(-PI/4))
        previous = steps[0]
        self.add(previous)
        for step in steps[1:]:
            self.play(ReplacementTransform(previous, step))
            self.wait(.2)
            previous = step
        self.wait()

        # manim -pqh discord.py get_dragon

        # manim -sqk discord.py get_dragon

class IFSTransform(object):
    def __init__(self):
        self.r = random.random()
        self.g = random.random()
        self.b = random.random()
    def transform(self, px:float, py:float) -> tuple[float, float]:
        # Default implementation, to be overriden in subclasses
        return (px, py)
    
class IFSFern(IFSTransform):
    def transform(self, x: float, y: float) -> tuple[float, float]:
        # Barnsley fern
        r = random.random()
        if r < 0.01:
            x, y =  0.00 * x + 0.00 * y,  0.00 * x + 0.16 * y + 0.00
        elif r < 0.86:
            x, y =  0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.60
        elif r < 0.93:
            x, y =  0.20 * x - 0.26 * y,  0.23 * x + 0.22 * y + 1.60
        else:
            x, y = -0.15 * x + 0.28 * y,  0.26 * x + 0.24 * y + 0.44
        return x, y


class ifs_fern(Scene):
    def setup(self):
        self.WIDTH = config.pixel_width
        self.HEIGHT = config.pixel_height
        self.img = Image.new("RGBA", (self.WIDTH, self.HEIGHT), "#00000000")
        self.img_mobj = ImageMobject(self.img, scale_to_resolution=self.HEIGHT)
        self.add(self.img_mobj)
        self.img_data = self.img.load()
        

    def ifs_points(self):
        ifs = IFSFern()
        px = random.uniform(-1, 1)
        py = random.uniform(-1, 1)
        r, g, b, a = .3, .9, .4, .99
        while True:
            px, py = ifs.transform(px, py)
            yield px, py, (r, g, b, a)

    def construct(self):
        iterator = iter(self.ifs_points())
        def updater(mobj, dt):
            for _ in range(1500):
                x, y, colour = next(iterator)
                self.add_pixel(x, y, colour)
            mobj.become(ImageMobject(self.img, scale_to_resolution=self.HEIGHT))
        self.wait(.1)
        self.img_mobj.add_updater(updater)
        self.wait(10)

    def add_pixel(self, x, y, colour):
        factor = min(self.WIDTH, self.HEIGHT) / 10
        x = int(x * factor + self.WIDTH / 2)
        y = int(-y * factor + self.HEIGHT)
        try:
            self.img_data[x, y] = tuple(int(255 * c) for c in colour)
        except IndexError:
            pass


        # manim -pql discord.py ifs_fern

        # manim -sqk discord.py ifs_fern


config.background_color= REANLEA_BACKGROUND_COLOR
def exFunc(x) :
    return (x**2-10*x+9)/(x**2-7*x+6)

class graphingEx(Scene):
    def construct(self):

        plane = NumberPlane(
            x_range = [-20, 20, 5], 
            y_range = [-20,20,5],
            x_length=14.1,
            y_length=8
        ).add_coordinates()
        graph = plane.plot(
            exFunc, 
            x_range = [-20, 20, 0.01], 
            discontinuities = [1,6],
            dt = 0.1,
            color=BLUE,
            use_smoothing=False,
        )
        self.play(FadeIn(plane))
        self.play(Create(graph, run_time=4))
        dot = Dot(radius=0.1, color=RED).move_to(plane.c2p(1,exFunc(1+1e-4))) 
        self.add(dot)
        self.wait()         


        # manim -pql discord.py graphingEx

        # manim -sqk discord.py graphingEx

class circle_inside_lines(Scene):
    def construct(self):
        rect = Rectangle(height=0.01).set_fill(BLUE, 1)
        lines = VGroup(*[rect.copy() for _ in range(50)])
        lines.arrange(DOWN, buff=0.1)
        circle = Circle()
        for line in lines:
            line.become(Intersection(line, circle))
        #self.add(lines.rotate(PI / 6), circle)

        self.play(
            Create(circle)
        )
        self.play(
            Create(lines)
        )
    

    # manim -pql discord.py circle_inside_lines

    # manim -sqk discord.py circle_inside_lines


class BeveledSquare(Square):
    def __init__(self, bevel=0.5, **kwargs):
        if bevel < 0 :
            bevel = 0
        if bevel > 1:
            bevel = 1
        scale = interpolate(2**.5, .5*2**.5, bevel)
        super().__init__(**kwargs)
        sq2 = self.copy().scale(scale).rotate(PI/4)
        result = Intersection(self, sq2)
        self.points = result.points[:]



class BeveledSquare_scene(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        #The side length of the square is 1.4, and the bevel 0.2*2
        VGIQ = VGroup(
            *[(IQBlock := BeveledSquare(bevel = 2/7, side_length = 1.4, color = "#57585a", fill_opacity = 1))
              .add(Square(side_length = 1, color = BLACK, fill_opacity = 1)
                   .move_to(IQBlock.get_center())) 
              for i in range(32)]).arrange_in_grid(rows = 4, cols = 8, buff = 0)
        
        PluggedHoles = VGroup(*[Polygram([[-2/7, 0, 0], [0, 2/7, 0], [2/7, 0, 0], [0, -2/7, 0]], color = "#57585a", fill_opacity = 1)
                              .move_to(Block.get_corner(DR)) 
                              for Block in VGroup(*[i for i in VGIQ if (list(VGIQ).index(i) + 1)%8 != 0])[:-7]])  #This is to iterate with a list that does not contain the squares of the southeast edges.

        #self.add(VGIQ, PluggedHoles) 

        self.play(
            Write(VGIQ)
        )
        self.play(
            Write(PluggedHoles)
        )


        # manim -pqh discord.py BeveledSquare_scene

        # manim -sqk discord.py BeveledSquare_scene

class MyCamera(ThreeDCamera):
    def transform_points_pre_display(self, mobject, points):
        if getattr(mobject, "fixed", False):
            return points
        else:
            return super().transform_points_pre_display(mobject, points)
      
class MyThreeDScene(ThreeDScene):
    def __init__(self, camera_class=MyCamera, ambient_camera_rotation=None,
                 default_angled_camera_orientation_kwargs=None, **kwargs):
        super().__init__(camera_class=camera_class, **kwargs)

def make_fixed(*mobs):
    for mob in mobs:
        mob.fixed = True
        for submob in mob.family_members_with_points():
            submob.fixed = True

class threed_txt_fxd(MyThreeDScene):
    def construct(self):
        r = Rectangle()
        tex = MathTex("{{f(x,y)}} = {{ e^{-(x^2 + y^2)} }}").to_edge(UP * 1.1)
        tex[2].set_color(YELLOW)
        texN = MathTex( "{{f(x,y)}} = " 
            "{{\\int_{-2}^2 \\int_{-2}^2 f(x,y) dx dy \\approx \\sum_{i=0}^{n-1} \\sum_{j=0}^{m-1} f(i^*,j^*) \\Delta a}}")
        make_fixed(tex, texN)
        texN.to_edge(UP * 1.1)
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.5)
        self.add(r)
        self.add(tex)
        self.wait()
        self.play(TransformMatchingTex(tex, texN))
        self.wait()

        # manim -pqh discord.py threed_txt_fxd

        # manim -sqk discord.py threed_txt_fxd


class Brace_cube(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(x_range=(0, 1, 1), 
                          y_range=(0, 1, 1),
                          z_range=(0, 1, 1),
                          x_length=1, y_length=1,z_length=1).scale(0.5)
        labels = axes.get_axis_labels().scale(0.5)
        Tex.set_default(font_size=30)
        MathTex.set_default(font_size=30)

        self.camera.background_color = DARKER_GRAY

        Titolo = Title("Cubo di un binomio",     font_size=35).set_color(RED).shift(DOWN*0.5)
        self.camera.add_fixed_in_frame_mobjects(Titolo)

        cube = Cube(side_length=2, fill_opacity=0.1, stroke_width=0.5)
        self.set_camera_orientation(phi=60 * DEGREES, theta=30* DEGREES)
        
        p1 = Dot3D((1, -1, -1))

        l1 = Line3D(start=(1, 1, -1), end=(1, -1, -1), color=RED)
        b1=Brace(l1,direction=(1,0,0))
        b1t= MathTex(r"a+b").rotate(PI/2).next_to(b1,RIGHT*0.5)
        

        l2 = Line3D(start=(1, 1, -1), end=(-1, 1, -1), color=BLUE)
        b2=Brace(l2,direction=(0,1,0))
        b2t= MathTex(r"a+b").rotate(PI).next_to(b2,UP*0.5)

        l3 = Line3D(start=(1, -1, -1), end=(1, -1, 1), color=GREEN)
        b3 = b1.copy().rotate(PI/2, axis=(1,0,0), about_point=p1.get_center())
        b3t= MathTex(r"a+b").next_to(l3, 1.5*RIGHT).rotate(-PI/2).rotate(-PI/2, axis=(1,0,0))

        self.add(cube, Titolo, b1,p1,l1,axes,b1t, labels, l2,b2,b2t, l3,b3,b3t)


        # manim -pqh discord.py Brace_cube

        # manim -sqk discord.py Brace_cube

class TriangleInsideColor(Scene):
    def construct(self):
        self.camera.background_color = "#140027"
        sq3 = math.sqrt(3)
        #   "#c7751e"
        tri_x_offset = 0
        tri_y_offset = 0
        tri_len = 1.2
        tri_p1 = [tri_x_offset, tri_len*math.sqrt(3) + tri_y_offset, 0]
        tri_p2 = [2*tri_len + tri_x_offset, -tri_len*math.sqrt(3) + tri_y_offset, 0]
        tri_p3 = [-2*tri_len + tri_x_offset, -tri_len*math.sqrt(3) + tri_y_offset, 0]

        tri = always_redraw(
            lambda : Polygon(
                tri_p1, tri_p2, tri_p3,
                color=WHITE
            )
        )

        point = Dot([-0.5, 0.6, 0])
        px = ValueTracker(point.get_arc_center()[0])
        py = ValueTracker(point.get_arc_center()[1])
        px.add_updater(lambda x: x.set_value(point.get_arc_center()[0]))
        py.add_updater(lambda x: x.set_value(point.get_arc_center()[1]))

        int_point_1 = Dot([0.25*(3*tri_x_offset + 3*tri_len + sq3*tri_y_offset + px.get_value() - sq3*py.get_value()), (1/(4*sq3))*(3*tri_x_offset + 3*tri_len + sq3*tri_y_offset + 3*sq3*py.get_value() - 3*px.get_value()), 0])
        int_point_2 = Dot([-0.25*(3 * tri_len - 3*tri_x_offset + sq3*tri_y_offset - px.get_value() - sq3 * py.get_value()), (1/(4*sq3))*(3*tri_len - 3*tri_x_offset + sq3*tri_y_offset + 3*sq3*py.get_value() + 3*px.get_value()), 0])
        int_point_3 = Dot([px.get_value(), -tri_len*sq3, 0])

        int_point_1.add_updater(lambda x: x.move_to([0.25*(3*tri_x_offset + 3*tri_len + sq3*tri_y_offset + point.get_arc_center()[0] - sq3*point.get_arc_center()[1]), (1/(4*sq3))*(3*tri_x_offset + 3*tri_len + sq3*tri_y_offset + 3*sq3*point.get_arc_center()[1] - 3*point.get_arc_center()[0]), 0]))
        int_point_2.add_updater(lambda x: x.move_to([-0.25*(3 * tri_len - 3*tri_x_offset + sq3*tri_y_offset - point.get_arc_center()[0] - sq3 * point.get_arc_center()[1]), (1/(4*sq3))*(3*tri_len - 3*tri_x_offset + sq3*tri_y_offset + 3*sq3*point.get_arc_center()[1] + 3*point.get_arc_center()[0]), 0]))
        int_point_3.add_updater(lambda x: x.move_to([point.get_arc_center()[0], -tri_len*sq3, 0]))

        tri1 = always_redraw(
            lambda : Polygon(
            tri_p1, int_point_1.get_arc_center(), point.get_arc_center(),
            color=WHITE, fill_color="#41145e", fill_opacity=1, stroke_width=0, stroke_color=WHITE
            )
        )
        tri2 = always_redraw(
            lambda: Polygon(
                tri_p2, int_point_1.get_arc_center(), point.get_arc_center(),
                color=WHITE, fill_color="#7a40a1", fill_opacity=1, stroke_width=0, stroke_color=WHITE
            )
        )
        tri3 = always_redraw(
            lambda: Polygon(
                tri_p2, int_point_3.get_arc_center(), point.get_arc_center(),
                color=WHITE, fill_color="#41145e", fill_opacity=1, stroke_width=0, stroke_color=WHITE
            )
        )
        tri4 = always_redraw(
            lambda: Polygon(
                tri_p3, int_point_3.get_arc_center(), point.get_arc_center(),
                color=WHITE, fill_color="#7a40a1", fill_opacity=1, stroke_width=0, stroke_color=WHITE
            )
        )
        tri5 = always_redraw(
            lambda: Polygon(
                tri_p3, int_point_2.get_arc_center(), point.get_arc_center(),
                color=WHITE, fill_color="#41145e", fill_opacity=1, stroke_width=0, stroke_color=WHITE
            )
        )
        tri6 = always_redraw(
            lambda: Polygon(
                tri_p1, int_point_2.get_arc_center(), point.get_arc_center(),
                color=WHITE, fill_color="#7a40a1", fill_opacity=1, stroke_width=0, stroke_color=WHITE
            )
        )

        tri_group = VGroup(tri1, tri2, tri3, tri4, tri5, tri6)

        self.play(Create(tri))
        self.play(
            DrawBorderThenFill(tri1, run_time=0.6),
            DrawBorderThenFill(tri2, run_time=0.9),
            DrawBorderThenFill(tri3, run_time=1.2),
            DrawBorderThenFill(tri4, run_time=1.5),
            DrawBorderThenFill(tri5, run_time=1.8),
            DrawBorderThenFill(tri6, run_time=2.1),
            Write(point)
        )
        self.play(
            Write(Dot(tri_p1).scale(0.7), run_time=0.7),
            Write(Dot(tri_p2).scale(0.7), run_time=0.7),
            Write(Dot(tri_p3).scale(0.7), run_time=0.7),
            Write(int_point_1.scale(0.6), run_time=0.7),
            Write(int_point_2.scale(0.6), run_time=0.7),
            Write(int_point_3.scale(0.6), run_time=0.7),
        )
        #  self.play(
            #  Write(Tex("A").next_to(tri_p1, UP, buff=0.1).scale(0.7), run_time=0.9),
            #  Write(Tex("B").next_to(tri_p2, RIGHT, buff=0.1).scale(0.7), run_time=0.9),
            #  Write(Tex("C").next_to(tri_p3, LEFT, buff=0.1).scale(0.7), run_time=0.9),
        #  )
        TexP = Tex("P").next_to(point.get_arc_center(), UR, buff=0.1).scale(0.7)
        TexD = Tex("D").next_to(int_point_1.get_arc_center(), UR, buff=0.07).scale(0.7)
        TexE = Tex("E").next_to(int_point_3.get_arc_center(), DOWN, buff=0.1).scale(0.7)
        TexF = Tex("F").next_to(int_point_2.get_arc_center(), UL, buff=0.07).scale(0.7)
        SecondaryTex = VGroup(TexP, TexD, TexE, TexF)
        #  self.play(Write(SecondaryTex, run_time=0.7))

        p_group = VGroup(point, TexP)
        self.play(p_group.animate.shift(DR))
        self.wait(2) 


        # manim -pqh discord.py TriangleInsideColor

        # manim -sqk discord.py TriangleInsideColor 


class SierpinskiCarpet(Scene):
    def construct(self):
        # Set the total number of times the process will be repeated
        total = 7

        # Calculate the size of the image
        size = 3**total

        # Create an empty image
        square = np.empty([size, size, 3], dtype=np.uint8)
        color = np.array([255, 255, 255], dtype=np.uint8)

        # Fill it with black
        square.fill(0)

        for i in range(0, total + 1):
            stepdown = 3**(total - i)
            for x in range(0, 3**i):
                if x % 3 == 1:
                    for y in range(0, 3**i):
                        if y % 3 == 1:
                            square[y * stepdown:(y + 1) * stepdown, x * stepdown:(x + 1) * stepdown] = color

            # Convert the NumPy array to an image and display it
            img = Image.fromarray(square)
            self.add(ImageMobject(img))
            self.wait(0.5)  # Adjust the animation speed as needed

###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2023\lab" 