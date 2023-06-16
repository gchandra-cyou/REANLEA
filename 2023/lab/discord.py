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
        
###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2023\lab" 