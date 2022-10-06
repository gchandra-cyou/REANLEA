from __future__ import annotations
from asyncore import poll2
from audioop import add
from cProfile import label
from distutils import text_file

import math
from math import pi

import os,sys
from manim import *
from numpy import array
import numpy as np
import random as rd
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
from round_corner import*
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import root_scalar
from round_corner import angle_between_vectors_signed


config.background_color= REANLEA_BACKGROUND_COLOR


###################################################################################################################






axis_size = 4
arrow_size = axis_size /2
axis_range = [-arrow_size,arrow_size,arrow_size*2]


def apply_transforms(arc, arc_transforms):
    for transform in arc_transforms:
        getattr(arc, transform['method'])(*transform['args'], **transform['kwargs'])

class UpdateValueRange:
    def __init__(self, name, start, end, transforms=()):
        self.name = name
        self.start = start
        self.end = end
        self.transforms = transforms

    def __call__(self, mobject, alpha):
        value = interpolate(self.start, self.end, alpha)
        mobject.become(
            AnnularSector(radius=1.0, inner_radius=0, start_angle=-pi, angle=value, fill_opacity=0.2)
        )
        apply_transforms(mobject, self.transforms)

class Scenex(ThreeDScene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)#, distance=8)

        axes = ThreeDAxes(tips=False, x_length=axis_size, y_length=axis_size, z_length=axis_size,
                          x_range=axis_range, y_range=axis_range, z_range=axis_range)


        self.add((axes))

        arrow = Vector(direction=np.array([0,0,arrow_size]))

        arc = AnnularSector(radius=1.0, inner_radius=0, start_angle=-pi, angle=0, fill_opacity=0.2)


        arc_transforms = [{'method': 'rotate', 'args': [pi/2,Y_AXIS], 'kwargs': {'about_point': ORIGIN}}]
        apply_transforms(arc, arc_transforms)


        self.add(axes, arrow, arc)

        updater = UpdateValueRange('angle', 0, pi/2, transforms=arc_transforms)

        self.play(
            UpdateFromAlphaFunc(arc, updater),
            Rotate(arrow, 90 * DEGREES,about_point=ORIGIN, axis=array([1.0, 0.0, 0.0]))
        )


if __name__ == "__main__":
    import os

    module_name = os.path.basename(__file__)
    command_A = "manim -ql -p  "
    command_B = module_name + " " + "Scene"
    os.system(command_A + command_B)


    # manim -pqh discord_3.py Scenex







num = 10

class Mouche(Dot):
    def __init__(self, position, velocity, color):
        Dot.__init__(self, position, color=color)
        self.position = position
        self.velocity = velocity
        self.path = TracedPath(self.get_center) #checkout the usage of get_center without () 


class ExoMouches(MovingCameraScene):
    
    def construct(self):
        
        self.create_mouches(num)
        self.polygon()
        self.velocity_vectors()
        self.play(self.camera.frame.animate(run_time=5).move_to(ORIGIN).scale(0.7))
        self.time=0
        def updcamera(mob,dt):
            self.time+=dt
            if self.time%2==0:
                mob.move_to(ORIGIN).scale(0.85)
            
        self.camera.frame.add_updater(updcamera)

        def update_mouche(mob, dt):
                index = self.mouche_list.index(mob)
                mob.shift(dt*mob.velocity)
                mob.position += mob.velocity*dt
                if index == len(self.mouche_list) - 1:
                    mob.velocity = (self.mouche_list[0].position - self.mouche_list[index].position)/np.linalg.norm(
                        self.mouche_list[0].position - self.mouche_list[index].position)
                else:
                    mob.velocity = (self.mouche_list[index+1].position - self.mouche_list[index].position)/np.linalg.norm(
                        self.mouche_list[index+1].position - self.mouche_list[index].position)

        def update_velocity_vector(velocity, dt):
                index = self.velocity_vectors_list.index(velocity)
                vector = Vector(self.mouche_list[index].velocity, color=self.mouche_list[index].color).move_to(
                    self.mouche_list[index].position).set_z_index(10)
                vector.shift(self.mouche_list[index].position - vector.get_start())
                velocity.become(vector)

        def update_polygon(mob, dt):
            mob.become(Polygon(*[mouche.position for mouche in self.mouche_list]).set_z_index(1))

        for i in range(len(self.mouche_list)):
            self.mouche_list[i].add_updater(update_mouche)
            self.velocity_vectors_list[i].add_updater(update_velocity_vector)

        
        self.polygon.add_updater(update_polygon)
        
        self.wait(20)


    def create_mouches(self, num):
        self.mouche_list = []
        for i in range(num):
            x = 3.5 * np.cos((2*i*np.pi)/num)
            y = 3.5 * np.sin((2*i*np.pi)/num)
            z = 0
            position = np.array([x, y, z])
            velocity = np.array([0, 0, 0])
            mouche = Mouche(position, velocity, RED).set_z_index(10)
            self.mouche_list.append(mouche)
        for i in range(len(self.mouche_list)):
            if i == len(self.mouche_list) - 1:
                self.mouche_list[i].velocity = (self.mouche_list[0].position - self.mouche_list[i].position)/np.linalg.norm(
                    self.mouche_list[0].position - self.mouche_list[i].position)
            else:
                self.mouche_list[i].velocity = (self.mouche_list[i+1].position - self.mouche_list[i].position)/np.linalg.norm(
                    self.mouche_list[i+1].position - self.mouche_list[i].position)
        
        self.play(*[Create(mouche) for mouche in self.mouche_list])
        self.add(*[mouche.path for mouche in self.mouche_list])
        self.wait(0.5)

    def polygon(self):
        self.polygon = Polygon(*[mouche.position for mouche in self.mouche_list]).set_z_index(1)
        self.play(Create(self.polygon))
        self.wait(0.5)

    def velocity_vectors(self):
        self.velocity_vectors_list = []
        for i in range(len(self.mouche_list)):
            vector = Vector(self.mouche_list[i].velocity, color=self.mouche_list[i].color).move_to(self.mouche_list[i].position).set_z_index(10)
            vector.shift(self.mouche_list[i].position - vector.get_start())
            self.velocity_vectors_list.append(vector)
        self.play(*[GrowArrow(vector) for vector in self.velocity_vectors_list])
        self.wait(0.5)



   



        # manim -pqh discord_3.py ExoMouches



class Shrink_sin_func(Scene):
    def construct(self):

        ax = Axes(
            x_range=[-5*PI,2*PI],#,-0.1, 5 * PI, 2*PI),
            y_range=[-3, 3]
        )

        tracker = ValueTracker(0.5)

        graph = always_redraw(lambda: ax.plot(
            lambda t,tracker=tracker: np.sin(tracker.get_value() * t),
            color=BLUE
        ))
        # Below mentioned two lines also work, and give the same output.
        # graph=ax.plot(lambda t: np.sin(t*tracker.get_value()),x_range=[-5*PI,2*PI],color=BLUE)
        # graph.add_updater(lambda m: m.become(ax.plot(lambda t: np.sin(t*tracker.get_value()),x_range=[-5*PI,2*PI],color=BLUE)))

        #self.add(ax)
        self.add(graph)

        self.play(tracker.animate(run_time=6).set_value(5))
        self.wait(3)



        # manim -pqh discord_3.py Shrink_sin_func



from colour import Color

class sin_func_shrink(Scene):
    def construct(self):
        ax = Axes(
            x_range=(-0.1, 10),
            y_range=(-2, 2)
        )

        sin_funcs = [ax.plot(
            lambda t: np.sin(t * alpha),
            color=RED
        ) for alpha in np.arange(0.5, 1.5+1, 0.1)]

        # self.add(ax)
        polys = [RegularPolygon(n=5, color=Color(
            hue=j/10, saturation=1, luminance=0.5))for j in range(12)]

        

        for i in range(len(sin_funcs)-1):
            self.play(Transform(sin_funcs[i], sin_funcs[i+1], rate_func=linear))


        self.wait(5)


        #  manim -pqh discord_3.py sin_func_shrink



value = [4, 5, 6, 1, 4]

class Pie(Scene):
    def construct(self):

        angles = [x/sum(value)*360*DEGREES for x in value]
        pies = []
        a = 0

        circle = Circle(stroke_color=GREY).set_fill(opacity=0)

        for i, angle in enumerate(angles):

            piece = ArcPolygon(circle.get_center(), circle.point_at_angle(a), circle.point_at_angle(angle), circle.get_center(), radius=1, arc_config={"color": random_color(), "stroke_color": GREY, "fill_opacity": 0.5})
            piece.add_points_as_corners([circle.get_center(), piece.get_start()])
            a += angle
            pies.append(piece)
        
        sq=ArcPolygon(ORIGIN, RIGHT, RIGHT+UP, UP, radius=1)
        

        self.play(Create(circle))
        self.play(Create(sq))
            
        for piece in pies:
            self.play(Create(piece))

        self.wait(1)


        #  manim -pqh discord_3.py Pie




class test17(Scene):
    def construct(self):
        baselen = 0.5
        max_n   = 10
        origo   = 3*LEFT+3*DOWN
        sq = Square(side_length=baselen)
        triangle = VGroup()  # create an empty group

        for x in np.arange(1, max_n+1, 1):
            for y in np.arange(1, x+1, 1):

                triangle += sq.copy().move_to(x*baselen*RIGHT + y*baselen*UP + origo) 

                self.add(triangle[-1]) # retrieve latest added sq and Create

        # now make a copy of the triangle
        new_triangle = triangle.copy().set_color(RED) 

        self.play(Create(new_triangle))

        self.play(Rotate(new_triangle, 720*DEGREES), run_time=4)


        # manim -pqh discord_3.py test17



class multiplication(Scene):
    def construct(self):

        
        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        for a in [1,2,3]:
            theTable = []
            for b in range(10):
                theTable.append([a, r"\times", b+1, r"=", a*(b+1)])
            texTable = MathTable(theTable,include_outer_lines=False).scale(0.6).to_edge(UP)
            texTable.remove(*texTable.get_vertical_lines())
            texTable.remove(*texTable.get_horizontal_lines())

            self.play(Write(texTable))

            self.wait(5)
            self.remove(texTable)


            # manim -pqh discord_3.py multiplication



class square_from_line(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        line1 = Line(start=ORIGIN, end=1*RIGHT)
        line2 = line1.copy().shift(1*RIGHT)        
        line3 = line2.copy().shift(1*RIGHT)        
        line4 = line3.copy().shift(1*RIGHT)
        all = VGroup(line1,line2,line3,line4)
        grp2 = VGroup(line2,line3,line4)       
        grp3 = VGroup(line3,line4)
        self.play(Write(all))
        self.wait(2)
        self.play(Rotate(grp2, angle=90*DEGREES, about_point=1*RIGHT))        
        self.play(Rotate(grp3, angle=90*DEGREES, about_point=1*RIGHT+1*UP))
        self.play(Rotate(line4, angle=90*DEGREES, about_point=1*UP))        
        self.wait(2)


        # manim -pqh discord_3.py square_from_line



class line2square(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        line = Line(start=1*RIGHT+3*DOWN, end=3*RIGHT+0*UP)

        self.play(Create(line))
        self.wait(2)
        
        length = line.get_length()
        angle  = line.get_angle()

        square = Square(side_length=length).move_to(ORIGIN, DL)
        self.play(Create(square))

        self.play(Rotate(square, angle=angle, about_point=ORIGIN))
        self.play(square.animate.shift(line.start))

        self.wait(2)


        # manim -pqh discord_3.py line2square


class MobsInFront(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        circ = Circle(radius=1,fill_color=PINK,fill_opacity=1,
            stroke_color=PINK,stroke_opacity=1).set_z_index(2)
        edge = Dot(circ.get_right())
        anim = Flash(edge,color=BLUE,run_time=2,line_length=1)
        circ.add_updater(
            lambda l: l.become(
                Circle(arc_center=[0,0,1],radius=1,fill_color=PINK,
                fill_opacity=1,stroke_color=PINK,stroke_opacity=1)
            )
        )

        self.add(circ)
        self.play(anim)
        self.wait()  


        # manim -pqh discord_3.py MobsInFront



class Combine(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        some_text = Tex(
            r"$P(\hspace{2em}$"
        )
        square = Square(0.5, color=RED, fill_color=RED, fill_opacity=0.8)
        square.next_to(some_text, buff=0.1)
        some_other_text = Tex (r"$)$")
        some_other_text.next_to(square,buff=0.1)

        grp=VGroup(some_text, square, some_other_text)

        self.play(Create(grp))


        # manim -pqh discord_3.py Combine




class tracesquare(MovingCameraScene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        grid = NumberPlane()
        self.add(grid)
        s = Square(side_length=2, color=RED, fill_opacity=0.2)
        self.add(grid, s)


        for dir in [UP * 2,LEFT * 2,DOWN * 5,RIGHT * 5,LEFT * 6]:
            a = s.copy()
            s.set_color(color=GREEN)
            s.set_fill(color=GREEN)
            s.set_opacity(0.4)
            self.play(a.animate.shift(dir))
            s = a

        self.wait(1)


        # manim -pqh discord_3.py tracesquare



class MeineSzene(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        line1 = Line(3*LEFT, 3*RIGHT).shift(UP).set_color(RED)
        line2 = Line(3*LEFT, 3*RIGHT).set_color(GREEN)

        d1 = Dot().move_to(line1.get_left())
        d2 = Dot().move_to(line2.get_left())

        label1 = Tex("smooth").next_to(line1, RIGHT)
        label2 = Tex("linear").next_to(line2, RIGHT)


        tr1=ValueTracker(-3)
        tr2=ValueTracker(-3)


        d1.add_updater(lambda z: z.set_x(tr1.get_value()))
        d2.add_updater(lambda z: z.set_x(tr2.get_value()))
        self.add(d1,d2)
        self.add(line1,line2,d1,d2,label1,label2 )

        self.play(tr1.animate(rate_func=smooth).set_value(3), tr2.animate(rate_func=linear).set_value(3))
        self.wait()


        # manim -pqh discord_3.py MeineSzene



class LightArcEx(Scene):
    @staticmethod
    def colorfunction(s, freq = 50, **kwargs):
        return interpolate_color(BLUE, RED, (1 + np.cos(freq * PI * s)) / 2)

    def light_arc(self, f, **kwargs):
        curve = ParametricFunction(f, **kwargs)
        pieces = CurvesAsSubmobjects(curve)

        length_diffs = [ curve.get_nth_curve_length(n)
                                    for n in range(curve.get_num_curves()) ]
        length_parts = np.cumsum(length_diffs)
        total_length = length_parts[-1]
        colors = [ self.colorfunction(length_parts[n]/total_length, **kwargs)
                                    for n in range(curve.get_num_curves()) ]

        pieces.set_color_by_gradient(*colors)

        return pieces
    
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        
        arcs = [ self.light_arc(lambda t : [-3 * np.cos(t), (2 + 0.01 *c) * np.sin(t) - 1, 0],
                          t_range=[0, PI, 0.01]) for c in range(100) ]

        self.play(ShowSubmobjectsOneByOne(arcs))


        # manim -pqh discord_3.py LightArcEx



class ex2(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        def get_slope_from_path(path, alpha, dx=0.001):
            sign = 1 if alpha < 1-dx else -1
            return angle_of_vector(sign * path.point_from_proportion(alpha + sign * dx) - sign * path.point_from_proportion(alpha))

        b = VMobject().set_points_smoothly([
            LEFT*3+UP*2,RIGHT*2+DOWN*1.7,DOWN*2+LEFT*2.5,UP*1.7+RIGHT*2
            ])
        arrow = Triangle(fill_opacity=1)\
                .set(width=0.3)\
                .move_to(b.get_start())
        arrow.save_state()

        arrow.rotate(get_slope_from_path(b,0)-PI/2)

        def update_arrow(mob,alpha):
            mob.restore()
            mob.move_to(b.point_from_proportion(alpha))
            mob.rotate(get_slope_from_path(b,alpha)-PI/2)

        self.add(b,arrow)
        self.play(
                UpdateFromAlphaFunc(
                    arrow, update_arrow
                    ),
                run_time=4
                )
        self.wait()




        # manim -pqh discord_3.py ex2




class TransformMatchingID(TransformMatchingShapes):
    @staticmethod
    def get_mobject_parts(mobject: Mobject) -> list[Mobject]:
        return mobject

    @staticmethod
    def get_mobject_key(mobject):
        return mobject.id


class Coin(VMobject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        p = np.random.rand()
        if p < 0.5:
            self.symbol = "P"
            color = BLUE
        else:
            self.symbol = "F"
            color = RED

        self.id = np.random.rand()
        self.contour = Circle(
            radius=0.5, color=color, fill_color=color, fill_opacity=1, stroke_width=1
        )
        self.add(self.contour)

    def __str__(self) -> str:
        return self.symbol


class Test2(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        v = VGroup(*[Coin() for _ in range(10)])
        v.arrange()
        self.play(FadeIn(v))
        self.wait()
        for _ in range(5):
            new_v = v.copy().arrange()
            new_v.sort(submob_func=lambda m: m.symbol)
            self.play(
                TransformMatchingID(
                    v,
                    new_v,
                )
            )
            v = new_v
        self.wait()



        # manim -pqh discord_3.py Test2




class teststuff(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        num_line = NumberLine()
        line = Line(LEFT,RIGHT,color=RED)
        b1 = Brace(line).add_updater(lambda m: m.next_to(line,DOWN))
        b1text = b1.get_tex("\\sum_{i=1}^{n} k_{i}").next_to(b1,DOWN)
        b2text = b1.get_tex("\\sum_{i=1}^{n} k_{i} + j").next_to(b1,DOWN)
        b3text = b1.get_tex("\\sum_{i=1}^{n} k_{i} - j").next_to(b1,DOWN)


        self.play(Write(num_line))
        self.play(Write(line))
        self.play(Write(b1),Write(b1text))

        b1text.add_updater(lambda m: m.next_to(b1,DOWN))
        b2text.add_updater(lambda m: m.next_to(b1,DOWN))
        b3text.add_updater(lambda m: m.next_to(b1,DOWN))

        self.play(line.animate.shift(LEFT), ReplacementTransform(b1text, b2text))
        self.wait(2)

        b1text = b1.get_tex("\\sum_{i=1}^{n} k_{i}").add_updater(lambda m: m.next_to(b1,DOWN))

        self.play(line.animate.shift(RIGHT), ReplacementTransform(b2text, b1text))
        self.wait(2)
        self.play(line.animate.shift(RIGHT), ReplacementTransform(b1text, b3text))


        # manim -pqh discord_3.py teststuff


