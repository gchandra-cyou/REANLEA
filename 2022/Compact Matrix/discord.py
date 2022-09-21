from __future__ import annotations
from asyncore import poll2
from audioop import add
from cProfile import label
from distutils import text_file

from math import pi

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

class Scene(ThreeDScene):
    def construct(self):

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


    # manim -pqh discord.py Scene




num = 10

class Mouche(Dot):
    def __init__(self, position, velocity, color):
        Dot.__init__(self, position, color=color)
        self.position = position
        self.velocity = velocity
        self.path = TracedPath(self.get_center) #checkout the usage of get_center without () 


class ExoMouches(MovingCameraScene):
    # def __init__(self, **kwargs):
    #     ZoomedScene.__init__(
    #         self,
    #         zoom_factor=0.4,
    #         zoomed_display_height=1.5,
    #         zoomed_display_width=9,
    #         image_frame_stroke_width=20,
    #         zoomed_camera_config={
    #             "default_frame_stroke_width": 3,
    #         },
    #         **kwargs
    #     )
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



        # manim -pqh discord.py ExoMouches



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



        # manim -pqh discord.py Shrink_sin_func


class Shrink_sin_func1(Scene):
    def construct(self):

        ax = Axes(
            x_range=(-1, 3),
            y_range=(-3, 3)
        )

        tracker = ValueTracker(0.1)


        sine_function = lambda t: np.sin(tracker.get_value() * t)
        sine_graph = always_redraw(lambda: ax.plot(
            sine_function,
            color=BLUE
        ))
        self.add(ax)
        self.add(sine_graph)

        # Animate the sine wave from y=sin(0.1*x) to y=sin(10*x) over the course of 6 seconds.
        self.play(tracker.animate(run_time=6).set_value(
            tracker.get_value() * 100))



        #  manim -pqh discord.py Shrink_sin_func1


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


        #  manim -pqh discord.py sin_func_shrink



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


        #  manim -pqh discord.py Pie





###########################################################################################################

from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import root_scalar
import numpy as np
from func import angle_between_vectors_signed



def Round_Corner_Param(radius,curve_points_1,curve_points_2):
    bez_func_1 = bezier(curve_points_1)
    diff_func_1 = bezier((curve_points_1[1:, :] - curve_points_1[:-1, :]) / 3)
    bez_func_2 = bezier(curve_points_2)
    diff_func_2= bezier((curve_points_2[1:, :] - curve_points_2[:-1, :]) / 3)

    def find_crossing(p1,p2,n1,n2):
        t = fsolve(lambda t: p1[:2]+n1[:2]*t[0]-(p2[:2]+n2[:2]*t[1]),[0,0])
        return t, p1+n1*t[0]

    def rad_cost_func(t):
        angle_sign = np.sign( angle_between_vectors_signed(diff_func_1(t[0]),diff_func_2(t[1])))
        p1 = bez_func_1((t[0]))
        n1 = normalize(rotate_vector(diff_func_1(t[0]),angle_sign* PI / 2))
        p2 = bez_func_2((t[1]))
        n2 = normalize(rotate_vector(diff_func_2(t[1]), angle_sign* PI / 2))
        d = (find_crossing(p1, p2, n1, n2))[0]
        # 2 objectives for optimization:
        #  - the normal distances should be equal to each other
        #  - the normal distances should be equal to the target radius
        # I'm hoping that in this form at least a tangent circle will be found (first goal),
        # even if there is no solution at the desired radius. I don't really know, fsolve() and roots() is magic.
        return ((d[0])-(d[1])),((d[1])+(d[0])-2*radius)

    k = root(rad_cost_func,np.asarray([0.5,0.5]),method='hybr')['x']

    p1 = bez_func_1(k[0])
    n1 = normalize(rotate_vector(diff_func_1(k[0]), PI / 2))
    p2 = bez_func_2(k[1])
    n2 = normalize(rotate_vector(diff_func_2(k[1]), PI / 2))
    d, center = find_crossing(p1, p2, n1, n2)
    r = abs(d[0])
    start_angle = np.arctan2((p1-center)[1],(p1-center)[0])
    cval = np.dot(p1-center,p2-center)
    sval = (np.cross(p1-center,p2-center))[2]
    angle = np.arctan2(sval,cval)

    out_param = {'radius': r, 'arc_center': center, 'start_angle': start_angle, 'angle': angle}

    return out_param, k


def Round_Corners(mob:VMobject,radius=0.2):
    i=0
    while i < mob.get_num_curves() and i<1e5:
        ind1 = i % mob.get_num_curves()
        ind2 = (i+1) % mob.get_num_curves()
        curve_1 = mob.get_nth_curve_points(ind1)
        curve_2 = mob.get_nth_curve_points(ind2)
        handle1 = curve_1[-1,:]-curve_1[-2,:]
        handle2 = curve_2[1, :] - curve_2[0, :]
        # angle_test = (np.cross(normalize(anchor1),normalize(anchor2)))[2]
        angle_test = angle_between_vectors_signed(handle1,handle2)
        if abs(angle_test)>1E-6:
            params, k = Round_Corner_Param(radius,curve_1,curve_2)
            cut_curve_points_1 = partial_bezier_points(curve_1, 0, k[0])
            cut_curve_points_2 = partial_bezier_points(curve_2, k[1], 1)
            loc_arc = Arc(**params,num_components=5)
            # mob.points = np.delete(mob.points, slice((ind1 * 4), (ind1 + 1) * 4), axis=0)
            # mob.points = np.delete(mob.points, slice((ind2 * 4), (ind2 + 1) * 4), axis=0)
            mob.points[ind1 * 4:(ind1 + 1) * 4, :] = cut_curve_points_1
            mob.points[ind2 * 4:(ind2 + 1) * 4, :] = cut_curve_points_2
            mob.points = np.insert(mob.points,ind2*4,loc_arc.points,axis=0)
            i=i+loc_arc.get_num_curves()+1
        else:
            i=i+1

        if i==mob.get_num_curves()-1 and not mob.is_closed():
            break

    return mob


class Test_round(Scene):
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

        mob1 = Round_Corners(mob1, 0.25)
        mob2 = Round_Corners(mob2, 0.25)
        mob3 = Round_Corners(mob3, 0.25)
        #self.add(mob1,mob2,mob3,mob4)

        self.play(
            Create(mob1)
        )
        self.wait(2)


        # manim -pqh discord.py Test_round




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

         # manim -pqh discord.py test_dimension_pointer


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


        # manim -pqh discord.py test_dimension_base



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



         # manim -pqh discord.py test_dash







###################################################################################################################