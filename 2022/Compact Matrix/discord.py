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


        # manim -pqh discord.py test17



class multiplication(Scene):
    def construct(self):
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


            # manim -pqh discord.py multiplication



class square_from_line(Scene):
    def construct(self):
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


        # manim -pqh discord.py square_from_line



class line2square(Scene):
    def construct(self):
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


        # manim -pqh discord.py line2square



class MobsInFront(Scene):
    def construct(self):
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


        # manim -pqh discord.py MobsInFront



class Combine(Scene):
    def construct(self):
        some_text = Tex(
            r"$P(\hspace{2em}$"
        )
        square = Square(0.5, color=RED, fill_color=RED, fill_opacity=0.8)
        square.next_to(some_text, buff=0.1)
        some_other_text = Tex (r"$)$")
        some_other_text.next_to(square,buff=0.1)

        grp=VGroup(some_text, square, some_other_text)

        self.play(Create(grp))


        # manim -pqh discord.py Combine



class tracesquare(MovingCameraScene):
    def construct(self):
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


        # manim -pqh discord.py tracesquare



class MeineSzene(Scene):
    def construct(self):
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


        # manim -pqh discord.py MeineSzene



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
        
        arcs = [ self.light_arc(lambda t : [-3 * np.cos(t), (2 + 0.01 *c) * np.sin(t) - 1, 0],
                          t_range=[0, PI, 0.01]) for c in range(100) ]

        self.play(ShowSubmobjectsOneByOne(arcs))


        # manim -pqh discord.py LightArcEx



class ex2(Scene):
    def construct(self):
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




        # manim -pqh discord.py ex2



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



        # manim -pqh discord.py Test2



class teststuff(Scene):
    def construct(self):
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


        # manim -pqh discord.py teststuff




quality_factor = 1
fps = 30

config['pixel_height'] = int(1920/quality_factor)
config['pixel_width'] = int(1080/quality_factor)
config['frame_height'] = 14
config['frame_width'] = 8
config['frame_rate'] = fps

class integral_1(Scene):
    def construct(self):
        ## Titulo y subtitulo
        title = Tex(r'Riemann Integration').to_edge(UP*1.5).scale_to_fit_width(config["frame_width"]-1.5)
        rect_title = SurroundingRectangle(title,buff = 0.4).set_color_by_gradient([REANLEA_AQUA_GREEN,REANLEA_BLUE])

        brand = Text(
            "LET Me Explain",
            fill_opacity = 1,
            color = WHITE,
            font = "Arial Rounded MT Bold",
            t2c = {"[:1]":BLUE,"[3:4]":BLUE,"[5:6]":BLUE} ## Los espacios no cuentan como caracteres
        ).scale(0.4).next_to(rect_title,DOWN,aligned_edge = RIGHT)

        subtitle = Tex('Gobinda Chandra').scale(0.55).set_opacity(0.6).next_to(rect_title,DOWN,aligned_edge = LEFT)

        self.add(title)
        self.play(Create(rect_title), FadeIn(brand), Write(subtitle))
        self.wait()

        ax = Axes(
            x_range=[0, 10],
            y_range=[0, 10], 
            x_length = 5, 
            y_length = 5, 
            axis_config={"include_tip": False}
        ).shift(DL*.5)
        labels = ax.get_axis_labels(x_label="x", y_label="y").set_opacity(0.6)
        graph = ax.get_graph(lambda x: ((x-4.5)/2)**3 - x + 11, x_range=[0, 8.5]).set_color(ORANGE)
        graph_lbl = ax.get_graph_label(graph,'y = f(x)',direction = UR, x_val = 8.5).set_color(ORANGE)

        x_1 = 2
        x_2 =  8

        riemann = ax.get_riemann_rectangles(graph, x_range=[x_1,x_2], dx=1, stroke_width=1)
        brace = VGroup(Line(ax.c2p(x_1,0),ax.c2p(x_1+1,0)).set_color(LIGHT_PINK),MathTex(r'\Delta x').set_color(LIGHT_PINK).scale(0.8).move_to(ax.c2p(x_1+1,-1)))

        a_line = ax.get_vertical_line(ax.i2gp(x_1,graph))
        a_line_lbl = MathTex(r'a').scale(0.7).next_to(ax.c2p(x_1,0),DOWN)
        a_graph_lbl = ax.get_graph_label(graph,'f(a)',direction = UP*1.3, x_val = x_1).set_color(ORANGE).scale(0.9)

        lines1_group = VGroup(a_line_lbl,a_line)

        b_line = ax.get_vertical_line(ax.i2gp(x_2,graph))
        b_line_lbl = MathTex(r'b').scale(0.6).next_to(ax.c2p(x_2,0),DOWN)
        b_graph_lbl = ax.get_graph_label(graph,'f(b)',direction = RIGHT, x_val = x_2).set_color(ORANGE).scale(0.9)

        lines2_group = VGroup(b_line_lbl,b_line)

        self.play(
            Create(ax),
            FadeIn(labels)
        )
        self.play(Create(graph))
        self.play(FadeIn(graph_lbl))
        self.play(
            Create(lines1_group)
        )
        self.play(
            Create(lines2_group)
        )

        self.play(FadeIn(riemann[0]))
        self.wait(0.5)
        self.play(FadeIn(a_graph_lbl))
        self.play(FadeIn(brace))
        self.wait(0.5)

        eq_0 = MathTex(r'A_a',r'=',r'f(a)\,',r'\Delta x').next_to(rect_title,DOWN,buff=1,aligned_edge = LEFT)
        eq_0[0].set_color(riemann[0].get_fill_color())
        eq_0[2].set_color(ORANGE)
        eq_0[3].set_color(LIGHT_PINK)

        self.play(FadeIn(eq_0,target_position=riemann))
        self.wait()
        self.play(FadeOut(a_graph_lbl))
        self.play(FadeIn(riemann[1:]),lag_ratio=1,run_time = 2.5)

        eq_1 = MathTex(r'A_{ab}',r'=',r'\sum_{i=a}^{b}',r'f(i)\,',r'\Delta x').next_to(rect_title,DOWN,buff=1,aligned_edge = LEFT)
        eq_1[0].set_color_by_gradient([riemann[0].get_fill_color(),riemann[-1].get_fill_color()])
        eq_1[3].set_color(ORANGE)
        eq_1[4].set_color(LIGHT_PINK)

        self.play(
            FadeOut(eq_0,shift=UP),
            FadeIn(eq_1,shift=UP),
        )
        self.wait()

        for i in [.5,.2,.1,.05]:
            if i == 0.05:
                self.play(
                    riemann.animate.become(ax.get_area(graph, x_range=[x_1,x_2], opacity=1)),
                    brace[1].animate.become(MathTex(r'dx').set_color(LIGHT_PINK).scale(0.8).move_to(ax.c2p(x_1+1,-1))),
                    brace[0].animate.become(Dot(color=LIGHT_PINK).move_to(ax.c2p(x_1,0)))
                )
            else:
                self.play(
                    riemann.animate.become(ax.get_riemann_rectangles(graph, x_range=[x_1,x_2], dx=i, stroke_width=i)),
                    brace[0].animate.become(Line(ax.c2p(x_1,0),ax.c2p(x_1+i,0)).set_color(LIGHT_PINK))
                )
            self.wait(0.5)

        eq_2 = MathTex(r'A_{ab}',r'=',r'\displaystyle \int_{a}^{b}',r'f(x)\,',r'dx').next_to(rect_title,DOWN,buff=1,aligned_edge = LEFT)
        eq_2[0].set_color_by_gradient([riemann[0].get_fill_color(),riemann[-1].get_fill_color()])
        eq_2[3].set_color(ORANGE)
        eq_2[4].set_color(LIGHT_PINK)

        self.play(
            FadeOut(eq_1,shift=UP),
            FadeIn(eq_2,shift=UP),
        )

        self.wait()

        self.play(eq_2.animate.set_color(WHITE))
        self.play(Indicate(eq_2,color=PURE_GREEN))

        self.wait()


        # manim -pqh discord.py integral_1




class VMobject(VMobject):
    def pfp(self, alpha):
        return self.point_from_proportion(alpha)


class TestVm(Scene):
    def construct(self):
        #c = Circle()
        c = VMobject(stroke_color=GREEN).set_points_smoothly([
                LEFT*2+UP*1.2, LEFT+UP*(-3), RIGHT*2+DOWN*1.7,
                DOWN*2+LEFT*2.5
            ])
        a = Dot(color = YELLOW)

        self.add(c, a)

        self.play(UpdateFromAlphaFunc(a, lambda x, alpha: x.move_to(c.pfp(alpha))), run_time = 3, rate_func= smooth)
        self.wait()


        # manim -pqh discord.py TestVm


class test22(Scene):
    def construct(self):
        clockface = VGroup()
        for t in range(12):
            clockface += Line(start = ORIGIN, end = 2*UP).rotate(-(t+1)/12*360*DEGREES, about_point=ORIGIN)
            lbl = MathTex(r"{:.0f}".format(t+1))
            clockface += lbl.move_to(clockface[-1].get_end())

        self.play(Create(clockface))
        self.wait(2)

        # manim -pqh discord.py test22


'''class ClockFaces(Scene):
    def draw_text_lines(self, line1, line2, offset=np.array([3.5, 0.5, 0])):
        text_heading = Text(line1)
        text_heading.shift(offset)
        text_body = Text(line2)
        text_body.next_to(text_heading, DOWN)

        return text_heading, text_body

    def construct(self):

        line, _ = self.draw_text_lines("Imagine a clockface", "")
        self.play(FadeIn(line))
        self.wait()

        ### DRAW 12HR CLOCK
        plane = PolarPlane(radius_max=2,
                           azimuth_step=12,
                           azimuth_units='degrees',
                           azimuth_direction='CW',
                           radius_config={
                               "stroke_width": 0,
                               "include_ticks": False
                           },
                           azimuth_offset=np.pi / 2).add_coordinates()
        plane.shift(np.array([-3.5, 0, 0]))
        self.play(LaggedStart(Write(plane), run_time=3, lag_ratio=0.5))
        self.wait()


        # manim -pqh discord.py ClockFaces
'''


class spinning(Scene):
    def construct(self):

        self.acc_time = 0
        self.vect1 = 0*LEFT
        self.vect2 = 0*LEFT
        self.vect1_ampl  = 2
        self.vect2_ampl  = self.vect1_ampl/2
        self.vect1_freq  = 1
        self.vect2_freq  = 2
        def sceneUpdater(dt):
            self.acc_time += 0.5*dt
            self.vect1 = self.vect1_ampl*(np.sin(self.acc_time*self.vect1_freq)*UP + np.cos(self.acc_time*self.vect1_freq)*RIGHT)
            self.vect2 = self.vect2_ampl*(np.sin(self.acc_time*self.vect2_freq)*UP + np.cos(self.acc_time*self.vect2_freq)*RIGHT)
        self.add_updater(sceneUpdater)
        
        dyn_vect1_arrow = VMobject()
        def vect1_updater(mobj):
            dyn_vect1_arrow.become(Arrow(start=ORIGIN,end=self.vect1,buff=0).set_color(BLUE))
        dyn_vect1_arrow.add_updater(vect1_updater)

        dyn_vect2_arrow = VMobject()
        def vect2_updater(mobj):
            dyn_vect2_arrow.become(Arrow(start=self.vect1,end=self.vect1+self.vect2,buff=0).set_color(RED))
        dyn_vect2_arrow.add_updater(vect2_updater)

        self.add(dyn_vect1_arrow, dyn_vect2_arrow)

        self.wait(30)   



        # manim -pqh discord.py spinning



class spinning1(Scene):
    def construct(self):

        self.acc_time = 0
        self.vect1 = 0*LEFT
        self.vect2 = 0*LEFT
        self.vect1_ampl  = 2
        self.vect2_ampl  = self.vect1_ampl/2
        self.vect1_freq  = 1
        self.vect2_freq  = 10

        def sceneUpdater(dt):
            self.acc_time += dt
            self.vect1 = self.vect1_ampl*(np.sin(self.acc_time*self.vect1_freq)*UP + np.cos(self.acc_time*self.vect1_freq)*RIGHT)
            self.vect2 = self.vect2_ampl*(np.sin(self.acc_time*self.vect2_freq)*UP + np.cos(self.acc_time*self.vect2_freq)*RIGHT)
        self.add_updater(sceneUpdater)
        
        dyn_vect1_arrow = VMobject()
        def vect1_updater(mobj):
            dyn_vect1_arrow.become(Arrow(start=ORIGIN,end=self.vect1,buff=0).set_color(BLUE))
        dyn_vect1_arrow.add_updater(vect1_updater)

        dyn_vect2_arrow = VMobject()
        def vect2_updater(mobj):
            dyn_vect2_arrow.become(Arrow(start=self.vect1,end=self.vect1+self.vect2,buff=0).set_color(RED))
        dyn_vect2_arrow.add_updater(vect2_updater)

        self.add(dyn_vect1_arrow, dyn_vect2_arrow)

        self.wait(30) 


        # manim -pqh discord.py spinning1


class test4(Scene):
    def construct(self):
        for i in range(5, -1, -1):
            Text = MathTex(r"{:d} \over {:d}".format(i, i+1)).move_to(i*LEFT)
            self.add(Text)
            self.wait(1)

        self.wait(2)


        # manim -pqh discord.py test4


class sep14(Scene):
    def construct(self):
        text = r"blabla askdjf  askdjfslk asdkfj asdfkj sdflajkd f"

        ques = Tex(text, tex_environment = "flushleft")
        op1 = Tex("(A) 1",
        tex_to_color_map={"(A)":YELLOW})
        op2 = Tex("(C) 2",
        tex_to_color_map={"(B)":YELLOW})
        op3 = Tex("(B) 3",
        tex_to_color_map={"(C)":YELLOW})
        op4 = Tex("(D) 4",
        tex_to_color_map={"(D)":YELLOW})
        ques.to_corner(UP, buff=2)
        options1 = VGroup(op1, op2, op3, op4).arrange(DOWN)
        options1.next_to(ques,DOWN, buff=1, aligned_edge = LEFT)

        self.play(FadeIn(ques))
        self.play(AnimationGroup(*[FadeIn(member) for member in options1], lag_ratio=0.5))
        
        self.wait(0)


        # manim -pqh discord.py sep14