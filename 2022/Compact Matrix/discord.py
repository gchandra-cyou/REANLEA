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
import requests
import io
from PIL import Image
from random import choice, seed
from random import random, seed
from enum import Enum


config.background_color= REANLEA_BACKGROUND_COLOR


###################################################################################################################




def makeTorus(major_radius=3, minor_radius=1, theta_step=30, theta_offset=0):
    torus = VGroup()
    for n in range(0,360,theta_step):
        torus.add(Circle(radius=major_radius-minor_radius*np.cos((n+theta_offset)*DEGREES)).shift(minor_radius*np.sin((n+theta_offset)*DEGREES)*OUT).set_color(BLUE))
    return torus

class try_torus(ThreeDScene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        major_radius = 3
        minor_radius = 1

        self.move_camera(phi=60*DEGREES)

        torus = makeTorus(major_radius=major_radius, minor_radius=minor_radius, theta_step=30, theta_offset=0)

        

        self.play(Create(torus))

        for offset in range(60):
            self.play(torus.animate.become(makeTorus(major_radius=major_radius, minor_radius=minor_radius, theta_step=30, theta_offset=offset)), run_time=0.2)

        self.wait(2)



        # manim -pqh discord.py try_torus




def mkVGroup(mainVMobject, indices):
    grp = VGroup()
    for i in indices:
        grp.add(mainVMobject[i].copy())
    return grp

class svg_electronics(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        bgrid = VGroup()
        for x in range(-7, 8):            
            for y in range(-4, 5):
                if (x==0) and (y==0):
                    bgrid.add((Dot(np.array([x, y, 0]),radius=0.03, color=RED)))        
                else:
                    bgrid.add(Dot(np.array([x, y, 0]),radius=0.03, color=DARK_GREY))   

        self.acc_time  = 0
        self.acc_speed = 0
        def sceneUpdater(dt):
            self.acc_time += dt * self.acc_speed  
        self.add_updater(sceneUpdater)     

        self.add(bgrid)

        circuit = SVGMobject("20221004_mosfet.svg", 
            stroke_color=WHITE, stroke_width=1, fill_color=WHITE, height=5)

        comp = {}
        
        comp['MOSFET'] = mkVGroup(circuit, [0,1,2,3,4,5,6,7,8,9,10,11]).set_color(WHITE)
        comp['G-term'] = mkVGroup(circuit, [16,12,13]).set_color(WHITE)
        comp['D-resistor'] = mkVGroup(circuit, [14,21,22,23,24,25,26]).set_color(WHITE)
        comp['D-term'] = mkVGroup(circuit, [27,17,18]).set_color(WHITE)
        comp['S-resistor'] = mkVGroup(circuit, [15,28,29,30,31,32,33]).set_color(WHITE)
        comp['S-term'] = mkVGroup(circuit, [34,19,20]).set_color(WHITE)

        for mobj in comp.values():
            self.play(Create(mobj))
            self.wait(1)

        for mobj in comp.values():
            self.play(Wiggle(mobj))
            self.wait(1)    

        self.wait(2)    


        # manim -pqh discord.py svg_electronics




class svg_electronics_index(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        for x in range(-7, 8):
            for y in range(-4, 5):
                if (x==0) and (y==0):
                    self.add(Dot(np.array([x, y, 0]),radius=0.03, color=RED))        
                else:
                    self.add(Dot(np.array([x, y, 0]),radius=0.03, color=DARK_GREY))

        circuit = SVGMobject("20221004_mosfet.svg", 
            stroke_color=WHITE, stroke_width=0.02, fill_color=WHITE, height=5)

        self.add(circuit)
        #self.wait(2)

        bbox = Rectangle(height=circuit.height, width=circuit.width).move_to(circuit.get_center(),ORIGIN)
        bbox.scale(1.4)
        loc = bbox.get_critical_point(UL)
        w = bbox.width
        h = bbox.height
        cf = 2*w + 2*h
        dl = cf / (len(circuit)+3)

        dir   = [dl,0,0]
        edge = 0
        positions = []
        for i in range(len(circuit)):
            positions.append(loc)
            loc = loc + dir
            if (edge == 0) and (loc[0] > bbox.get_critical_point(UP+RIGHT)[0]):
                edge = 1
                loc = loc - dir
                dir = [0,-dl,0]
                loc = loc + dir
                
            if (edge == 1) and (loc[1] < bbox.get_critical_point(DOWN+RIGHT)[1]):
                edge = 2
                loc = loc - dir
                dir = [-dl,0,0]
                loc = loc + dir

            if (edge == 2) and (loc[0] < bbox.get_critical_point(DOWN+LEFT)[0]):
                edge = 3
                loc = loc - dir
                dir = [0,+dl,0]
                loc = loc + dir

        for i in range(len(circuit)):
            shortest = 1e6
            found = 0
            for j in range(len(positions)):
                dist = np.sqrt((circuit[i].get_center()[0]-positions[j][0])**2 + (circuit[i].get_center()[1]-positions[j][1])**2)
                if dist < shortest:
                    shortest = dist
                    found = j

            txt = Text("{}".format(i)).scale(0.3).move_to(positions[found])                
                
            line = Line(circuit[i].get_center(),end=txt.get_center(), stroke_width=1)

            #self.add(line)
            #self.add(txt)
            # self.wait(1)
            self.play(Create(line),Create(txt))

            positions.pop(found)

        #self.wait(2)



        # manim -pqh discord.py svg_electronics_index



class ColoredCircuit(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        template = TexTemplate()
        template.add_to_preamble(r"\usepackage[siunitx, straightvoltages, RPvoltages, european]{circuitikz}")
        c = MathTex(
            r"""\draw (0,0) to[isource, l=$I_0$, v=$U_0$] (0,3);""", 
            r"""\draw (0,3) to[short, -*] (2,3);""",
            r"""\draw (2,3) to[R=$R_1$, i>_=$I_1$] (2,0);""",
            r"""\draw (2,3) -- (4,3);""", 
            r"\draw (4,3) to[R=$R_2$, i>_=$I_2$] (4,0);",
            r"\draw (4,0) to[short, -*] (2,0)--(0,0);"
            , stroke_width=2 
            , fill_opacity=0
            , stroke_opacity=1
            , tex_environment="circuitikz"
            , tex_template=template
            
            )
        # for cir, clr in zip(c[0,4],[RED, GREEN, BLUE, YELLOW]):
        #     cir.set_color(clr)
        c.set_color_by_tex_to_color_map({"I_0":RED, "R_1":YELLOW, "R_2":BLUE})
        upperWire = Group(c[1],c[3])
        upperWire.set_color(GREEN)
        c[3].stroke_width = 8
        self.play(Create(c, shift=UP, target_position=ORIGIN), run_time=10)
        self.play(ApplyWave(upperWire))
        self.play(Indicate(c[2], color=TEAL), run_time=2) 
        self.play(Circumscribe(c[4], fade_out=True, color=BLUE))
        self.wait(2)


        # manim -pqh discord.py ColoredCircuit



class eqn_underline_equal(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        some_eq = MathTex(
            r"x+1+2+4&",r"=x+1+6\\ &",r"= x+7",
        )
        self.play(
            AnimationGroup(
                *[Write(eq) for eq in some_eq],
                lag_ratio=2
            )
        )



        # manim -pqh discord.py eqn_underline_equal



def follow_bezier_path(start, stop, *control_points):
    pts = [start] + list(control_points) + [stop]
    b = bezier(np.array(pts))
    start = np.array(start)
    stop = np.array(stop)
    def bezier_path_func(src, dst, t):
        assert len(src) == len(dst)
        # Find where the center is
        curr_center = b(t)
        # Find where the center should be if it was just on a straight line
        # between start and stop
        lerped = (1 - t) * start + t * stop
        # Transform all the points as if they were just on a straight line
        # between start and stop
        lerped_pts = (1 - t) * src + t * dst
        # subtracting the two lerps gives us the relative distance between
        # the center and the partially transformed shape. Adding on the
        # location of where we are along the bezier line moves all the
        # points in accordance of that.
        return (lerped_pts - lerped) + curr_center
    return bezier_path_func


class Star_FollowBezier(Scene):
  def construct(self):
    #self.camera.background_color = WHITE

    # WATER MARK 

    water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
    self.add(water_mark)

    start = [-1, 0, 0]
    stop = [1, 1, 0]
    detour1 = [-2, 4, 0]
    detour2 = [3, -8, 0]

    rect = Rectangle(width=1, height=1, color=REANLEA_GREY, fill_color=BLACK, fill_opacity=1.0).move_to(start)
    star = Star(outer_radius=2, color=REANLEA_BLUE_DARKEST, fill_color=RED, fill_opacity=1.0).move_to(stop)

    bezier_path = follow_bezier_path(start, stop, detour1, detour2)

    self.add(rect)
    self.play(Transform(rect, star, path_func=bezier_path), run_time=3)
    self.pause()


    # manim -pqh discord.py Star_FollowBezier



class Flip_Plane(ThreeDScene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        axes = ThreeDAxes()
        plane1 = Surface(
            lambda u, v: np.array([
                u,
                v,
                v
            ]), u_range=[-1, 1], v_range=[-1, 1],
            checkerboard_colors=[RED_D, RED_E], resolution=(15, 32)
        )
        self.renderer.camera.light_source.move_to(3*IN) # changes the source of the light
        self.set_camera_orientation(phi=75 * DEGREES, theta=40 * DEGREES)
        self.add(plane1)
        self.begin_ambient_camera_rotation()
        self.wait(2)
        rotation_matrix = [[1, 0, 0],
                           [0, np.cos(np.pi/2), np.sin(np.pi/2)],
                           [0, -np.sin(np.pi/2), np.cos(np.pi/2)]]
        self.play(plane1.animate.apply_matrix(rotation_matrix))
        self.wait(10)
        self.stop_ambient_camera_rotation()



        # manim -pqh discord.py Flip_Plane



class colored_tex_with_indication(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        
        mathtex = MathTex(*r'f(x)  =  \ln(  x^2  +  3  x  +  2  )'.split('  '))
        colors =[RED,YELLOW,PURPLE,GREEN,BLUE]
        self.play(Write(mathtex))
        self.wait(1)                
        for i,piece in enumerate(mathtex):
            self.play(piece.animate.set_color(colors[i % 5]))
            self.add(MathTex("{}".format(i)).scale(0.4).next_to(piece, DOWN))
        self.wait(2)  


        # manim -pqh discord.py colored_tex_with_indication



class the_spider(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        graph = ImplicitFunction(
            lambda x, y:  np.cos(20*(np.arctan((x-1)/y)+np.arctan(y/(x+1)))),
            color=PURE_RED,
            min_depth=8, # <- Change this
        )

        #self.add(graph)


        self.play(
            Create(graph),
            run_time=5
        )
        self.wait(2)


        # manim -pqh discord.py the_spider

        # manim -sqk discord.py the_spider




L_SYSTEM = {
    "axiom": "F",
    "rules": {"F": "F[-F][+F]", "+": "+", "-": "-", "[": "[", "]": "]"},
    "iterations": 5,
    "length": 0.5,
    "degrees": 25,
    "inital_degrees": 90,
    "start": np.array([0, -1, 0]),
}


class Tree(Scene):
    def construct(self):
        (
            axiom,
            rules,
            iterations,
            length,
            degrees,
            initial_degrees,
            start,
        ) = L_SYSTEM.values()
        system = LSystem(axiom, rules, iterations)
        artist = LSystemArtist(system, length, degrees, initial_degrees, start)
        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        self.play(Create(artist.tree.set_color_by_gradient(PURE_RED,REANLEA_SLATE_BLUE)))
        self.wait(2)


class LSystemArtist:
    def __init__(self, system, length, degrees, initial_degrees, start):
        self.system = system
        self.commands = {
            "F": self.forward,
            "+": self.rotate_left,
            "-": self.rotate_right,
            "[": self.save,
            "]": self.restore,
        }
        self.theta = initial_degrees * PI / 180
        self.rotation = degrees * PI / 180
        self.length = length
        self.start = start
        self.positions = []
        self.rotations = []
        self.tree = VGroup()
        self.draw_system()

    def draw_system(self):
        for instruction in self.system.instructions:
            if instruction in self.commands.keys():
                self.commands[instruction]()

    def forward(self):
        end = (
            self.start
            + (self.length * np.cos(self.theta) * RIGHT)
            + (self.length * np.sin(self.theta) * UP)
        )
        new_line = Line(start=self.start, end=end)
        self.start = end
        self.tree.add(new_line)

    def rotate_left(self):
        self.theta += self.rotation

    def rotate_right(self):
        self.theta -= self.rotation

    def save(self):
        self.positions.append(self.start)
        self.rotations.append(self.theta)

    def restore(self):
        self.start = self.positions.pop()
        self.theta = self.rotations.pop()


class LSystem:
    def __init__(self, axiom, rules, iterations):
        self.rules = rules
        self.iterations = iterations
        self.instructions = axiom
        self.create_instructions()

    def create_instructions(self):
        new_string = ""
        for _ in range(self.iterations):
            new_string = ""
            for character in self.instructions:
                new_string += self.rules[character]
            self.instructions = new_string



        # manim -pqh discord.py Tree

        # manim -sqk discord.py Tree



class Vortices(MovingCameraScene):
    def construct(self):

        # WATER MARK 

        #water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        #self.add(water_mark)

        rigid = lambda pos: pos[0] * DOWN + pos[1] * RIGHT
        stream_rigid = StreamLines(rigid, x_range=[-1.5, 1.5, 0.2], y_range=[-1.5, 1.5, 0.2], colors=[REANLEA_WARM_BLUE, REANLEA_BLUE,PURE_RED,REANLEA_PURPLE,REANLEA_SLATE_BLUE,REANLEA_VIOLET_LIGHTER,REANLEA_BLUE_LAVENDER])
        stream_rigid.start_animation(warm_up=False)

        self.add(stream_rigid)
        self.wait()
        


        # manim -pqh discord.py Vortices

        # manim -sqk discord.py Vortices



class area_with_alpha(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        ax1 = NumberPlane(background_line_style={"stroke_color": WHITE,"stroke_width": 0.1,"stroke_opacity": 0.6})

        k=ValueTracker(-6)
        graph1 = ax1.plot(lambda x: 1/(x**k.get_value()), x_range=[0.5, 6.2]).set_color_by_gradient(YELLOW, ORANGE, RED)
        ts = MathTex(r"\alpha = ").set_color_by_gradient(YELLOW, ORANGE, RED).move_to(UP*3+RIGHT*3,aligned_edge=LEFT)
        number = DecimalNumber(0, color = RED if k.get_value() >=0 else ORANGE).next_to(ts,RIGHT)
        number.add_updater(lambda d: d.next_to(ts).set_value(k.get_value()))

        tt = MathTex(r"t\longmapsto \frac{1}{t^\alpha}").set_color_by_gradient(YELLOW, ORANGE, RED).to_edge(UP*3+RIGHT*2)
        doto = Dot(point=[1,0,0], radius=0.01)
        labelo = Tex("1", color=RED).next_to(doto, DOWN)
        area =  ax1.get_area(graph1, x_range = (1,6), color = ORANGE, opacity = 0.3)

        def sceneUpdater(dt):
            graph1.become(ax1.plot(lambda x: 1/(x**k.get_value()), x_range=[0.5, 6.2]).set_color_by_gradient(YELLOW, ORANGE, RED))
            area.become(ax1.get_area(graph1, x_range = (1,6), color = ORANGE, opacity = 0.3))
        self.add_updater(sceneUpdater)     

        self.play(Create(ax1))
        self.play(Write(labelo)) 
        self.play(Write(tt))
        self.play(Write(ts),Write(number))
        self.play(Create(graph1))
        self.play(Create(area))
        self.wait(0.25)
        self.add(Circle().shift(2*RIGHT+2*DOWN)) #mark the start
        self.play(k.animate.set_value(6), run_time=8,  rate_func= rate_functions.smooth)


        # manim -pqh discord.py area_with_alpha



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

class Arrow_3d_Area(ThreeDScene):
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


    # manim -pqh discord.py Arrow_3d_Area





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

class Arbitrary_ArcPolygon(Scene):
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


        #  manim -pqh discord.py Arbitrary_ArcPolygon



class sq_arange_trianle_rotate(Scene):
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


        # manim -pqh discord.py sq_arange_trianle_rotate



class multiplication_table(Scene):
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


            # manim -pqh discord.py multiplication_table



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


        # manim -pqh discord.py square_from_line



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


        # manim -pqh discord.py line2square



class MobsInFront_z_index(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        circ = Circle(radius=1,fill_color=PINK,fill_opacity=1,
            stroke_color=REANLEA_BLUE,stroke_opacity=1).set_z_index(2)
        edge = Dot(circ.get_center())
        anim = Flash(edge,color=PURE_RED,run_time=2,line_length=2)
        circ.add_updater(
            lambda l: l.become(
                Circle(arc_center=[0,0,1],radius=1,fill_color=REANLEA_BLUE,
                fill_opacity=1,stroke_color=PINK,stroke_opacity=1)
            )
        )

        self.add(circ)
        self.play(anim)
        self.wait()  


        # manim -pqh discord.py MobsInFront_z_index



class Combine_latex_square(Scene):
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


        # manim -pqh discord.py Combine_latex_square



class trace_square_and_fill(MovingCameraScene):
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


        # manim -pqh discord.py trace_square_and_fill



class smooth_linear_rate_function(Scene):
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


        # manim -pqh discord.py smooth_linear_rate_function



class Arc_Span(Scene):
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


        # manim -pqh discord.py Arc_Span




class Arrow_follow_path(Scene):
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




        # manim -pqh discord.py Arrow_follow_path




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


class Coin_transform_replace(Scene):
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



        # manim -pqh discord.py Coin_transform_replace


class brace_move_on_sum(Scene):
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


        # manim -pqh discord.py brace_move_on_sum



class VMobject(VMobject):
    def pfp(self, alpha):
        return self.point_from_proportion(alpha)


class Dot_move_on_curve_set_point_smoothly(Scene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)
        
        #c = Circle()
        c = VMobject(stroke_color=GREEN).set_points_smoothly([
                LEFT*2+UP*1.2, LEFT+UP*(-3), RIGHT*2+DOWN*1.7,
                DOWN*2+LEFT*2.5
            ])
        a = Dot(color = YELLOW)

        self.add(c, a)

        self.play(UpdateFromAlphaFunc(a, lambda x, alpha: x.move_to(c.pfp(alpha))), run_time = 3, rate_func= smooth)
        self.wait()


        # manim -pqh discord.py Dot_move_on_curve_set_point_smoothly



class analogue_clock_with_reading(Scene):
    def construct(self):
        clockface = VGroup()
        for t in range(12):
            clockface += Line(start = ORIGIN, end = 2*UP).rotate(-(t+1)/12*360*DEGREES, about_point=ORIGIN)
            lbl = MathTex(r"{:.0f}".format(t+1))
            clockface += lbl.move_to(clockface[-1].get_end()*1.5)

        self.play(Create(clockface))
        self.wait(2)

        # manim -pqh discord.py analogue_clock_with_reading



class Imagine_ClockFaces(Scene):
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


        # manim -pqh discord.py Imagine_ClockFaces



class spinning_arrow_on_tip_of_an_arrow(Scene):
    def construct(self):

        self.acc_time = 0
        self.vect1 = 0*LEFT
        self.vect2 = 0*LEFT
        self.vect1_ampl  = 2
        self.vect2_ampl  = self.vect1_ampl/2
        self.vect1_freq  = 1
        self.vect2_freq  = 10

        def sceneUpdater(dt):
            self.acc_time += dt                         # 0.5*dt will slow down the rotation/spinning rate
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

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        self.add(dyn_vect1_arrow, dyn_vect2_arrow)

        self.wait(30) 


        # manim -pqh discord.py spinning_arrow_on_tip_of_an_arrow




class sequence_of_n_by_nplus1(Scene):
    def construct(self):
        for i in range(6,-1 , -1):
            Text = MathTex(r"{:d} \over {:d}".format(i, i+1)).move_to(i*LEFT)
            self.add(Text)
            self.wait(1)

        self.wait(2)


        # manim -pqh discord.py sequence_of_n_by_nplus1


class MCQ_question(Scene):
    def construct(self):
        text = r"Which of the following are right?"

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


        # manim -pqh discord.py MCQ_question



class FrequencyShiftKying(Scene):
    def construct(self):
        
        bit_sequence = [1,0,1,1,0,1,1,1]
        self.bit_sequence = bit_sequence
        self.current_bit = 2
        self.stream_started = False

        self.establish_clock()
        self.show_debug_clock()
        self.enter_bit_stream(bit_sequence)
        self.setup_dot()
        self.setup_modulator_box()
        self.animate_bit_stream()
        self.setup_curve()

        dot = self.dot
        sine_curve_line = self.sine_curve_line
        sq = self.sq
        label = self.label
        arrow = self.arrow
        channel = self.channel
        bits = self.bits
        clock = self.clock
        time_group = self.time_group
        
        brace = Brace(sq, UP)
        brace_label = VGroup()
        info = Text("current bit:", font='/usr/share/fonts/truetype/baekmuk/dotum.ttf')
        current_bit_tag = Integer(self.current_bit).scale(2).set_color(BLACK)
        current_bit_tag.add_updater(lambda x : x.set_value(self.current_bit))
        brace_label.add(info)
        brace_label.add(current_bit_tag)
        brace_label.scale(0.5)
        brace_label.arrange(RIGHT)
        brace_label.add_updater(lambda x : x.next_to(brace, UP))
        colour_group = VGroup(brace, brace_label).add_updater(self.colour_control)

        title = Text("FSK - Frequency Shift Keying", font='/usr/share/fonts/truetype/baekmuk/dotum.ttf').scale(0.8)
        title.to_corner(UP+LEFT)

        bits.add_updater(self.align_bit_stream_top)
        self.play(Write(title))
        self.wait(0.5)
        enter = VGroup()
        enter.add(arrow, channel, label)
        self.play(Write(enter), FadeIn(sq), Write(time_group))
        self.add(dot, sine_curve_line, colour_group)
        self.wait(2)

        for j in range(len(bits)):
            self.stream_started = True
            the_bit = bits[0]
            the_bit.generate_target()
            the_bit.target.set_opacity(0)
            the_bit.target.shift(RIGHT*2)
            self.play(MoveToTarget(the_bit), run_time = 1.4)
            bits.remove(the_bit)
            self.wait(1.6)

        self.wait(1)

    def establish_clock(self):
        self.clock = 0
        self.increment = 0

    def enter_bit_stream(self, bit_sequence):
        self.bit_stream = bit_sequence
        self.bits = VGroup()
        for i in range(len(bit_sequence)):
            self.bits.add(Text(str(bit_sequence[i])))
        self.current_index = 0
        

    def align_bit_stream_top(self, bits):
        top = bits.get_top()
        sq = self.sq
        bits.shift(UP*(sq.get_center()[1] - top[1]))

    def animate_bit_stream(self):
        self.bits.arrange(RIGHT)
        self.play(Write(self.bits))
        self.wait(1)
        self.bits.generate_target()
        self.bits.target.scale(0.5)
        self.bits.target.arrange(DOWN)
        self.bits.target.to_edge(LEFT)
        for i in range(len(self.bits)):
            if self.bit_sequence[i] == 0:
                self.bits.target[i].set_color(BLUE_D)
            else:
                self.bits.target[i].set_color(RED_D)
        top = self.bits.target.get_top()
        sq = self.sq
        self.bits.target.shift(UP*(sq.get_center()[1] - top[1]))
        self.play(MoveToTarget(self.bits), run_time = 3)

    def oscillate_dot(self, the_dot, dt):
            self.clock += dt
            y_shift = 0.7*np.sin(self.clock*(1.5 + 2.5*self.current_amp(self.clock))*np.pi)
            the_dot.move_to(self.dot_origin + [0,y_shift,0])

    def current_amp(self, t):
        if(t < 2):
            return 0
        else:
            bit_index = int((t-2)/3)
            if(bit_index in range(len(self.bit_sequence))):
                self.current_bit = self.bit_sequence[bit_index]
            else:
                return 0

            if(self.current_bit):
                return 1
            else:
                return 0

    def setup_dot(self):
        self.dot = Dot()
        self.dot.shift(UP*0+LEFT*4)
        self.dot_origin = self.dot.get_center()
        self.dot.add_updater(self.oscillate_dot)

    def setup_modulator_box(self):
        self.sq = Square()
        self.sq.move_to(self.dot.get_center())
        self.label = Text("Modulator",font='/usr/share/fonts/truetype/baekmuk/dotum.ttf').scale(0.5)
        self.label.move_to(self.sq.get_bottom()).shift(DOWN*0.35)
        self.arrow = Arrow(self.sq.get_right()-[0.25,0,0], self.sq.get_right() + [8,0,0])
        self.channel = Text("to channel", font='/usr/share/fonts/truetype/baekmuk/dotum.ttf')
        self.channel.scale(0.5).move_to(self.arrow.get_right() + [0,-1.1,0])
    
    def move_signal(self, path):
        path.move_to(self.dot_origin + [self.clock/4, 0, 0])

    def get_curve(self):
        
        last_line = self.curve[-1]
        x = self.dot.get_center()[0]
        y = self.dot.get_center()[1]
        new_line = Line(last_line.get_end(),np.array([x,y,0]), color=YELLOW_D)
        if(self.stream_started):
            self.curve.add(new_line)

        return self.curve

    def setup_curve(self):
        self.curve = VGroup()
        self.curve.add(Line(self.dot.get_center(),self.dot.get_center()))
        self.sine_curve_line = always_redraw(self.get_curve)
        self.sine_curve_line.add_updater(self.move_signal)

    def show_debug_clock(self):
        box = Square().to_corner(corner = UP + RIGHT)
        label = Text("time (s)", font='/usr/share/fonts/truetype/baekmuk/dotum.ttf')
        label.move_to(box.get_bottom() - [0,0.4,0]).scale(0.5)
        watch = DecimalNumber(0, show_ellipsis=False, num_decimal_places=2, include_sign=False)
        watch.move_to(box)
        watch.add_updater(lambda x : x.set_value(self.clock))
        self.time_group = VGroup(box, watch, label).scale(0.8)

    def colour_control(self, thing):
        if self.current_bit == 1:
            thing.set_color(RED_D)
        elif self.current_bit == 0:
            thing.set_color(BLUE_D)
    


        # manim -pqh discord.py FrequencyShiftKying




class Dragon(MovingCameraScene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(6).set_opacity(0.15).set_z_index(-1)
        

        dragon_curve = VMobject(stroke_color=[REANLEA_PURPLE,REANLEA_SLATE_BLUE,REANLEA_WELDON_BLUE])
        dragon_curve_points = [LEFT, RIGHT]
        dragon_curve.set_points_as_corners(dragon_curve_points)
        dragon_curve.corners = dragon_curve_points
        self.add(dragon_curve)
        dragon_curve.add_updater(
            lambda mobject: mobject.set_style(stroke_width=self.camera.frame.width / 5),              #stroke_width=self.camera.frame.width / 10
        )
        dragon_curve.update()
        self.wait()

        def rotate_half_points(points, alpha):
            static_part = points[:len(points)//2]
            about_point = points[len(points)//2]
            mat = rotation_matrix(-PI/2 * alpha, OUT)
            rotated_part = [
                np.dot((point - about_point), mat.T) + about_point
                for point in reversed(static_part)
            ]
            return static_part + [about_point] + rotated_part

        def rotate_half_curve(mobject, alpha):
            corners = mobject.corners
            new_corners = rotate_half_points(corners, alpha)
            mobject.set_points_as_corners(new_corners)
            return mobject

        for it in range(15):
            rotated_curve = VMobject().set_points_as_corners(rotate_half_points(dragon_curve.corners, 1))
            self.play(
                UpdateFromAlphaFunc(dragon_curve, rotate_half_curve),
                self.camera.auto_zoom(rotated_curve, margin=1),
            )
            current_corners = rotate_half_points(dragon_curve.corners, 1)
            current_corners = current_corners + current_corners[-1::-1]
            dragon_curve.set_points_as_corners(current_corners)
            dragon_curve.corners = current_corners

        self.add(water_mark.shift(350*RIGHT+35*UP))
        self.wait()



        # manim -pqh discord.py Dragon

        # manim -sqk discord.py Dragon





class KochCurveEx(Scene):
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


            #  manim -pqh discord.py KochCurveEx




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
        self.play(Uncreate(Cycloid, reverse_rate_func=lambda t: linear(1 - t), run_time=4.5 + corr))
            
        RC.clear_updaters()
        self.wait()


        # manim -pqh discord.py Cycloid




class VonKoch(VMobject):
    def __init__(self, number_of_iterations: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.lines = Line(5*RIGHT, 5*LEFT, **kwargs)
        self.add(self.lines)
        for _ in range(number_of_iterations):
            for line in self:
                self.add(*self.get_iteration(line, **kwargs))
                self.remove(line)


    def get_iteration(self, side: VMobject, **kwargs) -> tuple:
        start, end = side.get_start(), side.get_end()
        first_point, second_point = 2/3 * start + 1/3 * end, 1/3 * start + 2/3 * end
        new_line_1 = Line(start, first_point, **kwargs)
        new_line_2 = Line(first_point, second_point, **kwargs).rotate(-PI/3, about_point=first_point)
        new_line_3 = Line(first_point, second_point, **kwargs).rotate(PI/3, about_point=second_point)
        new_line_4 = Line(second_point, end, **kwargs)
        return VMobject().add(new_line_1, new_line_2, new_line_3, new_line_4)



class VonKochFractal(Scene):
    def construct(self):
        vonkoch = VonKoch(0, stroke_width=6.5).shift(DOWN).set_color_by_gradient(REANLEA_AQUA,REANLEA_PURPLE)
        self.add(vonkoch)
        for k in range(8):
            self.play(Transform(vonkoch, VonKoch(k, stroke_width=0.5).shift(DOWN)))


    # manim -pqh discord.py VonKochFractal



class testingSine(Scene):
    def construct(self):
        
        bit_sequence = [1,0,1,1,0,1,1,1]
        self.bit_sequence = bit_sequence
        self.current_bit = 2
        self.stream_started = False

        self.establish_clock()
        self.show_debug_clock()
        self.enter_bit_stream(bit_sequence)
        self.setup_dot()
        self.setup_modulator_box()
        self.animate_bit_stream()
        self.setup_curve()

        dot = self.dot
        sine_curve_line = self.sine_curve_line
        sq = self.sq
        label = self.label
        arrow = self.arrow
        channel = self.channel
        bits = self.bits
        clock = self.clock
        time_group = self.time_group
        
        brace = Brace(sq, UP)
        brace_label = VGroup()
        info = Text("current bit:", font='/usr/share/fonts/truetype/baekmuk/dotum.ttf')
        current_bit_tag = Integer(self.current_bit).scale(2).set_color(BLACK)
        current_bit_tag.add_updater(lambda x : x.set_value(self.current_bit))
        brace_label.add(info)
        brace_label.add(current_bit_tag)
        brace_label.scale(0.5)
        brace_label.arrange(RIGHT)
        brace_label.add_updater(lambda x : x.next_to(brace, UP))
        colour_group = VGroup(brace, brace_label).add_updater(self.colour_control)

        title = Text("FSK - Frequency Shift Keying", font='/usr/share/fonts/truetype/baekmuk/dotum.ttf').scale(0.8)
        title.to_corner(UP+LEFT)

        bits.add_updater(self.align_bit_stream_top)
        self.play(Write(title))
        self.wait(0.5)
        enter = VGroup()
        enter.add(arrow, channel, label)
        self.play(Write(enter), FadeIn(sq), Write(time_group))
        self.add(dot, sine_curve_line, colour_group)
        self.wait(2)

        for j in range(len(bits)):
            self.stream_started = True
            the_bit = bits[0]
            the_bit.generate_target()
            the_bit.target.set_opacity(0)
            the_bit.target.shift(RIGHT*2)
            self.play(MoveToTarget(the_bit), run_time = 1.4)
            bits.remove(the_bit)
            self.wait(1.6)

        self.wait(1)

    def establish_clock(self):
        self.clock = 0
        self.increment = 0

    def enter_bit_stream(self, bit_sequence):
        self.bit_stream = bit_sequence
        self.bits = VGroup()
        for i in range(len(bit_sequence)):
            self.bits.add(Text(str(bit_sequence[i])))
        self.current_index = 0
        

    def align_bit_stream_top(self, bits):
        top = bits.get_top()
        sq = self.sq
        bits.shift(UP*(sq.get_center()[1] - top[1]))

    def animate_bit_stream(self):
        self.bits.arrange(RIGHT)
        self.play(Write(self.bits))
        self.wait(1)
        self.bits.generate_target()
        self.bits.target.scale(0.5)
        self.bits.target.arrange(DOWN)
        self.bits.target.to_edge(LEFT)
        for i in range(len(self.bits)):
            if self.bit_sequence[i] == 0:
                self.bits.target[i].set_color(BLUE_D)
            else:
                self.bits.target[i].set_color(RED_D)
        top = self.bits.target.get_top()
        sq = self.sq
        self.bits.target.shift(UP*(sq.get_center()[1] - top[1]))
        self.play(MoveToTarget(self.bits), run_time = 3)

    def oscillate_dot(self, the_dot, dt):
            self.clock += dt
            y_shift = 0.7*np.sin(self.clock*(1.5 + 2.5*self.current_amp(self.clock))*np.pi)
            the_dot.move_to(self.dot_origin + [0,y_shift,0])

    def current_amp(self, t):
        if(t < 2):
            return 0
        else:
            bit_index = int((t-2)/3)
            if(bit_index in range(len(self.bit_sequence))):
                self.current_bit = self.bit_sequence[bit_index]
            else:
                return 0

            if(self.current_bit):
                return 1
            else:
                return 0

    def setup_dot(self):
        self.dot = Dot()
        self.dot.shift(UP*0+LEFT*4)
        self.dot_origin = self.dot.get_center()
        self.dot.add_updater(self.oscillate_dot)

    def setup_modulator_box(self):
        self.sq = Square()
        self.sq.move_to(self.dot.get_center())
        self.label = Text("Modulator",font='/usr/share/fonts/truetype/baekmuk/dotum.ttf').scale(0.5)
        self.label.move_to(self.sq.get_bottom()).shift(DOWN*0.35)
        self.arrow = Arrow(self.sq.get_right()-[0.25,0,0], self.sq.get_right() + [8,0,0])
        self.channel = Text("to channel", font='/usr/share/fonts/truetype/baekmuk/dotum.ttf')
        self.channel.scale(0.5).move_to(self.arrow.get_right() + [0,-1.1,0])
    
    def move_signal(self, path):
        path.move_to(self.dot_origin + [self.clock/4, 0, 0])

    def get_curve(self):
        
        last_line = self.curve[-1]
        x = self.dot.get_center()[0]
        y = self.dot.get_center()[1]
        new_line = Line(last_line.get_end(),np.array([x,y,0]), color=YELLOW_D)
        if(self.stream_started):
            self.curve.add(new_line)

        return self.curve

    def setup_curve(self):
        self.curve = VGroup()
        self.curve.add(Line(self.dot.get_center(),self.dot.get_center()))
        self.sine_curve_line = always_redraw(self.get_curve)
        self.sine_curve_line.add_updater(self.move_signal)

    def show_debug_clock(self):
        box = Square().to_corner(corner = UP + RIGHT)
        label = Text("time (s)", font='/usr/share/fonts/truetype/baekmuk/dotum.ttf')
        label.move_to(box.get_bottom() - [0,0.4,0]).scale(0.5)
        watch = DecimalNumber(0, show_ellipsis=False, num_decimal_places=2, include_sign=False)
        watch.move_to(box)
        watch.add_updater(lambda x : x.set_value(self.clock))
        self.time_group = VGroup(box, watch, label).scale(0.8)

    def colour_control(self, thing):
        if self.current_bit == 1:
            thing.set_color(RED_D)
        elif self.current_bit == 0:
            thing.set_color(BLUE_D)
    


        # manim -pqh discord.py testingSine



#config.disable_caching=True

from numba import jit
from numba import njit

class MandelbrotAndJuliaScene(Scene):
    def construct(self):
        jited_MandelbrotSet = njit()(MandelbrotSet)
        jited_JuliaSet = njit()(JuliatSet)
        mandelbrot_set = ImageMobject(jited_MandelbrotSet(
            max_steps=50,
            resolution=1080,
        )).set_z_index(4)
        mandelbrot_set_complex_plane = ComplexPlane(
            x_range = np.array([-2, 0.5]),
            y_range = np.array([-1.15, 1.15]),
            x_length = mandelbrot_set.height,
            y_length = mandelbrot_set.width 
        )
        old_height = mandelbrot_set.height
        mandelbrot_set.height = 3.5
        rectangle = BackgroundRectangle(mandelbrot_set, color=WHITE, stroke_width=2, stroke_opacity=1, fill_opacity=0, buff=0).set_z_index(5)
        Group(mandelbrot_set, rectangle).to_corner(UR, buff=0.2)

        mandelbrot_set_complex_plane.move_to(mandelbrot_set).scale(3.5/old_height)


        c = ComplexValueTracker(array_of_complex_constants[0][0])
        max_steps = ValueTracker(20)

        dot = always_redraw(lambda: Dot(mandelbrot_set_complex_plane.c2p(c.get_value().real, c.get_value().imag), radius=DEFAULT_SMALL_DOT_RADIUS, color=PURE_BLUE).set_z_index(6))
        label = always_redraw(lambda: MathTex('c', stroke_color=BLACK, stroke_width=0.75).scale(1.25).next_to(dot, UL, buff=0.1).set_z_index(6))

        eq = MathTex('z \\mapsto z^2 + c').to_corner(UL, buff=0.27).set_z_index(4)
        rectangle_eq = SurroundingRectangle(eq, color=WHITE, buff=0.25).set_z_index(5)

        constant = MathTex('c =').to_edge(buff=0.22).to_edge(DOWN, buff=0.65).set_z_index(10)
        for complex_constant in array_of_complex_constants:
            complex_constant[1].next_to(constant, RIGHT)
        constant_value = array_of_complex_constants[0][1].set_z_index(10)
        background = always_redraw(lambda: BackgroundRectangle(Group(constant, constant_value), color=BLACK))
        surrounding = always_redraw(lambda: SurroundingRectangle(Group(constant, constant_value), color=WHITE, buff=0.2)).set_z_index(11)


        julia_set = ImageMobject(jited_JuliaSet(
            c.get_value(),
            max_steps=int(max_steps.get_value()),
            resolution=1440,
            x_range=np.array([-1.75, 1.75]),
            y_range=np.array([-1.5, 1.5])
        )).set_z_index(3)
        julia_set.height = config.frame_height

        self.play(FadeIn(Group(mandelbrot_set, rectangle, eq, rectangle_eq, julia_set, dot, label, constant, constant_value, background, surrounding)))
        self.wait(1.75)

        def update_julia_set(julia_set):
            julia_set.become(ImageMobject(jited_JuliaSet(
                c.get_value(),
                max_steps=int(max_steps.get_value()),
                resolution=1440,
                x_range=np.array([-1.75, 1.75]),
                y_range=np.array([-1.5, 1.5])
            )).set_z_index(3))
            julia_set.height = config.frame_height

        julia_set.add_updater(update_julia_set)

        for k in range(1, len(array_of_complex_constants)):
            if k == 4:
                self.play(
                    c.animate.set_value(array_of_complex_constants[4][0]),
                    max_steps.animate.set_value(35),
                    Transform(array_of_complex_constants[0][1], array_of_complex_constants[4][1]),
                    run_time=3
                    )
                self.wait(1.75)
            if k == 6:
                self.play(
                    c.animate.set_value(array_of_complex_constants[6][0]),
                    max_steps.animate.set_value(500),
                    Transform(array_of_complex_constants[0][1], array_of_complex_constants[6][1]),
                    run_time=3
                    )
                self.wait(1.75)
            if k == 11:
                self.play(
                    c.animate.set_value(array_of_complex_constants[11][0]),
                    max_steps.animate.set_value(70),
                    Transform(array_of_complex_constants[0][1], array_of_complex_constants[11][1]),
                    run_time=3
                    )
                self.wait(1.75)
            else:
                self.play(
                    c.animate.set_value(array_of_complex_constants[k][0]),
                    max_steps.animate.set_value(20),
                    Transform(array_of_complex_constants[k-1][1], array_of_complex_constants[k][1]),
                    run_time=3
                    )
                self.wait(1.75)




array_of_complex_constants = [
    [complex(np.pi/17, np.pi/17), MathTex('-\\frac{\\pi}{17} + \\frac{\\pi}{17}i')],
    [complex(np.pi/17, 42/1975), MathTex("-\\frac{\\pi}{17} + \\frac{42}{1975}i")],
    [complex(-1.975, 0.42), MathTex("-1.975 + 0.42i")],
    [complex(np.pi/17, np.exp(-42)), MathTex("\\frac{\\pi}{17} + e^{-42}i")],
    [complex(0.3, 0.03), MathTex("0.3 + 0.03i")],
    [complex(0.42, 1), MathTex("0.42 + i")],
    [complex(-1/15, 2/3), MathTex("-\\frac{1}{15} + \\frac{2}{3}i")],
    [complex(-1.877, 0.115), MathTex("-\\frac{\\pi}{17} + \\frac{\\pi}{17}i")],
    [complex(0, 0), MathTex("0")],
    [complex(1/42, np.pi/17), MathTex("\\frac{1}{42} + \\frac{\\pi}{17}i")],
    [complex(-1.683, 0.6977), MathTex("-1.683 + 0.6977i")],
    [complex(0.07, 0.7), MathTex("0.07 + 0.7i")]
]




def MandelbrotSet(max_steps: int = 50, resolution: int = 1080, x_range: np.ndarray = np.array([-2, 0.5]), y_range: np.ndarray = np.array([-1.15, 1.15])) -> np.ndarray:
        x_m, x_M, y_m, y_M = x_range[0], x_range[1], y_range[0], y_range[1]
        resolution_tuple = resolution, int(np.abs(x_M - x_m)/np.abs(y_M - y_m)*resolution)
        dx, dy = np.abs(x_M - x_m)/resolution_tuple[1], np.abs(y_M - y_m)/resolution_tuple[0]
        array = np.zeros(resolution_tuple, dtype='uint8')
        for n in range(len(array)):
            for p in range(len(array[0])):
                c = complex(x_m + p*dx, y_M - n*dy)
                z = c
                i = 0
                while i < max_steps and np.abs(z) <= 2:
                    z = z**2 + c
                    i += 1
                if i == max_steps:
                    array[n, p] = 0
                else:
                    array[n, p] = i*(255/max_steps)
        return array



def JuliatSet(complex_constant: complex = complex(0, 0), max_steps: int = 50, resolution: int = 1080, x_range: np.ndarray = np.array([-2, 0.5]), y_range: np.ndarray = np.array([-1.15, 1.15])) -> np.ndarray:
    x_m, x_M, y_m, y_M = x_range[0], x_range[1], y_range[0], y_range[1]
    resolution_tuple = resolution, int(np.abs(x_M - x_m)/np.abs(y_M - y_m)*resolution)
    dx, dy = np.abs(x_M - x_m)/resolution_tuple[1], np.abs(y_M - y_m)/resolution_tuple[0]
    array = np.zeros(resolution_tuple, dtype='uint8')
    for n in range(len(array)):
        for p in range(len(array[0])):
            c = complex(x_m + p*dx, y_M - n*dy)
            z = c
            i = 0
            while i < max_steps and np.abs(z) <= 2:
                z = z**2 + complex_constant
                i += 1
            if i == max_steps:
                array[n, p] = 0
            else:
                array[n, p] = i*(255/max_steps)
    return array


    # manim -pqh test2.py MandelbrotAndJuliaScene




###################################################################################################################

                        ##########  https://slama.dev/manim/3d-and-the-other-graphs/  #########


class Rotation3DExample(ThreeDScene):
    def construct(self):
        cube = Cube(side_length=3, fill_opacity=1).set_color_by_gradient(REANLEA_BLUE_LAVENDER,REANLEA_MAGENTA,REANLEA_BLUE)

        self.begin_ambient_camera_rotation(rate=0.3)

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        self.play(Write(cube), run_time=2)

        self.wait(3)

        self.play(Unwrite(cube), run_time=2)


        # manim -pqh discord.py Rotation3DExample



class Basic3DExample(ThreeDScene):
    def construct(self):
        cube = Cube(side_length=3, fill_opacity=0.5)

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        self.play(FadeIn(cube))

        for axis in [RIGHT, UP, OUT]:
            self.play(Rotate(cube, PI / 2, about_point=ORIGIN, axis=axis))

        self.play(FadeOut(cube))


        # manim -pqh discord.py Basic3DExample



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




class BinomialDistributionSimulation(Scene):
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



        # manim -pqh discord.py BinomialDistributionSimulation





class Plot_Stat(Scene):
    def construct(self):
        text = Tex(r"number of throws until all in [1,2,3] appeared once").to_edge(UP)
        self.add(text)
        stats = [.001 for i in range(3,31)]
        names = ["${}$".format(i) for i in range(3,30)]+["$30+$"]

        stats = [0 for i in range(3,31)]      
        
        num = DecimalNumber(0, unit=r"\text{~throws}", num_decimal_places=0).to_edge(DOWN)
        self.add(num)

        throws=[np.random.randint(low=1,high=4) for i in range(30)]
        try:
            first = max(throws.index(1),throws.index(2),throws.index(3))+1
        except:
            first = 30
        stats[first-3] += 1

        barchart = BarChart(stats, bar_names=names)
        self.add(barchart)

        num.increment_value(1)

        self.wait(1)

        for k in range(99):
            throws=[np.random.randint(low=1,high=4) for i in range(30)]
            try:
                first = max(throws.index(1),throws.index(2),throws.index(3))+1
            except:
                first = 30
            stats[first-3] += 1

            self.remove(barchart)
            barchart = BarChart(stats, bar_names=names)
            self.add(barchart)

            num.increment_value(1)

            self.wait(1/10)

        barlabels = barchart.get_bar_labels(font_size=24)
        self.play(Create(barlabels),run_time=4)
 
        self.wait(10)


        # manim -pqh discord.py Plot_Stat



class Tikz_Node_Coloring(Scene):
    def construct(self):
        template = TexTemplate()
        template.add_to_preamble(r"\usepackage{tikz}")
        template.add_to_preamble(r"\usetikzlibrary{arrows}")
        
        
        tex = Tex(
            r"""[->,>=stealth',node distance=3cm]
            \node[circle,draw=black,font=\sffamily\Large\bfseries] (1) {\textcolor{black}{a}};
            \node[circle,draw=black,text=black,font=\sffamily\Large\bfseries] (2) [right of=1] {b};
            \path[every node/.style={color=green,font=\sffamily\small}] (2) edge[bend left=90] node [left] {} (1);""",
            tex_template=template,
            tex_environment="tikzpicture",
            fill_opacity=0,
            stroke_width=2,
            stroke_opacity=1,
            font_size=30,
        )
        
        self.play(Create(tex), run_time = 5)
        
        for i, obj in enumerate(tex):
            for j, mobj in enumerate(obj):
                color = random_color()
                
                self.add(MathTex(r"{}".format(j)).next_to(mobj,UP).set_color(color))
                mobj.set_color(color)
                self.wait()
        self.wait()  

        # manim -pqh discord.py Tikz_Node_Coloring

###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix"