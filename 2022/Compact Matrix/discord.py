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
from scipy.stats import norm, gamma
from scipy.optimize import fsolve
import random


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
        c1=Circle()    
        a = Dot(color = YELLOW)

        self.add(c1, a)

        self.play(UpdateFromAlphaFunc(a, lambda x, alpha: x.move_to(c1.pfp(alpha))), run_time = 3, rate_func= smooth)
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

'''from numba import jit
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


'''

###################################################################################################################

                        ##########  https://slama.dev/manim/3d-and-the-other-graphs/  #########


class Rotation3DExample(ThreeDScene):
    def construct(self):
        cube = Cube(side_length=3, fill_opacity=1).set_color_by_gradient(REANLEA_BLUE_LAVENDER,REANLEA_MAGENTA,REANLEA_BLUE)

        a=MathTex(r"\mathbb{R}^{3}").shift(2*UR)

        self.begin_ambient_camera_rotation(rate=0.3)

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        self.play(Write(cube), run_time=2)
        self.play(
            Write(a)
        )

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






class ArbitraryShape(Scene):
    def construct(self):
        grid = NumberPlane(axis_config={"include_tip":True},
            background_line_style={
                "stroke_color": BLUE,
                "stroke_width": 0,
                "stroke_opacity": 0
            }
        ).set_opacity(0)
        graph1 = grid.plot_polar_graph(lambda theta: 1, [0, 2 * PI], color=YELLOW)
        r = lambda theta: 1 + 0.2 * np.sin(4*theta) + 0.01*theta*theta*(theta-2*np.pi)*(theta-2*np.pi)
        graph2 = grid.plot_polar_graph(r, [0, 2 * PI], color=YELLOW)
        self.add(graph1)

        dots = VGroup(*[
            Dot(grid.polar_to_point(1, i*PI/5))
            for i in range(1,11)
        ])

        self.play(*map(GrowFromCenter,dots))

        self.play(
            Transform(graph1, graph2),
            *[
                dot.animate.move_to(grid.polar_to_point(r(i*PI/5), i*PI/5))
                for i,dot in zip(range(1,11), dots)
            ],
            run_time=6
        )
        self.wait()


        # manim -pqh discord.py ArbitraryShape



class Fibonacci(Scene):
    def construct(self):
        fibos = [1,1] 
        for i in range(20):
            num1 = DecimalNumber(fibos[-2], num_decimal_places=0)
            num2 = DecimalNumber(fibos[-1], num_decimal_places=0)
            result = DecimalNumber(fibos[-2]+fibos[-1], num_decimal_places=0)
            result.to_edge(RIGHT)
            num2.next_to(result,LEFT)
            num1.next_to(num2,LEFT)
            question = MathTex(r"?").move_to(result)
            resSq = SurroundingRectangle(result).set_color(BLUE)
            num1Sq = SurroundingRectangle(num1).set_color(YELLOW)                 
            num2Sq = SurroundingRectangle(num2).set_color(YELLOW)
            self.add(num1,num1Sq,num2,num2Sq,question,resSq)

            older = VGroup()
            if len(fibos)>2:
                for j in range(len(fibos)-2):
                    older.add(
                        MathTex(r"{}".format(fibos[-3-j]))
                    )
                older.arrange(direction=LEFT)
                older.next_to(num1Sq,LEFT)
            self.add(older)
            fibos.append(result.get_value())
            self.wait(1)
            self.play(ReplacementTransform(question,result))
            self.wait(1)
            self.remove(older,num1,num1Sq,num2,num2Sq,result,resSq)


            # manim -pqh discord.py Fibonacci



class Rounded_Corner_Animation(Scene):
    def construct(self):
        radius = ValueTracker(0)
        plane = NumberPlane()
        self.add(plane)
        polygram1 = Polygram([[0, 0, 0], [3, 2, 0], [0, 3, 0], [2, -1, 0]]
                             ).set_fill(WHITE, opacity=0.8) 
        origpoly = polygram1.copy()

        self.play(Create(polygram1))
                   
        def polyUpdater(obj):
            obj.become(origpoly.copy().round_corners(radius=radius.get_value()))           
        polygram1.add_updater(polyUpdater)    

        self.play(radius.animate.set_value(0.5), runtime=3)



        # manim -pqh discord.py Rounded_Corner_Animation




class test_y(Scene):
    def construct(self):

        theta_tracker = ValueTracker(.01)

        vect_1=Arrow(start=LEFT,end=RIGHT,max_tip_length_to_length_ratio=0.125, buff=0).set_color(PURE_RED)
        vect_1_moving=Arrow(start=LEFT,end=RIGHT,max_tip_length_to_length_ratio=0.125, buff=0).set_color(PURE_RED).set_opacity(0.85)
        vect_1_ref=vect_1_moving.copy()
        vect_1_moving.rotate(
            theta_tracker.get_value() * DEGREES, about_point=vect_1_moving.get_start()
        )

        ang=Angle(vect_1, vect_1_moving, radius=1, other_angle=False).set_stroke(color=PURE_GREEN, width=3).set_z_index(-1)

        ang_theta=DecimalNumber(unit="^o").scale(.5).move_to(RIGHT+UP)



        vect_1_moving.add_updater(
            lambda x: x.become(vect_1_ref.copy()).rotate(
                theta_tracker.get_value() * DEGREES, about_point=vect_1_moving.get_start()
            )
        )
        

        ang.add_updater(
            lambda x: x.become(Angle(vect_1, vect_1_moving, radius=1, other_angle=False).set_stroke(color=PURE_GREEN, width=3).set_z_index(-1))
        )


        ang_theta.add_updater( lambda x: x.set_value(theta_tracker.get_value()).move_to(RIGHT+UP))
        






        self.play(
            Create(vect_1),
            Write(vect_1_moving),
        )
        self.play(theta_tracker.animate.set_value(40))
        self.wait()
        self.play(
            Create(ang)
        )
        self.play(
            Write(ang_theta)
        )
        self.wait(2)
    
        self.play(
            theta_tracker.animate.increment_value(80),
            ang_theta.animate.set_color(RED),
            ang.animate.set_stroke(color=RED, width=3),
            run_time=2
        )
        
        self.wait(2)


        # manim -pqh discord.py test_y




class doppler2(Scene):
    def construct(self):
        d = Dot()
        self.time = 0
        circle_group = VGroup()
        def update_time(dt):
            self.time += dt
        self.add_updater(update_time)
        direction = [1, 2]
        def make_circle(d,dt):

            if self.time > 0.15:
                self.time = 0
                c= Circle().scale(0.5)
                c.move_to(d.get_center())
                circle_group.add(c)
            d.shift((direction[0]*RIGHT+direction[1]*UP)*dt)
        # #dt = 1/self.camera.frame_rate
        d.add_updater(make_circle)
        
        def expand_circle(mob, dt):
            for i in range(len(mob)):
                mob[i].add_updater(lambda x: x.scale_to_fit_width(x.width+0.002, about_point = x.get_center()).set_stroke(opacity = 1-x.width/14))
        self.add(circle_group)
        circle_group.add_updater(expand_circle)
        
        self.add(d)
       # self.play(d.animate.move_to(2*RIGHT))
        self.wait(5)



        # manim -pqh discord.py doppler2



class doppler1(Scene):
    def construct(self):
        d = Dot()
        self.time = 0
        circle_group = VGroup()
        def update_time(dt):
            self.time += dt
        self.add_updater(update_time)
        direction = [RIGHT, 2*UP]
        def make_circle(d,dt):

            if self.time > 0.15:
                self.time = 0
                c= Circle().scale(0.5)
                c.move_to(d.get_center())
                circle_group.add(c)
            d.shift(direction[0]*dt)
        # #dt = 1/self.camera.frame_rate
        d.add_updater(make_circle)
        
        def expand_circle(mob, dt):
            for i in range(len(mob)):
                mob[i].add_updater(lambda x: x.scale_to_fit_width(x.width+0.002, about_point = x.get_center()).set_stroke(opacity = 1-x.width/14))
        self.add(circle_group)
        circle_group.add_updater(expand_circle)
        
        self.add(d)
       # self.play(d.animate.move_to(2*RIGHT))
        self.wait(5)


        # manim -pqh discord.py doppler1



class unfold2(Scene):
    def construct(self):
        t = ValueTracker(1)

        def getAnnularSector():
            newInnerRad = 1.8 * t.get_value()**3
            arcLength = 2*np.pi*(1.8/newInnerRad) 
            start = (2*np.pi - arcLength)/2 + 90*DEGREES
            return AnnularSector(inner_radius=newInnerRad, outer_radius=newInnerRad+0.2, start_angle=start, angle=arcLength).move_to(2*DOWN, aligned_edge=DOWN).set_color(REANLEA_BLUE_LAVENDER)
        donuts = always_redraw(getAnnularSector)

        self.add(donuts)

        self.wait()

        self.play(t.animate.set_value(10), run_time=5)



        # manim -pqh discord.py unfold2





class PythagoreanIdentity(Scene):
    def construct(self):
        title = Text("The Pythagorean Identity").shift(UP)
        #name = Text("By R E A N L E A")
        with RegisterFont("Montserrat") as fonts:
            name=Text("by    R E A N L E A", font=fonts[0])
            name.set_color_by_gradient(REANLEA_TXT_COL)
        #name.move_to(.75*LEFT+ 0.2*UP).rotate(20*DEGREES)
        credit = Text("Inspired by Burkard's (aka Mathologer) Twisted Squares video", font_size=26).next_to(name, DOWN)
        banner = ManimBanner().next_to(credit, DOWN).scale(0.3)
        self.play(Write(title), run_time=0.8)
        self.play(Write(name), run_time=0.8)
        self.play(Write(credit), banner.create(), runt_time=0.8)
        self.play(banner.expand())
        self.play(Unwrite(title), Unwrite(credit, reverse=False), Unwrite(banner), Unwrite(name[0:2]), run_time=0.8)
        self.play(name[2:].animate.scale(0.5).move_to(5 * RIGHT + 3.5 * DOWN))

        triangle = Polygon(2 * LEFT, 2 * RIGHT, 2 * LEFT + 3 * UP, stroke_color=WHITE, fill_color=REANLEA_BLUE, fill_opacity=1
        , stroke_width=DEFAULT_STROKE_WIDTH/2).scale(0.5).move_to(ORIGIN)
        self.play(DrawBorderThenFill(triangle), run_time=0.8)
        a = MathTex("a").next_to(triangle, direction=LEFT, buff=0.1)
        b = MathTex("b").next_to(triangle, direction=DOWN, buff=0.08)
        c = MathTex("c").next_to(Line(triangle.get_corner(DR), triangle.get_corner(UL)).get_center(), direction=UR, buff=0.05)
        labels = VGroup(a, b, c)
        self.play(Write(labels))

        square = Square(side_length=0.2, stroke_width=DEFAULT_STROKE_WIDTH/2).next_to(triangle.get_corner(DL), UR, buff=0)
        self.play(FadeIn(square), run_time=0.6)
        self.play(FadeOut(square), run_time=0.4)

        square = Square(side_length=3+4, fill_color=REANLEA_BLUE_LAVENDER, fill_opacity=1).scale(0.5)
        self.play(FadeOut(triangle), Unwrite(labels))
        self.play(Write(square))
        self.play(square.animate.move_to(2.5*RIGHT))

        triangles = [triangle.copy() for i in range(0, 8)]
        time = 0.3
        self.play(Write(triangles[0].next_to(square.get_corner(DL), UR, buff=0)), run_time=time)
        self.play(Write(triangles[1].rotate(-PI/2).next_to(square.get_corner(UL), DR, buff=0)), run_time=time)
        self.play(Write(triangles[2].rotate(PI).next_to(square.get_corner(UR), DL, buff=0)), run_time=time)
        self.play(Write(triangles[3].rotate(PI/2).next_to(square.get_corner(DR), UL, buff=0)), run_time=time)

        c_1 = MathTex("c").next_to(Line(triangles[0].get_corner(DR), triangles[0].get_corner(UL)).get_center(), direction=UR, buff=0.05)
        c_2 = MathTex("c").next_to(Line(triangles[1].get_corner(DL), triangles[1].get_corner(UR)).get_center(), direction=DR, buff=0.05)
        labels = VGroup(c_1, c_2)
        self.play(Write(labels))
        c2 = MathTex(r"c^2").move_to(square.get_center())
        self.play(ReplacementTransform(labels, c2))
        self.add_foreground_mobject(c2)

        square2 = square.copy()
        dupelicate = VGroup(square2, triangles[4].become(triangles[0]), triangles[5].become(triangles[1]), triangles[6].become(triangles[2])
        , triangles[7].become(triangles[3]))
        self.play(dupelicate.animate.move_to(2.5*LEFT))
        equal = MathTex("=")
        self.play(Write(equal))

        self.play(triangles[7].animate.move_to(Line(triangles[5].get_corner(DL), triangles[5].get_corner(UR)).get_center()))
        self.play(triangles[4].animate.next_to(square2.get_corner(DR), UL, buff=0))
        self.play(triangles[6].animate.next_to(square2.get_corner(DR), UL, buff=0))
        self.wait(0.5)

        a_1 = MathTex("a").next_to(triangles[7], direction=DOWN, buff=0.1)
        a_2 = MathTex("a").next_to(triangles[4], direction=LEFT, buff=0.1)
        b_1 = MathTex("b").next_to(triangles[7], direction=RIGHT, buff=0.1)
        b_2 = MathTex("b").next_to(triangles[6], direction=UP, buff=0.1)
        labels = VGroup(a_1, a_2, b_1, b_2)
        self.play(Write(labels))
        self.wait(0.5)

        c_square = Difference(square, Union(triangles[0], triangles[1], triangles[2], triangles[3]), fill_color=REANLEA_BLUE_LAVENDER, fill_opacity=1)
        a_square = Square(side_length=3, fill_color=REANLEA_BLUE_LAVENDER, fill_opacity=1).scale(0.5).next_to(square2.get_corner(DL), UR, buff=0)
        b_square = Square(side_length=4, fill_color=REANLEA_BLUE_LAVENDER, fill_opacity=1).scale(0.5).next_to(square2.get_corner(UR), DL, buff=0)
        c_square.set_stroke(width=DEFAULT_STROKE_WIDTH / 2)
        a_square.set_stroke(width=DEFAULT_STROKE_WIDTH / 2)
        b_square.set_stroke(width=DEFAULT_STROKE_WIDTH / 2)

        a2 = MathTex(r"a^2").move_to(a_square.get_center())
        b2 = MathTex(r"b^2").move_to(b_square.get_center())
        new_labels = VGroup(a2, b2)
        self.play(ReplacementTransform(labels, new_labels))
        self.add_foreground_mobject(a2)
        self.add_foreground_mobject(b2)
        self.wait(0.5)

        square.set_fill(opacity=0)
        self.add(c_square)
        square2.set_fill(opacity=0)
        left_over = VGroup(a_square, b_square)
        self.add(left_over)
        
        triangles_group = VGroup(triangles[0])
        for i in range(1, 8):
            triangles_group.add(triangles[i])
        self.play(Unwrite(triangles_group), Unwrite(square), Unwrite(square2))
        self.play(c_square.animate.set_stroke(width=DEFAULT_STROKE_WIDTH), a_square.animate.set_stroke(width=DEFAULT_STROKE_WIDTH), 
        b_square.animate.set_stroke(width=DEFAULT_STROKE_WIDTH))

        identity = MathTex("a^2", "+", "b^2", "=", "c^2").move_to(3.5 * DOWN)
        self.play(GrowFromCenter(triangle),
        c_square.animate.move_to(ORIGIN).shift(0.75 * RIGHT + 1 * UP),
        a_square.animate.next_to(triangle, LEFT, buff=0),
        b_square.animate.next_to(triangle, DOWN, buff=0),
        ReplacementTransform(a2, identity[0]),
        ReplacementTransform(b2, identity[2]),
        ReplacementTransform(equal, identity[3]),
        ReplacementTransform(c2, identity[4]),
        GrowFromCenter(identity[1]))

        a = MathTex("a").next_to(triangle, direction=LEFT, buff=0.1)
        b = MathTex("b").next_to(triangle, direction=DOWN, buff=0.08)
        c = MathTex("c").next_to(Line(triangle.get_corner(DR), triangle.get_corner(UL)).get_center(), direction=UR, buff=0.05)
        labels = VGroup(a, b, c)
        self.play(Write(labels))
        square = Square(side_length=0.2, stroke_width=DEFAULT_STROKE_WIDTH/2).next_to(triangle.get_corner(DL), UR, buff=0)
        self.play(FadeIn(square), run_time=0.6)
        
        self.wait()



        # manim -pqh discord.py PythagoreanIdentity





class TwistedSquares(Scene):
    def construct(self):
        title = Text("TWISTGONS").shift(UP)
        #name = Text("By R E A N L E A")
        with RegisterFont("Montserrat") as fonts:
            name=Text("by    R E A N L E A", font=fonts[0])
            name.set_color_by_gradient(REANLEA_TXT_COL)
        credit = Text("Inspired by Burkard's (aka Mathologer) Twisted Squares video", font_size=26).next_to(name, DOWN)
        banner = ManimBanner().next_to(credit, DOWN).scale(0.3)
        self.play(Write(title), run_time=0.8)
        self.play(Write(name), run_time=0.8)
        self.play(Write(credit), banner.create(), runt_time=0.8)
        self.play(banner.expand())
        self.play(Unwrite(title), Unwrite(credit, reverse=False), Unwrite(banner), Unwrite(name[0:2]), run_time=0.8)
        self.play(name[2:].animate.scale(0.3).move_to(5 * RIGHT + 3.5 * DOWN))

        color_of_polygons = WHITE
        stroke_width_of_polygons = 1
        fill_opacity_of_polygons = 0
        fix_angle = PI / 4
        size = 4
        max_num_of_polys = 100
        max_num_of_vertices = 17

        percent_label = Variable(0.5, label=Text("Percent"), num_decimal_places=5).move_to(5.5*LEFT + UP).scale(0.5)
        percent_label.label.set_color(GREEN)
        num_of_vertices_label = Variable(4, label=Text("#Sides"), var_type=Integer).move_to(5.5 * LEFT).scale(0.5)
        num_of_vertices_label.label.set_color(BLUE)
        num_of_polygons_label = Variable(16, label=Text("#Polygons"), var_type=Integer).move_to(5.5 * LEFT + DOWN).scale(0.5)
        num_of_polygons_label.label.set_color(RED)

        percent_tracker = percent_label.tracker
        num_of_vertices_tracker = num_of_vertices_label.tracker
        num_of_polygons_tracker = num_of_polygons_label.tracker

        def np_array_to_list(np_array):
            """Turns a np array of lists into a list of lists, it's a temporary fix!"""
            result = []
            for i in np_array:
                new_i = []
                for j in i:
                    new_i.append(j)
                result.append(new_i)
            return result

        def updater(mobj):
            p = int(num_of_polygons_tracker.get_value())
            v = int(num_of_vertices_tracker.get_value())
            x = percent_tracker.get_value()

            polygons[0].become(RegularPolygon(v, color=color_of_polygons, stroke_width=stroke_width_of_polygons,
                                                fill_opacity=fill_opacity_of_polygons).rotate(fix_angle).scale(size))
            vertices_of_polygons[0][:v] = np_array_to_list(polygons[0].get_vertices())
            for i in range(1, p):
                for j in range(v):
                    vertices_of_polygons[i][j] = [(1-x) * vertices_of_polygons[i-1][j][k] + x * vertices_of_polygons[i-1][(j+1) % v][k]
                                                  for k in range(3)]

                polygons[i].become(Polygon(*vertices_of_polygons[i][:v], color=color_of_polygons, stroke_width=stroke_width_of_polygons,
                                       fill_opacity=fill_opacity_of_polygons))
            for k in range(p, max_num_of_polys):
                polygons[k].become(Square(side_length=0))
            mobj.become(VGroup(*polygons))


        # a list of lists, vertices_of_polygons[i] is the ith polygon's list of vertices while vertices_of_polygons[i][j] is its jth vertix
        vertices_of_polygons = [list(range(max_num_of_vertices)) for i in range(max_num_of_polys)]
        polygons = [Square() for i in range(max_num_of_polys)]

        group = VGroup(*polygons).add_updater(updater)
        self.play(DrawBorderThenFill(group), run_time=1.5)
        self.play(Write(percent_label), run_time=0.7)
        self.play(Write(num_of_vertices_label), run_time=0.6)
        self.play(Write(num_of_polygons_label), run_time=0.5)

        self.play(percent_tracker.animate.set_value(0.001), num_of_polygons_tracker.animate.set_value(max_num_of_polys), run_time=3)
        self.play(percent_tracker.animate.set_value(0.5), num_of_polygons_tracker.animate.set_value(16), run_time=3)
        texts = Text("But wait, there is more!!!", gradient=[RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]).to_edge(UP).shift(0.1* UP)
        self.play(Write(texts))
        self.play(Unwrite(texts), num_of_polygons_tracker.animate.set_value(0))

        fix_angle = 0
        for i in range(3, 7):
            if i != 4:
                num_of_polygons_tracker.set_value(16)
                num_of_vertices_tracker.set_value(i)
                self.play(DrawBorderThenFill(group))
                self.wait(0.5)
                self.play(percent_tracker.animate.set_value(0.001), num_of_polygons_tracker.animate.set_value(max_num_of_polys), run_time=3)
                self.play(percent_tracker.animate.set_value(0.5), num_of_polygons_tracker.animate.set_value(0), run_time=3)
                self.wait(0.5)

        texts = Text("It's time to be faster...", gradient=[RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE])
        self.play(Write(texts), run_time=0.6)
        self.play(Unwrite(texts), run_time=0.5)
        num_of_polygons_tracker.set_value(max_num_of_polys)
        for i in range(7, 18):
            percent_tracker.set_value(0.5)
            num_of_vertices_tracker.set_value(i)
            self.play(DrawBorderThenFill(group), run_time=1.8)
            self.play(percent_tracker.animate.set_value(0.001), run_time=1.8)
            self.wait(0.25)

        self.wait()


        # manim -pqh discord.py TwistedSquares



myTexTemplate = TexTemplate(
                    tex_compiler="xelatex",
                    output_format='.xdv',
                    )
myTexTemplate.add_to_preamble(r"\usepackage{fontspec}")

MathTex.set_default(tex_template=myTexTemplate) 
Tex.set_default(tex_template=myTexTemplate) 

class tedxnitjalandhar(Scene):
    def construct(self):
        text1= Tex(r"\fontspec{arialbd.ttf}TEDx \fontspec{arial.ttf}NIT Jalandhar").scale(1.2).shift(2*UP)
        text2 = Tex(r"\fontspec{arial.ttf}Independently organized TED event").set_color("#DA291C")

        
        self.play(Write(text1))
        self.play(text1[0][3].animate.shift(0.1*UP))
        self.play(text1[0][:4].animate.set_color("#DA291C"))
        self.play(Write(text2))
        
        self.wait(2)


        # manim -pqh discord.py tedxnitjalandhar



class alphaGradient(Scene):
    def construct(self):
        
        imgsize = (100, 1) #The size of the image
        image = Image.new('RGBA', imgsize) #Create the image

        (r,g,b) = color_to_int_rgb( YELLOW )

        dist = norm(loc = 0.5, scale = .1)
        peak = dist.pdf(0.5)

        for xpix in range(imgsize[0]):
            x = xpix/imgsize[0]
            a = 255 * (dist.pdf(x)/peak)
            #Place the pixel        
            image.putpixel((xpix, 0), (int(r), int(g), int(b), int(a)))  
              
        distBar = ImageMobject(image)
        distBar.stretch_to_fit_width(10).stretch_to_fit_height(2)        
        self.add(distBar.shift(1*DOWN))
        ax = Axes(x_range=[0,1,.1],y_range=[0,peak],x_length=10,y_length=2).move_to(distBar.get_left(),LEFT)
        distGraph = ax.plot(dist.pdf).set_color(BLUE)
        self.add(distGraph) 
        self.wait(2)  

        (r,g,b) = color_to_int_rgb( "#ff0000" )

        dist = gamma(a=1.99)
        peak = 0.4

        for xpix in range(imgsize[0]):
            x = xpix/imgsize[0]*8
            a = 255 * (dist.pdf(x)/peak)
            #Place the pixel        
            image.putpixel((xpix, 0), (int(r), int(g), int(b), int(a)))  
              
        distBar2 = ImageMobject(image)
        distBar2.stretch_to_fit_width(10).stretch_to_fit_height(2)        
        self.add(distBar2.shift(1.5*UP))
        ax2 = Axes(x_range=[0,8,1],y_range=[0,peak],x_length=10,y_length=2).move_to(distBar2.get_left(),LEFT)
        distGraph2 = ax2.plot(dist.pdf).set_color(BLUE)
        self.add(distGraph2) 
        self.wait(2) 


        # manim -pqh discord.py alphaGradient



class imgrad(Scene):
    def construct(self):
        
        imgsize = (100, 100) #The size of the image

        image = Image.new('RGB', imgsize) #Create the image

        innerColor = [0, 255, 0] #Color at the center
        outerColor = [0, 0, 0] #Color at the corners

        def gradient(t: float) -> float:
            return (1-math.erf((t-30)/10))/2

        for y in range(imgsize[1]):
            for x in range(imgsize[0]):

                #Find the distance to the center
                distanceToCenter = gradient(math.sqrt((x - imgsize[0]/2) ** 2 + (y - imgsize[1]/2) ** 2))

                #Calculate r, g, and b values
                r = innerColor[0] * distanceToCenter + outerColor[0] * (1 - distanceToCenter)
                g = innerColor[1] * distanceToCenter + outerColor[1] * (1 - distanceToCenter)
                b = innerColor[2] * distanceToCenter + outerColor[2] * (1 - distanceToCenter)

                #Place the pixel        
                image.putpixel((x, y), (int(r), int(g), int(b)))  
              
        bgGlow = ImageMobject(image)
        bgGlow.height = 7        
        self.add(bgGlow)
        self.wait(2)


        # manim -pqh discord.py imgrad






class stickman_2(Scene):
    def construct(self):
        """
        n       parte           len     ang_i
        0       tronco          2.5     0
        1       braco esq       1.5     7*PI/6
        2       brado dir       1.5     5*PI/6
        3       antebra esq     1.5     PI
        4       antebra dir     1.5     PI
        5       coxa esq        1.5     7*PI/6
        6       coxa dir        1.5     5*PI/6
        7       perna esq       1.5     PI
        8       perna dir       1.5     PI
        9       pescoço         0.25    0
        -       cabeça          1       -

        """
        init_angs = (0, PI*7/6, PI*5/6, PI, PI, PI*7/6, PI*5/6, PI, PI, 0)
        init_comprs = (2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.25, 0.5)
        init_colors = (BLACK, PURE_BLUE, PURE_GREEN, BLUE, GREEN,
                       PURE_BLUE, PURE_GREEN, BLUE, GREEN, GRAY, RED)

        self.x0, self.y0 = ValueTracker(0.0), ValueTracker(0.0)
        self.compr, self.ang = [], []
        self.linhas = VGroup()
        self.direcao = "frente"
        self.raio_cabeca = 0.5
        self.espessura = 12

        for i in range(10):
            self.ang.append(ValueTracker(init_angs[i]))
            self.compr.append(ValueTracker(init_comprs[i]))
            self.linhas.add(
                Line(
                    color=init_colors[i],
                    stroke_width=self.espessura,
                    start=self.get_start(i),
                    end=self.get_end(i)
                )
            )
            self.play(Create(self.linhas[i]), run_time=0.25)
        self.cabeca = Circle(
            radius=self.raio_cabeca).set_fill(init_colors[-1], 1)
        self.cabeca.move_to(self.get_end(9) + 0.5 * self.get_versor(9))

        self.play(Create(self.cabeca), run_time=0.25)
        self.wait()

        # Parte 1
        self.atualiza_tudo()
        self.redimensiona(0.25, True)
        self.redimensiona(8, True)
        self.redimensiona(0.5, True)
        self.anda("direita", 4, 1)
        self.anda("esquerda", 4, 1)
        self.wait()
        self.acena("direito")
        self.acena("esquerdo")
        self.wait()
        self.recolore(BLACK)
        self.wait(2)
        self.recolore(ORANGE)
        self.wait(2)
        self.recolore(PINK)
        self.wait(2)
        self.anda("direita", 4, 0.5)
        self.anda("esquerda", 4, 0.5)
        self.wait()
        self.acena("direito")
        self.redimensiona(4, True)
        self.redimensiona(0.25, False)
        self.wait()

        # self.embed()
        # for i in range(len(self.linhas)):
        #     print("start{}:[{:.2f}, {:.2f}]\t\tend{}:[{:.2f}, {:.2f}]".format(
        #         i, self.get_start(i)[0], self.get_start(i)[1],
        #         i, self.get_end(i)[0], self.get_end(i)[1]))

        # Fim
        self.desatualiza_tudo()
        self.play(FadeOut(VGroup(self.cabeca, self.linhas)), run_time=2)
        self.wait()

    def get_compr(self, n):
        try:
            compr = self.compr[n].get_value()
        except IndexError as err:
            print(err)

        return compr

    def get_ang(self, n):
        try:
            ang = self.ang[n].get_value()
        except IndexError as err:
            print(err)

        return ang

    def get_versor(self, n):
        try:
            n = np.array([np.sin(self.get_ang(n)), np.cos(self.get_ang(n)), 0])
        except IndexError as err:
            print(err)

        return n

    def get_dif(self, n):
        return self.get_compr(n) * self.get_versor(n)

    def get_start(self, n):
        if n == 0:
            start = np.array([self.x0.get_value(), self.y0.get_value(), 0])
        elif n == 1 or n == 2:
            start = self.get_end(0)
        elif n == 3:
            start = self.get_end(1)
        elif n == 4:
            start = self.get_end(2)
        elif n == 5 or n == 6:
            start = self.get_start(0)
        elif n == 7:
            start = self.get_end(5)
        elif n == 8:
            start = self.get_end(6)
        elif n == 9:
            start = self.get_end(0)

        # print("start[{}]: [{:.2}, {:.2}]".format(n, start[0], start[1]))

        return start

    def get_end(self, n):
        end = self.get_start(n) + self.get_dif(n)

        # print("end[{}]: [{:.2}, {:.2}]".format(n, end[0], end[1]))

        return end

    def atualiza_linha(self, linha):
        try:
            for i in range(10):
                if linha == self.linhas[i]:
                    n = i
        except:
            print("Linha não encontrada.")

        linha.put_start_and_end_on(
            start=self.get_start(n), end=self.get_end(n))

    def atualiza_cabeca(self, cabeca):
        cabeca.move_to(
            self.get_end(9) + self.raio_cabeca * self.get_versor(9))

    def atualiza_tudo(self):
        for linha in self.linhas:
            linha.add_updater(self.atualiza_linha)
        self.cabeca.add_updater(self.atualiza_cabeca)

    def desatualiza_tudo(self):
        for linha in self.linhas:
            linha.remove_updater(self.atualiza_linha)
        self.cabeca.remove_updater(self.atualiza_cabeca)

    def vira(self, direcao):
        if direcao == "frente" and self.direcao != "frente":
            self.play(
                self.ang[0].animate.set_value(0),
                self.ang[1].animate.set_value(PI*7/6),
                self.ang[2].animate.set_value(PI*5/6),
                self.ang[3].animate.set_value(PI),
                self.ang[4].animate.set_value(PI),
                self.ang[5].animate.set_value(PI*7/6),
                self.ang[6].animate.set_value(PI*5/6),
                self.ang[7].animate.set_value(PI),
                self.ang[8].animate.set_value(PI),
                self.ang[9].animate.set_value(0)
            )
            self.direcao = "frente"
        elif direcao == "direita" and self.direcao != "direita":
            self.play(
                self.ang[0].animate.set_value(0),
                self.ang[1].animate.set_value(PI*7/6),
                self.ang[2].animate.set_value(PI*11/12),
                self.ang[3].animate.set_value(PI*23/24),
                self.ang[4].animate.set_value(PI*4/5),
                self.ang[5].animate.set_value(PI*13/12),
                self.ang[6].animate.set_value(PI*5/6),
                self.ang[7].animate.set_value(PI*7/6),
                self.ang[8].animate.set_value(PI*23/24),
                self.ang[9].animate.set_value(0)
            )
            self.direcao = "direita"
        elif direcao == "esquerda" and self.direcao != "esquerda":
            self.play(
                self.ang[0].animate.set_value(0),
                self.ang[1].animate.set_value(PI*(2-7/6)),
                self.ang[2].animate.set_value(PI*(2-11/12)),
                self.ang[3].animate.set_value(PI*(2-23/24)),
                self.ang[4].animate.set_value(PI*(2-4/5)),
                self.ang[5].animate.set_value(PI*(2-13/12)),
                self.ang[6].animate.set_value(PI*(2-5/6)),
                self.ang[7].animate.set_value(PI*(2-7/6)),
                self.ang[8].animate.set_value(PI*(2-23/24)),
                self.ang[9].animate.set_value(0)
            )
            self.direcao = "esquerda"

    def anda(self, direcao, passos=2, seg_p_passo=0.5):
        if direcao == "direita" and self.direcao != "direita":
            self.vira("direita")
        elif direcao == "esquerda" and self.direcao != "esquerda":
            self.vira("esquerda")

        assert passos % 2 == 0

        deltaX = 3 * abs(self.get_end(8)[0] - self.get_end(6)[0])
        ang_passos = PI/4

        if direcao == "esquerda":
            deltaX *= -1
            ang_passos *= -1

        for i in range(passos):
            ang_passos *= -1
            self.play(
                self.ang[1].animate.increment_value(ang_passos),
                self.ang[2].animate.increment_value(-ang_passos),
                self.ang[3].animate.increment_value(ang_passos/2),
                self.ang[4].animate.increment_value(-ang_passos/2),
                self.ang[5].animate.increment_value(ang_passos),
                self.ang[6].animate.increment_value(-ang_passos),
                self.ang[7].animate.increment_value(ang_passos),
                self.ang[8].animate.increment_value(-ang_passos),
                self.x0.animate.increment_value(deltaX),
                run_time=seg_p_passo,
                # rate_func=rate_functions.linear
            )

    def acena(self, braco="direito", acenadas=2, velocidade=0.5, angulo_mao=PI/3):
        if self.direcao != "frente":
            self.vira("frente")

        ang_bra = PI/12
        if braco == "esquerdo":
            bra, antebra = self.ang[1], self.ang[3]
            ang_antebra = antebra.get_value()
        else:
            bra, antebra = self.ang[2], self.ang[4]
            ang_bra *= -1
            ang_antebra = -antebra.get_value()

        self.play(
            bra.animate.increment_value(ang_bra),
            antebra.animate.increment_value(ang_antebra),
            run_time=velocidade)

        self.play(
            antebra.animate.increment_value(-angulo_mao/2),
            run_time=velocidade/2)
        for i in range(acenadas):
            self.play(
                antebra.animate.increment_value(angulo_mao),
                run_time=velocidade)
            self.play(
                antebra.animate.increment_value(-angulo_mao),
                run_time=velocidade)

        self.direcao = "errada"
        self.vira("frente")

    def redimensiona(self, fator=1.0, redi_espess=False):
        self.desatualiza_tudo()

        if redi_espess:
            self.espessura *= fator

        new_linhas = VGroup()
        for i in range(10):
            self.compr[i] *= fator
            new_linhas.add(
                Line(
                    color=self.linhas[i].get_color(),
                    stroke_width=self.espessura,
                    start=self.get_start(i),
                    end=self.get_end(i)
                )
            )

        self.raio_cabeca *= fator
        new_cabeca = Circle(radius=self.raio_cabeca)
        new_cabeca.set_fill(self.cabeca.get_color(), 1)
        new_cabeca.set_stroke(self.cabeca.get_color(), 1)
        new_cabeca.move_to(
            self.get_end(9) + self.raio_cabeca * self.get_versor(9))

        self.play(
            Transform(self.linhas, new_linhas),
            Transform(self.cabeca, new_cabeca),
            run_time=2)

        self.atualiza_tudo()

    def recolore(self, color):
        self.cabeca.set_fill(color, 1).set_stroke(color, opacity=1)
        self.cabeca.set_stroke(color, 1)
        for linha in self.linhas:
            linha.set_color(color)


     # manim -pqh discord.py stickman_2



class funcs(Scene):
    def construct(self):
        a = ValueTracker(1)
        b = ValueTracker(1)
        c = ValueTracker(1)
        d = ValueTracker(1)
        e = ValueTracker(1)

        ax = Axes(x_range=[-10,10],y_range=[-10,10])
        self.add(ax)

        f = always_redraw(lambda: ax.plot(lambda x: a.get_value()*x**2+b.get_value()*x+c.get_value()).set_color(RED))
        g = always_redraw(lambda: ax.plot(lambda x: d.get_value()*x+e.get_value()).set_color(BLUE))

        self.play(Create(f))
        self.play(Create(g))
        self.wait(2)

        roots = VGroup()
        def rootsUpdater(mobj):
            res = VGroup()
            f = lambda x: a.get_value()*x**2+b.get_value()*x+c.get_value()
            g = lambda x: d.get_value()*x+e.get_value()
            fg = lambda x: f(x)-g(x)
            zeros = fsolve(fg, [ax.x_range[0],ax.x_range[1]])
            for zero in zeros:
                res.add(Circle(radius=0.2).set_color(YELLOW).move_to(ax.c2p(zero,f(zero))))
            mobj.become(res)
        roots.add_updater(rootsUpdater)

        self.add(roots)

        self.play(c.animate.set_value(-3))
        self.wait()


        # manim -pqh discord.py funcs




class Gates_design(VGroup):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        top_arc = Arc(radius=1, start_angle=0, angle=PI).rotate(-PI/2)
        line1=Line()
        line1.move_to(top_arc.get_start()).shift((line1.get_length()/2)*LEFT)
        line2=line1.copy()
        line1.move_to(top_arc.get_end()).shift((line2.get_length()/2)*LEFT)
        line3=Line(line1.get_start(),line2.get_start())

        connector_line1=Line()
        connector_line1.move_to(line3.get_start()-[0,line3.get_length()/3,0]).shift((connector_line1.get_length()/2)*LEFT)
        connector_line2=Line()
        connector_line2.move_to(line3.get_start()-[0,line3.get_length()*2/3,0]).shift((connector_line1.get_length()/2)*LEFT)

        circle=Circle(radius=0.3)
        circle.move_to(top_arc.point_from_proportion(0.5)).shift(circle.get_radius()*RIGHT)

        line4=Line()
        line4.move_to(circle.get_right()).shift((line4.get_length()/2)*RIGHT)
        self.gate_vmobject=VGroup(top_arc,line1,line2,line3,connector_line1,connector_line2,circle,line4)
        

class Gates(Scene):
    def construct(self):
        gate1 = Gates_design()
        gate2 = Gates_design()
        gate_group= VGroup(gate1.gate_vmobject,gate2.gate_vmobject).arrange(DOWN)
        #self.add(gate_group)

        self.play(
            Write(gate_group)
        )



        # manim -pqh discord.py Gates


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

        plt = ax.plot(f, discontinuities = [0.5]).set_stroke(width=15, color=[PURE_GREEN,BLUE_C,PURE_RED])


        self.play(
            Create(plt)
        )

        self.wait(3)


        # manim -pqh discord.py defFun




class FancyText(Scene):
    def construct(self):
        text_orig = Text("Hello World!").scale_to_fit_width(config.frame_width - 1)

        text = DashedVMobject(
            VMobject().set_points(text_orig.get_all_points()),
            dashed_ratio=1,
            num_dashes=45,
        )
        text.save_state()

        text_segments = VGroup(*text.submobjects)
        anims = []
        random.shuffle(text_segments)
        for segment in text_segments:
            anims.append(FadeIn(segment, shift=DOWN))

        self.play(
            AnimationGroup(*anims, lag_ratio=0.1, run_time=3)
        )
        #self.play(FadeTransform(text_segments, text_orig))
        self.wait()


        # manim -pqh discord.py FancyText


 

class VectorFieldScene1(Scene):
    def construct(self):

        frame_width = config["frame_width"]
        frame_height = config["frame_height"]

        grid = NumberPlane(axis_config={"include_tip":True},
            background_line_style={
                "stroke_color": BLUE,
                "stroke_width": 0.8,
                "stroke_opacity": 0.2
            }
        )
        #self.play(Create(grid))
        self.wait()

        #State Space: UNSTABLE NODE
        def x_dot(x,y):
            return y

        def y_dot(x,y):
            return 2*x/3-(4/9)*(x*x*x)

        #Angle of vector
        def phi(x,y):
            return np.arctan2(y_dot(x,y),x_dot(x,y))

        def vec_x(x,y):
            return np.cos(phi(x,y))

        def vec_y(x,y):
            return np.sin(phi(x,y))

        func = lambda pos: (x_dot(pos[0],pos[1]))*RIGHT + (y_dot(pos[0],pos[1]))*UP
        vf = ArrowVectorField(func, x_range = [-7, 7, 0.4], y_range = [-5, 5, 0.4])#length_func = lambda x: x / 4)#, length_func  = length_func) #stroke_width does nothing :(
        #self.play(Create(vf))
        self.wait()

        self.t = 0

        #Circle
        a = 1.5
        R = 0.2
        t = 0.05
        x1 = -a-R
        x2 = -a+np.sqrt(R*R-t*t)
        x3 = -np.sqrt(R*R-t*t)
        x4 = np.sqrt(R*R-t*t)
        x5 = a-np.sqrt(R*R-t*t)
        x6 = a+R

        T = 6

        dot1 = Dot(color=BLUE).move_to(grid.c2p(-a, 0,0))
        dot2 = Dot(color=GREEN).move_to(grid.c2p(0, 0,0))
        dot3 = Dot(color=BLUE).move_to(grid.c2p(+a, 0,0))

        #self.play(Create(dot1))
        #self.play(Create(dot2))
        #self.play(Create(dot3))


        r = lambda theta: 2 + 0.2 * np.sin(4*theta) + 0.01*theta*theta*(theta-2*np.pi)*(theta-2*np.pi)
        graph1 = grid.plot_polar_graph(r, [0, 2 * PI], color=YELLOW)
        self.wait()
        self.play(Create(graph1))
        self.wait()





        def y_fun(x):
            if (x1<=x<=x2):
                return np.sqrt(R*R-(x+a)*(x+a))
            elif (x2<=x<=x3):
                return t
            elif (x3<=x<=x4):
                return np.sqrt(R*R-x*x)
            elif (x4<=x<=x5):
                return t
            elif (x5<=x<=x6):
                return np.sqrt(R*R-(x-a)*(x-a))
            else:
                return 0

        def x_para(u):
            if (u < T):
                return 2*(R+a)*u/T - (R+a)
            elif (u >= T):
                return -2*(R+a)*u/T + 3*(R+a)
        
        def y_para(u):
            if (u < T):
                return y_fun(2*(R+a)*u/T - (R+a))
            elif (u >= T):
                return -y_fun(-2*(R+a)*u/T + 3*(R+a))
        
        graph2 = grid.plot_parametric_curve(lambda u: [x_para(u), y_para(u), 0], t_range=(0,2*T),color = YELLOW)
        self.wait()
        #self.play(Transform(graph1,graph2))
        # self.play(Create(graph1),rate_func=linear)
        self.wait(3)



        # #DOT
        # dot = always_redraw(
        #     lambda: Dot(color=YELLOW).move_to(
        #         grid.c2p(x_para(self.t), y_para(self.t),0) #circle parameter eqns
        #     )
        # )
        # dot.set_z_index(1)
        # self.play(Create(dot))
        # self.wait()

        # #ARROW
        # L = 1
        # arrow = always_redraw(
        #     lambda: Arrow(grid.c2p(x_para(self.t), y_para(self.t),0), grid.c2p(x_para(self.t) + L*vec_x(x_para(self.t), y_para(self.t)), y_para(self.t) + L*vec_y(x_para(self.t), y_para(self.t)),0), buff=0, color=YELLOW)
        # )
        # self.play(Create(arrow))

        # def update_box(mob, dt):
        #     self.t = self.t + dt

        # dot.add_updater(update_box)
        # self.wait(2*T)
        # dot.remove_updater(update_box)
        # self.wait(3)



        # manim -pqh discord.py VectorFieldScene1

        # manim -sqk discord.py VectorFieldScene1





class ArbitraryShape(Scene):
    def construct(self):


        grid = NumberPlane(axis_config={"include_tip":True},
            background_line_style={
                "stroke_color": BLUE,
                "stroke_width": 0.8,
                "stroke_opacity": 0.2
            }
        )


        r1 = lambda theta: 2 + 0.2 * np.sin(4*theta) + 0.01*theta*theta*(theta-2*np.pi)*(theta-2*np.pi)
        graph1 = grid.plot_polar_graph(r1, [0, 2 * PI])
        graph1.set_stroke(width=15, color=[REANLEA_BLUE_LAVENDER,REANLEA_SLATE_BLUE]).scale(.75).shift(4*LEFT)

        r2 = lambda theta: 2 + 0.2 * np.cos(4*theta) + 0.01*theta*theta*(theta-2*np.pi)*(theta-2*np.pi)
        graph2 = grid.plot_polar_graph(r2, [0, 2 * PI])
        graph2.set_stroke(width=15, color=[REANLEA_SLATE_BLUE, REANLEA_TXT_COL_LIGHTER]).scale(.75).shift(4*RIGHT)


        self.play(Create(graph1))
        self.play(Create(graph2))
        self.wait(2)


        # manim -sqk discord.py ArbitraryShape



class VectorFieldScene2(Scene):
    def construct(self):

        grid = NumberPlane(axis_config={"include_tip":True},
            background_line_style={
                "stroke_color": BLUE,
                "stroke_width": 0.8,
                "stroke_opacity": 0.2
            }
        )
        self.play(Create(grid))
        self.wait()

        # length_func = lambda x: 2

        def x_dot(x,y):
            return y/2

        def y_dot(x,y):
            return x/2-x*x*x/8

        def phi(x,y):
            return np.arctan2(y_dot(x,y),x_dot(x,y))

        def vec_x(x,y):
            return np.cos(phi(x,y))

        def vec_y(x,y):
            return np.sin(phi(x,y))

        func = lambda pos: (x_dot(pos[0],pos[1]))*RIGHT + (y_dot(pos[0],pos[1]))*UP
        vf = ArrowVectorField(func, x_range = [-7, 7, 0.4], y_range = [-5, 5, 0.4])#, length_func  = length_func) #stroke_width does nothing :(
        self.play(Create(vf))
        self.wait()


        self.x1 = 3 #initial value
        self.v1 = 0
        self.t = 0
        dot = always_redraw(
            lambda: Dot().move_to(
                grid.c2p(self.x1, self.v1)
            )
        )
        self.play(Create(dot))
        L = 1

        path = VMobject()
        path.set_points_as_corners([dot.get_center(), dot.get_center()])
        def update_path(path):
            previous_path = path.copy()
            previous_path.add_points_as_corners([dot.get_center()])
            path.become(previous_path)
        path.add_updater(update_path)
        self.add(path)

        def update_box(mob, dt):
            # #update velocity
            self.v1 = self.v1 + y_dot(self.x1,self.v1) * dt
            #update position
            self.x1 = self.x1 + x_dot(self.x1,self.v1) * dt
            #update time
            self.t = self.t + dt

        dot.add_updater(update_box)

        # self.wait(1)

        # stream_lines = StreamLines(
        #     func, stroke_width=1,
        #     max_anchors_per_line=50
        # )
        # self.add(stream_lines)
        # stream_lines.start_animation(warm_up=True, flow_speed=1, max_anchors_per_line=5)
        # self.wait(stream_lines.virtual_time / stream_lines.flow_speed)

        T = 18.418
        self.wait(T)
        dot.remove_updater(update_box)
        self.wait()
        self.play(Uncreate(dot))

        dot1 = Dot(color=BLUE).move_to(grid.c2p(-2,0,0)) #center
        dot2 = Dot(color=GREEN).move_to(grid.c2p(0,0,0)) #saddle
        dot3 = Dot(color=BLUE).move_to(grid.c2p(2,0,0)) #center
        self.play(Create(dot1),Create(dot2),Create(dot3))

        self.wait()

        dot_right_loop = always_redraw(
            lambda: Dot().move_to(
                grid.c2p(self.x1, self.v1)
            )
        )
        path_right = VMobject()
        path_right.set_points_as_corners([dot_right_loop.get_center(), dot_right_loop.get_center()])
        def update_path(path):
            previous_path = path.copy()
            previous_path.add_points_as_corners([dot_right_loop.get_center()])
            path.become(previous_path)
        path_right.add_updater(update_path)
        self.add(path_right)

        self.wait()
        self.x1 = 2.5 #initial value
        self.v1 = 0
        self.t = 0
        self.play(Create(dot_right_loop))
        dot_right_loop.add_updater(update_box)
        T = 9.51 #IC: (2.5,0)
        self.wait(T)
        dot_right_loop.remove_updater(update_box)
        self.wait()
        self.play(Uncreate(dot_right_loop))


        dot_left_loop = always_redraw(
            lambda: Dot().move_to(
                grid.c2p(self.x1, self.v1)
            )
        )
        path_left = VMobject()
        path_left.set_points_as_corners([dot_left_loop.get_center(), dot_left_loop.get_center()])
        def update_path(path):
            previous_path = path.copy()
            previous_path.add_points_as_corners([dot_left_loop.get_center()])
            path.become(previous_path)
        path_left.add_updater(update_path)
        self.add(path_left)
        self.play(Uncreate(dot_right_loop))
        self.x1 = -2.5 #initial value
        self.v1 = 0
        self.t = 0
        self.play(Create(dot_left_loop))
        dot_left_loop.add_updater(update_box)
        T = 9.51 #IC: (-2.5,0)
        self.wait(T)
        dot_left_loop.remove_updater(update_box)
        self.wait()
        self.play(Uncreate(dot_left_loop))



        # manim -pqh discord.py VectorFieldScene2

        # manim -sqk discord.py VectorFieldScene2



from numpy import*

def credits(self, title: str, credit: str):
    title = Text(title).shift(UP).scale(0.6)
    name = Text("By R E A N L E A")
    credit = Text(credit, font_size=26).next_to(name, DOWN)
    banner = ManimBanner().next_to(credit, DOWN).scale(0.3)
    self.play(Write(title), run_time=0.8)
    self.play(Write(name), run_time=0.8)
    self.add_foreground_mobject(name)
    self.play(Write(credit), banner.create(), runt_time=0.8)
    self.play(banner.expand())
    self.play(Unwrite(title), Unwrite(credit, reverse=False), Unwrite(banner), Unwrite(name[0:2]), run_time=0.8)
    self.play(name[2:].animate.scale(0.3).move_to(5 * RIGHT + 3.5 * DOWN))

class ComplexSequences(Scene):
    def construct(self):
        credits(self, "Complex Sequences, Twisted Polygons and Cardioids", "I feel like this deserves a part 2, Burkard.")
        max_num_of_dots = 42
        segments_color = "#5200FF"

        a = 2.1
        plane = ComplexPlane(background_line_style={"stroke_color": "#00FFFF", "stroke_opacity": 0.7}).scale(a)
        x_label = plane.get_x_axis_label(Tex("Re"), RIGHT, DOWN, buff=0.1).set_color(BLUE).scale(0.7)
        y_label = plane.get_y_axis_label(Tex("Im"), UP, LEFT, buff=0.1).set_color(BLUE).scale(0.7)
        self.play(Write(plane), run_time=1)
        self.play(Write(x_label), Write(y_label))
        unit_circle = Circle(radius=1, color="#00FF4C").scale(a)
        self.play(Write(unit_circle))


        initial_point = (1/sqrt(2)) + (1/sqrt(2))*1j
        modulus_var = Variable(absolute(initial_point), Tex(r"$|z_0|$"), num_decimal_places=3).move_to(5.4*LEFT + 0.5*UP).scale(0.8)
        modulus_var.label.set_color(YELLOW)
        modulus_var.add_updater(lambda x: x.tracker.set_value(absolute(get_dot())))
        arg_var = Variable(angle(initial_point), Tex(r"Arg $z_0$"), num_decimal_places=3).move_to(5.4*LEFT + 0.5*DOWN).scale(0.8)
        arg_var.label.set_color(YELLOW)
        arg_var.add_updater(lambda x: x.tracker.set_value(angle(get_dot())))

        segments = list(range(max_num_of_dots))
        dots = list(range(max_num_of_dots))
        dots[0] = Dot(plane.number_to_point(initial_point), color=YELLOW)
        dot_label = MathTex("z_0", font_size=45).next_to(dots[0], UR, buff=0.1)
        dot_label.add_updater(lambda x: x.next_to(dots[0], UR, buff=0.1))

        def get_dot(): return (1/a)*(dots[0].get_center()[0] + dots[0].get_center()[1]*1j)

        self.play(GrowFromCenter(dots[0]))
        self.play(Write(dot_label))
        self.add_foreground_mobjects(dots[0], dot_label)
        self.play(Write(modulus_var), Write(arg_var))
        self.wait()


        texts = Tex("Consider the complex number $z_0$").move_to(3*UP)
        self.play(Write(texts))
        self.play(FadeOut(texts, shift=UP))
        texts = Tex(r"and the complex valued function $f(z)=z\cdot z_0$").move_to(3*UP)
        self.play(Write(texts))
        self.play(modulus_var.animate.shift(UP), arg_var.animate.shift(UP),
                    texts[0][27:].animate.move_to(5.4*LEFT + 0.5*DOWN).scale(0.8), FadeOut(texts[0][:27], shift=UP))
        
        function_text = texts[0][27:]
        frame = SurroundingRectangle(function_text, color=segments_color)
        self.play(Create(frame))


        texts = Tex(r"What do you think will happen if we keep applying $f$ to $z_0$?").move_to(3*UP)
        self.play(Write(texts))
        self.wait(0.5)
        self.play(FadeOut(texts, shift=UP))

        ###############################
        def nth_term(n, f, z0):
            result = z0
            for i in range(n):
                result = f(result)
            return result

        f = lambda z: z * get_dot()
        
        def group_updater(mobj):
            z0 = get_dot()
            for i in range(1, max_num_of_dots):
                dots[i].become(Dot(plane.n2p(nth_term(i, f, z0)), color=segments_color))
                segments[i].become(Line(dots[i-1].get_center(), dots[i].get_center(), color=segments_color))

        for i in range(1, max_num_of_dots):
            dots[i] = Dot(plane.n2p(nth_term(i, f, initial_point)), color=segments_color) 
            segments[i] = Line(dots[i-1].get_center(), dots[i].get_center(), color=segments_color)
            self.play(GrowFromCenter(dots[i]), Create(segments[i]), run_time=2**(-0.4*i))
        
        texts = Tex(r"Hmmm. But what if $z_0$ was different?").move_to(3*UP)
        self.play(Write(texts))
        self.wait(0.5)
        self.play(FadeOut(texts, shift=UP))
        
        group = VGroup().add_updater(group_updater)
        self.add(group)
        ###############################

        self.play(dots[0].animate.move_to(plane.n2p(-initial_point)), run_time = 3)

        r = ValueTracker(1)
        theta = ValueTracker(-(3/4)*PI)

        def dot_updater(d):
            d.move_to(plane.n2p(r.get_value()*(cos(theta.get_value()) + sin(theta.get_value())*1j)))
        
        dots[0].add_updater(dot_updater)


        self.play(r.animate.set_value(0.9), theta.animate.set_value(theta.get_value()-PI/2), run_time = 2)
        self.play(r.animate.set_value(0.7), theta.animate.set_value(theta.get_value()-(4/5)*PI), run_time = 2)
        self.play(r.animate.set_value(1.1), theta.animate.set_value(theta.get_value()-PI/3), run_time = 2)
        self.play(r.animate.set_value(0.4), theta.animate.set_value(theta.get_value()-PI), run_time = 2)
        self.play(r.animate.set_value(1), theta.animate.set_value(0), run_time = 2)
        self.play(r.animate.set_value(1.05), theta.animate.set_value((3/4)*PI), run_time = 2)
        self.play(r.animate.set_value(1.15), theta.animate.set_value((3/2)*PI), run_time = 1.5)
        self.play(r.animate.set_value(1.25), theta.animate.set_value(2*PI), run_time = 1)
        self.play(r.animate.set_value(1), theta.animate.set_value(0), run_time = 5)
        self.play(theta.animate.set_value(-PI), run_time = 40, rate_func=linear)
        self.play(r.animate.set_value(0.99))
        self.play(theta.animate.set_value(-2*PI), run_time = 40, rate_func=linear)
        self.play(r.animate.set_value(1))
        



        texts = Tex(r"Very pretty, don't you think!").move_to(3*UP)
        texts2 = Tex(r"-Burkard").next_to(texts, DR).scale(0.5).shift(LEFT)
        self.play(Write(texts))
        self.play(Write(texts2))
        self.play(FadeOut(texts, texts2, shift=UP))

        texts = Tex(r"Hmmm. But did you notice the twisted polygons? coincidence?").move_to(3*UP)
        self.play(Write(texts))
        self.wait(0.5)
        self.play(FadeOut(texts, shift=UP))
        self.wait()

        texts = Tex(r"Now let's see what happens if we change $f$").move_to(3*UP)
        self.play(Write(texts))
        self.wait(0.5)
        self.play(FadeOut(texts, shift=UP))

        self.play(Transform(function_text, Tex("$f(z) = z^2$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8)), 
                  Transform(frame, SurroundingRectangle(Tex("$f(z) = z^2$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8), color=segments_color)))

        f = lambda z: z*z

        self.play(r.animate.set_value(1.201), theta.animate.set_value(-PI), run_time = 2)
        self.play(r.animate.set_value(0.7), theta.animate.set_value(PI), run_time = 2)
        self.play(r.animate.set_value(1))
        self.play(theta.animate.set_value(2*PI), run_time = 2)
        theta.set_value(0)

        texts = Tex(r"A cardioid? Why?").move_to(3*UP)
        self.play(Write(texts))
        self.wait(0.5)
        self.play(FadeOut(texts, shift=UP))
        self.wait()

        texts = Tex(r"A quick forewarning, the following animations are not 100\% accurate!!!").move_to(3*UP).scale(0.8)
        self.play(Write(texts))
        self.wait(0.5)
        self.play(FadeOut(texts, shift=UP))


        self.play(Transform(function_text, Tex("$f(z) = z^{1.5}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8)), 
                  Transform(frame, SurroundingRectangle(Tex("$f(z) = z^{1.5}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8), color=segments_color)))

        f = lambda z: z**(1.5)
        self.play(theta.animate.set_value(2*PI), run_time = 2)
        theta.set_value(0)

        self.play(Transform(function_text, Tex("$f(z) = z^{2.5}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8)), 
                  Transform(frame, SurroundingRectangle(Tex("$f(z) = z^{2.5}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8), color=segments_color)))

        f = lambda z: z**(2.5)
        self.play(theta.animate.set_value(2*PI), run_time = 2)
        theta.set_value(0)


        self.play(Transform(function_text, Tex("$f(z) = z^{3}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8)), 
                  Transform(frame, SurroundingRectangle(Tex("$f(z) = z^{3}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8), color=segments_color)))

        f = lambda z: z**(3)
        self.play(theta.animate.set_value(2*PI), run_time = 2)
        theta.set_value(0)

        self.play(Transform(function_text, Tex("$f(z) = z^{3.5}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8)), 
                  Transform(frame, SurroundingRectangle(Tex("$f(z) = z^{3.5}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8), color=segments_color)))

        f = lambda z: z**(3.5)
        self.play(theta.animate.set_value(2*PI), run_time = 2)
        theta.set_value(0)

        self.play(Transform(function_text, Tex("$f(z) = z^{4}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8)), 
                  Transform(frame, SurroundingRectangle(Tex("$f(z) = z^{4}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8), color=segments_color)))

        f = lambda z: z**(4)
        self.play(theta.animate.set_value(2*PI), run_time = 2)
        theta.set_value(0)

        texts = Tex(r"Enough cardioids! let's try something... complex!").move_to(3*UP)
        self.play(Write(texts))
        self.wait(0.5)
        self.play(FadeOut(texts, shift=UP))

        self.play(Transform(function_text, Tex("$f(z) = z^{i}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8)), 
                  Transform(frame, SurroundingRectangle(Tex("$f(z) = z^{i}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8), color=segments_color)))

        f = lambda z: z**(1j)
        self.play(theta.animate.set_value(2.00111*PI), run_time = 2)
        theta.set_value(0)

        self.play(Transform(function_text, Tex("$f(z) = z^{-1}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8)), 
                  Transform(frame, SurroundingRectangle(Tex("$f(z) = z^{-1}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8), color=segments_color)))

        f = lambda z: z**(-1)
        self.play(theta.animate.set_value(2.0001*PI), run_time = 2)
        theta.set_value(0)

        self.play(Transform(function_text, Tex("$f(z) = z^{1+i}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8)), 
                  Transform(frame, SurroundingRectangle(Tex("$f(z) = z^{1+i}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8), color=segments_color)))

        f = lambda z: z**(1+1j)
        self.play(r.animate.set_value(1.2006), theta.animate.set_value(-PI), run_time = 2)
        self.play(r.animate.set_value(0.7), theta.animate.set_value(PI), run_time = 2)
        self.play(r.animate.set_value(1))
        self.play(theta.animate.set_value(2*PI), run_time = 2)
        theta.set_value(0)


        self.play(Transform(function_text, Tex("$f(z) = e^{z}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8)), 
                  Transform(frame, SurroundingRectangle(Tex("$f(z) = e^{z}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8), color=segments_color)))
        f = lambda z: e**(z)
        self.play(theta.animate.set_value(2*PI), run_time = 2)
        theta.set_value(0)

        self.play(Transform(function_text, Tex("$f(z) = z^{z}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8)), 
                  Transform(frame, SurroundingRectangle(Tex("$f(z) = z^{z}$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8), color=segments_color)))
        f = lambda z: z**(z)
        self.play(theta.animate.set_value(2*PI), run_time = 2)
        theta.set_value(0)

        self.play(Transform(function_text, Tex(r"$f(z) = \log(z)$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8)), 
                  Transform(frame, SurroundingRectangle(Tex(r"$f(z) = \log(z)$").move_to(5.4*LEFT + 0.5*DOWN).scale(0.8), color=segments_color)))

        f = lambda z: log(z)
        self.play(theta.animate.set_value(2*PI-0.1), run_time = 2)
        theta.set_value(0)

        texts = Tex(r"This last one suggests an interesting fact...").move_to(3*UP)
        self.play(Write(texts))
        self.wait(0.5)
        self.play(FadeOut(texts, shift=UP))

        self.wait()



        # manim -pqh discord.py ComplexSequences



class myDot(Scene):
    def construct(self):
        squares = [Square(side_length=1).move_to([x,0,0]) for x in [-5,-3,-1]]

        for square in squares:
            self.play(Create(square))  
        
        arcAngles = [ValueTracker(0*DEGREES) for i in range(len(squares)-1)]

        arcs = []

        arcs.append(always_redraw(lambda: Arc(
                start_angle = 0,
                arc_center = (squares[0].get_bottom()+squares[1].get_bottom())/2,
                angle = arcAngles[0].get_value(),
                radius = (squares[0].get_bottom()[0]-squares[1].get_bottom()[0])/2
            )
        ))

        arcs.append(always_redraw(lambda: Arc(
                start_angle = 0,
                arc_center = (squares[1].get_bottom()+squares[2].get_bottom())/2,
                angle = arcAngles[1].get_value(),
                radius = (squares[1].get_bottom()[0]-squares[2].get_bottom()[0])/2
            )
        ))
        for arc in arcs:
            self.add(arc)

        dot = always_redraw(lambda: Dot().move_to(arcs[0].get_end()))
        self.add(dot)
        self.play(arcAngles[0].animate.set_value(180*DEGREES),run_time=2)
        self.remove(dot)
        dot = always_redraw(lambda: Dot().move_to(arcs[1].get_end()))
        self.add(dot)
        self.play(arcAngles[1].animate.set_value(180*DEGREES),run_time=2)
        self.wait()
        self.play(arcAngles[1].animate.set_value(0*DEGREES),run_time=2)
        self.remove(dot)
        dot = always_redraw(lambda: Dot().move_to(arcs[0].get_end()))
        self.add(dot)
        self.play(arcAngles[0].animate.set_value(0*DEGREES),run_time=2)

        self.wait(2)


        # manim -pqh discord.py myDot



def inversion_homotopy(x,y,z,t):

    a = 1/(x*x+y*y)

    x_ = (1+(a-1)*t)*x
    y_ = (1+(a-1)*t)*y

    if x_ > 10000000:
        x_ = 10000000*x
    if y_ > 10000000:
        y_ = 10000000*y

    return (x_,y_,z)

def mobius(x,y,a,b,c,d):
    z = x + y * 1j
    z_ = (a*np.conj(z)+b)/(c*np.conj(z)+d)
    return z_.real, z_.imag

def mobius_homotopy_1(x,y,z,t):
    
    a=0.01+0.02j
    b=0.9+0.01j
    c=1
    d=0.03+0.07j

    # z -> (az+b)/(cz+d)
    
    a_ = (1-t/4)*1 + t*a/4 # a_: 1 to a
    b_ = (1-t/4)*0 + t*b/4 # b_: 0 to b
    c_ = (1-t/4)*0 + t*c/4 # c_: 0 to c
    d_ = (1-t/4)*1 + t*d/4 # d_: 1 to d
    x_, y_ = mobius(x,y,a_,b_,c_,d_)
    return (x_, y_, z)

def mobius_homotopy_2(x,y,z,t):
    
    a=0.91-0.07j
    b=0.10+0.03j
    c=0.01-1.05j
    d=0.04-0.06j

    # z -> (az+b)/(cz+d)
    
    a_ = (1-t/4)*1 + t*a/4 # a_: 1 to a
    b_ = (1-t/4)*0 + t*b/4 # b_: 0 to b
    c_ = (1-t/4)*0 + t*c/4 # c_: 0 to c
    d_ = (1-t/4)*1 + t*d/4 # d_: 1 to d
    x_, y_ = mobius(x,y,a_,b_,c_,d_)
    return (x_, y_, z)


class invert(Scene):

    def construct(self):

        unit_circle = Circle(fill_opacity = 0, radius = 1)

        #self.add(unit_circle)
        #self.play(Create(unit_circle),run_time = 2)
        
        circles = []

        '''for c in circles:
                c.insert_n_curves(10)'''

                
        circles.append(Circle(fill_opacity = 0, radius = np.sqrt(2)/2))
        circles.append(Circle(fill_opacity = 0, radius = np.sqrt(2)/2))
        circles.append(Circle(fill_opacity = 0, radius = np.sqrt(2)/2))
        circles.append(Circle(fill_opacity = 0, radius = np.sqrt(2)/2))

        circles[0].move_to([-1, 0,0])
        circles[1].move_to([ 0,-1,0])
        circles[2].move_to([ 1, 0,0])
        circles[3].move_to([ 0, 1,0])

        self.add(
            circles[0],
            circles[1],
            circles[2],
            circles[3]
        )
        self.play(
            FadeIn(circles[0]),
            FadeIn(circles[1]),
            FadeIn(circles[2]),
            FadeIn(circles[3]),
            run_time = 5
        )
        self.wait()
        self.play(
            Homotopy(homotopy = mobius_homotopy_1, mobject = circles[0]),
            Homotopy(homotopy = mobius_homotopy_1, mobject = circles[1]),
            Homotopy(homotopy = mobius_homotopy_1, mobject = circles[2]),
            Homotopy(homotopy = mobius_homotopy_1, mobject = circles[3]),
            run_time = 5)
        self.wait()


        

        # manim -pqh discord.py invert




import scipy as sp
class Counter(Scene):
    def construct(self):
        def transition(t):
            return (sp.special.erf(t)+1)/2
        def fading(t):
            return transition(20*(t-0.1))*transition(-20*(t-0.9))

        func = lambda pos: ((pos[0] * UR + pos[1] * LEFT) - pos) / 3
        streamer = StreamLines(
            func,
            x_range=[-4,4,1],
            y_range=[-4,4,1],
        ) 
        self.add(streamer)   
        dot = Dot().set_color(RED)
        self.add(dot)
        self.wait(1)
        dots = VGroup()
        for streamline in streamer.stream_lines:
            dot = always_redraw(lambda: Dot())
            dot.pprop = np.random.random()
            dot.clr = random_color()
            dot.move_to(streamline.point_from_proportion(0))
            dots.add(dot)

        self.play(Create(dots))
        def scene_updater(dt):
            for i, streamline in enumerate(streamer.stream_lines):
                dots[i].pprop = (dots[i].pprop + dt/2) % 1.0
                dots[i].move_to(streamline.point_from_proportion(dots[i].pprop))
                dots[i].set_color(dots[i].clr).set_opacity(fading(dots[i].pprop))
        self.add_updater(scene_updater)

        self.wait(4)


        # manim -pqh discord.py Counter




class squares2(Scene):
    def construct(self):
        field = np.random.rand(6,6)
        element_height = 8/(len(field)*1.1)

        squares = VGroup(
            *[Square(
                side_length=element_height, 
                stroke_width=0, 
                fill_opacity=1).set_color(rgb_to_color([field[y,x],0,0])).move_to([x*(element_height+0.1)-4, 4-y*(element_height+0.1), 0],UL)                
                for y in range(field.shape[0]) for x in range(field.shape[1]) 
            ]
        )
        self.play(Create(squares))
        self.wait(2)
        self.play(Rotate(squares, angle=2*PI))                
        self.wait(2)


        # manim -pqh discord.py squares2


class Rotation3DEx(ThreeDScene):
    def construct(self):
        cube = Cube(side_length=3, fill_opacity=1)

        a=MathTex(r"\mathbb{R}^{3}").shift(2*UR)

        self.begin_ambient_camera_rotation(rate=0.3)

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        self.play(Write(cube), run_time=2)
        self.play(
            Write(a)
        )

        self.wait(3)

        self.play(Unwrite(cube), run_time=2)


        # manim -pqh discord.py Rotation3DEx




class TwoToTheZ(Scene):
    def construct(self):
        #self.camera.background_color = "#334466"

        axes = (
            Axes(
                x_range=[-3.14, 5.50, 1],
                y_range=[-2.43, 2.43, 1],
                x_length=16,
                y_length=9,
                axis_config={"include_ticks": False}
            )
        )

        line_1 = Line(axes.c2p(0, 0), axes.c2p(6, 0), color=RED, stroke_width=2)
        line_2 = line_1.copy()

        dots = {}
        labels = {}
        for a in range(-1, 3):
            for b in range(-4, 5):
                num = np.exp(np.log(2) * (a + b*1j))
                dots[a + b*1j] = (
                    Dot(axes.c2p(num.real, num.imag), color=YELLOW)
                    .scale(2**(a-0.5))
                    .set_z_index(1)
                )
                shift = {
                    -4: 0.3*LEFT + 0.45*DOWN,
                    -3: 0.35*RIGHT + 0.45*DOWN,
                    -2: 0.6*RIGHT + 0.3*DOWN,
                    -1: 0.6*RIGHT + 0.1*UP,
                    0: 0.3*RIGHT + 0.25*DOWN,
                    1: 0.6*RIGHT + 0.1*UP,
                    2: 0.6*RIGHT + 0.3*UP,
                    3: 0.35*RIGHT + 0.45*UP,
                    4: 0.3*LEFT + 0.45*UP
                }

                text = "2^{"
                # Concat a
                if a != 0 or b == 0:
                    text += str(a)
                    if b > 0:
                        text += "+"

                # Concat b
                if b == 0:
                    text += "}"
                elif b == 1:
                    text += "i}"
                elif b == -1:
                    text += "-i}"
                else:
                    text += str(b) + "i}"

                labels[a + b*1j] = (
                    MathTex(text)
                    .scale(2**(a-0.5))
                    .move_to(dots[a + b*1j].get_center() + (2**a)*shift[b])
                    .set_z_index(1)
                )
                if a == 0:
                    labels[b*1j].scale(1.3).set_color(RED)

        self.play(DrawBorderThenFill(axes))
        self.play(
            *[DrawBorderThenFill(dots[a]) for a in range(-1, 3)],
            *[Write(labels[a]) for a in range(-1, 3)]
        )
        self.wait()

        def vector_sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def vector_smooth(t, inflection = 10.0):
            error = vector_sigmoid(-inflection / 2)
            return np.minimum(
                np.maximum((vector_sigmoid(inflection * (t - 0.5)) - error) / (1 - 2 * error), 0),
                1,
            )

        run_time = 6

        fps = config.frame_rate
        frames = int(round(run_time, 1) * fps)

        linspace = np.linspace(0, 1, frames + 1)
        angles = vector_smooth(linspace) * np.pi

        ln_2 = np.log(2)
        wait_times = [np.where(angles >= k * ln_2)[0][0] / fps for k in range(1, 5)]

        circle_animations = [
            AnimationGroup(
                # Generating lines
                Rotate(line_1, angle=PI, about_point=axes.c2p(0, 0)),
                Rotate(line_2, angle=-PI, about_point=axes.c2p(0, 0)),

                # Circles other than unit circle
                *[Create(Arc(
                    radius=(axes.c2p(2 ** k, 0)[0] - axes.c2p(0, 0)[0]),
                    angle=PI,
                    arc_center=axes.c2p(0, 0),
                    stroke_width=1,
                    stroke_color=BLUE
                )) for k in (-5, -4, -3, -2, -1, 1, 2)],
                *[Create(Arc(
                    radius=(axes.c2p(2 ** k, 0)[0] - axes.c2p(0, 0)[0]),
                    angle=-PI,
                    arc_center=axes.c2p(0, 0),
                    stroke_width=1,
                    stroke_color=BLUE
                )) for k in (-5, -4, -3, -2, -1, 1, 2)],

                # Unit circle
                Create(Arc(
                    radius=(axes.c2p(1, 0)[0] - axes.c2p(0, 0)[0]),
                    angle=PI,
                    arc_center=axes.c2p(0, 0),
                    stroke_width=2,
                    stroke_color=RED
                )),
                Create(Arc(
                    radius=(axes.c2p(1, 0)[0] - axes.c2p(0, 0)[0]),
                    angle=-PI,
                    arc_center=axes.c2p(0, 0),
                    stroke_width=2,
                    stroke_color=RED
                )),

                run_time=run_time
            )
        ]

        line_animations = [
            AnimationGroup(
                # Radial lines
                Create(
                    Line(axes.c2p(0, 0), axes.c2p(5, 0), stroke_width=1, stroke_color=BLUE)
                    .rotate(b * ln_2, about_point=axes.c2p(0, 0)),
                    run_time=0.1
                ),
                Create(
                    Line(axes.c2p(0, 0), axes.c2p(5, 0), stroke_width=1, stroke_color=BLUE)
                    .rotate(-b * ln_2, about_point=axes.c2p(0, 0)),
                    run_time=0.1
                ),

                # Dots and labels
                *[DrawBorderThenFill(dots[a + b*1j]) for a in range(-1, 2)],
                *[Write(labels[a + b*1j]) for a in range(-1, 2)],
                *[DrawBorderThenFill(dots[a - b*1j]) for a in range(-1, 2)],
                *[Write(labels[a - b*1j]) for a in range(-1, 2)],
            )
            for b in range(1, 5)
        ]

        self.play(Create(line_1), Create(line_2), run_time=0.5)
        self.play(
            AnimationGroup(
                *circle_animations,
                *[Succession(Wait(wait_times[i]), line_animations[i]) for i in range(4)],
                lag_ratio=0
            ),
            run_time=run_time
        )
        self.play(Uncreate(line_1), Uncreate(line_2), run_time=0.5)
        self.wait(2)


        # manim -pqh discord.py TwoToTheZ



class Diagonal(Scene):
    def construct(self):
        grid = MathTex(
            r"""
            \begin{array}{c c c c c c}
                0 & \rightarrow & 0 & 1 & 0 & \cdots \\
                1 & \rightarrow & 0 & 1 & 1 & \cdots \\
                2 & \rightarrow & 1 & 1 & 0 & \cdots \\
                \vdots & \vdots & \vdots & \vdots & \vdots & \ddots 
            \end{array}
            """ 
        ).set_color(BLUE).scale(2)
        for i in [0, 8, 16, 24, 25, 26]:
            grid[0][i].set_color(YELLOW)

        for i in [1, 9, 17, 27, 28, 29]:
            grid[0][i].set_color(GREEN)

        for i in [2, 11, 20, 39, 40, 41]:
            grid[0][i].set_color(RED)

        self.play(
            Write(grid)
        )
       # self.add(get_submobject_index_labels(grid[0]))
        self.wait()


        # manim -pqh discord.py Diagonal


class Test3D(ThreeDScene):
    def construct(self):
        c = Cube()
        t = Text("Hi cube!").next_to(c, UP)
        self.play(Create(c), Write(t))
        self.add_fixed_orientation_mobjects(t)
        self.move_camera(phi=PI/3, theta=PI/3, run_time=3)
        self.wait()


        # manim -pqh discord.py Test3DEx


class Test3Dex(ThreeDScene):
    def construct(self):
        c = ThreeDAxes()
        
        x = Text("x").next_to(c, RIGHT)
        y = Text("y").next_to(c, UP)
        #z = Text("Hi cube!").next_to(c, UP)
        
        self.play(Create(c), Write(x), Write(y))
        
        self.add_fixed_orientation_mobjects(x, y)
        
        self.move_camera(phi=PI/3, theta=PI/3, run_time=3)
        
        self.wait()

        # manim -pqh discord.py Test3DEx




class cosineIter(Scene):
  def construct(self):
    axes = Axes(x_range = [0, 2.5, 0.5],
                y_range = [0, 1, 0.2],
                x_axis_config = {"numbers_to_include": np.arange(0,2.5,0.5)},
                y_axis_config = {"numbers_to_include": np.arange(0,1,0.2)})

    axis_labels = axes.get_axis_labels()
      
    ident = axes.plot(lambda x: x, color=PURE_BLUE)
    cos_graph = axes.plot(lambda x: np.cos(x), color=RED)
    
    
    
    graphing_stuff = VGroup(axes, axis_labels)

    self.play(DrawBorderThenFill(axes), Write (axis_labels))
    self.wait()    
    self.play(Create(ident))
    self.wait()
    self.play(Create(cos_graph))
    self.wait()  

    x = 0
    y = 0
    i = 0

    while (i<10):
        y2 = np.cos(x)
        vline = Line(start=axes.c2p(x,y), end=axes.c2p(x,y2), color=YELLOW)
        self.play(Create(vline), run_time=2)
        y = y2
        hline = Line(start=axes.c2p(x,y),end=axes.c2p(y,y), color=BLUE)
        self.play(Create(hline), run_time=2)
        x = y

        i += 1


        # manim -pqh discord.py cosineIter



class weier(Scene):
    def construct(self):
        n = 300
        a = ValueTracker(0.5)
        b = ValueTracker(0.6)
        xrng = ValueTracker(4)

        ax = Axes()
        func = VMobject()
        def axUpdater(mobj):
            xmin = -xrng.get_value()
            xmax = +xrng.get_value()
            newax =Axes(x_range=[xmin,xmax,10**int(np.log10(xmax)-1)],y_range=[-1,4])
            newax.add_coordinates()
            newfunc = newax.plot(
                lambda x: sum([a.get_value()**k*np.cos(b.get_value()**k*PI*x) for k in range(n)]),
                x_range=[xmin,xmax,xrng.get_value()/200],
                use_smoothing=False,
                ).set_color(RED).set_stroke(width=3)
            mobj.become(newax)
            func.become(newfunc)            
        ax.add_updater(axUpdater)

        self.add(ax,func)

        self.play(
            b.animate.set_value(7),
            run_time=2
        )        
        self.wait(2)
        self.play(
            xrng.animate.set_value(0.01),
            run_time=10
        ) 


        # manim -pqh discord.py weier


class RealTest(Scene):
    def construct(self):
        transform_txt = VGroup(*[Text("1"), Text("2"), Text("3"), Text("4")])
        transform_txt.arrange(direction=RIGHT, buff=2)

        wrt_txt = VGroup(*[Text("1"), Text("2"), Text("3"), Text("4")])
        wrt_txt.arrange(direction=ORIGIN)

        print(f"debug: {wrt_txt._original__init__}")

        w = VGroup( *[x for x in wrt_txt[0:2]], *[x for x in wrt_txt[3:1:-1]])
        t = transform_txt[0:2] + transform_txt[3:1:-1]

        print(f"debug 2: {w._original__init__}")

        for wt, tt in zip(w, t):
            self.play(Write(wt))
            self.play(ReplacementTransform(wt, tt))

        self.wait(1)  


        # manim -pqh discord.py  RealTest




class sinex(Scene):
    def construct(self):

        tracker = ValueTracker(0)

        ax = Axes(x_range=[0,2*PI,PI/2], x_length=12,
                y_range=[-1.5,1.5,1], y_length=6,
                y_axis_config={"include_numbers": True},
                tips=False
                )
        
        g = ax.plot(
            lambda x: np.sin(x), x_range=[0,6*PI,PI/2], color=BLUE
                )
        line1 = ax.plot(
            lambda x: 1/2, x_range=[0,6*PI,PI/2]
                )

        self.last_y = g.underlying_function(tracker.get_value()) 

        dot = Dot(ax.c2p(tracker.get_value(),g.underlying_function(tracker.get_value())))
        def dotUpdater(mobj):
            y = g.underlying_function(tracker.get_value())
            dot.move_to(ax.c2p(tracker.get_value(),y))
            if np.sign(y-0.5) != np.sign(self.last_y-0.5):
                self.add(Dot(ax.c2p(tracker.get_value(),y), color=RED))
            self.last_y = y    
        dot.add_updater(dotUpdater)

        self.add(ax, g, dot, line1)
        self.play(tracker.animate.set_value(2*PI),rate_func=linear, run_time=3)
        self.wait()  


        # manim -pqh discord.py sinex


class SN(Scene):
    def construct(self):
        
        template = TexTemplate()       
        template.documentclass = r"\documentclass[tikz]{standalone}"
        template.add_to_preamble(r"\usepackage{tikz}")
        template.add_to_preamble(r"\usetikzlibrary{trees}")
        
        self.add(
            Tex(r"""
\begin{tikzpicture}[level distance=3cm] \tikzstyle{edge from parent}=[-,draw]
        \node {$\Omega$} [clockwise from=25, sibling angle=50]
            child {node {\fbox{$B$}} [clockwise from=15, sibling angle=30]
                child {node {\fbox{$G$}}
                edge from parent node[fill=black, inner sep=1pt] {$\frac{3}{5}$}}
                child {node {\fbox{$\overline{G}$}}
                edge from parent node[fill=black, inner sep=1pt] {$\frac{2}{5}$}}
            edge from parent node[fill=black, inner sep=1pt] {$\frac{1}{3}$}
            }
            child {node {\fbox{$R$}} [clockwise from=15, sibling angle=30]
                child {node {\fbox{$G$}}
                edge from parent node[fill=black, inner sep=1pt] {$0,3$}}
                child {node {\fbox{$\overline{G}$}}
                edge from parent node[fill=black, inner sep=1pt] {$0,7$}}
            edge from parent node[fill=black, inner sep=1pt] {$\frac{2}{3}$}
            };
    \end{tikzpicture}
                """,
                tex_template=template,
                stroke_width=2,
                font_size=50,
                )
            )

            # manim -pqh discord.py SN




class thousand(Scene):
    def construct(self):
        agents = 100
        vg_agents = VGroup(
            *[Circle(radius=np.random.random()*0.5, 
                     fill_opacity=0.8,
                     color=random_bright_color()
                     ).move_to([
                         np.random.normal(0, 3),
                         np.random.normal(0, 2),0]
                         ) for agent in range(agents)]
        )
        self.add(vg_agents)
        
        for t in range(100): 

            for i in range(agents):
                x = np.random.normal(0, 0.3)
                y = np.random.normal(0, 0.2)
                vg_agents[i].shift([x, y, 0])
  
            self.wait(0.1)
        self.remove(vg_agents)
        self.wait(2)

        # manim -pqh discord.py thousand




class circ_sur_ex(Scene):
    
    def construct(self):

        circle = Circle(radius=2, stroke_color = WHITE, stroke_width = 8)
        radius = Line(start=circle.get_center(), end = circle.get_right(), stroke_color = WHITE, stroke_width = 8).set_z_index(2)
        center = Dot(point= circle.get_center(), radius =0.1, color=WHITE)
        angle_1 = Arc(radius = 2, start_angle = 0, angle = 1, arc_center = circle.get_center(), stroke_width = 8).set_z_index(3).set_color(RED)
        angle_1.reverse_direction()
        vgroup = VGroup(radius, circle, center)
        self.add(vgroup)
        self.play(vgroup.animate(rate_func = smooth))
        self.play(Rotate(radius, about_point = circle.get_right(), angle =-PI/2), rate_func = smooth)
        self.wait(2)
        self.play(ReplacementTransform(radius, angle_1),run_tim=4)#
        self.wait(1)    
        colors = [BLUE,YELLOW,GREEN,PURPLE,ORANGE]
        segs = []
        for i in range(5):    
            segs.append(angle_1.copy().set_color(colors[i]))
            self.add(segs[-1])  # [-1] = last element in list
            dir = segs[-1].point_from_proportion(0.5)-center.get_center()
            dir = dir / np.linalg.norm(dir)
            self.play(segs[-1].animate.shift(0.3*dir))
            self.play(Rotate(segs[-1], 1+i, about_point=center.get_center()))
            dir = segs[-1].point_from_proportion(0.5)-center.get_center()
            dir = dir / np.linalg.norm(dir)
            self.play(segs[-1].animate.shift(-0.3*dir))
            self.wait(1)


            # manim -pqh discord.py circ_sur_ex

 
class Combinatorics(Scene):
    def construct(self):
        dots, triangles = VGroup(), VGroup()
        for i in range(6):
            dot = Dot(3 * RIGHT).rotate(i * TAU / 6, about_point=ORIGIN)
            dots.add(dot)
        cols = [BLUE, GREEN, RED, YELLOW, ORANGE]        
        
        for i in range(5):
            for j in range(i, 6):                
                triangle = Polygon(
                    dots[i].get_center(),
                    dots[i + 1].get_center(),
                    dots[j].get_center(),
                ).set_fill(cols[i], 0.2)   
                triangles.add(triangle)
        #self.add(triangles, dots)  
        self.play(
            Write(triangles),
            Write(dots)
        )
        self.wait(3) 

        # manim -pqh discord.py Combinatorics


class Shape_ex(Scene):
    def construct(self):
        vertices = [
            3 * RIGHT,
            2 * UR,
            UP,
            2 * UL,
            3 * LEFT,
            2 * DL,
            DOWN,
            2 * DR,
            3 * RIGHT
        ]
        shape = VMobject().set_points_smoothly(vertices)        
        hole = Circle(0.6).shift(LEFT)
        difference = Difference(shape, hole)
        difference.set_fill(BLUE, 0.3) 
        #self.add(difference)
        self.play(
            Create(shape)
        )
        self.play(
            Create(hole)
        )
        self.play(
            Write(difference)
        )
        self.wait(3)


        # manim -pqh discord.py Shape_ex



class QuadraticBezierSimpleEx(Scene):
    def construct(self):
        curve = Line(ORIGIN, ORIGIN)
        curve.add_quadratic_bezier_curve_to(2 * UR, 3 * RIGHT) 
        self.play(Create(curve))
        self.wait(3)

        # manim -pqh discord.py QuadraticBezierSimpleEx


class BallGetVectorEx(ThreeDScene):
    def construct(self):
        Faces = [RegularPolygon(n=5).set_fill(WHITE,1.0) for i in range(6)]
        Faces[0].set_fill(RED)
        Faces[2].set_fill(GREEN)
        Dode = VGroup(*Faces)
        self.play(*[Create(F) for F in Faces])
        for i in range(1,6):
            P = Faces[i]
            V = P.get_vertices()
            self.play(P.animate.rotate(
                        angle = PI - math.atan(2),
                        axis = Line(start = V[i%5],end = V[(i+1)%5]).get_vector(),
                        about_point = V[i%5])
                    )
        self.play(Rotate(Dode,-PI/2,RIGHT,about_point = ORIGIN))
        self.wait()

        # manim -pqh discord.py BallGetVectorEx




class ColoringVectorField(Scene):
    def construct(self):
       
        func = lambda x: x - ORIGIN
        colors = [REANLEA_BLUE_LAVENDER,REANLEA_SLATE_BLUE,REANLEA_AQUA,REANLEA_GREY]
        
        vf = ArrowVectorField(
            func, min_color_scheme_value=2, 
            max_color_scheme_value=10, 
            colors=colors
        )
       
        self.wait()
        self.play(
            Write(vf, run_time=1)
        )
        
        for i in range(280,290):
            self.play(Wiggle(vf[i], scale_value=2, rotation_angle=0.4))             

        for obj in vf:
            self.add(Dot().move_to(obj.get_end()))
        self.wait(2)


        # manim -pqh discord.py  ColoringVectorField


class inside4(Scene):
    def construct(self):

        def is_inside(mobj, point):
            # mobj - a convex polygon
            # point - a coordinate [x,y,z] ignoring the z-component
            inside = True 
            vertices = mobj.copy().force_direction("CCW").get_vertices()
            for i in range(len(vertices)):
                # https://stackoverflow.com/questions/2752725/finding-whether-a-point-lies-inside-a-rectangle-or-not
                # D = (x2 - x1) * (yp - y1) - (xp - x1) * (y2 - y1)
                i2 = (i + 1) % len(vertices)
                D = (vertices[i2][0] - vertices[i][0]) * (point[1] - vertices[i][1]) - (point[0] - vertices[i][0]) * (vertices[i2][1] - vertices[i][1])
                if (D < 0):
                    inside = False  
                    break
            return inside
 
        position_list = [
            [2*np.cos(a*DEGREES),2*np.sin(a*DEGREES),0] for a in np.arange(0,359.9,72)
        ]
        obj = Polygon(*position_list, color=PURPLE_B)

        objs = VGroup(
            obj.copy().set_color(PURPLE).shift(2*LEFT+1*UP),
            obj.copy().set_color(RED).rotate(PI).shift(2*RIGHT+1*DOWN),
        )

        points = VGroup(
            *[Dot().move_to([np.random.uniform(low=-6.5,high=6.5),
                            np.random.uniform(low=-3.5,high=3.5),
                            0]) for i in range(50)]
        )
        self.add(objs,points)
        
        all_inside = False
        iter = 0
        while ((all_inside == False) and (iter < 1000)):
            iter += 1
            all_inside = True
            outsiders = VGroup()
            for point in points:
                inside = False 
                for obj in objs:
                    inside = is_inside(obj, point.get_center())
                    if inside:
                        point.set_color(obj.get_color())
                        break
                if inside == False:
                    all_inside = False
                    point.set_color(YELLOW)
                    outsiders += point                    
            dx = []
            dy = []
            for point in outsiders:
                dx.append(np.random.uniform(low=-1,high=+1))
                if abs(point.get_center()[0]+dx[-1]) >= 7:
                    dx[-1] = -dx[-1]
                dy.append(np.random.uniform(low=-1,high=+1))
                if abs(point.get_center()[1]+dy[-1]) >= 4:
                    dy[-1] = -dy[-1]
            if (all_inside != True):
                self.play(
                    *[point.animate.shift([dx[i],dy[i],0]) for i,point in enumerate(outsiders)], 
                    rate_func=rate_functions.linear,
                    run_time=1/10
                ) 
        self.wait(2) 

        # manim -pqh discord.py inside4



class raster(Scene):
    def construct(self):
        circ = Circle(radius=2)
        self.add(circ)

        sq_length = 0.2
        sq = Square(side_length=sq_length, stroke_width=1)
        def sqUpdater(mobj):
            diff = Intersection(circ,mobj)
            if (len(diff)):
                mobj.set_fill(color=YELLOW, opacity=1)
            else:
                mobj.set_fill(color=BLUE, opacity=0)
        sq.add_updater(sqUpdater)

        grid = VGroup()
        y = -4
        while (y <= (4-sq_length)):
            x = -7
            while (x <= (7-sq_length)):
                grid += sq.copy().move_to([x,y,0],aligned_edge=DL)
                x += sq_length
            y += sq_length
        grid.update()
        self.add(grid)  
        self.wait(2)
        self.play(circ.animate.shift(3*RIGHT+2*UP))
        self.wait(2)

        # manim -pqh discord.py raster


class ColoredCircles(Scene):
    def construct(self):
        circles, numbers = VGroup(), VGroup()
        for _ in range(100):            
            circles.add(Circle(0.23))
        circles.arrange_in_grid(buff=0.18)
        for i in range(100):
            if i < 11: 
                num = Text("24", font_size=14)
            elif 11 <= i < 21:
                num = Text("25", font_size=14)
            else:
                num = Text("26", font_size=14)  
            numbers.add(num.move_to(circles[i]))
        circles[:11].set_stroke(GREEN, 2)
        circles[11:21].set_stroke(BLUE, 2)
        circles[21:].set_stroke(YELLOW, 2)
        #self.add(circles, numbers)
        self.play(
            Write(circles),
            Write(numbers)
        )

        # manim -pqh discord.py ColoredCircles



class HomographyEx(Scene):
    def construct(self):
        img = Square()
        self.play(FadeIn(img))

        H = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0.5, 1],
            ]
        )

        def apply_homography(pt):
            assert pt[2] == 0
            pt[2] = 1
            pt = np.dot(H, pt)
            pt /= pt[2]
            pt[2] = 0
            return pt

        self.play(ApplyPointwiseFunction(apply_homography, img))     

        # manim -pqh discord.py HomographyEx


class cases_ex(Scene):
    def construct(self):
        tex1 = MathTex(r"f(n) = \begin{cases}"
                r"n/2  & n \text{ is even} \\"
                r"3n+1 & n \text{ is odd}"
                r"\end{cases}"
        )
        self.play(Write(tex1))

        # manim -pqh discord.py cases_ex



class impli_fun_area(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-5,5],
            y_range=[-5,5]
        )          
        ax.add_coordinates()

        curve_pos = ax.plot_implicit_curve(
            lambda x,y: (x*y - 1)*(x>0),   # f(x,y) = 0 
        )
        points_pos = curve_pos.get_all_points()
        points_pos = np.append(points_pos, [ax.c2p(x,5) for x in np.linspace(points_pos[-1][0],5,100)], axis=0)

        area_pos = Polygon(
            *points_pos 
        ).set_fill(color=YELLOW,opacity=1)

        curve_neg = ax.plot_implicit_curve(
            lambda x,y: (x*y - 1)*(x<0),   # f(x,y) = 0 
        )
        points_neg = curve_neg.get_all_points()
        points_neg = np.append(points_neg, [ax.c2p(x,-5) for x in np.linspace(points_neg[-1][0],-5,100)], axis=0)

        area_neg = Polygon(
            *points_neg 
        ).set_fill(color=YELLOW,opacity=1)

        self.play(Create(ax))        
        self.play(Create(curve_pos))
        self.play(Create(area_pos))
        self.play(Create(curve_neg))
        self.play(Create(area_neg))
        self.wait(2)


        # manim -pqh discord.py impli_fun_area



class case_tex_ex(Scene):
    def construct(self):
        
        case_tex_1= MathTex(r"\lambda v' + \gamma v = 0 \longrightarrow \begin{cases}"
                r"n/2  & n \text{ is even} \\"
                r"3n+1 & n \text{ is odd}"
                r"\end{cases}"
        )

        self.wait()
        self.play(
            Write(case_tex_1)
        )

        self.wait(2)

        # manim -pqh discord.py case_tex_ex

        # manim -sqk discord.py case_tex_ex




class Derivative(Scene):
    def construct(self):
        k = ValueTracker(0)
        dx = ValueTracker(2)
        axes = Axes(

            x_range=[-1,10,1],
            y_range=[-1,6,1],
            axis_config={"include_numbers":True}
            )
        func = lambda x: -0.1*(x-5)**2 +2
        plot = axes.plot(func,x_range=[-1,9,0.01],color=RED)

        dot = always_redraw(
            lambda: Dot(
                axes.c2p(
                    k.get_value(),
                    func(k.get_value())
                ),
                color=YELLOW
            )
        )
        lines = always_redraw(
            lambda: VGroup(
                axes.get_lines_to_point(dot.get_center())  
            )
        )        
        secant = always_redraw(
            lambda: axes.get_secant_slope_group(
                x=k.get_value(),
                graph = plot,
                dx=dx.get_value(),
            )
        )        

        # Change the line to an arrow
        def upd_line(line):
            arrow = Arrow(
                dot.get_center(),
                secant[1].get_end(),
                buff=0,
                tip_length=0.25,
                color=YELLOW,
            )
            line.become(arrow)
        secant[2][0].add_updater(upd_line)

        velo = Line()
        velolength = 2
        def veloUpdater(mobj):
            x = k.get_value()
            y1 = func(x)
            y2 = func(x + 1e-6)
            mobj.become(
                Line(
                    start=axes.c2p(x,y1),
                    end=axes.c2p(x+velolength,y1 + velolength*1e6*(y2-y1)),
                    color=GREEN
                ).add_tip(at_start=False)
            )
        velo.add_updater(veloUpdater)
        velo.update()

        # Animations
        self.add(axes,plot,dot,secant,lines,velo)
        self.play(
            k.animate.set_value(8),
            run_time=3
        ) 


        # manim -pqh discord.py Derivative



class Hue_Test(Scene):
    def construct(self):
        square = Square().set_stroke(width=25)
        self.add(square)        
        def upd_square(square, dt):  
            hue = (square.get_color().get_hue() + dt/3) % 1     
            square.set_color(Color(hsl=(hue, 1, 0.5)))
        square.add_updater(upd_square)
        self.wait(3)


        # manim -pqh discord.py Hue_Test


class FunctionParts(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0,10,1],
            y_range=[0,10,1],
            tips=False,
        ).add_coordinates()
        def func(x):
            return x**2 if 0 <= x < 1 else x
        graph = ax.plot(func)
        #self.add(ax, graph)

        self.wait()
        self.play(
            Create(ax)
        )
        self.play(
            Create(graph)
        )
        self.wait(2)

        # manim -pqh discord.py FunctionParts



class VecCirc(VGroup):
    def __init__(self, vt, factor, scale, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.vt = vt
        self.circle = Circle(stroke_opacity=0.3).scale(scale)
        self.vec = Vector(UP)
        self.add(self.circle, self.vec)
        self.add_updater(self.update_state)
        self.set_value = self.vt.set_value
        self.get_value = self.vt.get_value

    @staticmethod
    def update_state(instance: "VecCirc", dt):
        instance.vec.put_start_and_end_on(
            instance.circle.get_center(),
            instance.circle.point_from_proportion(instance.get_current_t()),
        )

    def get_attachment_point(self):
        return self.circle.point_from_proportion(self.get_current_t())

    def get_current_t(self):
        return self.vt.get_value() * self.factor % 1


class FourierDraw(Scene):
    def construct(self):
        k = ValueTracker(0)
        degree = 10
        circles = [
            VecCirc(k, np.sin(i), (i + 1) / degree) for i in range(degree)
        ]
        circles.reverse()
        for i, c in enumerate(circles[1:]):
            c.add_updater(lambda x, i=i: x.move_to(circles[i].get_attachment_point()))
        path = TracedPath(
            circles[-1].get_attachment_point, stroke_width=2, #dissipating_time=1
        )

        self.add(*circles, path)

        self.play(k.animate.set_value(4), run_time=10)


        # manim -pqh discord.py FourierDraw




class LineGraphExample(Scene):
    def construct(self):
        x_values = [0, 1, 2, 3, 4, 5]
        y_values = [1, 1.25, 1.5, 2, 2.5, 3.5]
        plane = NumberPlane(
            x_range=(0, 10),
            y_range=(0, 10),
            x_length=5,
            y_length=5,
            axis_config={"include_numbers": True},
        )
        line_graph = plane.plot_line_graph(
            x_values=[0, 1, 2, 3, 4, 5],
            y_values=[1, 1.25, 1.5, 2, 2.5, 3.5],
            line_color=GOLD_E,
            vertex_dot_style=dict(stroke_width=3, fill_color=PURPLE),
            stroke_width=4,
        )
        plane.center()
        self.play(Write(plane))
        line = line_graph.submobjects[0]
        dots = line_graph.submobjects[1:]
        for dot in dots:
            self.play(Write(dot))
        self.play(Write(line))


        # manim -pqh discord.py LineGraphExample



class Focus2Dto3D(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        x_label = axes.get_x_axis_label(Tex("x"))
        y_label = axes.get_y_axis_label(Tex("y"))
        z_label = axes.get_z_axis_label(Tex("z"))

        p1 = axes.coords_to_point(2,2,2)
        proyection = axes.coords_to_point(2,2,0)

        dot = Dot3D(p1,color=MAROON, fill_opacity = 0.5)
        dashed_line = DashedLine(p1, proyection)
        dashed_line.set_color(MAROON_D)
        dashed_line_1 = DashedLine(proyection, p1)
        arrow = Arrow3D(start=axes.coords_to_point(0,0,0), end = p1, resolution = 8,color = ORANGE)
        hipotenusa2 = Line3D(start = axes.coords_to_point(0,0,0), end = proyection, color = PURPLE_D)
        hipotenusa1 = Line3D(start = axes.coords_to_point(0,0,2), end = p1, color = PURPLE_D)
        # Here
        self.set_camera_orientation(focal_distance=500)
        self.move_camera(phi = 75*DEGREES, theta = 25*DEGREES)
        self.add(axes, dashed_line, hipotenusa2, hipotenusa1, arrow, x_label, y_label,z_label)
        self.move_camera(phi = 90*DEGREES, theta = 0*DEGREES)
        self.wait()


        # manim -pqh discord.py Focus2Dto3D



class Project_Condition(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-1,5],
            y_range=[-1,5]
        )
        self.add(ax)

        NUMPOINTS = 5

        alpha = ValueTracker(70*DEGREES)
        line = always_redraw(lambda:
            Line(
                start=ax.c2p(0,0),
                end=ax.c2p(10*np.cos(alpha.get_value()),10*np.sin(alpha.get_value()))
            )
        )
        points = VGroup(
            *[Dot().move_to(ax.c2p(np.random.uniform(0,5),np.random.uniform(0,5))) for i in range(NUMPOINTS)],
            *[Line() for i in range(NUMPOINTS)]
        )
        def pointsUpdater(mobj):
            for i in range(NUMPOINTS):
                end = line.get_projection(mobj[i].get_center())
                mobj[NUMPOINTS+i].become(
                    Line(
                        start = mobj[i].get_center(),
                        end = end
                    )
                )
        points.add_updater(pointsUpdater)
        self.add(line, points)
        self.play(
            alpha.animate.set_value(90*DEGREES)
        )


        # manim -pqh discord.py Project_Condition




class rollOut(Scene):                
    def construct(self):
        ang = ValueTracker(0)
        circumferenceDifference = 3
        radiusDifference = circumferenceDifference/(2*PI)
        rad1 = 1
        rad2 = rad1 + radiusDifference

        arc1 = Circle(radius=rad1).move_to([-4,0,0], aligned_edge=DOWN)
        def arcUpdater1(mobj): 
            arc = Arc(
                    radius = rad1,
                    start_angle=ang.get_value(),
                    angle=2*PI-ang.get_value()
            ).rotate(1.5*PI-ang.get_value())
            arc.shift((rad1*ang.get_value() - arc.get_start()[0] - 4)*RIGHT - arc.get_start()[1]*UP)
            mobj.become(arc)
        arc1.add_updater(arcUpdater1)
        line1 = always_redraw(lambda:
                Line(
                    start=[-4,0,0],
                    end=[rad1*ang.get_value() - 4,0,0]
                )
        )

        arc2 = Circle(radius=rad2).move_to([-4,-radiusDifference,0], aligned_edge=DOWN)
        def arcUpdater2(mobj): 
            arc2 = Arc(
                    radius = rad2,
                    start_angle=ang.get_value(),
                    angle=2*PI-ang.get_value()
            ).rotate(1.5*PI-ang.get_value())
            arc2.shift((rad2*ang.get_value() - arc2.get_start()[0] - 4)*RIGHT - (arc2.get_start()[1]+radiusDifference)*UP)
            mobj.become(arc2)
        arc2.add_updater(arcUpdater2)
        line2 = always_redraw(lambda:
                Line(
                    start=[-4,-radiusDifference,0],
                    end=[rad2*ang.get_value() - 4,-radiusDifference,0]
                )
        )

        self.add(arc1,line1,arc2,line2)

        self.play(
            ang.animate.set_value(2*PI),
            rate_func = rate_functions.linear,
            run_time = 2
        )
        self.wait(1)

        self.play(
            ang.animate.set_value(0),
            run_time = 2
        )


        # manim -pqh discord.py rollOut





class InwardVector(Scene):
    #config.background_color = WHITE
    #Mobject.set_default(color=BLACK)
    #Tex.set_default(color=BLACK)
    def construct(self):

        ax = Axes(   
            x_range=[-7,7,2],         
            y_range=[-7,7,2],         
            x_length=7,
            y_length=7,
        )
        labels = ax.get_axis_labels(
            x_label=Tex("$x_1$"), 
            y_label=MathTex("x_2")
        )
        
        self.play(FadeIn(ax))
        self.play(Write(labels))

        eigenv1=VGroup()
        eigenv2=VGroup()

        for i in np.arange(0,6): 
            x=1
            y=1

            vec1=Arrow(ax.c2p(x*(i),y*(i)),ax.c2p(x*(i+1),y*(i+1)),buff=0)
            eigenv1.add(vec1)

            vec1=Arrow(ax.c2p(x*(-i),y*(-i)),ax.c2p(x*(-i-1),y*(-i-1)),buff=0)
            eigenv1.add(vec1)

            x=-1
            y=2

            vec2=Arrow(ax.c2p(x*(i+1),y*((i+1)/2)),ax.c2p(x*i,y*(i/2)),buff=0)
            eigenv2.add(vec2)

            vec2=Arrow(ax.c2p(x*(-i-1),y*((-i-1)/2)),ax.c2p(-x*i,-y*(i/2)),buff=0)
            eigenv2.add(vec2)


        self.play(Create(eigenv1))    
        self.play(Create(eigenv2))

        self.wait(2)  


        # manim -pqh discord.py InwardVector





class SortAlgorithm(Scene):
    def construct(self):
        arr = [1, 3, 4, 5, 7, 8]
        group = VGroup()        
        for s in arr:
            group.add(Square(1).add(Text(str(s))))
        group.arrange()     
        self.play(Write(group), run_time=2)
        self.wait()    
        box = SurroundingRectangle(group[:2], color=YELLOW)
        self.play(DrawBorderThenFill(box))    
        for i in range(4):
            self.play(box.animate.shift(1.25 * RIGHT))
            self.wait(0.25)
        
        # Use move_to
        self.play(box.animate.move_to(group[2:4]))
        self.wait()
        self.play(box.animate.move_to(group[:2]))
        self.wait()

        # Use strech
        self.play(box.animate.stretch(1.5, 0).move_to(group[3:6]))
        self.wait(3)


        # manim -pqh discord.py SortAlgorithm




class SecantAnimation(Scene):
    def construct(self):  
        axes = Axes([-5, 5, 1], [-100, 100, 25], 12, 6)
        axis_labels = axes.get_axis_labels(x_label="x", y_label="y")        
        def curve(x):
            return x ** 3 - 3 * x + 1        
        graph = axes.plot(curve, [-5, 5], color=BLUE)
        x = ValueTracker(2.5)
        dx = ValueTracker(2)        

        secant = always_redraw(
            lambda: axes.get_secant_slope_group(   
                x.get_value(),
                graph,             
                dx.get_value(),                
                YELLOW,
                ORANGE,
                "dx",                
                "dy",
                GREEN, 
                secant_line_length=axes.height * 0.9,                
            )
        )

        dot1 = always_redraw(
            lambda: Dot()
            .move_to(axes.c2p(x.get_value(), graph.underlying_function(x.get_value())))
        )
        dot2 = always_redraw(
            lambda: Dot()
                .move_to(axes.c2p(x.get_value() + dx.get_value(),
                graph.underlying_function(x.get_value() + dx.get_value())))
        )
        scn = VGroup(axes, axis_labels, graph, secant, dot1, dot2)        
        
        # Animations           
        self.add(scn)
        self.play(dx.animate.set_value(0.001), run_time=2)
        self.play(FadeOut(dot2), run_time=0.5)
        self.play(x.animate.set_value(-2.5), run_time=3)
        self.wait()
        self.play(x.animate.set_value(4.5), run_time=2)       
        
        secant = None
        dot1 = lambda: Dot().move_to(axes.c2p(x.get_value(), graph.underlying_function(x.get_value())))
        dot2 = None
        self.play(scn.animate.scale(0.25))
        self.wait()

        box = SurroundingRectangle(scn, buff=SMALL_BUFF)
        boxedScn = VGroup(scn, box)        
        self.play(Create(box))
        self.play(boxedScn.animate.to_edge(DL))

        self.wait(3)


        # manim -pqh discord.py SecantAnimation



class sincircle(ThreeDScene):
    def construct(self):
        k=0.6
        cr = ParametricFunction(
            lambda t: np.array([
                2*k*np.cos(t),
                2*k*np.sin(t),
                k*t
            ]), color=RED, t_range = np.array([-TAU, TAU, k])
        ).set_shade_in_3d(True)
        ax = ThreeDAxes()
        self.set_camera_orientation(phi=TAU/4, theta=-TAU/4)
        self.add(ax)
        self.play(Create(cr))
        self.wait()
        self.move_camera(phi=TAU/6, theta=-TAU/6, run_time=2)
        self.move_camera(phi=0, theta=-TAU/4, run_time=2)
        self.wait(2)


        # manim -pqh discord.py sincircle




class ExpEx01(Scene):
    def construct(self):

        def SternsDiatomicSeries(n:int):
          DP = [0]*(n+2)
          DP[0] = 0
          DP[1] = 1

          for i in range(2, n+1):
            if int(i % 2) == 0:
              DP[i] = DP[int(i/2)]
            else:
              DP[i] = DP[int((i-1)/2)] + DP[int((i+1)/2)]

          return DP[n]

        ax = Axes(
            x_range = [0, 1024, 256],
            y_range = [0, 84, 21],
            tips = False,
            axis_config = {
                "include_numbers": True,
                "decimal_number_config": {
                    "group_with_commas": False,
                    "num_decimal_places": 0,
                    },
            }          
        )
        
        lab = ax.get_axis_labels(
            x_label = MathTex("n"),
            y_label = MathTex("a_n")
        )

        SDSg = VGroup(*[
            Dot(ax.coords_to_point(n, SternsDiatomicSeries(n)), radius = 0.02, color = BLUE)
            for n in range(1024)
        ])

        #self.add(SDSg, ax, lab)

        self.play(
            Write(SDSg),
            run_time=6
        )


        # manim -pqh discord.py ExpEx01


class Optics(Scene):
    def construct(self):

        line = Line()
        dot_1, dot_2 = Dot(LEFT), Dot(RIGHT)        
        
        self.add(line, dot_1, dot_2)
        self.play(
            AnimationGroup(
                Flash(dot_1),
                *[MoveAlongPath(dot_1.copy(), line) for _ in range(10)],
                Flash(dot_2, rate_func=lambda t: smooth(1-t)),
                lag_ratio=0.1,
            )
        )


        # manim -pqh discord.py Optics



class Sin1byx(Scene):
    def construct(self):
        continuousAxes=Axes(
            x_range=[-10, 10, 2],
            x_length=5,
            color=BLUE,
            y_range=[-5, 5, 1],
            y_length=3.5,
            axis_config={
                "tip_width" :0.15,
                "tip_height": 0.15
            })
        wobbly = continuousAxes.plot(
            lambda x: 6*math.sin(3/x),
            x_range=[-10, 10, 0.001],
            color=PURPLE_B,
        )
        self.add(continuousAxes)
        self.play(Create(wobbly))     
        self.wait(2) 


        # manim -pqh discord.py Sin1byx


class boxMove(Scene):
    def construct(self):
        boxes = VGroup()
        for y in range(24):
            for x in range(25):
                sq = Square(side_length=0.2, fill_opacity=1).move_to([-6+0.2*x,3.5-0.2*y,0])
                boxes.add(sq)
        self.play(Create(boxes),run_time=6)
        self.wait(6)
        a =[11]*24+[12]*28
        
        b = np.random.permutation(a)

        index = 0
        rows = VGroup()
        for row in range(len(b)):
            color = random_bright_color()
            rowOfBoxes = VGroup()
            for i in range(b[row]):
                boxes[index].set_color(color)
                rowOfBoxes.add(boxes[index])
                index += 1
            rows += rowOfBoxes
        self.wait(2)

        ycount = int(len(rows)/2)
        for col in range(2):
            for y in range(ycount):
                self.play(
                    rows[col*ycount+y].copy().animate.arrange(RIGHT,buff=0).move_to([col*3+0,3.5-0.2*y,0],aligned_edge=LEFT)
                )


        
        # manim -pqh discord.py boxMove




from manim import *
#from numba import njit
from cmath import phase
from typing import Sequence, Callable

#@njit
def G(k: int, tau: complex, max_term_order: int = 10):
    lattice_values = np.array([(m, n) for m in range(1, max_term_order + 1) for n in range(1, max_term_order + 1)])
    return 1 + 1 / tau ** (2 * k) + np.sum(np.array([1 / (value[0] + value[1] * tau) ** (2 * k) for value in lattice_values]))

#@njit
def plot_eisenstein_series(
        k: int,
        max_term_order: int = 10,
        matrix_for_action: np.ndarray = np.eye(2),
        real_range: Sequence = np.array([-1.5, 1.5, 3 / 1000]),
        imag_range: np.ndarray = np.array([0.01, 1.5, 1.5 / (1000 * 1080 / 1920)])
    ) -> np.ndarray:
    x_min, x_max, dx = real_range
    y_min, y_max, dy = imag_range
    n_x, n_y = int(np.abs(x_max - x_min) / dx), int(np.abs(y_max - y_min) / dy)
    res_array = np.zeros((n_y, n_x, 3), dtype="uint8")
    a, b, c, d = matrix_for_action[0, 0], matrix_for_action[0, 1], matrix_for_action[1, 0], matrix_for_action[1, 1]

    for n in range(len(res_array)):
        for p in range(len(res_array[0])):
            z = complex(x_min + p * dx, y_max - n * dy)
            matrix_acting_on_z = (a * z + b) / (c * z + d)
            
            lattice_values = np.array([(m, n) for m in range(-max_term_order, max_term_order + 1) for n in range(-max_term_order, max_term_order + 1)])
            f_z = 1 / (c * z + d) ** (2 * k) * np.sum(np.array([1 / (value[0] + value[1] * matrix_acting_on_z) ** (2 * k) for value in lattice_values if value[0] != 0 or value[1] != 0]))

            h = (phase(f_z) + PI) / (2 * PI)
            l = (0.05 * np.log(np.abs(f_z)) / (1 + 0.05 * np.abs(np.log(np.abs(f_z)))) + 1) / 2

            H = int(360 * h)
            D = 1 - np.abs(2 * l - 1)
            m = 255 * (l - D / 2)
            x = D * (1 - np.abs((H / 60) % 2 - 1))

            if 0 <= H < 60:
                R = np.uint8(255 * D + m)
                G = np.uint8(255 * x + m)
                B = np.uint8(m)

            if 60 <= H < 120:
                R = np.uint8(255 * x + m)
                G = np.uint8(255 * D + m)
                B = np.uint8(m)

            if 120 <= H < 180:
                R = np.uint8(m)
                G = np.uint8(255 * D + m)
                B = np.uint8(255 * x + m)

            if 180 <= H < 240:
                R = np.uint8(m)
                G = np.uint8(255 * x + m)
                B = np.uint8(255 * D + m)

            if 240 <= H < 300:
                R = np.uint8(255 * x + m)
                G = np.uint8(m)
                B = np.uint8(255 * D + m)

            if 300 <= H < 360:
                R = np.uint8(255 * D + m)
                G = np.uint8(m)
                B = np.uint8(255 * x + m)

            res_array[n, p] = R, G, B
            #res_array[n, p] = h, l, 1.
    return res_array


class EisensteinSeriesG10(Scene):
    def construct(self):
        a, b, c, d = ValueTracker(1), ValueTracker(0), ValueTracker(0), ValueTracker(1)

        screen_rect = Rectangle(height=config.frame_height)

        im = always_redraw(
            lambda: ImageMobject(plot_eisenstein_series(
                k=10,
                max_term_order=10,
                matrix_for_action=np.array([[a.get_value(), b.get_value()], [c.get_value(), d.get_value()]])
            )).match_height(screen_rect)
        )

        self.add(im)
        self.wait()
        self.play(a.animate.set_value(1), b.animate.set_value(1), c.animate.set_value(1), d.animate.set_value(0), run_time=8, rate_func=linear)
        self.wait()


        # manim -pqh discord.py EisensteinSeriesG10




from manim import *
from random import *
import networkx as nx


class GraphAlgorithm(Scene):
    def construct(self):
        seed(0xDEADBEEF)

        n = 14
        p = 3 / n

        VISITED_COLOR = GREEN
        NEIGHBOUR_COLOR = BLUE

        graph = None
        while graph is None or not nx.is_connected(graph):
            graph = nx.generators.random_graphs.gnp_random_graph(n, p)

        g = (
            Graph(graph.nodes, graph.edges, layout_config={"seed": 0})
            .scale(2.7)
            .rotate(PI / 12)
        )

        # quickfix for a bug in AniomationGroup's handling of z_index
        for v in g.vertices:
            g.vertices[v].set_z_index(1)

        explored = set()

        def dfs(v, position_object):
            """Recursive DFS which moves the position_object."""
            neighbours = list(graph.neighbors(v))

            for w in neighbours:
                if w in explored:
                    continue

                edge = (v, w) if (v, w) in g.edges else (w, v)

                unexplored_neighbours = [w for w in neighbours if w not in explored]
                unexplored_neighbour_edges = [
                    (a, b)
                    for a, b in g.edges
                    if (a == v and b in unexplored_neighbours)
                    or (b == v and a in unexplored_neighbours)
                ]

                # while there exist unexplored neighbours, explore
                if len(unexplored_neighbours) != 0:
                    self.play(
                        *[
                            g.vertices[q].animate.set_color(NEIGHBOUR_COLOR)
                            for q in unexplored_neighbours
                        ],
                        *[
                            g.edges[e].animate.set_color(NEIGHBOUR_COLOR)
                            for e in unexplored_neighbour_edges
                        ],
                    )

                explored.add(w)

                # animation of transitioning to neighbouring vertex
                # has two parts - first initialize the move and then change color (+ flash)
                self.play(
                    AnimationGroup(
                        position_object.animate.move_to(g.vertices[w]),
                        AnimationGroup(
                            Flash(g.vertices[w], color=VISITED_COLOR, flash_radius=0.3),
                            g.edges[edge].animate.set_color(VISITED_COLOR),
                            g.vertices[w].animate.set_color(VISITED_COLOR),
                            *[
                                g.vertices[q].animate.set_color(WHITE)
                                for q in unexplored_neighbours
                                if q != w
                            ],
                            *[
                                g.edges[(a, b)].animate.set_color(WHITE)
                                for (a, b) in unexplored_neighbour_edges
                                if (a, b) != edge
                            ],
                        ),
                        lag_ratio=0.45,
                    )
                )

                dfs(w, position_object)
                self.play(position_object.animate.move_to(g.vertices[v]))

        self.play(Write(g))

        start_vertex = 0

        position_object = (
            Circle(fill_color=VISITED_COLOR, fill_opacity=1, stroke_color=VISITED_COLOR)
            .move_to(g.vertices[start_vertex])
            .scale(0.15)
        )

        self.play(
            Flash(g.vertices[start_vertex], color=VISITED_COLOR, flash_radius=0.3),
            g.vertices[start_vertex].animate.set_color(VISITED_COLOR),
        )

        self.add(position_object)

        # run DFS
        explored.add(start_vertex)
        dfs(start_vertex, position_object)

        self.remove(position_object)
        self.play(Unwrite(g))

        
        ## source --  https://slama.dev/manim/camera-and-graphs/

        # manim -pqh discord.py GraphAlgorithm



class GoursatCurve(ThreeDScene):
    def construct(self):
        
        title = Tex("Goursat Curve").to_edge(UP)
        
        a = ValueTracker(1.12)
        b = ValueTracker(0.74)
        c = ValueTracker(-0.86)             
        d = ValueTracker(0.06)  
        k = ValueTracker(6)
        
        id_to_label = {0:'a', 1:'b',2:'c',3:'d',4:'k'}
        label_to_tracker = {0:a,1:b,2:c,3:d,4:k}
        
        labels = []
        labels_number = []
        
        labels.append(Tex(id_to_label[0]+"= ").to_edge(UL))
        for i in range(1,5):
            labels.append(Tex(id_to_label[i]+"= ").next_to(labels[i-1], DOWN))
            
        for i in range(5):
            labels_number.append(DecimalNumber(label_to_tracker[i].get_value(), color=WHITE, num_decimal_places=2).next_to(labels[i], RIGHT))
            labels_number[i].add_updater(lambda mob,i=i: mob.set_value(label_to_tracker[i].get_value()))
        
        ax = ThreeDAxes(x_range=[-6,6,1], y_range=[-6,6,1])
        self.set_camera_orientation(phi=60*DEGREES)
        
        #self.add(ax)
        
        def func(t):
            x = (np.cos(t) * (a.get_value() + b.get_value() *                        np.cos(k.get_value()*t) + c.get_value() *                                np.cos(2*k.get_value()*t) + d.get_value() *                              np.cos(3*k.get_value()*t)) - np.sin(t) * (b.get_value() *                np.sin(k.get_value()*t) + c.get_value() *         np.sin(2*k.get_value()*t) + d.get_value() * np.sin(3*k.get_value()*t)))

            y = (np.sin(t) * (a.get_value() + b.get_value() * np.cos(k.get_value()*t) + c.get_value() * np.cos(2*k.get_value()*t) + d.get_value() * np.cos(3*k.get_value()*t)) + np.cos(t) * (b.get_value() * np.sin(k.get_value()*t) + c.get_value() * np.sin(2*k.get_value()*t) + d.get_value() * np.sin(3*k.get_value()*t)))
            return [x,y,0]
        
        fplot=ax.plot_parametric_curve(func, t_range=[0,10,0.01], color=ORANGE)
        
        fplot.add_updater(lambda mob:mob.become(ax.plot_parametric_curve(func, t_range=[0,10,0.01], color=ORANGE)))
        self.begin_ambient_camera_rotation(rate = 0.5)   
        
        self.play(Create(fplot, run_time=10,rate_function=linear))
        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES)
        self.stop_ambient_camera_rotation()
        #self.move_camera(phi=75 * DEGREES, theta=-30 * DEGREES)
        self.play(Write(title))
        self.play(Write(VGroup(*labels, *labels_number)))
        self.play(a.animate.set_value(2),run_time=5)



        # manim -pqh discord.py GoursatCurve



class GraphTest_01(Scene):
    def construct(self):
        vertices = [1, 2, 3]
        edges = [(2, 1), (3, 1), (3, 2)]
        g = Graph(vertices, edges,
                  # edge_type=Arrow,
                  # labels=True
                  )
        self.play(Create(g))
        self.wait()
        for _ in range(3):
            vertices = g.vertices
            print(vertices)
            for _ in range(len(vertices)):
                new_edges = [(len(g.vertices)+1, vertex) for vertex in g.vertices]
                print(new_edges)
                g.add_vertices(len(g.vertices)+1, positions={len(g.vertices)+1:[np.random.uniform(-1,1),np.random.uniform(-1,1),0]})
                g.add_edges(*new_edges)
            self.play(g.animate.change_layout("circular"))
            self.wait()


            # manim -pqh discord.py GraphTest_01





class manyNumbers2(Scene):
    def construct(self):
        cols = 20
        rows = 10
        numsize = 30

        numbers = VGroup(
            *[DecimalNumber(
                i+1, 
                font_size=numsize,
                num_decimal_places=0,
            ) for i in range(200)]
        )
        numbers.arrange_in_grid(rows=rows, cols=cols)

        self.play(Create(numbers))
        self.wait()    
        for num in numbers:
            if num.get_value() % 28 == 0:
                self.play(Wiggle(num))
                num.set_color(YELLOW) 
        self.wait() 


        # manim -pqh discord.py manyNumbers2




def disc_func(x):
    return np.cos(x) + 1

class Disc_Func_test(Scene):
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

        #self.add(axes, plot, disc_point)

        self.play(
            Write(axes)
        )
        self.play(
            Create(plot),
            Write(disc_point)
        )

        self.wait(2)


        # manim -pqh discord.py Disc_Func_test



class SampleSpaceLabelEx(Scene):
    def construct(self):
        number = 7
        list_num = [1 / number] * number
        geo = SampleSpace(4, 4, stroke_width=2, fill_opacity=0.5)
        geo.divide_horizontally(p_list=np.array(list_num), colors=[TEAL, TEAL])

        for i in range(number):
            geo[i].set_opacity(0)
            geo[i].divide_vertically(p_list=np.array(list_num), colors=[GREEN_D, GREEN])
            geo[i].label = MathTex(f"{i + 1}").next_to(geo[i], LEFT)     
            self.add(geo[i].label)       
            for j in range(0, i+1): 
                geo[i][j].set_opacity(0.1) 
        self.add(geo)

        self.play(
            Write(geo)
        )

        self.wait(2)


        # manim -pqh discord.py  SampleSpaceLabelEx




class Moving_Triangle_ex(Scene):
    def construct(self):

        circ = Circle(radius=2)
        alpha = ValueTracker(110)

        dotB = Dot(circ.point_at_angle(-160*DEGREES))
        dotC = Dot(circ.point_at_angle(-20*DEGREES))

        dotA =always_redraw(lambda:
        Dot(circ.point_at_angle(alpha.get_value()*DEGREES)))

        tri = always_redraw(lambda: 
        Polygon(dotA.get_center(), dotB.get_center(), dotC.get_center()))
        move = VGroup(dotA, tri)

        self.add(circ,move,dotB,dotC)
        self.play(alpha.animate.set_value(30),run_time=3)
        self.play(alpha.animate.set_value(160),run_time=3)
        self.wait()


        # manim -pqh discord.py Moving_Triangle_ex



class FluidFlowScene(Scene):
    def construct(self):
        k = 1

        title = Text("Idealized Flow Around an Infinite Cylinder")
        eq = MathTex( r"\vec{V}(x,y)=U\left(1-r^2\frac{x^2-y^2}{(x^2+y^2)^2}\right)\hat{i}-2U^2r^2\frac{xy}{(x^2+y^2)^2}\hat{j}")

        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title), run_time=2)

        div_func = lambda pos: (1-1*((pos[0]*pos[0]-pos[1]*pos[1])/((pos[0]*pos[0]+pos[1]*pos[1])*(pos[0]*pos[0]+pos[1]*pos[1]))))*k*RIGHT+(-2*1*((pos[0]*pos[1])/(((pos[0]*pos[0]+pos[1]*pos[1])*(pos[0]*pos[0]+pos[1]*pos[1])))))*UP*k*k
        
        stream_lines = StreamLines(div_func, stroke_width=4, max_anchors_per_line=100, padding=1, dt=.00001, noise_factor=2, y_range=[-4,3,.5])
        vf = VectorField(div_func)
        self.add(stream_lines)
        stream_lines.start_animation(warm_up=False, flow_speed=1)

        circ1 = Circle(radius=1,color=BLUE)
        circ1.set_fill(BLACK, opacity=1)
        self.add(circ1)
        squr = Square(side_length=.25).shift(LEFT*3+UP*.5)
        dsqur = squr.copy().set_color(WHITE)
        dx = Text('dx', color=WHITE).scale(.5).next_to(dsqur, DOWN)
        dy = Text('dy', color=WHITE).scale(.5).next_to(dsqur, RIGHT)
        self.add(dsqur,dx,dy)
        
        dot = Dot().move_to(squr)
        self.wait(stream_lines.virtual_time / stream_lines.flow_speed)


        vf.nudge(squr, -2, 60, True)
        vf.nudge(dot, -2, 60)

        squr.add_updater(vf.get_nudge_updater(pointwise=True))
        dot.add_updater(vf.get_nudge_updater())
        self.add(squr, dot)

        self.wait(12)
        self.play(FadeOut(stream_lines,circ1,squr,dsqur,dx,dy))
        self.play(Write(eq))
        self.wait(2)

        # not run


        # manim -pqh discord.py FluidFlowScene



class Clock_ex(Scene):
    def construct(self):
        body = Circle(radius=3, color=PURE_GREEN)
        ticks = VGroup()
        for tick in range(0,360,6):
            if tick % 30 == 0:
                self.add(Line(2.5*UP, 2.8*UP, color=PURE_GREEN).rotate(tick, about_point=ORIGIN))
            else:
                self.add(Line(2.65*UP, 2.8*UP, color=PURE_GREEN).rotate(tick, about_point=ORIGIN))

            n = VGroup()
            for num in range(1,13,1):
                self.add(Tex(r"{}".format(num), color=PURE_GREEN).scale(0.5).shift(2.2*UP))
        self.play(FadeIn(body), FadeIn(ticks))
        self.wait(3)
        self.play(FadeIn(n))
        self.wait(3)


        # manim -pqh discord.py Clock_ex



class para_ex(Scene):
    def construct(self):
        npl = NumberPlane()        
        self.add(npl)
        p = ValueTracker(1)
        pbola = npl.plot_implicit_curve(lambda x, y:y**2 - 2*p.get_value()*x).set_color(RED)
        self.add(pbola)
        focus = Dot([p.get_value()/2,0,0]).set_color(YELLOW)
        
        negFocus = Line(
            npl.c2p(-p.get_value()/2, npl.y_range[0]),
            npl.c2p(-p.get_value()/2, npl.y_range[1]),
        ).set_color(RED)
        self.add(negFocus)
        
        pos = ValueTracker(0)
        dot = always_redraw(lambda:
            Dot(pbola.point_from_proportion(pos.get_value()))
        )
        
        fline = Line()
        def flineUpdater(mobj):
            mobj.become(
                Line(
                    focus.get_center(),
                    dot.get_center(),
                    stroke_width = 2,
                )
            )
        fline.add_updater(flineUpdater)
        fline.update()
        
        hline = Line()
        def hlineUpdater(mobj):
            mobj.become(
                Line(
                    dot.get_center(),
                    negFocus.get_projection(dot.get_center()),
                    stroke_width = 2,
                )
            )
        hline.add_updater(hlineUpdater)
        hline.update()
        
        circle = Circle()
        def circleUpdater2(mobj):
            mobj.become(
                Circle(
                    radius=fline.get_arc_length(),
                ).set_color(GREEN).move_to(fline.get_end())
            )

        circle.add_updater(circleUpdater2)
        circle.update()
        
        self.add(focus, dot, circle, fline, hline)
        self.play(pos.animate.set_value(1),rate_func=linear,run_time=4)


        # manim -pqh discord.py para_ex
        


class streamline_function_ex(Scene):
    def construct(self):
        func = lambda pos: ((pos[0]*UR+pos[1]*LEFT) - pos)  
        mob = StreamLines(func,x_range=[-6,6,1.1], y_range=[-6,6,1.1],stroke_width=3)  
        self.add(mob) 

        # manim -pqh discord.py streamline_function_ex



class MovingTextAlongGraph(Scene):
    def construct(self):
        ax = Axes(x_range=[0,5], y_range=[0, 10])
        ax.add_coordinates()
        
        def func(x):
            return 1 / 4 * x ** 2
        
        points = [1,2,3,4,5,6]
        graph = ax.plot(func, color=BLUE)
        neutral_price = Tex("Neutral Price").scale(0.7)
        neutral_price.next_to(ax.c2p(0, 0), DOWN)         
        neutral_price_start = neutral_price.copy()
        def get_position(text):    
            text.become(neutral_price_start)         
            text.next_to(graph.points[-1], DOWN)
            if graph.get_arc_length() > 0.1:            
                slope = TangentLine(graph, 1).get_slope()
            else:
                slope = 0
            text.rotate(slope)   

        self.play(Create(ax))
        self.play(Write(neutral_price))
        neutral_price.add_updater(get_position)
        self.play(Create(graph), run_time=3)


        # manim -pqh discord.py MovingTextAlongGraph




class StairGraphExample(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0,7], 
            y_range=[0, 10]
        )
        ax.add_coordinates()
        
        def func(x):
            return 1 / 4 * x ** 2
        graph = ax.plot(func, color=BLUE)

        self.add(ax)  

        self.play(
            Create(graph)
        )    
        self.wait(2)      
                
        points =[1,2,3.5,5,6]
        for i in range(len(points)-1):
            hline = ax.plot(lambda x: func(points[i+1]), x_range=[points[i],points[i+1]]).set_color(YELLOW)
            '''self.add(
                hline,
                Line(
                    ax.c2p(points[i],func(points[i])), 
                    ax.c2p(points[i],func(points[i+1])), 
                ),
                ax.get_area(
                    graph, 
                    x_range=[points[i],points[i+1]],
                    opacity=0.4,
                    bounded_graph=hline,
                )
            )'''

            
            self.play(
                Create(
                   Line(
                    ax.c2p(points[i],func(points[i])), 
                    ax.c2p(points[i],func(points[i+1])), 
                    ) 
                )
            )
            self.play(
                Create(hline)
            )
            self.play(
                ax.animate.get_area(
                    graph, 
                    x_range=[points[i],points[i+1]],
                    opacity=0.4,
                    bounded_graph=hline,
                    color=PURE_GREEN
                )
            )

        
        self.wait(2)

        


            # manim -pqh discord.py StairGraphExample
                


class Mandala(Scene):
    def construct(self):
        n = ValueTracker(4.8)
        #a = ValueTracker(0.1*PI)
        a = ValueTracker(31.73)

        plane = PolarPlane()
        def r(theta):
            val = 1 / np.cos(theta * n.get_value()) 
            return np.abs(val) if -4<val<+4 else np.abs(4*np.sign(val))
        graph = plane.plot_polar_graph(
            r, 
            theta_range=[0.1*PI, a.get_value(), 1/500], 
            use_smoothing=False,
            color=ORANGE
        )
        self.add(plane, graph)   
        self.play(Rotate(graph,2*PI),run_time=4, rate_func=linear)


        # manim -pqh discord.py Mandala




class RectStretch(Scene):
    def construct(self):  
        rect = Rectangle(WHITE, 2, 2)
        self.play(FadeIn(rect))
        self.play(
            rect.animate
                .stretch(3, dim=0, about_point=DL)
                .stretch(2, dim=1, about_point=DL)
        )
        self.wait()
        
        rect_2 = Rectangle(WHITE, 1, 4).next_to(DL, UR, buff=0)
        
        self.play(Transform(rect, rect_2))
        self.wait()
        self.play(FadeOut(rect))

        h, w = ValueTracker(1), ValueTracker(2)
        polygon = always_redraw(lambda:
            Polygon(
                ORIGIN,
                h.get_value() * UP,
                h.get_value() * UP + w.get_value() * RIGHT,
                w.get_value() * RIGHT,
            )
        )
        self.play(FadeIn(polygon))
        self.play(h.animate.set_value(2))
        self.play(
            h.animate.set_value(1.5),
            w.animate.set_value(7), 
        )
        self.play(w.animate.set_value(4))


        # manim -pqh discord.py RectStretch



import random

def circles_intersect(circle1, circle2):
    distance = np.linalg.norm(circle1.get_center() - circle2.get_center())
    return distance < circle1.get_width() / 2 + circle2.get_width() / 2


class MovingCircles(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_points = 50
        self.ps = [
            Dot(0.1).shift(random.uniform(-1, 1) * UP + random.uniform(-1, 1) * RIGHT)
            for _ in range(num_points)
        ]
        self.add(*self.ps)
        for point in self.ps:
            point.velocity = random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0
        self.add_updater(self.update_self)

    @staticmethod
    def update_self(self, dt):
        for point in self.ps:
            point.shift(point.velocity)

            if point.get_center_of_mass()[0] < -2 or point.get_center_of_mass()[0] > 2:
                point.velocity = (
                    -point.velocity[0],
                    point.velocity[1],
                    point.velocity[2],
                )
            if point.get_center_of_mass()[1] < -2 or point.get_center_of_mass()[1] > 2:
                point.velocity = (
                    point.velocity[0],
                    -point.velocity[1],
                    point.velocity[2],
                )

            for other_point in self.ps:
                if point != other_point and circles_intersect(point, other_point):
                    point.velocity, other_point.velocity = (
                        other_point.velocity,
                        point.velocity,
                    )
        return self

class TestCirclesBouncing(Scene):
    def construct(self):
        box = Rectangle(width=4.4, height=4.4)
        self.play(FadeIn(box))
        circles = MovingCircles()
        self.add(circles)
        self.wait(5)


        # manim -pqh discord.py TestCirclesBouncing



from colour import Color

def circles_intersect(circle1, circle2):
    distance = np.linalg.norm(circle1.get_center() - circle2.get_center())
    return distance < circle1.get_width() / 2 + circle2.get_width() / 2

class veloDot(Dot):
    def __init__(self, velocity = np.ndarray([0,0,0]), mass = 1, tracker = None, **kwargs):
        super().__init__(**kwargs)
        self.velocity = velocity
        self.mass     = mass
        self.tracker  = tracker
        self.add_updater(self.updater)

    @staticmethod    
    def updater(mobj, dt):
        if mobj.tracker != None:
            pos = mobj.get_center() + mobj.tracker.get_value()*dt*mobj.velocity
        else:
            pos = mobj.get_center() + dt*mobj.velocity

        if (pos[0] >= 2):
            pos[0] = +2 - (pos[0] - 2)
            mobj.velocity[0] = -abs(mobj.velocity[0])
        elif (pos[0] <= -2):
            pos[0] = -2 - (pos[0] + 2)
            mobj.velocity[0] = +abs(mobj.velocity[0])
        if (pos[1] >= 2):
            pos[1] = +2 - (pos[1] - 2)
            mobj.velocity[1] = -abs(mobj.velocity[1])
        elif (pos[1] <= -2):
            pos[1] = -2 - (pos[1] + 2)
            mobj.velocity[1] = +abs(mobj.velocity[1])
        mobj.move_to(pos)

class ParticleCollisonScene(Scene):
    def construct(self):
        sloMo = ValueTracker(1)
        dots = VGroup()
        for i in range(20):
            dot = veloDot(
                point    = np.random.uniform(low=-1,high=1)*RIGHT+np.random.uniform(low=-1,high=1)*UP,
                velocity = np.random.uniform(low=-3,high=3)*RIGHT+np.random.uniform(low=-3,high=3)*UP,
                mass = np.random.uniform(low=0.5, high=2),
                tracker = sloMo
            )
            dots += dot

        def dotsUpdater(mobj):
            for i in range(len(mobj)):
                for j in range(i+1,len(mobj)):
                    if circles_intersect(dots[i],dots[j]):
                        v1 = dots[i].velocity
                        v2 = dots[j].velocity
                        m1 = dots[i].mass
                        m2 = dots[j].mass
                        dots[i].velocity = (m1*v1 + m2*(2*v2-v1))/(m1+m2)
                        dots[j].velocity = (m2*v2 + m1*(2*v1-v2))/(m1+m2)
                        color = Color(hsl=(np.random.uniform(0,1),1,0.5))
                        dots[i].set_color(color) 
                        dots[j].set_color(color) 

        dots.add_updater(dotsUpdater)                    
        self.add(dots, Square(side_length=4,stroke_width=2))
        self.wait(5)
        self.play(sloMo.animate.set_value(0.2))
        self.wait(5)
        self.play(sloMo.animate.set_value(2))
        self.wait(5)

        # manim -pqh discord.py ParticleCollisonScene




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


        # manim -pqh discord.py graphingEx

        # manim -sqk discord.py graphingEx



class Color_Gradient_example(Scene):
  def construct(self):

    # WATER MARK 

    water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
    self.add(water_mark)

    radius = 2
    colors = color_gradient(["#fef601", "#fef601", "#ba7946", "#780488", "#14a9ec"], 100)
    cs = VGroup(
       *[Circle(radius=(i+1)*radius/100,stroke_width=2*radius, color=colors[i]) for i in range(100)] 
    )
    #self.add(cs)
    self.play(
        Write(cs)
    )

    # manim -pqh discord.py Color_Gradient_example

    # manim -sqk discord.py Color_Gradient_example
    


import numpy as np

class FourierEpicycle(Scene):
    def construct(self):
        # Define the number of epicycles and the frequency of the signal
        num_circles = 20
        signal_freq = 5

        # Generate a random signal
        x_vals = np.linspace(-4*np.pi, 4*np.pi, 1000)
        y_vals = np.zeros_like(x_vals)
        for i in range(1, signal_freq+1):
            y_vals += np.sin(i*x_vals)

        # Plot the signal
        axes = Axes(
            x_range=[-4*np.pi, 4*np.pi, np.pi],
            y_range=[-2, 2, 1],
            x_length=8,
            y_length=4,
            axis_config={"color": WHITE},
            x_axis_config={"numbers_to_include": np.arange(-4*np.pi, 4*np.pi+1, np.pi)},
            y_axis_config={"numbers_to_include": np.arange(-2, 2.1, 1)}
        )
        signal = axes.get_graph(lambda x: np.interp(x, x_vals, y_vals))



        # Plot the Fourier Epicycle
        epicycles = self.get_epicycles(num_circles, signal_freq, signal)

        # Show the animation
        self.add(signal, *epicycles)

        self.wait(10)

    def get_epicycles(self, num_circles, signal_freq, signal):
        epicycles = []
        epicycle_radius = 0.5

        # Get the Fourier coefficients
        x_vals = np.linspace(-4*np.pi, 4*np.pi, 1000)
        y_vals = np.interp(x_vals, signal.get_points()[:, 0], signal.get_points()[:, 1])
        fourier_coeffs = np.fft.fft(y_vals)

        # Plot the epicycles
        circle = Circle(radius=epicycle_radius, color=WHITE)
        circle.move_to(ORIGIN)
        epicycles.append(circle)

        for i in range(1, num_circles+1):
            prev_circle = epicycles[-1]
            circle = Circle(radius=epicycle_radius/i, color=WHITE)
            circle.next_center = prev_circle.get_right()

            # Add the rotating animation
            rot_angle = -np.angle(fourier_coeffs[i])
            circle.rotate(rot_angle, about_point=prev_circle.get_center())

            epicycles.append(circle)

        # Plot the path traced by the last epicycle
        path = VMobject()
        path.set_points_smoothly(signal.get_points())
        traced_path = TracedPath(path, stroke_width=2, stroke_color=YELLOW)

        # Connect the epicycles with lines
        lines = VGroup()
        for i in range(len(epicycles)-1):
            line = Line(
                epicycles[i].get_right(),
                epicycles[i+1].get_left(),
                stroke_width=2,
                stroke_color=WHITE
            )
            lines.add(line)

        return [lines, traced_path, *epicycles]



        # manim -pqh discord.py FourierEpicycle





class FourierEpicycle_1(Scene):
    def construct(self):
        # Set up the parameters
        num_circles = 10
        amplitude = 1.5
        freqs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        phase_shifts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        circle_color = BLUE
        curve_color = YELLOW

        # Define the functions for the Fourier series
        def f(t):
            return sum(amplitude * np.exp(2 * np.pi * freqs[i] * t * 1j + phase_shifts[i]) for i in range(num_circles))

        def f_real(t):
            return np.real(f(t))

        def f_imag(t):
            return np.imag(f(t))

        # Create the circles and curve
        circles = VGroup()
        for i in range(num_circles):
            circle = Circle(radius=amplitude / (i + 1), color=circle_color, stroke_width=2)
            circles.add(circle)

        curve = ParametricFunction(lambda t: np.array([f_real(t), f_imag(t), 0]), t_range=[0,1], color=curve_color, stroke_width=2)

        # Animate the circles and curve
        self.add(circles, curve)

        self.play(*[ApplyMethod(circles[i].rotate, -360 * freqs[i], about_point=ORIGIN) for i in range(num_circles)], rate_func=linear, run_time=4)

        self.wait()


        # manim -pqh discord.py FourierEpicycle_1


class upd(Scene):
    def construct(self):
        color = [RED,BLUE,PURPLE,WHITE]
        dots = VGroup(*[Dot(fill_opacity=1).set_color(color[i%4]) for i in range(2000)])
        dots.arrange_in_grid(40,buff=0)
        self.add(dots)
        def mcomp(mob):
            return complex(mob.get_center()[0],mob.get_center()[1])
        def pcomp(comp):
            return [comp.real,comp.imag,0]
        def iter(mob,c = 1,n=5):
            
            for i in range(n):
                a = []
                for sub in mob:
                    z_old = mcomp(sub)
                    z_new = z_old**2 + c
                    a.append(sub.animate.move_to(pcomp(z_new)))
                    z_old = z_new
                self.play(*a)
        
        self.play(Write(dots))
        self.wait()
        iter(dots,complex(0,1),5)
        self.wait()
        self.play(Unwrite(dots))


        # manim -pqh discord.py upd




class FourierCircleBase(VGroup):
    def __init__(self, vt, n, cn, **kwargs):
        super().__init__(**kwargs)
        self.vt = vt
        self.cn = cn
        self.n = n
        self.circle = always_redraw(
            lambda: Circle(
                fill_color=BLUE_D,
                fill_opacity=0.1,
                stroke_color=BLUE_B,
                stroke_width=1,
                stroke_opacity=0.5,
            )
            .scale(np.linalg.norm(self.cn))
            .move_to(self.get_center())
        )
        # self.num_dots = 5 if np.linalg.norm(cn) > 0.2 else 0
        # self.dots = VGroup(
        #     *[
        #         Dot().scale(0.1).move_to(self.circle.point_from_proportion(i))
        #         for i in np.linspace(
        #             0,
        #             1,
        #             self.num_dots,
        #         )
        #     ]
        # )

        self.vec = Vector(RIGHT, stroke_width=2)
        self.add(self.circle, self.vec)  # , self.dots)
        self.add_updater(self.update_state)

    @staticmethod
    def update_state(instance: "FourierCircleBase", dt):
        instance.vec.put_start_and_end_on(
            instance.get_center(),
            instance.get_attachment_point(),
        )
        # if instance.num_dots > 0:
        #     for dot, i in zip(instance.dots, np.linspace(0, 1, instance.num_dots)):
        #         dot.move_to(
        #             instance.circle.point_from_proportion(
        #                 (i - instance.vt.get_value()) % 1
        #             )
        #         )

    def get_attachment_point(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def set_cns(self, n, cn):
        self.n = n
        self.cn = cn


class ComplexCircle(FourierCircleBase):
    def __init__(self, vt, factor, scale, **kwargs):
        super().__init__(vt, factor, scale, **kwargs)

    def get_vec_dir(self, t):
        return self.cn * np.e ** (1j * self.n * t * 2 * np.pi)

    def get_attachment_point(self):
        result = self.get_vec_dir(self.vt.get_value())
        return self.get_center() + result.real * RIGHT + result.imag * UP


class FourierManager(VGroup):
    def __init__(self, vt, constructor=ComplexCircle, **kwargs):
        super().__init__(**kwargs)
        self.vt = vt
        self.center_dot = Dot(color=BLUE).scale(0.5)
        self.add(self.center_dot)
        self.circles = []
        self.constructor = constructor

    def add_circle(self, factor, scale):
        vc = self.constructor(self.vt, factor, scale)
        attach_func = self.get_attachment_func()

        def tmp(instance):
            instance.move_to(attach_func())

        vc.add_updater(tmp)
        self.add(vc)
        self.circles.append(vc)

    def get_attachment_func(self):
        if len(self.circles) == 0:
            return lambda: self.center_dot.get_center()
        return self.circles[-1].get_attachment_point

    @staticmethod
    def create_producer(s):
        """This function generates a function which returns a complex point from the proportion of the given shape.

        Parameters
        ----------
        s : VMobject
            The shape to generate the function for.
        """

        def complex_from_proportion(t):
            point = s.point_from_proportion(t)
            return point[0] + point[1] * 1j

        return complex_from_proportion

    def init_for_shape(self, shape, degree, resolution=10000):
        produce_prportion = self.create_producer(shape)
        samples = np.fromiter(
            (produce_prportion(t) for t in np.linspace(0, 1, resolution)),
            count=resolution,
            dtype=np.complex128,
        )

        
        def generate_cns(samples, degree):
            ns = np.linspace(-degree / 2, degree / 2, degree + 1)
            idxs = np.argsort(np.abs(ns))
            ns = ns[idxs]
            cns = [calculate_cn(n, samples) for n in ns]
            return ns, cns

        ns, cns = generate_cns(samples, degree)
        self.ns = ns
        self.cns = cns
        for n, cn in zip(ns, cns):
            self.add_circle(n, cn)



def calculate_cn(n, samples):
    ts = np.linspace(0, 1, len(samples))
    return sum(
        [np.exp(-2 * np.pi * 1j * n * t) * ft for t, ft in zip(ts, samples)]
    ) / len(ts)
    # return integrate.quad(tmp, 0, 1, )[0]


def auto_zoom(self, *mobjects, margin=1, animate=False):
    bb = Group(*mobjects).get_bounding_box()
    lower_left = bb[0]
    upper_right = bb[2]
    new_width = upper_right[0] - lower_left[0]
    new_height = upper_right[1] - lower_left[1]
    x = lower_left[0] + new_width / 2
    y = lower_left[1] + new_height / 2
    m_target = self.animate if animate else self
    if new_width / self.get_width() > new_height / self.get_height():
        return m_target.set_x(x).set_y(y).set_width(new_width + margin)
    else:
        return m_target.set_x(x).set_y(y).set_height(new_height + margin)


class Test_epi(Scene):
    def construct(self):
        self.camera.background_rgba = [0, 0, 0, 0]
        t_tracker = ValueTracker(0)
        degree = 200
        resolution = 10000
        # shape = (
        #     Square(stroke_width=1, color=GREY_A, opacity=0.2).scale(2).rotate(PI / 4)
        # )
        shape = Text("Test", fill_opacity=0, stroke_width=2, stroke_color=GREY_A).scale(
            7
        )
        print("Initializing")

        # s = VMobject()
        # s.set_points_as_corners(shape.get_all_points())
        # s.push_self_into_submobjects()
        # shape = s
        # self.play(Write(shape))

        def factory_for_shape(shape):
            original_point = shape.get_center()
            m1 = FourierManager(t_tracker, constructor=ComplexCircle)
            m1.move_to(shape)
            shape.move_to(ORIGIN)
            m1.init_for_shape(shape, degree, resolution)
            f = m1.get_attachment_func()
            path = TracedPath(
                f,
                stroke_color=RED,
                stroke_width=2,
                stroke_opacity=0.8,
            ).suspend_updating()
            return VGroup(m1, path)

        t_tracker.set_value(0)
        shapes = VGroup()
        paths = VGroup()
        for s in shape:
            s1, p1 = factory_for_shape(s)
            s1.update(0)
            shapes.add(s1)
            paths.add(p1)

        self.play(
            AnimationGroup(*[Write(s) for s in shapes], lag_ratio=0.1),
            run_time=3,
        )
        self.add(paths)
        paths.resume_updating()

        self.play(t_tracker.animate.set_value(2), run_time=5, rate_func=linear)

        # decoy = VMobject()
        # decoy.add_updater(lambda m, dt: t_tracker.increment_value(dt / 10))
        # self.add(decoy)
        # self.wait(20)
        # frame = self.camera.frame
        # updater = decoy.add_updater(
        #     lambda m: auto_zoom(frame, shape2.circles[-1], margin=3)
        # )
        # self.wait(5)


        # manim -pql discord.py Test_epi




class rad_gard_1(Scene):
  def construct(self):
    radius = 2
    colors = color_gradient([WHITE,DARK_BLUE], 100)
    cs = VGroup(
       *[Circle(radius=(i+1)*radius/100,stroke_width=2*radius, color=colors[i]) for i in range(100)] 
    )
    self.add(cs)


    # manim -sqk discord.py rad_gard_1

    # manim -pqh discord.py rad_gard_1


###################################################################################################################

# NOTE :-

'''
Q1. How can I configure the output video format to be square or vertical? Can it be done directly with manim?
Ans: python3 -m manim -pql -r 1080,1920 my_file.py
'''


###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix"