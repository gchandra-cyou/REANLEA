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




def makeTorus(major_radius=3, minor_radius=1, theta_step=30, theta_offset=0):
    torus = VGroup()
    for n in range(0,360,theta_step):
        torus.add(Circle(radius=major_radius-minor_radius*np.cos((n+theta_offset)*DEGREES)).shift(minor_radius*np.sin((n+theta_offset)*DEGREES)*OUT).set_color(BLUE))
    return torus

class torus(ThreeDScene):
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

        # manim -pqh discord_2.py torus



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


        # manim -pqh discord_2.py svg_electronics




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



        # manim -pqh discord_2.py svg_electronics_index



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


        # manim -pqh discord_2.py ColoredCircuit



class Expl1(Scene):
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



        # manim -pqh discord_2.py Expl1




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


class FollowBezierEx(Scene):
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


    # manim -pqh discord_2.py FollowBezierEx



class Test2y(ThreeDScene):
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



        # manim -pqh discord_2.py Test2y





class colorTexex(Scene):
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


        # manim -pqh discord_2.py colorTexex



class GraphEx2(Scene):
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


        # manim -pqh discord_2.py GraphEx2

        # manim -sqk discord_2.py GraphEx2



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



        # manim -pqh discord_2.py Tree

        # manim -sqk discord_2.py Tree



class Vortices(MovingCameraScene):
    def construct(self):

        # WATER MARK 

        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15).set_z_index(-100)
        self.add(water_mark)

        rigid = lambda pos: pos[0] * DOWN + pos[1] * RIGHT
        stream_rigid = StreamLines(rigid, x_range=[-2, 2, 0.2], y_range=[-2, 2, 0.2])
        stream_rigid.start_animation(warm_up=False)

        self.add(stream_rigid)
        self.wait()
        self.play(
            self.camera.frame.animate.shift(2*RIGHT),
            water_mark.animate.shift(2*RIGHT)
        )
        self.wait()


        # manim -pqh discord_2.py Vortices

        # manim -sqk discord_2.py Vortices




class riemannint(Scene):
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


        # manim -pqh discord_2.py riemannint


