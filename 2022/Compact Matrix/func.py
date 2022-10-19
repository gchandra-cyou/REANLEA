from __future__ import annotations
from cProfile import label


import fractions
from imp import create_dynamic
from multiprocessing import context
from multiprocessing.dummy import Value
from numbers import Number
from tkinter import Y, Label
from imp import create_dynamic
from turtle import degrees
from typing import List
from manim import*
from math import*
from manim_fonts import*
import numpy as np
import random
from sklearn.datasets import make_blobs
from reanlea_colors import*
from manim.camera.moving_camera import MovingCamera
from manim.scene.scene import Scene
from manim.utils.family import extract_mobject_family_members
from manim.utils.iterables import list_update
from manim.camera.multi_camera import*
from manim.mobject.geometry.tips import ArrowTriangleFilledTip
from manim.mobject.geometry.tips import ArrowTip
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import root_scalar
from manim import *




config.background_color= REANLEA_BACKGROUND_COLOR


class DashedArrow(DashedLine):
    def __init__(
        self,
        *args,
        stroke_width=6,
        buff=MED_SMALL_BUFF,
        max_tip_length_to_length_ratio=0.25,
        max_stroke_width_to_length_ratio=5,
        dash_length=DEFAULT_DASH_LENGTH,
        dashed_ratio=0.5,
        **kwargs,
    ):
        self.max_tip_length_to_length_ratio = max_tip_length_to_length_ratio
        self.max_stroke_width_to_length_ratio = max_stroke_width_to_length_ratio
        self.dash_length = dash_length
        self.dashed_ratio = dashed_ratio
        tip_shape = kwargs.pop("tip_shape", ArrowTriangleFilledTip)
        super().__init__(*args, buff=buff, stroke_width=stroke_width, **kwargs)
        # TODO, should this be affected when
        # Arrow.set_stroke is called?
        self.initial_stroke_width = self.stroke_width
        self.add_tip(tip_shape=tip_shape)
        self._set_stroke_width_from_length()

    def scale(self, factor, scale_tips=False, **kwargs):
        if self.get_length() == 0:
            return self

        if scale_tips:
            super().scale(factor, **kwargs)
            self._set_stroke_width_from_length()
            return self

        has_tip = self.has_tip()
        has_start_tip = self.has_start_tip()
        if has_tip or has_start_tip:
            old_tips = self.pop_tips()

        super().scale(factor, **kwargs)
        self._set_stroke_width_from_length()

        if has_tip:
            self.add_tip(tip=old_tips[0])
        if has_start_tip:
            self.add_tip(tip=old_tips[1], at_start=True)
        return self


    def get_normal_vector(self) -> np.ndarray:
        p0, p1, p2 = self.tip.get_start_anchors()[:3]
        return normalize(np.cross(p2 - p1, p1 - p0))

    def reset_normal_vector(self):
        self.normal_vector = self.get_normal_vector()
        return self

    def get_default_tip_length(self) -> float:
        max_ratio = self.max_tip_length_to_length_ratio
        return min(self.tip_length, max_ratio * self.get_length())


    def _set_stroke_width_from_length(self):
        max_ratio = self.max_stroke_width_to_length_ratio
        if config.renderer == "opengl":
            self.set_stroke(
                width=min(self.initial_stroke_width, max_ratio * self.get_length()),
                recurse=False,
            )
        else:
            self.set_stroke(
                width=min(self.initial_stroke_width, max_ratio * self.get_length()),
                family=False,
            )
        return self



class DashedDoubleArrow(DashedArrow):
    def __init__(self, *args, **kwargs):
        if "tip_shape_end" in kwargs:
            kwargs["tip_shape"] = kwargs.pop("tip_shape_end")
        tip_shape_start = kwargs.pop("tip_shape_start", ArrowTriangleFilledTip)
        super().__init__(*args, **kwargs)
        self.add_tip(at_start=True, tip_shape=tip_shape_start)




class Spherez(Surface):
    def __init__(
        self,
        center=ORIGIN,
        radius=1,
        resolution=None,
        u_range=(0,TAU),
        v_range=(0,PI),
        fill_color=BLUE_D,
        fill_opacity=1,
        **kwargs
    ):  
        if config.renderer=="opengl":
            res_value=(101,51)
        else:
            res_value=(24,12)
    
        resolution = resolution if resolution is not None else res_value

        self.radius = radius

        super().__init__(
            self.func ,
            resolution=resolution,
            u_range=u_range,
            v_range=v_range,
            fill_color = fill_color,
            fill_opacity= fill_opacity,
            **kwargs,
        )

        self.shift(center)

    def func(self, u, v):
        return self.radius * np.array(
        [np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), -np.cos(v)],
    )



class UpdatedMovingCameraScene(Scene):

    def __init__(self, camera_class= MovingCamera, **kwargs):
        super().__init__(camera_class=camera_class, **kwargs)

    def get_moving_mobjects(self, *animations):
        moving_mobjects = super().get_moving_mobjects(*animations)
        all_moving_mobjects = extract_mobject_family_members(moving_mobjects)
        movement_indicators = self.renderer.camera.get_mobjects_indicating_movement()
        for movement_indicator in movement_indicators:
            if movement_indicator in all_moving_mobjects:
                return list_update(self.mobjects, moving_mobjects)
        return moving_mobjects




def get_glowing_surround_circle(
    circle, buff_min=0, buff_max=0.15, color=REANLEA_YELLOW, n=40, opacity_multiplier=1
):
    current_radius = circle.width / 2
    glowing_circle = VGroup(
        *[
            Circle(radius=current_radius+interpolate(buff_min, buff_max, b))
            for b in np.linspace(0, 1, n)
        ]
    )
    for i, c in enumerate(glowing_circle):
        c.set_stroke(color, width=0.5, opacity=1- i / n)
    return glowing_circle.move_to(circle.get_center())




def get_stripe(
    factor=0.25, buff_min=0, buff_max=5, color=REANLEA_TXT_COL_DARKER, n=500
):  
    line=Line(ORIGIN,RIGHT).scale(factor).set_opacity(0.2)

    k=(buff_max-buff_min)/(n-1)*50
    stripe=VGroup(
        *[
            line.copy().shift(DOWN*interpolate(buff_min,buff_max,b))
            for b in np.linspace(0,1,n)
        ]
    )

    for i,c in enumerate(stripe):
        c.set_stroke(color,width=k*line.get_stroke_width(),opacity=1-(1*i)/n)     # line.stroke_width=0.04 , default
#opacity=1-i/n
    return stripe.rotate(PI/2).move_to(ORIGIN).set_z_index(-10)



    
def get_surround_bezier(text):

    rad= max(text.width, text.height, 1.4)

    current_radius=rad

    p1 = ParametricFunction(
        lambda t: bezier(np.array([
            [1.68,.67,0],
            [1.65,.35, 0],
            [.21,.67, 0],
            [.3,1.13, 0],  
        ]))(t),
        [0, 1],
        color=REANLEA_CHARM,
    )
    p2 = ParametricFunction(
        lambda t: bezier(np.array([
            [.3, 1.13, 0],
            [.47, 1.65, 0],
            [2.93, .26, 0],
            [.93, .58, 0],
        ]))(t),
        [0, 1],
        color=REANLEA_CHARM,
    )

    grp= VGroup(p1,p2).flip(axis=RIGHT).rotate(-15*DEGREES)

    return grp.move_to(text.get_center()).scale(current_radius)



def ArrowCubicBezierUp():

        grp=VGroup()

        p1 = ParametricFunction(
            lambda t: bezier(np.array([
                [2.21,1.50,0],
                [2.19, .31, 0],
                [.33, 1.83, 0],
                [.40, .58, 0],  
            ]))(t),
            [0, 1],
            #color=REANLEA_CHARM,
        ).flip(RIGHT)

        p=CurvesAsSubmobjects(p1)
        p.set_color_by_gradient(REANLEA_YELLOW,REANLEA_CHARM)

        grp += p


        ar= Arrow(max_stroke_width_to_length_ratio=0,max_tip_length_to_length_ratio=0.15).move_to(p1.get_end()+.55*DOWN).rotate(PI/2)
        ar.set_color(REANLEA_CHARM)

        
        grp += ar

        #grp.set_color_by_gradient(REANLEA_CHARM)

        return grp#.next_to(text, .2*UP)



def ArrowQuadricBezierDown(text):

        grp=VGroup()

        p1 = ParametricFunction(
            lambda t: bezier(np.array([
                [2.30,2.21,0],
                [.9, 2.27, 0],
                [.76, .1, 0],  
            ]))(t),
            [0, 1],
            #color=REANLEA_CHARM,
        ).flip(RIGHT)

        p=CurvesAsSubmobjects(p1)
        p.set_color_by_gradient(REANLEA_BLUE_DARKER,REANLEA_BLUE,REANLEA_CHARM)

        grp += p


        ar= Arrow(max_stroke_width_to_length_ratio=0,max_tip_length_to_length_ratio=0.15).move_to(p1.get_end()+.55*DOWN).rotate(PI/2)
        ar.set_color(REANLEA_CHARM)

        
        grp += ar

        #grp.set_color_by_gradient(REANLEA_CHARM)

        return grp.next_to(text, .2*DOWN)



def under_line_bezier_arrow():

        grp=VGroup()

        p1 = ParametricFunction(
            lambda t: bezier(np.array([
                [.42,1.69,0],
                [.20, 1.04, 0],
                [1.30, 0.97, 0],
                [2.21, 1.36, 0],  
            ]))(t),
            [0, 1],
            #color=PURE_RED,
        ).flip(RIGHT)

        p=CurvesAsSubmobjects(p1)
        p.set_color_by_gradient(PURE_RED, REANLEA_BLUE_LAVENDER)

        grp += p


        ar= Arrow(max_stroke_width_to_length_ratio=0,max_tip_length_to_length_ratio=0.1).move_to(p1.get_end()+.65*DOWN).rotate(PI/2)
        ar.set_color(REANLEA_SLATE_BLUE_LIGHTEST)

        
        grp += ar

        #grp.set_color_by_gradient(REANLEA_CHARM)

        return grp



def bend_bezier_arrow():

        grp3=VGroup()
        p1 = ParametricFunction(
            lambda t: bezier_updated(t,
                np.array([
                    [1.91,.29,0],
                    [.2,1.1, 0],
                    [1.9, 2.53, 0],
                ]),
                np.array([1,1,1])),
            t_range=[0, 1],
            color=REANLEA_CHARM,
        )
        
        p1.move_to(ORIGIN).rotate(50*DEGREES)

        p=CurvesAsSubmobjects(p1)
        p.set_color_by_gradient(REANLEA_YELLOW_CREAM,REANLEA_CHARM).set_stroke(width=3)

        grp3 += p


        ar= Arrow(max_stroke_width_to_length_ratio=0,max_tip_length_to_length_ratio=0.09).move_to(p1.get_end()+.7*DOWN).rotate(PI/2)
        ar.set_color(REANLEA_CHARM)

        
        grp3 += ar
        grp3.move_to(2.15*DOWN+RIGHT).flip(RIGHT).rotate(130*DEGREES)


        return grp3



def low_frame_rate(t):
    return np.floor(t*10)/10



def bezier_updated(t,
    points: np.ndarray,
    weights: np.ndarray,
):
    n = len(points) - 1

    return  sum(
        ((1 - t) ** (n - k)) * (t**k) * choose(n, k) * point * weights[k]      
        for k, point in enumerate(points) 
    )



def focus_on(self, mobject, buff=2):
        return self.camera.frame.animate.set_width(mobject.width * buff).move_to(
            mobject
        )



def create_glow(vmobject, rad=1, col=YELLOW):
    glow_group = VGroup()
    for idx in range(60):
        new_circle = Circle(radius=rad*(1.002**(idx**2))/400, stroke_opacity=0, fill_color=col,fill_opacity=0.2-idx/300).move_to(vmobject)
        glow_group.add(new_circle)
    return glow_group


def get_mirror(
    factor=0.25, buff_min=0, buff_max=5, color=REANLEA_AQUA, n=25
):   
    scale_by=(buff_max-buff_min)

    line=Line(ORIGIN,RIGHT).scale(factor).rotate(PI/4).set_stroke(width=1).shift(LEFT)
    
    d_line=DashedLine(start=ORIGIN, end=RIGHT, stroke_width=1.5, dashed_ratio=0.75, dash_length=0.05).scale(scale_by*1.025)
    d_line.set_color_by_gradient(REANLEA_AQUA_GREEN,REANLEA_SLATE_BLUE_LIGHTER,REANLEA_VIOLET_LIGHTER)
    
    stripe=VGroup(
        *[
            line.copy().shift(DOWN*interpolate(buff_min,buff_max,b))
            for b in np.linspace(0,1,n)
        ]
    ).rotate(PI/2).move_to(0.575*RIGHT+0.15*DOWN)
    stripe.set_color_by_gradient(REANLEA_AQUA_GREEN,REANLEA_SLATE_BLUE_LIGHTER,REANLEA_VIOLET_LIGHTER)
    

    mirror=VGroup(d_line,stripe)

    
    return mirror.rotate(-PI/2)



def line_highlight(
    factor=0.1, length_factor=1, opacity_factor=0.2, buff_min=0, buff_max=1, color=REANLEA_YELLOW, n=500
):  
    line=Line(ORIGIN,RIGHT).scale(factor).set_opacity(opacity_factor)

    k=(buff_max-buff_min)/(n-1)*50
    stripe=VGroup(
        *[
            line.copy().shift(DOWN*interpolate(buff_min,buff_max,b))
            for b in np.linspace(0,1,n)
        ]
    )

    for i,c in enumerate(stripe):
        c.set_stroke(color,width=k*line.get_stroke_width())     # line.stroke_width=0.04 , default
#opacity=1-i/n
    return stripe.rotate(PI/2).move_to(ORIGIN).set_z_index(-10).scale(length_factor)

