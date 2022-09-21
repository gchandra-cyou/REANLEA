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


























#####################################                                              #####################################
##################################### BEGINING ROUNDED CORNER MOBJECT CONSTRUCTION #####################################
#####################################                                              #####################################


def angle_between_vectors_signed(v1,v2):
    cval = np.dot(v1, v2)
    sval = (np.cross(v1, v2))[2]
    return np.arctan2(sval, cval)

    
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

def Chamfer_Corner_Param(offset,curve_points_1,curve_points_2):
    # this is ugly, I know, don't judge
    if hasattr(offset,'iter'):
        ofs = [offset[0], offset[1]]
    else:
        ofs = [offset, offset]
    bez_func_1 = bezier(curve_points_1)
    bez_func_2 = bezier(curve_points_2)

    #copied from vectorized mobject length stuff
    def get_norms_and_refs(curve):
        sample_points = 10
        refs = np.linspace(0, 1, sample_points)
        points = np.array([curve(a) for a in np.linspace(0, 1, sample_points)])
        diffs = points[1:] - points[:-1]
        norms = np.cumsum(np.apply_along_axis(np.linalg.norm, 1, diffs))
        norms = np.insert(norms,0,0)
        return norms,refs

    norms1,refs1 = get_norms_and_refs(bez_func_1)
    norms2,refs2 = get_norms_and_refs(bez_func_2)
    a1 = (np.interp(norms1[-1]-ofs[0], norms1, refs1))
    a2 = (np.interp(ofs[1], norms2, refs2))
    p1 = bez_func_1(a1)
    p2 = bez_func_2(a2)
    param = {'start':p1,'end':p2}

    return param, [a1,a2]


def Chamfer_Corners(mob:VMobject,offset=0.2):
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
            params, k = Chamfer_Corner_Param(offset,curve_1,curve_2)
            cut_curve_points_1 = partial_bezier_points(curve_1, 0, k[0])
            cut_curve_points_2 = partial_bezier_points(curve_2, k[1], 1)
            loc_line = Line(**params)
            # mob.points = np.delete(mob.points, slice((ind1 * 4), (ind1 + 1) * 4), axis=0)
            # mob.points = np.delete(mob.points, slice((ind2 * 4), (ind2 + 1) * 4), axis=0)
            mob.points[ind1 * 4:(ind1 + 1) * 4, :] = cut_curve_points_1
            mob.points[ind2 * 4:(ind2 + 1) * 4, :] = cut_curve_points_2
            mob.points = np.insert(mob.points,ind2*4,loc_line.points,axis=0)
            i=i+loc_line.get_num_curves()+1
        else:
            i=i+1

        if i==mob.get_num_curves()-1 and not mob.is_closed():
            break

    return    


##################################### PATH MAPPER CONSTRUCTION #####################################
class Path_mapper(VMobject):
    def __init__(self,path_source:VMobject,num_of_path_points=100,**kwargs):
        super().__init__(**kwargs)
        self.num_of_path_points = num_of_path_points
        self.path= path_source
        self.generate_length_map()

    def generate_length_map(self):
        norms = np.array(0)
        for k in range(self.path.get_num_curves()):
            norms = np.append(norms, self.path.get_nth_curve_length_pieces(k,sample_points=11))
        # add up length-pieces in array form
        self.pathdata_lengths  = np.cumsum(norms)
        self.pathdata_alpha = np.linspace(0, 1, self.pathdata_lengths.size)

    def get_path_length(self):
        return self.pathdata_lengths[-1]

    def alpha_from_length(self,s):
        if hasattr(s, '__iter__'):
            return [np.interp(t, self.pathdata_lengths, self.pathdata_alpha) for t in s]
        else:
            return np.interp(s, self.pathdata_lengths, self.pathdata_alpha)

    def length_from_alpha(self,a):
        if hasattr(a, '__iter__'):
            return [np.interp(t, self.pathdata_alpha, self.pathdata_lengths) for t in a]
        else:
            return np.interp(a, self.pathdata_alpha, self.pathdata_lengths)

    def equalize_alpha(self, a):
        'used for inverting the alpha behavior'
        return self.alpha_from_length(a*self.get_path_length())

    def equalize_rate_func(self, rate_func):
        '''
        Specifically made to be used with Create() animation.
        :param rate_func: rate function to be equalized
        :return: callable new rate function
        Example:
        class test_path_mapper_anim(Scene):
            def construct(self):
                mob1 = round_corners(Triangle(fill_color=TEAL,fill_opacity=0).scale(3),0.5)
                PM = Path_mapper(mob1)
                mob2 = mob1.copy()
                mob1.shift(LEFT * 2.5)
                mob2.shift(RIGHT * 2.5)
                self.play(Create(mob1,rate_func=PM.equalize_rate_func(smooth)),Create(mob2),run_time=5)
                self.wait()
        '''
        def eq_func(t:float):
            return self.equalize_alpha(rate_func(t))
        return eq_func

    def point_from_proportion(self, alpha: float) -> np.ndarray:
        '''
         Override original implementation.
         Should be the same, except it uses pre calculated length table and should be faster a bit.
        '''
        if hasattr(alpha, '__iter__'):
            values = self.alpha_from_length(alpha * self.get_path_length())
            ret = np.empty((0,3))
            for a in values:
                if a == 1:
                    index = self.path.get_num_curves() - 1
                    remainder = 1
                else:
                    index = int(a * self.path.get_num_curves() // 1)
                    remainder = (a * self.path.get_num_curves()) % 1
                p = self.path.get_nth_curve_function(index)(remainder)
                ret = np.concatenate([ret,np.reshape(p,(1,3))],axis=0)
            return ret
        else:
            a = self.alpha_from_length(alpha*self.get_path_length())
            if a==1:
                index = self.path.get_num_curves()-1
                remainder = 1
            else:
                index = int(a * self.path.get_num_curves() // 1)
                remainder = (a * self.path.get_num_curves()) % 1
            return self.path.get_nth_curve_function(index)(remainder)

    def get_length_between_points(self,b,a):
        '''
        Signed arc length between to points.
        :param b: second point
        :param a: first point
        :return: length (b-a)
        '''
        return self.length_from_alpha(b)-self.length_from_alpha(a)

    def get_length_between_points_wrapped(self,b,a):
        ''' This function wraps around the length between two points similar to atan2 method.
        Useful for closed mobjects.
        Returns distance value between -L/2...L/2 '''
        AB = self.get_length_between_points(b,a)
        L = self.get_path_length()
        return (AB%L-L/2)%L-L/2

    def get_length_between_points_tuple(self,b,a):
        ''' Function to get the 2 absolute lengths between 2 parameters on closed mobjects.
        Useful for closed mobjects.
        :returns tuple (shorter, longer)'''

        AB = abs(self.get_length_between_points(b,a))
        L = self.get_path_length()
        if AB>L/2:
            return (L - AB), AB
        else:
            return AB, (L - AB)

    def get_bezier_index_from_length(self,s):
        a = self.alpha_from_length(s)
        nc = self.path.get_num_curves()
        indx = int(a * nc // 1)
        bz_a = a * nc % 1
        if indx==nc:
            indx = nc-1
            bz_a=1
        return (indx,bz_a)

    def get_tangent_unit_vector(self,s):
        # diff_bez_points = 1/3*(self.path.points[1:,:]-self.path.points[:-1,:])
        indx, bz_a = self.get_bezier_index_from_length(s)
        points = self.path.get_nth_curve_points(indx)
        dpoints = (points[1:,:]-points[:-1,:])/3
        bzf = bezier(dpoints)
        point = bzf(bz_a)
        return normalize(point)

    def get_tangent_angle(self,s):
        tv = self.get_tangent_unit_vector(s)
        return angle_of_vector(tv)

    def get_normal_unit_vector(self,s):
        tv = self.get_tangent_unit_vector(s)
        return rotate_vector(tv,PI/2)

    def get_curvature_vector(self,s):
        indx, bz_a = self.get_bezier_index_from_length(s)
        points = self.path.get_nth_curve_points(indx)
        dpoints = (points[1:, :] - points[:-1, :]) * 3
        ddpoints = (dpoints[1:, :] - dpoints[:-1, :]) * 2
        deriv = bezier(dpoints)(bz_a)
        dderiv = bezier(ddpoints)(bz_a)
        curv = np.cross(deriv, dderiv) / (np.linalg.norm(deriv)**3)
        return curv

    def get_curvature(self,s):
        return np.linalg.norm(self.get_curvature_vector(s))

# DashedVMobject

class Dashed_line_mobject(VDict):
    def __init__(self,target_mobject:VMobject,
                 num_dashes=15,
                 dashed_ratio=0.5,
                 dash_offset=0.0,**kwargs):
        super().__init__(**kwargs)
        self['path'] = Path_mapper(target_mobject,num_of_path_points=10*target_mobject.get_num_curves())
        # self['path'].add_updater(lambda mob: mob.generate_length_map())

        dshgrp = self.generate_dash_mobjects(
            **self.generate_dash_pattern_dash_distributed(num_dashes,dash_ratio = dashed_ratio,offset=dash_offset)
        )
        self.add({'dashes':dshgrp})

    def generate_dash_pattern_metric(self,dash_len,space_len, num_dashes, offset=0):
        ''' generate dash pattern in metric curve-length space'''
        period = dash_len + space_len
        n = num_dashes
        full_len = self['path'].get_path_length()
        dash_starts = [(i * period + offset) for i in range(n)]
        dash_ends = [(i * period + dash_len + offset) for i in range(n)]
        k=0
        while k<len(dash_ends):
            if dash_ends[k]<0:
                dash_ends.pop(k)
                dash_starts.pop(k)
            k+=1

        k = 0
        while k < len(dash_ends):
            if dash_starts[k] > full_len:
                dash_ends.pop(k)
                dash_starts.pop(k)
            k+=1
        return {'dash_starts':dash_starts,'dash_ends':dash_ends}

    def generate_dash_pattern_dash_distributed(self,num_dashes,dash_ratio = 0.5,offset=0.0):
        full_len = self['path'].get_path_length()
        period = full_len / num_dashes
        dash_len = period * dash_ratio
        space_len = period * (1-dash_ratio)
        n = num_dashes+2

        return self.generate_dash_pattern_metric(dash_len, space_len, n, offset=(offset-1)*period)

    def generate_dash_mobjects(self,dash_starts=[0],dash_ends=[1]):
        ref_mob = self['path'].path
        a_list = self['path'].alpha_from_length(dash_starts)
        b_list = self['path'].alpha_from_length(dash_ends)
        ret=[]
        for i in range(len(dash_starts)):
            mobcopy = VMobject().match_points(ref_mob)
            ret.append(mobcopy.pointwise_become_partial(mobcopy,a_list[i],b_list[i]))
        return VGroup(*ret)


class Path_Offset_Mobject(VMobject):
    def __init__(self,target_mobject, ofs_func,ofs_func_kwargs={}, num_of_samples=100, **kwargs):
        super().__init__(**kwargs)
        self.ofs_func_kwargs = ofs_func_kwargs
        self.PM = Path_mapper(target_mobject)
        self.PM.add_updater(lambda mob: mob.generate_length_map())
        self.t_range = np.linspace(0, 1, num_of_samples)
        self.ofs_func = ofs_func
        self.s_scaling_factor = 1/self.PM.get_path_length()
        self.points = self.generate_offset_paths()

    # this can be useful in lambdas for updaters
    def set_ofs_function_kwargs(self,ofs_func_kwargs):
        self.ofs_func_kwargs = ofs_func_kwargs

    def generate_bezier_points(self, input_func,t_range, Smoothing=True):
        # generate bezier 4-lets with numerical difference
        out_data = []
        for k in range(len(t_range)-1):
            t = t_range[k]
            t2 = t_range[k+1]
            val1 = input_func(t)
            val2 = input_func(t2)
            p1 = val1
            p4 = val2
            if Smoothing:
                diff1 = (input_func(t + 1e-6) - input_func(t)) / 1e-6
                diff2 = (input_func(t2) - input_func(t2 - 1e-6)) / 1e-6
                p2 = val1 + diff1 * (t2 - t) / 3
                p3 = val2 - diff2 * (t2 - t) / 3
            else:
                p2 = (val1 * 2 + val2) / 3
                p3 = (val1 + val2 * 2) / 3
            out_data.append([p1,p2,p3,p4])
        return out_data


    def generate_ref_curve(self):
        self.ref_curve = VMobject()
        bez_point = self.generate_bezier_points(self.PM.point_from_proportion,self.t_range)
        for point in bez_point:
            self.ref_curve.append_points(point)
        self.ref_curve_path = Path_mapper(self.ref_curve)

    def generate_offset_func_points(self,Smoothing=True):
        points = self.generate_bezier_points(lambda t: self.ofs_func(t,**self.ofs_func_kwargs),self.t_range,Smoothing=Smoothing)
        return points

    def generate_normal_vectors(self):
        s_range = self.t_range*self.PM.get_path_length()
        # generate normal vectors from tangent angles and turning them 90°
        # the angles can be interpolated with bezier, unit normal vectors would not remain 'unit' under interpolation
        angles = self.generate_bezier_points(self.PM.get_tangent_angle,s_range)
        out = []
        for angle in angles:
            out.append([np.array([-np.sin(a),np.cos(a),0])for a in angle])
        return out

    def generate_offset_paths(self,gen_ofs_point=True, gen_ref_curve=True):
        if gen_ref_curve:
            self.generate_ref_curve()
            self.norm_vectors = self.generate_normal_vectors()
        if gen_ofs_point:
            self.ofs_points = self.generate_offset_func_points()

        n = self.ref_curve.get_num_curves()
        ofs_vectors = np.empty((n*4,3))
        for k in range(len(self.ofs_points)):
            for j in range(len(self.ofs_points[k])):
                ofs_vectors[k*4+j,:] = self.norm_vectors[k][j] * self.ofs_points[k][j]

        return self.ref_curve.points + ofs_vectors

    def default_updater(self,gen_ofs_point=True, gen_ref_curve=True):
        self.points = self.generate_offset_paths(gen_ofs_point,gen_ref_curve)


class Curve_Warp(VMobject):
    def __init__(self,warp_source:VMobject,warp_curve:VMobject,anchor_point=0.5, **kwargs):

        self.warp_curve = warp_curve
        self.warp_source = warp_source
        self.PM = Path_mapper(self.warp_curve)
        self.anchor_point = anchor_point
        super().__init__(**kwargs)
        self.match_style(warp_source)

    def generate_points(self):

        s0 = self.PM.length_from_alpha(self.anchor_point)
        x_points = self.warp_source.points[:, 0] + s0
        y_points = self.warp_source.points[:, 1]
        L = self.PM.get_path_length()

        if self.warp_curve.is_closed():
            #if the curve is closed, out-of-bound x values can wrap around to the beginning
            x_points = x_points % L
            x_curve_points = self.PM.point_from_proportion(x_points/L)
            nv = [self.PM.get_normal_unit_vector(x) for x in x_points]
            y_curve_points = np.array( [tuplie[0] * tuplie[1] for tuplie in zip(nv,y_points)])
            self.points = x_curve_points + y_curve_points
        else:
            self.points = np.empty((0,3))
            for x,y in zip(x_points,y_points):
                if 0 < x < L:
                    p = self.PM.point_from_proportion(x / L) + self.PM.get_normal_unit_vector(x)*y
                    self.points = np.append(self.points,np.reshape(p,(1,3)),axis=0)
                elif x>L:
                    endpoint = self.PM.point_from_proportion(1)
                    tanv = self.PM.get_tangent_unit_vector(L)
                    nv = rotate_vector(tanv,PI/2)
                    x_1 = x-L
                    p = endpoint + x_1 * tanv + y * nv
                    self.points = np.append(self.points, np.reshape(p,(1,3)), axis=0)
                else:
                    startpoint = self.PM.point_from_proportion(0)
                    tanv = self.PM.get_tangent_unit_vector(0)
                    nv = rotate_vector(tanv, PI / 2)
                    x_1 = x
                    p = startpoint + x_1 * tanv + y * nv
                    self.points = np.append(self.points, np.reshape(p, (1, 3)), axis=0)



class Pointer_Label_Free(VDict):
    def __init__(self,point, text:str, offset_vector=(RIGHT+DOWN),**kwargs):
        text_buff = 0.1
        if not 'stroke_width' in kwargs:
            kwargs['stroke_width'] = DEFAULT_STROKE_WIDTH
        super().__init__(**kwargs)
        self.add({'arrow': Arrow(start=point + offset_vector, end=point, buff=0, **kwargs)})

        if isinstance(text,str):
            textmob = Text(text,**kwargs)
        elif isinstance(text,Mobject):
            textmob = text
        else:
            textmob = Text('A',**kwargs)
        self.add({'text': textmob})


        self.twidth = (self['text'].get_critical_point(RIGHT)-self['text'].get_critical_point(LEFT))[0]
        self.twidth = self.twidth + text_buff * 2
        self.add({'line':Line(start=point+offset_vector,
                              end=point+offset_vector+self.twidth*np.sign(offset_vector[0]*RIGHT))})

        theight = (self['text'].get_critical_point(UP)-self['text'].get_critical_point(DOWN))[1]
        self['text'].move_to(self['line'].get_center()+UP*theight*0.75)


    def update_point(self, point, offset_vector=(RIGHT+DOWN)):
        self['arrow'].put_start_and_end_on(start=point+offset_vector, end=point)
        self['line'].put_start_and_end_on(start=point+offset_vector,
                                          end=point + offset_vector + self.twidth * np.sign(offset_vector[0] * RIGHT))
        # self['arrow'].add_tip(self['arrow'].start_tip, at_start=True)

        theight = (self['text'].get_critical_point(UP)-self['text'].get_critical_point(DOWN))[1]
        self['text'].move_to(self['line'].get_center()+UP*theight*0.75)


class Pointer_To_Mob(Pointer_Label_Free):
    def __init__(self, mob:Mobject, proportion,  text:str, dist=1, **kwargs):
        point = mob.point_from_proportion(proportion)

        # if the mob center and point happens to be the same, it causes problems
        # it can happen if all the mob is 1 point
        offset_ref = point - mob.get_center()
        if np.linalg.norm(offset_ref)>1e-6:
            offset = normalize(point - mob.get_center())*dist
        else:
            # I had no better idea to handle this than to go upright
            offset = normalize(RIGHT+UP)*dist
        super().__init__(point,text, offset_vector=offset,**kwargs)

    def update_mob(self,mob, proportion, dist=1):
        point = mob.point_from_proportion(proportion)
        # if the mob center and point happens to be the same, it causes problems
        # it can happen if all the mob is 1 point
        offset_ref = point - mob.get_center()
        if np.linalg.norm(offset_ref) > 1e-6:
            offset = normalize(point - mob.get_center()) * dist
        else:
            # I had no better idea to handle this than to go upright
            offset = normalize(RIGHT + UP) * dist
        super().update_point(point,offset)


class Linear_Dimension(VDict):
    def __init__(self, start,end, text=None,direction=ORIGIN, outside_arrow=False, offset=2, **kwargs):
        super().__init__(**kwargs)
        diff_vect = end-start
        norm_vect = normalize(rotate_vector(diff_vect,PI/2))
        if not any(direction!=0):
            ofs_vect = norm_vect * offset
            ofs_dir = norm_vect
        else:
            ofs_vect = direction * offset
            ofs_dir = direction
        if not 'stroke_width' in kwargs:
            kwargs['stroke_width'] = DEFAULT_STROKE_WIDTH

        startpoint = start + ofs_dir * np.dot(end-start,ofs_dir)/2+ofs_vect
        endpoint = end - ofs_dir * np.dot(end - start, ofs_dir) / 2 + ofs_vect

        tip_len=0.2

        if not outside_arrow:
            main_line = Arrow(start=startpoint, end=endpoint, buff=0,
                              max_tip_length_to_length_ratio=1,
                              max_stroke_width_to_length_ratio=1000,
                              tip_length=tip_len,
                              **kwargs)
            main_line.add_tip(at_start=True,tip_length=tip_len)
        else:
            main_line = Line(start=startpoint, end=endpoint,**kwargs)
            arrow_line1 = Arrow(end=startpoint,
                                start=startpoint+tip_len*3*(normalize(startpoint-endpoint)),
                                buff=0,
                                max_tip_length_to_length_ratio=1,
                                max_stroke_width_to_length_ratio=1000,
                                tip_length=tip_len,
                                **kwargs
                               )
            arrow_line2 = Arrow(end=endpoint,
                                start=endpoint-tip_len*3*(normalize(startpoint-endpoint)),
                                buff=0,
                                max_tip_length_to_length_ratio=1,
                                max_stroke_width_to_length_ratio=1000,
                                tip_length=tip_len,
                                **kwargs
                               )
            main_line.add(arrow_line1)
            main_line.add(arrow_line2)

        self.add({'main_line': main_line})
        self.add({'ext_line_1': Line(start=start,
                                     end=startpoint + 0.25 * (normalize(startpoint-start)),
                                     **kwargs)})
        self.add({'ext_line_2': Line(start=end,
                                     end=endpoint + 0.25 * (normalize(endpoint-end)),
                                     **kwargs)})

        if isinstance(text,str):
            textmob = Text(text)
        elif isinstance(text,Mobject):
            textmob = text
        else:
            dist = np.linalg.norm(main_line.start-main_line.end)
            textmob = Text(f"{dist:.2}",**kwargs)

        angle = (main_line.get_angle()+PI/2)%PI-PI/2
        if abs(angle+PI/2)<1e-8:
            angle=PI/2
        self.text_h = textmob.height
        textmob.rotate(angle)
        textmob.move_to(self.submobjects[0].get_center() + rotate_vector(UP,angle)*self.text_h)
        self.add({'text': textmob})


class Angle_Dimension_3point(VGroup):
    def __init__(self,start,end, arc_center,offset=2,text=None, outside_arrow=False,**kwargs):
        super().__init__(**kwargs)
        if not 'stroke_width' in kwargs:
            kwargs['stroke_width'] = DEFAULT_STROKE_WIDTH

        self.angle = angle_between_vectors_signed(start-arc_center,end-arc_center)
        radius = (np.linalg.norm(start-arc_center)+np.linalg.norm(end-arc_center))/2 + offset
        angle_0 = angle_of_vector(start-arc_center)
        angle_1 = angle_between_vectors_signed(start-arc_center,end-arc_center)

        tip_len = 0.2
        base_arc = Arc(radius=radius,
                       start_angle=angle_0,
                       arc_center=arc_center,
                       angle=angle_1,
                       **kwargs)
        arc_p0 = base_arc.point_from_proportion(0)
        arc_p1 = base_arc.point_from_proportion(1)
        line1 = Line(start=start,
                     end=arc_p0 + normalize(arc_p0-start)*tip_len,
                     **kwargs
        )
        line2 = Line(start=end,
                     end=arc_p1 + normalize(arc_p1-end)*tip_len,
                     **kwargs
                     )
        self.add(line1,line2)

        if not outside_arrow:

            base_arc.add_tip(tip_length=tip_len)
            base_arc.add_tip(tip_length=tip_len,at_start=True)
            self.add(base_arc)
        else:
            angle_ext = tip_len*3/radius * np.sign(angle_1)
            ext_arc_1 = Arc(radius=radius,
                            start_angle=angle_0-angle_ext,
                            angle=+angle_ext,
                            arc_center=arc_center,
                            **kwargs)
            ext_arc_2 = Arc(radius=radius,
                            start_angle=(angle_0 + angle_1 + angle_ext)%TAU,
                            angle=-angle_ext,
                            arc_center=arc_center,
                            **kwargs)
            ext_arc_1.add_tip(tip_length=tip_len)
            ext_arc_2.add_tip(tip_length=tip_len)
            base_arc.add(ext_arc_1,ext_arc_2)
            self.add(base_arc)

        if isinstance(text,str):
            textmob = Text(text)
        elif isinstance(text,Mobject):
            textmob = text
        else:
            textmob = Text(f"{abs(angle_1/DEGREES):.2f}",**kwargs)

        pos_text = base_arc.point_from_proportion(0.5)
        angle_text = (angle_of_vector(base_arc.point_from_proportion(0.5+1e-6) -
                                      (base_arc.point_from_proportion(0.5-1e-6))) + PI / 2) % PI - PI / 2
        if abs(angle_text+PI/2)<1e-8:
            angle_text=PI/2
        self.text_h = textmob.height
        textmob.rotate(angle_text)
        textmob.move_to(pos_text + rotate_vector(UP,angle_text)*self.text_h)
        self.add(textmob)





####################################                                  #####################################
#################################### END OF ROUND CORNER CONSTRUCTION #####################################
####################################                                  #####################################