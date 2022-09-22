from __future__ import annotations
from asyncore import poll2
from audioop import add
from cProfile import label
from distutils import text_file


import fractions
from imp import create_dynamic
from multiprocessing import context
from multiprocessing.dummy import Value
from numbers import Number
from operator import is_not
from pickle import TRUE
from tkinter import CENTER, Y, Label, Scale
from imp import create_dynamic
from tracemalloc import start
from turtle import degrees, dot, width
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


config.background_color= REANLEA_BACKGROUND_COLOR


##################################################################################################################

# visit : https://github.com/huncut2016/videos/blob/master/fermats_point/index.py



class Index(Scene):
    def construct(self):
        triangle = (
            Polygon([5, -2, 0], [-1, 4, 0], [-3, -2, 0])
            .scale(0.5)
            .center()
            .shift(1.6 * UP)
        )

        p_coord = triangle.get_center()
        p = Dot(p_coord, color=RED)
        p_label = Tex("P").next_to(p, DOWN).set_color(RED)

        a_coord = triangle.get_boundary_point(LEFT)
        a = Dot(a_coord)
        a_label = Tex("A").next_to(a, DOWN)

        b_coord = triangle.get_boundary_point(RIGHT)
        b = Dot(b_coord)
        b_label = Tex("B").next_to(b, DOWN)

        c_coord = triangle.get_boundary_point(UP)
        c = Dot(c_coord)
        c_label = Tex("C").next_to(c, UP)

        ap = pa = DashedLine(a_coord, p_coord)
        bp = pb = DashedLine(b_coord, p_coord)
        cp = pc = DashedLine(c_coord, p_coord)

        dots = VGroup(a, b, c, p)
        d_labels = VGroup(a_label, b_label, c_label, p_label)
        lines = VGroup(ap, bp, cp)

        g = VGroup(triangle, dots, d_labels, lines)
        surr = SurroundingRectangle(g, color=WHITE)

        t = (
            MathTex(r"AP + BP + CP = min", substrings_to_isolate="P")
            .next_to(surr, DOWN)
            .shift(DOWN)
        )
        t.set_color_by_tex("P", RED)

        #self.add(g, t)
        self.play(
            Create(g),
            Create(t),
            run_time=2
        )
        self.add_foreground_mobject(p)
        self.wait(2)


        # manim -pqh srcgit.py Index




# VISIT : https://github.com/huncut2016/videos/blob/master/fermats_point/main.py





class Main(MovingCameraScene):
    def introduction(self):
        title = Text("Fermat point").center()
        mname = Text("András Zoller").scale(0.5).next_to(title, DOWN)
        self.play(Write(title))
        self.play(Write(mname))
        self.wait()
        self.play(FadeOut(VGroup(title, mname)))
        self.wait()

    def demonstration(self):
        triangle = Polygon([5, -2, 0], [-1, 4, 0], [-3, -2, 0]).center()

        p_coord = triangle.get_center()
        p = Dot(p_coord, color=TEAL)
        p_label = MathTex("P").next_to(p, LEFT).set_color(TEAL)

        a_coord = triangle.get_boundary_point(LEFT)
        a = Dot(a_coord)
        a_label = MathTex("A").next_to(a, DOWN)

        b_coord = triangle.get_boundary_point(RIGHT)
        b = Dot(b_coord)
        b_label = MathTex("B").next_to(b, DOWN)

        c_coord = triangle.get_boundary_point(UP)
        c = Dot(c_coord)
        c_label = MathTex("C").next_to(c, UP)

        ap = pa = DashedLine(a_coord, p_coord)
        bp = pb = DashedLine(b_coord, p_coord)
        cp = pc = DashedLine(c_coord, p_coord)

        a_side = (c_coord + b_coord) / 2 + 0.5 * self.perp(c_coord, b_coord)
        a_side_label = Tex("a").move_to(a_side)

        b_side = (a_coord + c_coord) / 2 + 0.5 * self.perp(a_coord, c_coord)
        b_side_label = Tex("b").move_to(b_side)

        c_side = (a_coord + b_coord) / 2 + 0.5 * self.perp(b_coord, a_coord)
        c_side_label = Tex("c").move_to(c_side)

        sides = VGroup(a_side_label, b_side_label, c_side_label)
        dots = VGroup(a, b, c, p)
        d_labels = VGroup(a_label, b_label, c_label, p_label)
        lines = VGroup(ap, bp, cp)

        g = VGroup(triangle, dots, d_labels, lines, sides)
        surr = SurroundingRectangle(g, color=WHITE)

        t = (
            MathTex("AP", "+", "BP", "+", "CP", "=", "min", substrings_to_isolate="P")
            .to_corner(UR)
            .shift(DL)
        )
        t.set_color_by_tex("P", RED)
        ts = SurroundingRectangle(t)

        self.play(Create(triangle))
        self.play(Write(d_labels - p_label), Create(dots - p))
        self.play(Create(sides))
        self.add_foreground_mobject(dots - p)
        self.wait()

        self.play(Create(lines))
        self.play(Create(p), Write(p_label))
        self.add_foreground_mobject(p)
        self.wait()

        for line, text_index in zip(lines, range(0, 9, 3)):
            text = t[text_index : text_index + 2]
            self.play(TransformFromCopy(line, text))
            self.play(Write(t[text_index + 2]))
            self.wait()

        self.play(Write(t[-1]))
        self.play(Create(ts))
        self.wait()

        self.play(
            FadeOut(VGroup(ts, t)),
            VGroup(lines, dots, d_labels, triangle, sides)
            .animate.scale(0.5)
            .shift(3 * LEFT),
        )

        inner_triangle: Polygon = (
            Polygon(c.get_center(), p.get_center(), b.get_center())
            .set_fill(TEAL, opacity=0.5)
            .set_stroke(TEAL, opacity=1)
        )

        self.play(FadeIn(inner_triangle))
        self.wait()

        ang = ValueTracker(0)

        def helper():

            # print(
            #     inner_triangle.get_vertices()[1],
            #     b.get_center(),
            #     p.get_center(),
            #     sep="   coord \n",
            # )
            # print("\n\n")
            # print(inner_triangle.get_vertices())

            an = Angle.from_three_points(
                inner_triangle.get_vertices()[1]
                * (1 - 1e-10),  # at the beginning P == P'
                b.get_center(),
                p.get_center(),
                radius=0.7,
            )

            ang.set_value(an.get_value(degrees=True))

            return an

        def mid_angle(angle, point):
            base = (angle.get_center() + point.get_center()) / 2
            shift = normalize(point.get_center() - angle.get_center())

            return base - 0.15 * shift

        angle = always_redraw(helper)

        _c = always_redraw(
            lambda: Dot(inner_triangle.get_boundary_point(UP)).scale(0.5)
        )
        _c_label = always_redraw(lambda: Tex("C'").scale(0.5).next_to(_c, RIGHT))

        _p = always_redraw(
            lambda: Dot(inner_triangle.get_vertices()[1], color=TEAL).scale(0.5)
        )

        _p_label = always_redraw(
            lambda: Tex("P'").scale(0.5).next_to(_p, LEFT).set_color(TEAL)
        )

        value = (
            Integer(ang.get_value(), unit="^{\circ}")
            .scale(0.5)
            .move_to(mid_angle(angle, b))
        )

        value.add_updater(lambda d: d.move_to(mid_angle(angle, b)))
        value.add_updater(lambda d: d.set_value(ang.get_value()))

        self.add(angle, _c, _c_label, value, ang, _p, _p_label)
        self.add_foreground_mobject(value)
        self.play(
            Rotate(inner_triangle, -60 * DEGREES, about_point=b.get_center()),
            run_time=2,
        )
        angle.clear_updaters()
        value.clear_updaters()

        self.wait()

        p_p = _pp = Line(p.get_center(), _p.get_center(), color=YELLOW_D)

        self.play(Create(p_p))
        self.wait()

        t1 = MathTex("B", "P'", "=", "B", "P").to_edge(RIGHT).shift(LEFT + 2.5 * UP)
        t1[1].set_color(TEAL)
        t1[-1].set_color(TEAL)

        b_p = _pb = Line(b, _p.get_center(), color=TEAL).set_opacity(0.5)

        self.play(TransformFromCopy(bp, t1[:3]))
        self.wait()
        self.play(TransformFromCopy(b_p, t1[3:]))
        self.wait()

        d_arrow = Arrow(t1.get_bottom(), t1.get_bottom() + DOWN)

        self.play(GrowArrow(d_arrow))

        t2 = MathTex("B", "PP'", "=", "B", "P'P").next_to(d_arrow, DOWN)
        t2.set_color_by_tex("P", TEAL)

        corner1 = Group(bp, p_p).copy()
        corner2 = Group(b_p, p_p).copy()

        self.play(
            corner1.animate.scale(2),
            Indicate(t2[:2], scale_factor=2),
        )
        self.play(corner1.animate.scale(0.5))

        self.wait()
        self.play(Write(t2[2]))
        self.wait()

        self.play(
            corner2.animate.scale(2),
            Indicate(t2[3:], scale_factor=2),
        )
        self.play(corner2.animate.scale(0.5))
        self.wait()

        self.remove(corner1, corner2)

        t3 = (
            MathTex(
                "B",
                "PP'",
                "+",
                "B",
                "P'P",
                r"&= 180^{\circ} -",
                "P",
                "B",
                "P'",
                r" \\ &= 120^{\circ}",
            )
            .next_to(t2, DOWN)
            .shift(0.6 * LEFT)
        )
        t3.set_color_by_tex("P", TEAL)

        bp_p = Angle.from_three_points(b.get_center(), p.get_center(), _p.get_center())
        b_pp = Angle.from_three_points(p.get_center(), _p.get_center(), b.get_center())

        self.play(FadeIn(t3[:2]), Create(bp_p))
        self.wait()

        self.play(Write(t3[2]))

        self.play(FadeIn(t3[3:5]), Create(b_pp))
        self.wait()

        self.play(Write(t3[5:]))

        self.wait()

        d2_arrow = Arrow(t3.get_bottom(), t3.get_bottom() + DOWN)

        self.play(GrowArrow(d2_arrow))

        t4 = MathTex("B", "PP'", "=", "B", "P'P", " = 60^{\circ}").next_to(
            d2_arrow, DOWN
        )
        t4.set_color_by_tex("P", TEAL)

        self.play(
            FadeIn(t4),
            VGroup(bp_p, b_pp, angle, value).animate.set_color(ORANGE).scale(2),
        )
        self.play(VGroup(bp_p, b_pp, angle, value).animate.scale(0.5))
        self.wait()

        b_p.set_color(YELLOW_D).set_opacity(1)

        t5 = MathTex("B", "P", "=", "PP'").to_edge(RIGHT).shift(2 * LEFT)
        t5.set_color_by_tex("P", TEAL)

        self.play(
            Create(b_p),
            Transform(bp, Line(p.get_center(), b.get_center(), color=YELLOW_D)),
            FadeOut(VGroup(t1, t2, t3, t4, d2_arrow, d_arrow), shift=UP),
            FadeIn(t5, shift=UP),
        )
        self.wait()

        self.play(
            t5.animate.shift(3 * UP),
        )

        # proof = BulletedList(*("If $APP'C'$ is a straight line the is minimum".split()))
        proof = (
            r"$A$, $B$, and $C'$ are fixed",
            r"$AB + BC'$ has a constant size",
            (r"If $A$", r"$PP'$", r"$C'$ is a straight line \\ the sum is minimum"),
        )

        b_c = _cb = Line(b.get_center(), _c.get_center(), color=TEAL).set_opacity(1)
        ab = ba = Line(a.get_center(), b.get_center(), color=BLUE)
        corner = (a, a_label, b, b_label, _c, _c_label)
        a_c = _ca = Line(a.get_center(), _c.get_center(), color=PINK)

        highlights = (
            AnimationGroup(
                *[Indicate(i, scale_factor=2, color=PURE_RED) for i in corner],
            ),
            VGroup(ab, b_c).animate(rate_func=there_and_back).scale(1.2),
            Create(
                _ca
                # rate_func=there_and_back,
            ),
        )

        proofs = VGroup()

        for index, (text, highlight) in enumerate(zip(proof, highlights)):
            tex: Tex = None

            if isinstance(text, tuple):
                tex = Tex(*text)
            else:
                tex = Tex(text)

            tex.scale(0.7).next_to(t5, DOWN, buff=1).shift(index * DOWN)
            tex.set_color_by_tex("P", TEAL)

            proofs.add(tex)

            self.play(FadeIn(tex, shift=DOWN), highlight, run_time=4)
            self.wait()

        self.play(Uncreate(a_c))
        self.wait()
        self.remove(b_c)

        # add dynamic scene

        self.remove(
            angle,
            value,
            inner_triangle,
            b_pp,
            b_pp,
            _p_label,
            _p,
            p_p,
            b_p,
            ap,
            bp,
            cp,
            bp_p,
        )

        p_label.add_updater(lambda m: m.next_to(p, LEFT))

        ap = always_redraw(lambda: DashedLine(a.get_center(), p.get_center()))
        cp = always_redraw(lambda: DashedLine(c.get_center(), p.get_center()))
        bp = always_redraw(lambda: Line(b.get_center(), p.get_center(), color=YELLOW_D))
        b_p = always_redraw(
            lambda: Line(b.get_center(), _p.get_center(), color=YELLOW_D)
        )

        p_p = always_redraw(
            lambda: Line(p.get_center(), _p.get_center(), color=YELLOW_D)
        )

        _p = always_redraw(
            lambda: Dot(inner_triangle.get_vertices()[1], color=TEAL).scale(0.5)
        )

        _p_label = always_redraw(
            lambda: Tex("P'").scale(0.5).next_to(_p, LEFT).set_color(TEAL)
        )

        bp_p = always_redraw(
            lambda: Angle.from_three_points(
                b.get_center(), p.get_center(), _p.get_center(), color=ORANGE
            )
        )
        b_pp = always_redraw(
            lambda: Angle.from_three_points(
                p.get_center(), _p.get_center(), b.get_center(), color=ORANGE
            )
        )

        inner_triangle = always_redraw(
            lambda: (
                Polygon(c.get_center(), p.get_center(), b.get_center())
                .rotate(-60 * DEGREES, about_point=b.get_center())
                .set_fill(TEAL, opacity=0.5)
                .set_stroke(TEAL, opacity=1)
            )
        )

        angle = always_redraw(
            lambda: Angle.from_three_points(
                inner_triangle.get_vertices()[1],
                b.get_center(),
                p.get_center(),
                radius=0.7,
                color=ORANGE,
            )
        )

        value = (
            Integer(60, unit="^{\circ}", color=ORANGE)
            .scale(0.5)
            .move_to(mid_angle(angle, b))
        )

        value.add_updater(lambda d: d.move_to(mid_angle(angle, b)))
        value.add_updater(lambda d: d.set_value(ang.get_value()))

        dy_values = (
            angle,
            value,
            inner_triangle,
            b_pp,
            b_pp,
            _p_label,
            _p,
            p_p,
            b_p,
            ap,
            bp,
            cp,
            bp_p,
        )

        self.add(*dy_values)

        self.play(p.animate.shift(LEFT))
        self.play(p.animate.shift(DOWN))
        self.play(p.animate.shift(RIGHT * 2))
        self.play(p.animate.shift(LEFT))
        self.play(p.animate.shift(UP))
        self.play(
            p.animate.move_to(
                self.calculate_fermats_point(
                    a.get_center(), b.get_center(), c.get_center()
                )
            )
        )
        self.wait()

        for i in dy_values:
            i.clear_updaters()

        self.play(
            Create(
                Line(a.get_center(), _c.get_center(), color=PINK),
                rate_func=there_and_back,
            )
        )
        self.wait()

        self.camera.frame.save_state()

        self.play(
            self.camera.frame.animate.move_to(p.get_center()).set(width=4),
        )
        apb = Angle.from_three_points(
            a.get_center(), p.get_center(), b.get_center(), radius=0.6
        )
        apb_label = MathTex(r"120 ^{\circ}").scale(0.5).move_to(mid_angle(apb, p))

        self.play(Create(apb), FadeIn(apb_label))
        self.wait()
        self.play(Restore(self.camera.frame))
        self.wait()

        self.camera.frame.save_state()

        self.play(
            self.camera.frame.animate.move_to(_p.get_center()).set(width=4),
        )

        b_p_c = Angle.from_three_points(
            b.get_center(), _p.get_center(), _c.get_center(), radius=0.6
        )
        b_p_c_label = MathTex(r"120 ^{\circ}").scale(0.5).move_to(mid_angle(b_p_c, _p))

        self.play(Create(b_p_c), FadeIn(b_p_c_label))
        self.wait()
        self.play(Restore(self.camera.frame))
        self.wait()

        bpc = Angle.from_three_points(
            b.get_center(), p.get_center(), c.get_center(), radius=0.6
        )
        bpc_label = MathTex(r"120 ^{\circ}").scale(0.5).move_to(mid_angle(bpc, p))

        self.play(Create(bpc), FadeIn(bpc_label))
        self.wait()
        self.play(
            *[Indicate(i) for i in (bpc, bpc_label, b_p_c, b_p_c_label)],
        )
        self.wait()

        cpa = Angle.from_three_points(
            c.get_center(), p.get_center(), a.get_center(), radius=0.6
        )
        cpa_label = MathTex(r"120 ^{\circ}").scale(0.5).move_to(mid_angle(cpa, p))

        self.camera.frame.save_state()

        self.play(
            self.camera.frame.animate.move_to(p.get_center()).set(width=4),
        )

        self.play(
            Create(cpa),
            FadeIn(cpa_label),
            # remove stuff
            *[FadeOut(i) for i in (p_label, b_p_c, b_p_c_label, bp_p, b_pp)],
        )

        self.wait()
        self.play(Restore(self.camera.frame))
        self.wait()

        sides = VGroup(a_side_label, b_side_label, c_side_label)
        dots = VGroup(a, b, c)
        d_labels = VGroup(a_label, b_label, c_label)
        lines = VGroup(ap, bp, cp)
        self.add_foreground_mobjects(*(dots + d_labels))

        self.remove(ab, b_c, b_p)
        self.play(Transform(bp, DashedLine(b.get_center(), p.get_center())))
        self.play(
            *[
                FadeOut(i)
                for i in (
                    proofs,
                    t5,
                    _c_label,
                    _p_label,
                    c,
                    p,
                    _p,
                    _c,
                    inner_triangle,
                    p_p,
                    angle,
                    value,
                )
            ],
            VGroup(
                lines,
                dots,
                d_labels,
                triangle,
                sides,
                apb,
                apb_label,
                cpa,
                cpa_label,
                bpc,
                bpc_label,
            )
            .animate.shift(3 * RIGHT)
            .scale(2),
        )

        t6 = Tex("The Fermat point").to_corner(UR).shift(DL)
        t6_u = Underline(t6, color=YELLOW)

        self.play(Write(t6))
        self.wait()
        self.play(ShowPassingFlash(t6_u, 0.5))
        self.wait()
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait()

    def ending(self):
        t = Tex("Thanks for watching!")
        self.play(Write(t))
        self.play(FocusOn(t))
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    def construct(self):
        self.introduction()
        self.demonstration()
        self.ending()

    def calculate_fermats_point(self, a, b, c):
        OMEGA = (c + b) / 2 + self.perp(c, b) * np.sqrt(3 / 4) * np.linalg.norm(c - b)
        LAMBDA = (a + b) / 2 + self.perp(b, a) * np.sqrt(3 / 4) * np.linalg.norm(a - b)

        return line_intersection((OMEGA, a), (LAMBDA, c))

    def perp(self, v1, v2):
        return normalize(np.cross(v1 - v2, OUT))



        # manim -pqh srcgit.py Main