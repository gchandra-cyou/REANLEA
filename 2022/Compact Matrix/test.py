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
from manim import*
from math import*
from manim_fonts import*
import numpy as np
import random
from sklearn.datasets import make_blobs
from reanlea_colors import*

config.background_color= REANLEA_BACKGROUND_COLOR


###############################################################################################################

class LineExample(Scene):
                def construct(self):
                    d = VGroup()
                    for i in range(0,10):
                        d.add(Dot())
                    d.arrange_in_grid(buff=1)
                    self.add(d)
                    l= Line(d[0], d[1])
                    self.add(l)
                    self.wait()
                    l.put_start_and_end_on(d[1].get_center(), d[2].get_center())
                    self.wait()
                    l.put_start_and_end_on(d[4].get_center(), d[7].get_center())
                    self.wait()


                    # manim -pqh test.py LineExample