############################################# by GOBINDA CHANDRA ###################################################

                                    # VISIT    : https://reanlea.com/ 
                                    # YouTube  : https://www.youtube.com/Reanlea/ 
                                    # Twitter  : https://twitter.com/Reanlea_ 

####################################################################################################################

from __future__ import annotations
from ast import Return
from cProfile import label
from calendar import c
from difflib import restore


import fractions
from imp import create_dynamic
import math
from multiprocessing import context
from multiprocessing import dummy
from multiprocessing.dummy import Value
from numbers import Number
import sre_compile
from tkinter import Y, Label, font
from imp import create_dynamic
from tracemalloc import start
from turtle import degrees, width
from manim import*
from math import*
from manim_fonts import*
import numpy as np
import random
from sklearn.datasets import make_blobs
from reanlea_colors import*
from func import*
from PIL import Image
#from func import EmojiImageMobject

config.max_files_cached=500

config.background_color= REANLEA_BACKGROUND_COLOR


###################################################################################################################


class reanlea_logo(Scene):
    def construct(self):

        # WATER-MARK
        water_mark=ImageMobject("watermark.png").scale(0.1).move_to(5*LEFT+3*UP).set_opacity(0.15)

        self.add(water_mark)



        cloud = PointCloudDot(radius=2,color=REANLEA_WHITE)
        logo=ImageMobject("watermark.png").shift(1.25*LEFT).scale(0.135).set_z_index(-1)
        self.wait()

        self.play(
            FadeIn(cloud)
        )
        self.wait()

        self.add_sound("piano.mp3")

        self.play(
            cloud.animate.apply_complex_function(lambda z: np.exp(z)),
            FadeIn(logo, run_time=2)
        )

        self.wait(2)

        self.play(
            Uncreate(cloud),
            FadeOut(logo)
        )



        self.wait(4)

        # manim -pqk logo.py reanlea_logo

        # manim -pqh logo.py reanlea_logo

        # manim -sqk logo.py reanlea_logo


###################################################################################################################

# Changing FONTS : import any font from Google
# some of my fav fonts: Cinzel,Kalam,Prata,Kaushan Script,Cormorant, Handlee,Monoton, Bad Script, Reenie Beanie, 
# Poiret One,Merienda,Julius Sans One,Merienda One,Cinzel Decorative, Montserrat, Cousine, Courier Prime , The Nautigal
# Marcellus SC,Contrail One,Thasadith,Spectral SC,Dongle,Cormorant SC,Comfortaa, Josefin Sans (LOVE), Fuzzy Bubbles

###################################################################################################################


                     #### completed on 15th March,2023 | 03:05am  ####


###################################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\Compact Matrix" 