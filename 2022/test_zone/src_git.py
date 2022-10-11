import sys
import copy
#sys.path.insert(1, "C:/Users/gchan/Desktop/REANLEA/2022/common")
#sys.path.insert(1, "C:\\Users\\gchan\\Desktop\\REANLEA\\2022\\common")
#sys.path.insert(1, r"C:\Users\gchan\Desktop\REANLEA\2022\common")
sys.path.insert(1,"common/")


from manim import*
#from reanlea_colors import*

config.background_color="#0b2149"


##########################################################################################################

class Ex(Scene):
    def construct(self):
        
        tex=Text("Gobinda Chandra")

        self.play(Create(tex))


        # manim -pqh src_git.py Ex 






##########################################################################################################

# cd "C:\Users\gchan\Desktop\REANLEA\2022\test_zone"



