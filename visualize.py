import re
from matplotlib.patches import Circle
from math import pow
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

class TestVisualizer:
  def __init__(self, image_path, actual_coords, pred_coords=None):
    self.image_path = image_path
    self.pred = pred_coords
    self.actual = actual_coords

  def draw_ellipse(self, im, val1, val2, color='red'):
    width, height = im.size
    radius = 5
    draw = ImageDraw.Draw(im)
    draw.ellipse((val1*width-radius, val2*height-radius, val1*width+radius, val2*height+radius), fill=color, outline=color)

    return

  def prep_image(self):
    im = Image.open(self.image_path)
    width, height = im.size
    val1, val2 = self.actual[0], self.actual[1]
    self.draw_ellipse(im, val1, val2, 'red')
    if self.pred != None:
      val1, val2 = self.pred[0], self.pred[1]
      self.draw_ellipse(im, val1, val2, 'blue')

    return im

  def get_error(self):
    if self.pred == None:
      self.pred = [0, 0]

    return pow(pow(self.pred[0]-self.actual[0], 2) + pow(self.pred[1] - self.actual[1], 2), 1/2)

  def show_image(self):
    im = self.prep_image()
    im.show()

  def save_image(self, save_path):
    im = self.prep_image()

    dirs = re.split("/", save_path)
    directory = ""

    for i in range(len(dirs)):
        directory += dirs[i] + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)

    res = re.split("[/.]", self.image_path)
    im.save(save_path+res[-2] + "_plot.jpg")