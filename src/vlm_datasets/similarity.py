
import numpy as np
import string
from PIL import Image, ImageDraw, ImageFont
import colorsys


first_12 = string.ascii_uppercase[:12]
all_letters = string.ascii_uppercase
all_colors = np.linspace(0,360,num=101)[:-1]
font_size = 50
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
CANVAS_SIZE = (500, 500)
STENCIL_SIZE = 50

def make_image_4items(random_colors):
    canvas = Image.new('RGB', CANVAS_SIZE, 'white')
    draw = ImageDraw.Draw(canvas)
    for k in range(4):
        row = k // 4
        col = k % 4
        hsv_color = (random_colors[k], 1.0, 1.0)  # Green in HSV
        rgb_color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hsv_color[0] / 360, hsv_color[1], hsv_color[2]))
        x = [75,175,275,375][col]
        y = [225][row]
        rect_position = (x, y, x+50, y+50)  # (x1, y1, x2, y2)
        draw.rectangle(rect_position, fill=rgb_color)
        text_bbox = font.getbbox(first_12[k])
        deltax = (STENCIL_SIZE - (text_bbox[2] - text_bbox[0]))//2
        draw.text((x+deltax,y+50), first_12[k], fill="black", font=font)
    return canvas

def make_image_6items(random_colors):
    canvas = Image.new('RGB', CANVAS_SIZE, 'white')
    draw = ImageDraw.Draw(canvas)
    for k in range(6):
        row = k // 3
        col = k % 3
        hsv_color = (random_colors[k], 1.0, 1.0)  # Green in HSV
        rgb_color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hsv_color[0] / 360, hsv_color[1], hsv_color[2]))
        x = [100,225,350][col]
        y = [125,275][row]
        rect_position = (x, y, x+50, y+50)  # (x1, y1, x2, y2)
        draw.rectangle(rect_position, fill=rgb_color)
        text_bbox = font.getbbox(first_12[k])
        deltax = (STENCIL_SIZE - (text_bbox[2] - text_bbox[0]))//2
        draw.text((x+deltax,y+50), first_12[k], fill="black", font=font)
    return canvas

def make_image_8items(random_colors):
    canvas = Image.new('RGB', CANVAS_SIZE, 'white')
    draw = ImageDraw.Draw(canvas)
    for k in range(8):
        row = k // 4
        col = k % 4
        hsv_color = (random_colors[k], 1.0, 1.0)  # Green in HSV
        rgb_color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hsv_color[0] / 360, hsv_color[1], hsv_color[2]))
        x = [75,175,275,375][col]
        y = [125,275][row]
        rect_position = (x, y, x+50, y+50)  # (x1, y1, x2, y2)
        draw.rectangle(rect_position, fill=rgb_color)
        text_bbox = font.getbbox(first_12[k])
        deltax = (STENCIL_SIZE - (text_bbox[2] - text_bbox[0]))//2
        draw.text((x+deltax,y+50), first_12[k], fill="black", font=font)
    return canvas

def make_image_10items(random_colors):
    canvas = Image.new('RGB', CANVAS_SIZE, 'white')
    draw = ImageDraw.Draw(canvas)
    for k in range(10):
        row = k // 5
        col = k % 5
        hsv_color = (random_colors[k], 1.0, 1.0)  # Green in HSV
        rgb_color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hsv_color[0] / 360, hsv_color[1], hsv_color[2]))
        x = [25,125,225,325,425][col]
        y = [150,300][row]
        rect_position = (x, y, x+50, y+50)  # (x1, y1, x2, y2)
        draw.rectangle(rect_position, fill=rgb_color)
        text_bbox = font.getbbox(first_12[k])
        deltax = (STENCIL_SIZE - (text_bbox[2] - text_bbox[0]))//2
        draw.text((x+deltax,y+50), first_12[k], fill="black", font=font)
    return canvas

def make_image_12items(random_colors):
    canvas = Image.new('RGB', CANVAS_SIZE, 'white')
    draw = ImageDraw.Draw(canvas)
    for k in range(12):
        row = k // 4
        col = k % 4
        hsv_color = (random_colors[k], 1.0, 1.0)  # Green in HSV
        rgb_color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hsv_color[0] / 360, hsv_color[1], hsv_color[2]))
        x = [75,175,275,375][col]
        y = [50,200,350][row]
        rect_position = (x, y, x+50, y+50)  # (x1, y1, x2, y2)
        draw.rectangle(rect_position, fill=rgb_color)
        text_bbox = font.getbbox(first_12[k])
        deltax = (STENCIL_SIZE - (text_bbox[2] - text_bbox[0]))//2
        draw.text((x+deltax,y+50), first_12[k], fill="black", font=font)
    return canvas


def make_image_multi(num_stencils,random_colors_indexes):
    random_colors = [all_colors[k] for k in random_colors_indexes]
    if num_stencils == 4:
        return make_image_4items(random_colors)
    if num_stencils == 6:
        return make_image_6items(random_colors)
    if num_stencils == 8:
        return make_image_8items(random_colors)
    if num_stencils == 10:
        return make_image_10items(random_colors)
    if num_stencils == 12:
        return make_image_12items(random_colors)
    else:
        return None

def make_image_query(col_index):
    canvas = Image.new('RGB', CANVAS_SIZE, 'white')
    draw = ImageDraw.Draw(canvas)
    hsv_color = (all_colors[col_index], 1.0, 1.0)  # Green in HSV
    rgb_color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hsv_color[0] / 360, hsv_color[1], hsv_color[2]))
    rect_position = (225, 225, 275, 275)  # (x1, y1, x2, y2)
    draw.rectangle(rect_position, fill=rgb_color)
    return canvas