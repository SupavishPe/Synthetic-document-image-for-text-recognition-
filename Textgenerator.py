import config

import os
path = os.environ.get('PATH')
os.environ['PATH'] = config.VIPS_BIN_PATH + ';' + path 

from utils import readlines_txt, read_json

import json
import glob
import random
import pyvips
import numpy as np
import cv2
import re 
import albumentations as A
from tqdm import tqdm
from PIL import Image, ImageOps
from joblib import Parallel, delayed

def transform_img(image): 
    img = np.array(image)
    transform = A.Compose([
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.OpticalDistortion(p=0.5),
        ])
    transformed_img = transform(image=img)["image"]
    transformed_img = Image.fromarray(np.uint8(transformed_img))
    return transformed_img

def rotate_text(image, bg, limit):
    img = np.array(image)
    mask = np.ones((img.shape[0], img.shape[1])) * 255
    rotate_f = A.SafeRotate(limit=limit, border_mode=0, value=(0,0,0), p=1)
    results = rotate_f(image=img, mask=mask)
    img = results['image']
    mask = results['mask']
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    rotate_img = Image.fromarray(img.astype('uint8'))
    rotate_mask = Image.fromarray(mask.astype('uint8'))
    image = bg.copy()
    image.paste(rotate_img, (0,0), mask= rotate_mask)
    w, h =image.size
    area = (1, 1, w-1, h-1)
    image = image.crop(area)
    return image

class Textgenerator: 
    def __init__(
        self, rotate_text_f=None, transform=None, dpi=300, n_job=1, saved_dir=None, saved_format='jpg', padding=10,
        img_height=32
        ): 
        assert saved_format in ['jpg', 'png'] , ' file format need be "jpg" or "png" '
        self.rotate_text = rotate_text_f
        self.transform=transform
        self.dpi=dpi
        self.n_job=n_job
        self.saved_dir=saved_dir
        self.saved_format=saved_format
        self.padding=padding
        self.img_height=img_height
        self.img_idx=0
        
    def convert_vips_to_pil(self, img):
        format_to_dtype = {
            'uchar': np.uint8,
            'char': np.int8,
            'ushort': np.uint16,
            'short': np.int16,
            'uint': np.uint32,
            'int': np.int32,
            'float': np.float32,
            'double': np.float64,
            'complex': np.complex64,
            'dpcomplex': np.complex128,
        }
        img_np = np.ndarray(
            buffer=img.write_to_memory(),
            dtype=format_to_dtype[img.format],
            shape=[img.height, img.width, img.bands]
        )
        if img_np.shape[-1] == 1: 
            img_pil = img_np.reshape(img_np.shape[0], img_np.shape[1])
            img_pil = Image.fromarray(img_pil.astype('uint8'))
        else: 
            img_pil = Image.fromarray(img_np.astype('uint8'))
        return img_pil

    def font_rendering(self, text, font, background_img, text_color, background_color=(255,255,255)): 
        # image and mask 
        img = pyvips.Image.text(
            text,
            dpi=self.dpi,
            fontfile=font)              
        mask_img_pil = self.convert_vips_to_pil(img)

        #text color
        assert len(text_color) == 3 , 'color need to be (R,G,B)'
        tc = (255-text_color[0], 255-text_color[1], 255-text_color[2])
        img /= 255
        img *= tc
        text_img_pil = self.convert_vips_to_pil(img)
        text_img_pil = ImageOps.invert(text_img_pil)
        font_w, font_h = text_img_pil.size

        #blackground
        W = font_w+(2*self.padding)
        H = font_h+(2*self.padding)
        if background_img is None: 
            bg = Image.new('RGBA', (W, H), color=background_color)
        else: 
            bg = Image.open(background_img)
            bg_w, bg_h = bg.size
            if (bg_w < W) or (bg_h < H) : 
                bg = bg.resize( (W, H) )
                bg_w, bg_h = bg.size
            rand_x = random.randint(0, bg_w-(font_w+(2*self.padding)))
            rand_y = random.randint(0, bg_h-(font_h+(2*self.padding)))
            area = (rand_x, rand_y, rand_x+(font_w+(2*self.padding)), rand_y+(font_h+(2*self.padding)))
            bg = bg.crop(area)
        image = bg.copy()
        image.paste(text_img_pil, (self.padding, self.padding), mask=mask_img_pil)
        
        # Rotate text
        if self.rotate_text is not None :  
            image = self.rotate_text(image, bg)

        # Transform image
        if self.transform is not None : 
            image = self.transform(image)

        # resize 
        w, h = image.size
        resize_h = self.img_height
        resize_w = int(w * (resize_h / h))
        resize_t = A.Resize(height=resize_h, width=resize_w, p=1)
        resized_img = resize_t(image=np.array(image))["image"]
        image = Image.fromarray(np.uint8(resized_img))

        if self.saved_dir is not None:
            if not os.path.isdir(self.saved_dir): 
                os.mkdir(self.saved_dir)
                self.img_idx +=1
            elif self.img_idx==0:
                self.img_idx = len(os.listdir(self.saved_dir)) + 1
                
            # save image
            cln_text = re.sub(r'[\/:*?<>|]', '', text)
            filename_img = self.saved_dir + f'{cln_text}_{self.img_idx}.{self.saved_format}'
            image.save(filename_img)
            self.img_idx+=1

        image = np.array(image)
        return image

    def run(self, n_img, txt_color_file, font_dir, bg_dir,  txt_file=None, txt=None, font=None):
        if txt is not None: 
            texts = np.array([txt]*n_img)
        elif txt_file is not None : 
            texts = readlines_txt(txt_file)
        else: 
            raise Exception('please specify text or text file')

        if font is not None: 
            fonts = np.array([font]*n_img)
        else: 
            fonts = glob.glob( f"{font_dir}/*/*.ttf")

        text_colors = read_json(txt_color_file)['color']
        bgs = glob.glob(f"{bg_dir}*.jpg") + glob.glob(f"{bg_dir}*.png")

        font_n = np.random.choice(fonts, n_img)
        bg_n = np.random.choice(bgs, n_img)
        text_n = np.random.choice(texts, n_img)
        text_color_n = [text_colors[i] for i in np.random.choice(np.arange(len(text_colors)), n_img)]

        styles = list(zip(text_n, font_n, bg_n, text_color_n))

        gen_img = Parallel(n_jobs=self.n_job, backend='multiprocessing')(
            delayed(self.font_rendering)(text, font, bg, text_color) for (text, font, bg, text_color) in tqdm(
                styles, ascii=True, desc='Generating text image')
        )
        return gen_img