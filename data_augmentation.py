import cv2
import numpy as np
import random

def scale_image(img, scale):
    copy = img.copy()
    old_w, old_h = (copy.shape[1], copy.shape[0])
    new_w, new_h = (int(scale*old_w), int(scale*old_h))
    crop = copy[int(old_h/2-new_h/2):int(old_h/2+new_h/2), int(old_w/2-new_w/2):int(old_w/2+new_w/2)]
    img_scaled = cv2.resize(crop, (old_w, old_h), interpolation=cv2.INTER_AREA)
    return img_scaled

def rotate_image(img, angle):
    copy = img.copy()
    center = tuple(np.array(copy.shape)/2)
    rot_mat = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1.0)
    return cv2.warpAffine(copy, rot_mat, (copy.shape[1], copy.shape[0]), flags=cv2.INTER_AREA)

def hsv_shift(img, f):
    copy = img.copy()
    hsv = cv2.cvtColor(copy, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    h, s, v = (np.clip(h*f, 0, 180).astype(np.uint8), 
               np.clip(s*f, 0, 255).astype(np.uint8), 
               np.clip(v*f, 0, 255).astype(np.uint8))
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)

def contrast_image(img, c):
    copy = img.copy()
    return np.where(255/(copy+1e-6) < c, 255, copy*c).astype(np.uint8)

def brighten_image(img, b):
    copy = img.copy()
    if b > 0:
        return np.where((255 - copy) < b, 255, copy+b).astype(np.uint8)
    elif b < 0:
        return np.where(copy < abs(b), 0, copy+b).astype(np.uint8)
    else:
        return copy

def random_scale(rgb_img, depth_img, max_scale=0.5):
    scale = random.uniform(max_scale, 1.0)
    rgb_scaled = scale_image(rgb_img, scale)
    depth_scaled = scale_image(depth_img, scale)
    depth_scaled = depth_scaled*scale
    return rgb_scaled, depth_scaled, scale

def random_rotation(rgb_img, depth_img, max_angle=5):
    angle = random.uniform(-max_angle, max_angle)
    rgb_rotated = rotate_image(rgb_img, angle)
    depth_rotated = rotate_image(depth_img, angle)
    return rgb_rotated, depth_rotated, angle

def random_flip(rgb_img, depth_img):
    rgb_flip, depth_flip = (rgb_img.copy(), depth_img.copy())
    idx = random.randint(0,1)
    if idx == 1:
        rgb_flip = cv2.flip(rgb_flip, 1)
        depth_flip = cv2.flip(depth_flip, 1)
    return rgb_flip, depth_flip

def random_HSV_shift(rgb_img):
    f = random.uniform(0.8, 1.2)
    return hsv_shift(rgb_img, f)

def random_contrast(rgb_img):
    c = random.uniform(0.8, 1.2)
    return contrast_image(rgb_img, c)

def random_brightness(rgb_img):
    b = random.uniform(-50, 50)
    return brighten_image(rgb_img, b)

def random_augmentation(rgb_img, depth_img):
    rgb_img_augmented = random_HSV_shift(random_contrast(random_brightness(rgb_img)))
    rgb_img_augmented, depth_img_augmented, _ = random_scale(rgb_img_augmented, depth_img)
    rgb_img_augmented, depth_img_augmented = random_flip(rgb_img_augmented, depth_img_augmented)
    rgb_img_augmented, depth_img_augmented, _ = random_rotation(rgb_img_augmented, depth_img_augmented) 
    return rgb_img_augmented, depth_img_augmented 