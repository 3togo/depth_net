import numpy as np
from scipy import interpolate

def interp_2d(depth_img, dtype=np.float32):
    depth_img_copy = depth_img.copy()
    x = np.arange(0, depth_img_copy.shape[1])
    y = np.arange(0, depth_img_copy.shape[0])
    
    array = np.ma.masked_invalid(depth_img_copy)
    xx, yy = np.meshgrid(x, y)
    
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    depth_img_copy = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear')
    return depth_img_copy.astype(dtype)

def fill_edges(depth_img, extrap_len):
    depth_img_copy = depth_img.copy()
    def mx_plus_b(x, m, b):
        return m*x + b

    def extrap(seg):
        mask = np.isnan(seg)
        line = np.polyfit(np.arange(len(seg))[~mask], seg[~mask], 1)
        seg[mask] = [mx_plus_b(x, line[0], line[1]) for x in np.arange(len(seg))[mask]]
        return seg

    for i in range(depth_img_copy.shape[0]):
        seg = depth_img_copy[i][:extrap_len]
        if np.sum(np.isnan(seg)) >= extrap_len-10:
            pass
        else:
            seg = extrap(seg)
        depth_img_copy[i][:extrap_len] = seg

    for i in range(depth_img_copy.shape[0]):
        seg = depth_img_copy[i][-extrap_len:]
        if np.sum(np.isnan(seg)) >= extrap_len-10:
            pass
        else:
            seg = extrap(seg)
        depth_img_copy[i][-extrap_len:] = seg
    return depth_img_copy

def combine_layers(depth_img, depth_layers):
    depth_img_copy = depth_img.copy()
    mask = np.isnan(depth_img_copy)
    for layer in depth_layers:
        layer_mask = np.isnan(layer)
        layer_vals = layer[~layer_mask*mask]
        depth_img_copy[~layer_mask*mask] = layer_vals
    return depth_img_copy

def fill_with_mean(depth_img):
    depth_img_copy = depth_img.copy()
    mask = np.isnan(depth_img_copy)
    mean = np.mean(depth_img_copy[~mask])
    depth_img_copy[mask] = mean
    return depth_img_copy

def smooth_values(depth_img, mask, dim):
    depth_img_copy = depth_img.copy()
    
    smooth = depth_img.copy()
    for i in range(depth_img_copy.shape[0]):
        for j in range(depth_img_copy.shape[1]):
            ymin = max(0, i-dim)
            ymax = min(depth_img_copy.shape[0], i+dim+1)
            xmin = max(0, j-dim)
            xmax = min(depth_img_copy.shape[1], j+dim+1)
            smooth[i][j] = np.mean(smooth[ymin:ymax, xmin:xmax])
    depth_img_copy[mask] = smooth[mask]
    return depth_img_copy

def fill_in_nan(depth_img, interp_len=100, smooth_kernel_dim=10):    
    depth_img_copy = depth_img.copy()
    nan_mask = np.isnan(depth_img_copy)
    
    depth_img_copy = interp_2d(depth_img_copy)
    for i in range(2):
        depth_img_h = fill_edges(depth_img_copy, interp_len)
        depth_img_v = fill_edges(depth_img_copy.transpose(1,0), interp_len).transpose(1,0) 
        depth_img_copy = combine_layers(depth_img_copy, [depth_img_h, depth_img_v])
    depth_img_copy = smooth_values(depth_img_copy, nan_mask, smooth_kernel_dim)
    depth_img_copy = fill_with_mean(depth_img_copy)
    assert(np.sum(np.isnan(depth_img_copy)) == 0)
    return depth_img_copy