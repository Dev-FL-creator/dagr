import numpy as np

def draw_events_on_image(img, x, y, p, alpha=0.5):
    """
    Draw events on image with precise floating point coordinates
    Uses bilinear interpolation for sub-pixel accuracy
    """
    img = img.astype(np.float32)
    img_copy = img.copy()
    
    for i in range(len(p)):
        x_coord = float(x[i])
        y_coord = float(y[i])
        
        # Get integer parts and fractional parts
        x0 = int(np.floor(x_coord))
        y0 = int(np.floor(y_coord))
        x1 = x0 + 1
        y1 = y0 + 1
        
        dx = x_coord - x0
        dy = y_coord - y0
        
        # Check bounds for all four pixels
        if (0 <= x0 < img.shape[1] and 0 <= y0 < img.shape[0] and
            0 <= x1 < img.shape[1] and 0 <= y1 < img.shape[0]):
            
            # Bilinear interpolation weights
            w00 = (1 - dx) * (1 - dy)
            w01 = (1 - dx) * dy
            w10 = dx * (1 - dy)
            w11 = dx * dy
            
            # Get polarity channel
            p_idx = int(p[i]) - 1
            if 0 <= p_idx < img.shape[2]:
                # Apply color to all four neighboring pixels with weights
                event_color = 255 * (1 - alpha)
                
                # Update pixels with precise weights
                img[y0, x0, :] = alpha * img_copy[y0, x0, :]
                img[y0, x0, p_idx] += event_color * w00
                
                img[y0, x1, :] = alpha * img_copy[y0, x1, :]
                img[y0, x1, p_idx] += event_color * w10
                
                img[y1, x0, :] = alpha * img_copy[y1, x0, :]
                img[y1, x0, p_idx] += event_color * w01
                
                img[y1, x1, :] = alpha * img_copy[y1, x1, :]
                img[y1, x1, p_idx] += event_color * w11
        
        # Fallback for edge cases - use nearest pixel
        elif 0 <= x_coord < img.shape[1] and 0 <= y_coord < img.shape[0]:
            x_nearest = int(round(x_coord))
            y_nearest = int(round(y_coord))
            p_idx = int(p[i]) - 1
            
            if (0 <= x_nearest < img.shape[1] and 0 <= y_nearest < img.shape[0] and
                0 <= p_idx < img.shape[2]):
                img[y_nearest, x_nearest, :] = alpha * img_copy[y_nearest, x_nearest, :]
                img[y_nearest, x_nearest, p_idx] += 255 * (1 - alpha)
    
    return img.astype(np.uint8)