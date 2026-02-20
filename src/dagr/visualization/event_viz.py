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
            
            # 极性处理 - 正确映射事件极性到颜色通道
            polarity = int(p[i])
            if polarity == 1:  # 正极性事件（变亮）- 蓝色 (BGR格式中的通道0)
                channel = 0
            else:  # 负极性事件（变暗，polarity == 0）- 红色 (BGR格式中的通道2)
                channel = 2
                
            if 0 <= channel < img.shape[2]:
                # Apply color to all four neighboring pixels with weights
                event_color = 255 * (1 - alpha)
                
                # Update pixels with precise weights
                img[y0, x0, :] = alpha * img_copy[y0, x0, :]
                img[y0, x0, channel] += event_color * w00
                
                img[y0, x1, :] = alpha * img_copy[y0, x1, :]
                img[y0, x1, channel] += event_color * w10
                
                img[y1, x0, :] = alpha * img_copy[y1, x0, :]
                img[y1, x0, channel] += event_color * w01
                
                img[y1, x1, :] = alpha * img_copy[y1, x1, :]
                img[y1, x1, channel] += event_color * w11
        
        # Fallback for edge cases - use nearest pixel
        elif 0 <= x_coord < img.shape[1] and 0 <= y_coord < img.shape[0]:
            x_nearest = int(round(x_coord))
            y_nearest = int(round(y_coord))
            
            # 极性处理
            polarity = int(p[i])
            if polarity == 1:  # 正极性事件（变亮）- 蓝色
                channel = 0
            else:  # 负极性事件（变暗）- 红色
                channel = 2
            
            if (0 <= x_nearest < img.shape[1] and 0 <= y_nearest < img.shape[0] and
                0 <= channel < img.shape[2]):
                img[y_nearest, x_nearest, :] = alpha * img_copy[y_nearest, x_nearest, :]
                img[y_nearest, x_nearest, channel] += 255 * (1 - alpha)
    
    # 确保像素值在有效范围内
    return np.clip(img, 0, 255).astype(np.uint8)