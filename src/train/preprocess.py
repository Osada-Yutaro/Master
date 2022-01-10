
def intersect(box1, box2):
    h1, w1, x1, y1 = box1
    h2, w2, x2, y2 = box2
    return max(x1, x2) <= min(x1 + w1, x2 + w2) and max(y1, y2) <= min(y1 + h1, y2 + h2)

"""
    win_size = (window_height, window_width)
    win_pos = (window_y, window_x)
    のウィンドウ内に
    bb = (height, width, x, y)
    のバウンディングボックスがあるときに
    ウィンドウ内のバウンディングボックスの大きさと位置を返す
"""
def boundingbox_in_window(image_size, win_size, win_pos, bb):
    import numpy as np

    height_w, width_w = win_size
    x_w, y_w = win_pos
    height_b, width_b, x_b, y_b = tuple(bb)

    if not intersect((height_w, width_w, x_w, y_w), bb):
        return None

    left_new, right_new = np.clip((x_b - x_w, x_b + width_b - x_w), 0, width_w)/width_w
    top_new, bot_new = np.clip((y_b - y_w, y_b + height_b - y_w), 0, height_w)/height_w

    width_new = right_new - left_new
    height_new = bot_new - top_new
    return (height_new, width_new, left_new, top_new)

def point_in_window(win_size, win_pos, center):
    import numpy as np

    window_height, window_width = win_size
    window_x, window_y = win_pos
    xc, yc = center
    
    new_xc = (xc - window_x)/window_width
    new_yc = (yc - window_y)/window_height

    if 0 <= new_xc <= 1 and 0 <= new_yc <= 1:
        return (new_xc, new_yc)
    else:
        return None

"""
data[フレーム番号][物体id] = (h, w, x, y) の辞書を返す
"""
def load_targets():
    import os
    import json

    FPS = 30

    targets_dir = os.path.join('/kw_resources', 'Master', 'Targets')
    ls = os.listdir(targets_dir)
    json_file_list = [f for f in ls if '.json' in f]

    data = {}

    for json_file in json_file_list:
        path = os.path.join(targets_dir, json_file)
        json_open = open(path, 'r')
        json_load = json.load(json_open)
        for region in json_load['regions']:
            left = region['boundingbox']['left']
            top = region['boundingbox']['top']
            width = region['boundingbox']['width']
            height = region['boundingbox']['height']
            xc = left + width/2
            yc = top + height/2

            asset_name = json_load['asset']['name']
            asset_name.split('#')
            mov = asset_name[0]
            time = float(asset_name[1])

            frame = int(FPS*time)
            id = int(json_load['tags'][0])
            data[frame][id] = (xc, yc)
    return data

def load_images(num):
    import os
    import cv2
    import numpy as np
    N = 5820
    view = os.path.join('/kw_resources', 'Master', 'Frames')

    filename = 'frame_' + str(num).zfill(4) + '.jpg'
    path = os.path.join(view, filename)
    res = cv2.imread(path)/255

    return res

def image_in_frame(frame_size, image):
    import numpy as np
    import cv2
    frame_height, frame_width = frame_size
    height, width, _ = image.shape
    frame = np.ones([frame_height, frame_width, 3])*0.5
    scale = 1/max(height/frame_height, width/frame_width)
    new_height = int(scale*height)
    new_width = int(scale*width)
    resized_image = cv2.resize(image, (new_width, new_height))
    top = frame_height//2 - new_height//2
    bottom = top + new_height
    left = frame_width//2 - new_width//2
    right = left + new_width
    frame[top:bottom, left:right] = resized_image
    return frame



