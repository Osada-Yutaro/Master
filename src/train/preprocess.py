
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
    y_w, x_w = win_pos
    height_b, width_b, x_b, y_b = tuple(map(int, bb))

    if not intersect((height_w, width_w, x_w, y_w), bb):
        return None

    left_new, right_new = np.clip((x_b - x_w, x_b + width_b - x_w), 0, width_w)/width_w
    top_new, bot_new = np.clip((y_b - y_w, y_b + height_b - y_w), 0, height_w)/height_w

    width_new = right_new - left_new
    height_new = bot_new - top_new
    return (height_new, width_new, left_new, top_new)

"""
data[フレーム番号][物体id] = (h, w, x, y) の辞書を返す
"""
def load_targets():
    import xml.etree.ElementTree as ET
    
    xmlfile = 'PETS2009-S1L1-1.xml'
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    data = {}
    for number in root:
        for objectlist in number:
            number = int(number.attrib['number'])
            data[number] = {}
            for obj in objectlist:
                obj_id = int(obj.attrib['id'])
                for box in obj:
                    h, w, x, y = map(float, box.attrib.values())

                    data[number][obj_id] = (h, w, x, y)
    return data

def load_images():
    import os
    import cv2
    import numpy as np
    PETS09 = os.environ['PETS09']
    S1 = 'S1'
    L1 = 'L1'
    Time_13_57 = 'Time_13-57'
    View_001 = 'View_001'
    view = os.path.join(PETS09, S1, L1, Time_13_57, View_001)
    N = 221

    res = []

    for i in range(N):
        filename = 'frame_' + str(i).zfill(4) + '.jpg'
        path = os.path.join(view, filename)
        img = cv2.imread(path)/255
        res.append(img)

    return np.array(res)

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



