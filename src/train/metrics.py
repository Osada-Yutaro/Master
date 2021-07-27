from tensorflow.keras import backend as K

def k_max(a, b):
    cond = K.cast(K.greater(a, b), K.floatx())
    return cond*a + (1 - cond)*b
def k_min(a, b):
    cond = K.cast(K.less(a, b), K.floatx())
    return cond*a + (1 - cond)*b

def clip(x, bottom=0., top=1.):
    return k_min(k_max(x, bottom), top)
def binalize(x, boarder):
    return K.cast(K.greater(x, boarder), K.cast(K.floatx()))

def parse_BB(boundingbox):
    return (
        clip(boundingbox[:, 0]),
        clip(boundingbox[:, 1]),
        clip(boundingbox[:, 2]),
        clip(boundingbox[:, 3]),
        binalize(boundingbox[:, 4], 0.5))


def TP(groundtruth, predict):
    h_ground, w_ground, x_ground, y_ground, c_ground = parse_BB(groundtruth)
    h_predic, w_predic, x_predic, y_predic, c_predic = parse_BB(predict)

    intersection_w = k_min(x_ground + w_ground, x_predic + w_predic) - k_max(x_ground, x_predic)
    intersection_h = k_min(y_ground + h_ground, y_predic + h_predic) - k_max(y_ground, y_predic)

    is_intersecting = K.cast(K.greater(intersection_w, 0), K.floatx())*K.cast(K.greater(intersection_h, 0), K.floatx())*c_ground*c_predic
    intersection = is_intersecting*intersection_w*intersection_h
    return intersection

def TN(groundtruth, predict):
    h_ground, w_ground, _, _, c_ground = parse_BB(groundtruth)

    truth = h_ground*w_ground*c_ground
    return truth - TP(groundtruth, predict)

def FP(groundtruth, predict):
    h_predic, w_predic, _, _, c_predic = parse_BB(predict)

    positive = h_predic*w_predic*c_predic
    return positive - TP(groundtruth, predict)

def FN(groundtruth, predict):
    return 1 - FP(groundtruth, predict)
