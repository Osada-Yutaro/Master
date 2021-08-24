import os
import numpy as np
from preprocess import load_images, load_targets, boundingbox_in_window, image_in_frame
import cv2
from metrics import TP, TN, FP, FN, IoU

def crop(image, bb, position, image_size, win_size):
    height, width = image_size
    x, y = position
    newbb = boundingbox_in_window(image_size, (win_size, win_size), position, bb)
    dst = image[y:y + win_size, x:x + win_size]
    X = image_in_frame((win_size, win_size), dst)
    Y = None
    if newbb is None:
        Y = [0, 0, 0, 0, 0]
    else:
        h_targ, w_targ, x_targ, y_targ = newbb
        Y = [h_targ, w_targ, x_targ, y_targ, 1]
    return X, Y

def key(bb):
    h, w, _, _, c = bb
    return h*w*c

def load_data(num):
    X = []
    Y = []

    image = load_images(num)
    targets = load_targets()[num]

    HEIGHT, WIDTH, _ = image.shape
    WIN_SIZE = 96

    x0 = np.random.randint(0, WIN_SIZE)
    y0 = np.random.randint(0, WIN_SIZE)
    for x in range(x0, WIDTH - WIN_SIZE, WIN_SIZE//3):
        for y in range(y0, HEIGHT - WIN_SIZE, WIN_SIZE//3):
            cropped_win = image[y:y + WIN_SIZE, x:x + WIN_SIZE]
            cropped_win = cv2.resize(cropped_win, (224, 224))
            targ = [0, 0, 0, 0, 0]
            for item in targets.items():
                id, bb = item
                _, newbb = crop(image, bb, (x, y), (HEIGHT, WIDTH), WIN_SIZE)
                targ = max(targ, newbb, key=key)
            X.append(cropped_win)
            Y.append(targ)
    return np.array(X), np.array(Y)

def draw(src, bb, color):
    print(bb)
    h, w, xc, yc, c = bb
    if c < 0.5:
        return src
    left_top = (xc - w/2, yc - h/2)
    right_bot = (xc + w/2, yc + h/2)
    dst = cv2.rectangle(src, left_top, right_bot, color, 4)
    return dst


def main():
    counter = 0
    path = os.path.join('/', 'kw_resources', 'Test')
    for i in range(221):
        X, Y = load_data(i)
        N = len(X)
        for j in range(N):
            x = X[j]
            y = Y[j]
            if y[4] < 0.5:
                continue
            predict = [0.4, 0.4, 0.5, 0.5, 1.0]
            img = draw(x, y, (0, 255, 0))
            img = draw(img, predict, (0, 255, 0))

            tp = TP(y, predict)
            tn = TN(y, predict)
            fp = FP(y, predict)
            fn = FN(y, predict)
            iou = IoU(y, predict)

            logpath = os.path.join(path, 'Log', str(counter) + '.txt')
            imgpath = os.path.join(path, 'Image', str(counter) + '.png')
            cv2.imwrite(imgpath, img)

            with open(logpath, mode='w') as f:
                message = 'TP: ' + str(tp) + '\n' + 'TN: ' + str(tn) + '\n' + 'FP: ' + str(fp) + '\n' + 'FN: ' + str(fn) + '\n' + 'IoU: ' + str(iou) + '\n'
                f.write(message)
            counter += 1

if __name__ == '__main__':
    main()