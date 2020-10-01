import cv2
import numpy as np
import sys


def splitRGB(img):
    B_map, G_map, R_map = cv2.split(img)
    return R_map, G_map, B_map

def splitHSV(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H_map, S_map, V_map = cv2.split(hsv)
    return H_map, S_map, V_map

def bilinear_interpolation(img, x, y, print_detail=False):
    x_l = int(np.floor(x))
    x_h = int(np.ceil(x))
    y_l = int(np.floor(y))
    y_h = int(np.ceil(y))
    a = img[x_l, y_l]
    b = img[x_l, y_h]
    c = img[x_h, y_l]
    d = img[x_h, y_h]
    wx = x - x_l
    wy = y - y_l
    p = a * (1-wx) * (1-wy) + b * wx * (1-wy) + c * (1-wx) * wy + d * wx * wy
    if print_detail:
        print('x_l(%d)<x(%.2f)<x_h(%d), wx = %.2f, y_l(%d)<y(%f)<y_h(%d), wy = %.2f' % (x_l, x, x_h, wx, y_l, y, y_h, wy))
        print(a, b, c, d, p)
    return p


def resize(img, size):
    height = img.shape[0]
    width = img.shape[1]
    new_h = int(np.around(height * size))
    new_w = int(np.around(width * size))
    new_image = np.zeros((new_h, new_w, 3), np.uint8)
    x_ratio = float(height - 1) / float(new_h -1)
    y_ratio = float(width -1) / float(new_w -1)
    print('[%f, %f]' % (x_ratio, y_ratio))
    for i in range(0, new_h):
        for j in range(0, new_w):
            x = float(i) * x_ratio
            y = float(j) * y_ratio
            print_detail = False
            if i % 100 == 0 and j % 100 == 0:
                print_detail = True
            new_image[i, j] = bilinear_interpolation(img, x, y, print_detail)
            if print_detail:
                print('[', i, ',', j, ']', new_image[i, j])
    return new_image


class MotionDetect(object):
    """docstring for MotionDetect"""
    def __init__(self, shape):
        super(MotionDetect, self).__init__()

        self.shape = shape
        self.avg_map = np.zeros((self.shape[0], self.shape[1], self.shape[2]), dtype='float')
        self.alpha = 0.8 # you can ajust your value
        self.threshold = 40 # you can ajust your value

        print("MotionDetect init with shape {}".format(self.shape))

    def getMotion(self, img):
        assert img.shape == self.shape, "Input image shape must be {}, but get {}".format(self.shape, img.shape)

        # Extract motion part (hint: motion part mask = difference between image and avg > threshold)
        motion = img - self.avg_map

        # Mask out unmotion part (hint: set the unmotion part to 0 with mask)
        motion_map = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if np.sum(np.absolute(motion[i, j])) < self.threshold:
                    motion_map[i, j] = np.array([0, 0, 0])
                else:
                    motion_map[i, j] = img[i, j]
        # Update avg_map
        self.avg_map = self.avg_map * self.alpha + img * (1 -self.alpha)
        return motion_map


def split():
    # ------------------ #
    #     RGB & HSV      #
    # ------------------ #
    name = "../data.png"
    img = cv2.imread(name)
    if img is not None:
        print("Reading {} success. Image shape {}".format(name, img.shape))
    else:
        print("Faild to read {}.".format(name))

    R_map, G_map, B_map = splitRGB(img)
    H_map, S_map, V_map = splitHSV(img)

    cv2.imwrite('data_R.png', R_map)
    cv2.imwrite('data_G.png', G_map)
    cv2.imwrite('data_B.png', B_map)
    cv2.imwrite('data_H.png', H_map)
    cv2.imwrite('data_S.png', S_map)
    cv2.imwrite('data_V.png', V_map)


def interpolation():
    # ------------------ #
    #   Interpolation    #
    # ------------------ #
    name = "../data.png"
    img = cv2.imread(name)
    if img is not None:
        print("Reading {} success. Image shape {}".format(name, img.shape))
    else:
        print("Faild to read {}.".format(name))

    height, width, channel = img.shape
    img_big = resize(img, 2)
    img_small = resize(img, 0.5)
    img_big_cv = cv2.resize(img, (width*2, height*2))
    img_small_cv = cv2.resize(img, (width//2, height//2))

    cv2.imwrite('data_2x.png', img_big)
    cv2.imwrite('data_0.5x.png', img_small)
    cv2.imwrite('data_2x_cv.png', img_big_cv)
    cv2.imwrite('data_0.5x_cv.png', img_small_cv)


def videoRW():
    # ------------------ #
    #  Video Read/Write  #
    # ------------------ #
    name = "../data.mp4"
    # Input reader
    cap = cv2.VideoCapture(name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output1.avi', fourcc, fps, (w, h), True)

    # Motion detector
    mt = MotionDetect(shape=(h,w,3))

    # Read video frame by frame
    while True:
        # Get 1 frame
        success, frame = cap.read()

        if success:
            motion_map = mt.getMotion(frame)

            # Write 1 frame to output video
            out.write(motion_map)
        else:
            break

    # Release resource
    cap.release()
    out.release()


def main():
    split()
    interpolation()
    videoRW()


if __name__ == '__main__':
    main()