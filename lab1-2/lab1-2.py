import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from simple_term_menu import TerminalMenu


def gammaCorrection(img, gamma=1.0):
    print(img.shape)
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    map = {}
    for i in range(0, 256):
        norm_val = float(i)/255.0
        map[i] = int(255.0 * math.pow(norm_val, gamma))
    # print(map)

    # Apply gamma correction using the lookup table
    img_g = np.zeros(img.shape, np.uint8)
    for j in range(0, img.shape[0]):
        for k in range(0, img.shape[1]):
            img_g[j, k] = map[img[j, k]]
    return img_g


def histEq(gray):
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    hist = hist / gray.size
    print(hist)

    # Convert the histogram to Cumulative Distribution Function
    cum_sum = np.zeros(hist.shape, float)
    cum_sum[0] = hist[0]
    for i in range(1, hist.shape[0]):
        cum_sum[i] = cum_sum[i -1] + hist[i]
    # print(cum_sum)
    # Build a lookup table mapping the pixel values [0, 255] to their new grayscale value
    map = {}
    for j in range(0, 256):
        map[j] = int(cum_sum[j] * 255.0)
    # Apply histogram equalization using the lookup table
    img_h = np.zeros(gray.shape, np.uint8)
    for k in range(0, gray.shape[0]):
        for l in range(0, gray.shape[1]):
            img_h[k, l] = map[gray[k, l]]
    return img_h


def gamma_correction():
    # ------------------ #
    #  Gamma Correction  #
    # ------------------ #
    name = "../data.mp4"
    cap = cv2.VideoCapture(name)
    success, frame = cap.read()
    if success:
        print("Success reading 1 frame from {}".format(name))
    else:
        print("Faild to read 1 frame from {}".format(name))
    cap.release()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray.png', gray)
    img_g1 = gammaCorrection(gray, 0.5)
    img_g2 = gammaCorrection(gray, 2)
    
    cv2.imwrite('data_g0.5.png', img_g1)
    cv2.imwrite('data_g2.png', img_g2)


def histogram_equalization():
    # ------------------------ #
    #  Histogram Equalization  #
    # ------------------------ #
    name = "../hist.png"
    img = cv2.imread(name, 0)

    img_h = histEq(img)
    img_h_cv = cv2.equalizeHist(img)
    cv2.imwrite("hist_h.png", img_h)
    cv2.imwrite("hist_h_cv.png", img_h_cv)

    # save histogram
    plt.figure(figsize=(18, 6))
    plt.subplot(1,3,1)
    plt.bar(range(1,257), cv2.calcHist([img], [0], None, [256], [0, 256]).reshape(-1))
    plt.subplot(1,3,2)
    plt.bar(range(1,257), cv2.calcHist([img_h], [0], None, [256], [0, 256]).reshape(-1))
    plt.subplot(1,3,3)
    plt.bar(range(1,257), cv2.calcHist([img_h_cv], [0], None, [256], [0, 256]).reshape(-1))
    plt.savefig('hist_plot.png')


def main():
    terminal_menu = TerminalMenu(["Gamma Correction", "Histogram Equalization"])
    menu_entry_index = terminal_menu.show()
    if menu_entry_index == 0:
        gamma_correction()
    elif menu_entry_index == 1:
        histogram_equalization()


if __name__ == '__main__':
    main()