import cv2
import numpy as np
from simple_term_menu import TerminalMenu

def avgFilter(img):
    img_avg = np.zeros(img.shape, np.uint8)
    for i in range(0, img.shape[0]):
        x_l = 0 if (i - 1) < 0 else i -1
        x_h = img.shape[0] if (i + 1) >= img.shape[0] else i + 2
        for j in range(0, img.shape[1]): 
            y_l = 0 if (j - 1) < 0 else j - 1
            y_h = img.shape[1] if (j + 1) >= img.shape[1] else j + 2
            kernel_sum = 0
            num_kernel = 0
            for k in range(x_l, x_h):
                for l in range(y_l, y_h):
                    kernel_sum += img[k, l]
                    num_kernel += 1
            img_avg[i, j] = kernel_sum // num_kernel
    return img_avg


def midFilter(img):
    img_mid = np.zeros(img.shape, np.uint8)
    for i in range(0, img.shape[0]):
        x_l = 0 if (i - 1) < 0 else i -1
        x_h = img.shape[0] if (i + 1) >= img.shape[0] else i + 2
        for j in range(0, img.shape[1]): 
            y_l = 0 if (j - 1) < 0 else j - 1
            y_h = img.shape[1] if (j + 1) >= img.shape[1] else j + 2
            kernel_list = []
            for k in range(x_l, x_h):
                for l in range(y_l, y_h):
                    kernel_list.append(img[k, l])
            img_mid[i, j] = np.median(kernel_list) 
    return img_mid


def edgeSharpen(img):
    avg_kernel = np.ones([3, 3], np.float64) / 9
    denoise_img = cv2.filter2D(img, ddepth=-1, dst=-1, kernel=avg_kernel)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    img_edge = cv2.filter2D(denoise_img, ddepth=-1, dst=-1, kernel=kernel)
    img_s = np.subtract(denoise_img, img_edge, dtype=np.int16)
    img_s[img_s < 0] = 0
    img_s = img_s.astype('uint8')
    return img_edge, img_s
    # return img_edge


def denoise():
    # ------------------ #
    #       Denoise      #
    # ------------------ #
    name1 = '../noise_impulse.png'
    name2 = '../noise_gauss.png'
    noise_imp = cv2.imread(name1, 0)
    noise_gau = cv2.imread(name2, 0)

    img_imp_avg = avgFilter(noise_imp)
    img_imp_mid = midFilter(noise_imp)
    img_gau_avg = avgFilter(noise_gau)
    img_gau_mid = midFilter(noise_gau)

    cv2.imwrite('img_imp_avg.png', img_imp_avg)
    cv2.imwrite('img_imp_mid.png', img_imp_mid)
    cv2.imwrite('img_gau_avg.png', img_gau_avg)
    cv2.imwrite('img_gau_mid.png', img_gau_mid)


def sharpen():
    # ------------------ #
    #       Sharpen      #
    # ------------------ #
    name = '../mj.tif'
    img = cv2.imread(name, 0)

    img_edge, img_s = edgeSharpen(img)
    cv2.imwrite('mj_edge.png', img_edge)
    cv2.imwrite('mj_sharpen.png', img_s)


def main():
    terminal_menu = TerminalMenu(["Denoise", "Sharpen"])
    menu_entry_index = terminal_menu.show()
    if menu_entry_index == 0:
        denoise()
    elif menu_entry_index == 1:
        sharpen()


if __name__ == '__main__':
    main()