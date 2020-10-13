import cv2
import numpy as np
import sys


class MotionDetect(object):
    """docstring for MotionDetect"""
    def __init__(self, shape):
        super(MotionDetect, self).__init__()

        self.shape = shape
        self.avg_map = np.zeros((self.shape[0], self.shape[1], self.shape[2]), dtype='float')
        self.alpha = 0.7 # you can ajust your value
        self.threshold = 35 # you can ajust your value

        print("MotionDetect init with shape {}".format(self.shape))

    def getMotion(self, img):
        assert img.shape == self.shape, "Input image shape must be {}, but get {}".format(self.shape, img.shape)
        avg_kernel = np.ones([3, 3], np.float64) / 9
        img = cv2.filter2D(img, ddepth=-1, dst=-1, kernel=avg_kernel)
        # Extract motion part (hint: motion part mask = difference between image and avg > threshold)
        motion = img - self.avg_map

        # Mask out unmotion part (hint: set the unmotion part to 0 with mask)
        motion_map = np.where(motion < self.threshold, 0, img)
        # motion_map = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)
        # for i in range(0, img.shape[0]):
        #    for j in range(0, img.shape[1]):
        #        if np.sum(np.absolute(motion[i, j])) < self.threshold:
        #            motion_map[i, j] = np.array([0, 0, 0])
        #        else:
        #            motion_map[i, j] = img[i, j]
        # Update avg_map
        self.avg_map = self.avg_map * self.alpha + img * (1 -self.alpha)
        return motion_map


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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    while True:
        # Get 1 frame
        success, frame = cap.read()

        if success:
            motion_map = mt.getMotion(frame)
            count += 1
            percent = int(count * 100)/total_frames
            if count % 5 == 0:
                print('Processed %d %%' % percent)
            # Write 1 frame to output video
            out.write(motion_map)
        else:
            break

    # Release resource
    cap.release()
    out.release()

def main():
    videoRW()


if __name__ == '__main__':
    main()

