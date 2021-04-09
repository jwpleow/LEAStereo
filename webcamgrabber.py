import cv2
import numpy as np
import os
import threading

class Arducam():
    def __init__(self, camera_string, calib_params="CalibParams_Stereo.yml"):
        self._import_params(calib_params)
        self._connect_to_camera(camera_string)
        self._calculate_matched_roi()

        self.thread_running = True
        self.frame_lock = threading.Lock()
        self.last_frame = np.zeros((self.image_size_[1], self.image_size_[0] * 2))

        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()

        
    def update(self):
        while self.thread_running:
            ret, frame = self._cam.read()
            if ret:
                self.frame_lock.acquire()
                self.last_frame = frame
                self.frame_lock.release()
    
    def read(self):
        self.frame_lock.acquire()
        image = self.last_frame.copy()
        self.frame_lock.release()

        # split
        left = image[0:self.image_size_[1], 0:self.image_size_[0]]
        right = image[0:self.image_size_[1],
                    self.image_size_[0]:2*self.image_size_[0]]
        Uleft = cv2.remap(left, self.left_map_1_,
                        self.left_map_2_, cv2.INTER_LINEAR)
        Uright = cv2.remap(right, self.right_map_1_,
                        self.right_map_2_, cv2.INTER_LINEAR)
        Uleft = Uleft[self.matchedRoi1_[1]: self.matchedRoi1_[
            1] + self.matchedRoi1_[3], self.matchedRoi1_[0]:self.matchedRoi1_[0]+self.matchedRoi1_[2]]
        Uright = Uright[self.matchedRoi2_[1]: self.matchedRoi2_[
            1] + self.matchedRoi2_[3], self.matchedRoi2_[0]:self.matchedRoi2_[0]+self.matchedRoi2_[2]]
        return Uleft, Uright

    def release(self):
        self.thread_running = False
        self.thread.join()

    def _connect_to_camera(self, camera_string):
        # connect to cam
        print("Attempting to open VideoCapture...")
        self._cam = cv2.VideoCapture(camera_string, cv2.CAP_GSTREAMER)
        if not (self._cam.isOpened()):
            raise Exception("Webcam could not be opened!")
        # self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_size_[0] * 2)
        # self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_size_[1])
        print("VideoCapture opened!")
        os.system("v4l2-ctl --set-ctrl=gain=5")



    def _import_params(self, calib_params):
        fs = cv2.FileStorage(calib_params,
                             cv2.FILE_STORAGE_READ)
        if (fs.isOpened()):
            self.left_camera_matrix_ = fs.getNode("left_camera_matrix").mat()
            self.left_distortion_coefficients_ = fs.getNode(
                "left_distortion_coefficients").mat()

            self.right_camera_matrix_ = fs.getNode("right_camera_matrix").mat()
            self.right_distortion_coefficients_ = fs.getNode(
                "right_distortion_coefficients").mat()
            self.R_ = fs.getNode("R").mat()
            self.T_ = fs.getNode("T").mat()
            self.E_ = fs.getNode("E").mat()
            self.F_ = fs.getNode("F").mat()
            self.image_size_ = (int(fs.getNode("image_width").real()), int(
                fs.getNode("image_height").real()))
            fs.release()
        else:
            raise Exception("calibration file could not be opened")

        self.R1_, self.R2_, self.P1_, self.P2_, self.Q_, self.validRoi1_, self.validRoi2_ = cv2.stereoRectify(
            self.left_camera_matrix_, self.left_distortion_coefficients_, self.right_camera_matrix_, self.right_distortion_coefficients_, self.image_size_, self.R_, self.T_)
        self.left_map_1_, self.left_map_2_ = cv2.initUndistortRectifyMap(
            self.left_camera_matrix_, self.left_distortion_coefficients_, self.R1_, self.P1_, self.image_size_, cv2.CV_16SC2)
        self.right_map_1_, self.right_map_2_ = cv2.initUndistortRectifyMap(
            self.right_camera_matrix_, self.right_distortion_coefficients_, self.R2_, self.P2_, self.image_size_, cv2.CV_16SC2)

    def _calculate_matched_roi(self):
        new_y_loc = max(self.validRoi1_[1], self.validRoi2_[1])
        new_x_loc_right = max(self.validRoi2_[0], self.image_size_[
                              0] - self.validRoi1_[0] - self.validRoi1_[2])
        new_height = min(self.validRoi1_[3] - (new_y_loc - self.validRoi1_[
                         1]), self.validRoi2_[3] - (new_y_loc - self.validRoi2_[1]))
        new_width = min(min((self.validRoi1_[0] + self.validRoi1_[2]) - new_x_loc_right, self.validRoi2_[
                        2]), self.image_size_[0] - self.validRoi2_[0] - new_x_loc_right)

        self.matchedRoi1_ = (self.image_size_[
                             0] - new_x_loc_right - new_width, new_y_loc, new_width, new_height)
        self.matchedRoi2_ = (new_x_loc_right, new_y_loc, new_width, new_height)

    def _disconnect_from_camera(self):
        self._cam.release()

if __name__ == "__main__":
    cam = Arducam()
    while True:
        left, right = cam.read()
        cv2.imshow("left", left)
        cv2.imshow("right", right)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.release()
            break
