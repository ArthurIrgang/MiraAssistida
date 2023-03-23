import cv2
from lib import calibration,video

live = video.openLive(720,30)



while True:
    ret, undistort = video.undistortVideo(live,'dispersion_videos\calibration_parameters_14_3_2023.yaml')
    ret, liveV = live.read()
    cv2.imshow("Live Video",liveV)
    cv2.imshow("Undistorted Live Video", undistort)

    if cv2.waitKey(17) & 0xFF == ord('q'):
        break


cv2.imwrite("undistorted.jpg", undistort)
cv2.imwrite("distorted.jpg", liveV)