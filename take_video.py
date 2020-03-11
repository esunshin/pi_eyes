from time import sleep

def record_video(usb, seconds, filename = None):
    filename = ("video" if filename is None else filename) + ".mp4"

    if usb:
        import os
        os.system("ffmpeg -f v4l2 -framerate 24 -t " + str(seconds) + \
                    " -video_size 1080x720 -i /dev/video0 " + filename)

    else:
        from picamera import PiCamera as camera
        # with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.framerate = 24
        camera.start_recording(filename)
        camera.wait_recording(seconds)
        camera.stop_recording()


