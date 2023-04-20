import cv2

def video_cut(input_video_pth, out_video_pth):
    video_caputre = cv2.VideoCapture(input_video_pth)
    
    fps = video_caputre.get(cv2.CAP_PROP_FPS)
    width = video_caputre.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_caputre.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
    print("fps:", fps)
    print("width:", width)
    print("height:", height)
 
    # size = (1920-620, 1080)
    size = (640, 480)
    videp_write = cv2.VideoWriter(out_video_pth, \
                                  cv2.VideoWriter_fourcc(*'H264'), \
                                  fps, \
                                  size)
    success, frame_src = video_caputre.read()
    while success: 
        # [width, height]
        frame_target = frame_src[60:1020:, 200:1480, :]
        # import ipdb; ipdb.set_trace()
        frame_target = cv2.resize(frame_target, size)
        videp_write.write(frame_target)
        success, frame_src = video_caputre.read()
    print("finish cutting") 
    video_caputre.release()
 
if __name__=="__main__":
    input_video_pth = "toy_bear.mp4"
    out_video_pth  = "toy_bear_cut.mp4"
    video_cut(input_video_pth, out_video_pth)