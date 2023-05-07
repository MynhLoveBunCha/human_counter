from ultil import *
import os
import pandas as pd
import datetime
import time


def main():
    # video_path = os.path.join('..', 'videos', 'crowd.mp4')
    video_path = 0
    config_path = os.path.join('..', 'models', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    model_path = os.path.join('..', 'models', 'frozen_inference_graph.pb')
    classes_path = os.path.join('..', 'models', 'coco.names')
    detector = Detector(video_path, config_path, model_path, classes_path)

    # Camera capture
    if type(detector.video_path) == int:
        cap = cv2.VideoCapture(detector.video_path, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(detector.video_path)

    # polygon drawer object
    poly_draw = None
    poly_pts = []
    have_poly = False
    pts_draw = []

    # fps calculation
    prev_time = time.time()

    # excel
    excel_path = '../data/main.xlsx'
    old_time = prev_time
    data = []
    time_interval_s = 5

    # init roi box
    roi_box = (0, 0, 0, 0)

    # main loop
    while cap.isOpened():
        pts = []
        mask = np.zeros((np.array(pts).shape[0], 1), dtype=np.bool_)
        right_now = None

        # read frame
        success, frame = cap.read()

        # Check if frame is read
        if not success:
            break

        # keyboard events
        key = cv2.waitKey(1)
        if key & 0xff == 27:
            break
        elif key == ord('k'):
            # Select ROI
            roi_box = cv2.selectROI(windowName="Webcam Feed",
                                    img=frame,
                                    showCrosshair=True,
                                    fromCenter=False)
            cv2.destroyWindow("Webcam Feed")
            have_poly = False

        # select region to detect
        if roi_box != (0, 0, 0, 0):
            # Crop image
            cv2.rectangle(frame, (roi_box[0], roi_box[1]), (roi_box[0] + roi_box[2], roi_box[1] + roi_box[3]),
                          color=(0, 0, 0), thickness=2)
            roi = frame[int(roi_box[1]):int(roi_box[1] + roi_box[3]), int(roi_box[0]):int(roi_box[0] + roi_box[2])]

            roi_cp = np.copy(roi)
            if not have_poly:
                poly_draw = PolygonDrawer("Choose region", roi, (255, 0, 0), (0, 0, 255))
                poly_pts = poly_draw.run()
                if poly_pts:
                    poly_pts.append(poly_draw.points[0])
                    have_poly = True

            # draw polygon if drawing is done
            if poly_draw is not None and poly_pts:
                if poly_draw.points:
                    pts_draw = np.copy(np.array(poly_pts))
                    pts_draw[:, 0] = np.array(poly_pts)[:, 0] + roi_box[0]
                    pts_draw[:, 1] = np.array(poly_pts)[:, 1] + roi_box[1]
                    cv2.polylines(frame, pts_draw.reshape((1, -1, 2)), True, poly_draw.final_color, 2)

            # TODO: tune nms and score threshold
            # detect
            class_label_ids, confidences, bboxes = detector.net.detect(roi_cp, confThreshold=0.45, nmsThreshold=0.1)

            if list(class_label_ids):
                # only human
                human_idx = np.where(class_label_ids == 1)[0]
                confidences = confidences[human_idx]
                bboxes = bboxes[human_idx]

                bboxes = list(bboxes)
                confidences = list(np.array(confidences).reshape(1, -1)[0])
                confidences = list(map(float, confidences))

                # draw bounding box
                if len(bboxes) != 0:
                    for bbox_id in range(len(bboxes)):
                        bbox = bboxes[bbox_id]
                        class_conf = confidences[bbox_id]
                        x, y, w, h = bbox
                        x += roi_box[0]
                        y += roi_box[1]
                        pts.append((x + w // 2, y + h))
                        cv2.rectangle(
                            img=frame,
                            pt1=(x, y),
                            pt2=(x + w, y + h),
                            color=(0, 255, 255),
                            thickness=2
                        )

                        cv2.putText(
                            img=frame,
                            text=f'{class_conf * 100:.0f} %',
                            org=(x, y - 10),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1.5,
                            color=(255, 0, 255),
                            thickness=2
                        )

                    if pts and list(pts_draw):
                        mask = is_inside_sm_parallel(np.array(pts), pts_draw)
                    in_idx = np.where(mask)[0]  # index of point that is inside polygon
                    right_now = str(datetime.datetime.now())
                    # print(np.sum(mask), right_now)

                    # draw chosen points
                    for idx in range(len(pts)):
                        if idx in in_idx:
                            cv2.circle(frame, pts[idx], 5, (0, 0, 255), thickness=-1)
                        else:
                            cv2.circle(frame, pts[idx], 5, (0, 255, 255), thickness=1)

        # FPS calculation
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time)
        prev_time = cur_time
        cv2.putText(
            img=frame,
            text=f'{int(fps)}',
            org=(0, 20),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1.5,
            color=(255, 0, 255),
            thickness=2
        )

        # excel records
        if cur_time - old_time > time_interval_s and right_now is not None:
            data.append([np.sum(mask), right_now])
            old_time = cur_time

        # display
        cv2.imshow('Webcam Feed', frame)

    # write excel
    df = pd.DataFrame(np.array(data).reshape(-1, 2), columns=['Number of people', 'Date and Time'])
    df.to_excel(excel_path, sheet_name='test1', index=False, header=True)

    # release webcam
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
