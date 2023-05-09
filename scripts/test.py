from ultil import *
import PySimpleGUI as Psg
import cv2
import numpy as np
import os
import pandas as pd
import datetime
import time


def main():
    Psg.theme('DarkAmber')
    font = ('Consolas', 12, 'bold')

    # Define the window layout
    webcam_column = [
        [Psg.Text("Live Webcam Feed", size=(60, 1), justification="center", font=font, expand_x=True, expand_y=True)],
        [Psg.Image(filename="", key="-IMAGE-", expand_x=True, expand_y=True)],
        [
            Psg.Button("Choose Or Reset Surveillance Region", auto_size_button=True, font=font, expand_x=True,
                       expand_y=True),
            Psg.Button("Exit", auto_size_button=True, font=font, expand_x=True, expand_y=True)
        ],
    ]

    result_display_column = [
        [
            Psg.Text("Number of customers: ", expand_x=True, expand_y=True),
            Psg.Text(text="", justification="center", auto_size_text=True, font=font, key="-CUS NUM-", expand_x=True,
                     expand_y=True)
        ]
    ]

    layout = [
        [
            Psg.Column(webcam_column),
            Psg.VSeperator(),
            Psg.Column(result_display_column)
        ]
    ]

    # Create the window
    main_window = Psg.Window("Customers counter",
                             layout,
                             background_color="#8DDF5F",
                             location=(200, 100),
                             resizable=True)

    # Capture webcam feed
    vid_source = None
    while vid_source is None or not vid_source.isdecimal():
        vid_source = Psg.popup_get_text(  # get video source
            message='Enter your camera ID (HINT: must be positive decimal number e.g, 0, 1, 2)',
            title='Enter camera ID',
            size=(5, 1)
        )
        if vid_source is None:
            main_window.close()
            exit()

    cap = cv2.VideoCapture(int(vid_source), cv2.CAP_DSHOW)
    while not cap.isOpened():  # check if video source is valid
        Psg.popup(f'Can not open camera with ID: {vid_source}', auto_close=True, auto_close_duration=2)
        vid_source = None
        while vid_source is None or not vid_source.isdecimal():
            vid_source = Psg.popup_get_text(  # get video source
                message='Enter your camera ID (HINT: must be positive decimal number e.g, 0, 1, 2)',
                title='Enter camera ID',
                size=(5, 1)
            )
            if vid_source is None:
                cap.release()
                main_window.close()
                exit()
        cap = cv2.VideoCapture(int(vid_source), cv2.CAP_DSHOW)

    # initialize object detector
    config_path = os.path.join('..', 'models', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    model_path = os.path.join('..', 'models', 'frozen_inference_graph.pb')
    classes_path = os.path.join('..', 'models', 'coco.names')
    detector = Detector(int(vid_source), config_path, model_path, classes_path)

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

    # Event loop
    while cap.isOpened():
        pts = []
        mask = np.zeros((np.array(pts).shape[0], 1), dtype=np.bool_)
        right_now = None
        customer_count = None

        main_event, main_values = main_window.read(timeout=1)
        success, frame = cap.read()

        if not success:
            Psg.popup("No camera detected!", font=font)
            break

        if main_event == "Exit" or main_event == Psg.WIN_CLOSED:
            break

        if main_event == "Choose Or Reset Surveillance Region":
            Psg.popup_scrolled(
                'Choose surveillance region using your mouse\n'
                'Hit <ENTER> to finish selection\n'
                'Hit <c> to cancel selection',
                title='Instruction',
            )
            # Select ROI
            roi_box = cv2.selectROI(windowName="Choose Surveillance Region",
                                    img=frame,
                                    showCrosshair=True,
                                    fromCenter=False)
            cv2.destroyWindow("Choose Surveillance Region")
            have_poly = False

        # select restricted region
        if roi_box != (0, 0, 0, 0):
            # Crop image
            cv2.rectangle(frame, (roi_box[0], roi_box[1]), (roi_box[0] + roi_box[2], roi_box[1] + roi_box[3]),
                          color=(0, 0, 0), thickness=2)
            roi = frame[int(roi_box[1]):int(roi_box[1] + roi_box[3]), int(roi_box[0]):int(roi_box[0] + roi_box[2])]

            roi_cp = np.copy(roi)
            if not have_poly:
                Psg.popup_scrolled(
                    'Draw restricted region using left click to select points\n'
                    'Right click to auto enclose the region\n'
                    'Hit <ESC> to confirm selection\n',
                    title='Instruction',
                )
                poly_draw = PolygonDrawer("Choose restricted region", roi, (255, 0, 0), (0, 0, 255))
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
                    customer_count = np.sum(mask)

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
            text=f'FPS: {int(fps)}',
            org=(0, 20),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1.5,
            color=(0, 255, 0),
            thickness=2
        )

        # excel records
        if cur_time - old_time > time_interval_s and right_now is not None:
            data.append([customer_count, right_now])
            old_time = cur_time

        # customer count update
        if customer_count is None:
            main_window["-CUS NUM-"].update('')  # display number of customers
        else:
            main_window["-CUS NUM-"].update(str(customer_count))  # display number of customers

        # update webcam feed viewer
        img_bytes = cv2.imencode(".png", frame)[1].tobytes()
        main_window["-IMAGE-"].update(data=img_bytes)

    # write excel
    df = pd.DataFrame(np.array(data).reshape(-1, 2), columns=['Number of people', 'Date and Time'])
    df.to_excel(excel_path, sheet_name='test1', index=False, header=True)

    # release webcam and close window
    main_window.close()
    cap.release()


if __name__ == '__main__':
    main()
