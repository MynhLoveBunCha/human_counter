import cv2
import numpy as np
import numba
from numba import jit, njit


class Detector:

    def __init__(self, video_path, config_path, model_path, classes_path):
        self.video_path = video_path
        self.config_path = config_path
        self.model_path = model_path
        self.classes_path = classes_path
        ########################
        self.net = cv2.dnn_DetectionModel(self.model_path, self.config_path)
        self.net.setInputSize((320, 320))
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
        self.classes_list = self.readClasses()
        ############

    def readClasses(self):
        with open(self.classes_path, 'r') as f:
            classes_list = f.read().splitlines()
        classes_list.insert(0, '__Background__')
        return classes_list


class PolygonDrawer(object):
    def __init__(self, window_name, img, working_color, final_color):
        self.window_name = window_name  # Name for our window
        self.img = img
        self.done = False  # Flag signalling we're done
        self.current = (0, 0)  # Current position, so we can draw the line-in-progress
        self.points = []  # List of points defining our polygon
        self.working_color = working_color
        self.final_color = final_color

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done:  # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            # print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            # print("Completing polygon with %d points." % len(self.points))
            self.done = True

    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, self.img)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while not self.done:
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            canvas = np.copy(self.img)
            if len(self.points) > 0:
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, self.final_color, 2)
                # And  also show what the current segment would look like
                cv2.line(canvas, self.points[-1], self.current, self.working_color, 1)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27:  # ESC hit
                self.done = True

        # User finished entering the polygon points, so let's make the final drawing
        canvas = self.img
        # of a filled polygon
        if len(self.points) > 0:
            cv2.polylines(canvas, np.array([self.points]), True, self.final_color, 2)
        # And show it
        cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        return self.points


@jit(nopython=True)
def is_inside_sm(polygon, point):
    length = len(polygon) - 1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii < length:
        dy = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy * dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy < 0 or dy2 < 0:
                F = dy * (polygon[jj][0] - polygon[ii][0]) / (dy - dy2) + polygon[ii][0]

                if point[0] > F:  # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F:  # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2 == 0 and (point[0] == polygon[jj][0] or (
                    dy == 0 and (point[0] - polygon[ii][0]) * (point[0] - polygon[jj][0]) <= 0)):
                return 2

        ii = jj
        jj += 1

#     # print 'intersections =', intersections
    return intersections & 1


@njit(parallel=True)
def is_inside_sm_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean)
    for i in numba.prange(ln):
        D[i] = is_inside_sm(polygon, points[i])
    return D
