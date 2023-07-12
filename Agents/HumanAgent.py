import cv2
import numpy as np
from EnvUtils import VehicleAction

class HumanAgent:
    def __init__(self, height=512, width=512):
        self.height = height
        self.width = width

        self.control_board = self.drawControlBoard()

        self.control_signal_received = False
        self.action = VehicleAction.getIdleAction()
        self.showControlBoard()


    def drawControlBoard(self) -> np.ndarray:
        # Initialize empty board
        control_board = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Draw the middle axis (vertical line)
        middle_x = self.width // 2
        cv2.line(control_board, (middle_x, 0), (middle_x, self.height), (255, 0, 0), 1)

        # Draw the middle axis (horizontal line)
        middle_y = self.height // 2
        cv2.line(control_board, (0, middle_y), (self.width, middle_y), (255, 0, 0), 1)

        # Draw throttle - brake split line
        cv2.line(control_board, (0, int(self.height * 0.25)), (self.width, int(self.height * 0.25)), (0, 255, 0), 1)
        cv2.line(control_board, (0, int(self.height * 0.75)), (self.width, int(self.height * 0.75)), (0, 255, 0), 1)

        # draw forward text
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 1.0
        text_thickness = 2

        text = "forward"
        text_size, _ = cv2.getTextSize(text, text_font, text_scale, text_thickness)
        text_position = ((self.width - text_size[0]) // 2, text_size[1] + 10)
        cv2.putText(control_board, text, text_position, text_font, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)

        # draw backward text
        text = "backward"
        text_size, _ = cv2.getTextSize(text, text_font, text_scale, text_thickness)
        text_position = ((self.width - text_size[0]) // 2, self.height - 10)
        cv2.putText(control_board, text, text_position, text_font, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)

        text = "left"
        text_size, _ = cv2.getTextSize(text, text_font, text_scale, text_thickness)
        text_position = (10, (self.height + text_size[1]) // 2)
        cv2.putText(control_board, text, text_position, text_font, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)

        text = "right"
        text_size, _ = cv2.getTextSize(text, text_font, text_scale, text_thickness)
        text_position = (self.width - text_size[0] - 10, (self.height + text_size[1]) // 2)
        cv2.putText(control_board, text, text_position, text_font, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)

        text = "brake"
        text_size, _ = cv2.getTextSize(text, text_font, text_scale, text_thickness)
        text_position = ((self.width - text_size[0]) // 2, (self.height + text_size[1]) // 2)
        cv2.putText(control_board, text, text_position, text_font, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)

        return control_board


    def onControlBoardClick(self, event, x, y, flags, param):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.action = VehicleAction((1 - y / self.height) * 4 - 2, x / self.width * 2 - 1)
            self.control_signal_received = True
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_LBUTTONDOWN:
            self.action = VehicleAction((1 - y / self.height) * 4 - 2, x / self.width * 2 - 1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.control_signal_received = False


    def showControlBoard(self):
        cv2.imshow('Control', self.control_board)
        cv2.setMouseCallback('Control', self.onControlBoardClick)


    def getAction(self, *args, **kwargs):
        window_status = cv2.getWindowProperty("Control", cv2.WND_PROP_VISIBLE)
        if window_status < 0:
            # window is closed
            self.showControlBoard()
        k = cv2.waitKey(1)
        return self.action, self.control_signal_received, k


if __name__ == "__main__":
    agent = HumanAgent()
    agent.showControlBoard()