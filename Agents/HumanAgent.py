import cv2
import numpy as np
from EnvUtils import VehicleAction

class HumanAgent:
    def __init__(self, height=512, width=512, demonstrate_mode=False):
        self.height = height
        self.width = width
        self.demonstrate_mode = demonstrate_mode
        self.mouse_pos = (256, 256)

        self.control_board = self.drawControlBoard()
        if self.demonstrate_mode:
            self.display_board = np.zeros_like(self.control_board)

        self.control_signal_received = False
        self.action = VehicleAction.getStopAction()
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
        cv2.line(control_board, (0, int(self.height * 0.375)), (self.width, int(self.height * 0.375)), (0, 255, 0), 1)
        cv2.line(control_board, (0, int(self.height * 0.625)), (self.width, int(self.height * 0.625)), (0, 255, 0), 1)

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


    def onControlBoardOperate(self, event, x, y, flags, param):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # if hold control, is_human_action = False
            human_action = flags != cv2.EVENT_FLAG_CTRLKEY
            self.action = VehicleAction((1 - y / self.height) * 4 - 2, x / self.width * 2 - 1, is_human_action=human_action)
            self.control_signal_received = True
            self.mouse_pos = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and \
            ((flags == cv2.EVENT_LBUTTONDOWN | cv2.EVENT_FLAG_CTRLKEY) or flags == cv2.EVENT_LBUTTONDOWN):

            # if hold mouse and control, is_human_action = False
            human_action = flags != cv2.EVENT_FLAG_CTRLKEY | cv2.EVENT_FLAG_LBUTTON
            self.action = VehicleAction((1 - y / self.height) * 4 - 2, x / self.width * 2 - 1, is_human_action=human_action)
            self.mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.control_signal_received = False
            self.mouse_pos = (self.width//2, self.height//2)



    def showControlBoard(self):
        if self.demonstrate_mode:
            cv2.imshow("Demo", self.display_board)
        cv2.imshow('Control', self.control_board)
        cv2.setMouseCallback('Control', self.onControlBoardOperate)


    def getAction(self, *args, **kwargs):
        window_status = cv2.getWindowProperty("Control", cv2.WND_PROP_VISIBLE)
        if window_status < 0:
            # window is closed
            self.showControlBoard()
        if self.demonstrate_mode:
            demo_control = self.control_board.copy()
            cv2.circle(demo_control, self.mouse_pos, 5, (255, 255, 255), 1)
            cv2.line(demo_control, self.mouse_pos, (512+100, 256), (255, 255, 255), 1)
            demo = np.zeros_like(demo_control)
            start_y = int(256 - 100 * (256 - self.mouse_pos[1]) / (512 + 100 - self.mouse_pos[0]))
            print(start_y)
            cv2.line(demo, (0, start_y), (100, 256), (255, 255, 255), 1)
            cv2.putText(demo, str(self.action), (100, 256), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Control", demo_control)
            cv2.imshow("Demo", demo)
        k = cv2.waitKey(1)
        return self.action, self.control_signal_received, k


if __name__ == "__main__":
    agent = HumanAgent()
    agent.showControlBoard()