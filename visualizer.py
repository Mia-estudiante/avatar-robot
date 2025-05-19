import torch
import cv2

class Visualizer:
    def __init__(
        self,
        video_width: int,
        video_height: int,
        window_name: str = "Segmented"
    ):
        self.video_width = video_width
        self.video_height = video_height
        self.window_name = window_name

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.video_width, self.video_height)

    def resize_mask(self, mask):
        mask = torch.tensor(mask, device='cpu')
        mask = torch.nn.functional.interpolate(
            mask,
            size=(self.video_height, self.video_width),
            mode="bilinear",
            align_corners=False,
        )
        return mask

    def add_frame(self, frame, mask):
        frame = cv2.resize(frame, (self.video_width, self.video_height))

        mask = self.resize_mask(mask)
        mask = (mask > 0.0).numpy()

        overlay = frame.copy()
        color = (255, 105, 180)
        for i in range(mask.shape[0]):
            obj = mask[i, 0, :, :]
            overlay[obj] = color

        vis = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        cv2.imshow(self.window_name, vis)
        cv2.waitKey(1)