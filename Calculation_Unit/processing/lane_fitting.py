import numpy as np
from skimage.morphology import skeletonize


class LaneFitting:
    """车道线拟合工具"""

    @staticmethod
    def fit_lane_points_and_draw(frame, skeleton, color=(0, 0, 255), lane_id=None):
        y_coords, x_coords = np.where(skeleton > 0)
        if len(x_coords) < 4:
            return frame, None, None

        center_x = int(np.mean(x_coords))
        poly_coeffs = np.polyfit(y_coords, x_coords, 3)
        poly_func = np.poly1d(poly_coeffs)

        y_vals = np.linspace(min(y_coords), max(y_coords), num=500)
        x_vals = poly_func(y_vals)
        curve_points = np.array([x_vals, y_vals]).T.astype(int)

        for point in curve_points:
            x, y = point
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                frame[y, x] = color

        if lane_id is not None:
            label_pos = (center_x, int(min(y_coords)))
            frame = cv2.putText(
                frame,
                f"L{lane_id}",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        return frame, center_x, curve_points
