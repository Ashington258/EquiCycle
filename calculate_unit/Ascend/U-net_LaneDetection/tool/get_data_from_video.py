import cv2
import os


def extract_frames(video_path, output_folder, interval):
    """
    Extracts frames from a video at a specific interval and saves them as images.

    :param video_path: Path to the input video file.
    :param output_folder: Folder where the extracted frames will be saved.
    :param interval: Interval (in seconds) between frames to be extracted.
    """
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(
            f"Error: Could not open video. Check the file path and format: {video_path}"
        )
        return

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Save the frame as an image
            frame_filename = os.path.join(
                output_folder, f"frame_{saved_frame_count:06d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_frame_count} frames at {interval} second intervals.")


# Example usage
video_path = r"F:\1.Project\2.Ongoing_Projects\EquiCycle\Calculation_Unit\Ascend\U-net_LaneDetection\data\WIN_20231127_11_17_26_Pro.mp4"
output_folder = r"F:\1.Project\2.Ongoing_Projects\EquiCycle\Calculation_Unit\Ascend\U-net_LaneDetection\extracted_frames"
interval = 1  # Extract a frame every 1 second

# Check if the video file exists
if not os.path.isfile(video_path):
    print(f"Error: The video file does not exist: {video_path}")
else:
    extract_frames(video_path, output_folder, interval)
