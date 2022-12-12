import cv2
from pathlib import Path
def video_to_img(video_path, output_path):
    
    # id = Path(video_path).stem
    Path(output_path).mkdir(exist_ok=True)
    # save_dir = Path(output_path) / id
    # save_dir_images_path = create_folder_template(save_dir)
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    success, image = video.read()
    count = 0
    list_of_frames = []
    while success:
        cv2.imwrite(str(output_path / "frame_{}.jpg".format(count)), image)     # save frame as JPEG file      
        success,image = video.read()
        print('Reading frame:', count)
        # list_of_frames.append(str(output_path / "{}_frame_{}.jpg".format(id, count)))
        count += 1

if __name__ == '__main__':
    video_to_img(
        "/root/data/ltnghia/projects/visual_communication/htluc/custom_code/00421.mp4",
        Path("/root/data/ltnghia/projects/visual_communication/htluc/custom_code/00421"),
    )