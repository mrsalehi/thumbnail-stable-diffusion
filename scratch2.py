import multiprocessing as mp
import subprocess



# a pool of processes where each one opens a subprocess that uses yt-dlp to download the videos ids of that channel and write it into channel_name/videos_ids.txt file
def get_video_ids(channel):
    subprocess.run(
        [
            "yt-dlp",
            "--get-id",
            "--skip-download",
            "--output",
            f"/home/user/videos_ids/{channel}/%(id)s.%(ext)s",
            f"https://www.youtube.com/user/{channel}/videos",
            ])