import subprocess

VIDEO_ID = "7FwXDMucDfM"

# download youtube video with id l4ODJpnzzx4 using yt-dlp in a subprocess and save it to disk
# subprocess.run(["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best", "-o", f"{VIDEO_ID}.mp4", f"https://www.youtube.com/watch?v={VIDEO_ID}"])

# subprocess that extracts the frames from second 5 to second 8 of the video as images and write them to the directory "frames"
subprocess.call([
    "ffmpeg", "-i", 
    f"/home/user/thumbnail-stable-diffusion/data/{VIDEO_ID}.mp4", 
    "-ss", "00:00:00", "-to", "00:02:43", "-vf", "fps=10", 
    "/home/user/thumbnail-stable-diffusion/data/frames-disney-junior-halloween/image-%03d.jpg"])