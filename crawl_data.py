import psutil
import pickle
from pytube import YouTube
from pytube import Playlist
import zipfile
import argparse
# from slugify import slugify
import ffmpeg
import csv
import pandas as pd
import subprocess
from multiprocessing import Pool
from pathlib import Path
from yt_dlp import YoutubeDL
import json
from tqdm import tqdm
import shutil

# def DownloadVideo(video_link,folder,maxres=None):
#     if maxres==None: 
#         print("Video Started")
#         video_file = YouTube(video_link).streams.order_by('resolution').desc().first().download()
#         print("Video Done") 
#     else:
#         print("Video Started")
#         video_file = YouTube(video_link).streams.filter(res=maxres).order_by('resolution').desc().first().download()
#         print("Video Done")
        
#     # print(video_file) 
#     video_name = slugify(video_file.replace(".webm","").split("/")[-1]) 
#     print("Audio Started")
#     audio_file = YouTube(video_link).streams.filter(only_audio=True).order_by('abr').desc().first().download(filename_prefix="audio_")
#     print("Audio Done")
#     source_audio = ffmpeg.input(audio_file)
#     source_video = ffmpeg.input(video_file)
#     print("Concatenation Started")
#     ffmpeg.concat(source_video, source_audio, v=1, a=1).output(f"{folder}/{video_name}.mp4").run()
#     print("Concatenation Done")
        
#     return None
        

# def DownloadChannel(channel_link,folder,maxres=None): 
#     pure_link = channel_link.replace("/featured","/videos")
#     print(pure_link)
#     list_videos = Playlist(pure_link).video_urls
#     video_count = 0 
#     # total_video = len(list_videos) 
#     print(list_videos)
#     exit(9)
#     print(f'{total_video} Videos Found') 
#     list_videos_downloaded = []

#     with open('youtube_export_history.csv', 'r', newline='') as csvfile:        
#         spamwriter = csv.reader(csvfile, quoting=csv.QUOTE_MINIMAL) 
#         for row in spamwriter: 
#             list_videos_downloaded.append(row[0])
            
#     for video in list_videos: 
#         if video in list_videos_downloaded: 
#             video_count = video_count + 1 
#             print(f'Video {video_count}/{total_video} already downloaded') 
#         else: 
#             print(video) 
#             video_count = video_count + 1
#             print(f'{video_count}/{total_video} Started')
#             DownloadVideo(video_link=video,maxres=maxres,folder=folder)
            
#             with open('youtube_export_history.csv', 'a', newline='') as csvfile:
#                 spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
#                 spamwriter.writerow([video])

#             print(f'{video_count}/{total_video} Done')
            
# DownloadChannel(
#     channel_link="https://www.youtube.com/user/MrBeast6000/videos",
#     folder="fullFolderPath",
#     maxres=None)

def zip_ytube_metadata(root_dir, channel_names, dst_zip_name):
    # zip all the subdirectories with their name in channel_names list and zip them all into a single file

    # create a zip file with context manager and write to it
    for channel_name in tqdm(channel_names):
        with zipfile.ZipFile(f"{root_dir}/{channel_name}.zip", "w") as zip_file:
            # get the path of the subdirectory
            sub_dir = Path(f"{root_dir}/{channel_name}")
            # iterate over all the files in the subdirectory
            for file in tqdm(list(sub_dir.iterdir())):
                # write the file to the zip file and it is still in the subdir in the zip file
                zip_file.write(file, arcname=file.name)


def remove_channel_dirs(root_dir, channel_names):
    """
    DANGER ZONE!!!!: afer you zip the dirs of channels, you can remove the dirs here.
    """
    for channel_name in channel_names:
        channel_dir = Path(f"{root_dir}/{channel_name}")
        shutil.rmtree(channel_dir)


def load_channel_data(csv_fpath):
    csv_path = Path(csv_fpath)
    df = pd.read_csv(csv_path)
    channel_names, channel_urls = df["channel_name"].tolist(), df["channel_url"].tolist()
    # apply preproc_channel_name on each one of the channel names
    channel_names = list(map(prproc_channel_name, channel_names))

    return channel_names, channel_urls


def prproc_channel_name(channel_name):
    # replace any space with underscore
    # channel_name = channel_name.replace(" ", "_")
    # lowecase all letters
    channel_name = channel_name.lower()
    
    return channel_name


def get_title_description(video_id):
    with YoutubeDL() as ydl: 
        info_dict = ydl.extract_info('https://www.youtube.com/watch?v=65fN_OUawjk', download=False)
        video_url = info_dict.get("url", None)
        video_id = info_dict.get("id", None)
        video_title = info_dict.get('title', None)
        video_description = info_dict.get('description', None)
        print("Title: " + video_title) # <= Here, you got the video title
        print("Description: " + video_description) # <= Here, you got the video description


def get_metadata(channel_data):
    channel_name, channel_url = channel_data

    dst_path = f"/home/user/stable-diffusion-webui/data/kids_metadata/{channel_name}/%(id)s.%(ext)s"
    subprocess.run(
        [
            "yt-dlp",
            "--write-info-json",
            "--skip-download",
            "--no-warnings",
            "--output",
            dst_path,
            channel_url,
            ])


def read_thumbnail_data(root_dir): 
    return 


def get_channel_thumbnails(channel_name, channel_url, dst_dir):
    pass


# def get_thumbnail(channel_data, dst_path):
#     channel_name, channel_url = channel_data
#     subprocess.run(
#     [
#         "yt-dlp",
#         "--write-thumbnail",
#         "--skip-download",
#         "--output",
#         f"/home/user/thumbnail-stable-diffusion/thumbnails/{channel_name}/%(id)s.%(ext)s",
#         channel_url,
#         ])

def write_titles_to_dir(src_dir, dst_dir):
    channel_dirs = Path(src_dir).glob("*")

    for channel_dir in tqdm(channel_dirs):
        channel_name = channel_dir.name
        Path(dst_dir, channel_name).mkdir(parents=True, exist_ok=True)
        metadata_files = Path(channel_dir).glob("*.info.json")
        for metadata_file in metadata_files:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            video_id = metadata["id"]
            video_title = metadata["title"]
            # video_description = metadata["description"]
            with open(f"{dst_dir}/{channel_name}/{video_id}.txt", "w") as f:
                f.write(video_title)
                # f.write("\n")
                # f.write(video_description)


def get_thumbnail_urls(root_dir, select_channels=None):
    """
    root_dir: a directoy that contains all of the subdirs that contain the json metadata files
    returns a list of (channel_name, video_id, thumbnail_url) tuples
    """
    # we need a list of tuples (channel_name, video_id, thumbnail_url)
    channel_dirs = Path(root_dir).glob("*")
    thumbnail_data = []
    for channel_dir in channel_dirs:
        channel_name = channel_dir.name
        if channel_name not in select_channels:
            continue
        channel_metadata_files = Path(channel_dir).glob("*.info.json")
        for file in channel_metadata_files:
            with open(file, "r") as f:
                metadata = json.load(f)
            video_id = metadata["id"]
            try: 
                thumbnail_url = metadata["thumbnail"]
                thumbnail_data.append((channel_name, video_id, thumbnail_url))
            except:
                pass
    return thumbnail_data


    # subprocess.run(
    #     [
    #         "yt-dlp",
    #         "--write-thumbnail",
    #         "--skip-download",
    #         "--output",
    #         f"/home/user/thumbnails/{channel}/%(id)s.%(ext)s",
    #         f"https://www.youtube.com/user/{channel}/videos",
    #         ])


def download_thumbnail(thumbnail_data):
    dst_dir = "/home/user/stable-diffusion-webui/data/thumbnail-imgs"
    channel_name, video_id, thumbnail_url = thumbnail_data
    dst_path = Path(dst_dir) / channel_name / f"{video_id}.jpg"
    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        # print(f"Downloading {video_id} to {dst_path}")
        # download the thumbnail from thumbnail_url to dst_path and suppress wget outputs
        subprocess.run(["wget", thumbnail_url, "-q", "-O", dst_path])
    else:
        pass
        # print(f"Skipping {video_id} because it already exists at {dst_path}")


def main():
    # Imagine you have the channel names in the channels list
    # channel_names, channel_urls = load_channel_data('/home/user/thumbnail-stable-diffusion/channels2.csv')
    # a pool of processes where each one opens a subprocess that uses yt-dlp to download the videos ids of a channel and write it into thumbnails/channel_name folder and it does not use get_thumbnail function

    #argparser that reads if we want to get metadata option
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_metadata", action="store_true")
    parser.add_argument("--get_thumbnail_urls", action="store_true")
    parser.add_argument("--download_thumbnails", action="store_true")
    parser.add_argument("--zip_select_channels_metadata", action="store_true")
    parser.add_argument("--write_titles_to_dir", action="store_true")
    # parse the args
    args = parser.parse_args()
    
    if args.get_metadata:
        channel_names, channel_urls = load_channel_data('/home/user/stable-diffusion-webui/thumbnail-stable-diffusion/channels_kids.csv')
        # 1. get the channel videos metadata and write them to disk
        with Pool(processes=len(channel_names)) as pool:
            pool.map(
                get_metadata,
                zip(channel_names, channel_urls),
            )
    elif args.get_thumbnail_urls:
        # 2. get the thumbnail urls and write it as a list to disk
        channel_names, channel_urls = load_channel_data('/home/user/stable-diffusion-webui/thumbnail-stable-diffusion/channels_kids.csv')
        thumbnail_data = get_thumbnail_urls(
            root_dir="/home/user/stable-diffusion-webui/data/kids_metadata", #"/user/disks/sdc/data/ytube_metadata",
            select_channels=channel_names  # only select the new channels that we have not downloaded their thumbnails yet
            )
        pickle.dump(thumbnail_data, open("/home/user/stable-diffusion-webui/data/thumbnail_list.pkl", "wb"))
    elif args.download_thumbnails:
        channel_names, channel_urls = load_channel_data("/home/user/stable-diffusion-webui/thumbnail-stable-diffusion/channels_kids.csv")
        thumbnail_data = pickle.load(open("/home/user/stable-diffusion-webui/data/thumbnail_list.pkl", "rb"))
        # count number of physical cpu cores with psutil
        num_cores = psutil.cpu_count(logical=False)

        # pool of num_cores processes that download the thumbnails of the videos using the thumbnail_data with tqdm progress bar imap_unordered
        with Pool(processes=num_cores) as pool:
            for _ in tqdm(pool.imap_unordered(download_thumbnail, thumbnail_data), total=len(thumbnail_data)):
                pass
    elif args.zip_select_channels_metadata:
        channel_names, channel_urls = load_channel_data('/home/user/user/thumbnail-stable-diffusion/channels.csv')
        zip_ytube_metadata(
            root_dir="/home/user/thumbnail-stable-diffusion/ytube_metadata", 
            channel_names=channel_names, 
            dst_zip_name="metadata_channels_1_19.zip")
    elif args.write_titles_to_dir:
        write_titles_to_dir("/home/user/stable-diffusion-webui/data/kids_metadata",
                            "/home/user/stable-diffusion-webui/data/thumbnail-titles")


if __name__ == "__main__":
    main()