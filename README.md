# FPV Auto Trimmer
[![Pylint](https://github.com/matthewww/fpv-auto-trimmer/actions/workflows/pylint.yml/badge.svg)](https://github.com/matthewww/fpv-auto-trimmer/actions/workflows/pylint.yml)
[![Publish to PyPI](https://github.com/matthewww/fpv-auto-trimmer/actions/workflows/publish.yml/badge.svg)](https://github.com/matthewww/fpv-auto-trimmer/actions/workflows/publish.yml)
- Detects frames when the drone is sitting on the ground at the start of of FPV drone video clips, and **trims the frames out.**
- Processes **multiple videos** automatically (sequentially).
- Uses **[Optical Flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)** to track motion.
- Detects **takeoff events** based on motion intensity.

![image](https://github.com/user-attachments/assets/92479965-0f6f-4f43-ae98-29b20ec7581a)
![image](https://github.com/user-attachments/assets/8aa4c082-385c-4aac-8fd0-114df681cb33)

![image](https://github.com/user-attachments/assets/e8b74f78-ecc8-45e4-a8a9-3a412144e491)

## **Running the Script**
``pip install opencv-python``
Add your MOV videos to \input
Run the script

- Scans all videos in `'input'` folder.
- Detects motion and trims videos.
- Saves results in `'output'` folder.
  
## Why?
I had a lot of videos taking up unnecessary space, but also didn't want to delete them. I wanted to build a tool that quickly and easily keep only the main flights.

Also I just wanted an excuse to try out a Python / Github CI/CD workflow 😛

## Compatibility
So far, only tested to be working on .MOV format H.264 codec videos (from a Runcam 3).

## Performance
Around 60s for a 60s clip (i7 6700k, 16 GB)

