# FPV Auto Trimmer
- Detects frames when the drone is sitting on the ground at the start of of FPV drone video clips, and **trims the frames out.**
- Processes **multiple videos** automatically (sequentially).
- Uses **Optical Flow** to track motion.
- Detects **takeoff events** based on motion intensity.

![image](https://github.com/user-attachments/assets/92479965-0f6f-4f43-ae98-29b20ec7581a)
![image](https://github.com/user-attachments/assets/8aa4c082-385c-4aac-8fd0-114df681cb33)

![image](https://github.com/user-attachments/assets/e8b74f78-ecc8-45e4-a8a9-3a412144e491)

## Why?
I had a lot of videos taking up unnecessary space, but also didn't want to delete them. I wanted to build a tool that quickly and easily keep only the main flights.

## Compatibility
So far, only tested to be working on .MOV format H.264 codec videos (from a Runcam 3).

## Performance
Approx realtime - takes around 60s for a 60s clip.

## **Running the Script**
``pip install opencv-python``
Add your MOV videos to \input
Run the script

- Scans all videos in `'input'` folder.
- Detects motion and trims videos.
- Saves results in `'output'` folder.
