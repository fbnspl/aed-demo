# aed-demo

Demonstrator and visualization for real time prediction of audio stream with python3.6 based on small CNN. 
Network was trained on 4 classes: key chain rattle [keys], quiet background noise [none], paper crumble [paper], whistling [whistle] 

## install
pip install -r requirements.txt

## usage
python3 demo.py

## notes
Only works in a quiet environment, no background noise or other source separation methods implemented.
Uses your default sound device to record audio.
