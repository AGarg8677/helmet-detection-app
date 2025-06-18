import time
import threading
import simpleaudio as sa

last_triggered = {}

def play_alarm(sound_path):
    try:
        wave_obj = sa.WaveObject.from_wave_file(sound_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()  # Wait for playback to finish
    except Exception as e:
        print(f"Failed to play alarm: {e}")

def trigger_alarm(camera_id, sound_path, cooldown):
    now = time.time()
    if camera_id not in last_triggered or now - last_triggered[camera_id] > cooldown:
        print(f"[ALARM] No helmet detected on Camera {camera_id}")
        threading.Thread(target=play_alarm, args=(sound_path,)).start()
        last_triggered[camera_id] = now

