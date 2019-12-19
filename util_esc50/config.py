sample_rate = 44100
window_size = 1764
hop_size = 882
mel_bins = 40
fmin = 50       # Hz
fmax = int(sample_rate/4)    # Hz

frames_per_second = sample_rate // hop_size
audio_duration = 5     # Audio recordings in ESC-50 are all 5 seconds
total_samples = sample_rate * audio_duration
audio_duration_clip = 5
audio_stride_clip = 1
total_frames = frames_per_second * audio_duration
frames_num_clip = int(frames_per_second * audio_duration_clip)
total_samples_clip = int(sample_rate * audio_duration_clip)
frames_num = frames_per_second * audio_duration_clip
audio_num = (audio_duration-audio_duration_clip)//audio_stride_clip + 1
labels = [ 'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects',
          'sheep', 'crow', 'rain', 'sea_waves', 'crackling_fire', 'crickets', 
          'chirping_birds', 'water_drops', 'wind', 'pouring_water', 'toilet_flush',
          'thunderstorm', 'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing',
          'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping',
          'door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks',
          'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm',
          'clock_tick', 'glass_breaking', 'helicopter', 'chainsaw', 'siren', 'car_horn',
          'engine', 'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw']
classes_num = len(labels)
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
