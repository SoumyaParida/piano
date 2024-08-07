from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import librosa
import numpy as np
from typing import Dict

app = FastAPI()

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    times = librosa.frames_to_time(onsets, sr=sr)
    durations = np.diff(times, append=librosa.get_duration(y=y, sr=sr))
    dynamics = librosa.feature.rms(y=y)[0]
    
    return pitches, times, durations, dynamics

def trim_features_to_same_length(*features):
    min_length = min(len(feature) for feature in features)
    return [feature[:min_length] for feature in features]

def compare_features(original_features, recorded_features) -> Dict[str, float]:
    scores = {}
    
    orig_pitches, orig_times, orig_durations, orig_dynamics = original_features
    rec_pitches, rec_times, rec_durations, rec_dynamics = recorded_features
    
    # Trim features to the same length
    orig_times, rec_times = trim_features_to_same_length(orig_times, rec_times)
    orig_durations, rec_durations = trim_features_to_same_length(orig_durations, rec_durations)
    orig_dynamics, rec_dynamics = trim_features_to_same_length(orig_dynamics, rec_dynamics)
    
    # Normalize dynamics
    orig_dynamics = (orig_dynamics - np.min(orig_dynamics)) / (np.max(orig_dynamics) - np.min(orig_dynamics))
    rec_dynamics = (rec_dynamics - np.min(rec_dynamics)) / (np.max(rec_dynamics) - np.min(rec_dynamics))


    # # Compare Pitch Accuracy
    pitch_accuracy = np.mean([np.any(np.isclose(op, rp, atol=0.5)) for op, rp in zip(orig_pitches.T, rec_pitches.T) if np.any(op) and np.any(rp)])
    scores['Pitch Accuracy'] = round(pitch_accuracy * 10)
    
    # Compare Timing/Rhythm
    timing_diff = np.mean(np.abs(orig_times - rec_times))
    scores['Timing/Rhythm'] = round(max(0, 10 - timing_diff))
    
    # Compare Note Duration
    duration_diff = np.mean(np.abs(orig_durations - rec_durations))
    scores['Note Duration'] = round(max(0, 10 - duration_diff))
    
    # Evaluate Tempo Consistency
    orig_intervals = np.diff(orig_times)
    rec_intervals = np.diff(rec_times)
    orig_intervals, rec_intervals = trim_features_to_same_length(orig_intervals, rec_intervals)
    tempo_diff = np.std(orig_intervals - rec_intervals)
    scores['Tempo Consistency'] = round(max(0, 10 - tempo_diff))
    
    # Compare Dynamics based on absolute differences and standard deviation
    dynamics_diff = np.mean(np.abs(orig_dynamics - rec_dynamics))
    dynamics_std = np.std(np.abs(orig_dynamics - rec_dynamics))
    dynamics_score = max(0, 10 - ((dynamics_diff * 15) + (dynamics_std * 30)))  # Adjusted sensitivity factor
    scores['Dynamics'] = round(dynamics_score)
    
    # Check Articulation
    articulation_diff = np.mean(np.abs(np.diff(rec_durations) - np.diff(orig_durations)))
    scores['Articulation'] = round(max(0, 10 - articulation_diff))
    
    # Assess Consistency
    consistency = np.mean([
        scores['Pitch Accuracy'],
        scores['Timing/Rhythm'],
        scores['Note Duration'],
        scores['Tempo Consistency'],
        scores['Dynamics'],
        scores['Articulation']
    ])
    scores['Consistency'] = round(consistency)
    
    return scores

def deliver_feedback(scores: Dict[str, float]) -> str:
    feedback = f"""
    Score Comparison:
    * Pitch Accuracy: {scores['Pitch Accuracy']}/10
    * Timing/Rhythm: {scores['Timing/Rhythm']}/10
    * Note Duration: {scores['Note Duration']}/10
    * Tempo Consistency: {scores['Tempo Consistency']}/10
    * Dynamics: {scores['Dynamics']}/10
    * Articulation: {scores['Articulation']}/10
    * Consistency: {scores['Consistency']}/10
    """
    return feedback

def analyze_piano_performance(original_audio, recorded_audio) -> Dict[str, float]:
    original_features = extract_features(original_audio)
    recorded_features = extract_features(recorded_audio)
    
    scores = compare_features(original_features, recorded_features)
    
    return scores

@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <body>
    <h2>Upload Original and Recorded MP3 Files</h2>
    <form action="/analyze/" enctype="multipart/form-data" method="post">
    <input name="original_file" type="file" accept="audio/mpeg">
    <input name="recorded_file" type="file" accept="audio/mpeg">
    <input type="submit">
    </form>
    </body>
    """
    return HTMLResponse(content=content)

@app.post("/analyze/")
async def analyze_performance(original_file: UploadFile = File(...), recorded_file: UploadFile = File(...)):
    original_audio = original_file.file.read()
    recorded_audio = recorded_file.file.read()

    with open("temp_original.mp3", "wb") as f:
        f.write(original_audio)
    with open("temp_recorded.mp3", "wb") as f:
        f.write(recorded_audio)

    scores = analyze_piano_performance("temp_original.mp3", "temp_recorded.mp3")
    feedback = deliver_feedback(scores)
    
    return {"feedback": feedback}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
