from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import librosa
import numpy as np
from typing import Dict

app = FastAPI()

def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, threshold=0.2,
                                       ref=np.mean)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    times = librosa.frames_to_time(onsets, sr=sr)
    durations = np.diff(times, append=librosa.get_duration(y=y, sr=sr))
    dynamics = librosa.feature.rms(y=y)[0]
    
    return pitches, times, durations, dynamics

def trim_features_to_same_length(*features):
    min_length = min(len(feature) for feature in features)
    return [feature[:min_length] for feature in features]

def compute_tempo_consistency(orig_times, rec_times) -> float:
    # Compute the intervals between notes
    orig_intervals = np.diff(orig_times)
    rec_intervals = np.diff(rec_times)
    
    # Ensure both intervals have the same length for comparison
    min_length = min(len(orig_intervals), len(rec_intervals))
    orig_intervals = orig_intervals[:min_length]
    rec_intervals = rec_intervals[:min_length]
    
    if len(orig_intervals) == 0 or len(rec_intervals) == 0:
        return 0  # No intervals to compare
    
    # Compute the differences between corresponding intervals
    interval_diff = np.abs(orig_intervals - rec_intervals)
    
    # Calculate mean and standard deviation of the interval differences
    mean_diff = np.mean(interval_diff)
    std_diff = np.std(interval_diff)
    
    # Normalize the mean difference
    mean_diff_norm = mean_diff / (np.max(orig_intervals) + 1e-10)  # Prevent division by zero
    tempo_score = 10 - (mean_diff_norm * 10 + std_diff * 10)  # Adjust sensitivity and scale
    
    return max(0, tempo_score)  # Ensure non-negative scores

def compute_pitch_accuracy(orig_pitches, rec_pitches) -> float:
    tolerance = 1.0  # Tolerance in semitones for pitch accuracy

    # Ensure pitches are in the same shape
    min_length = min(orig_pitches.shape[1], rec_pitches.shape[1])
    orig_pitches = orig_pitches[:, :min_length]
    rec_pitches = rec_pitches[:, :min_length]

    # Compute the pitch differences
    pitch_diffs = np.abs(orig_pitches - rec_pitches)
    
    # Determine the number of notes within tolerance
    within_tolerance = pitch_diffs <= tolerance
    accuracy_scores = np.mean(np.sum(within_tolerance, axis=0) / within_tolerance.shape[0])
    
    # Adjust accuracy score
    max_score = 10
    if accuracy_scores == 1.0:
        pitch_accuracy_score = max_score
    else:
        pitch_accuracy_score = max_score * (accuracy_scores)
    
    return round(min(pitch_accuracy_score, 10))  # Ensure score is within 0-10


def compute_note_duration_accuracy(orig_durations, rec_durations) -> float:
    # Ensure arrays are at least 2D
    if len(orig_durations.shape) < 2:
        orig_durations = np.expand_dims(orig_durations, axis=0)
    if len(rec_durations.shape) < 2:
        rec_durations = np.expand_dims(rec_durations, axis=0)
    # Ensure durations are in the same shape
    min_length = min(orig_durations.shape[1], rec_durations.shape[1])
    
    # Slice arrays to the same length
    orig_durations = orig_durations[:, :min_length]
    rec_durations = rec_durations[:, :min_length]
    
    # Check if arrays are empty after slicing
    if orig_durations.size == 0 or rec_durations.size == 0:
        return 0  # Or handle as needed
    
    # Compute the difference in note durations
    duration_diffs = np.abs(orig_durations - rec_durations)
    
    # Average difference in note durations
    avg_duration_diff = np.mean(duration_diffs)
    
    # Assuming a maximum allowable duration difference, adjust if needed
    max_allowed_diff = np.max(orig_durations)  # or a fixed value
    if max_allowed_diff == 0:  # Prevent division by zero
        return 10
    
    duration_score = max(0, 10 - (avg_duration_diff / max_allowed_diff) * 10)
    
    return round(duration_score)

def compute_articulation_accuracy(orig_durations, rec_durations) -> float:
    # Ensure arrays are at least 2D
    if len(orig_durations.shape) < 2:
        orig_durations = np.expand_dims(orig_durations, axis=0)
    if len(rec_durations.shape) < 2:
        rec_durations = np.expand_dims(rec_durations, axis=0)
    
    # Ensure durations are in the same shape
    min_length = min(orig_durations.shape[1], rec_durations.shape[1])
    
    # Slice arrays to the same length if they are longer
    orig_durations = orig_durations[:, :min_length]
    rec_durations = rec_durations[:, :min_length]
    
    # Check if arrays are empty after slicing
    if orig_durations.size == 0 or rec_durations.size == 0:
        return 0  # Or handle as needed
    
    # Compute the differences in the articulation (e.g., note spacing)
    orig_articulation = np.diff(orig_durations, axis=1)
    rec_articulation = np.diff(rec_durations, axis=1)
    
    # Calculate difference in articulation
    articulation_diffs = np.abs(orig_articulation - rec_articulation)
    
    # Average articulation difference
    avg_articulation_diff = np.mean(articulation_diffs)
    
    # Assuming a maximum allowable articulation difference, adjust if needed
    max_allowed_diff = np.max(orig_articulation) if np.max(orig_articulation) != 0 else 1  # Prevent division by zero
    articulation_score = max(0, 10 - (avg_articulation_diff / max_allowed_diff) * 10)
    
    return round(articulation_score)


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
    pitch_accuracy = compute_pitch_accuracy(orig_pitches, rec_pitches)
    scores['Pitch Accuracy'] = pitch_accuracy
    
    # Compare Timing/Rhythm
    timing_diff = np.mean(np.abs(orig_times - rec_times))
    scores['Timing/Rhythm'] = round(max(0, 10 - timing_diff))
    
    # Compare Note Duration
    note_duration_accuracy = compute_note_duration_accuracy(orig_durations, rec_durations)
    scores['Note Duration'] = note_duration_accuracy
    
    # Evaluate Tempo Consistency
    tempo_score = compute_tempo_consistency(orig_times, rec_times)
    scores['Tempo Consistency'] = round(tempo_score)
    
    # Compare Dynamics based on absolute differences and standard deviation
    dynamics_diff = np.mean(np.abs(orig_dynamics - rec_dynamics))
    dynamics_std = np.std(np.abs(orig_dynamics - rec_dynamics))
    dynamics_score = max(0, 10 - ((dynamics_diff * 20) + (dynamics_std * 30)))  # Adjusted sensitivity factor
    scores['Dynamics'] = round(dynamics_score)
    
    # Check Articulation
    articulation_score = compute_articulation_accuracy(orig_durations, rec_durations)
    scores['Articulation'] = round(articulation_score)
    
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