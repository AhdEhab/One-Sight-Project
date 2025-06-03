import os
import torch
import torchaudio
from moviepy.editor import VideoFileClip
import nemo.collections.asr as nemo_asr
import tempfile


def chunk_audio_tensor(audio_tensor, sample_rate=16000, chunk_duration=30):
    samples_per_chunk = chunk_duration * sample_rate
    num_samples = audio_tensor.size(0)

    chunks = []
    for start in range(0, num_samples, samples_per_chunk):
        end = start + samples_per_chunk
        chunk = audio_tensor[start:end]

        if chunk.size(0) < samples_per_chunk:
            pad_length = samples_per_chunk - chunk.size(0)
            chunk = torch.nn.functional.pad(chunk, (0, pad_length), "constant")

        chunks.append(chunk)

    return chunks


class VideoToAudioProcessor:
    def __init__(self, device=None, target_sample_rate=16000, chunk_duration=30):
        self.target_sample_rate = target_sample_rate
        self.chunk_duration = chunk_duration
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=44100,
            new_freq=target_sample_rate
        ).to(self.device)

        nvidia_model = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
        self.asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name=nvidia_model
        ).to(self.device)

    def extract_audio(self, video_path, audio_path):
        try:
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            return True
        except Exception as e:
            print(f"[ERROR] Audio extraction failed: {e}")
            return False

    def preprocess_audio(self, audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"[ERROR] Audio loading failed: {e}")
            return None

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.to(self.device)

        if sample_rate != self.resampler.orig_freq:
            self.resampler.orig_freq = sample_rate

        if sample_rate != self.target_sample_rate:
            waveform = self.resampler(waveform)

        return waveform.squeeze(0)

    def transcribe_video(self, video_path):
        temp_audio_dir = "./vtt/temp_audio"
        os.makedirs(temp_audio_dir, exist_ok=True)

        video_file = os.path.basename(video_path)
        audio_filename = os.path.splitext(video_file)[0] + ".wav"
        audio_path = os.path.join(temp_audio_dir, audio_filename)

        if not os.path.exists(audio_path):
            success = self.extract_audio(video_path, audio_path)
            if not success:
                return "[ERROR extracting audio]"

        processed_audio = self.preprocess_audio(audio_path)
        if processed_audio is None:
            return "[ERROR preprocessing audio]"

        chunks = chunk_audio_tensor(processed_audio, self.target_sample_rate, self.chunk_duration)

        full_transcription = ""

        for chunk in chunks:
            hash_value = hash(chunk.cpu().numpy().tobytes())
            temp_dir = tempfile.gettempdir()
            temp_wav_path = os.path.join(temp_dir, f"temp_{abs(hash_value)}.wav")
            torchaudio.save(temp_wav_path, chunk.unsqueeze(0).cpu(), self.target_sample_rate)

            try:
                transcription = self.asr_model.transcribe([temp_wav_path], verbose=False)[0].text
                full_transcription += transcription + " "
            except Exception as e:
                print(f"[ERROR] Transcription failed: {e}")

            os.remove(temp_wav_path)
        os.remove(audio_path)

        return full_transcription.strip()
        
def get_transcript(uploaded_file):
    import tempfile
    import shutil

    temp_video_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            shutil.copyfileobj(uploaded_file, temp_video)
            temp_video_path = temp_video.name

        processor = VideoToAudioProcessor()
        transcript = processor.transcribe_video(temp_video_path)

    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    return transcript


