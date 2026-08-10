import os
import shutil
import tempfile
import unittest
from unittest import mock

import librosa
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine

import bass_isolator
import instrument_isolator


def _notes(note_names, note_ms=600, gap_ms=200):
    """A short sequence of held, distinctly-pitched tones with silence
    between them, standing in for a real isolated bass stem."""
    track = AudioSegment.silent(duration=100)
    for note_name in note_names:
        freq = librosa.note_to_hz(note_name)
        tone = Sine(freq).to_audio_segment(duration=note_ms).fade_in(10).fade_out(30)
        track += tone + AudioSegment.silent(duration=gap_ms)
    return track


def _noisy_tone(note_name, duration_ms=2000, noise_amplitude=0.03, sr=44100):
    """A single held tone with a little broadband noise mixed in, standing
    in for the pitch-estimation jitter a real (imperfectly isolated) bass
    stem has even during one sustained note."""
    freq = librosa.note_to_hz(note_name)
    n = int(sr * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n, endpoint=False)
    tone = np.sin(2 * np.pi * freq * t)
    rng = np.random.default_rng(0)
    noisy = tone + rng.normal(scale=noise_amplitude, size=n)
    samples = np.clip(noisy * 32767 * 0.5, -32768, 32767).astype(np.int16)
    return AudioSegment(samples.tobytes(), sample_width=2, frame_rate=sr, channels=1)


def _chirp(start_note, end_note, duration_ms=1200, sr=44100):
    """A smooth linear pitch glide from start_note to end_note, standing in
    for a real bass slide/bend."""
    f_start = librosa.note_to_hz(start_note)
    f_end = librosa.note_to_hz(end_note)
    n = int(sr * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n, endpoint=False)
    freq_t = f_start + (f_end - f_start) * (t / t[-1])
    phase = 2 * np.pi * np.cumsum(freq_t) / sr
    tone = np.sin(phase)
    samples = (tone * 32767 * 0.5).astype(np.int16)
    return AudioSegment(samples.tobytes(), sample_width=2, frame_rate=sr, channels=1)


class TestIsolateBass(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.mp3_path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        with open(self.mp3_path, "wb") as f:
            f.write(b"fake mp3 bytes")
        # Everything for a song lives in the mp3's own folder now.
        self.song_dir = self.tmp_dir

    def tearDown(self):
        instrument_isolator.clear_stem_cache()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _fake_run_demucs(self, input_path, out_dir, model_name, two_stems=None, on_percent=None, should_cancel=None):
        track_name = os.path.splitext(os.path.basename(input_path))[0]
        stem_dir = os.path.join(out_dir, model_name, track_name)
        os.makedirs(stem_dir, exist_ok=True)
        with open(os.path.join(stem_dir, "bass.wav"), "wb") as f:
            f.write(b"bass")
        return stem_dir

    def _fake_trim_and_export(self, wav_path, trim_ms, out_path):
        shutil.copy(wav_path, out_path)

    @mock.patch("bass_isolator._apply_noise_gate")
    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_produces_a_bass_wav(self, mock_alignment, mock_run_demucs, mock_trim, mock_gate):
        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export

        result = bass_isolator.isolate_bass(self.mp3_path)

        self.assertTrue(result)
        basename = "Song - Artist (Isolated Bass at 120.000 BPM)"
        self.assertTrue(os.path.exists(os.path.join(self.song_dir, basename + ".wav")))
        mock_gate.assert_called_once_with(os.path.join(self.song_dir, basename + ".wav"))
        self.assertTrue(os.path.exists(os.path.join(self.song_dir, bass_isolator._SOURCE_MARKER_FILENAME)))

    @mock.patch("instrument_isolator.run_demucs")
    def test_skips_when_outputs_already_exist_and_match_the_source_mp3(self, mock_run_demucs):
        basename = "Song - Artist (Isolated Bass at 120.000 BPM)"
        with open(os.path.join(self.song_dir, basename + ".wav"), "wb") as f:
            f.write(b"x")
        instrument_isolator.write_source_marker(self.song_dir, self.mp3_path, bass_isolator._SOURCE_MARKER_FILENAME)

        result = bass_isolator.isolate_bass(self.mp3_path)

        self.assertFalse(result)
        mock_run_demucs.assert_not_called()

    @mock.patch("bass_isolator._apply_noise_gate")
    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_reruns_when_a_wav_is_there_but_no_marker_confirms_it(
        self, mock_alignment, mock_run_demucs, mock_trim, mock_gate
    ):
        basename = "Song - Artist (Isolated Bass at 120.000 BPM)"
        with open(os.path.join(self.song_dir, basename + ".wav"), "wb") as f:
            f.write(b"x")
        # No marker, so this wav can't be trusted as coming from
        # self.mp3_path and is reprocessed rather than accepted.

        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export

        result = bass_isolator.isolate_bass(self.mp3_path)

        self.assertTrue(result)
        mock_run_demucs.assert_called_once()
        self.assertTrue(os.path.exists(os.path.join(self.song_dir, basename + ".wav")))


class TestApplyNoiseGate(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_quiet_noise_lead_in_is_silenced(self):
        quiet_noise = _noisy_tone("A1", duration_ms=500, noise_amplitude=0.002).apply_gain(-40)
        loud_note = _notes(["A1"], note_ms=1000, gap_ms=0)
        track = quiet_noise + loud_note
        wav_path = os.path.join(self.tmp_dir, "bass.wav")
        track.export(wav_path, format="wav")

        bass_isolator._apply_noise_gate(wav_path)

        gated = AudioSegment.from_wav(wav_path)
        quiet_part = gated[:400]
        loud_part = gated[600:1400]
        self.assertLess(quiet_part.rms, loud_part.rms / 10)

    def test_a_normal_note_is_left_essentially_untouched(self):
        note = _notes(["A1"], note_ms=1000, gap_ms=0)
        wav_path = os.path.join(self.tmp_dir, "bass.wav")
        note.export(wav_path, format="wav")
        original_rms = AudioSegment.from_wav(wav_path).rms

        bass_isolator._apply_noise_gate(wav_path)

        gated_rms = AudioSegment.from_wav(wav_path).rms
        self.assertGreater(gated_rms, original_rms * 0.9)

    def test_silent_file_does_not_raise(self):
        wav_path = os.path.join(self.tmp_dir, "silence.wav")
        AudioSegment.silent(duration=500).export(wav_path, format="wav")

        bass_isolator._apply_noise_gate(wav_path)  # should not raise

        self.assertTrue(os.path.exists(wav_path))


class TestIsolateBassForFolder(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_no_mp3s_prints_message_and_does_nothing(self):
        with mock.patch("bass_isolator.isolate_bass") as mock_isolate:
            bass_isolator.isolate_bass_for_folder(self.tmp_dir)
        mock_isolate.assert_not_called()

    @mock.patch("bass_isolator.isolate_bass")
    def test_processes_each_mp3_in_folder(self, mock_isolate):
        for name in ("A - Artist.mp3", "B - Artist.mp3"):
            with open(os.path.join(self.tmp_dir, name), "wb") as f:
                f.write(b"x")

        bass_isolator.isolate_bass_for_folder(self.tmp_dir)

        self.assertEqual(mock_isolate.call_count, 2)
        called_paths = {call.args[0] for call in mock_isolate.call_args_list}
        self.assertEqual(
            called_paths,
            {os.path.join(self.tmp_dir, "A - Artist.mp3"), os.path.join(self.tmp_dir, "B - Artist.mp3")},
        )

    @mock.patch("bass_isolator.isolate_bass")
    def test_one_failure_does_not_stop_the_rest(self, mock_isolate):
        for name in ("A - Artist.mp3", "B - Artist.mp3"):
            with open(os.path.join(self.tmp_dir, name), "wb") as f:
                f.write(b"x")
        mock_isolate.side_effect = [RuntimeError("boom"), True]

        bass_isolator.isolate_bass_for_folder(self.tmp_dir)

        self.assertEqual(mock_isolate.call_count, 2)


class TestIsolateBassForPath(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @mock.patch("bass_isolator.isolate_bass")
    def test_single_file_dispatches_directly(self, mock_isolate):
        mp3_path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        with open(mp3_path, "wb") as f:
            f.write(b"x")

        bass_isolator.isolate_bass_for_path(mp3_path)

        mock_isolate.assert_called_once_with(mp3_path, context=None)

    @mock.patch("bass_isolator.isolate_bass_for_folder")
    def test_folder_dispatches_to_folder_handler(self, mock_folder):
        bass_isolator.isolate_bass_for_path(self.tmp_dir)
        mock_folder.assert_called_once_with(self.tmp_dir, context=None)


if __name__ == "__main__":
    unittest.main()
