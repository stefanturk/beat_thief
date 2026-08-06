import unittest
from unittest import mock

import beat_thief


class CliTestCase(unittest.TestCase):
    """What the terminal front end makes of its arguments. pipeline.run is
    stubbed out - what's under test is the request it gets handed, not the
    work."""

    def _instruments_for(self, argv):
        captured = {}

        def fake_run(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return {}

        with mock.patch("sys.argv", ["beat_thief.py"] + argv), \
             mock.patch("pipeline.run", side_effect=fake_run), \
             mock.patch("beat_thief._exit"):
            beat_thief.main()
        return captured


class TestChoosingInstruments(CliTestCase):
    URL = "https://youtu.be/abc"

    def test_no_instrument_named_means_just_the_song(self):
        captured = self._instruments_for([self.URL])

        self.assertEqual(captured["instruments"], [])
        self.assertEqual(captured["url"], self.URL)

    def test_a_bare_instrument_name_is_understood(self):
        captured = self._instruments_for([self.URL, "drums"])

        self.assertEqual(captured["instruments"], ["drums"])

    def test_vocals_can_be_asked_for(self):
        captured = self._instruments_for([self.URL, "vocals"])

        self.assertEqual(captured["instruments"], ["vocals"])

    def test_all_means_every_stem(self):
        captured = self._instruments_for([self.URL, "all"])

        self.assertEqual(captured["instruments"], ["drums", "bass", "harmony", "vocals"])

    def test_all_works_before_the_url_too(self):
        captured = self._instruments_for(["all", self.URL])

        self.assertEqual(captured["instruments"], ["drums", "bass", "harmony", "vocals"])
        self.assertEqual(captured["url"], self.URL)

    def test_all_as_a_dashed_flag_means_the_same_thing(self):
        captured = self._instruments_for([self.URL, "--all"])

        self.assertEqual(captured["instruments"], ["drums", "bass", "harmony", "vocals"])

    def test_named_instruments_still_combine(self):
        captured = self._instruments_for([self.URL, "drums", "vocals"])

        self.assertEqual(captured["instruments"], ["drums", "vocals"])


class TestChoosingMidi(CliTestCase):
    URL = "https://youtu.be/abc"

    def test_midi_is_off_unless_asked_for(self):
        captured = self._instruments_for([self.URL, "all"])

        self.assertEqual(captured["midi_for"], frozenset())

    def test_bare_midi_covers_everything_asked_for_that_has_it(self):
        captured = self._instruments_for([self.URL, "all", "midi"])

        self.assertEqual(captured["instruments"], ["drums", "bass", "harmony", "vocals"])
        self.assertEqual(captured["midi_for"], frozenset({"drums", "bass"}))

    def test_bare_midi_does_not_reach_instruments_that_were_not_asked_for(self):
        captured = self._instruments_for([self.URL, "drums", "midi"])

        self.assertEqual(captured["midi_for"], frozenset({"drums"}))

    def test_midi_can_name_one_instrument(self):
        captured = self._instruments_for([self.URL, "drums", "bass", "--midi", "drums"])

        self.assertEqual(captured["instruments"], ["drums", "bass"])
        self.assertEqual(captured["midi_for"], frozenset({"drums"}))

    def test_an_instrument_named_after_midi_is_not_also_isolated(self):
        # "--midi drums" is a MIDI choice, not a request for the drums.
        captured = self._instruments_for([self.URL, "bass", "--midi", "drums"])

        self.assertEqual(captured["instruments"], ["bass"])
        self.assertEqual(captured["midi_for"], frozenset({"drums"}))

    def test_a_url_after_midi_is_still_the_url(self):
        captured = self._instruments_for(["--midi", self.URL, "drums"])

        self.assertEqual(captured["url"], self.URL)
        self.assertEqual(captured["instruments"], ["drums"])
        self.assertEqual(captured["midi_for"], frozenset({"drums"}))

    def test_an_output_folder_named_after_a_stem_is_a_folder(self):
        captured = self._instruments_for([self.URL, "-o", "drums"])

        self.assertEqual(captured["output_dir"], "drums")
        self.assertEqual(captured["instruments"], [])

    def test_naming_an_instrument_with_no_midi_step_is_an_error(self):
        with self.assertRaises(SystemExit):
            self._instruments_for([self.URL, "vocals", "--midi", "vocals"])


if __name__ == "__main__":
    unittest.main()
