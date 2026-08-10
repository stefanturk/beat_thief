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
