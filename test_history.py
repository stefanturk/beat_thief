import os
import shutil
import tempfile
import unittest

import history


class HistoryTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp_dir, "history.json")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _song(self, name="Song - Artist.mp3"):
        song = os.path.join(self.tmp_dir, name)
        with open(song, "wb") as f:
            f.write(b"mp3")
        return song


class TestRemember(HistoryTestCase):
    def test_a_song_can_be_traced_back_to_its_link(self):
        song = self._song()

        history.remember("https://youtu.be/abc", [song], self.path)

        self.assertEqual(history.url_for(song, self.path), "https://youtu.be/abc")

    def test_an_unknown_song_has_no_link_rather_than_raising(self):
        # Songs downloaded before this existed, or dropped in by hand.
        self.assertEqual(history.url_for("/nope/Song.mp3", self.path), "")

    def test_re_downloading_updates_the_song_instead_of_duplicating_it(self):
        song = self._song()
        history.remember("https://youtu.be/old", [song], self.path)

        history.remember("https://youtu.be/new", [song], self.path)

        entries = history.entries(self.path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["url"], "https://youtu.be/new")

    def test_newest_song_comes_first(self):
        first = self._song("First - Artist.mp3")
        second = self._song("Second - Artist.mp3")
        history.remember("https://youtu.be/1", [first], self.path)
        history.remember("https://youtu.be/2", [second], self.path)

        titles = [os.path.basename(e["song"]) for e in history.entries(self.path)]

        self.assertEqual(titles, ["Second - Artist.mp3", "First - Artist.mp3"])

    def test_a_run_with_no_link_or_no_songs_records_nothing(self):
        history.remember("", [self._song()], self.path)
        history.remember("https://youtu.be/abc", [], self.path)

        self.assertEqual(history.entries(self.path), [])

    def test_a_deleted_song_drops_out_of_the_listing(self):
        song = self._song()
        history.remember("https://youtu.be/abc", [song], self.path)
        os.remove(song)

        self.assertEqual(history.entries(self.path), [])

    def test_a_corrupt_history_file_is_treated_as_empty(self):
        with open(self.path, "w") as f:
            f.write("{not json at all")

        self.assertEqual(history.entries(self.path), [])
        history.remember("https://youtu.be/abc", [self._song()], self.path)  # still writable
        self.assertEqual(len(history.entries(self.path)), 1)

    def test_an_unwritable_location_never_breaks_a_finished_run(self):
        # The files someone asked for already exist by this point; failing to
        # jot down the link must not undo that.
        history.remember("https://youtu.be/abc", [self._song()], "/nope/nowhere/history.json")


if __name__ == "__main__":
    unittest.main()
