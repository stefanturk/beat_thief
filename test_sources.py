import csv
import json
import os
import shutil
import tempfile
import unittest

import sources


class SourcesTestCase(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder, True)


class TestTheRecord(SourcesTestCase):
    def test_a_folder_with_no_record_has_nothing_in_it(self):
        self.assertEqual(sources.Sources(self.folder).songs, {})

    def test_what_is_recorded_is_there_next_time(self):
        record = sources.Sources(self.folder)
        record.record("spotify:abc", title="One", artist="Someone", number=1)
        self.assertEqual(sources.Sources(self.folder).get("spotify:abc")["title"], "One")

    def test_a_song_is_finished_only_while_its_file_is_there(self):
        path = os.path.join(self.folder, "One - Someone.mp3")
        with open(path, "wb") as f:
            f.write(b"mp3")
        record = sources.Sources(self.folder)
        record.record("spotify:abc", status="done", file=path)
        self.assertEqual(record.get("spotify:abc")["file"], "One - Someone.mp3")
        self.assertEqual(sources.Sources(self.folder).finished("spotify:abc"), path)
        os.remove(path)
        self.assertEqual(sources.Sources(self.folder).finished("spotify:abc"), "")

    def test_a_failed_song_is_not_finished(self):
        record = sources.Sources(self.folder)
        record.record("spotify:abc", status="failed", file="x.mp3")
        self.assertEqual(record.finished("spotify:abc"), "")

    def test_a_damaged_record_is_started_again_rather_than_crashing(self):
        with open(os.path.join(self.folder, sources.MANIFEST_FILENAME), "w") as f:
            f.write("{not json")
        self.assertEqual(sources.Sources(self.folder).songs, {})

    def test_the_record_is_hidden_from_finder(self):
        self.assertTrue(sources.MANIFEST_FILENAME.startswith("."))


class TestTheCsv(SourcesTestCase):
    def _rows(self):
        with open(os.path.join(self.folder, sources.CSV_FILENAME), encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def test_columns_are_the_ones_promised(self):
        sources.Sources(self.folder).record("k", title="One")
        with open(os.path.join(self.folder, sources.CSV_FILENAME), encoding="utf-8-sig") as f:
            self.assertEqual(next(csv.reader(f)), sources.CSV_COLUMNS)

    def test_rows_are_in_playlist_order(self):
        record = sources.Sources(self.folder)
        record.record("b", title="Two", number=2)
        record.record("a", title="One", number=1)
        record.flush()
        self.assertEqual([r["Song"] for r in self._rows()], ["One", "Two"])

    def test_the_bpm_comes_from_the_file_name(self):
        sources.Sources(self.folder).record("a", title="One", file="One - X (104.5 BPM).mp3")
        self.assertEqual(self._rows()[0]["BPM"], "104.5")

    def test_a_guess_says_to_check_it(self):
        sources.Sources(self.folder).record("a", title="One", check=True)
        self.assertIn("Check", self._rows()[0]["Note"])


class TestKeys(unittest.TestCase):
    def test_youtube_links_of_every_shape_agree(self):
        for url in ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=x",
                    "https://youtu.be/dQw4w9WgXcQ",
                    "https://music.youtube.com/watch?v=dQw4w9WgXcQ"):
            self.assertEqual(sources.youtube_key(url), "youtube:dQw4w9WgXcQ")


class TestWhichFolderAPasteBelongsIn(SourcesTestCase):
    def _folder_with(self, name, keys):
        folder = os.path.join(self.folder, name)
        os.makedirs(folder)
        record = sources.Sources(folder)
        for key in keys:
            record.record(key, title=key)
        return folder

    def test_the_folder_holding_the_start_of_a_longer_list(self):
        misco = self._folder_with("Misco", [f"k{i}" for i in range(100)])
        self._folder_with("Other", ["z1", "z2"])
        self.assertEqual(sources.folder_holding(self.folder, [f"k{i}" for i in range(1200)]), misco)

    def test_nothing_much_in_common_is_no_folder(self):
        self._folder_with("Misco", [f"k{i}" for i in range(100)])
        self.assertIsNone(sources.folder_holding(self.folder, ["k1"] + [f"n{i}" for i in range(50)]))


if __name__ == "__main__":
    unittest.main()


class TestALongPlaylistIsCheapToWriteDown(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder, True)

    def _csv_rows(self):
        with open(os.path.join(self.folder, sources.CSV_FILENAME), encoding="utf-8-sig") as f:
            return len(f.read().splitlines()) - 1

    def test_the_csv_is_written_every_so_often_and_at_the_end(self):
        record = sources.Sources(self.folder)
        record.record("a", title="First")
        self.assertEqual(self._csv_rows(), 1)       # there from the first song
        for n in range(sources.CSV_EVERY - 1):
            record.record(f"k{n}", title=str(n))
        self.assertEqual(self._csv_rows(), 1)       # not rewritten every time
        record.record("due", title="due")           # the CSV_EVERY-th change since
        self.assertEqual(self._csv_rows(), sources.CSV_EVERY + 1)
        record.record("late", title="late")
        record.flush()
        self.assertEqual(self._csv_rows(), sources.CSV_EVERY + 2)

    def test_a_folder_that_has_gone_is_not_made_again(self):
        record = sources.Sources(self.folder)
        record.record("a", title="First")
        shutil.rmtree(self.folder)
        record.record("b", title="Second")
        record.flush()
        self.assertFalse(os.path.exists(self.folder))

    def test_a_change_kept_for_later_is_written_with_the_next(self):
        record = sources.Sources(self.folder)
        record.record("a", save=False, title="Kept")
        self.assertEqual(sources.Sources(self.folder).songs, {})
        record.record("a", status="done")
        self.assertEqual(sources.Sources(self.folder).get("a"), {"title": "Kept", "status": "done"})

    def test_writing_down_song_5000_is_quick(self):
        import time
        record = sources.Sources(self.folder)
        for n in range(5000):
            record.songs[f"spotify:{n}"] = {"title": f"Song {n}", "artist": "Someone", "status": "done",
                                            "file": f"Song {n} - Someone (120 BPM).mp3", "number": n,
                                            "youtube_url": "https://www.youtube.com/watch?v=x" * 1}
        record._unwritten = 0
        started = time.monotonic()
        record.record("spotify:5000", title="Last", status="done")
        self.assertLess(time.monotonic() - started, 0.1)
