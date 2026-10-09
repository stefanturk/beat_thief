import json
import os
import shutil
import tempfile
import unittest

import song_queue


class TestSongQueue(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.path = os.path.join(self.tmp, "queue.json")
        self.queue = song_queue.SongQueue(self.path)

    def _add(self, url, label=None):
        return self.queue.add({"url": url, "options": {"drums": True}, "label": label or url})

    def test_songs_come_out_in_the_order_they_went_in(self):
        for n in range(3):
            self._add(f"https://youtu.be/{n}")
        self.assertEqual([self.queue.pop_next()["url"] for _ in range(3)],
                         [f"https://youtu.be/{n}" for n in range(3)])
        self.assertIsNone(self.queue.pop_next())

    def test_each_entry_gets_its_own_id_and_keeps_its_options(self):
        first, second = self._add("a"), self._add("b")
        self.assertNotEqual(first, second)
        self.assertEqual(self.queue.entries()[0]["options"], {"drums": True})

    def test_removing_one_leaves_the_rest_in_order(self):
        ids = [self._add(u) for u in "abc"]
        self.assertTrue(self.queue.remove(ids[1]))
        self.assertEqual([e["url"] for e in self.queue.entries()], ["a", "c"])
        self.assertFalse(self.queue.remove("nope"))

    def test_moving_up_and_down_stops_at_the_ends(self):
        ids = [self._add(u) for u in "abc"]
        self.queue.move(ids[2], -1)
        self.assertEqual([e["url"] for e in self.queue.entries()], ["a", "c", "b"])
        self.queue.move(ids[0], -1)
        self.assertEqual([e["url"] for e in self.queue.entries()], ["a", "c", "b"])
        self.queue.move(ids[1], 1)
        self.assertEqual([e["url"] for e in self.queue.entries()], ["a", "c", "b"])
        self.queue.move(ids[0], 1)
        self.assertEqual([e["url"] for e in self.queue.entries()], ["c", "a", "b"])

    def test_clearing_empties_it(self):
        self._add("a")
        self._add("b")
        self.queue.clear()
        self.assertEqual(self.queue.entries(), [])
        self.assertIsNone(self.queue.next())

    def test_up_next_and_last_added(self):
        self.assertIsNone(self.queue.next())
        self.assertIsNone(self.queue.last_added())
        self._add("a")
        self._add("b")
        self.assertEqual(self.queue.next()["url"], "a")
        self.assertEqual(self.queue.last_added()["url"], "b")

    def test_last_added_is_the_newest_even_after_reordering(self):
        ids = [self._add(u) for u in "abc"]
        self.queue.move(ids[2], -2)
        self.assertEqual(self.queue.last_added()["url"], "c")

    def test_it_comes_back_after_the_app_quits(self):
        self._add("a")
        self._add("b")
        again = song_queue.SongQueue(self.path)
        self.assertEqual([e["url"] for e in again.entries()], ["a", "b"])

    def test_the_song_in_hand_comes_back_first_after_a_crash(self):
        self._add("a")
        self._add("b")
        self.queue.pop_next()          # "a" is being worked on...
        again = song_queue.SongQueue(self.path)   # ...when the power goes
        self.assertEqual([e["url"] for e in again.entries()], ["a", "b"])

    def test_a_finished_song_does_not_come_back(self):
        self._add("a")
        self._add("b")
        self.queue.pop_next()
        self.queue.finish_current()
        again = song_queue.SongQueue(self.path)
        self.assertEqual([e["url"] for e in again.entries()], ["b"])

    def test_the_song_in_hand_can_be_put_back_at_the_front(self):
        self._add("a")
        self._add("b")
        self.queue.pop_next()
        self.queue.put_back_current()
        self.assertEqual([e["url"] for e in self.queue.entries()], ["a", "b"])

    def test_a_label_can_be_filled_in_later(self):
        entry_id = self._add("a")
        self.queue.relabel(entry_id, "Hey Ya!")
        self.assertEqual(self.queue.entries()[0]["label"], "Hey Ya!")

    def test_a_broken_file_is_an_empty_queue(self):
        with open(self.path, "w") as f:
            f.write("{nope")
        self.assertEqual(song_queue.SongQueue(self.path).entries(), [])

    def test_the_file_is_plain_json(self):
        self._add("a")
        with open(self.path) as f:
            saved = json.load(f)
        self.assertEqual(saved["entries"][0]["url"], "a")


class TestShortLabel(unittest.TestCase):
    def test_a_youtube_link_is_its_video_id(self):
        self.assertEqual(song_queue.short_label("https://www.youtube.com/watch?v=abc123&t=4"),
                         "YouTube abc123")
        self.assertEqual(song_queue.short_label("https://youtu.be/xyz"), "YouTube xyz")

    def test_spotify_says_what_kind(self):
        self.assertEqual(song_queue.short_label("https://open.spotify.com/track/1"), "Spotify track")
        self.assertEqual(song_queue.short_label("https://open.spotify.com/playlist/9"), "Spotify playlist")

    def test_pasted_text_is_its_first_line(self):
        self.assertEqual(song_queue.short_label("Hey Ya! - OutKast\nToxic - Britney"),
                         "Hey Ya! - OutKast (and more)")

    def test_a_stash_song_is_its_name(self):
        self.assertEqual(song_queue.short_label("", "/m/Hey Ya!/Hey Ya! (80 BPM).mp3"), "Hey Ya!")


if __name__ == "__main__":
    unittest.main()
