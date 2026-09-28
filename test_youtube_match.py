import unittest
from unittest import mock

import youtube_match


def _candidate(title, duration, want=213.0, url=None, channel="A Channel"):
    offset = None if duration is None else abs(duration - want)
    return {
        "url": url or ("https://www.youtube.com/watch?v=" + title[:11].replace(" ", "_")),
        "title": title,
        "channel": channel,
        "duration": duration,
        "offset": offset,
    }


class TestPicking(unittest.TestCase):
    """pick()'s job isn't to be right - it's to know when it can't be sure.
    Taking a wrong recording silently is the expensive failure here: a
    remaster or a live cut has its own tempo and its own beat 1, and every
    grid made downstream inherits them."""

    def test_one_obvious_match_is_taken_without_asking(self):
        right = _candidate("Rick Astley - Never Gonna Give You Up (Official Video)",
                           213.4, url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        url, asking = youtube_match.pick([right, _candidate("Some other song", 300.0)])
        self.assertFalse(asking)
        self.assertEqual(url, right["url"])

    def test_copies_of_the_same_recording_are_not_a_question(self):
        """The official video, the Topic upload and a lyric video come back
        together for any famous song. They are the same audio to the second,
        so there is nothing to decide - take the closest and get on with it."""
        best = _candidate("Never Gonna Give You Up", 213.4,
                          url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        url, asking = youtube_match.pick([
            best,
            _candidate("Never Gonna Give You Up (Official Video)", 213.6),
            _candidate("Rick Astley - Never Gonna Give You Up", 213.8),
        ])
        self.assertFalse(asking)
        self.assertEqual(url, best["url"])

    def test_a_remaster_wins_over_an_ordinary_upload_of_the_same_length(self):
        """Both are the same performance; the remaster is the better audio."""
        remaster = _candidate("Never Gonna Give You Up (2022 Remaster)", 213.4,
                              url="https://www.youtube.com/watch?v=remastered1")
        url, asking = youtube_match.pick([
            _candidate("Never Gonna Give You Up (Official Video)", 213.2),
            remaster,
        ])
        self.assertFalse(asking)
        self.assertEqual(url, remaster["url"])

    def test_a_remastered_live_take_is_not_rescued_by_the_word_remaster(self):
        """The mastering isn't what makes a live take the wrong recording."""
        _, asking = youtube_match.pick([
            _candidate("Never Gonna Give You Up (Live 1988 - Remastered)", 213.3),
        ])
        self.assertTrue(asking)

    def test_two_near_misses_and_nothing_exact_is_put_to_somebody(self):
        """Nothing here is clearly the recording asked for - both are a couple
        of seconds out, which is where a different arrangement starts."""
        url, asking = youtube_match.pick([
            _candidate("Never Gonna Give You Up", 210.5),
            _candidate("Never Gonna Give You Up (Official Video)", 215.8),
        ])
        self.assertTrue(asking)
        # Still hands back the best guess, for a caller with nobody to ask.
        self.assertIsNotNone(url)

    def test_a_remaster_at_the_same_length_is_the_one_to_take(self):
        """Same players, same room, same beat 1 - only the mastering differs,
        and it's usually the master Spotify is serving anyway."""
        url, asking = youtube_match.pick([
            _candidate("Never Gonna Give You Up (2022 Remaster)", 213.4),
        ])
        self.assertFalse(asking)
        self.assertIsNotNone(url)

    def test_a_remaster_of_some_other_length_is_still_a_question(self):
        """Three seconds out is a different edit, whatever the title says."""
        _, asking = youtube_match.pick([
            _candidate("Never Gonna Give You Up (2022 Remaster)", 216.0),
        ])
        self.assertTrue(asking)

    def test_other_versions_that_also_have_to_be_asked_about(self):
        for title in ("Song (Live at Wembley)", "Song - Remix", "Song (sped up)",
                      "Song [slowed + reverb]", "Song - Karaoke Version",
                      "Song (8D Audio)", "Song - Piano Cover", "Song (Instrumental)"):
            with self.subTest(title=title):
                _, asking = youtube_match.pick([_candidate(title, 213.2)])
                self.assertTrue(asking, f"{title} should not be taken silently")

    def test_which_titles_read_as_a_remaster(self):
        for title, expected in (("Song - 2011 Remastered", True),
                                ("Song (Remaster)", True),
                                ("Song - 2019 Re-Master", True),
                                ("Remasters - The Best Of", True),
                                ("Song (Official Video)", False)):
            with self.subTest(title=title):
                self.assertEqual(youtube_match.looks_remastered(title), expected)

    def test_a_song_whose_own_title_contains_one_of_the_words(self):
        """"Deliver" is not "live" and "Discover" is not "cover". Whole words
        only, because every title disqualified by accident is a question put
        to somebody who had nothing to decide. A title whose own words really
        are "Live" or "Cover" still gets asked about - that way round is only
        a wasted question, not a wrong recording."""
        for title, expected in (("Deliver Me", False), ("Discover", False),
                                ("Remixology", False), ("Song (Live)", True),
                                ("Song - 2011 Remastered", False),
                                ("Song (Piano Covers)", True)):
            with self.subTest(title=title):
                self.assertEqual(
                    youtube_match.looks_like_another_version(title), expected)

    def test_nothing_close_enough_is_asked_about_rather_than_refused(self):
        """A duration can simply be wrong - a legitimate upload with a long
        silent tail shouldn't be a dead end."""
        url, asking = youtube_match.pick([_candidate("Never Gonna Give You Up", 260.0)])
        self.assertTrue(asking)
        self.assertIsNotNone(url)

    def test_no_results_at_all(self):
        url, asking = youtube_match.pick([])
        self.assertIsNone(url)
        self.assertTrue(asking)

    def test_a_result_with_no_duration_is_not_a_confident_match(self):
        _, asking = youtube_match.pick([_candidate("Never Gonna Give You Up", None)])
        self.assertTrue(asking)


class TestWhatGetsOffered(unittest.TestCase):
    def test_everything_within_tolerance_plus_a_couple_of_outsiders(self):
        found = [
            _candidate("A", 213.1), _candidate("B", 214.0), _candidate("C", 216.0),
            _candidate("D", 240.0), _candidate("E", 300.0), _candidate("F", 400.0),
        ]
        offered = youtube_match.worth_offering(found)
        titles = [c["title"] for c in offered]
        self.assertEqual(titles[:3], ["A", "B", "C"])
        # The near misses stay reachable; the hopeless ones don't clutter.
        self.assertIn("D", titles)
        self.assertNotIn("F", titles)


class _FakeSearch:
    """Stands in for yt_dlp.YoutubeDL doing a ytsearch, with no network."""

    entries = []
    queries = []

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        _FakeSearch.queries.append(url)
        return {"entries": list(self.entries)}


class TestSearching(unittest.TestCase):
    def setUp(self):
        _FakeSearch.queries = []
        patcher = mock.patch("yt_dlp.YoutubeDL", _FakeSearch)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_results_come_back_nearest_the_wanted_length_first(self):
        _FakeSearch.entries = [
            {"id": "aaaaaaaaaaa", "title": "Far", "duration": 400, "uploader": "X"},
            {"id": "bbbbbbbbbbb", "title": "Near", "duration": 214, "uploader": "Y"},
        ]
        found = youtube_match.candidates("Rick Astley Never Gonna Give You Up", 213.0)
        self.assertEqual([c["title"] for c in found], ["Near", "Far"])
        self.assertAlmostEqual(found[0]["offset"], 1.0, places=3)

    def test_it_searches_for_what_it_was_asked_for(self):
        _FakeSearch.entries = []
        youtube_match.candidates("Rick Astley Never Gonna Give You Up", 213.0)
        self.assertEqual(_FakeSearch.queries,
                         ["ytsearch5:Rick Astley Never Gonna Give You Up"])

    def test_a_result_with_no_duration_is_kept_but_sorts_last(self):
        """It's still a real result - just one nothing is known about."""
        _FakeSearch.entries = [
            {"id": "aaaaaaaaaaa", "title": "Unknown length", "uploader": "X"},
            {"id": "bbbbbbbbbbb", "title": "Known", "duration": 214, "uploader": "Y"},
        ]
        found = youtube_match.candidates("q", 213.0)
        self.assertEqual([c["title"] for c in found], ["Known", "Unknown length"])
        self.assertIsNone(found[1]["offset"])

    def test_a_search_that_blows_up_is_no_results_rather_than_a_crash(self):
        with mock.patch("yt_dlp.YoutubeDL", side_effect=RuntimeError("boom")):
            self.assertEqual(youtube_match.candidates("q", 213.0), [])

    def test_nothing_is_downloaded_while_searching(self):
        _FakeSearch.entries = []
        youtube_match.candidates("q", 213.0)
        self.assertTrue(youtube_match._search_opts()["skip_download"])


class _FakeMusic:
    """yt_dlp.YoutubeDL for a Music search and then the one lookup that
    follows it. `found` is what the lookup says about the top result."""

    top_id = "BuzJ5NArvgw"
    found = {}
    calls = []

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False, process=True):
        _FakeMusic.calls.append(url)
        if "music.youtube.com/search" in url:
            return {"entries": [{"id": self.top_id, "title": "whatever"}] if self.top_id else []}
        return dict(self.found)


class TestYoutubeMusicFirst(unittest.TestCase):
    """The Music "songs" search names the release itself. Trusted only when
    length, artist and title all agree - anything less is None, and the
    ordinary search and pick() carry on exactly as before."""

    def setUp(self):
        _FakeMusic.calls = []
        _FakeMusic.top_id = "BuzJ5NArvgw"
        _FakeMusic.found = {"duration": 188, "title": "Them Changes",
                            "artists": ["Thundercat"], "channel": "Thundercat"}
        patcher = mock.patch("yt_dlp.YoutubeDL", _FakeMusic)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _match(self, want=188.0, artist="Thundercat"):
        return youtube_match.music_match("Thundercat Them Changes", want, artist)

    def test_the_release_at_the_right_length_by_the_right_artist_is_taken(self):
        hit = self._match()
        self.assertEqual(hit["url"], "https://www.youtube.com/watch?v=BuzJ5NArvgw")
        self.assertEqual(hit["offset"], 0.0)
        self.assertIn("music.youtube.com/search", _FakeMusic.calls[0])
        self.assertIn("#songs", _FakeMusic.calls[0])

    def test_the_wrong_length_is_left_to_the_ordinary_search(self):
        self.assertIsNone(self._match(want=214.0))

    def test_somebody_elses_recording_is_left_to_the_ordinary_search(self):
        _FakeMusic.found = dict(_FakeMusic.found, artists=["Cover Band"], channel="Cover Band")
        self.assertIsNone(self._match())

    def test_another_version_is_left_to_the_ordinary_search(self):
        _FakeMusic.found = dict(_FakeMusic.found, title="Them Changes (Sped Up)")
        self.assertIsNone(self._match())

    def test_no_result_or_a_failure_is_none_not_a_crash(self):
        _FakeMusic.top_id = None
        self.assertIsNone(self._match())
        with mock.patch("yt_dlp.YoutubeDL", side_effect=RuntimeError("boom")):
            self.assertIsNone(self._match())

    def test_artists_are_compared_ignoring_punctuation_and_topic(self):
        self.assertTrue(youtube_match.same_artist("Earth, Wind & Fire", ["Earth Wind & Fire - Topic"]))
        self.assertTrue(youtube_match.same_artist("Rufus, Chaka Khan", ["Chaka Khan"]))
        self.assertTrue(youtube_match.same_artist("Thundercat", ["Thundercat, Kendrick Lamar"]))
        self.assertFalse(youtube_match.same_artist("Thundercat", ["Vulfpeck"]))
        self.assertFalse(youtube_match.same_artist("", ["Anyone"]))


if __name__ == "__main__":
    unittest.main()
