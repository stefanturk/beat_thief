import os
import unittest
from unittest import mock

import spotify

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "fixtures")


def _fixture(name: str) -> str:
    with open(os.path.join(FIXTURES, name)) as f:
        return f.read()


class TestRecognisingALink(unittest.TestCase):
    """Whether this module gets involved at all comes down to spotify_id, so
    every shape a link arrives in has to be one it knows."""

    def test_an_ordinary_track_link(self):
        self.assertEqual(
            spotify.spotify_id("https://open.spotify.com/track/4PTG3Z6ehGkBFwjybzWkR8"),
            ("track", "4PTG3Z6ehGkBFwjybzWkR8"),
        )

    def test_the_share_button_adds_a_tracking_parameter(self):
        self.assertEqual(
            spotify.spotify_id("https://open.spotify.com/track/4PTG3Z6ehGkBFwjybzWkR8?si=9f2"),
            ("track", "4PTG3Z6ehGkBFwjybzWkR8"),
        )

    def test_a_localized_link(self):
        """What the app hands you outside the US - /intl-de/, /intl-pt-br/."""
        self.assertEqual(
            spotify.spotify_id("https://open.spotify.com/intl-de/track/4PTG3Z6ehGkBFwjybzWkR8"),
            ("track", "4PTG3Z6ehGkBFwjybzWkR8"),
        )

    def test_a_copied_uri(self):
        self.assertEqual(
            spotify.spotify_id("spotify:track:4PTG3Z6ehGkBFwjybzWkR8"),
            ("track", "4PTG3Z6ehGkBFwjybzWkR8"),
        )

    def test_playlists_and_albums(self):
        self.assertEqual(
            spotify.spotify_id("https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"),
            ("playlist", "37i9dQZF1DXcBWIGoYBM5M"),
        )
        self.assertEqual(
            spotify.spotify_id("https://open.spotify.com/album/1ATL5GLyefJaxhQzSPVrLX"),
            ("album", "1ATL5GLyefJaxhQzSPVrLX"),
        )

    def test_a_youtube_link_is_not_ours(self):
        """The case that matters most: returning anything but None here
        would divert every ordinary link into the Spotify path."""
        self.assertIsNone(spotify.spotify_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
        self.assertIsNone(spotify.spotify_id("https://youtu.be/dQw4w9WgXcQ"))
        self.assertIsNone(spotify.spotify_id(""))


class TestReadingATrack(unittest.TestCase):
    """Parsed from a saved copy of the embed page, so the suite never goes
    to the network - and so a Spotify outage isn't a test failure."""

    def test_title_artist_and_duration(self):
        with mock.patch.object(spotify, "_embed_entity",
                               return_value=spotify._entity_from_html(_fixture("spotify_track_embed.html"))):
            song = spotify.track("https://open.spotify.com/track/4PTG3Z6ehGkBFwjybzWkR8")
        self.assertEqual(song["title"], "Never Gonna Give You Up")
        self.assertEqual(song["artist"], "Rick Astley")
        self.assertAlmostEqual(song["duration_sec"], 213.573, places=2)

    def test_the_search_query_leads_with_the_artist(self):
        """Artist first is what pins down which recording of a common title
        is meant - "Hurt" alone finds the wrong one."""
        with mock.patch.object(spotify, "_embed_entity",
                               return_value=spotify._entity_from_html(_fixture("spotify_track_embed.html"))):
            song = spotify.track("https://open.spotify.com/track/4PTG3Z6ehGkBFwjybzWkR8")
        self.assertEqual(song["query"], "Rick Astley Never Gonna Give You Up")

    def test_a_playlist_link_is_not_a_track(self):
        with self.assertRaises(spotify.SpotifyUnavailable):
            spotify.track("https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M")

    def test_something_that_is_not_a_spotify_link(self):
        with self.assertRaises(spotify.SpotifyUnavailable):
            spotify.track("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


class TestReadingAPlaylist(unittest.TestCase):
    def test_every_row_comes_back_in_track_shape(self):
        with mock.patch.object(spotify, "_embed_entity",
                               return_value=spotify._entity_from_html(_fixture("spotify_playlist_embed.html"))):
            songs = spotify.playlist("https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M")
        self.assertEqual(len(songs), 50)
        for song in songs:
            self.assertTrue(song["title"])
            self.assertTrue(song["query"])
            self.assertIsInstance(song["duration_sec"], float)

    def test_a_track_link_is_still_a_list_of_one(self):
        """So a caller working through a playlist doesn't need two paths."""
        with mock.patch.object(spotify, "_embed_entity",
                               return_value=spotify._entity_from_html(_fixture("spotify_track_embed.html"))):
            songs = spotify.playlist("https://open.spotify.com/track/4PTG3Z6ehGkBFwjybzWkR8")
        self.assertEqual(len(songs), 1)
        self.assertEqual(songs[0]["title"], "Never Gonna Give You Up")


class TestReadingAWholeCollection(unittest.TestCase):
    """What a playlist or album is called, as well as what's on it - a
    queue of songs gets filed under the name it arrived with."""

    def test_the_playlist_name_comes_back_with_the_songs(self):
        with mock.patch.object(spotify, "_embed_entity",
                               return_value=spotify._entity_from_html(_fixture("spotify_playlist_embed.html"))):
            found = spotify.collection("https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M")
        self.assertEqual(found["name"], "Today\u2019s Top Hits")
        self.assertEqual(len(found["songs"]), spotify.EMBED_ROW_LIMIT)

    def test_a_single_track_has_no_collection_name(self):
        """One song doesn't need a folder built for it."""
        with mock.patch.object(spotify, "_embed_entity",
                               return_value=spotify._entity_from_html(_fixture("spotify_track_embed.html"))):
            found = spotify.collection("https://open.spotify.com/track/4PTG3Z6ehGkBFwjybzWkR8")
        self.assertEqual(found["name"], "")
        self.assertEqual(len(found["songs"]), 1)

    def test_an_album_is_read_as_a_track_list_too(self):
        """An album is a list of songs with a name. It needs no concept of
        its own, and its embed carries the same trackList."""
        with mock.patch.object(spotify, "_embed_entity",
                               return_value=spotify._entity_from_html(_fixture("spotify_playlist_embed.html"))) as read:
            found = spotify.collection("https://open.spotify.com/album/1ATL5GLyefJaxhQzSPVrLX")
        self.assertEqual(read.call_args.args[0], "album")
        self.assertTrue(found["songs"])


class TestTidyingSpotifysOwnStrings(unittest.TestCase):
    def test_a_non_breaking_space_never_reaches_the_search_box(self):
        """Playlist rows separate collaborators with \xa0. Left alone it
        travels all the way into the YouTube query as a character YouTube
        has no reason to match on."""
        song = spotify._song("Black Heart", "Stealth,\xa0The Dap-Kings", 174700)
        self.assertEqual(song["artist"], "Stealth, The Dap-Kings")
        self.assertNotIn("\xa0", song["query"])

    def test_runs_of_space_collapse_and_the_edges_come_off(self):
        song = spotify._song("  So   Easy ", " Olivia  Dean ", 1000)
        self.assertEqual(song["title"], "So Easy")
        self.assertEqual(song["query"], "Olivia Dean So Easy")


class TestWhenSpotifyChangesTheirPage(unittest.TestCase):
    """The scrape reads an undocumented internal blob. It is allowed to stop
    working; it is not allowed to fail as a stack trace, because the person
    on the other end has to be told what to do about it."""

    def test_a_page_without_the_blob(self):
        with self.assertRaises(spotify.SpotifyUnavailable) as caught:
            spotify._entity_from_html("<html><body>Sorry!</body></html>")
        self.assertIn("changed", str(caught.exception))

    def test_a_blob_that_moved(self):
        html = '<script id="__NEXT_DATA__">{"props": {"somethingElse": 1}}</script>'
        with self.assertRaises(spotify.SpotifyUnavailable) as caught:
            spotify._entity_from_html(html)
        self.assertIn("changed", str(caught.exception))

    def test_a_blob_that_is_there_but_empty(self):
        html = '<script id="__NEXT_DATA__">{"props":{"pageProps":{"state":{"data":{"entity":{}}}}}}</script>'
        with self.assertRaises(spotify.SpotifyUnavailable) as caught:
            spotify._entity_from_html(html)
        self.assertIn("song details", str(caught.exception))

    def test_a_dead_link_says_so_rather_than_blaming_the_page(self):
        import urllib.error

        with mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.HTTPError("u", 404, "Not Found", {}, None)):
            with self.assertRaises(spotify.SpotifyUnavailable) as caught:
                spotify._embed_entity("track", "nope")
        self.assertIn("private playlist", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
