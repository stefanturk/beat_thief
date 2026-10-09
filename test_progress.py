import unittest

import progress


class Clock:
    """Time that moves only when a test says so."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


def playlist_song(index, total, title, instruments=("drums", "bass")):
    """The events one song of a Spotify playlist sends, in the order the
    pipeline sends them, each with how long it took to arrive."""
    events = [
        (0, {"stage": "resolving", "index": index, "total": total, "song": title}),
        (2, {"stage": "found", "total": 1, "song": title, "index": index, "queue_total": total}),
        (1, {"stage": "downloading", "song": title, "index": index, "total": total, "percent": 0.0}),
        (4, {"stage": "downloading", "song": title, "index": index, "total": total, "percent": 50.0}),
        (4, {"stage": "downloaded", "song": title, "index": index, "total": total}),
        (1, {"stage": "sanitizing", "index": index, "total": total}),
        (5, {"stage": "tempo"}),
    ]
    for i, name in enumerate(instruments, 1):
        n = len(instruments)
        events += [
            (3, {"stage": "isolating", "instrument": name, "song": title, "index": i, "total": n,
                 "percent": None, "phase": None}),
            (20, {"stage": "isolating", "instrument": name, "song": title, "index": i, "total": n,
                  "percent": 60.0, "phase": None}),
            (10, {"stage": "isolated", "instrument": name, "song": title, "outputs": []}),
        ]
    return events


def tracker(total=3, instruments=("drums", "bass"), clock=None):
    return progress.RunProgress(progress.steps_for(True, True, instruments), total=total,
                                clock=clock or Clock())


def play(run, events, clock):
    views = []
    for wait, event in events:
        clock.tick(wait)
        run.feed(event)
        views.append((event, run.view()))
    return views


class TestTheBarNeverLosesItsPlace(unittest.TestCase):
    def test_the_whole_run_is_one_percentage_that_only_goes_forwards(self):
        clock = Clock()
        run = tracker(clock=clock)
        events = []
        for n, title in enumerate(("One", "Two", "Three"), 1):
            events += playlist_song(n, 3, title)

        percents = [view["percent"] for _, view in play(run, events, clock)]

        self.assertNotIn(None, percents)
        self.assertEqual(percents, sorted(percents))
        self.assertGreater(percents[-1], 95)

    def test_the_steps_after_the_download_stay_on_the_same_song(self):
        # "sanitizing", "tempo" and "isolating" don't say which song they're
        # on - "isolating"'s index is the instrument's.
        clock = Clock()
        run = tracker(clock=clock)
        play(run, playlist_song(1, 3, "One"), clock)
        views = play(run, playlist_song(2, 3, "Two"), clock)

        for event, view in views:
            with self.subTest(stage=event["stage"]):
                self.assertEqual((view["index"], view["total"]), (2, 3))
                self.assertGreaterEqual(view["percent"], 100 / 3 - 1e-9)

    def test_isolating_does_not_replace_the_runs_bar_with_its_own(self):
        clock = Clock()
        run = tracker(clock=clock)
        play(run, playlist_song(1, 3, "One"), clock)
        run.feed({"stage": "isolating", "instrument": "drums", "song": "One",
                  "index": 1, "total": 2, "percent": 99.0})

        self.assertLess(run.view()["percent"], 34)
        self.assertEqual(run.view()["step_percent"], 99.0)

    def test_it_says_which_step_is_in_hand_and_which_come_next(self):
        clock = Clock()
        run = tracker(clock=clock)
        play(run, playlist_song(1, 3, "One")[:6], clock)   # up to sanitizing

        states = {step["name"]: step["state"] for step in run.view()["steps"]}

        self.assertEqual(states, {"Match": "done", "Download": "done", "Clean up": "now",
                                  "Tempo": "next", "Drums": "next", "Bass": "next"})

    def test_match_is_only_a_step_for_songs_that_are_matched(self):
        run = tracker(total=1)
        run.feed({"stage": "found", "total": 1, "song": "Redbone"})

        self.assertNotIn("Match", [step["name"] for step in run.view()["steps"]])

    def test_nothing_is_claimed_before_there_is_anything_to_go_on(self):
        run = tracker()
        run.feed({"stage": "looking-up"})

        self.assertIsNone(run.view()["percent"])

    def test_a_resumed_playlist_starts_where_it_left_off(self):
        run = tracker(total=10)
        run.feed({"stage": "resuming", "done": 6, "total": 10})
        run.feed({"stage": "resolving", "index": 7, "total": 10, "song": "Seven"})

        self.assertGreaterEqual(run.view()["percent"], 60)
        self.assertEqual(run.view()["index"], 7)

    def test_the_questions_at_the_end_take_over_the_bar(self):
        run = progress.RunProgress(["download"], total=3, then="your questions", clock=Clock())
        run.feed({"stage": "downloading", "index": 3, "total": 3, "percent": 100.0, "song": "x"})
        self.assertEqual(run.view()["then"], "your questions")

        run.feed({"stage": "checking", "index": 1, "total": 2, "song": "One"})

        view = run.view()
        self.assertIsNone(view["percent"])     # the checking count is its own
        self.assertEqual((view["steps"], view["eta"], view["then"]), ([], "", ""))


class TestTheTimeLeft(unittest.TestCase):
    def test_none_until_a_song_has_been_timed(self):
        clock = Clock()
        run = tracker(clock=clock)
        play(run, playlist_song(1, 3, "One"), clock)
        self.assertEqual(run.view()["eta"], "")

        run.feed({"stage": "resolving", "index": 2, "total": 3, "song": "Two"})

        self.assertTrue(run.view()["eta"].startswith("about"))

    def test_it_learns_from_this_run_rather_than_the_guesses(self):
        clock = Clock()
        run = tracker(total=11, instruments=(), clock=clock)
        events = playlist_song(1, 11, "One", instruments=())
        # Downloads that take a minute rather than the guessed 12 seconds.
        events[3] = (60, events[3][1])
        play(run, events, clock)
        run.feed({"stage": "resolving", "index": 2, "total": 11, "song": "Two"})

        left = run.seconds_left()

        # Ten songs to go, each about 2+1+64+1+5 seconds as measured.
        self.assertAlmostEqual(left, 10 * 73 + 0, delta=10)

    def test_waiting_offline_is_not_counted_as_the_step_taking_long(self):
        clock = Clock()
        run = tracker(total=2, instruments=(), clock=clock)
        run.feed({"stage": "resolving", "index": 1, "total": 2, "song": "One"})
        clock.tick(3)
        run.feed({"stage": "found", "index": 1, "queue_total": 2, "song": "One"})
        run.feed({"stage": "offline", "index": 1, "total": 2})
        clock.tick(600)
        run.feed({"stage": "downloading", "index": 1, "total": 2, "percent": 100.0, "song": "One"})
        clock.tick(2)
        run.feed({"stage": "resolving", "index": 2, "total": 2, "song": "Two"})

        self.assertLess(run.expected("download"), 10)

    def test_a_question_on_the_page_is_not_counted_either(self):
        clock = Clock()
        run = tracker(total=1, instruments=(), clock=clock)
        run.feed({"stage": "found", "total": 1, "song": "One"})
        run.pause()
        clock.tick(300)
        run.resume()
        clock.tick(2)
        run.feed({"stage": "sanitizing"})

        self.assertAlmostEqual(run.expected("download"), 2)

    def test_one_song_waits_a_few_seconds_before_guessing(self):
        clock = Clock()
        run = tracker(total=1, clock=clock)
        run.feed({"stage": "found", "total": 1, "song": "One"})
        self.assertEqual(run.view()["eta"], "")

        clock.tick(progress.SINGLE_SONG_SETTLE_SEC)

        self.assertTrue(run.view()["eta"].startswith("about"))

    def test_the_first_instrument_is_the_slow_one(self):
        # Every later instrument reuses the first one's separation.
        run = tracker()
        self.assertGreater(run.expected("drums"), 5 * run.expected("bass"))


class TestTheWording(unittest.TestCase):
    def test_rounded_loosely_at_every_scale(self):
        cases = {None: "", 4: "almost done", 42: "about 40 sec left", 140: "about 2 min left",
                 3600: "about 1 hr left", 4850: "about 1 hr 21 min left"}
        for seconds, text in cases.items():
            with self.subTest(seconds=seconds):
                self.assertEqual(progress.eta_text(seconds), text)


class TestOtherKindsOfRun(unittest.TestCase):
    def test_taking_stems_from_a_song_already_here(self):
        clock = Clock()
        run = progress.RunProgress(progress.steps_for(True, True, ["drums", "vocals"], download=False),
                                   clock=clock)
        views = play(run, [(0, e) for _, e in playlist_song(1, 1, "One", ("drums", "vocals"))
                           if e["stage"] in ("isolating", "isolated")], clock)

        self.assertEqual([step["name"] for step in views[0][1]["steps"]], ["Drums", "Vocals"])
        self.assertEqual(views[-1][1]["percent"], 100)

    def test_tidying_a_folder(self):
        run = progress.RunProgress(["clean"], clock=Clock())
        run.feed({"stage": "tidying", "index": 3, "total": 4, "song": "Redbone"})

        self.assertEqual((run.view()["index"], run.view()["total"]), (3, 4))
        self.assertGreaterEqual(run.view()["percent"], 50)


if __name__ == "__main__":
    unittest.main()
