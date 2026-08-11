import unittest

import groove_reader


def grid(patterns: dict[str, str]) -> dict[tuple[int, str], int]:
    """A loop written out by hand. "X" is a hit at velocity 100, "x" one at
    40, "." nothing, and "|" is a bar line and is ignored.

    Every test here is a pattern and an assertion about a pattern - no audio,
    no model. That's what makes these rules arguable."""
    placed = {}
    for piece, pattern in patterns.items():
        for step, cell in enumerate(pattern.replace("|", "")):
            if cell == "X":
                placed[(step, piece)] = 100
            elif cell == "x":
                placed[(step, piece)] = 40
    return placed


def steps_of(placed: dict, piece: str) -> list[int]:
    return sorted(step for (step, this) in placed if this == piece)


def length(patterns: dict[str, str]) -> int:
    return len(next(iter(patterns.values())).replace("|", ""))


def refine(patterns: dict[str, str], steps_per_bar: int = 16):
    return groove_reader.refine(grid(patterns), length(patterns), steps_per_bar)


class TestTakingAVoiceOffTheSnare(unittest.TestCase):
    """The snare class arrives holding whatever the model couldn't name. What
    says it isn't a snare is what it plays, not what any one hit sounds
    like."""

    def test_a_voice_on_every_quarter_is_not_a_snare(self):
        # Four bars of the Brasstracks intro look exactly like this: the
        # snare class on all sixteen quarter notes, no gaps.
        out, inferred = refine({
            "kick":       "X..X..XX.X...X..|X..X...X.X...X..|X..X..XX.X...X..|X..X...X.X...X..",
            "tambourine": "X...X...X...X...|X...X...X...X...|X...X...X...X...|X...X...X...X...",
        })

        self.assertEqual(steps_of(out, "tambourine"), list(range(0, 64, 4)))
        self.assertEqual(steps_of(out, "snare"), [])
        self.assertEqual(inferred, 0)

    def test_a_voice_split_across_two_pads_is_made_one(self):
        # The per-hit vote is weak by design, so it comes back mixed. The
        # pulse is what says these eight hits are one instrument.
        out, inferred = refine({
            "tambourine": "X...X.......X...|X...X...X.......|X...X...X...X...|....X...X...X...",
            "snare":      "........X.......|............X...|................|X...............",
        })

        self.assertEqual(steps_of(out, "tambourine"), list(range(0, 64, 4)))

    def test_a_snare_on_the_pulse_keeps_its_pad_and_gains_a_neighbour(self):
        # One onset, two instruments: a tambourine struck with the backbeat
        # is inside the snare's window and cannot be heard out of it. The
        # snare stays; the tambourine is put back.
        out, inferred = refine({
            "tambourine": "X.......X...X...|X...X...X...X...",
            "snare":      "....X.......X...|................",
        })

        self.assertEqual(steps_of(out, "tambourine"), list(range(0, 32, 4)))
        self.assertEqual(steps_of(out, "snare"), [4, 12])
        self.assertEqual(inferred, 2)

    def test_a_voice_on_every_sixteenth_is_left_on_the_snare(self):
        # On a pulse that fine every snare in the loop is on the pulse by
        # definition, so a percussion line with a snare through it and a
        # snare playing sixteenths look identical - and confirming it doubles
        # the busiest bar in the song. The outro of the stem this was built
        # against is exactly this shape, and is a busy snare.
        out, inferred = refine({"tambourine": "X" * 32})

        self.assertEqual(steps_of(out, "snare"), list(range(32)))
        self.assertEqual(inferred, 0)

    def test_a_stray_bright_hit_is_still_a_snare(self):
        # One hit voted percussion and nothing else agrees with it. A lone
        # tambourine note scattered into a loop is the failure this stage
        # exists to prevent.
        out, inferred = refine({
            "snare":      "....X.......X...|....X.......X...",
            "tambourine": "..............X.|................",
        })

        self.assertEqual(steps_of(out, "tambourine"), [])
        self.assertEqual(steps_of(out, "snare"), [4, 12, 14, 20, 28])
        self.assertEqual(inferred, 0)

    def test_hits_off_the_pulse_stay_on_the_snare(self):
        # A tambourine on the quarters and a snare fill around it: the fill
        # isn't part of the pulse and doesn't become percussion.
        out, _ = refine({
            "tambourine": "X...X...X...X...|X...X...X...X...",
            "snare":      "..X...........X.|..X.............",
        })

        self.assertEqual(steps_of(out, "tambourine"), list(range(0, 32, 4)))
        self.assertEqual(steps_of(out, "snare"), [2, 14, 18])


class TestTakingAVoiceOffTheHihat(unittest.TestCase):
    """The hi-hat class's equivalent of TestTakingAVoiceOffTheSnare. Unlike
    the snare class, a real hi-hat is also on a pulse - so every pattern here
    is written to have exactly the same rhythm on both pieces, and what
    decides the outcome is only which piece the audio put the hits on
    ("closed hat" or "hat percussion"), not the shape of the pattern."""

    def test_a_voice_the_audio_heard_as_percussion_moves(self):
        out, inferred = refine({
            "hat percussion": "X...X...X...X...|X...X...X...X...|X...X...X...X...|X...X...X...X...",
        })

        self.assertEqual(steps_of(out, "tambourine"), list(range(0, 64, 4)))
        self.assertEqual(steps_of(out, "closed hat"), [])
        self.assertEqual(inferred, 0)

    def test_a_voice_the_audio_heard_as_a_real_hihat_stays(self):
        out, inferred = refine({
            "closed hat": "X...X...X...X...|X...X...X...X...|X...X...X...X...|X...X...X...X...",
        })

        self.assertEqual(steps_of(out, "closed hat"), list(range(0, 64, 4)))
        self.assertEqual(steps_of(out, "tambourine"), [])
        self.assertEqual(inferred, 0)

    def test_a_voice_split_across_two_pads_is_made_one(self):
        # Same pulse, mixed per-hit votes - the pulse is what says these are
        # one instrument, same as it does for the snare class.
        out, inferred = refine({
            "hat percussion": "X...X.......X...|X...X...X.......|X...X...X...X...|....X...X...X...",
            "closed hat":      "........X.......|............X...|................|X...............",
        })

        self.assertEqual(steps_of(out, "tambourine"), list(range(0, 64, 4)))

    def test_a_near_coin_flip_vote_is_not_trusted(self):
        # Same shape as what the fidelity tool found on a jazz kit ("Three",
        # Nicholas Payton, see beat-thief-jazz-genre-limits in project
        # memory): the audio's vote was barely different from a coin flip
        # (54% of 41 hits) - not the kind of evidence that should move a
        # whole voice off a real instrument. A flat 30%-of-hits threshold
        # (what the snare class still uses) would have happily called this a
        # voice; this pool requires the vote to actually be distinguishable
        # from noise, scaled to how many hits it's built from.
        steps_per_bar, bars = 16, 10
        total_steps = steps_per_bar * bars
        positions = list(range(0, total_steps, 2))  # an eighth-note pulse
        placed = {
            (step, "hat percussion" if i % 20 < 11 else "closed hat"): 100
            for i, step in enumerate(positions)
        }

        out, inferred = groove_reader.refine(placed, total_steps, steps_per_bar)

        self.assertEqual(steps_of(out, "tambourine"), [])
        self.assertEqual(inferred, 0)

    def test_a_lopsided_vote_is_trusted_even_with_the_same_pulse(self):
        # Same pulse and sample size as the coin-flip case above, but a vote
        # that's actually one-sided - the fidelity tool's confirmed case
        # (Officially Missing You's solo tambourine section) voted 82% of 28.
        steps_per_bar, bars = 16, 10
        total_steps = steps_per_bar * bars
        positions = list(range(0, total_steps, 2))
        placed = {
            (step, "hat percussion" if i % 20 < 17 else "closed hat"): 100
            for i, step in enumerate(positions)
        }

        out, inferred = groove_reader.refine(placed, total_steps, steps_per_bar)

        self.assertEqual(steps_of(out, "tambourine"), positions)

    def test_a_stray_bright_hit_is_still_a_closed_hat(self):
        # One hit voted percussion out of a steady closed-hat voice. A lone
        # tambourine note scattered into a loop is the failure this stage
        # exists to prevent, same as for the snare class.
        out, inferred = refine({
            "closed hat":     "....X...X...X...X...X...X...X.|....X...X...X...X...X...X...X.",
            "hat percussion": "................................|X...............................",
        })

        self.assertEqual(steps_of(out, "tambourine"), [])
        self.assertEqual(inferred, 0)


class TestLeavingRealDrumsAlone(unittest.TestCase):
    """Everything above has to happen without touching a beat that was
    transcribed correctly in the first place."""

    def test_a_backbeat_is_left_alone(self):
        patterns = {
            "kick":  "X.......X.......|X.......X.......|X.......X.......|X.......X.......",
            "snare": "....X.......X...|....X.......X...|....X.......X...|....X.......X...",
        }
        out, inferred = refine(patterns)

        self.assertEqual(out, grid(patterns))
        self.assertEqual(inferred, 0)

    def test_a_snare_fill_is_not_a_voice_keeping_time(self):
        # Dense over one bar and absent over the rest: its own hits are the
        # strays, which is what tells a fill from a part.
        patterns = {
            "snare": "....X.......X...|....X.......X...|....X.......X...|XXXXXXXXXXXXXXXX",
        }
        out, inferred = refine(patterns)

        self.assertEqual(out, grid(patterns))
        self.assertEqual(inferred, 0)

    def test_a_kick_is_never_filled_in(self):
        # A structural voice's silences are the composition. Even with a
        # crash covering the gap, nothing goes back.
        patterns = {
            "kick":  "X...X...X...X...|X...X.......X...|X...X...X...X...|X...X...X...X...",
            "crash": "................|........X.......|................|................",
        }
        out, inferred = refine(patterns)

        self.assertEqual(out, grid(patterns))
        self.assertEqual(inferred, 0)

    def test_three_hits_are_not_a_pattern(self):
        patterns = {"tambourine": "X...X...X.......|................"}
        out, inferred = refine(patterns)

        self.assertEqual(steps_of(out, "snare"), [0, 4, 8])
        self.assertEqual(inferred, 0)


class TestRestoringWhatWasCovered(unittest.TestCase):
    def test_a_hat_under_a_crash_comes_back(self):
        out, inferred = refine({
            "crash":      "X...............|................",
            "closed hat": "..X.X.X.X.X.X.X.|X.X.X.X.X.X.X.X.",
        })

        self.assertIn(0, steps_of(out, "closed hat"))
        self.assertEqual(inferred, 1)

    def test_a_hat_that_simply_stops_does_not_come_back(self):
        # Nothing is on step 16, so nothing was covering anything. The
        # drummer stopped, and that's part of the groove.
        out, inferred = refine({
            "closed hat": "X.X.X.X.X.X.X.X.|..X.X.X.X.X.X.X.",
        })

        self.assertNotIn(16, steps_of(out, "closed hat"))
        self.assertEqual(inferred, 0)

    def test_a_restored_hit_is_as_loud_as_the_ones_around_it(self):
        out, _ = refine({
            "crash":      "X...............|................",
            "closed hat": "..x.X.x.X.x.X.x.|X.x.X.x.X.x.X.x.",
        })

        # Step 0 is on a beat, and this voice accents the beat - the restored
        # hit has to be an accent too, not the one flat note in the bar.
        self.assertEqual(out[(0, "closed hat")], 100)

    def test_an_off_beat_restore_is_as_quiet_as_the_off_beats(self):
        out, _ = refine({
            "snare":      "..X.............|................",
            "closed hat": "X...X.x.X.x.X.x.|X.x.X.x.X.x.X.x.",
        })

        self.assertEqual(out[(2, "closed hat")], 40)


class TestGhostNotes(unittest.TestCase):
    """Which of what's left on the snare pad is quiet enough to be in the
    way of the hits that are meant to sing. Decided here and not in the
    transcriber, because until the voices have stopped moving between pads it
    isn't settled what's on the snare pad at all."""

    def test_a_quiet_snare_gets_its_own_pad(self):
        out, _ = refine({"snare": "....X.......x...|....X.......x..."})

        self.assertEqual(steps_of(out, "snare"), [4, 20])
        self.assertEqual(steps_of(out, "ghost snare"), [12, 28])

    def test_a_ghost_keeps_its_velocity(self):
        out, _ = refine({"snare": "....X.......x...|................"})

        self.assertEqual(out[(12, "ghost snare")], 40)

    def test_a_percussion_hit_never_becomes_a_ghost_note(self):
        # A tambourine played softly is a tambourine. Putting it on the ghost
        # pad would hide it under the snare all over again.
        out, _ = refine({"tambourine": "x...x...x...x...|x...x...x...x..."})

        self.assertEqual(len(steps_of(out, "tambourine")), 8)
        self.assertEqual(steps_of(out, "ghost snare"), [])

    def test_a_demoted_percussion_vote_is_judged_on_its_velocity(self):
        # It came back from the tambourine pad because nothing agreed with
        # it, and it was quiet - so it belongs on the ghost pad, not on the
        # snare. Getting this wrong promoted quiet hits to full snares.
        out, _ = refine({
            "snare":      "....X.......X...|....X.......X...",
            "tambourine": "..x.............|................",
        })

        self.assertEqual(steps_of(out, "ghost snare"), [2])
        self.assertEqual(steps_of(out, "snare"), [4, 12, 20, 28])


class TestHowFineTheGridIs(unittest.TestCase):
    """A loop is quantized to thirty-seconds, but a pulse is a musical thing
    and has to mean the same at any resolution. These are written twice, once
    on each grid, because the bug this guards against is a pulse silently
    becoming twice as fast when the grid does."""

    def test_the_same_music_reads_the_same_on_either_grid(self):
        # A tambourine on the quarters with the backbeat over it, written out
        # once in sixteenths and once in thirty-seconds. Same two bars.
        sixteenths, _ = refine({
            "tambourine": "X...X...X...X...|X...X...X...X...",
            "snare":      "....X.......X...|....X.......X...",
        }, steps_per_bar=16)
        thirty_seconds, _ = refine({
            "tambourine": "X.......X.......X.......X.......|X.......X.......X.......X.......",
            "snare":      "........X...............X.......|........X...............X.......",
        }, steps_per_bar=32)

        self.assertEqual(len(steps_of(sixteenths, "tambourine")), 8)
        self.assertEqual(len(steps_of(thirty_seconds, "tambourine")), 8)
        self.assertEqual(len(steps_of(sixteenths, "snare")),
                         len(steps_of(thirty_seconds, "snare")))

    def test_a_voice_on_every_thirty_second_is_a_shaker(self):
        # The pulse the finer grid can express that the old one couldn't. A
        # voice this fast isn't marking the beat, it's a swish.
        out, _ = refine({"tambourine": "X" * 64}, steps_per_bar=32)

        self.assertEqual(len(steps_of(out, "shaker")), 64)
        self.assertEqual(steps_of(out, "snare"), [])

    def test_a_voice_on_the_sixteenths_is_still_left_on_the_snare(self):
        # The refusal has to survive the change of grid: on thirty-seconds a
        # sixteenth voice is a pulse of period two rather than of period one,
        # and the rule is about sixteenths, not about "the finest thing the
        # grid can hold".
        out, inferred = refine({"tambourine": "X." * 32}, steps_per_bar=32)

        self.assertEqual(len(steps_of(out, "snare")), 32)
        self.assertEqual(steps_of(out, "tambourine"), [])
        self.assertEqual(inferred, 0)

    def test_a_pulse_finer_than_the_grid_is_not_looked_for(self):
        # steps_per_beat // division is zero for thirty-seconds on a grid of
        # eighths, and a period of zero is an infinite loop waiting to happen.
        out, _ = refine({"tambourine": "X" * 8}, steps_per_bar=8)

        self.assertEqual(len(out), 8)


class TestTheGridItself(unittest.TestCase):
    def test_an_empty_loop_comes_back_empty(self):
        out, inferred = groove_reader.refine({}, 32, 16)

        self.assertEqual(out, {})
        self.assertEqual(inferred, 0)

    def test_a_snare_heard_on_its_own_survives_a_percussion_voice(self):
        patterns = {
            "kick":       "X.....X.........|X.....X.........",
            "tambourine": "X.X.X.X.X.X.X.X.|X.X.X.X.X.X.X.X.",
            "snare":      "....X.......X...|....X.......X...",
        }
        out, _ = refine(patterns)

        self.assertEqual(steps_of(out, "snare"), [4, 12, 20, 28])
        self.assertEqual(steps_of(out, "kick"), [0, 6, 16, 22])

    def test_a_snare_vote_on_a_kick_step_is_not_believed(self):
        # The one measurement Part 0 was unambiguous about: a kick's shell
        # reaches into the band that says "this was a drum", so a hit sharing
        # a sixteenth with a kick reads as a drum whatever it was. On the real
        # bridge this was eight of the nine snares left behind on a voice that
        # is a tambourine all the way through.
        out, _ = refine({
            "kick":       "....X.......X...|....X.......X...",
            "tambourine": "X.X.X.X.X.X.X.X.|X.X.X.X.X.X.X.X.",
            "snare":      "....X.......X...|....X.......X...",
        })

        self.assertEqual(steps_of(out, "snare"), [])
        self.assertEqual(len(steps_of(out, "tambourine")), 16)

    def test_the_input_is_not_modified(self):
        patterns = {"tambourine": "X...X...X...X...|X...X...X...X..."}
        placed = grid(patterns)
        before = dict(placed)

        groove_reader.refine(placed, 32, 16)

        self.assertEqual(placed, before)


if __name__ == "__main__":
    unittest.main()
