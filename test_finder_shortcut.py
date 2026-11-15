import unittest

import finder_shortcut as fs


class TestEnsure(unittest.TestCase):
    def test_adds_when_missing(self):
        added = []
        self.assertTrue(fs.ensure("/Users/a/Music/Beat Thief", ["/Users/a/Desktop"], added.append))
        self.assertEqual(added, ["/Users/a/Music/Beat Thief"])

    def test_an_entry_made_by_hand_counts(self):
        added = []
        self.assertFalse(fs.ensure("/Users/a/Music/Beat Thief",
                                   ["/Users/a/Desktop", "/Users/a/Music/Beat Thief/"], added.append))
        self.assertEqual(added, [])

    def test_empty_paths_are_ignored(self):
        added = []
        fs.ensure("/x/Beat Thief", ["", None], added.append)
        self.assertEqual(added, ["/x/Beat Thief"])


if __name__ == "__main__":
    unittest.main()
