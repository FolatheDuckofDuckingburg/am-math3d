import unittest

from map_app import parse_byte_range


class ParseByteRangeTests(unittest.TestCase):
    def test_suffix_range_selects_bytes_from_end(self):
        self.assertEqual(parse_byte_range("bytes=-500", 1000), (500, 999))

    def test_suffix_larger_than_file_selects_entire_file(self):
        self.assertEqual(parse_byte_range("bytes=-2000", 1000), (0, 999))

    def test_open_ended_range(self):
        self.assertEqual(parse_byte_range("bytes=100-", 1000), (100, 999))

    def test_end_is_clamped_to_file_size(self):
        self.assertEqual(parse_byte_range("bytes=900-1200", 1000), (900, 999))

    def test_unsatisfiable_ranges_are_distinguished(self):
        self.assertIs(parse_byte_range("bytes=1000-", 1000), False)
        self.assertIs(parse_byte_range("bytes=20-10", 1000), False)
        self.assertIs(parse_byte_range("bytes=-0", 1000), False)

    def test_malformed_and_multiple_ranges_are_ignored(self):
        self.assertIsNone(parse_byte_range("bytes=-", 1000))
        self.assertIsNone(parse_byte_range("bytes=0-1,4-5", 1000))

    def test_empty_file_cannot_satisfy_a_range(self):
        self.assertIs(parse_byte_range("bytes=0-", 0), False)


if __name__ == "__main__":
    unittest.main()
