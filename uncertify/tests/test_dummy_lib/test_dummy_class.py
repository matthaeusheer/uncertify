import unittest

from uncertify.dummy_lib.dummy_module import DummyClass


class TestDummyClass(unittest.TestCase):
    def test_always_true(self) -> None:
        self.assertTrue(DummyClass.always_true())

    def test_always_false(self) -> None:
        self.assertFalse(DummyClass.always_false())
