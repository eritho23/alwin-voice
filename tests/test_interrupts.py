import unittest

from alwin_voice.interrupts import is_clear_question_or_clarification


class TestInterrupts(unittest.TestCase):
    def test_clear_question_and_clarification_cases(self) -> None:
        cases = [
            "Vad menar du?",
            "Kan du upprepa",
            "Ursäkta, vad sa du",
            "Could you explain that?",
            "What do you mean",
            "Hur fungerar det",
        ]

        for text in cases:
            with self.subTest(text=text):
                self.assertTrue(is_clear_question_or_clarification(text))

    def test_noise_and_statement_cases(self) -> None:
        cases = [
            "hmm",
            "ehm ehm",
            "12345",
            "hej",
            "Det här är bara ett påstående.",
            "random background noise noise",
        ]

        for text in cases:
            with self.subTest(text=text):
                self.assertFalse(is_clear_question_or_clarification(text))


if __name__ == "__main__":
    unittest.main()
