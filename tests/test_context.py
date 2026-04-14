import unittest

from alwin_voice.llm.context import ConversationContext


class TestConversationContext(unittest.TestCase):
    def test_context_keeps_last_turns(self) -> None:
        ctx = ConversationContext(max_turns=2)
        ctx.add_user("u1")
        ctx.add_assistant("a1")
        ctx.add_user("u2")
        ctx.add_assistant("a2")
        ctx.add_user("u3")
        ctx.add_assistant("a3")

        messages = ctx.as_ollama_messages("sys")
        contents = [m["content"] for m in messages]

        self.assertEqual(contents, ["sys", "u2", "a2", "u3", "a3"])


if __name__ == "__main__":
    unittest.main()
