import importlib.util
import unittest
from pathlib import Path


def load_conv():
    path = Path(__file__).resolve().parent / "conversation.py"
    spec = importlib.util.spec_from_file_location("yi_conversation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UserContentTest(unittest.TestCase):
    def test_a_text_part_is_kept_without_an_image(self):
        conv_mod = load_conv()
        conv = conv_mod.conv_templates["mm_default"].copy()
        image_file, has_image = conv_mod.append_chat_messages(
            conv,
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "where is the cat"}],
                }
            ],
            "<image>",
        )
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        self.assertIn("where is the cat", prompt)
        self.assertEqual(image_file, "")
        self.assertFalse(has_image)

        query, url = conv_mod.render_user_content(
            [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "a.png"}},
                {"type": "text", "text": "world"},
            ],
            "<image>",
        )
        self.assertEqual(query, "<image>\nhello\nworld")
        self.assertEqual(url, "a.png")

        query, url = conv_mod.render_user_content("hi there", "<image>")
        self.assertEqual(query, "hi there")
        self.assertEqual(url, "")


if __name__ == "__main__":
    unittest.main()
