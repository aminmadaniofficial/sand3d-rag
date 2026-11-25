"""Unit tests for core text, parsing, and tensor helpers."""

import unittest
import torch

from config import build_search_document, extract_feature_tensor, tokenize_bm25
from generate import build_generation_prompt, extract_python_code


class TestConfigHelpers(unittest.TestCase):
    def test_build_search_document_includes_taxonomy_and_style(self) -> None:
        row = {
            "prompt": "Mossy boulder with ferns",
            "domain": "geology",
            "family": "boulders",
            "subject": "granite boulder",
            "style": "stylized",
        }
        doc = build_search_document(row)
        self.assertIn("Mossy boulder with ferns", doc)
        self.assertIn("domain: geology", doc)
        self.assertIn("family: boulders", doc)
        self.assertIn("subject: granite boulder", doc)
        self.assertIn("style: stylized", doc)

    def test_tokenize_bm25_supports_persian_and_english(self) -> None:
        tokens = tokenize_bm25("صخره خزه‌دار Granite-Boulder 01!")
        self.assertIn("صخره", tokens)
        self.assertIn("granite", tokens)
        self.assertIn("boulder", tokens)
        self.assertIn("01", tokens)

    def test_extract_feature_tensor_variants(self) -> None:
        raw = torch.ones((2, 4))
        self.assertTrue(torch.equal(extract_feature_tensor(raw), raw))

        class PoolObj:
            pooler_output = torch.zeros((2, 4))

        self.assertTrue(torch.equal(extract_feature_tensor(PoolObj()), torch.zeros((2, 4))))


class TestGenerationHelpers(unittest.TestCase):
    def test_extract_python_code_from_fenced_block(self) -> None:
        reply = "Here is the script:\n```python\nimport sand3d as s3d\nprint('ok')\n```\nDone."
        self.assertEqual(extract_python_code(reply), "import sand3d as s3d\nprint('ok')")

    def test_extract_python_code_unclosed_fence(self) -> None:
        reply = "```python\nimport sand3d as s3d\nx = 1"
        self.assertEqual(extract_python_code(reply), "import sand3d as s3d\nx = 1")

    def test_build_generation_prompt_contains_examples_and_request(self) -> None:
        class DummyHit:
            payload = {
                "prompt": "Forest mushroom cluster",
                "domain": "fungi",
                "family": "mushrooms",
                "subject": "toadstool",
                "entry": "build_mushroom",
                "program": "import sand3d as s3d\ndef build_mushroom(seed=0): pass",
            }

        sys_p, usr_p = build_generation_prompt("قارچ جنگلی", [DummyHit()], "s3d.at(...)")
        self.assertIn("sand3d", sys_p)
        self.assertIn("build_mushroom", usr_p)
        self.assertIn("قارچ جنگلی", usr_p)


if __name__ == "__main__":
    unittest.main()
