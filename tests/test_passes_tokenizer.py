"""copy_tokenizer: files copied byte-identical, tokenizer_config.json gets the generic fast class, other keys kept."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p5_assemble import TOKENIZER_CLASS, copy_tokenizer  # noqa: E402


def test_copy_tokenizer_rewrites_class_and_keeps_rest(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "tokenizer.json").write_bytes(b'{"version": "1.0", "model": {"type": "BPE"}}')
    (src / "chat_template.jinja").write_text("{{ messages }}")
    (src / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "LlamaTokenizerFast", "add_bos_token": True, "bos_token": "<s>", "legacy": True}))
    copy_tokenizer(str(src), str(dst))
    assert (dst / "tokenizer.json").read_bytes() == (src / "tokenizer.json").read_bytes()
    assert (dst / "chat_template.jinja").read_text() == "{{ messages }}"
    cfg = json.load(open(dst / "tokenizer_config.json"))
    assert cfg["tokenizer_class"] == TOKENIZER_CLASS == "PreTrainedTokenizerFast"
    assert cfg["add_bos_token"] is True and cfg["bos_token"] == "<s>" and cfg["legacy"] is True
    assert not (dst / "tokenizer.model").exists()
