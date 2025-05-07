import re


# from tn.chinese.normalizer import Normalizer as ZhNormalizer
# from tn.english.normalizer import Normalizer as EnNormalizer
# import inflect

from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph, is_only_punctuation


# zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False, overwrite_cache=True)
# en_tn_model = EnNormalizer()
# inflect_parser = inflect.engine()


from modules.core.tn.TNPipeline import GuessLang
from modules.repos_static.cosyvoice.cosyvoice.utils.frontend_utils import (
    remove_bracket,
    replace_blank,
    replace_corner_mark,
)

from .base_tn import BaseTN

CosyVoiceTN = BaseTN.clone()
CosyVoiceTN.freeze_tokens = [
    "[laughter]",
    "[breath]",
    "<laughter>",
    "</laughter>",
    "<storng>",
    "</storng>",
    # <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
    "<|zh|>",
    "<|en|>",
    "<|jp|>",
    "<|yue|>",
    "<|ko|>",
]


@CosyVoiceTN.block()
def cv_tn(text: str, guess_lang: GuessLang) -> str:
    text = text.strip()
    if guess_lang.zh_or_en == "en":
        # en_tn_model.normalize(text)
        # text = spell_out_number(text, inflect_parser)
        return text
    # NOTE: 这个在这里大概率不会触发，因为 tn 之前会 chunker split
    # text = text.replace("\n", "")  # 源代码这里把 \n 全部去掉了??? 这样不会有问题吗？
    # text = replace_blank(text)
    # text = replace_corner_mark(text)
    # text = text.replace(".", "、")
    # text = text.replace(" - ", "，")
    # text = remove_bracket(text)
    # text = re.sub(r"[，,]+$", "。", text)
    # text = zh_tn_model.normalize(text)
    text = text.replace("\n", "")
    text = replace_blank(text)
    text = replace_corner_mark(text)
    text = text.replace(".", "。")
    text = text.replace(" - ", "，")
    text = remove_bracket(text)
    text = re.sub(r'[，,、]+$', '。', text)
    return text


if __name__ == "__main__":
    pass
