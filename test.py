import torchaudio,sys

import logging
import os

from modules.repos_static.sys_paths import setup_repos_paths
from modules.third_party_path import setup_third_party_paths

try:
    setup_third_party_paths()
    setup_repos_paths()
    # NOTE: 因为 logger 都是在模块中初始化，所以这个 config 必须在最前面
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
except BaseException:
    pass

from cosyvoice.cli.cosyvoice import CosyVoice2,CosyVoice
from cosyvoice.utils.file_utils import load_wav

sys.path.insert(0, "third_party/Matcha-TTS")

cosyvoice = CosyVoice('models/CosyVoice-300M-25Hz')
# cosyvoice = CosyVoice2('models/CosyVoice2-0.5B', load_jit=True, load_trt=True)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

texts = [
    '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
    'Speech AI Forge 是一个围绕 TTS 生成模型开发的项目，实现了 API Server 和 基于 Gradio 的 WebUI。',
    '这瓜啊，咬一口下去，汁水就满嘴巴都是，那瓜瓤是粉红色的，又脆又甜，口感跟普通的黑子西瓜、沙瓤西瓜都不一样，是那种清脆爽口的甜。',

]
for text in texts:
    for i, j in enumerate(cosyvoice.inference_zero_shot(text, '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
