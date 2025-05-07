import io
import torchaudio,sys

import logging
import os

from modules.repos_static.sys_paths import setup_repos_paths
from modules.third_party_path import setup_third_party_paths
from modules.ffmpeg_env import setup_ffmpeg_path

try:
    setup_third_party_paths()
    setup_repos_paths()
    setup_ffmpeg_path()
    # NOTE: 因为 logger 都是在模块中初始化，所以这个 config 必须在最前面
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
except BaseException:
    pass

from cosyvoice.cli.cosyvoice import CosyVoice2,CosyVoice
from cosyvoice.utils.file_utils import load_wav
from modules.core.models.tts.CosyVoiceModel import CosyVoiceTTSModel
from modules.core.pipeline.dcls import TTSPipelineContext, TTSSegment, TTSSpeaker
from modules.core.spk import spk_mgr, parse_wav_dtype
from modules.utils import audio_utils
from modules.core.models.AudioReshaper import AudioReshaper
import torch,numpy as np,wave
import soundfile as sf,librosa

# cosyvoice = CosyVoice('models/CosyVoice-300M', fp16=True)
# cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, fp16=True, use_flow_cache=True)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage

text = '我们走的每一步，都是我们策略的一部分；你看到的所有一切，包括我此刻与你交谈，所做的一切，所说的每一句话，都有深远的含义。'
prompt_text="希望你以后能够做的比我还好呦。"


def audio_to_bytes_and_get_sample_rate(audio_path):
    try:
        # 读取音频文件
        data, sample_rate = sf.read(audio_path)
        # 获取音频的字节流
        audio_bytes = data.tobytes()
        return audio_bytes, sample_rate,data.dtype
    except Exception as e:
        print(f"发生错误: {e}")
        return None, None

# audio_bytes, sample_rate,dtype = audio_to_bytes_and_get_sample_rate('./asset/zero_shot_prompt.wav')
# spk = TTSSpeaker.from_ref_wav_bytes((sample_rate, audio_bytes), prompt_text,str(dtype))

spk = TTSSpeaker.from_file("E:\code\speech\Speech-AI-Forge\data\speakers\男013.spkv1.json")
print("spk.id", spk.id)
def spk_to_ref_wav(spk: TTSSpeaker, emotion: str = "", target_sr=16000):
        ref_data = spk.get_ref(lambda x: x.emotion == emotion if len(emotion) > 0 else True)
        if ref_data is None:
            return None, None
        wav = audio_utils.bytes_to_librosa_array(
            audio_bytes=ref_data.wav, sample_rate=ref_data.wav_sr, dtype=ref_data.dtype
        )
        if ref_data.wav_sr != target_sr:
            _, wav = AudioReshaper.normalize_audio(
                audio=(ref_data.wav_sr, wav), target_sr=target_sr
            )
        return wav, ref_data.text

prompt_speech_16k, prompt_text = (
    spk_to_ref_wav(spk) if spk else (None, None)
)
prompt_speech_16k =torch.from_numpy(prompt_speech_16k).unsqueeze(0)
#prompt_speech_16k = np.frombuffer(prompt_speech_16k, dtype=np.int16)
# prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
# print(prompt_speech_16k)

texts = [
    '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
    'Speech AI Forge 是一个围绕 TTS 生成模型开发的项目，实现了 API Server 和 基于 Gradio 的 WebUI。',
    '这瓜啊，咬一口下去，汁水就满嘴巴都是，那瓜瓤是粉红色的，又脆又甜，口感跟普通的黑子西瓜、沙瓤西瓜都不一样，是那种清脆爽口的甜。',

]

cosyvoice =CosyVoiceTTSModel() # "cosy-voice2"
cosyvoice.load()
for text in texts:
    j = cosyvoice.inference_zero_shot([text], prompt_text,  prompt_speech_16k)
    torchaudio.save('zero_shot_{}.wav'.format('speech-300M'), j['tts_speech'], cosyvoice.sample_rate)


# sr, audio_data = cosyvoice.generate(
#     segment=TTSSegment(_type="text", text=text, spk=spk),
#     context=TTSPipelineContext(),
# )
# import soundfile as sf
# sf.write(f"test_cosyvoice{spk.name}.wav", audio_data, sr, format="WAV")


# cosyvoice = CosyVoice('models/CosyVoice-300M-25Hz', fp16=True)
# for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k)):
#     torchaudio.save('zero_shot_{}.wav'.format('CosyVoice-300M'), j['tts_speech'], cosyvoice.sample_rate)

# cosyvoice = CosyVoice2('models/CosyVoice2-0.5B')
# for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k)):
#     torchaudio.save('zero_shot_{}.wav'.format('CosyVoice2-0.5B'), j['tts_speech'], cosyvoice.sample_rate)




# cosyvoice = CosyVoiceTTSModel("cosy-voice2")
# cosyvoice.load()
# j = cosyvoice.inference_zero_shot([text], prompt_text,  prompt_speech_16k)
# torchaudio.save('zero_shot_{}.wav'.format('speech-2.5B'), j['tts_speech'], cosyvoice.sample_rate)