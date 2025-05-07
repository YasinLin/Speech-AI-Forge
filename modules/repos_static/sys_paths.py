import os
import sys

def get_base_path():
    """获取应用程序的基础路径，兼容开发模式和PyInstaller打包模式"""
    if getattr(sys, 'frozen', False):
        # 打包环境
        if hasattr(sys, '_MEIPASS'):
            # --onefile模式下的临时解压目录（资源文件在这里）
            return sys._MEIPASS
        else:
            # --onedir模式下的可执行文件所在目录
            return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def REPO_DIR(name):
    base_path = get_base_path()
    return os.path.abspath(os.path.join(base_path, name))

# REPO_DIR = lambda name: os.path.abspath(os.path.join(os.path.dirname(__file__), name))

paths = [
    REPO_DIR("cosyvoice"),
    REPO_DIR("openvoice"),
    REPO_DIR("fish_speech"),
    REPO_DIR("FireRedTTS"),
    REPO_DIR("F5TTS"),
]


def setup_repos_paths():
    for pth in paths:
        if pth not in sys.path:
            sys.path.insert(0, pth)
