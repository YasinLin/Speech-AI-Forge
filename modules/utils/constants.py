import os
import sys

cwd = os.getcwd()

utils_path = os.path.dirname(os.path.realpath(__file__))
modules_path = os.path.dirname(utils_path)
def get_root_path():
    """获取应用程序的基础路径，兼容开发模式和PyInstaller打包模式"""
    if getattr(sys, 'frozen', False):
        # 打包环境
        if hasattr(sys, '_MEIPASS'):
            # --onefile模式下的临时解压目录（资源文件在这里）
            return os.path.dirname(sys.executable)  # 返回真实可执行文件路径
        else:
            # --onedir模式下的可执行文件所在目录
            return os.path.dirname(sys.argv)        # 返回执行路径
    return os.path.dirname(modules_path)

ROOT_DIR = get_root_path()
DATA_DIR = os.path.join(ROOT_DIR, "data")

MODELS_DIR = os.path.join(ROOT_DIR, "models")

SPEAKERS_DIR = os.path.join(DATA_DIR, "speakers")
