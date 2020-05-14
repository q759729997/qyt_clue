import os
import io
import traceback
import logging
from logging import LoggerAdapter
from logging import Logger
from logging import addLevelName
from logging import currentframe
from logging import setLoggerClass
from logging import setLogRecordFactory
from logging import LogRecord

__all__ = ['logger', 'Log']


color = {
    'DEBUG': '34',
    'INFO': None,
    'WARNING': '33',
    'ERROR': '35'
}

color_cfg = {
    'LEVEL_CHAR_COLOR': True,
    'MSG_COLOR': False
}

color_fmt = '\033[;%sm%s\033[0m'


def format_msg_color(level_name, msg):
    """ 日志文本颜色格式化.

    @params:
        level_name - 日志级别.
        msg - 日志信息.

    @return:
        On success - 颜色格式化后的日志文本.
    """
    if level_name in color and color[level_name]:
        return color_fmt % (color[level_name], msg)
    else:
        return msg


class Log(LoggerAdapter):
    __date_fmt = "%Y-%m-%d %H:%M:%S"
    __format_str = "%(asctime)s %(levelname_first_char_with_color)s [%(filename)s:%(lineno)s] %(message)s"

    def __init__(self, **kwargs):
        super().__init__(logging.getLogger(__name__), {})
        self.level = 'DEBUG'
        self.formatter = logging.Formatter(self.__format_str, self.__date_fmt)
        handler = logging.StreamHandler()
        handler.setFormatter(self.formatter)
        self.file_handler = None
        self.logger.addHandler(handler)
        self.set_level(self.level)

    def debug(self, msg, *args, **kwargs):
        msg = format_msg_color('DEBUG', msg) if color_cfg['MSG_COLOR'] else msg
        super().debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        msg = format_msg_color('INFO', msg) if color_cfg['MSG_COLOR'] else msg
        super().info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        msg = format_msg_color('WARNING', msg) if color_cfg['MSG_COLOR'] else msg
        super().warning(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        msg = format_msg_color('WARNING', msg) if color_cfg['MSG_COLOR'] else msg
        super().warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        msg = format_msg_color('ERROR', msg) if color_cfg['MSG_COLOR'] else msg
        super().error(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        super().exception(msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg, *args, **kwargs):
        super().critical(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        super().log(level, msg, *args, **kwargs)

    def set_level(self, level: str):
        """ 设置 log level，低于 level 的log将被忽视.

        @params:
            level - ERROR, WARNING, INFO, DEBUG.
        """
        self.level = level
        self.logger.setLevel({
                                'ERROR': logging.ERROR,
                                'WARNING': logging.WARNING,
                                'INFO': logging.INFO,
                                'DEBUG': logging.DEBUG
                             }[level])

    def add_file_handler(self, filename: str, mode='w'):
        """ 增加一个输出日志文件位置，可以通过 `remove_file_handler`_ 函数取消，在完整的程序生命周期中，一个时间点只能存在一个文件输出位置.

        @params:
            filename - 输出日志的文件.
            mode - 默认为'w'，覆盖写模式，如果希望使用附加模式则改为 'a'.
        """
        fh = logging.FileHandler(filename, mode=mode, encoding='utf8')
        if self.file_handler is not None:
            raise RuntimeError('adding multi file_handler on global logger.')
        self.file_handler = fh
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def remove_file_handler(self):
        """
        停止向之前配置的日志文件输出日志。
        """
        try:
            if self.file_handler is not None:
                self.logger.removeHandler(self.file_handler)
            self.file_handler = None
        except Exception:
            pass

    @staticmethod
    def level_color_activate(enable=True):
        """
        将log的level设置为彩色。通过 ``level_color_activate(False)`` 取消。

        :param enable: 默认True
        """
        color_cfg['LEVEL_CHAR_COLOR'] = enable

    @staticmethod
    def msg_color_activate(enable=True):
        """
        将log的信息设置为彩色。通过 ``msg_color_activate(False)`` 取消。

        :param enable: 默认True
        """
        color_cfg['MSG_COLOR'] = enable


def get_logger():
    return Log()


_srcfile = [os.path.normcase(f) for f in [addLevelName.__code__.co_filename, get_logger.__code__.co_filename]]


class HackLoggingLogger(Logger):

    def findCaller(self, stack_info=False):
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.
        """
        f = currentframe()
        # On some versions of IronPython, currentframe() returns None if
        # IronPython isn't run with -X:Frames.
        if f is not None:
            f = f.f_back
        rv = "(unknown file)", 0, "(unknown function)", None
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename in _srcfile:
                f = f.f_back
                continue
            sinfo = None
            if stack_info:
                sio = io.StringIO()
                sio.write('Stack (most recent call last):\n')
                traceback.print_stack(f, file=sio)
                sinfo = sio.getvalue()
                if sinfo[-1] == '\n':
                    sinfo = sinfo[:-1]
                sio.close()
            rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
            break
        return rv


class HackLoggingLogRecord(LogRecord):
    def __init__(self, name, level, pathname, lineno, msg, args, exc_info, func=None, sinfo=None, **kwargs):
        super().__init__(name, level, pathname, lineno, msg, args, exc_info, func, sinfo, **kwargs)
        level_name = self.levelname.upper()
        colored = format_msg_color(level_name, level_name[0]) if color_cfg['LEVEL_CHAR_COLOR'] else level_name[0]
        self.levelname_first_char_with_color = colored


setLogRecordFactory(HackLoggingLogRecord)
setLoggerClass(HackLoggingLogger)

logger = get_logger()
