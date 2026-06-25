from PySide6.QtWidgets import QMessageBox

from core.error_handler import friendly_error_message


def show_error(parent, title: str, message: str):
    #显示错误弹窗
    QMessageBox.critical(parent, title, message)


def show_warning(parent, title: str, message: str):
    #显示警告弹窗
    QMessageBox.warning(parent, title, message)


def show_exception_dialog(parent, exc: Exception, i18n_func=None, error_title="Error"):
    #分类异常 + 弹窗
    msg = friendly_error_message(exc, i18n_func)
    QMessageBox.critical(parent, error_title, msg)
