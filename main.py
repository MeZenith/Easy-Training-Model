import os
import sys

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from core.config import AppConfig
from setup_icon import set_window_icon, setup_app_icon
from ui.theme import ThemeManager
from utils.i18n import I18n
from utils.logger import setup_logger


#玄学
def buddha_bless():
    print(r"""
                              _ooOoo_
                             o8888888o
                             88" . "88
                             (| -_- |)
                             O\\  =  //O
                           ____/`---'\\____
                         .'  \\\\|     |//  `.
                        /  \\\\|||  :  |||//  \\
                       /  _||||| -:- |||||-  \\
                       |   | \\\\\\  -  /// |   |
                       | \\_|  ''\\---/''  |   |
                       \\  .-\\__  `-`  ___/-. /
                     ___`. .'  /--.--\\  `. . __
                  ."" '<  `.___\\_<|>_/___.'  >'"".
                 | | :  `- \\`.;`\\ _ /`;.`/ - ` : | |
                 \\  \\ `-.   \\_ __\\ /__ _/   .-` /  /
            ======`-.____`-.___\\_____/___.-`____.-'======
                               `=---='
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       佛祖保佑        永无BUG
              佛曰:
                     写字楼里写字间，写字间里程序员；
                     程序人员写程序，又拿程序换酒钱。
                     酒醒只在网上坐，酒醉还来网下眠；
                     酒醉酒醒日复日，网上网下年复年。
                     但愿老死电脑间，不愿鞠躬老板前；
                     奔驰宝马贵者趣，公交自行程序员。
                     别人笑我忒疯癫，我笑自己命太贱；
                     不见满街漂亮妹，哪个归得程序员？
""")


#加载配置
def load_config():
    config = AppConfig()
    return config

#初始化日志
def init_log(config):
    logger = setup_logger(config.workspace)
    logger.info("Easy Tinking starting...")
    return logger

#初始化国际化
def init_i18n(config):
    i18n = I18n.instance()
    lang = config.get("language", "")
    if not lang:
        lang = i18n.detect_system_language()
        config.set("language", lang)
    i18n.load_language(lang)
    return i18n

#创建Qt应用
def make_app():
    app = QApplication(sys.argv)
    app.setApplicationName("Easy Tinking")
    app.setOrganizationName("BlueCornerStudio")
    set_window_icon(app)
    return app

#加载主题
def load_theme(app, config):
    theme_manager = ThemeManager.instance()
    theme = config.get("theme", "dark")
    theme_manager.apply_theme(app, theme)

#创建主窗口
def make_window(config, i18n):
    from ui.app import MainWindow
    window = MainWindow(config, i18n)
    set_window_icon(window)
    return window


def main():
    buddha_bless()

    #设置任务栏图标
    setup_app_icon()

    config = load_config()
    logger = init_log(config)
    i18n = init_i18n(config)
    app = make_app()

    #启动图 — QApplication创建之后才能用
    splash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res", "splash.png")
    if os.path.isfile(splash_path):
        splash = QSplashScreen(QPixmap(splash_path))
        splash.show()
        #强制刷新，让图片立刻显示
        app.processEvents()

    #全局异常捕获
    from ui.app import global_exception_handler
    sys.excepthook = global_exception_handler

    load_theme(app, config)

    window = make_window(config, i18n)
    window.show()

    #主窗口出来了，关掉启动图
    if os.path.isfile(splash_path):
        splash.finish(window)

    logger.info("Easy Tinking started successfully")
    exit_code = app.exec()
    logger.info("Easy Tinking shutting down...")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
