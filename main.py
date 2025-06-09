# main.py

import sys
from PyQt5.QtWidgets import QApplication
# 导入 app 模块和我们新的 shared 模块
from app import MainWindow, set_dark_style
import shared

if __name__ == '__main__':
    app = QApplication(sys.argv)
    set_dark_style(app)

    # 创建主窗口实例
    main_win_instance = MainWindow()

    # !!! 关键步骤：将创建好的实例赋值给共享变量 !!!
    shared.main_window = main_win_instance

    # 显示窗口
    shared.main_window.show()

    sys.exit(app.exec_())