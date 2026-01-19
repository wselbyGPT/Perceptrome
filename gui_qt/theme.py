from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication

def apply_dark_mode(app: QApplication):
    app.setStyle("Fusion")

    p = QPalette()
    p.setColor(QPalette.Window, QColor(30, 30, 30))
    p.setColor(QPalette.WindowText, QColor(230, 230, 230))
    p.setColor(QPalette.Base, QColor(20, 20, 20))
    p.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
    p.setColor(QPalette.ToolTipBase, QColor(230, 230, 230))
    p.setColor(QPalette.ToolTipText, QColor(20, 20, 20))
    p.setColor(QPalette.Text, QColor(230, 230, 230))
    p.setColor(QPalette.Button, QColor(45, 45, 45))
    p.setColor(QPalette.ButtonText, QColor(230, 230, 230))
    p.setColor(QPalette.Link, QColor(80, 160, 255))
    p.setColor(QPalette.Highlight, QColor(80, 160, 255))
    p.setColor(QPalette.HighlightedText, QColor(10, 10, 10))
    app.setPalette(p)

    app.setStyleSheet("""
        QMainWindow { background: #1e1e1e; }
        QLabel { color: #e6e6e6; }

        QTabWidget::pane { border: 1px solid #3a3a3a; top: -1px; background: #1e1e1e; }
        QTabBar::tab {
            background: #2b2b2b; color: #dcdcdc; padding: 8px 14px;
            border: 1px solid #3a3a3a; border-bottom: none; margin-right: 2px;
            border-top-left-radius: 6px; border-top-right-radius: 6px;
        }
        QTabBar::tab:selected { background: #1e1e1e; }
        QTabBar::tab:hover { background: #333333; }

        QLineEdit, QPlainTextEdit {
            background: #141414; border: 1px solid #3a3a3a;
            padding: 6px; border-radius: 6px;
        }
        QSpinBox, QDoubleSpinBox {
            background: #141414; border: 1px solid #3a3a3a;
            padding: 4px; border-radius: 6px;
        }

        QPushButton {
            background: #2b2b2b; border: 1px solid #3a3a3a;
            padding: 8px 12px; border-radius: 8px;
        }
        QPushButton:hover { background: #333333; }
        QPushButton:pressed { background: #222222; }
        QPushButton:disabled { color: #777; background: #222; border: 1px solid #2c2c2c; }

        QProgressBar {
            border: 1px solid #3a3a3a; border-radius: 8px; text-align: center;
            background: #141414; padding: 2px;
        }
        QProgressBar::chunk { background: #50a0ff; border-radius: 6px; }

        QTableWidget { background: #141414; border: 1px solid #3a3a3a; gridline-color: #2c2c2c; }
        QHeaderView::section {
            background: #2b2b2b; color: #dcdcdc; padding: 6px;
            border: 1px solid #3a3a3a;
        }
    """)
