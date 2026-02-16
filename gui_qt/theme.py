from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication

def apply_dark_mode(app: QApplication):
    app.setStyle("Fusion")

    p = QPalette()
    p.setColor(QPalette.Window, QColor(16, 18, 26))
    p.setColor(QPalette.WindowText, QColor(236, 239, 244))
    p.setColor(QPalette.Base, QColor(12, 14, 20))
    p.setColor(QPalette.AlternateBase, QColor(20, 23, 32))
    p.setColor(QPalette.ToolTipBase, QColor(236, 239, 244))
    p.setColor(QPalette.ToolTipText, QColor(17, 24, 39))
    p.setColor(QPalette.Text, QColor(236, 239, 244))
    p.setColor(QPalette.Button, QColor(30, 33, 45))
    p.setColor(QPalette.ButtonText, QColor(236, 239, 244))
    p.setColor(QPalette.Link, QColor(96, 165, 250))
    p.setColor(QPalette.Highlight, QColor(59, 130, 246))
    p.setColor(QPalette.HighlightedText, QColor(8, 12, 22))
    app.setPalette(p)

    app.setStyleSheet("""
        QMainWindow { background: #10121a; }
        QWidget { color: #e5e7eb; }
        QLabel { color: #e5e7eb; }
        QLabel#Logo { color: #f9fafb; }

        QGroupBox {
            border: 1px solid #2a2f3b;
            border-radius: 10px;
            margin-top: 14px;
            padding: 12px;
            background: #151925;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: #93c5fd;
            font-weight: 600;
        }

        QTabWidget::pane {
            border: 1px solid #2a2f3b;
            top: -1px;
            background: #10121a;
            border-radius: 10px;
        }
        QTabBar::tab {
            background: #1b1f2b;
            color: #d1d5db;
            padding: 10px 16px;
            border: 1px solid #2a2f3b;
            border-bottom: none;
            margin-right: 4px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        }
        QTabBar::tab:selected {
            background: #10121a;
            color: #f3f4f6;
        }
        QTabBar::tab:hover { background: #23283a; }

        QLineEdit, QPlainTextEdit {
            background: #0f131b;
            border: 1px solid #2a2f3b;
            padding: 8px 10px;
            border-radius: 8px;
            selection-background-color: #2563eb;
        }
        QLineEdit:focus, QPlainTextEdit:focus {
            border: 1px solid #60a5fa;
            background: #111827;
        }
        QSpinBox, QDoubleSpinBox {
            background: #0f131b;
            border: 1px solid #2a2f3b;
            padding: 6px 8px;
            border-radius: 8px;
        }
        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #60a5fa;
            background: #111827;
        }

        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #2f3648, stop:1 #222939);
            border: 1px solid #364057;
            padding: 8px 16px;
            border-radius: 10px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #3a435a, stop:1 #283044);
            border: 1px solid #4b5a74;
        }
        QPushButton:pressed {
            background: #1f2434;
            border: 1px solid #2f394f;
        }
        QPushButton:disabled {
            color: #6b7280;
            background: #1a1f2e;
            border: 1px solid #262c3b;
        }

        QProgressBar {
            border: 1px solid #2a2f3b;
            border-radius: 10px;
            text-align: center;
            background: #0f131b;
            padding: 3px;
            color: #e5e7eb;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #3b82f6, stop:1 #22d3ee);
            border-radius: 8px;
        }

        QTableWidget {
            background: #0f131b;
            border: 1px solid #2a2f3b;
            gridline-color: #22283a;
            selection-background-color: #1d4ed8;
            selection-color: #f8fafc;
        }
        QHeaderView::section {
            background: #1b1f2b;
            color: #e5e7eb;
            padding: 8px;
            border: 1px solid #2a2f3b;
            font-weight: 600;
        }

        QScrollBar:vertical {
            background: #0f131b;
            width: 12px;
            margin: 4px 2px 4px 2px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background: #2b3347;
            min-height: 20px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical:hover { background: #3b4762; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar:horizontal {
            background: #0f131b;
            height: 12px;
            margin: 2px 4px 2px 4px;
            border-radius: 6px;
        }
        QScrollBar::handle:horizontal {
            background: #2b3347;
            min-width: 20px;
            border-radius: 6px;
        }
        QScrollBar::handle:horizontal:hover { background: #3b4762; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
    """)
