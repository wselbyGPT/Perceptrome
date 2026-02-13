from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication


COLORS = {
    "accent": "#50a0ff",
    "accent_hover": "#6fb3ff",
    "accent_pressed": "#3b86db",
    "window": "#1e1e1e",
    "surface_0": "#141414",
    "surface_1": "#202020",
    "surface_2": "#2b2b2b",
    "surface_3": "#333333",
    "border": "#3a3a3a",
    "border_muted": "#2c2c2c",
    "text_primary": "#e6e6e6",
    "text_secondary": "#dcdcdc",
    "text_muted": "#a6a6a6",
    "text_disabled": "#777777",
}

SPACING = {
    "xs": "2px",
    "sm": "4px",
    "md": "6px",
    "lg": "8px",
    "xl": "12px",
}

RADIUS = {
    "sm": "4px",
    "md": "6px",
    "lg": "8px",
}


def _build_dark_stylesheet() -> str:
    c = COLORS
    s = SPACING
    r = RADIUS
    return f"""
        QMainWindow, QWidget {{ background: {c['window']}; color: {c['text_primary']}; }}
        QLabel {{ color: {c['text_primary']}; }}
        QLabel:disabled {{ color: {c['text_disabled']}; }}

        QGroupBox {{
            margin-top: {s['xl']};
            padding: {s['lg']};
            border: 1px solid {c['border']};
            border-radius: {r['lg']};
            background: {c['surface_1']};
            color: {c['text_primary']};
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: {s['lg']};
            padding: 0 {s['sm']};
            color: {c['text_secondary']};
            background: {c['window']};
        }}

        QTabWidget::pane {{ border: 1px solid {c['border']}; top: -1px; background: {c['window']}; }}
        QTabBar::tab {{
            background: {c['surface_2']}; color: {c['text_secondary']}; padding: {s['lg']} {s['xl']};
            border: 1px solid {c['border']}; border-bottom: none; margin-right: {s['xs']};
            border-top-left-radius: {r['md']}; border-top-right-radius: {r['md']};
        }}
        QTabBar::tab:selected {{ background: {c['window']}; color: {c['text_primary']}; }}
        QTabBar::tab:hover {{ background: {c['surface_3']}; }}

        QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QTableWidget {{
            background: {c['surface_0']};
            border: 1px solid {c['border']};
            color: {c['text_primary']};
            border-radius: {r['md']};
        }}
        QLineEdit, QPlainTextEdit {{ padding: {s['md']}; }}
        QSpinBox, QDoubleSpinBox {{ padding: {s['sm']}; }}

        QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTableWidget:focus {{
            border: 1px solid {c['accent']};
        }}

        QLineEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
        QTableWidget:disabled, QHeaderView::section:disabled {{
            color: {c['text_disabled']};
            background: {c['surface_1']};
            border-color: {c['border_muted']};
        }}

        QPushButton {{
            background: {c['surface_2']}; border: 1px solid {c['border']};
            padding: {s['lg']} {s['xl']}; border-radius: {r['lg']}; color: {c['text_primary']};
        }}
        QPushButton:hover {{ background: {c['surface_3']}; }}
        QPushButton:pressed {{ background: {c['surface_1']}; }}
        QPushButton:focus {{ border: 1px solid {c['accent']}; }}
        QPushButton:disabled {{ color: {c['text_disabled']}; background: {c['surface_1']}; border: 1px solid {c['border_muted']}; }}

        QProgressBar {{
            border: 1px solid {c['border']}; border-radius: {r['lg']}; text-align: center;
            background: {c['surface_0']}; padding: {s['xs']}; color: {c['text_primary']};
        }}
        QProgressBar::chunk {{ background: {c['accent']}; border-radius: {r['md']}; }}

        QTableWidget {{ gridline-color: {c['border_muted']}; selection-background-color: {c['accent']}; selection-color: #101010; }}
        QTableWidget::item:selected {{ background: {c['accent']}; color: #101010; }}
        QHeaderView::section {{
            background: {c['surface_2']}; color: {c['text_secondary']}; padding: {s['md']};
            border: 1px solid {c['border']};
        }}

        QToolTip {{
            background-color: {c['surface_2']};
            color: {c['text_primary']};
            border: 1px solid {c['border']};
            padding: {s['sm']};
            border-radius: {r['sm']};
        }}

        QScrollBar:vertical {{
            background: {c['surface_1']};
            width: 12px;
            margin: 0;
            border-radius: {r['sm']};
        }}
        QScrollBar::handle:vertical {{
            background: {c['surface_3']};
            min-height: 24px;
            border-radius: {r['sm']};
        }}
        QScrollBar::handle:vertical:hover {{ background: {c['accent']}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        QScrollBar:horizontal {{
            background: {c['surface_1']};
            height: 12px;
            margin: 0;
            border-radius: {r['sm']};
        }}
        QScrollBar::handle:horizontal {{
            background: {c['surface_3']};
            min-width: 24px;
            border-radius: {r['sm']};
        }}
        QScrollBar::handle:horizontal:hover {{ background: {c['accent']}; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}

        QLineEdit#monospaceInput, QPlainTextEdit#monospaceLog {{
            font-family: "Consolas", "Menlo", "Monaco", "Courier New", monospace;
        }}
    """

def apply_dark_mode(app: QApplication):
    app.setStyle("Fusion")

    p = QPalette()
    p.setColor(QPalette.Window, QColor(COLORS["window"]))
    p.setColor(QPalette.WindowText, QColor(COLORS["text_primary"]))
    p.setColor(QPalette.Base, QColor(COLORS["surface_0"]))
    p.setColor(QPalette.AlternateBase, QColor(COLORS["surface_2"]))
    p.setColor(QPalette.ToolTipBase, QColor(COLORS["surface_2"]))
    p.setColor(QPalette.ToolTipText, QColor(COLORS["text_primary"]))
    p.setColor(QPalette.Text, QColor(COLORS["text_primary"]))
    p.setColor(QPalette.Button, QColor(COLORS["surface_2"]))
    p.setColor(QPalette.ButtonText, QColor(COLORS["text_primary"]))
    p.setColor(QPalette.Link, QColor(COLORS["accent"]))
    p.setColor(QPalette.Highlight, QColor(COLORS["accent"]))
    p.setColor(QPalette.HighlightedText, QColor(10, 10, 10))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(COLORS["text_disabled"]))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(COLORS["text_disabled"]))
    app.setPalette(p)
    app.setStyleSheet(_build_dark_stylesheet())
