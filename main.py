import json
import logging
import os

import subprocess
import sys
import time
import toml
import numpy as np
from pathlib import Path
import onnxruntime  # workaround: Pre-import to avoid ImportError: DLL load failed while importing onnxruntime_pybind11_state: 动态链接库(DLL)初始化例程失败。
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtWidgets import QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QButtonGroup
from PyQt6.QtWidgets import QGroupBox, QMessageBox, QGraphicsDropShadowEffect, QFrame
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QFont, QIcon, QPainter, QColor
import PyQt6.QtCore as QtCore

import loadData
import auto_fetch
import similar_history_match
import recognize
from recognize import MONSTER_COUNT
from specialmonster import SpecialMonsterHandler
import winrt_capture
from config import FIELD_FEATURE_COUNT, MONSTER_DATA
from simular_history_match_ui import HistoryMatchUI
from input_panel_ui import InputPanelUI

logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("PIL").setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
stream_handler.setFormatter(formatter)
logging.getLogger().addHandler(stream_handler)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


try:
    from predict import CannotModel
    from train import UnitAwareTransformer

    logger.info("Using PyTorch model for predictions.")
except:
    from predict_onnx import CannotModel

    logger.info("Using ONNX model for predictions.")


class ADBConnectorThread(QThread):
    """
    Worker thread to run loadData.AdbConnector.connect() without blocking the UI.
    """

    connect_finished = pyqtSignal()

    def __init__(self, app: "ArknightsApp"):
        super().__init__()
        self.app = app

    def run(self):
        self.app.adb_connector.connect()
        self.connect_finished.emit()


class ArknightsApp(QMainWindow):
    # 添加自定义信号
    update_button_signal = pyqtSignal(str)  # 用于更新按钮文本
    update_monster_signal = pyqtSignal(list)
    update_prediction_signal = pyqtSignal(float)
    update_statistics_signal = pyqtSignal()  # 用于更新统计信息
    auto_collect_status_signal = pyqtSignal(str)   # 自动收集进度信号（安全跨线程）
    auto_collect_complete_signal = pyqtSignal(bool, str)  # 自动收集完成信号
    train_status_signal = pyqtSignal(str, bool)  # 训练进度信号（文本, 是否最终状态）
    qt_button_style = """
        QPushButton {
            background-color: #313131;
            color: #F3F31F;
            border-radius: 16px;
            padding: 8px;
            font-weight: bold;
            min-height: 30px;
        }
        QPushButton:hover {
            background-color: #414141;
        }
        QPushButton:pressed {
            background-color: #212121;
        }
    """

    def __init__(self):
        super().__init__()
        # 捕获模式：ADB, PC, WIN
        self.current_capture_mode = "ADB"

        # 尝试连接模拟器
        self.adb_connector = loadData.AdbConnector()
        self.pc_connector = loadData.PcConnector()
        self.adb_connector_thread = ADBConnectorThread(self)
        self.adb_connector_thread.connect_finished.connect(self.on_adb_connected)
        self.adb_connector_thread.start()

        self.auto_fetch_running = False
        self.is_invest = False
        self.game_mode = "单人"
        self._session_loading = False

        # 模型：优先 PyTorch，失败则回退 ONNX
        self.cannot_model = CannotModel()
        if not self.cannot_model.is_model_loaded:
            logger.info("PyTorch 模型未加载，尝试 ONNX...")
            try:
                import predict_onnx
                session = self._load_session_config().get("session_name", "")
                onnx_path = predict_onnx.resolve_model_path(session)
                # 如果 ONNX 不存在但 .backup 存在，自动恢复备份
                if not Path(onnx_path).exists():
                    backup_path = Path(onnx_path + ".backup")
                    if backup_path.exists():
                        logger.info(f"检测到备份模型，正在恢复: {backup_path}")
                        backup_path.rename(onnx_path)
                        data_backup = Path(onnx_path + ".data.backup")
                        if data_backup.exists():
                            data_backup.rename(onnx_path + ".data")
                self.cannot_model = predict_onnx.CannotModel(onnx_path)
                if self.cannot_model.is_model_loaded:
                    logger.info(f"ONNX 模型加载成功: {onnx_path}")
                else:
                    logger.warning("ONNX 模型也未加载")
            except Exception as e:
                logger.warning(f"ONNX 回退失败: {e}")

        # 怪物识别模块
        self.recognizer = recognize.RecognizeMonster(method="ADB")

        # 初始化UI后加载历史数据
        logger.info("尝试获取错题本")
        self.history_match = None
        session_for_history = self._load_session_config().get("session_name", "")
        self.history_match = similar_history_match.HistoryMatch(session_name=session_for_history)
        # Ensure feat_past and N_history are initialized
        try:
            self.history_match.feat_past = np.hstack([self.history_match.past_left, self.history_match.past_right])
        except Exception:
            self.history_match.feat_past = None
        self.history_match.N_history = 0 if self.history_match.labels is None else len(self.history_match.labels)
        logger.info("错题本加载成功")

        # 初始化特殊怪物语言触发处理程序
        self.special_monster_handler = SpecialMonsterHandler()

        self.init_ui()

        # 如果模型未加载，显示提示并禁用预测相关按钮
        if not self.cannot_model.is_model_loaded:
            self.recognize_button.setEnabled(False)
            self.recognize_button.setToolTip("模型未加载，无法使用此功能")
            self.input_panel.predict_button.setEnabled(False)
            self.input_panel.predict_button.setToolTip("模型未加载，无法使用此功能")

    def init_ui(self):
        try:
            with open("pyproject.toml", "r", encoding="utf-8") as f:
                pyproject_data = toml.load(f)
                version = pyproject_data["project"]["version"]
        except (FileNotFoundError, KeyError):
            version = "unknown"
        model_name = Path(self.cannot_model.model_path).name if self.cannot_model.model_path else "未加载"
        self.setWindowTitle(f"铁鲨鱼_Arknights Neural Network - v{version} - model: {model_name}")
        self.setWindowIcon(QIcon("ico/icon.ico"))
        self.setGeometry(100, 100, 500, 580)
        self.setMinimumWidth(580)
        self.setMaximumWidth(580)
        self.background = QPixmap("ico/background.png")

        # 初始化动画对象
        self.size_animation = QPropertyAnimation(self, b"size")
        self.size_animation.setDuration(300)
        self.size_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # 左侧面板
        self.input_panel = InputPanelUI()
        self.input_panel.setFixedWidth(528)
        self.input_panel.predict_requested.connect(self.predict)
        self.input_panel.reset_requested.connect(self.reset_entries)
        self.input_panel.input_changed.connect(self.update_input_display)
        self.input_panel.terrain_changed.connect(self.predict)

        # 中央面板 - 结果和控制区
        center_panel = QWidget()
        center_panel.setFixedWidth(550)  # 固定右侧面板宽度
        center_layout = QVBoxLayout(center_panel)

        # 顶部区域 - 输入显示
        input_display = QGroupBox()
        input_display.setStyleSheet(
            """
                QGroupBox {
                    background-color: rgba(0, 0, 0, 120);
                    border-radius: 15px;
                    border: 5px solid #F5EA2D;
                    margin-top: 10px;
                    padding: 10px 0;
                }
                QGroupBox::title {
                    color: white;
                    subcontrol-origin: margin;
                    left: 15px;
                    padding: 0 5px;
                }
            """
        )
        input_layout = QHBoxLayout(input_display)

        # 左侧人物显示
        left_input_group = QWidget()
        left_input_layout = QHBoxLayout(left_input_group)
        self.left_input_content = QWidget()
        self.left_input_layout = QHBoxLayout(self.left_input_content)
        self.left_input_layout.setSpacing(5)
        left_input_layout.addWidget(self.left_input_content)

        # 右侧人物显示
        right_input_group = QWidget()
        right_input_layout = QHBoxLayout(right_input_group)
        self.right_input_content = QWidget()
        self.right_input_layout = QHBoxLayout(self.right_input_content)
        self.right_input_layout.setSpacing(5)
        right_input_layout.addWidget(self.right_input_content)

        # 将左右两部分添加到主输入布局
        input_layout.addWidget(left_input_group)
        input_layout.addWidget(right_input_group)

        center_layout.addWidget(input_display)

        # 中部区域 - 预测结果
        result_group = QGroupBox()
        result_group.setStyleSheet(
            """
            QGroupBox {
                background-color: rgba(120, 120, 120, 10);
                border-radius: 15px;
                border: 1px solid #747474;
            }
            """
        )
        result_layout = QVBoxLayout(result_group)
        result_layout.setSpacing(10)
        result_layout.setContentsMargins(10, 10, 10, 10)

        self.result_label = QLabel("预测结果将显示在这里")
        self.result_label.setFont(QFont("Microsoft YaHei", 12))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(self.result_label)

        # 添加模型名称显示
        model_name = Path(self.cannot_model.model_path).name if self.cannot_model.model_path else "未加载"
        self.model_name_label = QLabel(f"model: {model_name}")
        self.model_name_label.setFont(QFont("Microsoft YaHei", 8))
        self.model_name_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        self.model_name_label.setStyleSheet("color: #888888;")  # 小字灰色
        result_layout.addWidget(self.model_name_label)

        # 第二行按钮result_identify_group
        result_identify_group = QWidget()
        result_identify_layout = QHBoxLayout(result_identify_group)

        self.recognize_button = QPushButton("识别并预测")
        self.recognize_button.clicked.connect(self.recognize_and_predict)
        self.recognize_button.setStyleSheet(self.qt_button_style)
        result_identify_layout.addWidget(self.recognize_button)

        self.recognize_only_button = QPushButton("仅识别")
        self.recognize_only_button.clicked.connect(self.recognize_only)
        self.recognize_only_button.setStyleSheet(self.qt_button_style)
        self.recognize_only_button.setFixedWidth(80)  # 小按钮
        result_identify_layout.addWidget(self.recognize_only_button)

        result_layout.addWidget(result_identify_group)

        center_layout.addWidget(result_group)

        # 底部区域 - 控制面板和连接设置
        self.bottom_group = QWidget()
        self.bottom_layout = QHBoxLayout(self.bottom_group)

        # 左侧垂直布局：控制面板 + 连接设置
        left_column = QVBoxLayout()

        # 深色主题样式(适配暗色模式)
        dark_group_box_style = """
            QGroupBox {
                background-color: rgba(0, 0, 0, 120);
                border-radius: 8px;
                border: 1px solid #555555;
                margin-top: 10px;
                padding: 10px 5px;
                color: #E0E0E0;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #E0E0E0;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """

        # --- 控制面板 ---
        control_group = QGroupBox("控制面板")
        control_group.setStyleSheet(dark_group_box_style)
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(4)

        # 第零行 - 训练时长 + 会话名
        row0 = QWidget()
        row0_layout = QHBoxLayout(row0)
        row0_layout.setContentsMargins(0, 0, 0, 0)
        row0_layout.setSpacing(8)

        self.duration_label = QLabel("训练时长(h):")
        self.duration_entry = QLineEdit("325")
        self.duration_entry.setFixedWidth(45)

        self.session_name_label = QLabel("会话名:")
        self.session_name_entry = QLineEdit("")
        self.session_name_entry.setFixedWidth(130)
        self.session_name_entry.setPlaceholderText("留空=按时间戳")
        self.session_name_entry.setToolTip(
            "同一会话名的数据和模型会累积复用。\n不同会话名完全隔离。\n留空则每次使用不同时间戳目录。"
        )

        # 加载缓存的会话名
        cached = self._load_session_config()
        self._session_loading = True
        if cached["session_name"]:
            self.session_name_entry.setText(cached["session_name"])
        self._session_loading = False
        self.session_name_entry.textChanged.connect(self._on_session_name_changed)

        row0_layout.addWidget(self.duration_label)
        row0_layout.addWidget(self.duration_entry)
        row0_layout.addWidget(self.session_name_label)
        row0_layout.addWidget(self.session_name_entry)
        row0_layout.addStretch()

        # 单独一行 - 训练设备选择
        row_dev = QWidget()
        row_dev_layout = QHBoxLayout(row_dev)
        row_dev_layout.setContentsMargins(0, 0, 0, 0)
        row_dev_layout.setSpacing(8)

        self.device_label = QLabel("训练设备:")
        self.device_menu = QComboBox()
        self.device_menu.addItem("自动检测", "")
        self.device_menu.addItem("CPU", "cpu")
        try:
            import torch
            if torch.cuda.is_available():
                self.device_menu.addItem("CUDA", "cuda")
        except ImportError:
            pass
        self.device_menu.setToolTip(
            "选择训练设备。\n自动检测: 优先GPU。\nCPU: 仅使用CPU。\nCUDA: 强制使用NVIDIA GPU。"
        )
        self.device_menu.setFixedWidth(100)

        if cached["device_type"]:
            idx = self.device_menu.findData(cached["device_type"])
            if idx >= 0:
                self.device_menu.setCurrentIndex(idx)
        self.device_menu.currentIndexChanged.connect(self._on_device_changed)

        row_dev_layout.addWidget(self.device_label)
        row_dev_layout.addWidget(self.device_menu)
        row_dev_layout.addStretch()

        # 第一行 - 游戏操作 + 统计
        row1 = QWidget()
        row1_layout = QHBoxLayout(row1)
        row1_layout.setContentsMargins(0, 0, 0, 0)
        row1_layout.setSpacing(8)

        self.auto_fetch_button = QPushButton("自动获取数据")
        self.auto_fetch_button.clicked.connect(self.toggle_auto_fetch)
        self.auto_fetch_button.setStyleSheet(self.qt_button_style)
        self.auto_fetch_button.setFixedWidth(120)

        self.mode_menu = QComboBox()
        self.mode_menu.addItems(["单人", "30人"])
        self.mode_menu.currentTextChanged.connect(self.update_game_mode)
        self.mode_menu.setFixedWidth(70)

        self.invest_checkbox = QCheckBox("投资")
        self.invest_checkbox.stateChanged.connect(self.update_invest_status)

        self.stats_label = QLabel()
        self.stats_label.setFont(QFont("Microsoft YaHei", 10))

        row1_layout.addWidget(self.auto_fetch_button)
        row1_layout.addWidget(self.mode_menu)
        row1_layout.addWidget(self.invest_checkbox)
        row1_layout.addWidget(self.stats_label)
        row1_layout.addStretch()

        # 第二行 - 训练 ONNX 模型
        row2 = QWidget()
        row2_layout = QHBoxLayout(row2)
        row2_layout.setContentsMargins(0, 0, 0, 0)
        row2_layout.setSpacing(8)

        self.train_onnx_button = QPushButton("🧠 训练ONNX模型")
        self.train_onnx_button.clicked.connect(self.train_onnx_model)
        self.train_onnx_button.setStyleSheet(self.qt_button_style)
        self.train_onnx_button.setFixedWidth(140)
        row2_layout.addWidget(self.train_onnx_button)

        self.train_status_label = QLabel("")
        self.train_status_label.setFont(QFont("Microsoft YaHei", 9))
        self.train_status_label.setStyleSheet("color: #888888;")
        row2_layout.addWidget(self.train_status_label)

        # 第三行 - 一键数据收集和训练
        row3 = QWidget()
        row3_layout = QHBoxLayout(row3)
        row3_layout.setContentsMargins(0, 0, 0, 0)
        row3_layout.setSpacing(8)

        self.auto_collect_train_button = QPushButton("🔄 一键收集并训练")
        self.auto_collect_train_button.clicked.connect(self.start_auto_collect_and_train)
        self.auto_collect_train_button.setStyleSheet(
            self.qt_button_style + """
            QPushButton {
                background-color: #2E7D32;
                color: white;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            """
        )
        self.auto_collect_train_button.setToolTip(
            "一键完成：自动收集数据（追加到当前会话）→ 到达时长后停止 → 基于历史模型微调 → 导出新ONNX"
        )
        self.auto_collect_train_button.setFixedWidth(230)
        row3_layout.addWidget(self.auto_collect_train_button)

        self.stop_auto_collect_button = QPushButton("⏹️ 停止")
        self.stop_auto_collect_button.clicked.connect(self.stop_auto_collect_and_train)
        self.stop_auto_collect_button.setEnabled(False)
        self.stop_auto_collect_button.setStyleSheet(
            self.qt_button_style + """
            QPushButton {
                background-color: #C62828;
                color: white;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            """
        )
        self.stop_auto_collect_button.setToolTip("停止当前的自动数据收集流程")
        self.stop_auto_collect_button.setFixedWidth(70)
        row3_layout.addWidget(self.stop_auto_collect_button)

        self.auto_collect_status_label = QLabel("")
        self.auto_collect_status_label.setFont(QFont("Microsoft YaHei", 9))
        self.auto_collect_status_label.setStyleSheet("color: #4CAF50;")
        row3_layout.addWidget(self.auto_collect_status_label)

        # GitHub链接
        github_label = QLabel(
            '<a href="https://github.com/Ancientea/CannotMax" style="color: #2196F3; text-decoration: none;">https://github.com/Ancientea/CannotMax</a>'
        )
        github_label.setMargin(0)
        github_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        github_label.setOpenExternalLinks(True)
        github_label.setFont(QFont("Microsoft YaHei", 9))
        github_label.setContentsMargins(0, 0, 0, 0)

        # 添加到控制布局
        control_layout.addWidget(row0)
        control_layout.addWidget(row_dev)
        control_layout.addWidget(row1)
        control_layout.addWidget(row2)
        control_layout.addWidget(row3)
        control_layout.addWidget(github_label)

        # --- 连接设置 ---
        connection_group = QGroupBox("连接设置")
        connection_group.setStyleSheet(dark_group_box_style)
        connection_layout = QVBoxLayout(connection_group)

        # 模式选择行
        mode_row = QWidget()
        mode_row_layout = QHBoxLayout(mode_row)
        mode_row_layout.setContentsMargins(0, 0, 0, 0)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)

        self.adb_mode_btn = QPushButton("安卓端-ADB")
        self.pc_mode_btn = QPushButton("PC端(UI比例100%)")
        self.win_mode_btn = QPushButton("窗口截取")

        mode_btns = [self.adb_mode_btn, self.pc_mode_btn, self.win_mode_btn]
        mode_style = """
            QPushButton {
                background-color: #313131;
                color: #F3F31F;
                border-radius: 10px;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #414141;
            }
            QPushButton:checked {
                background-color: #F5EA2D;
                color: #313131;
            }
        """
        for btn in mode_btns:
            btn.setCheckable(True)
            btn.setStyleSheet(mode_style)
            self.mode_group.addButton(btn)
            mode_row_layout.addWidget(btn)

        self.adb_mode_btn.setChecked(True)
        self.adb_mode_btn.clicked.connect(lambda: self.on_mode_changed("ADB"))
        self.pc_mode_btn.clicked.connect(lambda: self.on_mode_changed("PC"))
        self.win_mode_btn.clicked.connect(lambda: self.on_mode_changed("WIN"))
        connection_layout.addWidget(mode_row)

        # 序列号行
        conn_row1 = QWidget()
        conn_row1_layout = QHBoxLayout(conn_row1)
        conn_row1_layout.setContentsMargins(0, 0, 0, 0)

        self.serial_label = QLabel("模拟器序列号:")
        self.serial_entry = QComboBox()
        self.serial_entry.setEditable(True)
        self.serial_entry.setFixedWidth(150)
        self.serial_entry.lineEdit().setPlaceholderText("127.0.0.1:5555")

        self.serial_button = QPushButton("更新")
        self.serial_button.clicked.connect(self.update_device_serial)

        conn_row1_layout.addWidget(self.serial_label)
        conn_row1_layout.addWidget(self.serial_entry)
        conn_row1_layout.addWidget(self.serial_button)

        # 捕获设置行
        conn_row2 = QWidget()
        conn_row2_layout = QHBoxLayout(conn_row2)
        conn_row2_layout.setContentsMargins(0, 0, 0, 0)

        self.choose_window_button = QPushButton("选择截屏窗口")
        self.choose_window_button.clicked.connect(self.choose_capture_window)
        self.reselect_button = QPushButton("选择范围")
        self.reselect_button.clicked.connect(self.reselect_roi)

        # 初始为 ADB 模式，禁用窗口捕获相关按钮
        self.choose_window_button.setEnabled(False)
        self.reselect_button.setEnabled(False)

        conn_row2_layout.addWidget(self.choose_window_button)
        conn_row2_layout.addWidget(self.reselect_button)

        connection_layout.addWidget(conn_row1)
        connection_layout.addWidget(conn_row2)

        # 将两个组框添加到左侧列
        left_column.addWidget(control_group)
        left_column.addWidget(connection_group)

        # 第五行按钮 (纵向排列的功能按钮)
        row5 = QWidget()
        row5_layout = QVBoxLayout(row5)

        self.simulate_button = QPushButton("显示沙盒模拟")
        self.simulate_button.clicked.connect(self.run_simulation)
        self.simulate_button.setStyleSheet(self.qt_button_style)
        row5_layout.addWidget(self.simulate_button)

        # 在右侧面板添加显示输入面板按钮
        self.toggle_input_button = QPushButton("显示输入面板")
        self.toggle_input_button.clicked.connect(self.toggle_input_panel)
        self.toggle_input_button.setStyleSheet(self.qt_button_style)
        row5_layout.addWidget(self.toggle_input_button)

        # 在右侧面板添加历史对局按钮
        self.history_button = QPushButton("显示历史对局")
        self.history_button.clicked.connect(self.toggle_history_panel)
        self.history_button.setStyleSheet(self.qt_button_style)
        row5_layout.addWidget(self.history_button)

        # 窗口置顶按钮
        self.always_on_top_button = QPushButton("窗口置顶")
        self.always_on_top_button.clicked.connect(self.toggle_always_on_top)
        self.always_on_top_button.setStyleSheet(self.qt_button_style)
        row5_layout.addWidget(self.always_on_top_button)

        # 排布底部布局
        self.bottom_layout.addLayout(left_column)
        self.bottom_layout.addWidget(row5)

        center_layout.addWidget(self.bottom_group)

        # 创建并添加HistoryMatchUI实例
        self.history_match_ui = HistoryMatchUI(self.history_match)
        self.history_match_ui.setVisible(False)  # 初始隐藏

        main_layout.addWidget(center_panel, 1)
        main_layout.addWidget(self.input_panel)
        main_layout.addWidget(self.history_match_ui)  # 添加到主布局

        self.setCentralWidget(main_widget)
        # 初始化输入面板状态
        self.input_panel_visible = False
        self.input_panel.setVisible(False)  # 默认折叠左侧输入面板

        # 连接AutoFetch信号到槽
        self.update_button_signal.connect(self.auto_fetch_button.setText)
        self.update_monster_signal.connect(self.update_monster)
        self.update_prediction_signal.connect(self.update_prediction)
        self.update_statistics_signal.connect(self.update_statistics)
        # 连接自动收集信号（保证GUI操作在主线程执行）
        self.auto_collect_status_signal.connect(self._on_auto_collect_status)
        self.auto_collect_complete_signal.connect(self.on_auto_collect_complete)
        self.train_status_signal.connect(self._on_train_status)
        self.refresh_device_list()

    def toggle_input_panel(self):
        """切换输入面板的显示"""
        target_width = self.width()
        is_visible = self.input_panel.isVisible()
        self.input_panel.setVisible(not is_visible)
        if not is_visible:
            self.toggle_input_button.setText("隐藏输入面板")
            target_width += self.input_panel.width()
        else:
            self.toggle_input_button.setText("显示输入面板")
            target_width -= self.input_panel.width()
        self.animate_size_change(target_width)

    def animate_size_change(self, target_width, target_height=None):
        """通用的尺寸动画方法"""
        if target_height is None:
            target_height = self.height()
        if self.size_animation.state() == QPropertyAnimation.State.Running:
            self.size_animation.stop()

        self.setMinimumWidth(min(self.width(), target_width))
        self.setMaximumWidth(max(self.width(), target_width))

        self.size_animation.setStartValue(self.size())
        self.size_animation.setEndValue(QtCore.QSize(target_width, target_height))
        self.size_animation.start()

        def set_fixed_after_animation():
            self.setFixedWidth(self.width())

        self.size_animation.finished.connect(set_fixed_after_animation)

    @property
    def active_connector(self):
        if self.current_capture_mode == "PC":
            return self.pc_connector
        return self.adb_connector

    def on_mode_changed(self, mode):
        """切换捕获模式"""
        self.current_capture_mode = mode
        logger.info(f"切换捕获模式为: {mode}")

        is_win_mode = (mode == "WIN")
        is_adb_mode = (mode == "ADB")
        is_pc_mode = (mode == "PC")

        # 切换窗口捕获相关控件
        self.choose_window_button.setEnabled(is_win_mode)
        self.reselect_button.setEnabled(is_win_mode)
        
        # 切换 ADB 相关控件
        self.serial_label.setEnabled(is_adb_mode)
        self.serial_entry.setEnabled(is_adb_mode)
        self.serial_button.setEnabled(is_adb_mode)

        if mode == "ADB":
            self.refresh_device_list()
            self.recognizer = recognize.RecognizeMonster(method="ADB")
            if not self.adb_connector.device_serial:
                self.adb_connector_thread.start()
        elif mode == "WIN":
            if self.recognizer.method != "WIN":
                self.recognizer = recognize.RecognizeMonster(method="WIN")
            if self.recognizer._winrt is None:
                self.choose_capture_window()
        elif mode == "PC":
            self.recognizer = recognize.RecognizeMonster(method="ADB") # reuse ADB reading methodology but on PC Connector
            if not self.pc_connector.is_connected:
                self.pc_connector.connect()
                if not self.pc_connector.is_connected:
                    QMessageBox.warning(self, "警告", "未能连接到PC端窗口(明日方舟)。")

    def on_adb_connected(self):
        logger.info("模拟器初始化完成")

    def choose_capture_window(self):
        """弹出窗口选择器，切换 WinRT 截屏源（窗口标题或整屏）。"""
        import traceback, cv2

        if getattr(self, "_switching_source", False):
            return
        self._switching_source = True
        self.choose_window_button.setEnabled(False)
        try:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            dlg = winrt_capture.WindowPickerDialog(self)
            if dlg.exec():
                sel = dlg.get_selection()
                logger.info(f"选择了截屏源: {sel}")
                if not sel:
                    QMessageBox.information(self, "提示", "未选择任何项")
                    return
                hint = ""
                if "window_name" in sel:
                    self.recognizer = recognize.RecognizeMonster(method="WIN", window_name=sel["window_name"], monitor_index=None)
                    hint = f"已切换至窗口：{sel['window_name']}"
                else:
                    idx = max(1, sel["monitor_index"])
                    self.recognizer = recognize.RecognizeMonster(method="WIN", window_name=None, monitor_index=idx)
                    hint = f"已切换至整屏：显示器 {sel['monitor_index']}"

                self.no_region = True
                QMessageBox.information(self, "成功", hint + "\n建议重新选择范围。")
        except Exception as e:
            QMessageBox.critical(self, "异常", f"{e}\n\n{traceback.format_exc()}")
        finally:
            self._switching_source = False
            self.choose_window_button.setEnabled(self.current_capture_mode == "WIN")

    def paintEvent(self, event):
        painter = QPainter(self)
        # 深色底色（适配暗色模式，防止白色文字在浅色背景上不可见）
        painter.fillRect(self.rect(), QColor("#1e1e1e"))
        # 缩放图片以适应窗口（保持宽高比）
        scaled_pixmap = self.background.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        # 居中绘制
        painter.drawPixmap(
            (self.width() - scaled_pixmap.width()) // 2,
            (self.height() - scaled_pixmap.height()) // 2,
            scaled_pixmap,
        )
        # 半透明遮罩：降低背景亮度，保证所有文字可读
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

    def update_input_display(self):
        left_monsters_dict, right_monsters_dict = self.input_panel.get_monster_counts()

        def update_input_display_half(input_layout, monsters_dict):
            # 清除现有显示
            for i in reversed(range(input_layout.count())):
                widget = input_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            has_input = False
            for i in range(1, MONSTER_COUNT + 1):
                value = monsters_dict[str(i)].text()
                if value.isdigit() and int(value) > 0:
                    has_input = True
                    monster_widget = self.create_monster_display_widget(i, value)
                    input_layout.addWidget(monster_widget)
            # 如果没有输入，显示提示
            if not has_input:
                input_layout.addWidget(QLabel("无"))

        update_input_display_half(self.left_input_layout, left_monsters_dict)
        update_input_display_half(self.right_input_layout, right_monsters_dict)

    def create_monster_display_widget(self, monster_id, count):
        """创建人物显示组件"""
        widget = QWidget()
        widget.setFixedWidth(67)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(0)  # 模糊半径（控制发光范围）
        shadow.setColor(QColor("#313131"))  # 发光颜色
        shadow.setOffset(2)  # 偏移量（0表示均匀四周发光）
        widget.setGraphicsEffect(shadow)

        widget.setStyleSheet(
            """
                QWidget {
                    border-radius: 0px;
                }
            """
        )

        layout = QVBoxLayout(widget)
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 人物图片
        img_label = QLabel()
        img_label.setFixedSize(70, 70)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        try:
            pixmap = QPixmap(f"images/{MONSTER_DATA['原始名称'][monster_id]}.png")
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    70, 70, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
                img_label.setPixmap(pixmap)
        except Exception as e:
            logger.error(f"加载人物{monster_id}图片错误: {str(e)}")
            pass

        # 添加鼠标悬浮提示
        if monster_id in MONSTER_DATA.index:
            data = MONSTER_DATA.loc[monster_id].to_dict()
            tooltip_text = ""
            for key, value in data.items():
                tooltip_text += f"{key}: {value}\n"
            img_label.setToolTip(tooltip_text.strip())

        # 数量标签
        count_label = QLabel(count)
        count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        count_label.setStyleSheet(
            """
            color: #EDEDED;
            font: bold 20px SimHei;
            border-radius: 5px;
            padding: 2px 5px;
            min-width: 20px;
        """
        )

        layout.addWidget(img_label)
        layout.addWidget(count_label)

        return widget

    def reset_entries(self):
        self.result_label.setText("预测结果将显示在这里")
        self.result_label.setStyleSheet("color: #E0E0E0;")
        self.update_input_display()

    def get_prediction(self):
        try:
            left_monsters_dict, right_monsters_dict = self.input_panel.get_monster_counts()
            left_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
            right_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)

            for name, entry in left_monsters_dict.items():
                value = entry.text()
                left_counts[int(name) - 1] = int(value) if value.isdigit() else 0

            for name, entry in right_monsters_dict.items():
                value = entry.text()
                right_counts[int(name) - 1] = int(value) if value.isdigit() else 0

            # 构建包含地形的完整特征向量
            full_features = self.input_panel.build_terrain_features(left_counts, right_counts)

            prediction = self.cannot_model.get_prediction_with_terrain(full_features)
            return prediction
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", "未找到模型文件，请先训练")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                QMessageBox.critical(self, "错误", "模型结构不匹配！请删除旧模型并重新训练")
            else:
                QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
        except ValueError:
            QMessageBox.critical(self, "错误", "请输入有效的数字（0或正整数）")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测时发生错误: {str(e)}")

        return 0.5

    def update_prediction(self, prediction):
        """更新预测结果显示"""
        # 模型结果处理
        right_win_prob = prediction
        left_win_prob = 1 - right_win_prob

        # 判断胜负方向
        winner = "左方" if left_win_prob > 0.5 else "右方"
        if 0.6 > left_win_prob > 0.4:
            winner = "难说"

        # 设置结果标签样式
        if winner == "左方":
            self.result_label.setStyleSheet("color: #E23F25; font: bold,14px;")
        else:
            self.result_label.setStyleSheet("color: #25ace2; font: bold,14px;")

        left_monsters_dict, right_monsters_dict = self.input_panel.get_monster_counts()
        # 生成结果文本
        if winner != "难说":
            result_text = f"预测胜方: {winner}\n" f"左 {left_win_prob:.2%} | 右 {right_win_prob:.2%}\n"
        else:
            result_text = (
                f"这一把{winner}\n" f"左 {left_win_prob:.2%} | 右 {right_win_prob:.2%}\n" f"难道说？难道说？难道说？\n"
            )
            self.result_label.setStyleSheet("color: #E0E0E0; font: bold,24px;")

        # 添加特殊干员提示
        special_messages = self.special_monster_handler.check_special_monsters(
            left_monsters_dict, right_monsters_dict, winner
        )
        if special_messages:
            result_text += "\n" + special_messages

        self.result_label.setText(result_text)

    def predict(self):
        prediction = self.get_prediction()
        self.update_prediction(prediction)
        self.update_input_display()

        if self.history_match_ui.isVisible():
            left_monsters_dict, right_monsters_dict = self.input_panel.get_monster_counts()
            self.history_match_ui.render_similar_matches(left_monsters_dict, right_monsters_dict)

    def get_recognize(self):
        """
        根据当前模式获取截图并识别
        """
        screenshot = None
        if self.current_capture_mode in ["ADB", "PC"]:
            screenshot = self.active_connector.capture_screenshot()
            if screenshot is None:
                # 尝试重新连接一次
                self.active_connector.connect()
                screenshot = self.active_connector.capture_screenshot()
            if screenshot is None:
                logger.error(f"{self.current_capture_mode} 截图失败")
            
            results = self.recognizer.process_regions(screenshot)
        else:
            # WIN 模式，recognizer 内部处理 WinRT 或 PIL
            results = self.recognizer.process_regions(None)

        return results

    def update_monster(self, results):
        """
        根据识别结果更新怪物面板
        """
        left_counts = {}
        right_counts = {}
        for res in results:
            if "error" not in res:
                region_id = res["region_id"]
                matched_id = res["matched_id"]
                number = res["number"]
                if matched_id != 0:
                    if region_id < 3:
                        left_counts[str(matched_id)] = int(number)
                    else:
                        right_counts[str(matched_id)] = int(number)
        self.input_panel.set_monster_counts(left_counts, right_counts)

    def recognize_only(self):
        recognize_results = self.get_recognize()
        self.update_monster(recognize_results)

    def recognize_and_predict(self):
        recognize_results = self.get_recognize()
        self.update_monster(recognize_results)
        prediction = self.get_prediction()
        self.update_prediction(prediction)
        # 历史对局
        if self.history_match_ui.isVisible():
            left_monsters_dict, right_monsters_dict = self.input_panel.get_monster_counts()
            self.history_match_ui.render_similar_matches(left_monsters_dict, right_monsters_dict)

    def toggle_history_panel(self):
        """切换历史对局面板的显示"""
        target_width = self.width()
        if self.history_match is None:
            QMessageBox.warning(self, "警告", "历史数据加载失败，无法显示历史对局")
            return

        is_visible = self.history_match_ui.isVisible()
        self.history_match_ui.setVisible(not is_visible)
        if not is_visible:
            self.history_button.setText("隐藏历史对局")
            left_monsters_dict, right_monsters_dict = self.input_panel.get_monster_counts()
            self.history_match_ui.render_similar_matches(left_monsters_dict, right_monsters_dict)
            target_width += 540
        else:
            self.history_button.setText("显示历史对局")
            target_width -= 540
        self.animate_size_change(target_width)

    def reselect_roi(self):
        self.recognizer.select_roi()

    def toggle_auto_fetch(self):
        if not (hasattr(self, "auto_fetch") and self.auto_fetch.auto_fetch_running):
            self.auto_fetch = auto_fetch.AutoFetch(
                self.active_connector,
                self.game_mode,
                self.is_invest,
                update_prediction_callback=self.update_prediction_callback,
                update_monster_callback=self.update_monster_callback,
                updater=self.update_statistics_callback,
                start_callback=self.start_callback,
                stop_callback=self.stop_callback,
                training_duration=float(self.duration_entry.text()) * 3600,  # 获取训练时长
                session_name=self.session_name_entry.text().strip(),
            )
            self.auto_fetch.start_auto_fetch()
        else:
            self.auto_fetch.stop_auto_fetch()

    def train_onnx_model(self):
        """运行训练+转ONNX全流程（子进程，不阻塞UI）"""
        # 检查：采集运行时不能训练
        if hasattr(self, "auto_fetch") and self.auto_fetch.auto_fetch_running:
            QMessageBox.warning(self, "冲突", "请先停止「自动获取数据」再进行训练")
            return
        import subprocess, threading

        def _train_thread():
            try:
                self.train_onnx_button.setText("⏳ 训练中...")
                self.train_onnx_button.setEnabled(False)
                self.train_status_signal.emit("训练中，请稍候...", False)

                # 使用 uv run 执行 train_onnx.py
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                cmd = ["uv", "run", "python", "train_onnx.py"]
                session = self.session_name_entry.text().strip()
                if session:
                    cmd += ["--session", session]
                device = self.device_menu.currentData() or ""
                if device:
                    cmd += ["--device", device]
                proc = subprocess.Popen(
                    cmd,
                    cwd="D:\\CannotMax",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                )

                # 非阻塞读取：同时检测子进程是否崩溃
                import queue
                q = queue.Queue()

                def _reader(pipe, q):
                    try:
                        for line in iter(pipe.readline, b''):
                            q.put(line)
                    finally:
                        pipe.close()

                reader_thread = threading.Thread(target=_reader, args=(proc.stdout, q), daemon=True)
                reader_thread.start()

                while True:
                    try:
                        line = q.get(timeout=1.0)
                    except queue.Empty:
                        if proc.poll() is not None:
                            # 子进程已退出且没有更多输出
                            break
                        continue

                    try:
                        decoded_line = line.decode('utf-8').rstrip()
                    except UnicodeDecodeError:
                        decoded_line = line.decode('gbk', errors='ignore').rstrip()
                    if decoded_line:
                        logger.info(decoded_line)
                        self.train_status_signal.emit(decoded_line[-80:], False)

                reader_thread.join(timeout=3)
                proc.wait()
                return_code = proc.returncode

                if return_code == 0:
                    # 训练成功，重新加载模型
                    from importlib import reload
                    import predict_onnx
                    reload(predict_onnx)
                    session = self.session_name_entry.text().strip()
                    model_path = predict_onnx.resolve_model_path(session)
                    self.cannot_model = predict_onnx.CannotModel(model_path)
                    model_name = Path(self.cannot_model.model_path).name if self.cannot_model.model_path else "未加载"
                    self.setWindowTitle(
                        self.windowTitle().rsplit(" - model:", 1)[0] + f" - model: {model_name}"
                    )
                    if self.cannot_model.is_model_loaded:
                        self.recognize_button.setEnabled(True)
                        self.recognize_button.setToolTip("")
                        self.input_panel.predict_button.setEnabled(True)
                        self.input_panel.predict_button.setToolTip("")
                        self.train_status_signal.emit("✅ 训练完成！ONNX 模型已加载", True)
                    else:
                        self.train_status_signal.emit("⚠️ 模型生成但加载失败，请检查", True)
                else:
                    self.train_status_signal.emit(f"❌ 训练失败 (code={return_code})", True)

            except Exception as e:
                logger.error(f"训练出错: {e}")
                self.train_status_signal.emit(f"❌ 训练异常: {str(e)[:60]}", True)
            finally:
                self.train_onnx_button.setText("🧠 训练ONNX模型")
                self.train_onnx_button.setEnabled(True)

        threading.Thread(target=_train_thread, daemon=True).start()

    def start_auto_collect_and_train(self):
        """启动自动数据收集和训练流程"""
        # 检查是否已经在运行
        if hasattr(self, 'auto_collect_train') and self.auto_collect_train.is_running:
            reply = QMessageBox.question(
                self,
                "确认停止",
                "自动数据收集已在运行中，是否要停止？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.auto_collect_train.stop()
                self.auto_collect_train_button.setText("🔄 一键收集并训练")
                self.auto_collect_train_button.setEnabled(True)
                self.auto_collect_status_label.setText("⏹️ 已手动停止")
            return

        # 获取训练时长和会话名
        try:
            training_hours = float(self.duration_entry.text())
        except ValueError:
            training_hours = 1.0
        session_name = self.session_name_entry.text().strip()
        device_type = self.device_menu.currentData() or ""

        # 设置运行状态
        self.auto_collect_train_button.setEnabled(False)
        self.auto_collect_train_button.setText("⏳ 收集中...")
        self.stop_auto_collect_button.setEnabled(True)
        self.auto_collect_status_label.setText("准备启动...")

        # 导入自动收集模块
        from auto_collect_and_train import AutoCollectAndTrain

        # 创建实例
        self.auto_collect_train = AutoCollectAndTrain(
            connector=self.active_connector,
            game_mode=self.game_mode,
            training_duration_hours=training_hours,
            progress_callback=self.update_auto_collect_status,
            completion_callback=self._emit_auto_collect_complete,
            session_name=session_name,
            device_type=device_type,
            is_invest=self.is_invest,
        )

        # 启动流程
        self.auto_collect_train.start()
    
    def stop_auto_collect_and_train(self):
        """停止自动数据收集流程"""
        if hasattr(self, 'auto_collect_train') and self.auto_collect_train.is_running:
            msg = QMessageBox(self)
            msg.setWindowTitle("确认停止")
            msg.setText("自动数据收集仍在运行中，请选择操作：")
            btn_stop_train = msg.addButton("停止并训练", QMessageBox.ButtonRole.AcceptRole)
            btn_stop_only = msg.addButton("仅停止", QMessageBox.ButtonRole.DestructiveRole)
            btn_cancel = msg.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            msg.exec()

            if msg.clickedButton() == btn_stop_only:
                self.auto_collect_train.stop(start_training=False)
                self.auto_collect_train_button.setEnabled(True)
                self.auto_collect_train_button.setText("🔄 一键收集并训练")
                self.stop_auto_collect_button.setEnabled(False)
                self.auto_collect_status_label.setText("⏹️ 已手动停止")
            elif msg.clickedButton() == btn_stop_train:
                self.auto_collect_train.stop(start_training=True)
                self.auto_collect_train_button.setEnabled(False)
                self.auto_collect_train_button.setText("⏳ 训练中...")
                self.auto_collect_status_label.setText("正在准备训练...")
        else:
            QMessageBox.information(self, "提示", "当前没有运行中的自动收集流程")
    
    def update_auto_collect_status(self, message: str):
        """自动收集状态回调（当前在工作线程中），通过信号安全传到主线程"""
        self.auto_collect_status_signal.emit(message)

    def _on_auto_collect_status(self, message: str):
        """主线程：更新自动收集状态显示"""
        self.auto_collect_status_label.setText(message[-80:])  # 显示最后80个字符

    def _on_train_status(self, message: str, is_final: bool):
        """主线程：更新训练状态显示"""
        self.train_status_label.setText(message[-80:])
    
    def _emit_auto_collect_complete(self, success: bool, message: str):
        """完成回调（工作线程），通过信号安全转到主线程"""
        self.auto_collect_complete_signal.emit(success, message)

    def on_auto_collect_complete(self, success: bool, message: str):
        """自动收集流程完成回调"""
        # 恢复按钮状态
        self.auto_collect_train_button.setEnabled(True)
        self.auto_collect_train_button.setText("🔄 一键收集并训练")
        self.stop_auto_collect_button.setEnabled(False)
        
        if success:
            self.auto_collect_status_label.setText("✅ 完成！")
            QMessageBox.information(self, "成功", message)
            
            # 重新加载模型
            try:
                from importlib import reload
                import predict_onnx
                reload(predict_onnx)
                session = self.auto_collect_train.session_name if hasattr(self, 'auto_collect_train') else ""
                model_path = predict_onnx.resolve_model_path(session)
                self.cannot_model = predict_onnx.CannotModel(model_path)
                
                if self.cannot_model.is_model_loaded:
                    model_name = Path(self.cannot_model.model_path).name
                    self.setWindowTitle(
                        self.windowTitle().rsplit(" - model:", 1)[0] + f" - model: {model_name}"
                    )
                    self.recognize_button.setEnabled(True)
                    self.input_panel.predict_button.setEnabled(True)
                    logger.info("新模型已成功加载")
            except Exception as e:
                logger.error(f"加载新模型失败: {e}")
        else:
            self.auto_collect_status_label.setText("❌ 失败")
            QMessageBox.critical(self, "失败", message)

    def update_statistics(self):
        elapsed_time = time.time() - self.auto_fetch.start_time if self.auto_fetch.start_time else 0
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, _ = divmod(remainder, 60)
        stats_text = (
            f"总共填写次数: {self.auto_fetch.total_fill_count},    "
            f"填写×次数: {self.auto_fetch.incorrect_fill_count},    "
            f"当次运行时长: {int(hours)}小时{int(minutes)}分钟"
        )
        self.stats_label.setText(stats_text)

    def refresh_device_list(self):
        """刷新并更新模拟器序列号下拉列表"""
        current_text = self.serial_entry.currentText()
        devices = self.adb_connector.get_device_list()
        self.serial_entry.clear()
        if devices:
            self.serial_entry.addItems(devices)
            if current_text in devices:
                self.serial_entry.setCurrentText(current_text)
            else:
                self.serial_entry.setCurrentIndex(0)
        else:
            self.serial_entry.addItem("127.0.0.1:5555")
            self.serial_entry.setCurrentText(current_text if current_text else "127.0.0.1:5555")

    def update_device_serial(self):
        new_serial = self.serial_entry.currentText()
        device_serial = self.adb_connector.update_device_serial(new_serial)
        self.adb_connector.connect()  # 尝试连接新设备
        self.serial_entry.setCurrentText(device_serial)
        QMessageBox.information(self, "提示", f"已更新模拟器序列号为: {device_serial}")

    def start_callback(self):
        self.update_button_signal.emit("停止自动获取数据")

    def stop_callback(self):
        self.update_button_signal.emit("自动获取数据")

    def update_monster_callback(self, results: list):
        self.update_monster_signal.emit(results)

    def update_prediction_callback(self, prediction: float):
        self.update_prediction_signal.emit(prediction)

    def update_statistics_callback(self):
        self.update_statistics_signal.emit()

    def run_simulation(self):
        """
        获取左右怪物信息，转换为JSON格式，并通过stdin传递给main_sim.py子进程。
        """
        left_monsters_data = {}
        right_monsters_data = {}

        left_monsters_dict, right_monsters_dict = self.input_panel.get_monster_counts()

        # 获取左侧怪物信息
        for monster_id, entry in left_monsters_dict.items():
            count = entry.text()
            if count.isdigit() and int(count) > 0:
                # Need to map monster_id (string) to monster name
                # Assuming MONSTER_MAPPING is accessible or can be imported
                try:
                    # Convert monster_id string to int for mapping
                    monster_name = self.get_monster_name_by_id(int(monster_id))
                    if monster_name:
                        left_monsters_data[monster_name] = int(count)
                except ValueError:
                    logger.error(f"Invalid monster ID: {monster_id}")
                except Exception as e:
                    logger.error(f"Error getting monster name for ID {monster_id}: {e}")

        # 获取右侧怪物信息
        for monster_id, entry in right_monsters_dict.items():
            count = entry.text()
            if count.isdigit() and int(count) > 0:
                try:
                    # Convert monster_id string to int for mapping
                    monster_name = self.get_monster_name_by_id(int(monster_id))
                    if monster_name:
                        right_monsters_data[monster_name] = int(count)
                    else:
                        logger.error(f"Monster name not found for ID {monster_id}")
                except ValueError:
                    logger.error(f"Invalid monster ID: {monster_id}")
                except Exception as e:
                    logger.error(f"Error getting monster name for ID {monster_id}: {e}")

        simulation_data = {"left": left_monsters_data, "right": right_monsters_data}

        json_data = json.dumps(simulation_data, ensure_ascii=False)
        logger.info(f"Simulation data JSON: {json_data}")

        try:
            # 启动main_sim.py子进程 (非阻塞)
            # Use sys.executable to ensure the same Python interpreter is used
            process = subprocess.Popen(
                [sys.executable, "main_sim.py"], stdin=subprocess.PIPE, text=True, encoding="utf-8"
            )
            # 通过stdin传递JSON数据并关闭stdin
            process.stdin.write(json_data)
            process.stdin.close()
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", "未找到 main_sim.py 文件，请检查路径。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动模拟器时发生错误: {str(e)}")

    def get_monster_name_by_id(self, monster_id: int):
        """根据怪物ID获取怪物名称"""
        # Need to import MONSTER_MAPPING from simulator.utils
        try:
            from simulator.utils import MONSTER_MAPPING

            # Adjust for 1-based UI IDs vs 0-based mapping keys
            monster_name = MONSTER_MAPPING.get(monster_id - 1)
            if not monster_name:
                logger.error(f"Monster ID {monster_id} not found in MONSTER_MAPPING.")
            return monster_name
        except ImportError:
            logger.error("Error importing MONSTER_MAPPING from simulator.utils")
            return None

    def update_game_mode(self, mode):
        self.game_mode = mode

    def update_invest_status(self, state):
        self.is_invest = state == Qt.CheckState.Checked.value

    def update_result(self, text):
        self.result_label.setText(text)

    def update_stats(self, total, incorrect, duration):
        stats_text = f"总共: {total}, 错误: {incorrect}, 时长: {duration}"
        self.stats_label.setText(stats_text)

    def update_image_display(self, qimage):
        self.image_display.setPixmap(
            QPixmap.fromImage(qimage).scaled(
                self.image_display.width(), self.image_display.height(), Qt.AspectRatioMode.KeepAspectRatio
            )
        )

    def toggle_always_on_top(self):
        if self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
            self.always_on_top_button.setText("窗口置顶")
        else:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
            self.always_on_top_button.setText("取消置顶")
        self.show() # Reapply window flags

    def closeEvent(self, event):
        """窗口关闭时的处理"""
        if hasattr(self, "auto_fetch") and self.auto_fetch.auto_fetch_running:
            self.auto_fetch.stop_auto_fetch()

        if hasattr(self, "auto_collect_train") and self.auto_collect_train.is_running:
            logger.info("正在停止自动数据收集流程...")
            self.auto_collect_train.stop()

        self._save_session_config()
        event.accept()

    CONFIG_PATH = Path(__file__).parent / "app_config.json"

    def _load_session_config(self):
        """加载缓存的配置（会话名和设备）"""
        try:
            if self.CONFIG_PATH.exists():
                data = json.loads(self.CONFIG_PATH.read_text(encoding="utf-8"))
                return {
                    "session_name": data.get("session_name", "").strip(),
                    "device_type": data.get("device_type", ""),
                }
        except Exception as e:
            logger.debug(f"加载配置失败: {e}")
        return {"session_name": "", "device_type": ""}

    def _save_session_config(self):
        """保存配置到文件"""
        try:
            session = self.session_name_entry.text().strip()
            device_data = self.device_menu.currentData()
            data = {
                "session_name": session,
                "device_type": device_data if device_data else "",
            }
            self.CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"保存配置失败: {e}")

    def _on_session_name_changed(self):
        """会话名变更时自动缓存"""
        if self._session_loading:
            return
        self._save_session_config()

    def _on_device_changed(self):
        """训练设备变更时自动缓存"""
        if self._session_loading:
            return
        self._save_session_config()


if __name__ == "__main__":
    app = QApplication([])
    window = ArknightsApp()
    window.show()
    app.exec()
