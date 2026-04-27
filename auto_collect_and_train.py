"""
自动数据收集和训练模块
功能：一键完成数据收集 -> 模型训练 -> 模型加载的完整流程
"""
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional
import subprocess

logger = logging.getLogger(__name__)


class AutoCollectAndTrain:
    """
    自动数据收集和训练流程管理器
    
    工作流程：
    1. 检查并忽略当前错误的ONNX模型
    2. 启动自动游戏收集数据（固定选左）
    3. 到达设定时间后停止游戏
    4. 自动触发模型训练
    5. 重新加载新模型
    """
    
    def __init__(
        self,
        adb_connector,
        game_mode: str = "单人",
        training_duration_hours: float = 1.0,
        progress_callback: Optional[Callable[[str], None]] = None,
        completion_callback: Optional[Callable[[bool, str], None]] = None,
        session_name: str = "",
        device_type: str = "",
    ):
        """
        Args:
            adb_connector: ADB连接器实例
            game_mode: 游戏模式（"单人"或"30人"）
            training_duration_hours: 训练时长（小时）
            progress_callback: 进度回调函数，接收状态字符串
            completion_callback: 完成回调函数，接收(success: bool, message: str)
            session_name: 会话名称，用于分组数据和模型
            device_type: 训练设备（"cpu" 或 "cuda"，空=自动检测）
        """
        self.adb_connector = adb_connector
        self.game_mode = game_mode
        self.training_duration_seconds = int(training_duration_hours * 3600)
        self.progress_callback = progress_callback
        self.completion_callback = completion_callback
        self.session_name = session_name
        self.device_type = device_type

        self.is_running = False
        self.auto_fetch_instance = None
        self._stop_event = threading.Event()
        
    def _update_progress(self, message: str, log: bool = True):
        """更新进度信息"""
        if log:
            logger.info(message)
        if self.progress_callback:
            self.progress_callback(message)
    
    def start(self):
        """启动自动收集和训练流程"""
        if self.is_running:
            logger.warning("自动收集流程已在运行中")
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        # 在新线程中运行整个流程
        thread = threading.Thread(target=self._run_workflow, daemon=True)
        thread.start()
    
    def stop(self):
        """停止自动收集流程"""
        if not self.is_running:
            return
        
        self._update_progress("正在停止自动收集...")
        self._stop_event.set()
        
        # 停止auto_fetch
        if self.auto_fetch_instance and self.auto_fetch_instance.auto_fetch_running:
            self.auto_fetch_instance.stop_auto_fetch()
        
        self.is_running = False
    
    def _run_workflow(self):
        """执行完整的工作流程"""
        try:
            # 步骤1: 准备阶段
            self._update_progress("🚀 开始自动数据收集和训练流程")
            self._update_progress(f"⏱️ 预计时长: {self.training_duration_seconds / 3600:.1f} 小时")

            # 清理上次中断遗留的临时模型文件
            prefix = f"{self.session_name}_" if self.session_name else ""
            for name in ["best_model_acc.pth", "best_model_loss.pth", "best_model_full.pth"]:
                p = Path(f"models/{prefix}{name}")
                if p.exists():
                    p.unlink()
                    self._update_progress(f"已清理临时文件: {p.name}")
            
            # 步骤2: 检查并备份旧模型（如果存在）
            prefix = f"{self.session_name}_" if self.session_name else ""
            old_model_path = Path(f"models/{prefix}best_model_full.onnx")
            backup_path = old_model_path.with_suffix(".onnx.backup")
            if old_model_path.exists():
                self._update_progress("📦 检测到旧模型，将在使用前禁用")
                if backup_path.exists():
                    backup_path.unlink()
                old_model_path.rename(backup_path)
                old_data_path = old_model_path.with_suffix(".onnx.data")
                if old_data_path.exists():
                    backup_data_path = backup_path.with_suffix(".backup.data")
                    if backup_data_path.exists():
                        backup_data_path.unlink()
                    old_data_path.rename(backup_data_path)
                self._update_progress("✓ 旧模型已备份")
            
            # 步骤3: 启动数据收集
            self._update_progress("🎮 启动数据收集（固定选左）...")
            self._update_progress("ℹ️ 提示: 此模式不使用模型预测，仅收集原始数据")
            
            # 导入auto_fetch模块
            import auto_fetch
            
            # 创建临时的回调函数
            def dummy_update_prediction(pred):
                pass  # 不使用预测结果
            
            def dummy_update_monster(monsters):
                pass  # 不更新怪物显示
            
            def dummy_updater():
                pass  # 不更新统计信息
            
            def on_fetch_start():
                self._update_progress("✓ 数据收集已开始")
                self._update_progress(f"📊 游戏模式: {self.game_mode}")
                self._update_progress("🔒 策略: 固定观望（不投资），确保数据纯净性")
                self._update_progress("ℹ️ 注意: 此模式忽略GUI的投资复选框设置")
            
            def on_fetch_stop():
                self._update_progress("✓ 数据收集已停止")
            
            # 创建AutoFetch实例（强制不投资，固定观望）
            # 注意：即使模型不存在或加载失败，也不影响数据收集
            self.auto_fetch_instance = auto_fetch.AutoFetch(
                adb_connector=self.adb_connector,
                game_mode=self.game_mode,
                is_invest=False,  # 固定不投资，这样会固定选左/观望
                update_prediction_callback=dummy_update_prediction,
                update_monster_callback=dummy_update_monster,
                updater=dummy_updater,
                start_callback=on_fetch_start,
                stop_callback=on_fetch_stop,
                training_duration=self.training_duration_seconds,
                session_name=self.session_name,
            )
            
            # 检查模型状态并给出提示
            if not self.auto_fetch_instance.cannot_model.is_model_loaded:
                self._update_progress("ℹ️ 模型未加载（已备份或不存在），这是正常的")
                self._update_progress("   数据收集不依赖模型，可以正常进行")
            else:
                self._update_progress("⚠️ 警告: 检测到模型已加载")
                self._update_progress("   但本模式不使用预测，请放心")
            
            # 启动数据收集
            self.auto_fetch_instance.start_auto_fetch()
            
            # 等待数据收集完成或手动停止
            self._update_progress("⏳ 数据收集中，请稍候...")
            last_update_time = time.time()
            
            while self.auto_fetch_instance.auto_fetch_running and not self._stop_event.is_set():
                time.sleep(0.5)  # 更频繁地检查
                
                current_time = time.time()
                # 每秒更新一次剩余时间
                if current_time - last_update_time >= 1.0:
                    last_update_time = current_time
                    
                    # 显示剩余时间（仅更新GUI，不写入控制台日志）
                    if self.auto_fetch_instance.start_time:
                        elapsed = current_time - self.auto_fetch_instance.start_time
                        remaining = self.training_duration_seconds - elapsed
                        if remaining > 0:
                            mins = int(remaining / 60)
                            secs = int(remaining % 60)
                            hours = int(mins / 60)
                            mins_remaining = mins % 60
                            if hours > 0:
                                time_str = f"{hours}小时{mins_remaining}分{secs}秒"
                            else:
                                time_str = f"{mins}分{secs}秒"
                            self._update_progress(f"⏱️ 剩余时间: {time_str}", log=False)
                        else:
                            self._update_progress("⏱️ 时间到，正在停止...", log=False)
            
            # 确保停止数据收集
            if self.auto_fetch_instance.auto_fetch_running:
                self.auto_fetch_instance.stop_auto_fetch()
            
            # 等待一下确保文件写入完成
            time.sleep(2)
            
            # 步骤4: 统计收集到的数据
            self._update_progress("📊 统计收集到的数据...")
            data_count = self._count_collected_data()
            self._update_progress(f"✓ 共收集到 {data_count} 条对战数据")
            
            if data_count < 10:
                self._update_progress(f"⚠️ 警告: 数据量较少 ({data_count}条)，训练效果可能不佳")
                self._update_progress("   建议至少收集50条以上数据")
            
            # 步骤5: 开始训练模型
            self._update_progress("🧠 开始训练新模型...")
            train_success = self._train_model()
            
            if not train_success:
                self._update_progress("❌ 模型训练失败")
                if self.completion_callback:
                    self.completion_callback(False, "模型训练失败")
                return
            
            # 步骤6: 恢复旧模型备份（如果新训练失败则使用旧的）
            # 这里我们假设训练成功，删除备份
            for p in [backup_path, backup_path.with_suffix(".backup.data")]:
                if p.exists():
                    p.unlink()
            self._update_progress("🗑️ 已清理旧模型备份")
            
            # 步骤7: 完成
            self._update_progress("✅ 自动数据收集和训练流程完成！")
            self._update_progress("🎉 新模型已就绪，可以开始使用了")
            
            if self.completion_callback:
                self.completion_callback(True, f"成功！收集{data_count}条数据，新模型已就绪")
        
        except Exception as e:
            logger.exception(f"自动收集流程出错: {e}")
            self._update_progress(f"❌ 流程异常: {str(e)}")
            if self.completion_callback:
                self.completion_callback(False, f"流程异常: {str(e)}")
        
        finally:
            self.is_running = False
            self.auto_fetch_instance = None
    
    def _count_collected_data(self) -> int:
        """统计收集到的数据条数"""
        try:
            import pandas as pd

            data_dir = Path("data")
            if self.session_name:
                target = data_dir / self.session_name / "arknights.csv"
                if not target.exists():
                    return 0
                try:
                    df = pd.read_csv(target)
                    return len(df)
                except Exception:
                    return 0
            else:
                csv_files = list(data_dir.rglob("arknights.csv"))
                total_rows = 0
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        total_rows += len(df)
                    except Exception:
                        pass
                return total_rows
        except Exception as e:
            logger.error(f"统计数据失败: {e}")
            return 0
    
    def _train_model(self) -> bool:
        """执行模型训练"""
        try:
            self._update_progress("🔄 启动训练进程...")

            # 使用subprocess运行train_onnx.py
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            cmd = ["uv", "run", "python", "train_onnx.py"]
            if self.session_name:
                cmd += ["--session", self.session_name]
            if self.device_type:
                cmd += ["--device", self.device_type]
            proc = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
            
            # 实时输出训练日志（处理编码问题）
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                
                try:
                    # 尝试UTF-8解码，失败则使用GBK
                    try:
                        decoded_line = line.decode('utf-8').rstrip()
                    except UnicodeDecodeError:
                        decoded_line = line.decode('gbk', errors='ignore').rstrip()
                    
                    if decoded_line:
                        logger.debug(f"训练日志: {decoded_line}")
                        # 每10行更新一次进度，避免刷屏
                        if "Epoch" in decoded_line or "Loss" in decoded_line or "训练" in decoded_line:
                            self._update_progress(f"训练中: {decoded_line[:60]}")
                except Exception as e:
                    logger.debug(f"日志解码错误: {e}")
                    continue
            
            proc.wait()
            return_code = proc.returncode
            
            if return_code == 0:
                self._update_progress("✓ 模型训练成功")
                return True
            else:
                self._update_progress(f"❌ 模型训练失败 (退出码: {return_code})")
                return False
        
        except Exception as e:
            logger.exception(f"训练过程异常: {e}")
            self._update_progress(f"❌ 训练异常: {str(e)}")
            return False
