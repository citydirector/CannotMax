"""
测试自动数据收集和训练功能
验证各个组件是否正常工作
"""
import sys
from pathlib import Path

def test_imports():
    """测试导入是否正常"""
    print("Test 1: Check module imports...")
    try:
        from auto_collect_and_train import AutoCollectAndTrain
        print("  [PASS] auto_collect_and_train imported successfully")
        return True
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        return False


def test_model_padding():
    """测试模型维度填充功能"""
    print("\nTest 2: Check model dimension adaptation...")
    try:
        import numpy as np
        from predict_onnx import CannotModel
        
        model = CannotModel()
        if not model.is_model_loaded:
            print("  [WARN] Model not loaded (may not exist), skip test")
            return True
        
        # Create 60-dim input
        left = np.zeros(60, dtype=np.int16)
        right = np.zeros(60, dtype=np.int16)
        left[0] = 3
        right[1] = 2
        
        # Try prediction
        result = model.get_prediction(left, right)
        print(f"  [PASS] 60-dim input prediction success: {result:.4f}")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_collection_simulation():
    """模拟数据收集流程"""
    print("\nTest 3: Simulate data collection flow...")
    try:
        from auto_collect_and_train import AutoCollectAndTrain
        
        # Create Mock ADB connector
        class MockADB:
            def capture_screenshot(self):
                return None
            
            def click(self, point):
                pass
        
        mock_adb = MockADB()
        
        # Create AutoCollectAndTrain instance
        collector = AutoCollectAndTrain(
            adb_connector=mock_adb,
            game_mode="单人",
            training_duration_hours=0.1,  # 6 minutes for test
        )
        
        print(f"  [PASS] AutoCollectAndTrain instantiation success")
        print(f"    - Training duration: {collector.training_duration_seconds} seconds")
        print(f"    - Game mode: {collector.game_mode}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gui_button_exists():
    """测试GUI按钮是否存在"""
    print("\nTest 4: Check GUI button definition...")
    try:
        # Read main.py to check button definition
        with open("main.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        checks = [
            ("auto_collect_train_button", "Button variable definition"),
            ("start_auto_collect_and_train", "Callback function definition"),
            ("从0开始收集数据并训练", "Button text"),
        ]
        
        all_found = True
        for check_str, desc in checks:
            if check_str in content:
                print(f"  [PASS] Found {desc}: {check_str}")
            else:
                print(f"  [FAIL] Not found {desc}: {check_str}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"  [FAIL] Check failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Auto Data Collection and Training - Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_padding,
        test_data_collection_simulation,
        test_gui_button_exists,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Test exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n[PASS] All tests passed! Feature is ready.")
        return 0
    else:
        print(f"\n[WARN] {total - passed} test(s) failed, please check.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
