from pathlib import Path
import zipfile
import re
from datetime import datetime
import shutil


def create_zip_package(output_zip_path, session_name=""):
    data_folder = Path("data")

    date_pattern = re.compile(r"^\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}$")

    if session_name:
        target = data_folder / session_name
        if target.is_dir():
            folders = [target]
        else:
            print(f"未找到会话目录: {target}")
            return
    else:
        folders = [
            folder for folder in data_folder.iterdir()
            if folder.is_dir() and date_pattern.match(folder.name)
        ]

    if not folders:
        print("未找到输出目录！")
        return

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for folder in folders:
            for file_path in folder.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(data_folder)
                    zipf.write(file_path, arcname=str(arcname))

    for folder in folders:
        try:
            shutil.rmtree(folder)
            print(f"已删除文件夹：{folder}")
        except Exception as e:
            print(f"删除文件夹 {folder} 时出错：{e}")

    print(f"压缩包已创建：{output_zip_path}")


def package_data(session_name=""):
    current_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    prefix = f"{session_name}_" if session_name else ""
    output_zip = f"{prefix}arknights_package_{current_time}.zip"

    create_zip_package(output_zip, session_name=session_name)
    return output_zip


if __name__ == "__main__":
    package_data()
