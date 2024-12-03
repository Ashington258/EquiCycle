import logging
import os

# 确保日志文件夹存在
log_folder = "log"
os.makedirs(log_folder, exist_ok=True)

# 配置日志
log_file_path = os.path.join(log_folder, "application.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    force=True,
)
