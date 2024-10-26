from hdfs import InsecureClient
import os

from config import config


class HDFSUtils:
    client = InsecureClient(config['hdfs']['url'])

    @classmethod
    def upload(cls, local_path, hdfs_path):
        cls.client.upload(hdfs_path, local_path)

    @classmethod
    def download(cls, hdfs_path, temp_dir):
        file_name = os.path.basename(hdfs_path)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        local_path = os.path.join(temp_dir, file_name)
        # 检查文件是否已经存在
        if os.path.exists(local_path):
            return local_path
        cls.client.download(hdfs_path, local_path)
        return local_path

    @classmethod
    def download_bytes(cls, hdfs_path):
        with cls.client.read(hdfs_path) as reader:
            img_bytes = reader.read()
        return img_bytes
