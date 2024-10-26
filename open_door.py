import datetime
import json
import os
import threading
import requests
import cv2

url = 'https://open.ys7.com/api/lapp/token/get'
headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
}
data = {
    'appKey': '4292b4c8368a4635b728ea01670c9324',
    'appSecret': '2e4bac7f589244cdc6d661657dccb021'
}


# 获取Token
def get_accessToken():
    response = requests.post(url, headers=headers, data=data)
    # print(response.text)
    # 解析返回的json数据
    result = json.loads(response.text)
    accessToken = result['data']['accessToken']
    return accessToken


token = get_accessToken()

# print(token)
# 获取设备的rtmp视频流地址
def get_rtmp(device_id):
    url3 = 'https://open.ys7.com/api/lapp/v2/live/address/get'
    headers3 = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    data3 = {
        'accessToken': token,
        'deviceSerial': device_id,  # 起始页
        'protocol': 3,  # rtmp - 3
        'type': 1
    }
    response = requests.post(url3, headers=headers3, data=data3)
    # 解析返回的json数据
    result = json.loads(response.text)
    return result


"""
{'msg': '操作成功', 'code': '200', 
'data': {'id': '767025116881100800', 'url': 'rtmp://rtmp03open.ys7.com:1935/v3/openlive/K27000199_1_1?expire=1729231438&id=767025116881100800&t=0ebaefdcc5a5c7901310c2aeedd1faf604725abe81552a3d43f546d4285841a3&ev=100', 'expireTime': '2024-10-18 14:03:58'}}
"""


def get_images_by_device_id(device_id):
    rtmp_url = get_rtmp(device_id)
    # print(rtmp_url)
    rtmp_url = rtmp_url.get('data').get('url')
    # 创建VideoCapture对象
    cap = cv2.VideoCapture(rtmp_url)
    counts = 0
    # 检查是否成功打开
    if not cap.isOpened():
        print("无法打开RTMP流，请检查URL是否正确以及服务器是否支持.")
    else:
        # 获取视频的一些属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"FPS: {fps}, Frame Count: {frame_count}")

        # 循环读取视频帧
        while True:
            ret, frame = cap.read()

            if not ret:
                print("无法获取帧，检查视频流是否仍然可用.")
                continue
            image_path = r"E:\Work_Ysy\images\1019"
            image_path = os.path.join(image_path, device_id)
            image_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            save_name = os.path.join(image_path, f'{image_name}_{(counts)}.png')
            if counts % 5 == 0:
                os.makedirs(image_path, exist_ok=True)
                # 保存帧
                cv2.imwrite(save_name, frame)
                print("image,保存成功")

            counts += 1
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 画面无门：K26431181, K26431523, K26997759, K26998222
    # list_device_id = ['K26431400', 'K26431181', 'K27001152', 'K26431523', 'K26431513', 'K26997759', 'K26998222', 'K26431643']
    # for device in list_device_id:
    #     threading.Thread(target=get_images_by_device_id, args=(device,)).start()
    device_id = 'K26431702'
    get_images_by_device_id(device_id)
