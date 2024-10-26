import datetime

import cv2
import numpy as np
import torch
from common_utils._utils import is_in_poly, is_backlit, is_blurred, calculate_insert_over_union

from yolo.yolov5.utils.general import non_max_suppression, scale_boxes

from logger_conf import setup_logger
from yolo.model.model_rep import ModelRep, HelmetModel, PersonModel

logger = setup_logger("model_rep")


class HelmetDetect(object):

    def __init__(self, poly=None, new_h=640, new_w=640):
        if poly is None:
            poly = []
        self.helmet_model = ModelRep.get_model(HelmetModel.model_key)
        self.person_model = ModelRep.get_model(PersonModel.model_key)
        self.cam_id = None
        self.non_max_suppression_nm = 32
        self.poly = poly
        self.new_h = 640 if new_h <=0 else (new_h // 32) * 32
        self.new_w = 640 if new_w <=0 else (new_w // 32) * 32

    def get_info(self, img_shape_2):
        h, w = img_shape_2
        r = min(self.new_h / h, self.new_w / w)
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = self.new_w - new_unpad[0], self.new_h - new_unpad[1]  # wh padding
        # dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding  tensorRT needless
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        info = [top, bottom, left, right, new_unpad]
        return info

    def run(self, img_bytes):
        origin_img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), -1)  # 读取图片路径
        if not isinstance(origin_img, np.ndarray):
            return None

        # 4通道图片转换3通道
        if origin_img.shape[2] == 4:
            origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGRA2RGB)

        device = self.helmet_model.get_model().device
        names = self.helmet_model.get_model().names
        hat_receive_image_time = datetime.datetime.now()
        # type_map = {'person': '未佩戴安全帽', 'hat1': '佩戴安全帽', 'hat0': '佩戴安全帽'}
        type_map = {'class2': '未佩戴安全帽', 'class1': '佩戴安全帽', 'class0': '佩戴安全帽'}
        illegal = ['未佩戴安全帽']

        origin_img_shape = origin_img.shape[:2]
        top, bottom, left, right, new_unpad = self.get_info(origin_img_shape)
        origin_resize_img = cv2.resize(origin_img, new_unpad, interpolation=cv2.INTER_LINEAR)
        origin_resize_img = cv2.copyMakeBorder(origin_resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                               value=(114, 114, 114))

        ################################################
        origin_resize_img = origin_resize_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        origin_resize_img = np.ascontiguousarray(origin_resize_img)  # contiguous
        origin_resize_img = torch.from_numpy(origin_resize_img).to(device)
        torch.cuda.synchronize()

        origin_resize_img = origin_resize_img.float()  # uint8 to fp16/32
        origin_resize_img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(origin_resize_img.shape) == 3:
            origin_resize_img = origin_resize_img[None]  # expand for batch dim

        person_predicted = self.person_model.predict(origin_resize_img)  # pt
        person_predicted = non_max_suppression(
            person_predicted,
            conf_thres=0.45,
            iou_thres=0.45,
            classes=None,
            agnostic=True,
            max_det=100,
            nm=0  # 目标检测设置为0
        )
        # rescale boxes to im0 size
        person_result = []
        person_predicted[0][:, :4] = scale_boxes(origin_resize_img.shape[2:], person_predicted[0][:, :4],
                                                 origin_img.shape).round()
        for *xyxy, conf, cls in person_predicted[0]:
            p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            # 判断人体是否在安全帽检测区域内
            feet_point = (int((p1[0] + p2[0]) / 2), p2[1] - 5)  # 矩形底边的中点
            poly = self.poly
            # print("安全帽红线区域", poly, cam_id)
            # poly = self.parent.algorithm_roi.get(cam_id, [])
            if is_in_poly(feet_point, poly):
                person_result.append([p1, p2])
            else:
                pass
                # print("人在检测区域外", p1, p2)

        person_h, person_w = origin_img_shape
        # 按照坐标将人体剪裁，并进行安全帽识别========================================
        results = {"illegal": [], "legal": []}
        hat_start_time = datetime.datetime.now()

        for person_i in person_result:
            # print('person_i',person_i)
            x1, y1, x2, y2 = person_i[0][0], person_i[0][1], person_i[1][0], person_i[1][1]
            x1, y1, x2, y2 = x1 - 10 if x1 - 10 > 0 else 0, y1 - 10 if y1 - 10 > 0 else 0, \
                x2 + 10 if x2 + 10 < person_w else person_w, y2 + 10 if y2 + 10 < person_w else person_w
            image_person = origin_img[y1:y2, x1:x2, :]
            # with self.model_lock:
            # 图像预处理
            image_person_shape_2 = image_person.shape[:2]
            top, bottom, left, right, new_unpad = self.get_info(image_person_shape_2)
            img_person_resize = cv2.resize(image_person, new_unpad, interpolation=cv2.INTER_LINEAR)
            img_person_resize = cv2.copyMakeBorder(img_person_resize, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                                   value=(114, 114, 114))

            img_person_resize = img_person_resize.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img_person_resize = np.ascontiguousarray(img_person_resize)  # contiguous
            img_person_resize = torch.from_numpy(img_person_resize).to(device)
            img_person_resize = img_person_resize.float()  # uint8 to fp16/32
            img_person_resize /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img_person_resize.shape) == 3:
                img_person_resize = img_person_resize[None]  # expand for batch dim

            # 对处理后的图像检测
            # helmet_predicted = self.model(img_person_resize)[0]  # pt
            helmet_predicted = self.helmet_model.predict(img_person_resize)  # engine
            helmet_predicted = non_max_suppression(
                helmet_predicted,
                conf_thres=0.5,
                iou_thres=0.45,
                classes=None,
                agnostic=True,
                max_det=100,
                nm=0  # 目标检测设置为0
            )

            for i, det in enumerate(helmet_predicted):
                if len(det):
                    det[:, :4] = scale_boxes(img_person_resize.shape[2:], det[:, :4],
                                             image_person.shape).round()  # rescale boxes to im0 size
                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

                        # 添加图像明暗、模糊判断================================================================
                        # 袁天娇
                        # 读取图像
                        image1 = image_person[p1[1]:p2[1], p1[0]:p2[0], :]
                        y_, con_ = is_backlit(image1)
                        lap_var = is_blurred(image1)

                        if con_ > 25 and con_ < 40 and y_ > 80:
                            t = ''
                        elif y_ < 30:
                            # 异常数据，较暗
                            t = '_01'
                            # continue
                        elif con_ < 10:
                            t = '_02'
                        elif y_ + con_ < 70:
                            t = '_03'
                        elif (y_ > 50 and y_ < 60) and lap_var > 2500:
                            t = '_04'
                        elif lap_var > 10000:
                            t = '_05'
                        elif lap_var / (y_ + con_) > 80:
                            t = '_06'
                        else:
                            t = ''
                        # ===================================================================================
                        # 大小判断
                        if p2[0] - p1[0] < 25 and p2[1] - p1[1] < 35:
                            t = t + '_s'
                        if (p2[0] - p1[0]) / (p2[1] - p1[1]) < 0.5 or (p2[1] - p1[1]) / (p2[0] - p1[0]) < 0.5:
                            t = t + '_d'

                        cls = type_map.get(names[int(cls)], names[int(cls)])

                        # 对坐标进行转换，将相对人体坐标转化为相对于整张照片
                        p1 = (p1[0] + x1, p1[1] + y1)
                        p2 = (p2[0] + x1, p2[1] + y1)
                        skip = False

                        # 计算当前矩形与之前检测出的矩形交并比，当交并比大于0.45说明两个框重合，去掉一个
                        for results_list in results.values():
                            del_list = list()
                            for ri, results_i in enumerate(results_list):
                                iou_t, iou_be, iou_now = calculate_insert_over_union(
                                    [results_i[2], results_i[3]], [p1, p2])
                                # print('iou_t', round(float(conf), 4), results_i[1], iou_t, iou_be, iou_now)
                                # 说明是同一个框

                                if iou_t > 0.45:
                                    # print('删除进入')
                                    if round(float(conf), 4) > results_i[1]:
                                        # print('删除', ri)
                                        del_list.append(ri)
                                        # del results_list[-ri - 1]
                                    else:
                                        # 不需要保存当前
                                        # print('跳过')
                                        skip = True
                                        break
                                elif iou_be > 0.45:
                                    # print('重复面积1', ri)
                                    # 说明此时对比的两个矩形中，之前矩形的被交集区域占据了0.45以上，去掉
                                    # del results_list[-ri - 1]
                                    del_list.append(ri)
                                elif iou_now > 0.45:
                                    # print('重复面积2')
                                    # 说明此时对比的两个矩形中，当前的被交集区域占据了0.45以上，跳过，不再保存
                                    skip = True
                                    break
                            for del_i in sorted(set(del_list))[::-1]:
                                del results_list[del_i]
                        if skip:
                            # print('跳过')
                            continue

                        if cls in illegal:
                            results["illegal"].append([cls + t, round(float(conf), 4), p1, p2])
                        else:
                            results["legal"].append([cls + t, round(float(conf), 4), p1, p2])
        output = results
        hat_handle_time = datetime.datetime.now()
        logger.info(
            f"安全帽算法接收到图片：{hat_receive_image_time} 安全帽算法开始处理时间：{hat_start_time} 安全帽算法完成处理时间：{hat_handle_time} 花费时间：{hat_handle_time - hat_receive_image_time}  裁剪个数：{len(person_result)}")
        return output
