# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 23:25
# @Author  : FengZeTao zetao.feng@hand-china.com
# @File    : tool.py
# 如遇到看不懂且无备注的部分，请联系相关人
import os
import re
from random import random

import cv2
import numpy as np
from datetime import date, datetime
from PIL import Image, ImageEnhance
import pyzbar.pyzbar as pyzbar


def convert2yolov5(size, box):
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    x = x_center / size[0]
    y = y_center / size[1]

    w = (box[2] - box[0]) / size[0]
    h = (box[3] - box[1]) / size[1]

    return round(x.item(), 6), round(y.item(), 6), round(w.item(), 6), round(h.item(), 6)


def get_num(string):
    string.replace('l', '1').replace('I', '1').replace('i', '1')
    comp = re.compile('-?[0-9]\d*')
    return ''.join(comp.findall(string))


def get_tax(string):
    comp = re.compile('-?[0-9]\d*[a-zA-Z]*')
    return ''.join(comp.findall(string))


def get_title(string):
    comp = re.compile('-?[^:：]*')
    return ''.join(comp.findall(string))


def get_addr_bank(string):
    comp = re.compile('[0-9\-]*$')
    pre = comp.split(string.replace(':', '').replace('：', ''))[0]
    return f'{pre} {string[len(pre):]}'


def get_float(string):
    if not string:
        return '¥ 0.00'
    try:
        comp = re.compile('-?[0-9]\d*\.*')
        str_list = list(''.join(comp.findall(string)))
        if str_list[0] == '-' and str_list[1] == '0' and len(str_list) > 2:
            str_list[1] = '8'
        elif str_list[0] == '0' and len(str_list) > 1:
            str_list[0] = '8'

        return π
    except:
        return '¥ 0.00'

def get_amount(string):
    """
    提取金额并最大化容错：
    - 统一全角/半角字符、货币符号与中文“元/圆”
    - 修正常见 OCR 误识别（O→0、S→5 等）
    - 支持货币符号在前/后、括号表示负数、末尾减号
    返回格式统一为 `¥ xx.xx`，找不到有效数字时返回 `¥ 0.00`。
    """
    if not string:
        return '¥ 0.00'
    try:
        raw = str(string).strip()
        # 全角转半角并统一货币/中文符号
        trans_map = {
            '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
            '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
            '，': ',', '．': '.', '－': '-', '＋': '+', '﹣': '-', '–': '-', '—': '-',
            '￥': '¥', '元': '¥', '圆': '¥', ' ': ''
        }
        raw = raw.translate(str.maketrans(trans_map))

        # 常见 OCR 误识别字符替换
        raw = raw.translate(str.maketrans({
            'O': '0', 'o': '0', 'D': '0',
            'S': '5', 'B': '8', 'l': '1', 'I': '1',
            'Y': '¥'  # 兼容把 ¥ 识别为 Y
        }))

        # 去掉明显无关的符号（星号、装饰符号、竖线等）
        raw = re.sub(r'[★☆※*•·●⊙◎¤■◆◇▪▎▏▍▌▋▊▉|｜~`^_=+<>《》〈〉【】\[\]{}（）()]', '', raw)
        raw = re.sub(r'\s+', '', raw)  # 金额中去除所有空白

        # 负数标记：括号或末尾减号
        is_bracket_negative = '(' in raw and ')' in raw
        has_trailing_minus = bool(re.search(r'-\s*$', raw))

        def parse_candidates(pattern):
            vals = []
            for m in re.finditer(pattern, raw, flags=re.IGNORECASE):
                num = m.group(1).replace(',', '')
                try:
                    vals.append(float(num))
                except ValueError:
                    continue
            return vals

        # 优先：带货币符号的匹配
        currency_vals = parse_candidates(r'(?:¥|RMB|CNY)\s*([-+]?\d[\d,]*(?:\.\d+)?)')
        # 次优：数字后跟货币符号
        suffix_vals = parse_candidates(r'([-+]?\d[\d,]*(?:\.\d+)?)(?=\s*(?:¥|RMB|CNY))')
        # 兜底：任意数字串
        generic_vals = parse_candidates(r'([-+]?\d[\d,]*(?:\.\d+)?)')

        candidates = currency_vals or suffix_vals or generic_vals
        if not candidates:
            return '¥ 0.00'

        # 选择最有可能的金额：优先最后出现的金额，其次绝对值最大
        value = candidates[-1]
        if len(candidates) > 1 and abs(max(candidates, key=abs)) != abs(value):
            # 如果列表里有更大的绝对值，则用最大绝对值防止截取错位
            value = max(candidates, key=abs)

        if is_bracket_negative and value > 0:
            value = -value
        if has_trailing_minus and value > 0:
            value = -value

        return f'¥ {value:.2f}'
    except Exception:
        return '¥ 0.00'


def get_page(string):
    try:
        string = string.replace('|', '1').replace('I', '1').replace('l', '1')
        comp = re.search('第(.*)页/共(.*)页', string)
        return '{0}/{1}'.format(comp.group(1) or 1, comp.group(2) or 1)
    except:
        try:
            comp = re.compile('-?[0-9]\d*')
            return '{0}/{1}'.format(*comp.findall(string))
        except:
            return '-1/-1'


def get_date(string):
    try:
        raw = str(string).strip()
        # 统一全角/半角并修正常见 OCR 误识别，去掉无关符号
        trans_map = {
            '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
            '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
            '－': '-', '–': '-', '—': '-', '﹣': '-', '／': '/', '。': '.',
            '年': None, '月': None, '日': None, '号': None,
            ' ': None, '\t': None, '\n': None
        }
        raw = raw.translate(str.maketrans(trans_map))
        raw = raw.translate(str.maketrans({
            'O': '0', 'o': '0', 'D': '0',
            '|': '1', 'I': '1', 'l': '1'
        }))
        raw = re.sub(r'[★☆※*•·●⊙◎¤■◆◇▪▎▏▍▌▋▊▉|｜~`^_=+<>《》〈〉【】\[\]{}（）()]', '', raw)
        raw = re.sub(r'\s+', '', raw)

        date_str = get_num(raw)
        date_len = len(date_str)
        if date_len < 8:
            curr_date = date.today().strftime('%Y%m%d')
            date_str = curr_date[:8 - date_len] + date_str
        return datetime.strptime(date_str, '%Y%m%d').strftime('%Y年%m月%d日')
    except:
        return string


def remove_red_seal(image):
    """
    去除红色印章
    """
    # 获得红色通道
    blue_c, green_c, red_c = cv2.split(image)
    # 多传入一个参数cv2.THRESH_OTSU，并且把阈值thresh设为0，算法会找到最优阈值
    thresh, ret = cv2.threshold(red_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 实测调整为95%效果好一些
    filter_condition = int(thresh * 0.95)
    _, red_thresh = cv2.threshold(red_c, filter_condition, 255, cv2.THRESH_BINARY)
    # 把图片转回 3 通道
    result_img = np.expand_dims(red_thresh, axis=2)
    result_img = np.concatenate((blue_c, green_c, result_img), axis=-1)
    return result_img


def _vat_qrcode(barcodeData, invoice):
    barcode_list = barcodeData.split(',')
    invoice['invoice_type'] = barcode_list[1].strip()
    invoice['invoice_code'] = barcode_list[2].strip()
    invoice['invoice_number'] = barcode_list[3].strip()
    if barcode_list[1] in ('31', '32'):
        invoice['amount_with_tax'] = get_float(barcode_list[4])
        invoice['total_amount'] = '¥ 0.00'
        invoice['tax'] = '¥ 0.00'
    else:
        invoice['total_amount'] = get_float(barcode_list[4])
        invoice['tax'] = '¥ 0.00'
        invoice['amount_with_tax'] = '¥ 0.00'
    invoice['billing_date'] = datetime.strptime(barcode_list[5], '%Y%m%d').strftime('%Y年%m月%d日')
    if invoice['invoice_type'] in ('04', '10'):
        invoice['check_code'] = get_num(barcode_list[6])


def _stock_qrcode(barcodeData, stock):
    barcode_list = [x.split(':')[1] for x in barcodeData.split(' ')]
    if len(barcode_list[0]) in (11, 12):
        code = get_num(barcode_list[0])
        invoice_number = barcode_list[2].replace('/', '、')
    else:
        code = get_num(barcode_list[2])
        invoice_number = barcode_list[0].replace('/', '、')
    stock['code'] = code
    stock['total_amount'] = get_float(barcode_list[1])
    stock['invoice_number'] = invoice_number
    stock['page'] = get_page(barcode_list[4])


def get_qrcode_data(img, index=0):
    if index > 3:
        return ''
    optimization = ["img = ImageEnhance.Brightness(img).enhance(2.0)",
                    "img = ImageEnhance.Sharpness(img).enhance(1.5)",
                    "img = ImageEnhance.Contrast(img).enhance(2.0)"]
    for x in optimization[0:index]:
        dic = {"img": img, "ImageEnhance": ImageEnhance}
        exec(x, dic)
        img = dic.get('img')
    barcodes = pyzbar.decode(img.convert('L'))
    if barcodes:
        return barcodes[0].data.decode("utf-8")
    else:
        return get_qrcode_data(img, index + 1)


def qrcode_pyzbar(image, val, is_stock=False):
    try:
        img = Image.fromarray(image)
        #
        # #锐利化
        # #增加对比度
        img = img.convert('L')  # 灰度化
        barcodeData = get_qrcode_data(img)
        if barcodeData:
            print(barcodeData)
            if is_stock:
                _stock_qrcode(barcodeData, val)
            else:
                _vat_qrcode(barcodeData, val)
            val['qrcode'] = True
            return True
        else:
            return False
    except Exception as e:
        return False


if __name__ == '__main__':
    invoice = {}
    path = '/Users/fengzetao/PycharmProjects/ocr/code/images/invoice/qrcode.png'
    image = cv2.imread(path)
    qrcode_pyzbar(image, invoice, False)
    print(invoice)
