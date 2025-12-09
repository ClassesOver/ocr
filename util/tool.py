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
        date_str = get_num(string)
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
