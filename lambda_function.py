import cv2
import base64
import json
import numpy as np


def convert_b64_string_to_bynary(s):
    """base64をデコードする"""
    return base64.b64decode(s.encode("UTF-8"))

def base64_to_cv2(image_base64):
    """base64 image to cv2"""
    image_bytes = base64.b64decode(image_base64)
    np_array = np.fromstring(image_bytes, np.uint8)
    image_cv2 = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image_cv2


def cv2_to_base64(image_cv2):
    """cv2 image to base64"""
    image_bytes = cv2.imencode('.jpg', image_cv2)[1].tostring()
    image_base64 = base64.b64encode(image_bytes).decode()
    return image_base64

def sub_color(src, K):
    # 次元数を1落とす
    Z = src.reshape((-1, 3))

    # float32型に変換
    Z = np.float32(Z)

    # 基準の定義
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # K-means法で減色
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # UINT8に変換
    center = np.uint8(center)

    res = center[label.flatten()]

    # 配列の次元数と入力画像と同じに戻す
    return res.reshape((src.shape))


def anime_filter(img, K):
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # ぼかしでノイズ低減
    edge = cv2.blur(gray, (3, 3))

    # Cannyアルゴリズムで輪郭抽出
    edge = cv2.Canny(edge, 50, 150, apertureSize=3)

    # 輪郭画像をRGB色空間に変換
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    # 画像の領域分割
    img = sub_color(img, K)

    # 差分を返す
    return cv2.subtract(img, edge)




def lambda_handler(event, context):
    # requestbodeyの中のjsonはeventに辞書型に変化されて保存されている
    # なので、eventの中には {"mypng": "base64でエンコードされた文字列"}が入力される想定。
    base_64ed_image = event['mypng']
    # バケット作成を作成してbynary変換して保存する。
    cvimg = base64_to_cv2

    anime = anime_filter(cvimg, 30)
    
    body = cv2_to_base64(anime)
    # とりあえずOKを返す。
    return body