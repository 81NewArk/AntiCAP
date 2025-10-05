import AntiCAP

if __name__ == '__main__':

    # 初始化
    Atc = AntiCAP.Handler()

    # 文字类验证码 字母 数字 汉字
    result = Atc.OCR(img_base64="")

    # 算术类验证码
    result = Atc.Math(img_base64="")

    # 图标侦测
    result = Atc.Detection_Icon(img_base64="")

    # 图标点选 按序输出
    result = Atc.ClickIcon_Order(order_img_base64="",target_img_base64="")

    # 汉字侦测
    result = Atc.Detection_Text(img_base64="")

    # 文字点选 按序输出
    result = Atc.ClickText_Order(order_img_base64="", target_img_base64="")

    # 缺口滑块
    result = Atc.Slider_Match(target_base64="",background_base64="")

    # 阴影滑块
    result = Atc.Slider_Comparison(target_base64="",background_base64="")

    # 图像相似度对比 图片中的文字
    result= Atc.Compare_Image_Similarity(image1_base64="",image2_base64="")

    # 单图旋转
    result = Atc.Single_Rotate(img_base64="")

    # 双旋转
    result= Atc.Compare_Image_Similarity(image1_base64="", image2_base64="")


    print(result)
















