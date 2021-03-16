import requests
# 图片地址
img_url = "http://api.map.baidu.com/staticimage?center=四川成都新津锦绣路幸福路路口&zoom=16"
img = requests.get(img_url)
f = open('test.jpg', 'ab')  # 存储图片，多媒体文件需要参数b（二进制文件）
f.write(img.content)  # 多媒体存储content
f.close()
