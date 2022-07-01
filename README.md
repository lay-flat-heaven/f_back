# f_back
Futrauma backend

PF_AFN 模型下需要`checkpoint`文件夹才可以运行其中路径为：
checkpoints/PFAFN/XXX.pth

详细接口在`urls.py`中
通过向 root/get_image/getImage 接口发送POST请求('img_1'(cloth), 'img_2'(people))
返回Json格式:
{
'file_name':<file_name>,
'file_content':<base64 code of img>
}
