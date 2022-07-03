import os
import subprocess
from unittest import result
import cv2
import base64

from django.http import HttpResponse, JsonResponse
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from . import utils


@api_view(['GET'])
def apiTest(request):
    if request.method == 'GET':
        return Response(str('hello!, GET!'))
    return Response(str("None!"))

# @api_view([])
def getImage(request):
    if request.method  == 'POST':

        # return HttpResponse("Connected!")
        # 先清空对应路径
        utils.clean_files()
        img_1 = request.FILES.get('img_1')
        img_name = img_1.name
        mobile = os.path.splitext(img_name)[0]
        ext = os.path.splitext(img_name)[1]
        img_name = f'{mobile}{ext}'
        img_path = os.path.join(settings.CLOTH_UPLOAD, img_name)
     
        with open(img_path, 'ab') as fp:
        # 如果上传的图片非常大，就通过chunks()方法分割成多个片段来上传
            for chunk in img_1.chunks():
                fp.write(chunk)
        
        
        img_2 = request.FILES.get('img_2')
        img_name = img_2.name
        mobile = '1'
        ext = os.path.splitext(img_name)[1]
        img_name = f'avatar-{mobile}{ext}'
        img_path = os.path.join(settings.PEOPLE_UPLOAD, img_name)
        with open(img_path, 'ab') as fp:
            # 如果上传的图片非常大，就通过chunks()方法分割成多个片段来上传
            for chunk in img_2.chunks():
                fp.write(chunk) 
        
        rc = subprocess.call(["python", "test.py"], cwd=("/home/lzn/futurama/f_back/get_image/PF_AFN/"))
        # if rc == 0:
        try:
            if os.listdir(settings.RESULT_PATH) == []:
                return HttpResponse(str("No result!" + str(rc)))
            result_img = cv2.imread(settings.RESULT_PATH + os.listdir(settings.RESULT_PATH)[0])
        except FileNotFoundError:
            return HttpResponse(status_code=400, content=str('Generate Failed' + str(rc)))
        img_decoded=base64.b64encode(cv2.imencode('.jpg',result_img)[1]).decode()
        data = {
            'file_name': os.listdir(settings.RESULT_PATH)[0],
            'file_content': img_decoded,
        }
        return JsonResponse(data)
        # else:
            # return HttpResponse(str('Fail To Generate Img'))

def getPath(request):
    if request.method == 'GET':
        return HttpResponse(str(settings.PEOPLE_UPLOAD))