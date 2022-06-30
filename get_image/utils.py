import subprocess
import os

from django.conf import settings

# subprocess.call(["python", "test.py"], cwd="PF_AFN")
def clean_files():
    path_list = [settings.CLOTH_UPLOAD,
                 settings.PEOPLE_UPLOAD,
                 settings.MASK_PATH,
                 settings.RESULT_PATH
                ]
    for root_path in path_list:
        for fp_name in os.listdir(root_path):
            os.remove(os.path.join(root_path, fp_name))

def generate_result():
    rc = subprocess.call(["python", "test.py"], cwd="PF_AFN")
    return rc