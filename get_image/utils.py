import subprocess
import os

from django.conf import settings
from subprocess import check_output, CalledProcessError, STDOUT, run, call


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

# system_call
def system_call(command, root):
    """ 
    params:
        command: list of strings, ex. `["ls", "-l"]`
    returns: output, success
    """
    try:
        output = check_output(command, stderr=STDOUT, cwd=root).decode()
        success = True 
    except CalledProcessError as e:
        output = e.output.decode()
        success = False
    return output, success