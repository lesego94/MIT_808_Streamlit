import os

# Set the directory you want to start from
root_dir = '/var/folders/d9/zs98dym14hj8prgt5vm0p3580000gn/T/'

for dir_name, subdirs, files in os.walk(root_dir):
    for filename in files:
        if filename.endswith('.jpg'):
            # construct full file path
            file_path = os.path.join(dir_name, filename)
            # remove the file
            os.remove(file_path)
            print(f'Removed {file_path}')
