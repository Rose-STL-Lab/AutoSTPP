import os
import shutil
import datetime


def create_zip():
    # set the names of the directories to zip
    project_dir = os.getcwd()
    aim_dir = os.path.join(project_dir, '.aim')
    ray_results_dir = os.path.expanduser('~/ray_results')

    # create a timestamp for the archive name
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # create a directory to store the archives
    results_dir = os.path.join(project_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # create a .zip archive of the aim directory
    aim_archive_name = f'aim-{timestamp}.zip'
    aim_archive_path = os.path.join(results_dir, aim_archive_name)
    shutil.make_archive(aim_archive_path[:-4], 'zip', aim_dir)

    # create a .zip archive of the ray_results directory
    ray_archive_name = f'ray_results-{timestamp}.zip'
    ray_archive_path = os.path.join(results_dir, ray_archive_name)
    shutil.make_archive(ray_archive_path[:-4], 'zip', ray_results_dir)

    # return the paths to the created archives
    return aim_archive_path, ray_archive_path


if __name__ == '__main__':
    create_zip()
