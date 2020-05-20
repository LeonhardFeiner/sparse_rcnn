# %%
from functools import partial
from multiprocessing import Pool
import os
from tempfile import NamedTemporaryFile

# %%


def process_scene_raw(source_path, destination_path, cmd_raw, scene_id):
    input_path = source_path.format(scene_id)
    output_path = destination_path.format(scene_id)
    if os.path.exists(output_path):
        return
    cmd = cmd_raw.format(input_path, output_path)
    print(cmd)
    print(scene_id, os.system(cmd))  


def process_scene_list(
        source_path, destination_path, command, scene_ids, processes=None):

    if processes in (0, False):
        for scene_id in scene_ids:
            process_scene_raw(
                source_path, destination_path, command, scene_id)
    else:
        process_scene = partial(
            process_scene_raw, source_path, destination_path, command)
        p = Pool(processes=processes)
        p.map(process_scene, scene_ids)
        p.close()
        p.join()


# %%

def add_normals(
        meshlab_path, source_path, destination_path, scene_ids,
        processes=None):
    add_normals_command = ' -i {} -o {} -m vc vn'
    meshlab_command = meshlab_path + add_normals_command
    process_scene_list(
        source_path, destination_path, meshlab_command, scene_ids, processes)


def point_cloud_selection(
        radius, meshlab_path, source_path, destination_path, scene_ids,
        processes=None):
    filter_text = (
        f"""<!DOCTYPE FilterScript>
        <FilterScript>
        <filter name="Point Cloud Simplification">
        <Param name="SampleNum" value="0" type="RichInt"/>
        <Param name="Radius" value="{radius}" max="10.1558" min="0" type="RichAbsPerc"/>
        <Param name="BestSampleFlag" value="true" type="RichBool"/>
        <Param name="BestSamplePool" value="10" type="RichInt"/>
        <Param name="ExactNumFlag" value="false" type="RichBool"/>
        </filter>
        </FilterScript>""")

    with NamedTemporaryFile(mode='w+') as filter_file:
        filter_file.write(filter_text)
        filter_file_path = filter_file.name

        meshlab_command = (
            f'{meshlab_path} -i {{}} -o {{}} -m vc vn -s {filter_file_path}')

        process_scene_list(
            source_path, destination_path, meshlab_command, scene_ids,
            processes)