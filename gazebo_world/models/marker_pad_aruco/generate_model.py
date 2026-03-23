import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "model.sdf")

STEP = 0.6  # WIDTH (0.5) + SPACE (0.1)
COORDS = [-STEP, 0, STEP]  # -0.6, 0, 0.6

SDR_START = """<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='marker_pad_aruco'>
    <pose>0 0 0.0001 0 0 0</pose>
    <static>true</static>
    
    <!-- Asphalt patch -->
    <link name='marker_pad_aruco/base_link'>
      <pose>0 0 -0.0001 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>5 5</size>
          </plane>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>5 5</size>
          </plane>
        </geometry>
        <material>
          <diffuse>1 1 1 1</diffuse>
          <specular>1 1 1 1</specular>
          <pbr>
            <metal>
              <albedo_map>materials/textures/tarmac.png</albedo_map>
              <normal_map>materials/textures/tarmac.png</normal_map>
            </metal>
          </pbr>
        </material>
      </visual>
    </link>

    <!-- MARKERS -->
"""

MARKER_TEMPLATE = """
    <link name="marker_pad_aruco/marker_{idx}">
    <pose>{x} {y} 0 0 0 0</pose>
      <collision name="marker_pad_aruco/marker_{idx}_collision">
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>0.5 0.5</size>
          </plane>
        </geometry>
      </collision>
      <visual name="marker_pad_aruco/marker_{idx}_visual">
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>0.5 0.5</size>
          </plane>
        </geometry>
        <material>
          <diffuse>1 1 1 1</diffuse>
          <specular>1 1 1 1</specular>
          <pbr>
            <metal>
              <albedo_map>materials/textures/markers/{tex}.png</albedo_map>
              <normal_map>materials/textures/markers/{tex}.png</normal_map>
            </metal>
          </pbr>
        </material>
      </visual>
    </link>
    <joint name="marker_pad_aruco/marker_{idx}_joint" type="fixed">
      <parent>marker_pad_aruco/base_link</parent>
      <child>marker_pad_aruco/marker_{idx}</child>
    </joint>
"""

SDR_END = """
  </model>
</sdf>"""

def generate_sdf():
    content = SDR_START
    marker_idx = 0
    # Проход по сетке 3x3 (row - Y, col - X)
    for row in COORDS:
        for col in COORDS:
            # Текстуры: 101.png ... 109.png
            tex_num = 101 + marker_idx
            content += MARKER_TEMPLATE.format(
                idx=marker_idx,
                x=col,
                y=row,
                tex=tex_num
            )
            marker_idx += 1
    
    content += SDR_END
    return content

if __name__ == '__main__':
    sdf_content = generate_sdf()
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(sdf_content)
    