import blenderproc as bproc
import numpy as np
import os
import bpy
import trimesh

# 空間を単一の立方体メッシュとして作成
def create_space_mesh(bounds_min, bounds_max):
    center = [(bmax + bmin) / 2 for bmax, bmin in zip(bounds_max, bounds_min)]
    size = [bmax - bmin for bmax, bmin in zip(bounds_max, bounds_min)]
    bpy.ops.mesh.primitive_cube_add(size=size[0], location=center)
    space_mesh = bpy.context.object
    return space_mesh

# ブーリアン演算で占有部分を抽出
def apply_boolean(space_mesh, obj):
    bpy.context.view_layer.objects.active = space_mesh
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.modifier_add(type='BOOLEAN')
    modifier = space_mesh.modifiers[-1]
    modifier.operation = 'INTERSECT'
    modifier.object = obj.blender_obj
    override = bpy.context.copy()
    override['active_object'] = space_mesh
    override['object'] = space_mesh
    with bpy.context.temp_override(**override):
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    return space_mesh

# メッシュを三角形化
def triangulate_mesh(mesh_obj):
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.modifier_add(type='TRIANGULATE')
    modifier = mesh_obj.modifiers[-1]
    override = bpy.context.copy()
    override['active_object'] = mesh_obj
    override['object'] = mesh_obj
    with bpy.context.temp_override(**override):
        bpy.ops.object.modifier_apply(modifier=modifier.name)

# カスタムボクセル化関数（trimeshを使用）
def voxelize_object(space_mesh, grid_size, bounds_min, bounds_max):
    triangulate_mesh(space_mesh)
    vertices = [v.co for v in space_mesh.data.vertices]
    faces = [[i for i in p.vertices] for p in space_mesh.data.polygons]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # ボクセルサイズ（pitch）を計算
    pitch = (bounds_max[0] - bounds_min[0]) / grid_size

    try:
        # binvoxメソッドを試す（originをサポート）
        origin = [(bmax + bmin) / 2 for bmax, bmin in zip(bounds_max, bounds_min)]
        voxel_grid = mesh.voxelized(pitch=pitch, method='binvox', origin=origin).fill()
        occupancy_grid = voxel_grid.matrix.astype(np.uint8)
    except Exception as e:
        print(f"binvox failed: {e}. Falling back to default voxelization.")
        # フォールバック: デフォルトボクセル化
        voxel_grid = mesh.voxelized(pitch=pitch).fill()
        occupancy_grid = voxel_grid.matrix.astype(np.uint8)

    # [y, x, z]を[x, y, z]に変換（インデックス順を変更）
    occupancy_grid = np.transpose(occupancy_grid, (1, 0, 2))  # [j, i, k] -> [i, j, k]

    # グリッドをシフトして中心を(i=32, j=32, k=32)に
    target_shape = (grid_size, grid_size, grid_size)
    shifted_grid = np.zeros(target_shape, dtype=np.uint8)
    offset = np.array([grid_size // 2, grid_size // 2, grid_size // 2])
    for i, j, k in np.ndindex(occupancy_grid.shape):
        new_i = i + offset[0] - occupancy_grid.shape[0] // 2
        new_j = j + offset[1] - occupancy_grid.shape[1] // 2
        new_k = k + offset[2] - occupancy_grid.shape[2] // 2
        if 0 <= new_i < grid_size and 0 <= new_j < grid_size and 0 <= new_k < grid_size:
            shifted_grid[new_i, new_j, new_k] = occupancy_grid[i, j, k]
    occupancy_grid = shifted_grid

    # グリッドサイズを調整
    if occupancy_grid.shape != target_shape:
        occupancy_grid = np.pad(
            occupancy_grid[:grid_size, :grid_size, :grid_size],
            [(0, max(0, grid_size - s)) for s in occupancy_grid.shape],
            mode='constant',
            constant_values=0
        )

    return occupancy_grid

# 出力ディレクトリを作成
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# BlenderProcの初期化
bproc.init()

# シンプルなオブジェクト（MONKEY）を作成
obj = bproc.object.create_primitive("MONKEY")

# ポイントライトを追加
light = bproc.types.Light()
light.set_location([2, -2, 0])
light.set_energy(300)

# カメラをオブジェクトの前方に設定
cam_pose = bproc.math.build_transformation_mat([0, -5, 0], [np.pi / 2, 0, 0])
bproc.camera.add_camera_pose(cam_pose)

# シーンのレンダリング（RGBと深度マップ）
data = bproc.renderer.render()

# レンダリング結果をHDF5ファイルに保存
bproc.writer.write_hdf5(output_dir + "/", data)
print(f"Rendered data saved to {output_dir}/")

# Blenderのレンダリング設定を使用してPNG画像を保存
# RGB画像の保存
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = os.path.join(output_dir, "rgb.png")
bpy.ops.render.render(write_still=True)  # RGB画像をレンダリングして保存

# ボクセル化: 空間を単一メッシュとしてボクセル化
grid_size = 64
bounding_box_min = [-5, -5, -2.5]  # x, y, z min
bounding_box_max = [5, 5, 2.5]  # x, y, z max

# 空間メッシュを作成し、ブーリアン演算を適用
space_mesh = create_space_mesh(bounding_box_min, bounding_box_max)
space_mesh = apply_boolean(space_mesh, obj)

# 占有グリッドを生成
occupancy_grid = voxelize_object(space_mesh, grid_size, bounding_box_min, bounding_box_max)

# 占有グリッドの座標を確認（デバッグ用）
voxel_size = [(bmax - bmin) / grid_size for bmax, bmin in zip(bounding_box_max, bounding_box_min)]
occupied_indices = np.where(occupancy_grid == 1)
for i, j, k in zip(occupied_indices[0], occupied_indices[1], occupied_indices[2]):
    x = bounding_box_min[0] + (i + 0.5) * voxel_size[0]  # i -> X
    y = bounding_box_min[1] + (j + 0.5) * voxel_size[1]  # j -> Y
    z = bounding_box_min[2] + (k + 0.5) * voxel_size[2]  # k -> Z
    print(f"Index: ({i}, {j}, {k}), World: ({x:.2f}, {y:.2f}, {z:.2f})")

# 占有グリッドを保存
np.save(os.path.join(output_dir, "occupancy_grid.npy"), occupancy_grid)
print(f"Occupancy grid saved to {output_dir}/occupancy_grid.npy")