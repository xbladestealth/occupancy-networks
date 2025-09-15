import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非GUIバックエンド
import matplotlib.pyplot as plt
import os

output_dir = "output"
grid_size = 64
bounding_box_min = [-5, -5, -2.5]
bounding_box_max = [5, 5, 2.5]

occupancy_grid = np.load(os.path.join(output_dir, "occupancy_grid.npy"))
print(f"Loaded occupancy grid with shape: {occupancy_grid.shape}")

def visualize_voxel_grid_slice(occupancy_grid, grid_size, bounds_min, bounds_max, output_dir, slice_axis='z', slice_index=None):
    if slice_index is None:
        slice_index = grid_size // 2

    if slice_axis == 'z':
        slice_data = occupancy_grid[:, :, slice_index]  # [i, j] -> [x, y]
        xlabel, ylabel = 'X', 'Y'
        extent = [bounds_min[0], bounds_max[0], bounds_min[1], bounds_max[1]]  # X, Y
    elif slice_axis == 'y':
        slice_data = occupancy_grid[:, slice_index, :]  # [i, k] -> [x, z]
        xlabel, ylabel = 'X', 'Z'
        extent = [bounds_min[0], bounds_max[0], bounds_min[2], bounds_max[2]]
    elif slice_axis == 'x':
        slice_data = occupancy_grid[slice_index, :, :]  # [j, k] -> [y, z]
        xlabel, ylabel = 'Y', 'Z'
        extent = [bounds_min[1], bounds_max[1], bounds_min[2], bounds_max[2]]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(slice_data, cmap='binary', origin='lower', extent=extent)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Voxel Grid Slice ({slice_axis.upper()}={slice_index})')

    output_path = os.path.join(output_dir, f"voxel_grid_slice_{slice_axis}_{slice_index}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"2D voxel grid slice saved to {output_path}")

# Z軸スライスを可視化（Bird's-eye View）
visualize_voxel_grid_slice(occupancy_grid, grid_size, bounding_box_min, bounding_box_max, output_dir, slice_axis='z')

# 3D可視化（オプション）
def visualize_voxel_grid_3d(occupancy_grid, grid_size, bounds_min, bounds_max, output_dir):
    """
    Visualize the 3D occupancy grid and save it as an image.
    
    Parameters:
    - occupancy_grid: NumPy array of shape (grid_size, grid_size, grid_size) with binary values (0 or 1)
    - grid_size: Integer, size of the grid in each dimension (e.g., 64 for a 64x64x64 grid)
    - bounds_min: List of [x_min, y_min, z_min] for the bounding box
    - bounds_max: List of [x_max, y_max, z_max] for the bounding box
    - output_dir: Directory to save the output image
    """
    # Calculate voxel size for each dimension
    voxel_size = [(bmax - bmin) / grid_size for bmax, bmin in zip(bounds_max, bounds_min)]
    
    # Create a figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Get indices of occupied voxels
    occupied = np.where(occupancy_grid == 1)
    x, y, z = occupied
    
    # Convert voxel indices to world coordinates for visualization
    x_world = bounds_min[0] + (x + 0.5) * voxel_size[0]
    y_world = bounds_min[1] + (y + 0.5) * voxel_size[1]
    z_world = bounds_min[2] + (z + 0.5) * voxel_size[2]
    
    # Plot occupied voxels
    ax.scatter(x_world, y_world, z_world, c='blue', marker='s', s=10)
    
    # Set axis labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(bounds_min[0], bounds_max[0])
    ax.set_ylim(bounds_min[1], bounds_max[1])
    ax.set_zlim(bounds_min[2], bounds_max[2])
    
    # Save the plot
    output_path = os.path.join(output_dir, "voxel_grid_3d.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"3D voxel grid visualization saved to {output_path}")

visualize_voxel_grid_3d(occupancy_grid, grid_size, bounding_box_min, bounding_box_max, output_dir)