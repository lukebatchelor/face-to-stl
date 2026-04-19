import cv2
import numpy as np
import trimesh
import os


def create_grid_points(width, height, object_width, object_height):
    x = np.linspace(0, object_width, width)
    y = np.linspace(0, object_height, height)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def create_base_points(xx, yy):
    return np.column_stack((xx.flatten(), yy.flatten(), np.zeros_like(xx.flatten())))


def create_top_points(xx, yy, top_z):
    return np.column_stack((xx.flatten(), yy.flatten(), top_z.flatten()))


def create_faces(height, width):
    faces = []

    # Create base and top faces
    for y in range(height - 1):
        for x in range(width - 1):
            i00 = y * width + x
            i10 = y * width + (x + 1)
            i01 = (y + 1) * width + x
            i11 = (y + 1) * width + (x + 1)

            # Base face (clockwise)
            faces.extend([[i00, i01, i10], [i10, i01, i11]])

            # Top face (counter-clockwise)
            i00_top = i00 + width * height
            i10_top = i10 + width * height
            i01_top = i01 + width * height
            i11_top = i11 + width * height
            faces.extend([[i00_top, i10_top, i01_top], [i10_top, i11_top, i01_top]])

    # Add side faces
    for y in range(height - 1):
        # Left side
        i0 = y * width
        i1 = (y + 1) * width
        i0_top = i0 + width * height
        i1_top = i1 + width * height
        faces.extend([[i0, i0_top, i1], [i1, i0_top, i1_top]])

        # Right side
        i0 = (y + 1) * width - 1
        i1 = (y + 2) * width - 1
        i0_top = i0 + width * height
        i1_top = i1 + width * height
        faces.extend([[i0, i1, i0_top], [i1, i1_top, i0_top]])

    for x in range(width - 1):
        # Front side
        i0 = x
        i1 = x + 1
        i0_top = i0 + width * height
        i1_top = i1 + width * height
        faces.extend([[i0, i1, i0_top], [i1, i1_top, i0_top]])

        # Back side
        i0 = (height - 1) * width + x
        i1 = (height - 1) * width + (x + 1)
        i0_top = i0 + width * height
        i1_top = i1 + width * height
        faces.extend([[i0, i0_top, i1], [i1, i0_top, i1_top]])

    return faces


def create_and_process_mesh(points, faces, target_reduction):
    mesh = trimesh.Trimesh(vertices=points, faces=faces)
    mesh = mesh.simplify_quadric_decimation(target_reduction)
    mesh.fix_normals()

    # Rotate the mesh
    # rotation_matrix = trimesh.transformations.rotation_matrix(
    #     np.radians(-90), [1, 0, 0]
    # )
    # mesh.apply_transform(rotation_matrix)
    # rotation_matrix = trimesh.transformations.rotation_matrix(
    #     np.radians(180), [0, 1, 0]
    # )
    # mesh.apply_transform(rotation_matrix)

    return mesh


def generate_stl(
    height_map_image,
    color_palette,
    base_height=5.0,
    image_height=2.0,
    object_height=40,
    object_width=70,
    target_reduction=0.9,
):
    height, width = height_map_image.shape[:2]
    print("height, width", height, width)
    print("object_height, object_width", object_height, object_width)

    xx, yy = create_grid_points(width, height, object_width, object_height)
    base_points = create_base_points(xx, yy)

    color_heights = base_height + np.linspace(0, image_height, len(color_palette))
    color_to_height = {
        tuple(color): height for color, height in zip(color_palette, color_heights)
    }

    top_z = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            color = tuple(height_map_image[i, j])
            top_z[i, j] = color_to_height.get(color, base_height)

    top_points = create_top_points(xx, yy, top_z)

    points = np.vstack((base_points, top_points))
    faces = create_faces(height, width)

    mesh = create_and_process_mesh(points, faces, target_reduction)

    return mesh


if __name__ == "__main__":
    # Load the image mask
    input_path = os.path.join("..", "client", "public", "danny_mask.png")
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Unable to load image from {input_path}")
        exit(1)

    # Generate STL
    mesh = generate_stl(
        img, base_height=5, image_height=2, target_reduction=0.9, bend_factor=8
    )

    # Save the STL file
    output_path = os.path.join("..", "client", "public", "output.stl")
    mesh.export(output_path)

    print(f"STL file generated and saved to {output_path}")
    print(f"Final number of faces: {len(mesh.faces)}")
