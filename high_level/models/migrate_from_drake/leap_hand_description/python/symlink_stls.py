import os
import sys


def create_relative_symlinks(source_folder, target_folder):
    # Ensure the source folder ends with a '/'
    source_folder = os.path.join(source_folder, "")

    # Get all STL files in the target folder
    target_stl_files = {file for file in os.listdir(target_folder) if file.lower().endswith(".stl")}

    # Traverse the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(".stl"):
                # Check if the file exists in the target folder
                if file in target_stl_files:
                    target_file_path = os.path.join(target_folder, file)
                    symlink_path = os.path.join(root, file)

                    # Calculate the relative path from the source file location to the target file
                    relative_path = os.path.relpath(target_file_path, root)

                    # Remove the existing file if it's not a symlink
                    if not os.path.islink(symlink_path):
                        os.remove(symlink_path)

                    # Create a relative symlink
                    os.symlink(relative_path, symlink_path)
                    print(f"Relative symlink created for {file} at {symlink_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <target_folder> <source_folder>")
    else:
        target_folder = sys.argv[1]
        source_folder = sys.argv[2]
        create_relative_symlinks(source_folder, target_folder)
