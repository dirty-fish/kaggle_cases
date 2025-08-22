import os

def generate_tree(start_path, prefix=""):
    entries = sorted(os.listdir(start_path))
    tree_lines = []
    entries_count = len(entries)
    for idx, entry in enumerate(entries):
        path = os.path.join(start_path, entry)
        connector = "└── " if idx == entries_count - 1 else "├── "
        tree_lines.append(f"{prefix}{connector}{entry}")
        if os.path.isdir(path):
            extension = "    " if idx == entries_count - 1 else "│   "
            tree_lines.extend(generate_tree(path, prefix + extension))
    return tree_lines

def count_dirs_and_files(start_path):
    dir_count = 0
    file_count = 0
    for _, dirs, files in os.walk(start_path):
        dir_count += len(dirs)
        file_count += len(files)
    return dir_count, file_count

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))  # корень скрипта
    output_file = os.path.join(root_dir, "files_structure.txt")

    # Игнорируем сам скрипт и файл вывода
    project_root = root_dir  # можно заменить на нужную папку, например os.path.join(root_dir, "project")
    
    # Формируем дерево
    tree = generate_tree(project_root)

    # Считаем количество директорий и файлов
    dirs, files = count_dirs_and_files(project_root)

    # Записываем результат
    with open(output_file, "w", encoding="utf-8") as f:
        for line in tree:
            f.write(line + "\n")
        f.write(f"\n{dirs} directories, {files} files\n")

    print(f"Файл структуры создан: {output_file}")