import os


def read_lines(filepath: str) -> [str]:
    with open(filepath, encoding='utf-8') as f:
        lines = [line.rstrip('\n').rstrip(' ').lstrip(' ') for line in f]
    return [line for line in lines if len(line) != 0]


def read_lines_dir(path: str) -> [(str, [str])]:
    files = os.listdir(path)

    # return [(file, read_lines(f'{path}/{file}')) for file in files if file == '1994_195.txt']
    return [(file, read_lines(f'{path}/{file}')) for file in files]

