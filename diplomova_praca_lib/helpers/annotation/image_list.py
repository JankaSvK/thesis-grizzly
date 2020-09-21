import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    files = []


    for dir, subdirs, file in os.walk(args.root):
        for name in file:
            path = os.path.join(dir, name)
            path = os.path.relpath(path, args.root)
            files.append(path + "\n")


    with open(args.output, "w") as out:
        files = sorted(files)
        out.writelines(files)

if __name__ == '__main__':
    main()
