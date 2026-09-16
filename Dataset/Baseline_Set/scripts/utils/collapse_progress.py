"""Reads a training process's combined stdout/stderr from stdin and emulates real terminal line-overwrite behavior for '\r'-based progress bars (like Ultralytics' tqdm-style epoch bars), so the resulting log file contains only the final, fully-drawn state of each line -- one line per epoch -- instead of every intermediate redraw."""
import re
import sys

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def main():
    stdin = sys.stdin
    stdin.reconfigure(newline="")
    stdout = sys.stdout
    stdout.reconfigure(newline="\n")

    line = []
    col = 0

    def flush_line():
        text = "".join(line).rstrip()
        text = ANSI_RE.sub("", text)
        if text:
            stdout.write(text + "\n")
            stdout.flush()

    while True:
        ch = stdin.read(1)
        if ch == "":
            break
        if ch == "\r":
            col = 0
        elif ch == "\n":
            flush_line()
            line.clear()
            col = 0
        else:
            if col < len(line):
                line[col] = ch
            else:
                line.append(ch)
            col += 1

    flush_line()


if __name__ == "__main__":
    main()
