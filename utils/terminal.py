import sys


def clear_previous_line():
    sys.stdout.write("\033[F")  # move cursor to the previous line
    sys.stdout.write("\033[K")  # clear line from the cursor
    print("\r", end="")         # move the cursor to the beginning of the line


def clear_current_line():
    sys.stdout.write("\033[2K") # clear the whole line
    print("\r", end="")         # move the cursor to the beginning of the line
