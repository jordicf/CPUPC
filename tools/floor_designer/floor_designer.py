
import sys
import argparse
from PySide6.QtWidgets import QApplication
from .mainwindow import MainWindow
from typing import Any

def parse_options(prog: str | None = None, args: list[str] | None = None) -> dict[str, Any]:
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = argparse.ArgumentParser(prog=prog, description="Opens the GUI to edit a floorplan.")
    return vars(parser.parse_args(args))


def main(prog: str | None = None, args: list[str] | None = None) -> None:
    """Main function."""
    _ = parse_options(prog, args)
    
    app = QApplication(sys.argv)

    window = MainWindow(app)
    window.show()

    app.exec()



if __name__ == "__main__":
    main()
