import subprocess
import os


def main():
    """
    Finds and runs the Streamlit application.
    """
    # Get the directory of the current script (cli.py)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Construct the full path to app.py
    app_path = os.path.join(dir_path, "app.py")

    # Command to run streamlit
    command = ["streamlit", "run", app_path]

    # Run the command
    subprocess.run(command)


if __name__ == "__main__":
    main()
