"""
- get the imessage exporter working + explore the data DONE
- fine tune llama on the data using qLORA
- store fine tuned model in s3 bucket (or somewhere else); 
    - associate with person's first name to keep it simple
- call the model somehow
"""

"""
instructions:
- install the imessage exporter
- change security and privacy settings for terminal
- run the imessage exporter
"""

import subprocess
import os
import shutil

def execute_imessage_exporter(delete_after=True):
    command = [
        "imessage-exporter",
        "-f",
        "txt",
        "-c",
        "disabled",
        "-a",
        "MacOS",
        "-o",
        "./imessages",
    ]

    try:
        subprocess.run(command, check=True)
        print("Command executed successfully")
        if delete_after:
            delete_imessages()
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
    except FileNotFoundError:
        print("The 'imessage-exporter' command was not found. Make sure it is installed and in your PATH.")

def execute_imessage_exporter_install():
    command = ["brew", "install", "imessage-exporter"]

    try:
        subprocess.run(command, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
    except FileNotFoundError:
        print("Homebrew is not installed on your system or the 'brew' command is not in your PATH.") 

def delete_imessages():
    folder_path = "./imessages"  # Change this path to the actual folder path

    try:
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Remove all .txt files in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(folder_path, file_name)
                    os.remove(file_path)
                    print(f"Deleted file: {file_name}")

            # Delete the folder itself
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")
        else:
            print(f"The folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    execute_imessage_exporter_install()
    execute_imessage_exporter()
