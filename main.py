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
import openai
import dotenv
import tiktoken
import json

# Load environment variables from .env file
dotenv.load_dotenv()

# Get the API key from the environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

MAX_TOKENS = 4000


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
        print(
            "The 'imessage-exporter' command was not found. Make sure it is installed and in your PATH."
        )


def execute_imessage_exporter_install():
    command = ["brew", "install", "imessage-exporter"]

    try:
        subprocess.run(command, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
    except FileNotFoundError:
        print(
            "Homebrew is not installed on your system or the 'brew' command is not in your PATH."
        )


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


def num_tokens_from_string(string: str, encoding_name: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# parse_imessages_to_json()
# for each file:
#   for each "cell":
#       count # of tokens in message
#       if # of tokens + counted > max tokens:
#          add current datum to dataset
#          clear datum and token count
#       if "me":
#           create role as assistant
#       else:
#           create role as user
#    if end of file, add to dataset


def create_dataset() -> list:
    # go thru each .txt file, count the max number of tokens that it can
    dataset = []
    folder_path = "./imessages"  # Change this path to the actual folder path

    try:
        curr_datum = []
        curr_token_count = 0
        was_assistant_seen = False
        # i = 0
        # j=0

        # Check if the folder exists
        if os.path.exists(folder_path):
            # Iterate over each .txt file in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, "r") as file:
                        # print(file_name)
                        content = file.read()
                        if len(content.split("\n", 1)) > 1:
                            content = content.split("\n", 1)[1]
                        # print(content)
                        # i+=1

                        message_cells = content.split("\n\n")
                        # Process the messages as needed
                        for message_cell in message_cells:
                            # if j < 2:
                            # print(message_cell+"\n")
                            # j+=1
                            if len(message_cell.split("\n")) > 2:
                                # print(message_cell.split("\n"))
                                # Read the second line of the message and store it in the 'sender' variable
                                sender: str = message_cell.split("\n")[1]
                                # if j < 2:
                                #     print(sender + "\n")

                                # Trim the rest of the body and store it in the 'actual_message' variable
                                actual_message: str = message_cell.split("\n", 2)[
                                    2
                                ].strip()

                                actual_message_tokens: int = num_tokens_from_string(
                                    actual_message
                                )

                                if (
                                    actual_message_tokens + curr_token_count
                                    > MAX_TOKENS
                                ):
                                    if was_assistant_seen:
                                        dataset.append({"messages": curr_datum})
                                    curr_datum = []
                                    curr_token_count = 0
                                    was_assistant_seen = False
                                    break

                                role = ""
                                if sender.lower() == "me":
                                    role = "assistant"
                                    was_assistant_seen = True
                                else:
                                    role = "user"
                                curr_datum.append(
                                    {"role": role, "content": actual_message}
                                )
                                curr_token_count += num_tokens_from_string(
                                    actual_message
                                )
                    if curr_datum and was_assistant_seen:
                        dataset.append({"messages": curr_datum})
                    curr_datum = []
                    curr_token_count = 0
                    was_assistant_seen = False
                    # j=0
                    # if i == 2:
                    #     break
        else:
            print(f"The folder '{folder_path}' does not exist.")
            raise FileNotFoundError
        return dataset
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e


def dataset_to_jsonl(dataset: list, filename: str = "dataset.jsonl"):
    if dataset:
        with open(filename, "w") as f:
            for datum in dataset:
                f.write(json.dumps(datum))
                f.write("\n")


def upload_file_to_openai(filename: str = "dataset.jsonl"):
    try:
        with open(filename, "rb") as f:
            response = openai.File.create(
                file=f,
                purpose="fine-tune",
            )
        file_id = response["id"]
        print(f"File uploaded successfully with file id: {file_id}")
    except Exception as e:
        raise e


def prepare_finetune_dataset():
    dataset = create_dataset()
    dataset_to_jsonl(dataset)
    upload_file_to_openai()


def finetune_model():
    file_id = "file-yzvKT0Gelto0BDZcP5srcj7J"
    model_name = "gpt-3.5-turbo"
    response = openai.FineTuningJob.create(
        training_file=file_id,
        model=model_name,
    )
    job_id = response["id"]
    print(f"Job created successfully with job id: {job_id}")


def rizzler_completion(good_rizz: bool = True, messages:list =[]):
    rizz = ""
    if good_rizz:
        if not messages:
            completion = openai.ChatCompletion.create(
                model="ft:gpt-3.5-turbo-0613:interlock-labs::7x5aTsmx",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an attractive man who has gotten a phone number of an attractive woman, and you are trying to take her on a nice date. You are not trying to do customer support with her. Play along as if you want to try to take her on a date and have a meaningful relationship with her. Use very casual language.",
                    },
                    {"role": "user", "content": "Hey :)"},
                ],
            )
            rizz = completion.choices[0].message
        else:
            m = messages.copy()
            m.insert(
                0,
                {
                    "role": "system",
                    "content": "You are an attractive man who has gotten a phone number of an attractive woman, and you are trying to take her on a nice date. You are not trying to do customer support with her. Play along as if you want to try to take her on a date and have a meaningful relationship with her. Use very casual language.",
                }
            )
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=m,
            )
            rizz = completion.choices[0].message
    else:
        if not messages:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system",
                    "content": "You are a man who has gotten a phone number of an attractive woman and you are trying to take her on a nice date, but you are not good at talking to people. It is unlikely that she will want to go out with you. You are not trying to do customer support with her. Play along as if you want to try to take her on a date and have a meaningful relationship with her. Use very casual language. Remember, you're trying to be unsuccessful.",
                },
                {"role": "user", "content": "Hey :)"},]
            )
            rizz = completion.choices[0].message
        else:
            m = messages.copy()
            m.insert(
                0,
                {
                    "role": "system",
                    "content": "You are a man who has gotten a phone number of an attractive woman and you are trying to take her on a nice date, but you are not good at talking to people. It is unlikely that she will want to go out with you. You are not trying to do customer support with her. Play along as if you want to try to take her on a date and have a meaningful relationship with her. Use very casual language. Remember, you're trying to be unsuccessful.",
                }
            )
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=m,
            )
            rizz = completion.choices[0].message
    print("RIZZ",rizz)
    return rizz


def get_woman_response(rizz: str, msgs: list):
    if not msgs:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": rizz},
            ],
        )
        print("WOMAN", completion.choices[0].message)
        return completion.choices[0].message
    else:
        m = msgs.copy()
        m.insert(
            0,
            {
                "role": "system",
                "content": "You are an attractive woman who has given a man you might be interested in your phone number, and he is trying to date you. You are not trying to do customer support with him. Play along as if you were open to flirting with him. Use very casual language.",
            },
        )
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=m,
        )
        print("WOMAN", completion.choices[0].message)
        return completion.choices[0].message


def perform_rizz(good_rizz: bool = True):
    print('CALLING RIZZLER FIRST')
    first_rizzler_response = rizzler_completion(good_rizz=good_rizz)

    messages_for_rizzler = [{"role": "assistant", "content": first_rizzler_response["content"]}]
    messages_for_woman = [{"role": "user", "content": first_rizzler_response["content"]}]


    print('CALLING WOMAN FIRST')
    # set up the prompt for the woman ai
    first_woman_response = get_woman_response(first_rizzler_response, messages_for_woman)
    messages_for_rizzler.append({"role": "user", "content": first_woman_response["content"]})
    messages_for_woman.append({"role": "assistant", "content": first_woman_response["content"]})
    # print("woman",first_woman_response.choices[0].message)
    for i in range(3):
        print(f'CALLING RIZZLER {i}')
        # call rizzler
        rizzler_response = rizzler_completion(good_rizz=good_rizz, messages=messages_for_rizzler)
        messages_for_rizzler.append({"role": "assistant", "content": rizzler_response["content"]})
        messages_for_woman = [{"role": "user", "content": rizzler_response["content"]}]

        print(f'CALLING WOMAN {i}')
        # call woman
        woman_response = get_woman_response(rizz=None, msgs=messages_for_woman)
        messages_for_rizzler.append({"role": "user", "content": woman_response["content"]})
        messages_for_woman.append({"role": "assistant", "content": woman_response["content"]})

    return messages_for_rizzler


def judge_rizz(rizz_messages_1, rizz_messages_2):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are Life Coach GPT. You can judge conversations to see which one has better odds of becoming a relationship.",
            },
            {
                "role": "user",
                "content": "Take these two conversations and judge which one has better odds of becoming a relationship. Conversation 1: {} Conversation 2: {} Please answer with '1' or '2'.".format(
                    rizz_messages_1, rizz_messages_2
                ),
            },
        ],
    )
    return completion.choices[0].message


def check_finetune_job_status():
    job_id = "ftjob-7eqafcvvz8PrjKWQOR9S6o00"
    response = openai.FineTuningJob.retrieve(job_id)
    print(response)

    r2 = openai.FineTuningJob.list_events(id="ftjob-7eqafcvvz8PrjKWQOR9S6o00")
    print(r2)


if __name__ == "__main__":
    # execute_imessage_exporter_install()
    # execute_imessage_exporter(delete_after=False)
    # prepare_finetune_dataset()
    # finetune_model()

    # check_finetune_job_status()
    # perform_rizz()
    # print("this rizz is good")
    # rizzler_completion(good_rizz=True)
    # print("this rizz is bad")
    # rizzler_completion(good_rizz=False)

    good_rizz_messages = perform_rizz(good_rizz=True)
    bad_rizz_messages = perform_rizz(good_rizz=False)

    print(judge_rizz(good_rizz_messages, bad_rizz_messages))


# query = "Tell me something cool."

# openai.api_base = "https://api.endpoints.anyscale.com/v1"
# openai.api_key = openai_api_key

# chat_completion = openai.ChatCompletion.create(
#     model="meta-llama/Llama-2-13b-chat-hf",
#     messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": query}],
#     temperature=0.1,
# )
# print(chat_completion)
# for message in chat_completion:
#     m = message["choices"][0]["message"]
#     if "content" in m:
#         print(message["content"], end="", flush=True)
