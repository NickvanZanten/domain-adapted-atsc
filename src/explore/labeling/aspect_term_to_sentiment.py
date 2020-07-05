import prodigy
from prodigy.components.loaders import JSONL

@prodigy.recipe(
    "sentiment",
    dataset=("The dataset to save to", "positional", None, str),
    file_path=("Path to texts", "positional", None, str),
)
def sentiment(dataset, file_path):
    """Annotate the sentiment of texts using different mood options."""
    stream = JSONL(file_path)     # load in the JSONL file
    stream = add_options(stream)  # add options to each task

    return {
        "dataset": dataset,   # save annotations in this dataset
        "view_id": "blocks",  # use the choice interface
        "stream": stream,
        "config": {
            "choice_style": "multiple",
            "blocks": [
                {"view_id": "classification"},
                {"view_id": "choice", "text": None},
            ] 
        }
    }

def add_options(stream):
    # Helper function to add options to every task in a stream
    options = [
        {"id": "positive", "text": "positive"},
        {"id": "neutral", "text": "neutral"},
        {"id": "negative", "text": "negative"},
    ]
    for task in stream:
        task["options"] = options
        task['label'] = task['aspectTerm']
        yield task
