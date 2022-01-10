import re



def clean_text(text):
    """
    This function cleans the text in the following ways
    1. Replace websites with URL
    2. Replace 's with <space>'s (e.g., her's --> her 's)
    """
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "URL", text) # Replace urls with special token
    #text = text.replace("\'s", "")
    #text = text.replace("\'", "")
    #text = text.replace("n\'t", " n\'t")
    text = text.replace("@", "")
    text = text.replace(":", "")
    text = text.replace("#", "")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("&amp;", "")
    text = text.replace("&gt;", "")
    text = text.replace("\"", "")
    text = text.replace("$MENTION$", '')
    text = text.replace("$ URL $", '')
    text = text.replace("$URL$", '')
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("<end>", "")
    text = text.replace("|", "")
    text = text.lower()
    return text.strip()

def change_label(label):
    if label == 'non-rumours':
        label = 0
    elif label == 'rumours':
        label = 1
    
    return label

    