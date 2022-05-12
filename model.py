# Will hold all the model unitils and the model itself



def tokenize_function(examples, tokenizer):
    # TODO: Im not sure why, but altough i removing NaN, we are stil encountering some nan values
    # This is just a workaround that needs to be fixed
    if not examples['tweet']:
        examples['tweet'] = 'this is fake tweet'
    return tokenizer(examples["tweet"], padding="max_length", truncation=True)