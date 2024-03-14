from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import shutil
from tqdm import tqdm

# import model and tokenizer
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# example = 'This is a blog post on how to do sentiment analysis with BERT'
# tokens = tokenizer.tokenize(example)
# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# print(tokens)
# print(token_ids)
model.summary()



dataset = pd.read_csv("./all-data.csv",encoding='ISO-8859-1')
# print(dataset.head())
# print(dataset.shape)
dataset.rename(columns={'neutral' : 'sentiment'},inplace= True)
dataset.rename(columns={'According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .' : 'review'},inplace=True)

# print(dataset.head())
# 0   neutral  Technopolis plans to develop in stages an area...                                                                             
# 1  negative  The international electronic industry company ...                                                                             
# 2  positive  With the new production plant the company woul...                                                                             
# 3  positive  According to the company 's updated strategy f...                                                                             
# 4  positive  FINANCING OF ASPOCOMP 'S GROWTH Aspocomp is ag...                                                                             
# (4845, 2)



def convert2num(value):
    if value=='positive': 
        return 1
    # elif value == 'neutral': 
    #     return 1
    else:
        return 0

    
dataset['sentiment']  =  dataset['sentiment'].apply(convert2num)
print(dataset.head())
print(dataset.shape[0])
train = dataset[:int(dataset.shape[0]/5 * 4)]    
test = dataset[int(dataset.shape[0] / 5):]


def convert2inputexamples(train, test, review, sentiment): 
    trainexamples = train.apply(lambda x: InputExample(guid=None, text_a=x[review], label=x[sentiment]), axis=1)
    validexamples = test.apply(lambda x: InputExample(guid=None, text_a=x[review], label=x[sentiment]), axis=1)
  
    return trainexamples, validexamples




def convertexamples2tf(examples, tokenizer, max_length=128):
    features = []

    # Iterate over each example
    for i in tqdm(examples):
        # Tokenize the text and obtain necessary inputs for BERT
        input_dict = tokenizer.encode_plus(
            i.text_a,
            add_special_tokens=True,    # Add 'CLS' and 'SEP'
            max_length=max_length,      # Truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,     # Pads to the right by default
            truncation=True
        )
        # 1. Tokenization: Splits the text into tokens (words or subwords).
        # 2. Add Special Tokens: Adds special tokens like [CLS] at the beginning and [SEP] at the end of the tokenized text. These tokens are required by BERT for classification tasks and separating segments, respectively.
        # 3. Truncation: If the tokenized text is longer than max_length, it truncates it to fit this maximum size.
        # 4. Token Type IDs: Generates token type IDs which are used to distinguish between different segments in models that require them (not used in single-segment tasks like sentence classification).
        # 5. Attention Mask: Generates an attention mask to differentiate the real tokens from padding tokens. This mask tells the model which tokens should be attended to and which should not.
        # 6. Padding: Ensures that all tokenized texts have the same length by adding padding tokens if necessary (pad_to_max_length=True).

        # Extract tokenized information
        input_ids, token_type_ids, attention_mask = (
            input_dict["input_ids"], #A sequence of integers representing the tokens in the text.
            input_dict["token_type_ids"], # A sequence of integers (usually 0s and 1s) indicating the segment to which each token belongs. 
            input_dict['attention_mask'] # A sequence of 1s and 0s indicating which tokens should be attended to (1s) and which are padding tokens (0s).
        )
        # Create InputFeatures object (assuming this is a defined class similar to InputExample)
        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=i.label
            )
        )
        # This object is designed to hold all the information needed to feed an example into 
        # BERT for training or prediction. The InputFeatures class is assumed to be defined 
        # elsewhere in the code or in an imported module.

    # Generator function to yield features
    def generate():
        for f in features:
            yield ( #Yield: In each iteration, the generator yields a tuple consisting of two elements:
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    # Create a TensorFlow dataset from the generator
    
    # tf.data.Dataset.from_generator: This function creates a TensorFlow 
    # Dataset object from a generator. The Dataset API is a powerful 
    # TensorFlow feature that allows for efficient data manipulation and 
    # pipelining.
    return tf.data.Dataset.from_generator(
        generate, # The first parameter is the generator function (generate) that will be used to produce elements for the dataset.
        output_types=(
            {"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32},
            tf.int64
        ), # specifies the data types of the output elements yielded by the generator. In this case, the first element (the dictionary of model inputs) has values of type tf.int32, and the second element (the label) is of type tf.int64.
        output_shapes=(
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )
    
    
# Assuming convertexamples2tf, tokenizer, trainexamples, and validexamples are defined correctly

# Correctly calling the function
trainexamples, validexamples = convert2inputexamples(train, test, 'review', 'sentiment')

# Convert trainexamples to TensorFlow dataset
train_data = convertexamples2tf(list(trainexamples), tokenizer)

# Shuffle, batch, and repeat the training dataset
train_data = train_data.shuffle(100).batch(32).repeat(2)

# Convert validexamples to TensorFlow dataset
validation_data = convertexamples2tf(list(validexamples), tokenizer)

# Batch the validation dataset
validation_data = validation_data.batch(32)

print("data transpose successfully\n")

model.compile(optimizer=tf.keras.optimizers.Adam(
              learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),             
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])
              
print("ready for train")
model.fit(train_data, epochs=2, validation_data=validation_data)


sentences = ['The comprehensive analysis and insightful recommendations provided in the report have significantly enhanced our understanding of the current financial landscape. It\'s evident that meticulous effort was put into evaluating various financial issues, leading to strategic decisions that are both innovative and practical. This review not only addresses the challenges but also highlights potential opportunities for growth, making it an invaluable resource for anyone looking to navigate financial complexities with confidence.',
            'The report on financial issues falls disappointingly short of expectations. Despite covering a broad range of topics, it lacks depth in analysis and critical insights, leaving readers with more questions than answers. The recommendations provided seem detached from the real-world challenges businesses face today, making them impractical and of little value. Overall, this review fails to deliver the clarity and direction needed to navigate the complex financial landscape effectively.']

tokenized_sentences = tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='tf')
outputs = model(tokenized_sentences)                                  
predictions = tf.nn.softmax(outputs[0], axis=-1)
labels = ['Negative','Positive']
label = tf.argmax(predictions, axis=1)
label = label.numpy()
for i in range(len(sentences)):
    print(sentences[i], ": ", labels[label[i]])
