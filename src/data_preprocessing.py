import os
import logging
import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
nltk.download('punkt')

# Ensure the log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok = True)


# Setting up logging
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')


console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    """ Transform the input text by converting it into lowercase,removing stopwords and punctuation and stemming."""
    ps = PorterStemmer()
    
    # Convert to lowercase
    text = text.lower()
    
    # tokenize the text
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    
    # Remove the stopwords and punctuation
    text = [word for word in text if word not in stopwords.words("english") and word not in string.punctuation]
    
    # stem the words
    text = [ps.stem(word) for word in text]
    
    # join the token back into a single string
    return " ".join(text)


def preprocess_data(df,text_column = 'text',target_column = 'target'):
    """
    Preprocess the DataFrame by encoding the target column,removing duplicates , and transforming the text column.
    """
    
    try:
        logger.debug("Started preprocessing data for dataframe")
        
        # Encode the target column
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
        logger.debug("Target column encoded successfully.")
        
        # Remove duplicates
        df = df.drop_duplicates(keep='first')
        logger.debug("Duplicates removed")
        
        
        # Apply the text transformation to the text column
        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug("Text transformation applied successfully.")
        
        return df
    
    except KeyError as e:
        logger.error("Missing expected columns in the data: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred during data preprocessing: %s", e)
        raise
    
def main(text_column = 'text',target_column = 'target'):
    """
    Main function to load raw data,preprocess it , and save the processed data.
    """
    
    try :
        # fetch the data from the data/raw directory
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug("Raw data loaded successfully.")
        
        # Transform the data
        train_processed_data = preprocess_data(train_data,text_column,target_column)
        test_processed_data = preprocess_data(test_data,text_column,target_column)
        
        # store the the data in the data/processed directory      
        data_path = os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok = True)
        
        train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'),index = False)  
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'),index = False)  
        logger.debug("Processed data saved successfully at %s", data_path)
        
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", e)
        raise
    except pd.error.EmptyDataError as e:
        logger.error("Data file is empty: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred in main function and failed data transformation: %s", e)
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()
        

    