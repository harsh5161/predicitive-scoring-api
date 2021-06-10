# Numeric Engineering 1(To be tested)
# For converting allnumeric data in columns like currency remperature, numbers in numeric form etc.. into numeric form
from wordcloud import WordCloud, STOPWORDS
from IPython.display import Image
from gensim import corpora, models
import nltk
import pandas as pd
import numpy as np
import time
import itertools
from math import sin, cos, sqrt, pow
from urlextract import URLExtract
from urllib.parse import urlparse, urlsplit
import holidays
import swifter
import spacy
from collections import Counter
from string import punctuation
from textblob import TextBlob
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import matplotlib.pyplot as plt
np.random.seed(2018)
nltk.download('wordnet', quiet=True)
stemmer = SnowballStemmer('english')
############################################
############## NUMERIC ENGINEERING #########
############################################


def formatter(x):
    try:
        return x.astype(str).str.strip(' %$€£¥+').str.lower()
    except:
        return x


def list_or_dict(x):
    if isinstance(x, list):
        return "List"
    elif isinstance(x, dict):
        return "Dict"
    else:
        return np.nan


def numeric_engineering(df):
    print("\n\n")
    print(">>>>>>[[Numeric Engineering]]>>>>>")
    start = time.time()

    def returnMoney(col):
        # Remove Commas from currencies
        try:
            return pd.to_numeric(col.str.replace([',', '$', '€', '£', '¥'], ''))
        except:
            return col

    obj_columns = list(df.dtypes[df.dtypes == np.object].index)
    # print(f'object type columns are {obj_columns}')
    print(f'\t\t stripping spaces, symbols, and lower casing all entries')
    for col in obj_columns:
        df[col] = df[col].apply(lambda x: formatter(x))
    print('done ...')
    print(f'\t\t Replacing empty and invalid strings')
    possible_empties = ['-', 'n/a', 'na', 'nan', 'nil', np.inf, -np.inf, '']
    for col in obj_columns:
        df[col] = df[col].replace(possible_empties, np.nan)
    print('done ...')
    print(f'\t\t Replacing commas if present in Currencies')
    for col in obj_columns:
        df[col] = df[col].apply(lambda x: returnMoney(x))
    print('done ...')

    drop_list = []
    sampled_df = df.sample(100).dropna(how='all')
    for col in sampled_df.columns:
        counter = sampled_df[col].apply(lambda x: list_or_dict(x)).to_list()
        if counter.count("List") > 0.50*len(sampled_df) or counter.count("Dict") > 0.50*len(sampled_df):
            drop_list.append(col)
    if drop_list:
        print(
            f"Dropping columns {drop_list} because they either contain lists or dicts")
        df.drop(drop_list, axis=1, inplace=True)

    obj_columns = list(df.dtypes[df.dtypes == np.object].index)
    df1 = df[obj_columns].copy()
    print(f'\t\t Finding Numeric Columns')
    df1 = df1.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    df1.dropna(axis=1, thresh=0.65*len(df), inplace=True)
    new_num_cols = df1.columns
    df[new_num_cols] = df[new_num_cols].apply(
        lambda x: pd.to_numeric(x, errors='coerce'))
    print('done ...')

    # for i in df.columns :
    # print(f'\t\t   {i} is of type {df[i].dtypes}')

    # # End of Testing codes
    end = time.time()
    print('Numeric Engineering Completed...:', end - start)
    print('\n')
    print(">>>>>>[[Numeric Engineering]]>>>>>>")
    return(df)


############################################
############## DATE ENGINEERING ############
############################################

def getDateColumns(df, withPossibilies=0):
    '''
    This method Identifies all columns with 'DATE' data by maximizing out the possibilities
    '''
    print("\n\n")
    print(">>>>>>[[Date Engineering]]>>>>>")
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    # First get all non-numerical Columns
    non_numeric_cols = df.select_dtypes('object')
    # This dictionary stores possibilities of a column containing 'DATES' based on the name of the column
    Possibility = {}
    for column in non_numeric_cols:
        if 'date' in column.lower():
            Possibility[column] = int(len(df)*0.1)
        else:
            Possibility[column] = 0
        # ITERATE THROUGH EVERY ENTRY AND TRY SPLITTING THE VALUE AND INCREMENT/DECREMENT POSSIBILITY
        for entry in df[column]:
            try:                                                                      # USING EXCEPTION HANDLING
                if len(entry.split('/')) == 3 or len(entry.split('-')) == 3 or len(entry.split(':')) == 3:
                    Possibility[column] += 1
                    for month in months:
                        if month in entry.lower():
                            Possibility[column] += 1
                else:
                    Possibility[column] -= 1
            except:
                Possibility[column] -= 1
      # This contains the final DATE Columns
    DATE_COLUMNS = []
    for key, value in Possibility.items():
        # IF THE POSSIBILITY OF THE COLUMN IN GREATER THAN 1, THEN IT IS DEFINITELY A 'DATE COLUMN'
        if value > 0.8 * len(df):
            DATE_COLUMNS.append(key)

    # to find missed date columns
    def finddate(entry):    # function to find the presence of a month in an entry
        a = 0
        for month in months:
            if month in str(entry).lower():
                a = 1
        return a

    Y = non_numeric_cols
    Y = Y.drop(DATE_COLUMNS, axis=1)
    Possible_date_col = []
    for col in Y.columns:
        # returns a series where value is one if the entry has a month in it
        a = Y[col].apply(finddate)
        # if there is a name of a month in atleast 80% entries of the column
        if sum(a) > 0.8*len(Y[col]):
            Possible_date_col.append(col)

    print("Finding Date Columns Completed...")

    if not withPossibilies:
        return DATE_COLUMNS, Possible_date_col
    else:
        return DATE_COLUMNS, Possible_date_col, Possibility


def date_engineering(df, possible_datecols, DateTime=None, validation=False):
    import itertools

    # def fixdate(entry):    # function to introduce '-' before and after month and and removing timestamp if it is seperated from date by':'
    #  months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    #  for month in months:
    #      if month in str(entry).lower():
    #          index1= entry.find(month)
    #          index2= index1+3
    #          entry = entry[:index1]+'-'+entry[index1:index2]+'-'+entry[index2:]
    #          index3=entry.find(':')  #only specific to messy dataset
    #          entry=entry[:index3]
    #  return entry

    start = time.time()

    # if possible_datecols:
    # for col in possible_datecols:
    # df[col]=df[col].apply(fixdate)

#    print(f'Before pdtodatetime: {df.head(10)}')
    df = df.apply(pd.to_datetime, errors='coerce')
 #   print(f'After pdtodatetime: {df.head(10)}')
    # print('\nPrinting Missing % of date columns')
    MISSING = pd.DataFrame(((df.isnull().sum().sort_values(
        ascending=False)*100/len(df)).round(2)), columns=['Missing in %'])[:10]
    # print(MISSING)
    before = df.columns.to_list()
    if validation == False:   # dropping columns with missing greater than 35% only in training not scoring
        print('Dropping Columns with missing greater than 35% of total number of rows completed...')
        df.dropna(thresh=len(df)*0.65, axis=1, inplace=True)
        after = df.columns.to_list()
        dropped_cols = list(list(set(before)-set(after)) +
                            list(set(after)-set(before)))
    try:
        for c in df.columns:
            # replacing null values with mode
            df[c].fillna(df[c].mode()[0], inplace=True)

    except:
        for c in df.columns:
            # if error in mode then replacing null values with mean
            df[c].fillna(df[c].mean(), inplace=True)

    date_cols = df.columns
    if validation == False:
        # Extracting Time and Date columns separately from Date Time columns
        possibleDateTime = []
        date_sample = df.sample(n=100)
        for col in date_cols:
            try:
                ser = date_sample[col].dt.hour
                if ser.nunique() > 1:
                    possibleDateTime.append(col)
            except Exception as e:
                print(e)
        print("Analysis to find DateTime columns completed...")
        # print(f'Possible DateTime columns are {possibleDateTime}')
    else:
        possibleDateTime = DateTime
    # removing datetime cols from date_cols
    date_cols = list(date_cols)
    for col in possibleDateTime:
        date_cols.remove(col)

    # Extracting date column from possible datetime cols and adding it back into df,date_cols
    for col in possibleDateTime:
        try:
            df[str(col)+"_Date"] = pd.to_datetime(df[col].dt.date, errors='coerce')
            date_cols.append(str(col)+"_Date")
        except:
            pass
    # Creating Time Columns from the possible datetime columns
    possibleTimeCols = []
    for col in possibleDateTime:
        try:
            df[str(col)+"_HMSTime"] = pd.to_datetime(df[col].dt.time,
                                                     format="%H:%M:%S", errors='coerce')
            possibleTimeCols.append(str(col)+"_HMSTime")
        except:
            pass
    # Remove possible DateTimes from the main dataframe
    df.drop(possibleDateTime, axis=1, inplace=True)

    # creating separate month and year columns, and difference from current date
    visualize_dict = dict.fromkeys(date_cols, [])
    for i in date_cols:
        df[str(i)+"_month"] = df[str(i)].dt.month.astype(int)
        df[str(i)+"_year"] = df[str(i)].dt.year.astype(int)
        df[str(i)+"_day"] = df[str(i)].dt.day.astype(int)
        df[str(i)+"-today"] = (pd.to_datetime('today') -
                               df[str(i)]).dt.days.astype(int)
        visualize_dict[str(i)] = visualize_dict[str(
            i)] + [str(i)+"_month"] + [str(i)+"_year"]+[str(i)+"_day"]+[str(i)+"-today"]

    # create difference columns
    diff_days = list()
    if (len(date_cols) > 1):
        for i in itertools.combinations(date_cols, 2):
            diff_days = diff_days + [str(i[0])+"-"+str(i[1])]
            df[str(i[0])+"-"+str(i[1])] = (df[i[0]] -
                                           df[i[1]]).dt.days.astype(int)
    print("Extraction of timebound features completed...")
    # See Near Holiday or not

    def nearHol(currentDate, us_hols, currentYear):
        new_list = []
        append = new_list.append
        for date, occasion in us_hols:
            if(date.year == currentYear):
                append(date)
        flag = 1
        for i in new_list:
            a = (currentDate.date()-i).days

            if abs(a) <= 5:
                flag = 1
                break
            else:
                flag = 0

        return 0 if flag == 0 else 1

    for col in date_cols:
        #         print('LOOP')
        # creating a unique list of all years corresponding to a column to minimise mapping
        us_hols = holidays.US(
            years=df[str(col)+'_year'].unique(), expand=False)
        # creating a new columns to check if a date falls near a holiday
        df[str(col)+'_Holiday'] = df.apply(lambda x: nearHol(x[col],
                                                             us_hols.items(), x[str(col)+'_year']), axis=1)
        visualize_dict[str(col)] = visualize_dict[str(col)
                                                  ] + [str(col)+"_nearestHoliday"]

    for col in possibleTimeCols:
        df[str(col)+"_Hour"] = df[col].dt.hour
        df[str(col)+"_Minute"] = df[col].dt.minute
        df[str(col)+"_Seconds"] = df[col].dt.second
        visualize_dict[str(col)] = [str(col)+"_Hour"] + \
            [str(col)+"_Minute"] + [str(col)+"_Seconds"]
    # removing the possible timecols from the dataframe
    df.drop(possibleTimeCols, axis=1, inplace=True)

    # print("\nVisualizing Coloumns Generated\n {}" .format(visualize_dict))
    # print("\nThe Following columns were generated to get days between dates of two seperate date columns\n {}".format(diff_days))
    print(f"Columns Impacted : {visualize_dict.keys()}")
    print(f"Columns Created: {visualize_dict.values()}")
    end = time.time()
    print('\nDate Engineering Completed...: {}'.format(end-start))
    print(">>>>>>[[Date Engineering]]>>>>>>")
    print("\n\n")
    if validation == False:
        return df.drop(date_cols, axis=1), dropped_cols, possibleDateTime
    else:
        return df.drop(date_cols, axis=1)


############################################
############## TEXT ANALYTICS ##############
############################################
def returnAddressCounter(string):
    try:
        numbers = re.findall("\d+",string)
        for number in numbers:
            if len(number) >= 5:
                return True
        return False
    except:
            return False
        
def findAddressColumns(df):
    possibleAddressColumns = []
    for col in df.columns:
        temp = df[col].apply(lambda x: returnAddressCounter(x)).to_list()
        if temp.count(True) > 0.50*len(df):
            possibleAddressColumns.append(col)
    return possibleAddressColumns

def findReviewColumns(df):  # input main dataframe
    print("\n\n")
    print(">>>>>>[[Text Engineering]]>>>>>")
    rf = df.sample(n=150, random_state=1).dropna(axis=0) if len(
        df) > 150 else df.dropna(axis=0)  # use frac=0.25 to get 25% of the data

    # df.dropna(axis=0,inplace=True) #dropping all rows with null values

    #categorical_variables = []
    col_list = []
    for col in rf.columns:
        if df[col].nunique() < 100:
            # define threshold for removing unique values #replace with variable threshold
            col_list.append(col)
            # here df contains object columns, no null rows, no string-categorical,
            rf.drop(col, axis=1, inplace=True)

    rf.reset_index(drop=True, inplace=True)
    for col in rf.columns:
        count1, count2, count3, count4 = 0, 0, 0, 0
        for i in range(len(rf)):
            val = len(str(rf.at[i, col]).split())
            if val == 1:
                count1 = count1+1
            elif val == 2:
                count2 = count2+1
            elif val == 3:
                count3 = count3+1
            elif val == 4:
                count4 = count4+1
        # print(col,"count of words is",count1,"-",count2,"-",count3,"-",count4,"-")

        if count1+count2+count3+count4 >= 0.75*len(rf):
            col_list.append(col)
            # print("dropping column",col)
            rf.drop(col, axis=1, inplace=True)

    #Additional logic to find and drop columns that may be address!
    possibleAddressColumns = findAddressColumns(rf)
    for col in possibleAddressColumns:
        col_list.append(col)
        rf.drop(col,axis=1,inplace=True)

    start = time.time()
    # print(rf.shape)
    nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'textcat'])
    sf = pd.DataFrame()
    for col in rf.columns:
        sf[col] = rf[col].apply(nlp)

    end = time.time()
    print("Tokenizing DataFrame for text extraction completed...")

    #print("Tokenised Sampled DataFrame",sf)
    #print("Sampled DataFrame",rf)
    #print("Actual Dataframe",df)

    start = time.time()
    #testf = sf.sample(frac=0.10,random_state=44)

    # code to eliminate columns of name, city, address
    for col in sf.columns:
        entity_list = []
        # converting one column into tokens
        tokens = nlp(''.join(str(sf[col].tolist())))
        #print("the tokens of each column are:", tokens)
        token_len = sum(1 for x in tokens.ents)
        # print("Length of token entities",token_len)                                    #create two lists that hold the value of actual token entities and matched token entities respectively
        if token_len > 0:
            for ent in tokens.ents:
                # matching is done on the basis of whether the entity label is
                if (ent.label_ == 'GPE') or (ent.label_ == 'PERSON'):
                    # countries, cities, state, person (includes fictional), nationalities, religious groups, buildings, airports, highways, bridges, companies, agencies, institutes, DATE etc.
                    entity_list.append(ent.text)

            entity_counter = Counter(
                entity_list).elements()  # counts the match
            counter_length = sum(1 for x in entity_counter)
        #   print("Length of matched entities",counter_length) #if there is at least a 50% match, we drop that column TLDR works better on large corpus
            if (counter_length >= 0.60*token_len):
                col_list.append(col)
        else:
            #   print("Length of token entities 0")
            #   print("Length of matched entities 0")
            pass
        counter_length = 0
        token_len = 0

    # list of columns that need to be removed
    # print("Columns that are going to be removed are ", col_list)
    print("Analysis to determine non-text columns completed...")
    ##########IMPORTANT LINE NEXT###############
    rf = df.copy()  # unhide this to immediately work with the entire dataset and not just sampled dataset and vice-versa to work with sampled
    ##########DO NOT IGNORE ABOVE LINE##########
    for val in col_list:
        rf.drop(val, axis=1, inplace=True)
    end = time.time()
    # print("Time taken for completion of excess column removal:", end-start)
    print("Finding text columns completed...")
    if (len(rf.columns) == 0):
        print("No Remarks or Comments Found ")
        flag = 0
        return None, None
    else:
        flag = 1

    if (flag == 1):
        main_list = []  # holds all the review columns
        append = main_list.append
        for col in rf.columns:
            append(col)

        return main_list, col_list


def sentiment_analysis(rf):
    bf = pd.DataFrame()
    print("Running sentiment analysis...")

    def getSubjectivity(text):
        try:
            # returns subjectivity of the text
            return TextBlob(text).sentiment.subjectivity
        except:
            return None

    def getPolarity(text):
        try:
            # returns polarity of the sentiment
            return TextBlob(text).sentiment.polarity
        except:
            return None

    for col in rf.columns:  # creating a new DataFrame with new columns
        col_pname = "{}-{}".format(col, "Polarity")
        col_sname = "{}-{}".format(col, "Subjectivity")
        bf[col_pname] = rf[col].apply(getPolarity)
        print("Polarity Analysis Completed...")
        bf[col_sname] = rf[col].apply(getSubjectivity)
        print("Subjectivity Analysis Completed...")

    return bf


def lemmatize_stemming(text):
    # performs lemmatization
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        # removes stopwords and tokens with len>3
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


def preprocessWithoutLematizer(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        # removes stopwords and tokens with len>3
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(token)
    return result


def topicExtraction(df, validation=False, lda_model_tfidf=None):

    data_text = df.copy()
    data_text['index'] = data_text.index
    documents = data_text

    headline = list(documents.columns)[0]  # review column

    processed_docs = documents[headline].map(
        preprocess)  # preprocessing review column

    #print("Processed Docs are as follows",processed_docs[:10])

    dictionary = gensim.corpora.Dictionary(
        processed_docs)  # converting into gensim dict
    # taking most frequent tokens
    dictionary.filter_extremes(no_below=10, no_above=0.25, keep_n=1000)

    if validation == False:
        print("Generating Wordcloud Completed...")
        word_str = ""
        for k, v in dictionary.token2id.items():
            word_str = word_str + k + " "
        # Generating wordcloud
        wordcloud = WordCloud(width=1000, height=800, random_state=42, background_color='white',
                              colormap='twilight', collocations=False, stopwords=STOPWORDS).generate(word_str)
        # Saving the image
        file = str(df.columns[0])+"_wordcloud.png"
        wordcloud.to_file(file)
        image = Image(filename=file)
        display(image)

    # document to bag of words
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    if validation == False:
        #print("BOW Corpus", bow_corpus[:10])
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]  # generating the TF-IDF of the corpus
        # print("!!!!!!", len(corpus_tfidf))
        start = time.time()
        # multiprocessing Latent Dirilichtion Allocation Model , passes=1, workers=6
        lda_model_tfidf = gensim.models.LdaModel(
            corpus_tfidf, num_topics=10, id2word=dictionary)
        end = time.time()
        # print(end-start)
        for idx, topic in lda_model_tfidf.print_topics(-1):
            # printing topics in the corpus
            # print('Topic: {} Word: {}'.format(idx, topic))
            pass

    ser = []
    append = ser.append
    # print("Bag of Words Corpus length", len(bow_corpus))
    start = time.time()
    count = 0
    for i in range(len(bow_corpus)):
        val = bow_corpus[i]
        for i in range(len(val)):
            sentence = "Word {} (\"{}\") appears {} time.".format(
                val[i][0], dictionary[val[i][0]], val[i][1])  # Messy3.csv error
        count = count+1
        try:
            for idx, topic in sorted(lda_model_tfidf[bow_corpus[i]], key=lambda tup: -1*tup[1]):
                append(idx)
                break
        except:
            append(idx)
        # print("Loop ran for ",count)
    end = time.time()
    asf = pd.DataFrame(ser)
    # print("Time for append", end-start)
    print("Topic Extraction Completed...")
    print("\n\n")
    return asf, lda_model_tfidf


def returnPol(x):
    if float(x) > 0.0:
        return "Positive"
    elif float(x) < 0.0:
        return "Negative"
    elif float(x) == 0.0:
        return "Neutral"


def returnCountSum(x, val):
    if val in x[0]:
        return x[2]
    else:
        return np.nan


def sentenceRecalibration(text):
    nlp = spacy.load('en_core_web_sm')
    # excluded tags
    excluded_tags = {"AUX", "ADP"}
    temp_text = []
    for token in nlp(text):
        if token.pos_ not in excluded_tags:
            temp_text.append(token.text)
    return " ".join(temp_text)


def topicWordVisualiser(analyticsFrame, topicColumn,topicClass=None,topicNumber=None):
    groupByTopic = analyticsFrame.groupby(topicColumn)
    groups = [groupByTopic.get_group(x) for x in groupByTopic.groups]
    group_keys = list(groupByTopic.groups.keys())
    i = 0
    resultDict = {}
    for groupFrame in groups:
        # currentGroupKey = group_keys[i]
        temporaryFrame = groupFrame
        reviewList = []
        reviewList.extend(temporaryFrame['Review'])
        flatList = [item for sublist in reviewList for item in sublist]
        mostCommonWords = Counter(flatList).most_common(10)
        tempDict = {}
        for words in mostCommonWords:
            word, count = words
            tempDict[word] = count
        resultDict[i] = tempDict
        i += 1
    i = 1
    for key in resultDict:
        if not resultDict[key]:
            del resultDict[key]
            continue
        currentTopicDict = resultDict[key]
        print(f"Topic {i}")
        plt.bar(list(currentTopicDict.keys()),
                currentTopicDict.values(), color='g')
        plt.xticks(rotation=45)

        # Uncomment the above two lines are run messy8.csv to understand how the output looks, but implement these plots in the front end using plotly so that it looks better. It goes into the new text analytics tab, users will have an option to select the topic. extract resultDict after this for loop runs so that there aren't any empty dictionaries,then you can use the length of resultDict as the different topics that is present.
        plt.show()
        i += 1


def text_analytics(review, col, mode, target, LE, topic_frame):
    analyticsFrame = topic_frame.copy()
    topicColumn = analyticsFrame.columns[0]
    analyticsFrame['Review'] = review[col]
    # print("Removing Auxillary's...")
    # analyticsFrame['Review'] = analyticsFrame['Review'].apply(
    #     lambda x: sentenceRecalibration(x))
    processed_docs = analyticsFrame['Review'].map(preprocessWithoutLematizer)
    analyticsFrame['Review'] = processed_docs
    if mode == 'Classification':
        analyticsFrame['Target'] = target
        analyticsFrame['Target'].fillna(
            analyticsFrame['Target'].mode()[0], inplace=True)
        analyticsFrame['Target'] = analyticsFrame['Target'].astype('int')
        analyticsFrame['Target'] = LE.inverse_transform(
            analyticsFrame['Target'])

        groupByClass = analyticsFrame.groupby('Target')
        groups = [groupByClass.get_group(x) for x in groupByClass.groups]
        group_keys = list(groupByClass.groups.keys())
        i = 0
        for groupFrame in groups:
            print(f"Class: {group_keys[i]}")
            topicWordVisualiser(groupFrame, topicColumn,group_keys[i],i)
            i += 1
    else:
        topicWordVisualiser(analyticsFrame, topicColumn)

    print("Text Analytics Completed")
    print(">>>>>>[[Text Engineering]]>>>>>")
    print("\n\n")
    # print(output_df)
    # output_df.to_csv("output_df.csv")

############################################
############## EMAIL ENGINEERING ##############
############################################

################### EMAIL AND URL IDENTIFICATION FUNCTIONS ###################


def identifier(x):
    try:
        if x.split('@'):
            if x.split('@')[1].split('.'):
                return True
            else:
                return False
        else:
            return False
    except:
        return False


def identifyEmailColumns(df):
    print("\n\n")
    print(">>>>>>[[EMAIL Engineering]]>>>>>")
    email_cols = []
    for col in df.columns:
        a = df[col].apply(lambda x: identifier(x)).to_list()
        if a.count(True) > 0.50*len(df):
            email_cols.append(col)
    print("Email Columns Identification Completed...")
    return email_cols


################### EMAIL ENGINEERINGS ###################
# Default email parameter is true for email engineering
def emailUrlEngineering(df, email=True, validation=False):
    # Email parameter is false for URL engineering
    ############################## EMAIL ENGINEERING ##############################

    start = time.time()

    # if email is True:
    #     print('\n########## EMAIL ENGINEERING RUNNING ##########')
    # else:
    #     print('\n########## URL ENGINEERING RUNNING ##########')

    def getEmailDomainName(col):
        # Get the first domain name, example: a@b.gov.edu.com, in this b alone is taken
        try:
            # print("Inside email")
            # print(col[1].split('.')[0])
            return col[1].split('.')[0]
        except:
            return np.nan  # Invalid Entry

    def getUrlDomainName(col):
        try:
            # print("Inside url")
            # print(col.split('://')[1].split('/')[0].split('.')[0])
            return col.split('://')[1].split('/')[0].split('.')[0]
        except:
            return np.nan  # Invalid Entry

    def rformatter(x):
        try:
            return x.rsplit('@')
        except:
            return np.nan

    # Making a note of newly created columns
    newCols = []
    # For every column in email columns, get Domain Name, create a new column and check the missing percentage
    for column in df.columns:
        domain_name = column + '_domain'
        if email is True:
            ser = df[column].apply(lambda x: rformatter(x))
            ser.reset_index(drop=True, inplace=True)
            df[domain_name] = ser.apply(getEmailDomainName)
        else:
            df[domain_name] = df[column].apply(getUrlDomainName)
        # Checking percentage of missing values
        if validation == False:
            if df[domain_name].isnull().sum()/len(df) >= 0.5:
                # print('The newly created \'{}\' column has 50% or more missing values!'.format(
                # domain_name))
                # print('And hence will be dropped!')
                df.drop(domain_name, axis=1)
            else:
                newCols.append(domain_name)
        else:
            newCols.append(domain_name)
    print("Extracting Domain Names from Email Columns Completed...")
    if len(newCols) == 0:
        return_df = pd.DataFrame(None)  # Will help return empty dataframe
    else:
        # Returning DataFrame that contain only newly created columns
        return_df = pd.DataFrame(df[newCols].fillna('missing'))
        # With missing imputation done

    end = time.time()
    if email is True:
        print('\nFeatures Created in Email Engineering are: {}'.format(newCols))
        print('\nEmail Engineering Completed... {}'.format(end-start))
    else:
        print('\nFeatures Created in URL Engineering are: {}'.format(newCols))
        print('\nURL Engineering Completed... {}'.format(end-start))

    print(">>>>>>[[EMAIL Engineering]]>>>>>")
    print("\n\n")
    return return_df
############################################
############## EMAIL ENGINEERING ##############
############################################


############################################
############## URL ENGINEERING ##############
############################################

def urlCheck(x, extractor):
    try:
        if extractor.has_urls(str(x)):
            return True
        else:
            return False
    except TypeError:
        return False


def findURLS(df):
    print("\n\n")
    print(">>>>>>[[URL Engineering]]>>>>>")
    extractor = URLExtract()
#    extractor.update()
    url_cols = []
    for col in df.columns:
        a = df[col].apply(lambda x: urlCheck(x, extractor)).to_list()
        if a.count(True) > 0.75*len(df):
            url_cols.append(col)
        # print(f"Trying column {col} and the percentage of urls are {a.count(True)}")
    if url_cols:
        print("The URL Columnns found are", url_cols)
    return url_cols


def getDomain(x):
    if x.split('.')[0].lower() == 'www':
        return x.split('.')[1].split('.')[0].lower()
    else:
        return x.split('.')[0].lower()


def urlparser(x, extractor):
    try:
        for url in extractor.gen_urls(x):
            url_obj = urlparse(url)
            if len(url_obj.scheme) > 0:
                return getDomain(url_obj.netloc)
            else:
                return getDomain(url_obj.path)
    except:
        return 'missing'


def URlEngineering(df):
    urls = {}
    extractor = URLExtract()
    # extractor.update()
#    print("Updating if extractor TLD's haven't been updated in seven days")
 #   extractor.update_when_older(7) #updates when list is older than 7 days
    print("URLs Parsed Successfully...")
    print("URL Domain Extraction Completed... ")
    for col in df.columns:
        ser = df[col].apply(lambda x: urlparser(x, extractor))
        urls[f'{col}_domain'] = ser
    if urls:
        print(
            f"Features Generated are : {pd.DataFrame.from_dict(urls).columns} ")
    print(">>>>>>[[URL Engineering]]>>>>>")
    print("\n\n")
    if urls:
        return pd.DataFrame.from_dict(urls)
    else:
        return None


############################################
############## URL ENGINEERING ##############
############################################

############################################
############## LAT-LONG ENGINEERING ##############
############################################

def floatCheck(x):
    # Accepts floating point numbers as well as np.nan  (testing)#and float.is_integer(x) is False:
    if isinstance(x, float) is True:
        return True
    else:
        return False


def checkFormat(x):
    try:
        if (decimal.Decimal(str(x)).as_tuple().exponent <= -3) and (x > -180.0 and x < 180.0):
            return True
        else:
            return False
    except:
        return False


def checkLatLong(x):
    if x > -180.0 and x < 180.0:
        if x > -90.0 and x < 90.0:
            return "Lat"
        return "Long"
    else:
        return np.nan


def checkCondition(x):
    # Sometimes if we have columns with values that have [] around them then they are considered as lists, this will help it out.
    if isinstance(x, list):
        if len(x) == 2:
            return True
        else:
            return False

    # if isinstance(x,dict): #Sometimes if we have columns with values that have {} around them then they are considered as lists, this will help it out.
    #     if len(x) == 2:
    #         return True
    #     else:
    #         return False
    try:
        if x[0] == '(' and x[-1] == ')' and len(x.split(',')) == 2:
            return True
        else:
            return False
    except TypeError:
        return np.nan


def Floater(df, value):
    print(value)
    floaters = []
    for column in df.columns:  # add try catch block
        # print(f"testing column {column}")
        if value == "returnFloat":
            a = df[column].apply(lambda x: floatCheck(x)).to_list()
        elif value == "confirmLatLong":
            a = df[column].apply(lambda x: checkFormat(x)).to_list()
        # print(f"printing true value counts {a.count(True)}")
        if a.count(True) > 0.9*len(df):
            floaters.append(column)
    return floaters


def segregator(df):
    lat_cols = []
    long_cols = []
    for col in df.columns:
        a = df[col].apply(lambda x: checkLatLong(x)).to_list()
        if a.count("Lat") > 0.9*len(df):
            lat_cols.append(col)
        elif a.count("Long") > 0.9*len(df):
            long_cols.append(col)
    if not long_cols:
        try:
            for i in range(1, len(lat_cols), 2):
                long_cols.append(lat_cols[i])
                lat_cols.remove(lat_cols[i])
        except:
            print("Lat-Long Length Mismatch")
            lat_cols = []
            long_cols = []
    return lat_cols, long_cols


def pseudoFormat(df):
    lat_long_cols = []
    for col in df.columns:
        # print(f"testing column {col}")
        a = df[col].apply(lambda x: checkCondition(x)).to_list()
        if a.count(True) > 0.9*len(df):
            lat_long_cols.append(col)
    return lat_long_cols


def findLatLong(df):
    print("\n\n")
    print(">>>>>>[[Latitude Longitude Engineering]]>>>>>")
    lat_cols = []
    long_cols = []
    # will add logic for lat-long columns of the form (lat,long)
    lat_long_cols = []
    lat_long_cols = pseudoFormat(df)
    # if lat_long_cols:
    #     print("Columns that are are of the form Lat-Long are as follows", lat_long_cols)
    print("Analysis of columns in special format completed...")
    # List of float columns that could be lat or long
    columns = Floater(df, "returnFloat")
    print(f"The columns that could be Lat/Long are as follows {columns}")
    for ex in columns[:]:
        if "lat" in ex.lower() or "latitude" in ex.lower():
            lat_cols.append(ex)
            columns.remove(ex)
        elif "long" in ex.lower() or "longitude" in ex.lower():
            long_cols.append(ex)
            columns.remove(ex)
    desired = []
    # print(f"lat-cols are {lat_cols}")
    # print(f"long-cols are {long_cols}")
    # print(f"columns are {columns}")
    requisites = ["Lat", "Long", "Latitude", "Longitude"]
    if len(columns) > 1:
        for val in itertools.product(columns, requisites):
            if val[0].lower().find(val[1].lower()) != -1 and val[0] not in desired:
                desired.append(val[0])
                columns.remove(val[0])
    # Removing columns with low nunique()
    for col in columns[:]:
        if df[col].nunique() < 100:
            columns.remove(col)
    # We check if there are any lat or long columns present in the rest of the float columns
    if columns:
        possible = Floater(df[columns], "confirmLatLong")
        if possible:  # If they are of Lat Long format then add it to desired list
            desired.extend(possible)

    if desired:
        lat_cols, long_cols = segregator(df[desired])
    print("Analysis of Columns to find Lat-Long Completed...")
    return lat_cols, long_cols, lat_long_cols


def distanceCalc(x_list):
    x = cos(float(x_list[0])) * cos(float(x_list[1]))
    y = cos(float(x_list[0])) * sin(float(x_list[1]))
    z = sin(float(x_list[0]))
    return sqrt(pow(x-1, 2)+pow(y-0, 2)+pow(z-0, 2))


def convertCartesian(x):
    try:
        x_list = x[x.find('(')+1:x.find(')')].split(',')
        return distanceCalc(x_list)
    except AttributeError:
        return 0.0


def originGenerator(latitude, longitude):
    temp = {}
    temp[latitude.name] = latitude
    temp[longitude.name] = longitude
    temp_df = pd.DataFrame.from_dict(temp)
    ser = temp_df.apply(lambda x: distanceCalc([x[0], x[1]]), axis=1)
    return ser


def latlongEngineering(df, lat_cols, long_cols, lat_long_cols):
    req = {}
    if lat_long_cols:
        for c in lat_long_cols:
            ser = df[c].apply(lambda x: convertCartesian(x))
            req[f'{c}-Origin'] = ser
    print("Converting to Cartesian Completed...")
    if lat_cols and long_cols and len(lat_cols) == len(long_cols):
        for i in range(len(lat_cols)):
            ser = originGenerator(df[lat_cols[i]], df[long_cols[i]])
            ser.fillna(0.0, inplace=True)
            req[f'{lat_cols[i]}_{long_cols[i]}-Origin'] = ser
    else:
        # print("Lat columns and Long columns length mismatch")
        pass
    print("Calculating Distance from Origin Completed...")
    print(">>>>>>[[Lat Long Engineering]]>>>>>")
    print("\n\n")
    if req:
        return pd.DataFrame.from_dict(req)
    else:
        return None

############################################
############## LAT-LONG ENGINEERING ##############
############################################
