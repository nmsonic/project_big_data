
!cp '/content/drive/MyDrive/Bigdata_project/WestburyLab.Wikipedia.Corpus.txt.bz2' 'wikipedia.txt.bz2'

!bunzip2 'wikipedia.txt.bz2'

import os

# Install java
! apt-get install -y openjdk-8-jdk-headless -qq > /dev/null
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
! java -version

# Install pyspark
! pip install --ignore-installed pyspark==2.4.5

# Install Spark NLP
! pip install --ignore-installed spark-nlp==2.4.5

# Install nltk
! pip install nltk

import sparknlp

spark = sparknlp.start()

import os
import socket
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import BooleanType, IntegerType, LongType, StringType, ArrayType, FloatType, StructType, StructField
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType
from pyspark import StorageLevel
from jinja2 import Environment, FileSystemLoader
import pyspark.sql.types as T

from  pyspark.sql.functions import *
from pyspark.sql.functions import udf, length, when, col
#from emoji import get_emoji_regexp, unicode_codes
import re

#read wikipedia texts
test_df = spark.read.text("/content/drive/MyDrive/Bigdata_project/WestburyLab.Wikipedia.Corpus.txt", lineSep="---END.OF.DOCUMENT---\n")\
       .withColumn("pp_id", monotonically_increasing_id()).cache()


test_df.show()

#converting data into Spark NLP annotation format
from sparknlp.base import DocumentAssembler

documentAssembler = DocumentAssembler() \
     .setInputCol('value') \
     .setOutputCol('document')

#tokenization
from sparknlp.annotator import Tokenizer

tokenizer = Tokenizer() \
     .setInputCols(['document']) \
     .setOutputCol('tokenized')

#cleaning of the data and transforming to lowercase
from sparknlp.annotator import Normalizer

normalizer = Normalizer() \
     .setInputCols(['tokenized']) \
     .setOutputCol('normalized') \
     .setLowercase(True)

#lemmatization
from sparknlp.annotator import LemmatizerModel

lemmatizer = LemmatizerModel.pretrained() \
     .setInputCols(['normalized']) \
     .setOutputCol('lemmatized')

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

eng_stopwords = stopwords.words('english')

#removing stopwords
from sparknlp.annotator import StopWordsCleaner

stopwords_cleaner = StopWordsCleaner() \
     .setInputCols(['lemmatized']) \
     .setOutputCol('unigrams') \
     .setStopWords(eng_stopwords)

#creaing PartOfSentensce for each word
from sparknlp.annotator import PerceptronModel

pos_tagger = PerceptronModel.pretrained('pos_anc') \
    .setInputCols(['document', 'lemmatized']) \
    .setOutputCol('pos')

#preparing data for futher using
from sparknlp.base import Finisher

finisher = Finisher() \
     .setInputCols(['unigrams', 'pos']) \

#creating pipeline with all transforamtions
from pyspark.ml import Pipeline

pipeline = Pipeline() \
     .setStages([documentAssembler,                  
                 tokenizer,
                 normalizer,                  
                 lemmatizer,                  
                 stopwords_cleaner, 
                 pos_tagger,  
                 finisher])

processed_review = pipeline.fit(test_df).transform(test_df)

processed_review.limit(5).show()

#creating word count matrix
from pyspark.ml.feature import CountVectorizer

tfizer = CountVectorizer(inputCol='finished_unigrams', outputCol='tf_features', vocabSize=5000, minDF=10.0)
tf_model = tfizer.fit(processed_review)
tf_result = tf_model.transform(processed_review)

#creating TF-IDF matrix
from pyspark.ml.feature import IDF

idfizer = IDF(inputCol='tf_features', outputCol='tf_idf_features')
idf_model = idfizer.fit(tf_result)
tfidf_result = idf_model.transform(tf_result)

tfidf_result.show(5)

#initializing and training LDA model
from pyspark.ml.clustering import LDA

num_topics = 10
max_iter = 10

lda = LDA(k=num_topics, maxIter=max_iter, featuresCol='tf_idf_features')
lda_model = lda.fit(tfidf_result)

#creating vocabulary: word_id -> word
vocab = tf_model.vocabulary

def get_words(token_list):
     return [vocab[token_id] for token_id in token_list]
       
udf_to_words = udf(get_words, T.ArrayType(T.StringType()))

#printing top 10 topics
num_top_words = 10

topics = lda_model.describeTopics(num_top_words).withColumn('topicWords', udf_to_words(col('termIndices')))
topics.select('topic', 'topicWords').show(truncate=90)



"""### Finding the most isolated articles"""

#performing pca
from pyspark.ml.feature import PCA
pca = PCA(k=50, inputCol='tf_idf_features', outputCol='tf_idf_pca')
pca_model = pca.fit(tfidf_result)

pca_results = pca_model.transform(tfidf_result)

pca_results.show(5)

df = pca_results.select('pp_id', col('tf_idf_pca'))

df.write.parquet("/content/drive/MyDrive/Bigdata_project/pca_results1.parquet")

df.show(5)

from pyspark.ml.clustering import KMeans

kmeans = KMeans(featuresCol='tf_idf_pca', predictionCol='cluster_num', k=20)
kmeans_model = kmeans.fit(df)

cluster_coordinates = kmeans_model.clusterCenters()

clusters_cols = ["clusters_centers"]
clusters_df = spark.createDataFrame(data=cluster_coordinates, schema = clusters_cols)

import numpy as np

@udf(returnType=FloatType())
def cos_sim_udf(a,b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

#calculationg top separated articles (не досчитался за 2 дня)
new_df = df.crossJoin(clusters_df.select(col("clusters_centers").alias('vectors_2')))\
    .withColumn("product", cos_sim_udf(col('tf_idf_pca'), col('vectors_2')))\
    .where("product<1").groupBy(col("pp_id")).agg(max("product").name('max_product'))\
    .orderBy('max_product')

new_df.show()



import time

while True:
  time.sleep(60)

