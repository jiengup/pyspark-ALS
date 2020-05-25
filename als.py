import time

from pyspark import SQLContext
from pyspark import SparkContext
import pymysql
from pyspark.mllib.recommendation import ALS
from pyspark.sql import SparkSession
import os
import sys

# os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"
sys.path.append("..")
sc = SparkContext('local[*]')
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)
spark = SparkSession.builder.appName('test').getOrCreate()
url = "jdbc:mysql://59.110.152.103:3306/asking?user=lkm&password=123"
db = pymysql.connect(host="59.110.152.103",
                     port=3306,
                     database="asking",
                     user="lkm",
                     password="123",
                     charset="utf8")
cursor = db.cursor()


def save(data, id_email):
    for item in data.collect():
        df = spark.createDataFrame(item[1]).select("user", "product", "rating").rdd.map(
            lambda x: (id_email[x.user], x.product, x.rating),
        ).toDF()
        df = df.withColumnRenamed("_1", "user").withColumnRenamed("_2", "product").withColumnRenamed("_3", "rating")
        df.write.jdbc(url=url, mode="append", table="recommend",
                      properties={"driver": "com.mysql.cj.jdbc.Driver"})


def clear_data():
    cursor.execute("truncate recommend")
    db.commit()
    print("recommend table clear")


def get_dict(data):
    dict_id = {}
    dict_email = {}
    cnt = 1
    rows = data.collect()
    for row in rows:
        dict_id[row["user_id"]] = cnt
        dict_email[cnt] = row["user_id"]
        cnt += 1
    return dict_id, dict_email


def init():
    while True:
        try:
            dataframe_mysql = sqlContext.read.format('jdbc').options(url="jdbc:mysql://59.110.152.103:3306/asking",
                                                                     dbtable="click_rec",
                                                                     user="lkm",
                                                                     password="123").load()
            df_user = dataframe_mysql.select("user_id").dropDuplicates()
            email_id, id_email = get_dict(df_user)
            dataframe_mysql = dataframe_mysql.select("user_id", "question_id").rdd.map(
                lambda x: ((email_id[x.user_id], x.question_id), 1)
            )
            data = dataframe_mysql.reduceByKey(lambda a, b: a + b).map(lambda x: (x[0][0], x[0][1], x[1]))
            model = ALS.train(data, 10, 10, 0.01)
            res = model.recommendProductsForUsers(2)
            clear_data()
            save(res, id_email=id_email)
            time.sleep(60 * 60)
        except Exception as e:
            print(e)
            time.sleep(60 * 60)
            continue


if __name__ == "__main__":
    init()
