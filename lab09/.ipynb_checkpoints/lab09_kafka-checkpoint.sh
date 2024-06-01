
echo "Now start the services needed for spark + kafka. " 

## Step 1: open a tmux window to run the zookeeper service
workloc="/opt/module/kafka/kafka_2.13-3.7.0"
$workloc/bin/zookeeper-server-start.sh $workloc/config/zookeeper.properties

## Step 2: open another tmux window to run the kafka service
workloc="/opt/module/kafka/kafka_2.13-3.7.0"
$workloc/bin/kafka-server-start.sh $workloc/config/server.properties
# $workloc/bin/kafka-topics.sh --list --bootstrap-server localhost:9092

## Step 3: start a kafka producer by reading contents in a file
python /shareddata/lab09/lab09_run_producer.py 

echo "Now start the jupyter notebook. "


# ##
# sparkloc="/usr/local/spark-3.3.1"
# $sparkloc/bin/pyspark --jars /data/jupyter-data/lab08/spark-sql-kafka-0-10_2.12-3.3.1.jar,/data/jupyter-data/lab08/spark-token-provider-kafka-0-10_2.13-3.3.1.jar,/data/jupyter-data/lab08/commons-pool2-2.11.1.jar

# sparkloc="/usr/local/spark-3.3.1"
# export PYSPARK_PYTHON=/root/miniconda3/envs/sp/bin/python 
# $sparkloc/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2 --jars /data/jupyter-data/lab08/spark-token-provider-kafka-0-10_2.13-3.3.1.jar,/data/jupyter-data/lab08/commons-pool2-2.11.1.jar --archives /data/jupyter-data/lab08/pyspark_conda_env.tar.gz  /data/jupyter-data/lab08_spark_kafka.py

## Step 3: open a third tmux window to run the HDFS service
# /opt/module/hadoop-3.3.1/sbin/start-dfs.sh

# $sparkloc/bin/spark-submit  --jars /data/jupyter-data/lab08/spark-sql-kafka-0-10_2.12-3.3.1.jar,/data/jupyter-data/lab08/spark-token-provider-kafka-0-10_2.13-3.3.1.jar,/data/jupyter-data/lab08/commons-pool2-2.11.1.jar /data/jupyter-data/lab08_spark_kafka.py


