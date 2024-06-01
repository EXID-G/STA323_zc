echo "Now start the services needed for spark + kafka. " 

## Step 1: open a tmux window to run the zookeeper service
workloc="/opt/module/kafka/kafka_2.13-3.7.0"
$workloc/bin/zookeeper-server-start.sh $workloc/config/zookeeper.properties
## 通过改 properties，改成自己的4040端口号。可以搭建自己的kafka工作环境。

## Step 2: open another tmux window to run the kafka service
workloc="/opt/module/kafka/kafka_2.13-3.7.0"
$workloc/bin/kafka-server-start.sh $workloc/config/server.properties
# $workloc/bin/kafka-topics.sh --list --bootstrap-server localhost:9092

## Step 3: start a kafka producer by reading contents in a file
python ass3_q3_runproducer.py 

echo "Now start the jupyter notebook. "
