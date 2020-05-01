hdfs_input_path="/train/*"
hdfs_output_path="/test_output"
hdfs dfs  -rm -r ${hdfs_output_path}
hadoop jar /data/hadoop_data/hadoop-3.2.1/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar \
-D mapreduce.job.queuename=default \
-D mapred.map.tasks=3  \
-D mapred.reduce.tasks=1 \
-input ${hdfs_input_path} \
-output ${hdfs_output_path} \
-file basheline.py \
-mapper "python basheline.py mapper" \
-reducer "python basheline.py reducer"
