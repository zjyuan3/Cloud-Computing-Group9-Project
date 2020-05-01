hdfs_input_path="/train/*"
hdfs_output_path="/test_output"
hdfs dfs  -rm -r ${hdfs_output_path}
hadoop jar /data/hadoop_data/hadoop-3.2.1/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar \
-D mapreduce.job.queuename=default \
-D mapred.map.tasks=3  \
-D mapred.reduce.tasks=1 \
-input ${hdfs_input_path} \
-output ${hdfs_output_path} \
-file hadoop_model_stacking_3gram.py \
-mapper "python hadoop_model_stacking_3gram.py mapper" \
-reducer "python hadoop_model_stacking_3gram.py reducer"
