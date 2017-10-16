## Install hadoop

This is a pain in the arse to do.

install the latest version of ruby `\curl -L https://get.rvm.io | bash -s stable --ruby`
install hadoop `brew search hadoop` & `brew install hadoop`

Edit the following lines of `/usr/local/Cellar/hadoop/<your hadoop version>/libexec/etc/hadoop/hadoop-env.sh` to the code below:

```
export HADOOP_OPTS="$HADOOP_OPTS -Djava.net.preferIPv4Stack=true -Djava.security.krb5.realm= -Djava.security.krb5.kdc="
```

Add below configuration to `/usr/local/Cellar/hadoop/<your hadoop version>/libexec/etc/hadoop/core-site.xml `.  This is wrong in the tuturial!

```
<configuration>
<property>
<name>hadoop.tmp.dir</name>
<value>/usr/local/Cellar/hadoop/hdfs/tmp</value>
<description>A base for other temporary directories.</description>
</property>
<property>
<name>fs.default.name</name>
<value>hdfs://localhost:9000</value>
</property>
</configuration>
```

Add below configuration to `/usr/local/Cellar/hadoop/<your hadoop version>/libexec/etc/hadoop/mapred-site.xml`

```
<configuration>
 <property>
  <name>mapred.job.tracker</name>
  <value>localhost:9010</value>
 </property>
</configuration>
```

Add below configuration to `/usr/local/Cellar/hadoop/<your hadoop version>/libexec/etc/hadoop/hdfs-site.xml`

```
<configuration>
 <property>
  <name>dfs.replication</name>
  <value></value>
 </property>
</configuration>
```

Add below configuration to `~/.profile`or `./bash_profile` and type `source ~/.profile` or `source ~/.bash_profile`

```
export PATH=$PATH:/usr/local/hadoop/bin
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
alias hstart="/usr/local/Cellar/hadoop/<your hadoop version>/sbin/start-dfs.sh;/usr/local/Cellar/hadoop/<your hadoop version>/sbin/start-yarn.sh"
alias hstop="/usr/local/Cellar/hadoop/<your hadoop version>/sbin/stop-yarn.sh;/usr/local/Cellar/hadoop/<your hadoop version>/sbin/stop-dfs.sh"

```
Run `hdfs namenode -format`

Check if `~/.ssh/id_rsa` and `~/.ssh/id_rsa.pub` exist. If not type `ssh-keygen -t rsa` and `cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys`

Enable Remote Login: “System Preferences” -> “Sharing”. Check “Remote Login” on mac
Authorize SSH Keys: `cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys`

Test login `$ ssh localhost` and `$ exit`

Run `hstart` to start hadoop.

you can now type `jps` in terminal to see the processes:

```
61156 SecondaryNameNode
61045 DataNode
61365 Jps
60951 NameNode
```
Also while hadoop is running go to Resource Manager at http://localhost:50070, JobTracker at http://localhost:8088/ and Node Specific Info at http://localhost:8042/

 and `hstop` to check you're good.

## Setup your HDFS user directory

hadoop fs -mkdir /user/
hadoop fs -mkdir /user/opringle

## Upload json data into hadoop

hadoop fs -copyFromLocal Documents/Repos/cmbc_analysis/data/transit_data.json  /user/opringle/transit_data.json

hadoop fs -chown admin:hadoop /user/admin

# Install pyspark

# Train a model on the data using pyspark

The script below configures a local spark session, reads data from HDFS into pyspark and fits a regression model to predict the run time of a transit service:

```

```

# Host the model

The trained model below must be run in spark.  Below I serialize the model using MLeap, which will allow me to use it anywhere!

```

```



