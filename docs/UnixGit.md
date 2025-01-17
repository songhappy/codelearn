# Unix command and git command CheetSeet


## unix, testcases, git
1. print word frequencies in a file
`cat words.txt |  tr '\n' ' '| tr -s ' ' | tr ' ' '\n' | sort | uniq -c |sort -k 2 -rn| awk '{print $2" "$1}' `
2. find patterns, or and
`grep 'pattern1\|pattern2' filename`  # or
find correct phone numbers
`cat file.txt | egrep ^'(\([0-9]{3}\) |[0-9]{3}-)[0-9]{3}-[0-9]{4}'$`
^: start of string, $: end of string 
find any numbers of a file, replace none numbers to nothing
`echo "I am 999 years old." | sed 's/[^0-9]*//g' | awk {print}`
`echo "I am 999 years old." | tr -dc '0-9' | awk {print}`
-c: complimentary, -d: delete
^:not
find pattern 1 and pattern2
`grep -E 'pattern1.*pattern2' filename`
3. display the tenth line
`tail -n +10 file.txt | head -n 1`
head -n: lines
4. loops in bash
 `   #!/bin/bash
        for i in `seq 1 10`;
        do
            echo $i
            echo hello
        done 
       # 
        start=1
        end=10
        for ((i=start; i<=end; i++))
        do
           echo "i: $i"
        done
  `
        
5. awk usage, https://www.geeksforgeeks.org/awk-command-unixlinux-examples/
(a) Scans a file line by line
(b) Splits each input line into fields
(c) Compares input line/fields to pattern
(d) Performs action(s) on matched lines
 I use it to print separate fields.
 `cat users.dat | awk -F:: '{print $4}' | sort -u`  #split by :: get the 4th column and get distinct values 
6. sort
   -n, on numbers; -k 2, on second column; -r, in reverse order; 
       
       
7. paste file1 file2 file3
    concatenate file1 file2 file3 by columns 
           
## test cases
1. connecting to a server for a small file
`input = "hdfs://blabla`



## git
1. Git fetch all branches from remote
```
Git fetch --all
```
2. git copy branches from a remote and build the same ones on local
```
git branch -r | grep -v '\->' | while read keras; do git branch --track "${keras#origin/}" "$keras"; done
git pull --all
```
3. git remote add url
```
git remote add intel-analytics https://github.com/intel-analytics/analytics-zoo.git
```

4. Set up local of spark in intellij
-Dspark.master="local[1]”


5. Get the loss
```
cat output  | grep "Loss is" |  awk 'NF>1{print $NF}' | sed -e "s/.\n/\n/g" > loss.txt
```
analytics-zoo-bigdl_0.7.1-spark_2.1.0-0.4.0-SNAPSHOT-jar-with-dependencies.jar
6. Install jars in local environment
```
mvn install:install-file -Dfile=bigdl-SPARK_2.2-0.5.0-jar-with-dependencies.jar -DgroupId=com.intel.analytics.bigdl -DartifactId=bigdl-spark-2.2 -Dversion=0.5.0 -Dpackaging=jar
mvn install:install-file -Dfile=analytics-zoo-bigdl_0.7.1-spark_2.1.0-0.4.0-SNAPSHOT-jar-with-dependencies.jar -DgroupId=com.intel.analytics.zoo -DartifactId=analytics-zoo-bigdl_0.7.1-spark_2.1.0  -Dversion=0.4.0-SNAPSHOT -Dpackaging=jar
```

7. Git rebase
```
git pull --rebase  intel-analytics master
```

8. CTRL SHIFT CMD + for superscript
CTRL CMD - for subscript

9. disable proxy
export http_proxy=
export https_proxy=

