JOBS=$(sqlite3 results.sqlite "SELECT instanceID FROM results WHERE resolved=0 ORDER BY instanceID;" | tr '\n' ' ')

parallel -j 3 --timeout 20 --progress --results ./tmp/ --joblog ./tmp/parallel.log --resume-failed --resume --bar --shuf \
    sh single_e.sh {1} {2} ::: $JOBS ::: true false
