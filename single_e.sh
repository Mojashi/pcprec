INSTANCEID=$1
REVERSE=$2

ALREADY_RESOLVED=$(sqlite3 results.sqlite "SELECT resolved FROM results WHERE instanceID=$INSTANCEID;")
if [ $ALREADY_RESOLVED -eq 1 ]; then
    echo "Instance $INSTANCEID already resolved"
    exit 0
fi

PROBLEM=$(sqlite3 results.sqlite "SELECT problem FROM results WHERE instanceID=$INSTANCEID;")
echo "Running problem $PROBLEM"

echo $PROBLEM | ./target/release/experiment $REVERSE


LAST_EXIT_CODE=$?
if [ $LAST_EXIT_CODE -eq 0 ]; then
    sqlite3 results.sqlite "UPDATE results SET unsatProof='resolved' WHERE instanceID=$INSTANCEID"
elif [ $LAST_EXIT_CODE -eq 1 ]; then
    sqlite3 results.sqlite "UPDATE results SET satProof='resolved' WHERE instanceID=$INSTANCEID"
else
    exit $LAST_EXIT_CODE
fi