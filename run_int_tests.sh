for test in $(ls test*_int.py)
do
    echo "-----------"
    echo "RUNNING INTEGRATION TEST FILE: $test"
    echo "-----------"
    python $test
done
