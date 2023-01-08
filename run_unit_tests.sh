for test in $(ls test*_unit.py)
do
    echo "-----------"
    echo "RUNNING TEST FILE: $test"
    echo "-----------"
    python $test
done
