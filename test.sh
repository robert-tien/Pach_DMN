#echo "accuracy=0.98" | cut -d "=" -f 2
echo "<td>"
grep -m 1 "max_input_len" $1|cut -d "=" -f 2
echo "</td>"
