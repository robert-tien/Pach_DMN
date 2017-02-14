#!/bin/bash
#
echo "<tr>"
echo "<td>"
echo "<a href=\"log/"
echo $1
echo "\">"
echo $1
echo "</a>"
echo "</td>"
echo "<td>"
grep -m 1 "max_input_len" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "max_allowed_sent_len" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "min_vocab_cnt" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "episodic memory num_hops" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "embed_size" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "vocab len" $1|cut -d "=" -f 4|cut -d ' ' -f 1
echo "</td>"
echo "<td>"
grep -m 1 "training: " $1|cut -d ":" -f 2|cut -d ' ' -f 2
echo "</td>"
echo "<td>"
grep -m 1 "training: " $1|cut -d ":" -f 3
echo "</td>"
echo "<td>"
grep -m 1 "all test categories tasks" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "Best validation accuracy=" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "Best validation loss    =" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "Best training   accuracy=" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "Best training   loss    =" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "Best validation epoch   =" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "tnpg_recall" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "trecall" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "tprecision" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "tTest accuracy" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "vrecall" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "vprecision" $1|cut -d "=" -f 2
echo "</td>"
echo "<td>"
grep -m 1 "vTest accuracy" $1|cut -d "=" -f 2
echo "</td>"
echo "</tr>"
