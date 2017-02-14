source getDataResults.sh $1 > tmp
if [[ -e result.html ]]; then 
echo "result.html exists"
else 
cp ResHeader.html result.html
fi
cat tmp >> result.html
sudo cp result.html /var/www/html/
sudo cp $1 /var/www/html/log
