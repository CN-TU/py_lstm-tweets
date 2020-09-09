 
NUMOFLINES=$(wc -l < "train.csv")

head -$(($NUMOFLINES * 7 / 10)) train.csv > train_small.csv
tail -$(($NUMOFLINES * 3 / 10)) train.csv > aux.csv

NUMOFLINES=$(wc -l < "aux.csv")
head -$(($NUMOFLINES * 9 / 10)) aux.csv > test_small.csv
tail -$(($NUMOFLINES * 1 / 10)) aux.csv > valid_pre.csv

cut -d, -f2-6 --complement valid_pre.csv > valid_small.csv
rm aux.csv
rm valid_pre.csv
