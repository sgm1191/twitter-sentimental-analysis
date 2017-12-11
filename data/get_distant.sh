rm *.ds;
wget https://www.dropbox.com/s/igi52x84qllp56b/tw-data.neg?dl=0;
wget https://www.dropbox.com/s/y93poywml4xfi9o/tw-data.pos?dl=0;
mv tw-data.neg?dl=0 tw-data.neg.ds;
mv tw-data.pos?dl=0 tw-data.pos.ds;
## se separa con $ñ$ porque \t daba problemas
awk '{$NF=$NF" ññ negative"; print}' tw-data.neg.ds > tw-data.neg-l.ds;
rm -f tw-data.neg.ds;
mv tw-data.neg-l.ds tw-data.neg.ds;
awk '{$NF=$NF" ññ positive"; print}' tw-data.pos.ds > tw-data.pos-l.ds;
rm -f tw-data.pos.ds
mv tw-data.pos-l.ds tw-data.pos.ds;
awk 'BEGIN { srand() } { print rand(), $0 }' tw-data.neg.ds tw-data.pos.ds | sort -n | cut -f2- -d" " > distant-data.ds;
rm -f tw-data.*;
