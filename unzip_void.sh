mkdir void_release
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
unzip -o void_150.zip -d void_release/
unzip -o void_500.zip -d void_release/
unzip -o void_1500.zip -d void_release/

cd void_release
mkdir -p 'void_150/data'
mkdir -p 'void_500/data'
mkdir -p 'void_1500/data'

file_id=0
for p in 150 500 1500
do
  while [ $file_id -le 56 ]; do
    unzip -o 'void_'${p}'-'${file_id}'.zip' -d 'void_'${p}'/data/'
    rm 'void_'${p}'-'${file_id}'.zip'
    file_id=$(( ${file_id} + 1 ))
  done
done