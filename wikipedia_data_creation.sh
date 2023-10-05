WIKIPEDIA_FOLDER='wikipedia_data'
mkdir $WIKIPEDIA_FOLDER

category_link='https://archive.org/download/enwiki-20181220/enwiki-20181220-category.sql.gz'
wget -P $WIKIPEDIA_FOLDER $category_link
zcat $WIKIPEDIA_FOLDER/enwiki-20181220-category.sql.gz | python3 mysqldump_to_csv.py >> $WIKIPEDIA_FOLDER/category.csv

categorylinks_link='https://archive.org/download/enwiki-20181220/enwiki-20181220-categorylinks.sql.gz'
wget -P $WIKIPEDIA_FOLDER $categorylinks_link
zcat $WIKIPEDIA_FOLDER/enwiki-20181220-categorylinks.sql.gz | python3 mysqldump_to_csv.py >> $WIKIPEDIA_FOLDER/categorylinks.csv

page_link='https://archive.org/download/enwiki-20181220/enwiki-20181220-page.sql.gz'
wget -P $WIKIPEDIA_FOLDER $page_link
zcat $WIKIPEDIA_FOLDER/enwiki-20181220-page.sql.gz | python3 mysqldump_to_csv.py >> $WIKIPEDIA_FOLDER/page.csv
