WIKIPEDIA_FOLDER=wikipedia_data
mkdir $WIKIPEDIA_FOLDER

echo "cat_id,cat_title,cat_pages,cat_subcats,cat_files" > $WIKIPEDIA_FOLDER/category.csv
category_link=https://archive.org/download/enwiki-20181220/enwiki-20181220-category.sql.gz
wget -P $WIKIPEDIA_FOLDER $category_link
zcat $WIKIPEDIA_FOLDER/enwiki-20181220-category.sql.gz | python3 mysqldump_to_csv.py >> $WIKIPEDIA_FOLDER/category.csv

echo "cl_from,cl_to,cl_sortkey,cl_timestamp,cl_sortkey_prefix,cl_collation,cl_type" > $WIKIPEDIA_FOLDER/categorylinks.csv
categorylinks_link=https://archive.org/download/enwiki-20181220/enwiki-20181220-categorylinks.sql.gz
wget -P $WIKIPEDIA_FOLDER $categorylinks_link
zcat $WIKIPEDIA_FOLDER/enwiki-20181220-categorylinks.sql.gz | python3 mysqldump_to_csv.py >> $WIKIPEDIA_FOLDER/categorylinks.csv


echo "page_id,page_namespace,page_title,page_restrictions,page_counter,page_is_redirect,page_is_new,page_random,page_touched,page_links_updated,page_latest,page_len,page_content_model,page_lang" > $WIKIPEDIA_FOLDER/page.csv
page_link=https://archive.org/download/enwiki-20181220/enwiki-20181220-page.sql.gz
wget -P $WIKIPEDIA_FOLDER $page_link
zcat $WIKIPEDIA_FOLDER/enwiki-20181220-page.sql.gz | python3 mysqldump_to_csv.py >> $WIKIPEDIA_FOLDER/page.csv
