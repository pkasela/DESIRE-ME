import pandas as pd
import json
import os
import unicodedata
import tqdm
import subprocess
import click

def text_to_id(text):
    """
    Convert input text to id.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    text = unicodedata.normalize('NFC', str(text).encode().decode('utf8'))
    text = str(text).encode('utf-8').decode('utf-8')
    text = str(text).replace(' ', '_')
    return text

def utf_page_title(text):
    text = str(text).encode('latin-1').decode('utf-8')
    text = unicodedata.normalize('NFC', str(text).encode().decode('utf8'))
    return text

def load_jsonl(file: str):
    with open(file, 'r') as f:
        for lne in f:
            yield json.loads(lne)
            
def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def write_jsonl(file_name, data_jsonl):
    with open(file_name, 'w') as f:
        for row in tqdm.tqdm(data_jsonl, desc='Writing jsonl'):
                json.dump(row, f)
                f.write('\n')


@click.command()
@click.option(
    "--wiki_folder",
    type=str,
    required=True,
)
@click.option(
    "--dataset",
    type=str,
    required=True,
)
def main(wiki_folder, dataset):
    def get_term_categories(term, term_id):
        if term in base_categories:
            return [term]
        current_categories = category_to_list_dict.get(term_id, [])
        ipdb.set_trace()
            
        depth = 0
        if set(current_categories) & base_categories:
            return list(set(current_categories) & base_categories)
        while True:
            depth += 1
            try:
                new_categories = set(sum([category_to_list_dict.get(str(cat_to_id[x]), []) for x in current_categories], []))
            except:
                print(current_categories)
                return
            if new_categories & base_categories:
                break
            current_categories = new_categories
            if depth > 100:
                # print('Warn: No Category Found for: ', term, ' ', term_id)
                return []
                
        return list(new_categories & base_categories)

    corpus_file = f'{dataset}/corpus.jsonl'
    out_corpus = f'{dataset}/wiki_corpus.jsonl'

    category_to_label = {
        "Business": 0, "Communication": 1, "Concepts": 2, "Culture": 3, "Economy": 4, "Education": 5, "Energy": 6, "Engineering": 7, 
        "Entertainment": 8, "Ethics": 9, "Food_and_drink": 10, "Geography": 11, "Government": 12, "Health": 13, "History": 14, 
        "Human_behavior": 15, "Humanities": 16, "Information": 17, "Internet": 18, "Knowledge": 19, "Language": 20, "Law": 21, 
        "Life": 22, "Mass_media": 23, "Mathematics": 24, "Military": 25, "Nature": 26, "People": 27, "Philosophy": 28, "Politics": 29, 
        "Religion": 30, "Science": 31, "Society": 32, "Sports": 33, "Technology": 34, "Time": 35, "Universe": 36
    } 
    with open(f'{dataset}/category_to_label.json', 'w') as f:
        json.dump(category_to_label, f)
        
    page_df = pd.read_csv(f'{wiki_folder}/page.csv', encoding='latin-1', usecols=['page_id', 'page_title', 'page_namespace'])
    
    category_pages = page_df[page_df['page_namespace']==14]
    cat_to_id = dict(zip(category_pages.page_title, category_pages.page_id))
    
    if os.path.exists(f'{wiki_folder}/category_to_list.json'):
        print('Category file exists, loading')
        with open(f'{wiki_folder}/category_to_list.json', 'r') as f:
            category_to_list_dict = json.load(f) 
    else:
        print('Creating the dictionary from scratch')
        category_df = pd.read_csv(f'{wiki_folder}/category.csv', encoding='latin-1', usecols=['cat_id','cat_title','cat_pages','cat_subcats','cat_files'])
        categorylinks_df = pd.read_csv(f'{wiki_folder}/categorylinks.csv', encoding='latin-1', usecols=['cl_from', 'cl_to', 'cl_type'])

        cat_id = pd.merge(category_df, 
                        category_pages, #page_df[page_df['page_namespace']==14], 
                        how='inner', 
                        left_on='cat_title', 
                        right_on='page_title'
        )[['cat_title','page_id']]
        
        categorylinks_df = categorylinks_df[categorylinks_df['cl_to'].isin(cat_id['cat_title'])]

        category_to_list = categorylinks_df.groupby('cl_from').agg({'cl_to': list})

        category_to_list_dict = dict(zip(category_to_list.index, category_to_list.cl_to))

        with open(f'{wiki_folder}/category_to_list.json', 'w') as f:
            json.dump(category_to_list_dict, f)
            
    base_categories = [
        'Academic disciplines', 'Business', 
        'Communication', 'Concepts', 'Culture',
        'Economy', 'Education', 'Energy', 'Engineering',
        'Entertainment', 'Entities', 'Ethics', 
        'Food_and_drink', 'Geography', 'Government',
        'Health', 'History', 'Human_behavior', 'Humanities',
        'Information', 'Internet', 'Knowledge', 'Language',
        'Law', 'Life', 'Mass_media', 'Mathematics', 'Military',
        'Nature', 'People', 'Philosophy', 'Politics', 'Religion',
        'Science', 'Society', 'Sports', 'Technology', 
        'Time', 'Universe'
    ]
    base_categories = set(base_categories)
    
    page_df['utf_page_title'] = page_df.page_title.apply(utf_page_title)
    page_df_basic = page_df.sort_values('page_namespace').drop_duplicates(subset=['page_title'], keep='first')

    page_to_id_dict = dict(zip(page_df_basic['utf_page_title'], page_df_basic['page_id'].apply(str)))
    
    test_corpus = load_jsonl(corpus_file)

    test_corpus_with_category = []
    title_cat = {}

    for t in tqdm.tqdm(test_corpus, total=file_len(corpus_file)):
        found_category = title_cat.get(text_to_id(t['title']), [])
        if not found_category:
            import ipdb
            ipdb.set_trace()
            found_category = get_term_categories(text_to_id(t['title']), page_to_id_dict.get(text_to_id(t['title']), -1))
        
        if found_category:
            title_cat[text_to_id(t['title'])] = found_category
        else:
            if ":" in text_to_id(t['title']):
                title = text_to_id(t['title']).split(":")[1]
                found_category = get_term_categories(text_to_id(title), page_to_id_dict.get(text_to_id(title), -1))
        
        t['category'] = found_category
        
        test_corpus_with_category.append(t)
    
    write_jsonl(out_corpus, test_corpus_with_category)
    
    found = len([x for x in test_corpus_with_category if x['category']])
    total = len(test_corpus_with_category)
    print(f"found categories for {found} over {total} ({found/total})")

if __name__ == "__main__":
    main()