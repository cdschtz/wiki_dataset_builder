import os
import json
import datetime
import requests

session = requests.Session()
url = "https://en.wikipedia.org/w/api.php"
headers = {
    'User-Agent': 'MetadataLoaderBot 0.0',
    'From': os.getenv('EMAIL')
}
session.headers.update(headers)


def extract_metadata(ids):
    query_value = '|'.join(ids)
    params = {
        "action": "query",
        "pageids": query_value,
        "format": "json",
        "prop": "info|pageviews|categories"
    }
    response = session.get(url=url, params=params)
    pages_data = response.json()["query"]["pages"]
    categories = {}
    for id in ids:
        try:
            cats = pages_data[id]["categories"]
            for cat in cats:
                if cat['title'] in categories:
                    categories[cat['title']] += 1
                else:
                    categories[cat['title']] = 1
        except:
            print('Error occured for id ' + id)
    return categories


def extract_other_data(ids):
    # as ids are duplicate only pick every second one
    query_value = '|'.join(ids)
    problem_ids = []
    params = {
        "action": "query",
        "pageids": query_value,
        "format": "json",
        "prop": "info|pageviews"
    }
    response = session.get(url=url, params=params)
    pages_data = response.json()["query"]["pages"]
    other_info = {}
    for id in ids:
        touched = None
        length = None
        pageviews = None

        try:
            touched = pages_data[id]["touched"]
            length = pages_data[id]["length"]
            if "pageviews" in pages_data[id]:
                pageviews = sum([i for i in pages_data[id]["pageviews"].values() if i])
        except:
            print('Error occurred for id ' + id)
            problem_ids.append(id)

        other_info[id] = {
            "touched": touched,
            "length": length,
            "pageviews": pageviews
        }
    return other_info, problem_ids


def actualize_data_points(file_path, other_info, ids, tmps, problem_ids):
    with open(file_path, 'w') as update_file:
        counter = 0
        for id in ids:
            human_data_point = tmps[counter]
            machine_data_point = tmps[counter+1]
            new_meta_info = other_info[id]
            for key, val in zip(new_meta_info.keys(), new_meta_info.values()):
                human_data_point['meta'][key] = val
                machine_data_point['meta'][key] = val
            if id in problem_ids:
                pass
            else:
                update_file.write(json.dumps(human_data_point) + '\n')
                update_file.write(json.dumps(machine_data_point) + '\n')
            counter += 2


def get_all_files(file_range=(0,1), in_path='/Users/christopher/Coding/BA_Code/thesis-data/output_after_gen/', verbose=False):
    all_files = []
    # get all files (also in subdirs)
    for root, directories, filenames in os.walk(in_path):
        for filename in filenames:
            if 'DS_Store' in filename:
                continue
            else:
                all_files.append(os.path.join(root, filename))

    all_files = sorted(all_files)
    return all_files[file_range[0]:file_range[1]]


def build_metadata(file_range):
    all_files = get_all_files(file_range)
    categories_object = {}
    with open('./categories.json', 'r') as categories_file:
        categories_object = json.load(categories_file)
        for mynum, file in enumerate(all_files):
            ids = []
            tmps = []
            with open(file, 'r') as r_f:
                for line in r_f:
                    tmp = json.loads(line)
                    tmps.append(tmp)
                    ids.append(tmp['meta']['id'])

            # actualize pageviews, length, last edit
            # ids_processed = 0
            # step_size = 100
            # other_info = {}
            # problem_ids = []
            # while ids_processed < len(ids):
            #     other_information, problem_ids_new = extract_other_data(ids[ids_processed:ids_processed + step_size:2])
            #     other_info.update(other_information)
            #     problem_ids = problem_ids + problem_ids_new
            #     ids_processed += step_size
            # actualize_data_points(file, other_info, ids[::2], tmps, problem_ids)
            # print(f"Finished file {mynum + 1}")

            # MediaWiki Api has a limit of 50 ids per generator
            # get category info
            ids_processed = 0
            step_size = 2
            while ids_processed < len(ids):
                cats = extract_metadata(ids[ids_processed:ids_processed+step_size-1])
                ids_processed += step_size
                for cat in cats.keys():
                    if cat in categories_object:
                        categories_object[cat] += cats[cat]
                    else:
                        categories_object[cat] = cats[cat]

    with open('./categories.json', 'w') as categories_file:
        json.dump(categories_object, categories_file, indent=4, sort_keys=True)


build_metadata((600, 700))

time_info = datetime.datetime.now().isoformat(timespec='minutes')
# with open(os.getcwd() + '/example_results/' + time_info + '.json', 'w') as fp:
#     import json
#     json.dump(DATA, fp, ensure_ascii=False, indent=4)

# if DATA['query']['search'][0]['title'] == SEARCHPAGE:
#     print("Your search page '" + SEARCHPAGE + "' exists on English Wikipedia")
