import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, coo_matrix, find
import implicit
import time
from collections import defaultdict
import ast


diversity = 0.5

df = pd.read_csv('data/movielens/ratings_lens.csv', sep=',')
df = df[['userId', 'movieId', 'rating']]

df_meta = pd.read_csv('data/movielens/movies.csv', sep=',')
df_meta = df_meta[['movieId','genres']]
df_meta = df_meta[df_meta['genres']!='(no genres listed)']

item_category = pd.merge(df, df_meta, left_on='movieId', right_on='movieId')
item_category = item_category[item_category['genres']!='(no genres listed)']
item_category.head()

genres_items = defaultdict(list)
for i,j in df_meta.iterrows():
    for g in j['genres'].split('|'):
        genres_items[g] += [j['movieId']]
        
for cat_1 in list(genres_items.keys())[:]:
    print(cat_1)
    list_cat = list(genres_items.keys())
    #cat_1 = 'Comedy' # 'Horror' # 'IMAX' # 'Musical' # 'Mystery' # 'Romance' # 'Sci-Fi' # 'Thriller' # 'War' # 'Western' 
    list_cat.remove(cat_1)
    cat_2 = 'Others'


    others = set([j for i in [genres_items[i] for i in list_cat] for j in i]).difference(set(genres_items[cat_1]))
    movies_cat = set(genres_items[cat_1]).union(set(others))
    
    dict_group = {}
    for m in movies_cat:
        if m in genres_items[cat_1]:
            dict_group[m] = cat_1
        else:#elif m in genres_items[cat_2]:
            dict_group[m] = cat_2
            
    df_categories = item_category[item_category.movieId.isin(movies_cat)]
    df_categories['group'] = df_categories['movieId'].apply(lambda x: dict_group[x])
    
    user_to_id = {u:i for i,u in enumerate(list(df_categories.userId.unique()))}
    item_to_id = {u:i for i,u in enumerate(list(df_categories.movieId.unique()))}

    df_categories['user_id'] = df_categories['userId'].apply(lambda x: user_to_id[x])
    df_categories['item_id'] = df_categories['movieId'].apply(lambda x: item_to_id[x])

    groupby_df = df_categories.groupby('userId').mean()
    user_avg = {u: groupby_df['rating'][u] for u in groupby_df.index}

    df_categories['avg_user_rat'] = df_categories['userId'].apply(lambda x: user_avg[x])
    df_categories['adj_rating'] = df_categories['rating'] - df_categories['avg_user_rat']

    id_to_user = {u:i for i,u in user_to_id.items()}
    id_to_item = {u:i for i,u in item_to_id.items()}
    
    groupby_df_item = df_categories.groupby('item_id').first()[['movieId', 'group']]
    dictionary_cat = {cat_1 : 'red',
                      cat_2 : 'blue'}
    
    red = groupby_df_item[groupby_df_item['group'] == cat_1]
    blue = groupby_df_item[groupby_df_item['group'] == cat_2]
    
    
    
    cols, rows = list(df_categories['user_id']), list(df_categories['item_id'])
    data = np.array(df_categories['adj_rating'], dtype=float)
    n_users = len(df_categories.user_id.unique())
    n_items = len(df_categories.item_id.unique())

    M = coo_matrix((data, (rows, cols)), shape=(n_items, n_users))
    M = csr_matrix(M)
    
    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20, random_state=50)

    s = time.time()

    data_conf = M.astype('double')


    model.fit(data_conf)

    print(time.time()-s)
    
    from scipy.stats import powerlaw

    a = .3
    x = np.linspace(0,
                    1, 21)[1:]
    rv = powerlaw(a)
    values = rv.pdf(x)
    normalized = np.array(values)/np.sum(values)
    
    mean_rating = df_categories.groupby('movieId').mean()
    num_ratings = df_categories.groupby('movieId').count()
    
    num_rats = {}
    for i in num_ratings.iterrows():#.iloc[1]['rating']
        obj = i[1]
        item = item_to_id[i[0]]
        num_rats[item] = obj['rating']/len(df_categories)
        
    mean_rats = {}
    for i in mean_rating.iterrows():#.iloc[1]['rating']
        obj = i[1]
        item = item_to_id[i[0]]
        mean_rats[item] = obj['rating']
        
    score = {}
    for i in mean_rats:
        score[i] = mean_rats[i]*num_rats[i]
        
    edges = []
    count = 0
    list_items = df_categories.item_id.unique()
    print(len(list_items))

    col_dic = {}
    for i in red.index:
        col_dic[i] = 'red'
    for i in blue.index:
        col_dic[i] = 'blue'
        
        
    for item in blue.index:
        # Get new weights
        #item_weight = np.array([i*score[item] for i in normalized])
        #item_norm = item_weight/np.sum(item_weight)


        sim = model.similar_items(item, N=21)[1:]
        edges_w = [(item, i[0], normalized[e]*score[i[0]]) for e,i in enumerate(sim) if item != i[0]] # map(lambda x: (item, x[0], x[1]), sim) 
        somma = np.sum(list(zip(*edges_w))[-1])
        edges += [(i,j,k/somma) for i,j,k in edges_w]
        count += 1
        if count % 1000 == 0:
            print(count)
            
            
    for item in red.index:
        sim = model.similar_items(item, N=len(list_items))[1:]
        #sim = model.similar_items(item, N=21)[1:]
        col_sim = [(i,j,col_dic[i]) for i,j in sim]
        tot = 20
        
        num_red = int(tot*diversity)
        num_blue = tot - num_red

#         reds_nodes = [i for i in sim if col_dic[i[0]]=='red']
#         blues_nodes = [i for i in sim if col_dic[i[0]]=='blue']
        
        max_col = {'red': num_red,
                  'blue': num_blue}
        tot_color = defaultdict(int)
        edges_w = [] 

        for i,j,c in col_sim:
            
            if tot_color[c] < max_col[c]:
                tot_color[c] += 1
                edges_w += [(item, i, normalized[len(edges_w)-1]*score[i])]#*score[i]
            
            if len(edges_w) >= tot:
                #print(tot_color['red'], 'red')
                #print(tot_color['blue'], 'blue')
                break
        
        else:
            edges_w = [(item, i[0], normalized[e]*score[i[0]]) for e,i in enumerate(sim) if item != i[0]] # map(lambda x: (item, x[0], x[1]), sim) 
            

        #print(edges_w)      
        somma = np.sum(list(zip(*edges_w))[-1])
        edges += [(i,j,k/somma) for i,j,k in edges_w]
        #edges_tot += edges_w
        count += 1
        if count % 1000 == 0:
            print(count)
    
    df_edges = pd.DataFrame(edges, columns=['source','target','similarity'])
    df_edges['source_cat'] = df_edges['source'].apply(lambda x: dict_group[id_to_item[x]])
    df_edges['target_cat'] = df_edges['target'].apply(lambda x: dict_group[id_to_item[x]])
    df_edges.sort_values(by=['source'], inplace=True)
    
    with open('data/movielens/'+ cat_1 + '/nodes/' + 'clickstream_weighted_edges_diversity.tsv', 'w') as f:
        for i in df_edges.iterrows():
            obj = i[1]
            src = obj['source']
            trg = obj['target']
            clicks = obj['similarity']

            f.write('{}\t{}\t{}\n'.format(src, trg, clicks))
            
    print('\n\n\n\n\n\n\n')