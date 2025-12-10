import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from flask import Flask, render_template_string, request, jsonify

# --- データ読み込みとモデル学習 ---

# 映画の評価データを読み込み
df_ratings = pd.read_csv("./ratings_100k.csv", sep=",")
df_ratings = df_ratings.iloc[:,0:3]

# df_ratingsのカラム名を変更: 'movieId' -> 'movie_id'
df_ratings = df_ratings.rename(columns={'movieId': 'movie_id'})

# 映画のデータを読み込み
# 💡 修正点 1: skipinitialspace=Trueを追加 (区切り文字の前後の空白を無視)
# 💡 修正点 2: ファイルにヘッダーデータが紛れ込んでいる可能性を考慮し、header=Noneを維持しつつ、
#             データ型を正しく推測できるようにします。
df_movies = pd.read_csv(
    "./movies_100k.csv", 
    sep="|", 
    header=None, 
    encoding="latin-1",
    names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url'] + [f'genre_{i}' for i in range(19)],
    # skipinitialspace=True # 今回はセパレータが'|'なので不要だが、念のため。
)
# movie_idとmovie_titleのみを抽出
df_movies = df_movies[['movie_id', 'movie_title']]
df_movies['movie_title_clean'] = df_movies['movie_title'].str.replace(r' \(\d{4}\)', '', regex=True)

# 💡 修正点 3: データ型変換時にエラーを無視し、変換できなかった値をNaN (欠損値) にする
#             その後、欠損値を0で埋めてint型に変換することで、不正なデータ行を処理から除外する
try:
    # エラーが発生した場合、その要素はNaNになる
    df_movies['movie_id'] = pd.to_numeric(df_movies['movie_id'], errors='coerce') 
    # NaN（欠損値）を削除またはゼロで埋める（ここでは簡単のためドロップ）
    df_movies.dropna(subset=['movie_id'], inplace=True) 
    # 残った値を整数型に変換
    df_movies['movie_id'] = df_movies['movie_id'].astype(int)

    df_ratings['movie_id'] = df_ratings['movie_id'].astype(int) # こちらは通常成功するはず
except ValueError as e:
    print(f"型変換エラー: {e}")
    # プログラムを終了せず続行するため、ここでは pass

# 評価データと映画データを 'movie_id' でマージ
df_merged = pd.merge(df_ratings, df_movies, on='movie_id')

# ユーザー×映画のピボットテーブルを作成 (モデル学習用)
df_piv = df_merged.pivot(index="movie_id", columns="userId", values="rating").fillna(0) 

# 疎行列に変換
df_sp = csr_matrix(df_piv.values)

# 類似度計算モデルを作成
rec = NearestNeighbors(n_neighbors=11, algorithm="brute", metric="cosine")
rec_model = rec.fit(df_sp)

# --- (以下、関数定義 get_recommendations, get_top_rated_movies, Flaskルート関数は変更なし) ---
# ※ 長くなるため省略しますが、前回お送りしたコードの続きをご使用ください。

def get_recommendations(movie_ids):
    # ... (前回のコードの内容をそのまま使用) ...
    recommendations = {}
    
    for movie_id in movie_ids:
        try:
            movie_idx = df_piv.index.get_loc(movie_id)
            
            distance, indice = rec_model.kneighbors(df_sp[movie_idx], n_neighbors=11)
            
            similar_movie_indices = indice.flatten()
            similar_movie_ids = [df_piv.index[i] for i in similar_movie_indices]
            
            scores = 1 - distance.flatten()
            
            for i in range(1, len(similar_movie_ids)): 
                rec_id = similar_movie_ids[i]
                rec_score = scores[i]
                recommendations[rec_id] = recommendations.get(rec_id, 0) + rec_score
        
        except KeyError:
            continue

    sorted_recs = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)
    
    final_recs = []
    for rec_id, _ in sorted_recs:
        if rec_id not in movie_ids:
            final_recs.append(rec_id)
        if len(final_recs) >= 5:
            break
            
    rec_titles = []
    for movie_id in final_recs:
        title_series = df_movies[df_movies['movie_id'] == movie_id]['movie_title']
        if not title_series.empty:
             rec_titles.append(title_series.iloc[0])
        
    return rec_titles

def get_top_rated_movies():
    df_mean_rating = df_merged.groupby('movie_id')['rating'].mean().reset_index()

    df_count_rating = df_merged.groupby('movie_id')['rating'].count().reset_index()
    min_ratings_threshold = df_count_rating['rating'].median()
    popular_movies = df_count_rating[df_count_rating['rating'] >= min_ratings_threshold]['movie_id']
    df_mean_rating = df_mean_rating[df_mean_rating['movie_id'].isin(popular_movies)]

    top_5_ids = df_mean_rating.sort_values(by='rating', ascending=False).head(5)['movie_id'].tolist()

    top_5_titles = []
    for movie_id in top_5_ids:
        title_series = df_movies[df_movies['movie_id'] == movie_id]['movie_title']
        if not title_series.empty:
             top_5_titles.append(title_series.iloc[0])
        
    return top_5_titles


app = Flask(__name__)

movie_list = df_movies[['movie_id', 'movie_title']].sort_values(by='movie_title').values.tolist()


@app.route('/')
def index():
    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>映画推薦システム</title>
    <style>
        body {{ font-family: sans-serif; padding: 20px; }}
        .container {{ max-width: 600px; margin: auto; border: 1px solid #ccc; padding: 30px; border-radius: 8px; }}
        h2 {{ color: #333; }}
        select, button {{ padding: 10px; margin: 10px 0; width: 100%; box-sizing: border-box; }}
        button {{ background-color: #007bff; color: white; border: none; cursor: pointer; }}
        button:hover {{ background-color: #0056b3; }}
        .recommendation-list {{ margin-top: 20px; }}
        .recommendation-list ol {{ padding-left: 20px; }}
        .recommendation-list li {{ margin-bottom: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>お好きな映画を3つ選択してください</h2>
        <form id="recommendation-form" action="/recommend" method="post">
            
            <div>
                <label for="movie1">1つ目の映画:</label>
                <select name="movie1" id="movie1">
                    <option value="">-- 映画を選択してください --</option>
                    {"".join([f'<option value="{mid}">{title}</option>' for mid, title in movie_list])}
                </select>
            </div>

            <div>
                <label for="movie2">2つ目の映画:</label>
                <select name="movie2" id="movie2">
                    <option value="">-- 映画を選択してください --</option>
                    {"".join([f'<option value="{mid}">{title}</option>' for mid, title in movie_list])}
                </select>
            </div>

            <div>
                <label for="movie3">3つ目の映画:</label>
                <select name="movie3" id="movie3">
                    <option value="">-- 映画を選択してください --</option>
                    {"".join([f'<option value="{mid}">{title}</option>' for mid, title in movie_list])}
                </select>
            </div>

            <button type="submit">オススメ映画を表示</button>
        </form>
        
        <div id="recommendation-result" class="recommendation-list">
            </div>
    </div>
</body>
</html>
"""


@app.route('/recommend', methods=['POST'])
def recommend():
    movie_ids = []
    for key in ['movie1', 'movie2', 'movie3']:
        movie_id_str = request.form.get(key)
        if movie_id_str and movie_id_str.isdigit():
            movie_ids.append(int(movie_id_str))
    
    valid_selection_count = len(set(movie_ids))

    if valid_selection_count >= 3:
        recommendations = get_recommendations(list(set(movie_ids)))
        header = f"🎬 選択された{valid_selection_count}作品に基づくオススメ映画トップ5"
    else:
        recommendations = get_top_rated_movies()
        header = "⭐ 好きな映画が未選択のため、総合的に評価値が高い映画トップ5"

    recommendation_html = f"<h3>{header}</h3><ol>"
    for i, title in enumerate(recommendations, 1):
        recommendation_html += f"<li>{title}</li>"
    recommendation_html += "</ol><p><a href='/'>選び直す</a></p>"
    
    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>オススメ映画トップ5</title>
    <style>
        body {{ font-family: sans-serif; padding: 20px; }}
        .container {{ max-width: 600px; margin: auto; border: 1px solid #ccc; padding: 30px; border-radius: 8px; }}
        h2, h3 {{ color: #333; }}
        a {{ display: inline-block; margin-top: 15px; padding: 8px 15px; background-color: #f0f0f0; text-decoration: none; border: 1px solid #ccc; border-radius: 4px; color: #333; }}
        a:hover {{ background-color: #ddd; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>オススメ映画トップ5</h2>
        {recommendation_html}
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    print("サーバーを起動します... http://127.0.0.1:5000/")
    app.run(host='127.0.0.1', port=5000, debug=True)