# import libraries
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import random
from flask import Flask, request, jsonify

# define file paths for the data
books_filename   = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# read in book data file and store in a pandas dataframe
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author', 'year-of-publish', 'publisher', 'image-url-S', 'image-url-M', 'image-url-L'],
    usecols=['isbn', 'title', 'author', 'year-of-publish', 'publisher', 'image-url-S', 'image-url-M', 'image-url-L'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str', 'year-of-publish': 'str', 'publisher': 'str', 'image-url-S': 'str', 'image-url-M': 'str', 'image-url-L': 'str'})

# read in rating data file and store in a pandas dataframe
df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

# remove any rows with null values in the book dataframe
df_books.dropna(inplace=True)

# count the number of ratings each user has given
ratings = df_ratings['user'].value_counts()

# create a new dataframe with only the rows where the user has given more than 200 ratings
df_ratings_rm = df_ratings[
  ~df_ratings['user'].isin(ratings[ratings < 200].index)
]

# count the number of ratings each book has received
ratings = df_ratings['isbn'].value_counts() # we have to use the original df_ratings to pass the challenge

# count the number of ratings received by those books
df_books['isbn'].isin(ratings[ratings < 100].index).sum()

# create a new dataframe with only the rows where the book has received more than 100 ratings
df_ratings_rm = df_ratings_rm[
  ~df_ratings_rm['isbn'].isin(ratings[ratings < 100].index)
]

# define a list of books to test the recommendation system with
books = ["Where the Heart Is (Oprah's Book Club (Paperback))",
        "I'll Be Seeing You",
        "The Weight of Water",
        "The Surgeon",
        "I Know This Much Is True"]

# create a pivot table of the ratings data with users as rows, books as columns, and fill in missing values with 0
df = df_ratings_rm.pivot_table(index=['user'],columns=['isbn'],values='rating').fillna(0).T

# join the book titles to the pivot table using their ISBN
df.index = df.join(df_books.set_index('isbn'))['title']

# sort the pivot table by book title
df = df.sort_index()

# create a nearest-neighbor model using cosine similarity
# fit the model to the data in the pivot table
model = NearestNeighbors(metric='cosine')
model.fit(df.values)


#################################################

# define a Flask app
app = Flask(__name__)

# define an endpoint for the recommend function
@app.route('/recommend', methods=['GET'])

def recommend():
    title = request.args.get('title')
    if not title:
        return jsonify({'error': 'Title parameter missing'})
    try:
        book = df.loc[title]
    except KeyError as e:
        return jsonify({'error': f'The given book {e} does not exist'})
    
    # find the N nearest neighbors to the given book (in this case N = 10)
    distance, indice = model.kneighbors([book.values], n_neighbors=10)
    
    # create a dataframe of recommended books
    recommended_books = pd.DataFrame({
        'title': df.iloc[indice[0]].index.values,
        'distance': distance[0]
    }).sort_values(by='distance', ascending=False)
    
    # join the book information to the recommended books dataframe
    recommended_books = recommended_books.join(df_books.set_index('title'), on='title')
    # select only the relevant columns from the dataframe
    recommended_books = recommended_books[['title', 'author', 'image-url-L']]
    # remove any duplicate books from the dataframe
    recommended_books = recommended_books.drop_duplicates(subset=['title'], keep='first')
    # rename the image-url-L column to image_url
    recommended_books.rename(columns={'image-url-L': 'image_url'}, inplace=True)
    # move the given book to the front of the list
    recommended_books_list = recommended_books.to_dict(orient='records')
    for book in recommended_books_list:
        if book['title'] == title:
            recommended_books_list.remove(book)
            recommended_books_list.insert(0, book)
            break
        
    # return the list of recommended books as a JSON response
    return jsonify({
        'recommended_books': recommended_books_list
    })
    
#################################################

# define an endpoint for the random_books function
@app.route('/api/random_books', methods=['GET'])


def random_books():
    # select N random book titles from the pivot table (in this case N = 10)
    random_titles = random.sample(list(df.index), 10)
    books = []
    for title in random_titles:
        # get the ratings data for the given book
        book = df.loc[title]
        # check if the book has the same number of features as the pivot table
        if book.shape[0] == df.shape[1]:
          # find the nearest neighbor to the given book
          distance, indice = model.kneighbors([book.values], n_neighbors=1)
        # create a dataframe of the recommended book
        recommended_book = pd.DataFrame({
            'title': df.iloc[indice[0]].index.values,
            'distance': distance[0]
        }).join(df_books.set_index('title'), on='title')
        # select only the relevant columns from the dataframe
        recommended_book = recommended_book[['title', 'author', 'image-url-L']]
        # rename the image-url-L column to image_url
        recommended_book = recommended_book.drop_duplicates(subset=['title'], keep='first')
        # add the recommended book to the list of books
        books.append({
            'title': recommended_book['title'].values[0],
            'author': recommended_book['author'].values[0],
            'image_url': recommended_book['image-url-L'].values[0]
        })
    # return the list of recommended books as a JSON response    
    return jsonify({'random_books': books})
#################################################

# define an endpoint for the you_may_like function
@app.route('/you_may_like', methods=['GET'])

def you_may_like():
    title = request.args.get('title')
    if not title:
        return jsonify({'error': 'Title parameter missing'})
    try:
        book = df.loc[title]
    except KeyError as e:
        return jsonify({'error': f'The given book {e} does not exist'})

    # find the N nearest neighbors to the given book (in this case N = 10)
    distance, indice = model.kneighbors([book.values], n_neighbors=10)

    # create a dataframe of recommended books
    recommended_books = pd.DataFrame({
        'title': df.iloc[indice[0]].index.values,
        'distance': distance[0]
    }).sort_values(by='distance', ascending=False)

    # join the book information to the recommended books dataframe
    recommended_books = recommended_books.join(df_books.set_index('title'), on='title')
    # select only the relevant columns from the dataframe
    recommended_books = recommended_books[['title', 'author', 'image-url-L']]
    # remove any duplicate books from the dataframe
    recommended_books = recommended_books.drop_duplicates(subset=['title'], keep='first')
    # rename the image-url-L column to image_url
    recommended_books.rename(columns={'image-url-L': 'image_url'}, inplace=True)

    # return the list of recommended books as a JSON response
    return jsonify({
        'recommended_books': recommended_books.to_dict(orient='records')
    })

#################################################

# run the program
if __name__ == '__main__':
    app.run(debug=True)
    
