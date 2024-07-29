import json


import pandas as pd

import nltk
import numpy as np
import plotly.express as px
import folium

nltk.download('punkt')

reviewfile = '/Users/bohu/Dropbox/UIUC_Computer_Science/CS598_DataMiningCapstone/application/datasets/yelp_academic_dataset_review.json'
businessfile = '/Users/bohu/Dropbox/UIUC_Computer_Science/CS598_DataMiningCapstone/application/datasets/yelp_academic_dataset_business.json'


# Reading JSON from a file

def read_review_data(reviewfile, businessfile, dish_name=None):
    if dish_name is None:
        dish_name = ['prime rib', 'cheese']
    if reviewfile is None:
        reviewfile = '/Users/bohu/Dropbox/UIUC_Computer_Science/CS598_DataMiningCapstone/application/datasets/yelp_academic_dataset_review.json'
    if businessfile is None:
        businessfile = '/Users/bohu/Dropbox/UIUC_Computer_Science/CS598_DataMiningCapstone/application/datasets/yelp_academic_dataset_business.json'

    with open(reviewfile, 'r') as file:
        review_from_file = file.readlines()
    with open(businessfile, 'r') as file:
        business_from_file = file.readlines()

    review_dict = {}
    for i in range(len(review_from_file)):
        review_dict[json.loads(review_from_file[i])['text']] = (
            json.loads(review_from_file[i])['business_id'], json.loads(review_from_file[i])['stars'])

    business_dict = {}
    for i in range(len(business_from_file)):
        business_dict[json.loads(business_from_file[i])['business_id']] = {
            'Restaurant': json.loads(business_from_file[i])['name'],
            'categories': json.loads(business_from_file[i])['categories'],
            'Latitude': json.loads(business_from_file[i])['latitude'],
            'Longitude': json.loads(business_from_file[i])['longitude']}
    print('line 1')
    rid2rating = {}
    rid2reviews = {}
    r = ['Restaurants']

    # cuisine = ['American (New)', 'Chinese', 'Indian', 'Italian', 'Mediterranean', 'Mexican']
    dish_name = dish_name
    for review, (business_id, rating) in review_dict.items():
        # check if the business category is restaurant
        business_categories = business_dict[business_id]['categories']
        intersection = set(r) & set(business_categories)

        if business_id in rid2reviews:
            rid2reviews[business_id].append(rating)
        else:
            rid2reviews[business_id] = [rating]

        if intersection:
            # check if dish name is in a review
            dishes = [dish for dish in dish_name if dish.lower() in review.lower()]
            if dishes:
                if business_id in rid2rating:
                    rid2rating[business_id].append(rating)
                else:
                    rid2rating[business_id] = [rating]
    print('line 2')
    restaurant_dish_score = {rid: {'mean': np.mean(ratings).round(2), 'n': len(ratings)} if ratings else {} for
                             rid, ratings in rid2rating.items()}

    restaurant_score = {rid: {'mean': np.mean(ratings).round(2), 'n': len(ratings)} if ratings else {} for rid, ratings
                        in rid2reviews.items()}

    restaurant_dish_score_sorted = pd.DataFrame.from_dict(restaurant_dish_score, orient='index')
    restaurant_dish_score_sorted.reset_index(inplace=True)
    restaurant_dish_score_sorted.rename(columns={'index': 'rid', 'mean': 'Average Rating about the Queried Dish',
                                                 'n': 'Number of Reviews about the Queried Dish'}, inplace=True)

    restaurant_score_sorted = pd.DataFrame.from_dict(restaurant_score, orient='index')
    restaurant_score_sorted.reset_index(inplace=True)
    restaurant_score_sorted.rename(columns={'index': 'rid', 'mean': 'Average Rating', 'n': 'Number of Reviews'},
                                   inplace=True)
    print('line 3')
    # output in a DataFrame
    ranked_restaurant_score = pd.DataFrame(restaurant_score_sorted, columns=['rid', 'Rating'])
    restaurant_info = pd.DataFrame.from_dict(business_dict, orient='index')
    restaurant_info.reset_index(inplace=True)
    restaurant_info.rename(columns={'index': 'rid'}, inplace=True)

    restaurants_df_1 = pd.merge(restaurant_dish_score_sorted, restaurant_score_sorted, on='rid', how='left')
    restaurants_df = pd.merge(restaurants_df_1, restaurant_info, on='rid', how='left')
    print('line 4')
    numberofreview_avg = restaurants_df['Number of Reviews'].mean()
    numberdishreview_avg = restaurants_df['Number of Reviews about the Queried Dish'].mean()
    restaurants_df['Ratio'] = restaurants_df['Number of Reviews about the Queried Dish'] / restaurants_df[
        'Number of Reviews']
    # Sort by 'Dish' and then by 'Rating' within each 'Dish'
    sorted_restaurants_df = restaurants_df[(restaurants_df['Number of Reviews'] > numberofreview_avg) & (
            restaurants_df['Number of Reviews about the Queried Dish'] > numberdishreview_avg) & (
                                                   restaurants_df['Average Rating'] > 3)].sort_values(
        by=['Average Rating about the Queried Dish', 'Ratio', 'Restaurant'], ascending=[False, False, True]).iloc[:10]
    return sorted_restaurants_df


# Plotting the sorted restaurants on a map
def plot_restaurants_on_map(restaurants_df):
    # map_center = [restaurants_df['Latitude'].mean(), restaurants_df['Longitude'].mean()]
    map_center = [37.761111, -100.018333]
    my_map = folium.Map(location=map_center, zoom_start=5)

    for idx, row in restaurants_df.iterrows():
        popup = folium.Popup(
            f"{row['Restaurant']}<br>Restaurant Rating: {row['Average Rating']}<br>Dish Rating: {row['Average Rating about the Queried Dish']}<br>Number of Reviews: {row['Number of Reviews']}",
            max_width=300)
        folium.Marker([row['Latitude'], row['Longitude']],
                      popup=popup,
                      icon=folium.Icon(icon='home', color='lightgreen')).add_to(my_map)
    map_html = my_map._repr_html_()
    return map_html


def plot_popular_dishes(dishes_data):
    fig = px.bar(dishes_data, x='Restaurant', y='Average Rating about the Queried Dish', title='Popular Restaurant')
    fig.update_layout(xaxis_title='Restaurant',
                      yaxis_title='Popularity',
                      xaxis_tickangle=45,
                      xaxis=dict(tickfont=dict(size=10)))
    plot_html = fig.to_html(full_html=False, include_plotlyjs=True)
    # Assuming `data` is your object (e.g., a DataFrame)
    # with open('plot_second.pkl', 'wb') as f:
    #    pickle.dump(plot_html, f)

    return plot_html

    # Create a bar plot
    # plt.figure(figsize=(16, 8))
    # plt.bar(dishes_data['Restaurant'], dishes_data['Average Rating about the Queried Dish'], color='skyblue')
    # plt.title('Popular Restaurant')
    # plt.xlabel('Restaurant')
    # plt.ylabel('Popularity')
    # plt.xticks(rotation=90, )

    # Save plot to a PNG image in memory
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # plt.close()
    # image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    # return f'<img src="data:image/png;base64,{image_base64}" alt="Bar Plot">'
