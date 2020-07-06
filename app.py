import dash
import nltk
import string
import base64
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import psycopg2
from nltk.corpus import stopwords
from wordcloud import WordCloud
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import tensorflow as tf
import plotly.express as px
from keras.models import load_model
from sklearn.externals import joblib
from nltk.stem.snowball import SnowballStemmer
from tensorflow.python.keras.backend import set_session

sess = tf.Session()
set_session(sess)

ext_font = "https://use.fontawesome.com/releases/v5.8.1/css/all.css"
ext_css = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
dashapp = dash.Dash(__name__,url_base_pathname='/',external_stylesheets=[dbc.themes.BOOTSTRAP,ext_font,ext_css])

server = dashapp.server

# Switching to multiple tabs. Allowed callbacks outside of Main Layout
dashapp.config['suppress_callback_exceptions'] = True

env = "local"

if env == 'local':
    conn = psycopg2.connect(host="localhost", port="5432", user="Shubham", password="", dbname="evoc_core")

custom_stopwords_df = pd.read_csv('./Wordcloud/custom_stopwords.csv')

full_merge_df = pd.read_sql_query("SELECT * FROM full_merge", conn)
side_effects_df = pd.read_sql_query("SELECT * FROM drug_side_effects", conn)


# Tab 1 datasets
full_merge_sorted_drugs = full_merge_df.sort_values(by='Drug')
emotions_df = full_merge_df[['Drug','anger','anticipation','disgust','fear','joy','sadness','surprise','trust']]
recommended_drugs_df = full_merge_df[['Drug','Condition','Sentiment','Predicted_rating']]

all_drugs_list = full_merge_sorted_drugs['Drug'].unique()
full_merge_sorted_condition = full_merge_df.sort_values(by='Condition')
all_conditions_list = full_merge_sorted_condition['Condition'].unique()

# Tab 2 datasets
dashboard_drugs_df = full_merge_sorted_drugs[full_merge_sorted_drugs['Website']=='webmd']
dashboard_drugs_df = dashboard_drugs_df.sort_values(by='Drug')
drugs_dashboard_list = dashboard_drugs_df['Drug'].unique()

drugs_age_df = side_effects_df[(side_effects_df['Age']!= " ")]
drugs_sex_df = side_effects_df[(side_effects_df['Sex']!= " ")]

age_list = drugs_age_df['Age'].unique()
sex_list = drugs_sex_df['Sex'].unique()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

# Load ML model and vectorizer
model = load_model('./models/sequential_model.h5')
model._make_predict_function()
graph = tf.get_default_graph()
tfidf_vect = joblib.load('./models/tfidf_vectorizer.pkl')

# Stemmer
stemmer = SnowballStemmer('english')

# ******************************* Custom CSS Styling ***********************************
tabs_styles = {
    'height': '40px'
}

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'background-color': 'silver',
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'border-bottom-color': 'white',
    'padding': '6px',
    'fontWeight': 'bold',
    'background-color': 'silver',
    'border-block-end-width':'5px',
    'border-top-right-radius':'25px',
    'border-top-left-radius':'25px',
}

title_style = {
    'text-align': 'center',
    'width': '100%',
    'display': 'inline-block',
    'vertical-align': 'middle',
    'font-family':'Helvetica',
    'background-image': 'url("./static/images/background.jpg")',
    'background-repeat':'no-repeat',
    'background-position': 'center',
    'height': '150px',
}

predict_label_style = {
    "width": "7rem",
    "height": "3rem",
    'text-align': 'center',
    'font-family': 'Helvetica',
    'display': 'inline-block',
    'vertical-align': 'middle',
    'padding-top':'13.5px',
    'font-size':'20px'
    }

# ******************************* End of CSS Styling ***********************************

# ******************************* Webapp layout ****************************************

# Webapp layout
dashapp.layout = html.Div(
children=[
    html.Div([
        html.Div([
            html.H1('Drug Review Analysis Work',style={'color':'black','font-size':'3.5rem','margin-top':'42px'}),
        ], style=title_style),
        dcc.Tabs(id="tabs", value='recommend', children=[
            dcc.Tab(label='Drug Recommendation', value='recommend',style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Data Insights', value='insights',style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Real Time Sentiment Classification', value='predict',style=tab_style, selected_style=tab_selected_style),
        ], style=tabs_styles),
        html.Div(id='tabs-content')
        ],style={'width': '100%','background-position': 'initial initial', 'background-repeat': 'initial initial'},
    )
])


# Render pageview according to tabs
@dashapp.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'recommend':
        return html.Div([
            html.Div([
                html.Div([
                    html.H5(
                        'Enter or Select Medical Condition',
                        className="control_label",
                        style={'textAlign': 'center'}
                    ),
                    dcc.Dropdown(
                        id='condition-column',
                        options=[{'label': condition, 'value': condition} for condition in all_conditions_list],
                        value='adhd'
                    ),
                ], style={'width': '25%', 'display': 'inline-block','margin-left':'15%','margin-bottom':'1%','margin-top':'1%'}),
                html.Div([
                    html.H5(
                        'Select Drug or Medication',
                        className="control_label",
                        style={'textAlign': 'center'}
                    ),
                    dcc.Dropdown(
                        id='all-drugs-column',
                        options=[{'label': drug, 'value': drug} for drug in all_drugs_list],
                        value='abilify'
                    ),
                ],
                    style={'width': '25%', 'display': 'inline-block','margin-left':'20%'}),
            ],
                style={
                    'borderBottom': 'thin lightgrey solid',
                    'backgroundColor': 'rgb(112, 200, 228)',
                }),

            html.Div([
                html.Div([
                    html.H5(
                        'Top recommended drugs for chosen medical condition',
                        className="control_label",
                        style={'textAlign': 'center', 'margin-right': '95px'}
                    ),
                    dcc.Graph(id='drug-recommendation')
                ], style={'display': 'inline-block', 'width': '54%', 'margin-left': '2.2%'}, className='six columns'),
                html.Div([
                    html.H5(
                        'Emotions associated with chosen drug',
                        className="control_label",
                        style={'textAlign': 'center'}
                    ),
                    dcc.Graph(id='emotion-classification')
                ], style={'display': 'inline-block', 'width': '43%'}, className='six columns'),
            ], className='row', style={'padding': '60px'}),

        ])


    elif tab == 'insights':
        return html.Div([
            html.Div([
                html.Div([
                    html.H5(
                        'Select Drug or Medication',
                        className="control_label",
                        style={'textAlign': 'center'}
                    ),
                    dcc.Dropdown(
                        id='drug-column',
                        options=[{'label': drug, 'value': drug} for drug in drugs_dashboard_list],
                        value='benadryl'
                    ),
                ],
                    style={'width': '25%', 'display': 'inline-block', 'margin-left': '38%'}),
            ],
                style={
                    'borderBottom': 'thin lightgrey solid',
                    'backgroundColor': 'rgb(112, 200, 228)',
                    'padding': '10px 5px'
                }),
                html.Div([
                html.Div([
                    html.H5(
                        'Review Count per Age Group',
                        className="control_label",
                        style={'textAlign': 'center'}
                    ),
                    dcc.Graph(id='age-distribution')
                    ], style={'display': 'inline-block', 'width': '49%'}, className='six columns'),

                    html.Div([
                        html.H5(
                            'Rating Distribution per Age Group',
                            className="control_label",
                            style={ 'textAlign': 'center'}
                        ),
                        dcc.Graph(id='age-rating-distribution')
                    ], style={'display': 'inline-block', 'width': '49%'}, className='six columns'),
                ], className='row',
                    style={'padding': '40px 5px','margin-left': '3%'
                }),
                html.Div([
                html.Div([
                    html.H5(
                        'Side Effects Wordcloud',
                        className="control_label",
                        style={'textAlign': 'center'}
                    ),
                    html.Img(id="side-effects-wordcloud"),
                    ], style={'display': 'inline-block', 'width': '49%'}, className='six columns'),
                html.Div([
                    html.H5(
                        'Gender Distribution',
                        className="control_label",
                        style={'textAlign': 'center'}
                    ),
                    dcc.Graph(id='sex-distribution'),
                ], style={'display': 'inline-block', 'width': '49%'}, className='six columns'),
                ], className='row',
                style={'padding': '10px 5px','margin-left': '12%'}),
        ])

    elif tab == 'predict':
        return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H5('Enter drug review for sentiment classification',style={'textAlign': 'center'}
            ),
        ],style={'width': '97%', 'display': 'inline-block'}),
        ],style={'borderBottom': 'thin lightgrey solid',
                'backgroundColor': 'rgb(112, 200, 228)',
                'padding': '28px 5px'
                }),

        html.Div([
            dcc.Textarea(
                id='input-box',
                placeholder = 'Write Review...',
                value='',
                style={'width': '100%','height':'200px'}
            )
        ],
        style={'display': 'inline-block', 'width': '62%','margin-left':'18%','margin-top':'1%'}),
        html.Div([
            html.Div([
                dbc.Button("Predict", color="dark", className="mr-1", size="lg", id='button')
            ],style={'display': 'inline-block','margin-right':'5%','margin-left':'2%'},className='six columns'),
            html.Div([
                html.H5(["Predicted Sentiment", dbc.Badge(color='info',id='predict-sentiment', className="ml-1",
                                                          style=predict_label_style)]),
                ],style={'display': 'inline-block', 'margin-right': '5%', 'margin-left': '2%','vertical-align': 'middle'}, className='six columns'),
        ],className='row',style={'margin-top': '10px','margin-left': '31%'}),
        html.Div([
                html.H5(
                    't-SNE Word grouping for Drug Reviews',
                    className="control_label",
                    style={'textAlign': 'center', 'width': '50%', 'margin-left': '25%',
                           'padding': '10px 10px 10px 10px', 'borderBottom': 'thin lightgrey solid',
                           'backgroundColor': 'silver', 'font-family': 'Helvetica', 'font-size': '18px'}
                ),
                html.Img(src="./static/images/tsne-bi.jpg", style={'width': '98%', 'height': '1000px'})
            ], style={'display': 'inline-block', 'width': '80%', 'margin-left': '11%','margin-top':'4%'}),
        html.Div([
                html.H5(
                    't-SNE Word grouping for Side Effects',
                    className="control_label",
                    style={'textAlign': 'center', 'width': '50%', 'margin-left': '25%',
                           'padding': '10px 10px 10px 10px', 'borderBottom': 'thin lightgrey solid',
                           'backgroundColor': 'silver', 'font-family': 'Helvetica', 'font-size': '18px'}
                ),
                html.Img(src="./static/images/tsne-side_effects.jpg", style={'width': '98%', 'height': '1000px'})
            ], style={'display': 'inline-block', 'width': '80%', 'margin-left': '11%','margin-top':'4%'}),
        ]),
        ])


# ******************************* End of Webapp layout *********************************

# Plot Top drugs given condition
def recommend_top_drugs(df,condition):
    return {
        'data': [{
            'x': df['Drug'],
            'y': df['Predicted_rating'],
            'type': 'bar',
            'width':0.5,
            'marker' : dict(color= df['Sentiment'],colorbar= dict(title=dict(text='Average Sentiment Rating')),colorscale= 'Blues',showscale= True,reversescale=True),
        }],
        'layout': {
            'height': 500,
            'margin': {'l': 40, 'b': 100, 'r': 10, 't': 40},
            #'title': "Top recommended drugs for " + condition,
            'xaxis': {
                'title': "Drug name",
            },
            'yaxis': {
                'title': "Average Predicted Rating",
            },
        }
    }

# Recommend Top Drugs based on Condition
@dashapp.callback(
    Output('drug-recommendation', 'figure'),
    [Input('condition-column', 'value'),
     ])
def top_drugs_for_condition(input_condition_name):
    condition_filtered = recommended_drugs_df[recommended_drugs_df['Condition'] == input_condition_name]
    best_drugs = condition_filtered.groupby(['Condition', 'Drug']).agg('mean').reset_index()
    best_drugs = best_drugs.sort_values(by=['Sentiment', 'Predicted_rating'], ascending=False).head(5)
    return recommend_top_drugs(best_drugs,input_condition_name)


# Emotion classification Pie chart
@dashapp.callback(
    Output('emotion-classification', 'figure'),
    [Input('all-drugs-column', 'value'),
     ])
def update_emotions(input_drug_name):
    drug_emotion = emotions_df[emotions_df['Drug'] == input_drug_name]
    df = drug_emotion.groupby('Drug').agg('mean').reset_index()
    fig = px.pie(df, values=df.values.flatten().tolist()[1:], names=df.columns[1:],hole=.3)
    fig.update_layout(height=500)
    fig.update_traces(textinfo='percent+label',hoverinfo='label+percent')
    return fig

# Age Group Distribution
@dashapp.callback(
    Output('age-distribution', 'figure'),
    [Input('drug-column', 'value'),
     ])
def age_distribution(input_drug_name):
    drug_age_groups = drugs_age_df[drugs_age_df['Drug'] == input_drug_name]
    age_grouped = drug_age_groups.groupby(['Drug','Age']).size().reset_index().rename(columns={0:'Count'})
    fig = px.bar_polar(age_grouped, r=age_grouped['Count'], theta=age_grouped['Age'], color=age_grouped['Count'],
                           color_discrete_sequence=px.colors.sequential.Plasma_r)
    fig.update_layout(height=500)
    return fig

# Rating Distribution per Age Group
@dashapp.callback(
    Output('age-rating-distribution', 'figure'),
    [Input('drug-column', 'value'),
     ])
def age_rating_distribution(input_drug_name):
    drug_age_groups = drugs_age_df[drugs_age_df['Drug'] == input_drug_name]
    age_grouped = drug_age_groups.groupby(['Drug','Age','Rating']).size().reset_index().rename(columns={0:'Count'})
    fig = px.bar_polar(age_grouped, r=age_grouped['Rating'], theta=age_grouped['Age'], color=age_grouped['Count'],
                           color_discrete_sequence=px.colors.sequential.Plasma_r)
    fig.update_layout(height=500)
    return fig


# Plot Gender Distribution
def plot_sex_distribution(df):
   return{
   'data': [{
       'values' : df['Count'].values,
       'type' : 'pie',
       'hoverinfo':'label+percent+name',
       'hole':0.4,
       'labels':df['Sex'],
   }],
   'layout': {
           'height': 430,
           'margin': {'l': 40, 'b': 30, 'r': 10, 't': 40},
       }
   }

# Gender Distribution
@dashapp.callback(
    Output('sex-distribution', 'figure'),
    [Input('drug-column', 'value'),
     ])
def gender_distribution(input_drug_name):
    drug_sex_groups = drugs_sex_df[drugs_sex_df['Drug'] == input_drug_name]
    sex_grouped = drug_sex_groups.groupby(['Drug','Sex']).size().reset_index().rename(columns={0:'Count'})
    return plot_sex_distribution(sex_grouped)


# Clean Side Effects Review Text
def clean_text(df):
    df['Side_Effects'] = df['Side_Effects'].str.replace("&#039;",'\'')
    df['Side_Effects'] = df['Side_Effects'].str.replace("\"","").str.lower()
    df['Side_Effects'] = df['Side_Effects'].str.replace( r"(\\r)|(\\n)|(\\t)|(\\f)|(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(&#039;)|(\d\s)|(\d)|(\/)","")
    df['Side_Effects'] = df['Side_Effects'].str.replace("\"","").str.lower()
    df['Side_Effects'] = df['Side_Effects'].str.replace( r"(\$)|(\-)|(\\)|(\s{2,})"," ")
    df['Side_Effects'] = df['Side_Effects'].str.replace('\d+', '')
    return df

# Generate Wordcloud
def plot_wordcloud(df,drug_name):
    new_df = df.loc[df['Drug'] == drug_name]
    new_df = clean_text(new_df)
    txt = new_df.Side_Effects.str.replace(r'\|', ' ').str.cat(sep=' ')
    tokenized_words = nltk.tokenize.word_tokenize(txt)
    stop_words = stopwords.words('english')
    stop_words.extend(custom_stopwords_df['stopwords'].tolist())
    stop_words.append(drug_name)
    words = [word for word in tokenized_words if word not in stop_words]
    word_dist = nltk.FreqDist(words)

    # The top 30 words
    rslt = pd.DataFrame(word_dist.most_common(30), columns=['Word', 'Frequency'])
    if len(rslt) == 0:
        text = 'None'
    else:
        text = " ".join(review for review in rslt.Word)
    wc_image = np.array(Image.open('./static/images/wc_mask.jpg'))
    wc = WordCloud(max_font_size=50, mask = wc_image, max_words=25, width=500,contour_width=1, contour_color='red', background_color="white").generate(text)
    return wc.to_image()


# Side Effects Wordcloud
@dashapp.callback(
    Output('side-effects-wordcloud', 'src'),
    [Input('drug-column', 'value'),
     ])
def side_effects_wordcloud(input_drug_name):
    img = BytesIO()
    plot_wordcloud(side_effects_df,input_drug_name).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

# Sentiment Classification
@dashapp.callback(
    dash.dependencies.Output('predict-sentiment', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')])
def update_output(n_clicks, value):
    with graph.as_default():
        set_session(sess)
        sentiment = ''
        if value != '' and value != ' ':
            predicted_class = model.predict_classes(tfidf_vect.transform([str(value)]))
            if predicted_class == 2:
                sentiment = 'Positive'
            elif predicted_class == 1:
                sentiment = 'Neutral'
            else:
                sentiment = 'Negative'
        return sentiment


# Run Dash server
if __name__ == '__main__':
    dashapp.run_server(debug=True)

