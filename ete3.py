import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from PIL import Image
import io
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer

# Generate Dataset
def generate_dataset():
    tracks = ['AI & ML', 'Cybersecurity', 'IoT', 'Blockchain']
    states = ['Karnataka', 'Maharashtra', 'Tamil Nadu', 'Delhi', 'Kerala']
    colleges = ['IIT Bombay', 'NIT Trichy', 'Anna University', 'Delhi University', 'IISc Bangalore']
    
    data = {
        'Participant_ID': [f'P{1000 + i}' for i in range(400)],
        'Name': [f'Participant_{i}' for i in range(400)],
        'College': [random.choice(colleges) for _ in range(400)],
        'State': [random.choice(states) for _ in range(400)],
        'Track': [random.choice(tracks) for _ in range(400)],
        'Day': [random.randint(1, 4) for _ in range(400)],
        'Presentation_Title': [f'Title_{i}' for i in range(400)],
        'Score': [random.randint(1, 10) for i in range(400)],
        'Feedback': [random.choice(["Great session!", "Needs improvement", "Very informative", "Excellent work!", "Could be better"," very bad", "Terrible","Amazing corordination","Interesting"])
                     for i in range(400)]
    }
    return pd.DataFrame(data)

df = generate_dataset()

# Streamlit App Styling
st.set_page_config(layout="wide", page_title="Poster Presentation Dashboard")
st.markdown("""
    <style>
    .stApp {
        background-color: #E5989B;
    }
             [data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
    }
    [data-testid="stSidebar"] {
        background-color: #B5838D ;
        background-size: cover;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] svg {
        color: #800080  ;
    }
            [data-testid="stBaseButton-secondary"] {
        color: #D8BFD8  ;
        background-color: #800080;
    }
    [data-testid="stFileUploader"] p{
        color: white  ;
    }
    [data-testid="stFileUploaderDropzone"] {
        color: #800080  ;
        background-color: #B5838D;
    }    
       
""", unsafe_allow_html=True)

st.title(" National Poster Presentation Analysis")
st.sidebar.header(" Filters")

# Sidebar Sections
selection = st.sidebar.radio("Select a Section", ["Data Visualisation", "Word Cloud", "Feedback Analysis", "Gallery"])

# Filters
selected_track = st.sidebar.selectbox("Select Track", ['All'] + list(df['Track'].unique()))
selected_day = st.sidebar.selectbox("Select Day", ['All'] + sorted(df['Day'].unique()))
selected_college = st.sidebar.selectbox("Select College", ['All'] + list(df['College'].unique()))
selected_state = st.sidebar.selectbox("Select State", ['All'] + list(df['State'].unique()))

filtered_df = df.copy()
if selected_track != 'All':
    filtered_df = filtered_df[filtered_df['Track'] == selected_track]
if selected_day != 'All':
    filtered_df = filtered_df[filtered_df['Day'] == selected_day]
if selected_college != 'All':
    filtered_df = filtered_df[filtered_df['College'] == selected_college]
if selected_state != 'All':
    filtered_df = filtered_df[filtered_df['State'] == selected_state]

# Data Visualization
if selection == "Data Visualisation":
    st.subheader(" Participation Analysis")

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(filtered_df, x='Track', title='Track-wise Participation', color='Track')
        fig1.update_layout(
            plot_bgcolor='#B5838D',  # Background color for the plot area inside the graph
            paper_bgcolor='#B5838D',  # Background color for the area outside the plot (paper)
            title_font=dict(size=20, family="Arial", color='black')  # Optional: Title font settings
            )
        st.plotly_chart(fig1, use_container_width=True)
    
        fig3 = px.pie(filtered_df, names='College', title='College-wise Participation')
        fig3.update_layout(
            plot_bgcolor='#B5838D',  # Background color for the plot area inside the graph
            paper_bgcolor='#B5838D',  # Background color for the area outside the plot (paper)
            title_font=dict(size=20, family="Arial", color='black')  # Optional: Title font settings
            )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig2 = px.line(filtered_df, x='Day', title='Day-wise Participation', markers=True, line_shape='spline')
        fig2.update_layout(
            plot_bgcolor='#B5838D',  # Background color for the plot area inside the graph
            paper_bgcolor='#B5838D',  # Background color for the area outside the plot (paper)
            title_font=dict(size=20, family="Arial", color='black')  # Optional: Title font settings
            )
        st.plotly_chart(fig2, use_container_width=True)
        
        fig4 = px.bar(filtered_df, x='State', title='State-wise Participation', color='State')
        fig4.update_layout(
            plot_bgcolor='#B5838D',  # Background color for the plot area inside the graph
            paper_bgcolor='#B5838D',  # Background color for the area outside the plot (paper)
            title_font=dict(size=20, family="Arial", color='black')  # Optional: Title font settings
            )
        st.plotly_chart(fig4, use_container_width=True)

    # Heatmap for participation across tracks and days
    st.subheader(" Track-Day Participation Heatmap")
    track_day_data = filtered_df.groupby(['Track', 'Day']).size().reset_index(name='Participation')
    heatmap_fig = px.density_heatmap(track_day_data, x='Day', y='Track', z='Participation', color_continuous_scale='Viridis')
    heatmap_fig.update_layout(
            plot_bgcolor='#B5838D',  # Background color for the plot area inside the graph
            paper_bgcolor='#B5838D',  # Background color for the area outside the plot (paper)
            title_font=dict(size=20, family="Arial", color='black')  # Optional: Title font settings
            )
    st.plotly_chart(heatmap_fig, use_container_width=True)

    st.subheader(" Score Distribution")
    fig5 = px.histogram(filtered_df, x='Score', nbins=10, title='Score Distribution', color='Score' )
    fig5.update_layout(
            plot_bgcolor='#B5838D',  # Background color for the plot area inside the graph
            paper_bgcolor='#B5838D',  # Background color for the area outside the plot (paper)
            title_font=dict(size=20, family="Arial", color='black')  # Optional: Title font settings
            )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Word Cloud
if selection == "Word Cloud":
    st.subheader(" Feedback Word Cloud")
    feedback_text = ' '.join(filtered_df['Feedback'])
    wordcloud = WordCloud(width=800, height=400, background_color='#B5838D').generate(feedback_text)
    st.image(wordcloud.to_array())

# Feedback Similarity Analysis
if selection == "Feedback Analysis":
    st.subheader("🔗 Feedback Similarity Analysis")
    if len(filtered_df) > 1:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(filtered_df['Feedback'])
        similarity_scores = cosine_similarity(tfidf_matrix, tfidf_matrix).mean(axis=1)
        fig6 = px.bar(x=filtered_df['Participant_ID'], y=similarity_scores, title='Feedback Similarity Scores', labels={'x': 'Participant ID', 'y': 'Similarity Score'})
        fig6.update_layout(
            plot_bgcolor='#B5838D',  # Background color for the plot area inside the graph
            paper_bgcolor='#B5838D',  # Background color for the area outside the plot (paper)
            title_font=dict(size=20, family="Arial", color='black')  # Optional: Title font settings
            )
        st.plotly_chart(fig6, use_container_width=True)

    sia = SentimentIntensityAnalyzer()
    filtered_df['Sentiment'] = filtered_df['Feedback'].apply(lambda feedback: 
        'Positive' if sia.polarity_scores(feedback)['compound'] > 0.05 else 
        'Negative' if sia.polarity_scores(feedback)['compound'] < -0.05 else 
        'Neutral'
    ) 

    # Sentiment Distribution Visualization
    sentiment_counts = filtered_df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    fig_sentiment = px.pie(sentiment_counts, names='Sentiment', values='Count', 
                           title="Sentiment Distribution of Feedback", 
                           color='Sentiment', 
                           color_discrete_map={'Positive': '#6cc879', 'Neutral': '#c10000', 'Negative': '#0098c1'})

    fig_sentiment.update_layout(
        plot_bgcolor='#B5838D',
        paper_bgcolor='#B5838D',
        title_font=dict(size=20, family="Arial", color='black')
    )
    
    st.plotly_chart(fig_sentiment, use_container_width=True)

# Gallery
if selection == "Gallery":
    st.subheader(" Image Processing")
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Original Image", use_container_width=True)

        # Custom Image Processing Component: Apply color filter (Sepia and Invert)
        img_array = np.array(image)
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, 100, 150)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(gray_image, caption="Grayscale Image", use_container_width=True, channels="GRAY")
        with col2:
            st.image(edges, caption="Edge Detection", use_container_width=True, channels="GRAY")

        # Carousel of Day-wise Images (as example)
        st.subheader(" Day-wise Image Gallery")
        day_images = ["img1.jpeg", "img2.png", "img3.jpeg", "img4.jpeg"] 
        for idx, img_path in enumerate(day_images):
            with st.expander(f"📅 Day {idx + 1} Image"):
                img = Image.open(img_path)
                st.image(img, use_container_width=True, caption=f"Day {idx + 1} Image")
        st.markdown("---")
        
        st.subheader("🎡 Carousel View")
        carousel_images = [Image.open(img) for img in day_images]

        selected_index = st.slider("Select Image", min_value=0, max_value=len(day_images)-1, value=0)
        st.image(carousel_images[selected_index], use_container_width=True, caption=f"Day {selected_index + 1} Image")
    st.markdown("</div>", unsafe_allow_html=True)
