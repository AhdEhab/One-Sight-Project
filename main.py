import streamlit as st
import plotly.graph_objects as go
import numpy as np
import json

from model_ml import load_model, predict 
from preprocessing import preprocess
from embedding import embed 
from Text_Generation import generate_negative_points
from Text_Generation import generate_positive_points
from image_model import arabic_text_from_image
from vid_model import get_transcript

# Streamlit UI setup
st.set_page_config(page_title="Dashboard", layout="wide")
st.image("logo-black.png", width=200)  


def preprocess_text(text):
    return preprocess(text)


# Load xgboost model 
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

#uploading multiple files
uploaded_files = st.file_uploader(
    "Upload .txt, .json, image, or video files for analysis",
    type=["txt", "json", "jpg", "jpeg", "png", "mp4", "mov"],
    accept_multiple_files=True
)

if uploaded_files:
    texts = []
    filenames = []
    vectors = []

    for file in uploaded_files:
        content = ""

        if file.type == "application/json":
            data = json.load(file)  
            if isinstance(data, dict) and "reviews" in data:
                for review in data["reviews"]:
                    comment = review.get("comment", "").strip()
                    if comment:
                        processed = preprocess_text(comment)
                        embedding = embed(processed)
                        texts.append(comment)
                        filenames.append(file.name)
                        vectors.append(embedding)
            else:
                st.warning(f"Invalid JSON format in: {file.name}")
                continue

        elif file.type == "text/plain":
            content = file.read().decode("utf-8")
            processed = preprocess_text(content)
            embedding = embed(processed)
            texts.append(content)
            filenames.append(file.name)
            vectors.append(embedding)

        elif file.type in ["image/jpeg", "image/png"]:
            content = arabic_text_from_image(file)
            processed = preprocess_text(content)
            embedding = embed(processed)
            texts.append(content)
            filenames.append(file.name)
            vectors.append(embedding)

        elif file.type in ["video/mp4", "video/quicktime"]:
            with st.spinner(f"Transcribing video: {file.name}"):
                content = get_transcript(file)
                processed = preprocess_text(content)
                embedding = embed(processed)
                texts.append(content)
                filenames.append(file.name)
                vectors.append(embedding)

        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue

    X_input = np.array(vectors)

    # Predict
    raw_predictions = predict(X_input)

    predictions = ["positive" if p == 1 else "negative" for p in raw_predictions]

    # Count sentiment
    positive_value = sum(1 for p in predictions if p == "positive")
    negative_value = sum(1 for p in predictions if p == "negative")
    total = positive_value + negative_value
    positive_percentage = int((positive_value / total) * 100)
    negative_percentage = 100 - positive_percentage

    #Separating positives and negatives 
    positive_reviews = []
    negative_reviews = []

    for text, label in zip(texts, predictions):
        if label == "positive":
            positive_reviews.append(text)
        else:
            negative_reviews.append(text)

    # Determine styling
    if positive_value >= negative_value:
        larger_color = "#A8DEE6"
        smaller_color = "#01516D"
        center_text = "Positive"
        center_color = "#0077B6"
        percentage_display = f"{positive_percentage}%"
    else:
        larger_color = "#FF9B8C"
        smaller_color = "#C72C4E"
        center_text = "Negative"
        center_color = "#D62828"
        percentage_display = f"{negative_percentage}%"

    values = [positive_value, negative_value]
    colors = [larger_color, smaller_color] if positive_value >= negative_value else [smaller_color, larger_color]
    labels = ["Positive", "Negative"]

    # Donut chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors),
        textinfo='none',
    )])

    fig.update_layout(
        annotations=[
            dict(
                x=0.5, y=0.56,
                text=f"<b style='color:{center_color};font-size:20px'>{center_text}</b>",
                showarrow=False,
                font_size=20,
                align='center'
            ),
            dict(
                x=0.5, y=0.44,
                text=f"<b style='font-size:16px;color:#555'>{percentage_display}</b>",
                showarrow=False,
                font_size=16
            )
        ],
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20)
    )

    # Placeholder summaries
    positive_text = generate_positive_points(positive_reviews)

    negative_text = generate_negative_points(negative_reviews)

    # Layout
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🟢 Positive Points")
        st.markdown(f"""
        <div style="background-color:rgba(69, 170, 180, 0.2);padding:15px;border-radius:10px;">
            <pre style="font-size: 15px;">{positive_text.strip()}</pre>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔴 Negative Points")
        st.markdown(f"""
        <div style="background-color:rgba(254, 121, 102, 0.2);padding:15px;border-radius:10px;">
            <pre style="font-size: 15px;">{negative_text.strip()}</pre>
        </div>
        """, unsafe_allow_html=True)

    # File predictions
    with st.expander("📄 Show Predictions for Each File"):
        for name, pred in zip(filenames, predictions):
            st.markdown(f"**{name}** — `{pred}`")

else:
    st.warning("Please upload at least one .txt file.")
