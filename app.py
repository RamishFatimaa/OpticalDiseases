from flask import Flask, render_template_string
from data_preprocessing import preprocess_data
from model_pipeline import train_and_evaluate, prepare_data, plot_correlation_matrix
#from ClusteringNLP import execute_all_nlp_and_clustering_functions
import pandas as pd

app = Flask(__name__)

# HTML template as a Python string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Results</title>
</head>
<body>
    <h1>Data Head</h1>
    <pre>{{ data_head }}</pre>
    <h1>Model Evaluation Results</h1>
    <pre>{{ model_results }}</pre>
</body>
</html>
"""

@app.route('/')
def show_results():
    # Path to your dataset
    df_path = 'D:\\Northeastern\\Semester 2\\Python\\Optical Diseases\\full_df.csv'
    df_cleaned = preprocess_data(df_path)

    # Mock-up for displaying purposes - adjust as necessary
    X = df_cleaned[['Patient Age', 'Patient Sex', 'D', 'H']]  # Example feature selection
    y = df_cleaned['MONR']  # Example target variable
    
    model_results = train_and_evaluate(X, y)
    print("Model Results from train_and_evaluate:", model_results)  # Debug print
    #operations_summary = execute_all_nlp_and_clustering_functions(df_cleaned)
    from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score


def create_age_groups(df_cleaned):
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
    df_cleaned['Age Group'] = pd.cut(df_cleaned['Patient Age'], bins=bins, labels=labels, right=False)

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Age Group', data=df_cleaned, hue='Patient Sex')
    plt.title('Distribution of Patients Across Age Groups')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Patients')
    plt.show()


def disease_prevalence_across_age_groups(df_cleaned):
    disease_age_df = df_cleaned.groupby(['Age Group', 'labels']).size().unstack(fill_value=0)
    disease_age_proportion = disease_age_df.div(disease_age_df.sum(axis=1), axis=0)

    disease_age_proportion.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Proportion of Diseases Across Age Groups')
    plt.xlabel('Age Group')
    plt.ylabel('Proportion of Patients')
    plt.legend(title='Disease')
    plt.show()


def cluster_analysis(df_cleaned):
    features_for_clustering = ['Patient Age', 'Patient Sex', 'D', 'G', 'C', 'A', 'H', 'M']
    X = df_cleaned[features_for_clustering]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    num_clusters = 4
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X_scaled)
    cluster_labels = kmeans.labels_
    df_cleaned['Cluster'] = cluster_labels
    cluster_summary = df_cleaned.groupby('Cluster').mean()
    print(cluster_summary)

def generate_word_cloud(df_cleaned):
    medical_notes = df_cleaned['Left-Diagnostic Keywords'].str.cat(df_cleaned['Right-Diagnostic Keywords'], sep=' ')
    medical_notes = medical_notes.str.lower().str.replace('[^\w\s]', '')
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(medical_notes))
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud of Diagnoses in Medical Notes')
    plt.axis('off')
    plt.show()


def topic_modeling(df_cleaned):
    medical_notes = df_cleaned['Left-Diagnostic Keywords'].str.cat(df_cleaned['Right-Diagnostic Keywords'], sep=' ')
    medical_notes = medical_notes.str.lower()
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    dtm = vectorizer.fit_transform(medical_notes)
    num_topics = 3
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(dtm)
    for idx, topic in enumerate(lda_model.components_):
        print(f"Topic {idx}:", ", ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]))


def classify_medical_notes(df_cleaned):
    X = df_cleaned['Left-Diagnostic Keywords'].str.cat(df_cleaned['Right-Diagnostic Keywords'], sep=' ')
    y = df_cleaned['labels']
    X = X.str.lower()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    # Assuming df_cleaned is already defined in your environment
    create_age_groups(df_cleaned)
    disease_prevalence_across_age_groups(df_cleaned)
    cluster_analysis(df_cleaned)
    generate_word_cloud(df_cleaned)
    topic_modeling(df_cleaned)
    classify_medical_notes(df_cleaned)
    
    #summary_str = "\n".join(f"{key}: {value}" for key, value in operations_summary.items())

    return render_template_string(HTML_TEMPLATE, data_head=df_cleaned.head().to_string(), model_results=model_results)


if __name__ == '__main__':
    app.run(debug=True)


