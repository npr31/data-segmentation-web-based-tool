import requests
import pandas as pd
from django.conf import settings

class ClusterAnalyzer:
    def __init__(self, df):
        """
        Initialize the ClusterAnalyzer with a DataFrame.
        Args:
            df (pd.DataFrame): DataFrame containing the dataset with clusters.
        """
        self.df = df
        self.api_key = getattr(settings, 'GOOGLE_GENAI_API_KEY', None)
        self.model = 'gemini-2.0-flash'
        self.api_url = f'https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent'

    def analyze_clusters(self):
        """
        Profiles the segmented clusters by analyzing their characteristics based on categorical and numerical data.
        Generates a detailed report using the Gemini API.
        Returns:
            str: Generated analysis report
        """
        try:
            categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
            if 'Cluster' in self.df.columns:
                categorical_columns.insert(0, 'Cluster')
            grouped_df = self.df.groupby(categorical_columns).agg({
                col: ['mean', 'median', 'min', 'max']
                for col in self.df.select_dtypes(include=['number']).columns
            })
            prompt = f"""
                Analyze the segmented clusters in the provided DataFrame, emphasizing each cluster's unique characteristics and differentiating factors.

                **Data:** {grouped_df.to_json()}

                using this data, perform profiling segments and return data (don't mention mean and median directly refer to the value) in the format mentioned below and don't mention about yourself.

                **Output Format:**
                #### <span style='color:lightgreen'>Clusters Information
                1. cluster 0 to cluster n-1 each one, denoting their features

                #### <span style='color:lightgreen'>Behavioral Segmentation (Show only if there is any)
                * Details for each cluster (Segments based on their buying behavior, usage rates, brand loyalty, purchase occasion, and benefits sought.)

                #### <span style='color:lightgreen'>Demographic Segmentation (Show only if there is any)
                * Details for each cluster (Divides based on measurable characteristics like age, gender, income, education, occupation, family size, etc.)

                #### <span style='color:lightgreen'>Geographic Segmentation (Show only if there is any)
                * Details for each cluster (Categorizes based on their geographic location, such as country, state, city, neighborhood, or climate.)

                #### <span style='color:lightgreen'>Psychographic Segmentation (Show only if there is any)
                * Details for each cluster (Groups based on their values, lifestyles, interests, attitudes, opinions, and personality traits.)

                #### Summarizing the segmentation
                * Summarize the key findings of the segmentation analysis, highlighting the most significant differences between clusters
                """
            headers = {
                'Content-Type': 'application/json',
            }
            params = {
                'key': self.api_key
            }
            data = {
                "contents": [
                    {"parts": [{"text": prompt}]}
                ]
            }
            response = requests.post(self.api_url, headers=headers, params=params, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            # Extract the text from the response
            if 'candidates' in result and result['candidates']:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "No analysis could be generated."
        except Exception as e:
            return f"An error occurred during profiling segments: {str(e)}" 