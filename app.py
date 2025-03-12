import streamlit as st
import torch
from transformers import AutoTokenizer
from src.models.toxic_classifier import ToxicClassifier
import os
import numpy as np
import plotly.graph_objects as go
from typing import Dict

class ToxicPredictor:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = ToxicClassifier().to(self.device)
        
        try:
            # Load trained weights with weights_only=True for security
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # Handle both old and new model state dict formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Load state dict and handle any missing/unexpected keys
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                st.warning(f"Missing keys in state dict: {missing_keys}")
            if unexpected_keys:
                st.warning(f"Unexpected keys in state dict: {unexpected_keys}")
            
            self.model.eval()
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            raise
        
        # Category names
        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def predict(self, text: str) -> Dict[str, float]:
        """Predict toxicity scores for a single text"""
        try:
            # Tokenize
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
            
            # Create results dictionary
            results = {
                category: float(prob)
                for category, prob in zip(self.categories, probabilities)
            }
            
            return results
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            raise

def create_gauge_chart(value: float, title: str) -> go.Figure:
    """Create a gauge chart for toxicity scores"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,  # Convert to percentage
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=200)
    return fig

def main():
    st.set_page_config(
        page_title="Toxic Comment Classifier",
        page_icon="🔍",
        layout="wide"
    )
    
    # Title and description
    st.title("💬 Toxic Comment Classifier")
    st.markdown("""
    This app uses a BERT-based model to detect toxic comments. 
    Enter your text below to analyze it for different types of toxicity.
    """)
    
    # Load model
    model_path = os.path.join("models", "saved", "best_model.pt")
    
    if not os.path.exists(model_path):
        st.error("Model file not found! Please train the model first.")
        return
    
    try:
        # Initialize predictor
        @st.cache_resource(show_spinner=False)
        def load_predictor():
            with st.spinner("Loading model..."):
                return ToxicPredictor(model_path)
        
        predictor = load_predictor()
        
        # Text input
        text = st.text_area(
            "Enter text to analyze:",
            height=100,
            placeholder="Type or paste your text here..."
        )
        
        if st.button("Analyze", type="primary"):
            if not text:
                st.warning("Please enter some text to analyze.")
                return
            
            with st.spinner("Analyzing text..."):
                try:
                    # Get predictions
                    predictions = predictor.predict(text)
                    
                    # Display results
                    st.markdown("### Analysis Results")
                    
                    # Create columns for the gauge charts
                    col1, col2, col3 = st.columns(3)
                    
                    # Display gauge charts in columns
                    with col1:
                        st.plotly_chart(create_gauge_chart(predictions['toxic'], "Toxic"), use_container_width=True)
                        st.plotly_chart(create_gauge_chart(predictions['obscene'], "Obscene"), use_container_width=True)
                    
                    with col2:
                        st.plotly_chart(create_gauge_chart(predictions['severe_toxic'], "Severe Toxic"), use_container_width=True)
                        st.plotly_chart(create_gauge_chart(predictions['threat'], "Threat"), use_container_width=True)
                    
                    with col3:
                        st.plotly_chart(create_gauge_chart(predictions['insult'], "Insult"), use_container_width=True)
                        st.plotly_chart(create_gauge_chart(predictions['identity_hate'], "Identity Hate"), use_container_width=True)
                    
                    # Overall assessment
                    st.markdown("### Overall Assessment")
                    max_toxicity = max(predictions.values())
                    max_category = max(predictions.items(), key=lambda x: x[1])[0]
                    
                    if max_toxicity > 0.5:
                        st.error(f"⚠️ This text may be toxic (highest score: {max_toxicity:.2%} for {max_category})")
                    else:
                        st.success(f"✅ This text appears to be non-toxic (highest score: {max_toxicity:.2%})")
                
                except Exception as e:
                    st.error(f"Error analyzing text: {str(e)}")
        
        # Add information about the categories
        with st.expander("ℹ️ About the Toxicity Categories"):
            st.markdown("""
            The model analyzes text for six types of toxicity:
            
            * **Toxic**: General category for unpleasant content
            * **Severe Toxic**: Extreme cases of toxicity
            * **Obscene**: Explicit or vulgar content
            * **Threat**: Expressions of intent to harm
            * **Insult**: Disrespectful or demeaning language
            * **Identity Hate**: Prejudiced language against protected characteristics
            
            Scores range from 0% to 100%, where higher scores indicate stronger presence of that category.
            """)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "Built with ❤️ using Streamlit and BERT. "
            "Model trained on the Toxic Comment Classification Dataset."
        )
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()