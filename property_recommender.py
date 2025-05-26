import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import json
import traceback
from datetime import datetime
import joblib
import os

class PropertyRecommender:
    def __init__(self, model_path='models/property_recommender.joblib'):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.feature_weights = {
            'gla': 0.3,  # Gross Living Area
            'lot_size_sf': 0.2,  # Lot Size
            'year_built': 0.15,  # Year Built
            'num_beds': 0.1,  # Number of Bedrooms
            'num_baths': 0.1,  # Number of Bathrooms
            'room_count': 0.05,  # Total Rooms
            'basement_area': 0.05,  # Basement Area
            'effective_age': 0.05  # Effective Age
        }
        self.feedback_history = []
        
    def load_data(self, data_path):
        """Load and prepare the appraisal dataset."""
        try:
            print("Loading data from", data_path)
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            print(f"Found {len(data['appraisals'])} appraisals")
            
            # Extract subjects and properties
            subjects = []
            properties = []
            selected_comps = []
            
            for i, appraisal in enumerate(data['appraisals']):
                try:
                    # Add subject
                    subject = appraisal['subject']
                    subject['appraisal_id'] = i
                    subjects.append(subject)
                    
                    # Add properties and track selected comps
                    selected_comp_ids = set()
                    for comp in appraisal.get('selected_comps', []):
                        selected_comp_ids.add(comp['id'])
                    
                    for prop in appraisal.get('properties', []):
                        prop['appraisal_id'] = i
                        prop['is_selected_comp'] = prop['id'] in selected_comp_ids
                        properties.append(prop)
                    
                    if i % 10 == 0:
                        print(f"Processed {i} appraisals...")
                        
                except Exception as e:
                    print(f"Error processing appraisal {i}: {str(e)}")
                    continue
            
            print(f"Total subjects: {len(subjects)}")
            print(f"Total properties: {len(properties)}")
            return pd.DataFrame(subjects), pd.DataFrame(properties)
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            print(traceback.format_exc())
            raise

    def preprocess_data(self, subjects_df, properties_df):
        """Preprocess the data for model training."""
        try:
            print("\nPreprocessing data...")
            # Select numerical features
            numerical_features = list(self.feature_weights.keys())
            
            # Find common features between subjects and properties
            common_features = list(set(subjects_df.columns) & set(properties_df.columns) & set(numerical_features))
            print(f"Common features between subjects and properties: {common_features}")
            
            if not common_features:
                raise ValueError("No common numerical features found between subjects and properties")
            
            # Convert features to numeric
            print("Converting features to numeric...")
            for feature in common_features:
                subjects_df[feature] = pd.to_numeric(subjects_df[feature], errors='coerce')
                properties_df[feature] = pd.to_numeric(properties_df[feature], errors='coerce')
            
            # Handle missing values
            print("Handling missing values...")
            X_subjects = subjects_df[common_features].copy()
            X_properties = properties_df[common_features].copy()
            
            # Check for completely missing features
            missing_in_subjects = X_subjects.columns[X_subjects.isna().all()].tolist()
            missing_in_properties = X_properties.columns[X_properties.isna().all()].tolist()
            missing_features = list(set(missing_in_subjects + missing_in_properties))
            
            if missing_features:
                print(f"Features with all missing values: {missing_features}")
                X_subjects = X_subjects.drop(columns=missing_features)
                X_properties = X_properties.drop(columns=missing_features)
                common_features = [f for f in common_features if f not in missing_features]
                print(f"Remaining features after removing missing: {common_features}")
            
            # Impute missing values
            X_subjects_imputed = self.imputer.fit_transform(X_subjects)
            X_properties_imputed = self.imputer.transform(X_properties)
            
            # Convert back to DataFrame
            X_subjects_imputed = pd.DataFrame(X_subjects_imputed, columns=common_features)
            X_properties_imputed = pd.DataFrame(X_properties_imputed, columns=common_features)
            
            return X_subjects_imputed, X_properties_imputed, common_features
            
        except Exception as e:
            print(f"Error preprocessing data: {str(e)}")
            print(traceback.format_exc())
            raise

    def train(self, subjects_df, properties_df):
        """Train the recommendation model."""
        try:
            print("\nTraining recommendation model...")
            # Preprocess data
            X_subjects, X_properties, features = self.preprocess_data(subjects_df, properties_df)
            
            # Scale features
            print("Scaling features...")
            X_properties_scaled = self.scaler.fit_transform(X_properties)
            X_subjects_scaled = self.scaler.transform(X_subjects)
            
            # Create labels for properties (1 for selected comps, 0 for others)
            property_labels = properties_df['is_selected_comp'].astype(int)
            
            # Train KNN model
            print("Training KNN model...")
            self.knn.fit(X_properties_scaled, property_labels)
            
            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump({
                'knn': self.knn,
                'scaler': self.scaler,
                'imputer': self.imputer,
                'features': features
            }, self.model_path)
            
            print("Model trained and saved successfully")
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            print(traceback.format_exc())
            raise

    def find_comps(self, subject_df, properties_df, n_neighbors=3):
        """Find comparable properties for a subject property."""
        try:
            print("\nFinding comparable properties...")
            # Load model if not already loaded
            if not hasattr(self, 'knn'):
                model_data = joblib.load(self.model_path)
                self.knn = model_data['knn']
                self.scaler = model_data['scaler']
                self.imputer = model_data['imputer']
                self.features = model_data['features']
            
            # Preprocess data
            X_subject, X_properties, _ = self.preprocess_data(subject_df, properties_df)
            
            # Scale features
            X_subject_scaled = self.scaler.transform(X_subject)
            X_properties_scaled = self.scaler.transform(X_properties)
            
            # Find nearest neighbors
            distances, indices = self.knn.kneighbors(X_subject_scaled)
            
            # Get comparable properties
            comps = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                comp_property = properties_df.iloc[idx]
                similarity_score = np.exp(-distance)
                
                # Generate explanation
                explanation = self._generate_explanation(
                    subject_df.iloc[0],
                    comp_property,
                    X_subject.iloc[0],
                    X_properties.iloc[idx]
                )
                
                comps.append({
                    'address': comp_property['address'],
                    'similarity_score': similarity_score,
                    'explanation': explanation
                })
            
            return comps
            
        except Exception as e:
            print(f"Error finding comps: {str(e)}")
            print(traceback.format_exc())
            raise

    def _generate_explanation(self, subject, comp, subject_features, comp_features):
        """Generate explanation for why a property was selected as a comp."""
        explanations = []
        
        # Compare key features
        for feature, weight in self.feature_weights.items():
            if feature in subject_features and feature in comp_features:
                subj_val = subject_features[feature]
                comp_val = comp_features[feature]
                
                # Calculate similarity percentage
                if subj_val != 0:  # Avoid division by zero
                    similarity = 1 - abs(subj_val - comp_val) / max(subj_val, comp_val)
                    if similarity > 0.8:  # Only mention very similar features
                        explanations.append(
                            f"Similar {feature.replace('_', ' ')}: "
                            f"{comp_val:.0f} vs {subj_val:.0f}"
                        )
        
        # Add location-based explanation if available
        if 'city' in subject and 'city' in comp:
            if subject['city'] == comp['city']:
                explanations.append("Located in the same city")
        
        return " | ".join(explanations)

    def add_feedback(self, subject_id, comp_id, is_good_comp, feedback_reason=None):
        """Add user feedback to improve the model."""
        feedback = {
            'subject_id': subject_id,
            'comp_id': comp_id,
            'is_good_comp': is_good_comp,
            'feedback_reason': feedback_reason,
            'timestamp': datetime.now().isoformat()
        }
        self.feedback_history.append(feedback)
        
        # If we have enough new feedback, retrain the model
        if len(self.feedback_history) >= 10:
            self._retrain_with_feedback()

    def _retrain_with_feedback(self):
        """Retrain the model incorporating user feedback."""
        try:
            print("\nRetraining model with user feedback...")
            # Convert feedback to training data
            feedback_df = pd.DataFrame(self.feedback_history)
            
            # Update property labels based on feedback
            # This would need to be implemented based on your specific data structure
            
            # Retrain the model
            # This would need to be implemented based on your specific data structure
            
            print("Model retrained with feedback")
            
        except Exception as e:
            print(f"Error retraining model: {str(e)}")
            print(traceback.format_exc())

def main():
    try:
        # Initialize recommender
        recommender = PropertyRecommender()
        
        # Check if data file exists
        data_path = 'appraisals_dataset.json'
        if not os.path.exists(data_path):
            print(f"Error: Data file '{data_path}' not found.")
            print("Please ensure the appraisals_dataset.json file exists in the current directory.")
            return
        
        # Load and split data
        print("\nLoading data...")
        subjects_df, properties_df = recommender.load_data(data_path)
        
        if len(subjects_df) == 0 or len(properties_df) == 0:
            print("Error: No data loaded. Please check the data file format.")
            return
        
        print(f"\nLoaded {len(subjects_df)} subjects and {len(properties_df)} properties")
        
        # Split data into training and validation sets
        print("\nSplitting data into training and validation sets...")
        train_subjects, val_subjects = train_test_split(subjects_df, test_size=0.2, random_state=42)
        print(f"Training set size: {len(train_subjects)}")
        print(f"Validation set size: {len(val_subjects)}")
        
        # Train model
        print("\nTraining model...")
        recommender.train(train_subjects, properties_df)
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        for i, (_, subject) in enumerate(val_subjects.iterrows(), 1):
            print(f"\nProcessing subject {i}/{len(val_subjects)}")
            print(f"Subject: {subject['address']}")
            
            try:
                comps = recommender.find_comps(
                    pd.DataFrame([subject]),
                    properties_df
                )
                
                print("Comparable properties:")
                for j, comp in enumerate(comps, 1):
                    print(f"{j}. {comp['address']}")
                    print(f"   Similarity: {comp['similarity_score']:.2f}")
                    print(f"   Explanation: {comp['explanation']}")
                    
            except Exception as e:
                print(f"Error processing subject {i}: {str(e)}")
                continue
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 